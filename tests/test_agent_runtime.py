"""Comprehensive unit and integration tests for LLMBrain agent runtime."""

import json
from typing import Any, AsyncGenerator, Callable

import pytest

from llmbrain.agent.runtime import (
    AgentRuntime,
    AgentState,
    CancellationToken,
    ContextBudgetError,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    StateTransitionError,
)
from llmbrain.agent.safety import PermissionLevel, SafetyMode
from llmbrain.agent.tools import AgentTool, ToolResult
from llmbrain.llm.base import BaseLLMProvider
from llmbrain.models.llm import LLMRequest, LLMResponse

# ── Provider Adapter 1: Raw ModelProvider ────────────────────────────


class MockRawModelProvider(ModelProvider):
    """Direct implementation of the new ModelProvider protocol."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.call_count = 0

    async def generate(
        self,
        request: ModelRequest,
        cancellation_token: CancellationToken | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        if cancellation_token:
            cancellation_token.check()

        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return ModelResponse(
                message=json.dumps(resp),
                finish_reason="stop",
                input_tokens=100,
                output_tokens=50,
            )
        # Default fallback
        finish_resp = {
            "thought": "All steps executed. Completing.",
            "finish_response": "Task complete.",
            "finish_verification": "Verified successfully.",
        }
        return ModelResponse(
            message=json.dumps(finish_resp),
            finish_reason="stop",
            input_tokens=10,
            output_tokens=10,
        )


# ── Provider Adapter 2: Adapted BaseLLMProvider ─────────────────────


class MockAdaptedBaseProvider(BaseLLMProvider):
    """Implementation of the legacy BaseLLMProvider class."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.call_count = 0

    async def generate(self, request: LLMRequest, stream_callback=None) -> LLMResponse:
        return LLMResponse(raw='{"thought": "Test response", "finish_response": "Test complete."}')

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        yield '{"thought": "Test response", '
        yield '"finish_response": "Test complete."}'

    async def generate_structured(self, request: LLMRequest, schema: dict, stream_callback=None) -> LLMResponse:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return LLMResponse(
                raw=json.dumps(resp),
                parsed=resp,
                model="base-mock",
                is_valid=True,
            )
        finish_resp = {
            "thought": "All steps done.",
            "finish_response": "Legacy task complete.",
            "finish_verification": "Legacy verify success.",
        }
        return LLMResponse(
            raw=json.dumps(finish_resp),
            parsed=finish_resp,
            model="base-mock",
            is_valid=True,
        )


# ── Custom Mock Tools for Test Scenarios ─────────────────────────────


class EchoTool(AgentTool):
    name = "echo"
    description = "Echoes back the inputs."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict:
        return {"type": "object", "properties": {"input": {"type": "string"}}}

    def get_output_schema(self) -> dict:
        return {"type": "object", "properties": {"output": {"type": "string"}}}

    async def execute(self, arguments: dict) -> ToolResult:
        val = arguments.get("input", "")
        return ToolResult(success=True, output=val)


class WriteMockTool(AgentTool):
    name = "write_mock"
    description = "A writing tool for testing permissions."
    permission_level = PermissionLevel.WRITE

    def get_input_schema(self) -> dict:
        return {"type": "object", "properties": {"value": {"type": "string"}}}

    def get_output_schema(self) -> dict:
        return {"type": "object", "properties": {"status": {"type": "string"}}}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(success=True, output="written")


class FailingTool(AgentTool):
    name = "failing_tool"
    description = "Fails systematically."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict:
        return {"type": "object"}

    def get_output_schema(self) -> dict:
        return {"type": "object"}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(success=False, output="", error="Systematic failure.")


# ── Pytest Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def temp_project(tmp_path):
    import subprocess

    project_dir = tmp_path / "test_agent_project"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=project_dir)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_dir)

    # Add dummy file
    (project_dir / "app.py").write_text("print('test')\n", encoding="utf-8")
    return project_dir


# ── Test Scenarios ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_final_answer_without_tools(temp_project):
    """Test when LLM completes a task immediately without invoking any tools."""
    responses = [
        {
            "thought": "I know the answer immediately.",
            "finish_response": "The answer is 42.",
            "finish_verification": "Mental validation.",
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project, provider=provider, prompt_func=lambda msg: True
    )
    record = await runtime.execute_task("What is the meaning of life?")

    assert record.status == "completed"
    assert "The answer is 42." in record.summary


@pytest.mark.asyncio
async def test_one_successful_tool_call(temp_project):
    """Test a single successful read-only tool invocation."""
    responses = [
        {
            "thought": "Let's call echo tool first.",
            "tool_name": "echo",
            "tool_arguments": {"input": "hello echo"},
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(EchoTool(temp_project))

    record = await runtime.execute_task("Run echo")
    assert record.status == "completed"
    assert record.error is None


@pytest.mark.asyncio
async def test_multiple_sequential_tool_calls(temp_project):
    """Test multiple sequential tool calls in a single session trajectory."""
    responses = [
        {
            "thought": "Step 1: Echo hello.",
            "tool_name": "echo",
            "tool_arguments": {"input": "first"},
        },
        {
            "thought": "Step 2: Echo world.",
            "tool_name": "echo",
            "tool_arguments": {"input": "second"},
        },
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(EchoTool(temp_project))

    record = await runtime.execute_task("Double echo")
    assert record.status == "completed"
    # Fallback response should complete successfully
    assert "Task complete." in record.summary


@pytest.mark.asyncio
async def test_denied_tool_call(temp_project):
    """Verify write tool is blocked under read-only safety mode."""
    responses = [
        {
            "thought": "Trying to write.",
            "tool_name": "write_mock",
            "tool_arguments": {"value": "secret"},
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.READ_ONLY,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(WriteMockTool(temp_project))

    record = await runtime.execute_task("Write mock value")
    assert record.status == "failed"
    assert "Permission denied" in record.error


@pytest.mark.asyncio
async def test_approval_required_tool_call(temp_project):
    """Verify approval callback blocks or allows execution accordingly."""
    responses = [
        {
            "thought": "I need to write something.",
            "tool_name": "write_mock",
            "tool_arguments": {"value": "approved_data"},
        }
    ]
    provider = MockRawModelProvider(responses)

    # 1. Deny via approval prompt returning False
    runtime_deny = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.ASK_BEFORE_WRITE,
        prompt_func=lambda msg: False,
    )
    runtime_deny.tools.register(WriteMockTool(temp_project))
    record_deny = await runtime_deny.execute_task("Write with approval deny")
    assert record_deny.status == "failed"
    assert "rejected" in record_deny.error

    # 2. Allow via approval prompt returning True
    provider_allow = MockRawModelProvider(responses)
    runtime_allow = AgentRuntime(
        project_root=temp_project,
        provider=provider_allow,
        safety_mode=SafetyMode.ASK_BEFORE_WRITE,
        prompt_func=lambda msg: True,
    )
    runtime_allow.tools.register(WriteMockTool(temp_project))
    record_allow = await runtime_allow.execute_task("Write with approval allow")
    assert record_allow.status == "completed"


@pytest.mark.asyncio
async def test_unknown_tool(temp_project):
    """Test model calling a tool not registered in the registry."""
    responses = [
        {
            "thought": "Calling unknown tool.",
            "tool_name": "unknown_tool_xyz",
            "tool_arguments": {},
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project, provider=provider, prompt_func=lambda msg: True
    )

    record = await runtime.execute_task("Call unknown")
    assert record.status == "failed"
    assert "Tool not found" in record.error


@pytest.mark.asyncio
async def test_tool_failure(temp_project):
    """Verify tool failures are logged as observations and runtime continues."""
    responses = [
        {
            "thought": "Let's call failing tool.",
            "tool_name": "failing_tool",
            "tool_arguments": {},
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project, provider=provider, prompt_func=lambda msg: True
    )
    runtime.tools.register(FailingTool(temp_project))

    record = await runtime.execute_task("Run failing tool")
    assert record.status == "completed"


@pytest.mark.asyncio
async def test_provider_timeout(temp_project):
    """Test provider failure handling during completions."""

    class TimeoutProvider(ModelProvider):
        async def generate(self, r, c=None, stream_callback=None):
            raise TimeoutError("Model response timeout.")

    runtime = AgentRuntime(
        project_root=temp_project, provider=TimeoutProvider(), prompt_func=lambda msg: True
    )
    record = await runtime.execute_task("Expect timeout")
    assert record.status == "failed"
    assert "Model call failed" in record.error


@pytest.mark.asyncio
async def test_malformed_model_response_with_recovery(temp_project):
    """Test bounded recovery from a malformed JSON output."""

    class MalformedProvider(ModelProvider):
        def __init__(self) -> None:
            self.call_count = 0

        async def generate(self, request: ModelRequest, cancellation_token=None, stream_callback=None) -> ModelResponse:
            self.call_count += 1
            if self.call_count == 1:
                return ModelResponse(
                    message="Not a valid JSON text!",
                    finish_reason="stop",
                )
            # Second call corrected response
            resp = {
                "thought": "I corrected my output.",
                "finish_response": "Recovery succeeded.",
                "finish_verification": "JSON parsed successfully.",
            }
            return ModelResponse(
                message=json.dumps(resp),
                finish_reason="stop",
            )

    runtime = AgentRuntime(
        project_root=temp_project, provider=MalformedProvider(), prompt_func=lambda msg: True
    )
    record = await runtime.execute_task("Trigger recovery")
    assert record.status == "completed"
    assert "Recovery succeeded." in record.summary


@pytest.mark.asyncio
async def test_maximum_iteration_limit(temp_project):
    """Ensure agent loop stops when exceeding configured iteration limit."""
    responses = [
        {
            "thought": "Looping forever.",
            "tool_name": "echo",
            "tool_arguments": {"input": "loop"},
        }
    ] * 5
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(EchoTool(temp_project))

    # set max_iterations to 2
    record = await runtime.execute_task("Force limit", max_iterations=2)
    assert record.status == "failed"
    assert "Maximum iteration limit" in record.error


@pytest.mark.asyncio
async def test_repeated_identical_tool_call_protection(temp_project):
    """Verify loop is aborted if model makes identical tool calls repeatedly."""
    responses = [
        {
            "thought": "Echoing same input.",
            "tool_name": "echo",
            "tool_arguments": {"input": "same"},
        }
    ] * 5
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(EchoTool(temp_project))

    record = await runtime.execute_task("Detect duplicate calls")
    assert record.status == "failed"
    assert "repeated more than 3 times" in record.error


@pytest.mark.asyncio
async def test_cancellation_during_execution(temp_project):
    """Test cancellation of execution loop via CancellationToken."""
    responses = [
        {
            "thought": "Running loop step.",
            "tool_name": "echo",
            "tool_arguments": {"input": "test"},
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(EchoTool(temp_project))

    token = CancellationToken()
    # Cancel task in a separate task or immediately
    token.cancel()

    record = await runtime.execute_task("Cancel this task", cancellation_token=token)
    assert record.status == "cancelled"
    assert "cancelled" in record.error


@pytest.mark.asyncio
async def test_context_token_budget_enforcement(temp_project):
    """Verify that assembler throws ContextBudgetError if context size is strictly exceeded."""
    from llmbrain.agent.runtime import ContextAssembler

    assembler = ContextAssembler(token_budget=2)  # very small budget

    # Mock large retrieved items
    items = {
        "entities": [
            {"id": "e1", "name": "long_name" * 20, "type": "module", "confidence": "high"}
        ],
        "facts": [],
        "relations": [],
    }

    with pytest.raises(ContextBudgetError):
        assembler.assemble(
            items,
            task_query="query",
            project_name="test",
            project_id="test_id",
        )


def test_state_transitions(temp_project):
    """Test valid and invalid state transitions."""
    # Test valid transition created -> failed -> completed (rejection)
    _ = AgentRuntime(
        project_root=temp_project, provider=MockRawModelProvider([]), prompt_func=lambda msg: True
    )

    # We can test state transitions validation manually by triggering a mock loop
    # or checking AgentRuntime transitions.
    # To check StateTransitionError validation, we can trigger execute_task
    # or raise StateTransitionError explicitly.
    # Let's assert StateTransitionError is raised when trying to transition
    # invalid state in execute_task. Actually, we tested StateTransitionError
    # in runtime.py. Let's make sure it raises if next state is wrong.
    # We can write a quick transition unit test:
    from llmbrain.agent.runtime import _VALID_TRANSITIONS

    def test_transition(curr, new):
        allowed = _VALID_TRANSITIONS.get(curr, [])
        if new not in allowed:
            raise StateTransitionError(f"Invalid transition from {curr} to {new}")

    # Valid transitions
    test_transition(AgentState.CREATED, AgentState.RETRIEVING_CONTEXT)
    test_transition(AgentState.RETRIEVING_CONTEXT, AgentState.ASSEMBLING_CONTEXT)

    # Invalid transition
    with pytest.raises(StateTransitionError):
        test_transition(AgentState.CREATED, AgentState.COMPLETED)


@pytest.mark.asyncio
async def test_verification_passed(temp_project):
    """Verify verification passed is logged correctly on test suite success."""
    responses = [
        {
            "thought": "I ran some test commands.",
            "tool_name": "run_tests",
            "tool_arguments": {},
        }
    ]
    provider = MockRawModelProvider(responses)

    # Mock run_tests tool to return success
    class MockSuccessRunTestsTool(AgentTool):
        name = "run_tests"
        description = "mock tests"
        permission_level = PermissionLevel.SHELL

        def get_input_schema(self) -> dict:
            return {}

        def get_output_schema(self) -> dict:
            return {}

        async def execute(self, arguments: dict) -> ToolResult:
            return ToolResult(success=True, output="Tests passed.")

    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(MockSuccessRunTestsTool(temp_project))

    record = await runtime.execute_task("Run validation tests")
    assert record.status == "completed"
    assert record.verification.status == "passed"


@pytest.mark.asyncio
async def test_verification_failed(temp_project):
    """Verify verification failed is logged correctly when tests fail."""
    responses = [
        {
            "thought": "I ran some failing tests.",
            "tool_name": "run_tests",
            "tool_arguments": {},
        }
    ]
    provider = MockRawModelProvider(responses)

    # Mock run_tests tool to return failure
    class MockFailedRunTestsTool(AgentTool):
        name = "run_tests"
        description = "mock tests"
        permission_level = PermissionLevel.SHELL

        def get_input_schema(self) -> dict:
            return {}

        def get_output_schema(self) -> dict:
            return {}

        async def execute(self, arguments: dict) -> ToolResult:
            return ToolResult(success=False, output="Tests failed.", error="Failing test case.")

    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(MockFailedRunTestsTool(temp_project))

    record = await runtime.execute_task("Run validation tests")
    assert record.status == "failed"
    assert record.verification.status == "failed"


@pytest.mark.asyncio
async def test_provider_independence_with_base_llm_adapter(temp_project):
    """Test provider independence by running task using legacy adapted BaseLLMProvider."""
    responses = [
        {
            "thought": "Inside the adapted BaseLLMProvider flow.",
            "finish_response": "Legacy task complete.",
            "finish_verification": "Legacy verify success.",
        }
    ]
    provider = MockAdaptedBaseProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project, provider=provider, prompt_func=lambda msg: True
    )

    record = await runtime.execute_task("Expect success from legacy adapted provider")
    assert record.status == "completed"
    assert "Legacy task complete." in record.summary


@pytest.mark.asyncio
async def test_event_ordering(temp_project):
    """Verify execution telemetry events are emitted in chronological order."""
    events_log = []

    def listener(evt):
        events_log.append(evt.event_type)

    responses = [
        {
            "thought": "Doing a quick step.",
            "tool_name": "echo",
            "tool_arguments": {"input": "event_check"},
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_project,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        event_listener=listener,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(EchoTool(temp_project))

    await runtime.execute_task("Event telemetries")

    # Assert expected events ordering
    expected_flow = [
        "task_created",
        "state_changed",  # RETRIEVING_CONTEXT
        "memory_retrieval_started",
        "memory_retrieved",
        "state_changed",  # ASSEMBLING_CONTEXT
        "context_assembled",
        "state_changed",  # WAITING_FOR_MODEL
        "model_request_started",
        "model_response_received",
        "state_changed",  # VALIDATING_TOOL_CALL
        "tool_call_requested",
        "permission_checked",
        "state_changed",  # EXECUTING_TOOL
        "tool_execution_started",
        "tool_execution_completed",
        "state_changed",  # PROCESSING_OBSERVATION
    ]

    # Check that early lifecycle events are correctly logged in chronological order
    for idx, name in enumerate(expected_flow):
        assert events_log[idx] == name
