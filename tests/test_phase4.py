import json
from typing import Any, Callable

import pytest

from llmbrain.agent.agents import (
    AgentDefinition,
    AgentOutputSchemaFailureError,
    AgentRegistry,
    AgentRouter,
    DelegationDepthLimitError,
    DelegationLoopError,
    DuplicateAgentNameError,
    InvalidAgentConfigurationError,
    compose_prompt,
    validate_agent_output,
)
from llmbrain.agent.runtime import (
    AgentRuntime,
    CancellationToken,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)

# ── Mock Model Provider for Deterministic Tests ──────────────────────


class MockModelProvider(ModelProvider):
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.call_count = 0

    async def generate(
        self,
        request: ModelRequest,
        cancellation_token: CancellationToken | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return ModelResponse(
                message=json.dumps(resp),
                finish_reason="stop",
                input_tokens=100,
                output_tokens=50,
            )
        # Fallback
        finish_resp = {
            "thought": "Finish task",
            "finish_response": "Default response",
            "finish_verification": "Verified",
        }
        return ModelResponse(
            message=json.dumps(finish_resp),
            finish_reason="stop",
            input_tokens=10,
            output_tokens=10,
        )


# ── Registry & Definition Tests ──────────────────────────────────────


def test_registry_builtins():
    registry = AgentRegistry()
    assert len(registry.list_agents()) == 7

    ask = registry.get_agent("ask")
    assert ask.name == "ask"
    assert ask.display_name == "Ask Agent"
    assert ask.permissions.get("mode") == "read-only"

    plan = registry.get_agent("plan")
    assert plan.name == "plan"

    build = registry.get_agent("build")
    assert build.name == "build"
    assert build.permissions.get("mode") == "ask-before-write"


def test_registry_duplicate_registration():
    registry = AgentRegistry()
    duplicate = AgentDefinition(
        name="ask",
        display_name="Duplicate Ask",
        description="Duplicate",
        system_prompt="prompts/ask.md",
    )
    with pytest.raises(DuplicateAgentNameError):
        registry.register_agent(duplicate)


def test_registry_invalid_config():
    registry = AgentRegistry()

    # Invalid max iterations
    invalid_iter = AgentDefinition(
        name="invalid_iter",
        display_name="Invalid Iter",
        description="invalid",
        system_prompt="prompts/ask.md",
    )
    invalid_iter.limits.max_iterations = 0
    with pytest.raises(InvalidAgentConfigurationError):
        registry.register_agent(invalid_iter)

    # Invalid token budget
    invalid_budget = AgentDefinition(
        name="invalid_budget",
        display_name="Invalid Budget",
        description="invalid",
        system_prompt="prompts/ask.md",
    )
    invalid_budget.context.token_budget = -1
    with pytest.raises(InvalidAgentConfigurationError):
        registry.register_agent(invalid_budget)

    # Prohibited tool allowed
    invalid_tool = AgentDefinition(
        name="invalid_tool",
        display_name="Invalid Tool",
        description="invalid",
        system_prompt="prompts/ask.md",
    )
    invalid_tool.tools.allow = ["prohibited"]
    with pytest.raises(InvalidAgentConfigurationError):
        registry.register_agent(invalid_tool)


def test_registry_load_project_config(tmp_path):
    # Setup temp project agents.yaml
    dot_llmbrain = tmp_path / ".llmbrain"
    dot_llmbrain.mkdir()
    agents_yaml = dot_llmbrain / "agents.yaml"

    config_data = {
        "agents": {
            "custom-planner": {
                "display_name": "Custom Planner",
                "description": "Custom plan agent",
                "extends": "plan",
                "context": {"token_budget": 20000},
            }
        }
    }
    with open(agents_yaml, "w") as f:
        json.dump(config_data, f)

    registry = AgentRegistry()
    registry.load_project_config(tmp_path)

    custom = registry.get_agent("custom-planner")
    assert custom.display_name == "Custom Planner"
    assert custom.extends == "plan"
    assert custom.context.token_budget == 20000
    assert custom.permissions.get("mode") == "read-only"  # inherited from plan


def test_registry_reject_security_weakening(tmp_path):
    dot_llmbrain = tmp_path / ".llmbrain"
    dot_llmbrain.mkdir()
    agents_yaml = dot_llmbrain / "agents.yaml"

    # Attempting to change write-only plan to ask-before-write
    config_data = {"agents": {"plan": {"permissions": {"mode": "ask_before_write"}}}}
    with open(agents_yaml, "w") as f:
        json.dump(config_data, f)

    registry = AgentRegistry()
    with pytest.raises(InvalidAgentConfigurationError) as exc_info:
        registry.load_project_config(tmp_path)
    assert "cannot weaken permission mode" in str(exc_info.value)


# ── Explicit & Router Selection Tests ───────────────────────────────


@pytest.mark.asyncio
async def test_router_routing():
    registry = AgentRegistry()
    router = AgentRouter(registry)

    # debug query
    agent, reason = await router.route("Why does this test fail with status code 500?")
    assert agent.name == "debug"
    assert "debug" in reason

    # build query
    agent, reason = await router.route("Implement refresh token rotation inside src/auth.py")
    assert agent.name == "build"

    # ask query
    agent, reason = await router.route("What is the architecture of database backend?")
    assert agent.name == "ask"

    # security query
    agent, reason = await router.route(
        "Audit credentials validation for sql injection vulnerability"
    )
    assert agent.name == "security"

    # test query
    agent, reason = await router.route("Write pytest cases for the new validator")
    assert agent.name == "test"

    # plan query
    agent, reason = await router.route("Generate step by step architecture design plan")
    assert agent.name == "plan"

    # review query
    agent, reason = await router.route("Review my changes on commit history")
    assert agent.name == "review"


# ── Prompt Composition Tests ────────────────────────────────────────


def test_prompt_composition(tmp_path):
    # Setup common and ask prompt files
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    common = prompts_dir / "common.md"
    common.write_text("Common Instructions")

    ask_prompt = prompts_dir / "ask.md"
    ask_prompt.write_text("Ask Instructions")

    registry = AgentRegistry()
    ask_def = registry.get_agent("ask")

    composed = compose_prompt(tmp_path, ask_def)
    assert "Common Instructions" in composed
    assert "Ask Instructions" in composed
    assert "Allowed Tools: [git_log, git_status, glob, grep, read_file, read_files]" in composed
    assert "Denied Capabilities: []" in composed


# ── Structured Output Tests ─────────────────────────────────────────


def test_structured_output_validation():
    # Valid ask output
    payload = {"answer": "This is the answer.", "sources": ["src/main.py"], "uncertainties": []}
    validated = validate_agent_output("ask", payload)
    assert validated["answer"] == "This is the answer."

    # Invalid output structure
    invalid_payload = {"sources": "not-a-list"}
    with pytest.raises(AgentOutputSchemaFailureError):
        validate_agent_output("ask", invalid_payload)


# ── Runtime & Tool Execution Permission Tests ───────────────────────


@pytest.mark.asyncio
async def test_runtime_unauthorized_tool_block(tmp_path):
    # Setup a mock provider that wants to execute a write_file tool
    # but the ask agent is read-only and does not allow write_file.
    provider = MockModelProvider(
        [
            {
                "thought": "I want to edit src/auth.py",
                "tool_name": "write_file",
                "tool_arguments": {"target_file": "src/auth.py", "content": "injected"},
            }
        ]
    )

    runtime = AgentRuntime(tmp_path, provider, agent_name="ask")

    # Run ask agent
    record = await runtime.execute_task("How does login work?")
    assert record.status == "failed"
    assert "is not authorized to use tool" in record.summary


# ── Delegation & Security Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_delegation_loop_prevention(tmp_path):
    # We will test delegation loops.
    # Set up mock responses where root agent delegates to 'ask' which delegates back to 'ask'
    # Wait, the runtime captures delegation loop.
    # Let's mock a provider response calling delegate_task
    provider = MockModelProvider(
        [
            {
                "thought": "I will delegate review to review agent",
                "tool_name": "delegate_task",
                "tool_arguments": {
                    "target_agent": "review",
                    "objective": "Review the applied changes.",
                },
            }
        ]
    )

    # Setup a review agent child runtime response that tries to delegate back to 'build'
    MockModelProvider(
        [
            {
                "thought": "I will delegate back to build",
                "tool_name": "delegate_task",
                "tool_arguments": {"target_agent": "build", "objective": "Fix this code."},
            }
        ]
    )

    # We can test loop checks on parent_chain directly
    runtime = AgentRuntime(tmp_path, provider, agent_name="build")

    # Verify depth and loop checks by executing directly
    # Call execute_task on 'build' with parent_chain containing 'build'
    with pytest.raises(DelegationLoopError):
        await runtime.execute_task("Fix code", parent_chain=["build"])


@pytest.mark.asyncio
async def test_delegation_depth_limit(tmp_path):
    runtime = AgentRuntime(tmp_path, MockModelProvider([]), agent_name="build")
    # parent chain length exceeds limit (max_delegations is 2 for build)
    with pytest.raises(DelegationDepthLimitError):
        await runtime.execute_task("Fix code", parent_chain=["build", "review"])


@pytest.mark.asyncio
async def test_delegation_permission_escalation(tmp_path):
    # ask agent is read-only (safety mode level 1)
    # It cannot delegate to build agent (level 3, ask-before-write)
    # We mock ask agent trying to delegate to build
    provider = MockModelProvider(
        [
            {
                "thought": "I will delegate to build",
                "tool_name": "delegate_task",
                "tool_arguments": {"target_agent": "build", "objective": "Write code."},
            }
        ]
    )

    runtime = AgentRuntime(tmp_path, provider, agent_name="ask")
    record = await runtime.execute_task("Find code structure")
    assert record.status == "failed"
    assert "Escalation blocked" in record.summary


# ── Concurrency & Isolation Tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_agents_concurrency(tmp_path):
    # Run two Ask Agents in parallel and ensure no state leaks
    provider1 = MockModelProvider(
        [
            {
                "thought": "Ask 1 response",
                "finish_response": '{"answer":"Response 1","sources":[],"uncertainties":[]}',
                "finish_verification": "Verified 1",
            }
        ]
    )
    provider2 = MockModelProvider(
        [
            {
                "thought": "Ask 2 response",
                "finish_response": '{"answer":"Response 2","sources":[],"uncertainties":[]}',
                "finish_verification": "Verified 2",
            }
        ]
    )

    runtime1 = AgentRuntime(tmp_path, provider1, agent_name="ask")
    runtime2 = AgentRuntime(tmp_path, provider2, agent_name="ask")

    import asyncio

    t1 = runtime1.execute_task("Question 1")
    t2 = runtime2.execute_task("Question 2")

    r1, r2 = await asyncio.gather(t1, t2)

    assert r1.status == "completed"
    assert r2.status == "completed"

    # Assert session isolation
    assert r1.task_id != r2.task_id
    assert r1.session_id != r2.session_id
    assert "Response 1" in r1.summary
    assert "Response 2" in r2.summary
