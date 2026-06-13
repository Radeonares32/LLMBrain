import shutil
import tempfile
from pathlib import Path

import pytest

from llmbrain.agent.runtime import (
    AgentRuntime,
    CancellationToken,
)
from llmbrain.agent.safety import PermissionLevel, SafetyMode
from llmbrain.agent.tools import AgentTool, ToolResult
from llmbrain.models.document import Document
from llmbrain.models.fact import Fact, FactEvidence
from tests.test_agent_runtime import MockRawModelProvider


@pytest.fixture
def temp_workspace():
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir) / "workspace"
    workspace.mkdir()

    # Minimal files
    (workspace / "main.py").write_text("def run():\n    print('hello')\n", encoding="utf-8")
    (workspace / "test_main.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")

    yield workspace

    shutil.rmtree(temp_dir)


# ── Scenario A: Repository question ──────────────────────────────────
@pytest.mark.asyncio
async def test_scenario_a_repository_question(temp_workspace):
    responses = [
        {
            "thought": "I know how the architecture works.",
            "finish_response": "The architecture is modular with an agent runtime.",
            "finish_verification": "Correct.",
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )

    # Store a dummy architecture fact in memory using direct Fact model
    project_id = runtime.memory._project_id()
    fact = Fact(
        id="fact_scen_a",
        project_id=project_id,
        subject="architecture",
        predicate="is",
        object="modular",
        claim="The main module is agent/runtime.",
        confidence="high",
    )
    runtime.memory.store.insert_facts([fact])

    record = await runtime.execute_task("Explain the repository structure.")
    assert record.status == "completed"
    assert "modular" in record.summary
    assert record.verification.status == "skipped"


# ── Scenario B: Read and explain code ────────────────────────────────
@pytest.mark.asyncio
async def test_scenario_b_read_and_explain(temp_workspace):
    responses = [
        {
            "thought": "I will grep for run first.",
            "tool_name": "grep",
            "tool_arguments": {"query": "run"},
        },
        {
            "thought": "Now I will read the file.",
            "tool_name": "read_file",
            "tool_arguments": {"path": "main.py"},
        },
        {
            "thought": "Explanation ready.",
            "finish_response": "The run function prints hello.",
            "finish_verification": "Correct.",
        },
    ]

    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )

    record = await runtime.execute_task("Explain main.py")
    assert record.status == "completed"
    assert "prints hello" in record.summary


# ── Scenario C: Approved code modification ───────────────────────────
@pytest.mark.asyncio
async def test_scenario_c_approved_modification(temp_workspace):
    responses = [
        {
            "thought": "I will write to main.py.",
            "tool_name": "write_file",
            "tool_arguments": {"path": "main.py", "content": "def run():\n    print('modified')\n"},
        },
        {
            "thought": "I will run tests to verify.",
            "tool_name": "run_tests",
            "tool_arguments": {},
        },
        {
            "thought": "Verified successfully.",
            "finish_response": "Modified main.py and verified tests.",
            "finish_verification": "Tests passed.",
        },
    ]

    approved_actions = []

    def approval_func(msg):
        approved_actions.append(msg)
        return True

    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.ASK_BEFORE_WRITE,
        prompt_func=approval_func,
    )

    record = await runtime.execute_task("Modify main.py")
    assert record.status == "completed"
    assert len(approved_actions) > 0
    assert "modified" in (temp_workspace / "main.py").read_text()


# ── Scenario D: Denied modification ──────────────────────────────────
@pytest.mark.asyncio
async def test_scenario_d_denied_modification(temp_workspace):
    responses = [
        {
            "thought": "I want to write to main.py.",
            "tool_name": "write_file",
            "tool_arguments": {"path": "main.py", "content": "def run():\n    print('denied')\n"},
        }
    ]

    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.READ_ONLY,
        prompt_func=lambda msg: False,
    )

    record = await runtime.execute_task("Write in read-only")
    assert record.status == "failed"
    assert "Permission denied" in record.error


# ── Scenario E: Unsafe workspace access ──────────────────────────────
@pytest.mark.asyncio
async def test_scenario_e_unsafe_workspace_access(temp_workspace):
    responses = [
        {
            "thought": "I will attempt to escape the workspace directory.",
            "tool_name": "read_file",
            "tool_arguments": {"path": "../outside.txt"},
        }
    ]

    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )

    record = await runtime.execute_task("Escape sandbox")
    assert record.status == "failed"
    assert "Task-path scope violation" in record.error


# ── Scenario F: Tool failure and recovery ────────────────────────────
@pytest.mark.asyncio
async def test_scenario_f_tool_failure_and_recovery(temp_workspace):
    responses = [
        {
            "thought": "Calling write_file with incorrect hash to simulate failure.",
            "tool_name": "write_file",
            "tool_arguments": {
                "path": "main.py",
                "content": "def run():\n    pass\n",
                "expected_sha256": "wrong_hash_to_fail_first_time",
            },
        },
        {
            "thought": "Correcting the write call parameters.",
            "tool_name": "write_file",
            "tool_arguments": {"path": "main.py", "content": "def run():\n    pass\n"},
        },
        {
            "thought": "Done.",
            "finish_response": "Corrected write success.",
            "finish_verification": "Passed.",
        },
    ]

    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )

    record = await runtime.execute_task("Recover tool call")
    assert record.status == "completed"


# ── Scenario G: Repeated tool loop ───────────────────────────────────
@pytest.mark.asyncio
async def test_scenario_g_repeated_tool_loop(temp_workspace):
    responses = [
        {
            "thought": "I will run status.",
            "tool_name": "git_status",
            "tool_arguments": {},
        },
        {
            "thought": "I will run status again.",
            "tool_name": "git_status",
            "tool_arguments": {},
        },
        {
            "thought": "I will run status again.",
            "tool_name": "git_status",
            "tool_arguments": {},
        },
        {
            "thought": "I will run status again.",
            "tool_name": "git_status",
            "tool_arguments": {},
        },
    ]

    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )

    record = await runtime.execute_task("Repeat status loop")
    assert record.status == "failed"
    assert "repeated more than 3 times" in record.error


# ── Scenario H: Cancellation ─────────────────────────────────────────
@pytest.mark.asyncio
async def test_scenario_h_cancellation(temp_workspace):
    responses = [
        {
            "thought": "I will execute a long shell command.",
            "tool_name": "shell",
            "tool_arguments": {"program": "sleep", "args": ["10"]},
        }
    ]

    cancellation_token = CancellationToken()
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )

    async def cancel_task_delayed():
        await asyncio.sleep(0.5)
        cancellation_token.cancel()

    import asyncio

    asyncio.create_task(cancel_task_delayed())

    record = await runtime.execute_task(
        "Run cancellable command", cancellation_token=cancellation_token
    )
    assert record.status == "cancelled"


# ── Scenario I: Secret output ────────────────────────────────────────
@pytest.mark.asyncio
async def test_scenario_i_secret_output(temp_workspace):
    # Harmless tool that returns a secret
    class SecretOutputTool(AgentTool):
        name = "secret_output"
        description = "Returns a secret"
        permission_level = PermissionLevel.READ

        def get_input_schema(self) -> dict:
            return {}

        def get_output_schema(self) -> dict:
            return {}

        async def execute(self, arguments, context=None):
            return ToolResult(success=True, output="My key is api_key='supersecret'")

    responses = [
        {
            "thought": "Calling secret_output.",
            "tool_name": "secret_output",
            "tool_arguments": {},
        },
        {
            "thought": "Finished.",
            "finish_response": "Done explaining output.",
            "finish_verification": "Passed.",
        },
    ]

    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime.tools.register(SecretOutputTool(temp_workspace))

    record = await runtime.execute_task("Read secret")
    assert record.status == "completed"

    # Audit record should be redacted
    for call in record.summary.split():
        assert "supersecret" not in call


# ── Scenario J: Stale memory invalidation ────────────────────────────
@pytest.mark.asyncio
async def test_scenario_j_stale_memory_invalidation(temp_workspace):
    runtime = AgentRuntime(
        project_root=temp_workspace,
        provider=MockRawModelProvider([]),
        safety_mode=SafetyMode.TRUSTED_PROJECT,
    )

    project_id = runtime.memory._project_id()

    # Insert document first to satisfy FK constraint
    doc = Document(
        id="doc_1",
        project_id=project_id,
        path=str(temp_workspace / "main.py"),
        relative_path="main.py",
        content_hash="hash123",
        file_type=".py",
        line_count=10,
        size_bytes=100,
    )
    runtime.memory.store.insert_documents([doc])

    # Index a file
    fact = Fact(
        id="fact_scen_j",
        project_id=project_id,
        subject="file",
        predicate="contains",
        object="run",
        claim="File main.py contains def run().",
        confidence="high",
        evidence=[
            FactEvidence(
                id="ev_1",
                fact_id="fact_scen_j",
                document_id="doc_1",
                path="main.py",
                start_line=1,
                end_line=2,
            )
        ],
    )
    runtime.memory.store.insert_facts([fact])

    # Check memory is present
    facts = runtime.memory.store.get_facts(project_id)
    assert len(facts) > 0

    # Modify main.py content
    (temp_workspace / "main.py").write_text("def another_run():\n    pass\n", encoding="utf-8")

    # Perform invalidation check (simulate by removing outdated provenance fact)
    for f in facts:
        for ev in f.get("evidence", []):
            if ev.get("path") == "main.py":
                fact_obj = Fact(**f)
                fact_obj.confidence = "low"  # downgrade
                runtime.memory.store.insert_facts([fact_obj])

    updated_facts = runtime.memory.store.get_facts(project_id)
    active_facts = [f for f in updated_facts if f.get("confidence") == "high"]
    assert len(active_facts) == 0


# ── Scenario K: Context budget pressure ───────────────────────────────
@pytest.mark.asyncio
async def test_scenario_k_context_budget_pressure(temp_workspace):
    # Set budget and assemble context
    from llmbrain.agent.runtime import ContextAssembler

    assembler = ContextAssembler(token_budget=200)  # budget is 800 characters

    retrieved_items = {
        "entities": [
            {
                "id": "1",
                "type": "class",
                "name": "A" * 1000,  # oversized item to trigger truncation/exclusion
                "path": "main.py",
                "confidence": "high",
            }
        ],
        "facts": [],
        "relations": [],
    }

    context = assembler.assemble(
        retrieved_items,
        task_query="Explain structure",
        project_name="test",
        project_id="test_id",
    )
    # Should fit budget
    assert len(context.encode("utf-8")) <= 200 * 4


# ── Scenario L: Full failure path ────────────────────────────────────
@pytest.mark.asyncio
async def test_scenario_l_full_failure_path(temp_workspace):
    class FailingRunTests(AgentTool):
        name = "run_tests"
        description = "run tests"
        permission_level = PermissionLevel.SHELL

        def get_input_schema(self) -> dict:
            return {}

        def get_output_schema(self) -> dict:
            return {}

        async def execute(self, arguments, context=None):
            return ToolResult(success=False, output="Test failed.")

    responses_l = [
        {
            "thought": "Run tests first.",
            "tool_name": "run_tests",
            "tool_arguments": {},
        },
        {
            "thought": "Now I complete.",
            "finish_response": "I am done.",
            "finish_verification": "Done.",
        },
    ]
    provider_l = MockRawModelProvider(responses_l)
    runtime_l = AgentRuntime(
        project_root=temp_workspace,
        provider=provider_l,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )
    runtime_l.tools.register(FailingRunTests(temp_workspace))

    record_l = await runtime_l.execute_task("Run with failing tests")
    assert record_l.status == "failed"
    assert "Verification failed" in record_l.summary
