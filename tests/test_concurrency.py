import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from llmbrain.agent.runtime import (
    AgentRuntime,
    CancellationToken,
)
from llmbrain.agent.safety import SafetyMode
from llmbrain.agent.tools import WriteFileTool
from tests.test_agent_runtime import MockRawModelProvider


@pytest.fixture
def test_workspace():
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir) / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("initial", encoding="utf-8")
    yield workspace
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_two_readonly_sessions_independent(test_workspace):
    """Scenario 333: Ensure two read-only task sessions can execute
    simultaneously without resource collision."""
    responses_1 = [{"thought": "Q1", "finish_response": "Ans1", "finish_verification": "V1"}]
    responses_2 = [{"thought": "Q2", "finish_response": "Ans2", "finish_verification": "V2"}]

    provider_1 = MockRawModelProvider(responses_1)
    provider_2 = MockRawModelProvider(responses_2)

    runtime_1 = AgentRuntime(test_workspace, provider_1, SafetyMode.READ_ONLY)
    runtime_2 = AgentRuntime(test_workspace, provider_2, SafetyMode.READ_ONLY)

    # Run concurrently
    rec1, rec2 = await asyncio.gather(
        runtime_1.execute_task("Task 1"), runtime_2.execute_task("Task 2")
    )

    assert rec1.status == "completed"
    assert rec2.status == "completed"
    assert rec1.task_id != rec2.task_id
    assert "Ans1" in rec1.summary
    assert "Ans2" in rec2.summary


@pytest.mark.asyncio
async def test_concurrent_writes_optimistic_protection(test_workspace):
    """Scenario 334: Ensure concurrent writes trigger optimistic locking hash mismatch."""
    import hashlib

    initial_sha = hashlib.sha256(b"initial").hexdigest()

    # Both try to write expecting the initial hash
    tool = WriteFileTool(test_workspace)

    # First write changes to "first"
    res1 = await tool.execute(
        {"path": "main.py", "content": "first", "expected_sha256": initial_sha}
    )

    # Second write expects the initial hash but the file has been modified to "first"
    res2 = await tool.execute(
        {"path": "main.py", "content": "second", "expected_sha256": initial_sha}
    )

    assert res1.success
    assert not res2.success
    assert res2.error["code"] == "stale_hash"


@pytest.mark.asyncio
async def test_cancellation_completion_race(test_workspace):
    """Scenario 335: Verify race between cancellation and completion resolves cleanly."""
    responses = [{"thought": "Ending", "finish_response": "Finished", "finish_verification": "Yes"}]
    provider = MockRawModelProvider(responses)

    cancellation_token = CancellationToken()
    runtime = AgentRuntime(test_workspace, provider, SafetyMode.TRUSTED_PROJECT)

    # Cancel immediately
    cancellation_token.cancel()

    record = await runtime.execute_task("Run race task", cancellation_token=cancellation_token)
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_no_task_id_leakage(test_workspace):
    """Scenario 338: Verify task IDs and event logs do not leak between sessions."""
    events_1 = []
    events_2 = []

    runtime_1 = AgentRuntime(
        test_workspace,
        MockRawModelProvider([{"thought": "T1", "finish_response": "R1"}]),
        SafetyMode.TRUSTED_PROJECT,
        event_listener=lambda e: events_1.append(e),
    )
    runtime_2 = AgentRuntime(
        test_workspace,
        MockRawModelProvider([{"thought": "T2", "finish_response": "R2"}]),
        SafetyMode.TRUSTED_PROJECT,
        event_listener=lambda e: events_2.append(e),
    )

    await asyncio.gather(runtime_1.execute_task("T1"), runtime_2.execute_task("T2"))

    task_id_1 = events_1[0].task_id
    task_id_2 = events_2[0].task_id

    assert task_id_1 != task_id_2

    # Ensure all events in events_1 only refer to task_id_1
    for ev in events_1:
        assert ev.task_id == task_id_1
    for ev in events_2:
        assert ev.task_id == task_id_2
