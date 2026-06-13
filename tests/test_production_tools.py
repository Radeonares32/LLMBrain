import shutil
import tempfile
from pathlib import Path

import pytest

from llmbrain.agent.context import (
    CommandPolicy,
    OutputLimiter,
    PathResolver,
    SecretRedactor,
)
from llmbrain.agent.runtime import (
    AgentRuntime,
)
from llmbrain.agent.safety import SafetyMode
from llmbrain.agent.tools import (
    ToolResult,
    WriteFileTool,
)
from tests.test_agent_runtime import MockRawModelProvider


@pytest.fixture
def test_env_dir():
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir) / "workspace"
    workspace.mkdir()

    # Create some dummy files
    (workspace / "file1.txt").write_text("Hello World", encoding="utf-8")
    (workspace / "subdir").mkdir()
    (workspace / "subdir" / "file2.txt").write_text("API_KEY='12345'", encoding="utf-8")

    yield workspace

    shutil.rmtree(temp_dir)


def test_path_resolver_safety(test_env_dir):
    resolver = PathResolver(test_env_dir)

    # 1. Normal resolution
    assert resolver.resolve("file1.txt") == test_env_dir / "file1.txt"
    assert resolver.resolve("subdir/file2.txt") == test_env_dir / "subdir" / "file2.txt"

    # 2. Escape attempts
    with pytest.raises(ValueError, match="escapes the workspace boundary"):
        resolver.resolve("../outside.txt")

    with pytest.raises(ValueError, match="escapes the workspace boundary"):
        resolver.resolve("/etc/passwd")

    # 3. Sensitive/credential files
    with pytest.raises(PermissionError, match="Access to .env file is denied"):
        resolver.resolve(".env")

    with pytest.raises(PermissionError, match="Access to sensitive credential path is denied"):
        resolver.resolve(".ssh/id_rsa")

    # 4. Symlink safety
    outside_file = test_env_dir.parent / "outside_secret.txt"
    outside_file.write_text("secret content")

    symlink_path = test_env_dir / "bad_link.txt"
    symlink_path.symlink_to(outside_file)

    with pytest.raises(ValueError, match="escapes the workspace boundary"):
        resolver.resolve("bad_link.txt")


def test_secret_redactor():
    redactor = SecretRedactor()

    text1 = "my API_KEY = 'secret_val_123' and password: 'some_password_xyz'"
    redacted1, has_secrets1 = redactor.redact(text1)
    assert has_secrets1
    assert "secret_val_123" not in redacted1
    assert "some_password_xyz" not in redacted1
    assert "[REDACTED]" in redacted1

    text2 = "normal text with no keys"
    redacted2, has_secrets2 = redactor.redact(text2)
    assert not has_secrets2
    assert redacted2 == text2

    text3 = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    redacted3, has_secrets3 = redactor.redact(text3)
    assert has_secrets3
    assert "MIIEow" not in redacted3
    assert "[REDACTED]" in redacted3


def test_output_limiter():
    limiter = OutputLimiter(max_bytes=100, max_lines=5)

    # Under limits
    text_small = "line 1\nline 2\nline 3"
    res_small, truncated_small, _, _ = limiter.limit(text_small)
    assert not truncated_small
    assert res_small == text_small

    # Over line limits
    text_many_lines = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7"
    res_lines, truncated_lines, _, _ = limiter.limit(text_many_lines)
    assert truncated_lines
    assert "Truncated" in res_lines
    assert "line 1" in res_lines
    assert "line 7" in res_lines

    # Over byte limits
    text_large_bytes = "a" * 500
    res_bytes, truncated_bytes, _, _ = limiter.limit(text_large_bytes)
    assert truncated_bytes
    assert "Truncated" in res_bytes


def test_command_policy():
    policy = CommandPolicy(SafetyMode.DENY_SHELL)
    # all shell/commands are denied under DENY_SHELL
    assert policy.evaluate("pytest", []) == "deny"

    policy_ask = CommandPolicy(SafetyMode.ASK_BEFORE_WRITE)
    assert policy_ask.evaluate("rm", []) == "deny"
    assert policy_ask.evaluate("git", ["push"]) == "deny"
    assert policy_ask.evaluate("git", ["status"]) == "require_approval"
    assert policy_ask.evaluate("python", ["app.py"]) == "require_approval"

    policy_trusted = CommandPolicy(SafetyMode.TRUSTED_PROJECT)
    assert policy_trusted.evaluate("python", ["app.py"]) == "allow"
    assert policy_trusted.evaluate("rm", []) == "deny"
    assert policy_trusted.evaluate("git", ["push"]) == "deny"


@pytest.mark.asyncio
async def test_auto_retry_prevention(test_env_dir):
    # Setup agent runtime
    responses = [
        {
            "thought": "I want to write something that fails.",
            "tool_name": "write_file",
            "tool_arguments": {"path": "escaping_test/out.txt", "content": "hello"},
        },
        {
            "thought": "Let me try to write the exact same thing again.",
            "tool_name": "write_file",
            "tool_arguments": {"path": "escaping_test/out.txt", "content": "hello"},
        },
    ]

    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=test_env_dir,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )

    # Mock a tool that inherits from WriteFileTool but fails
    class MockFailingWriteTool(WriteFileTool):
        async def execute(self, arguments, context=None):
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary="Simulated write failure",
            )

    runtime.tools.register(MockFailingWriteTool(test_env_dir))

    record = await runtime.execute_task("Write some files")
    assert record.status == "failed"
    assert "Auto-retry of failed side-effecting tool" in record.error


@pytest.mark.asyncio
async def test_task_path_scope_violation_in_runtime(test_env_dir):
    responses = [
        {
            "thought": "I will try to read a file with path traversal.",
            "tool_name": "read_file",
            "tool_arguments": {"path": "../outside.txt"},
        }
    ]
    provider = MockRawModelProvider(responses)
    runtime = AgentRuntime(
        project_root=test_env_dir,
        provider=provider,
        safety_mode=SafetyMode.TRUSTED_PROJECT,
        prompt_func=lambda msg: True,
    )

    record = await runtime.execute_task("Read external file")
    assert record.status == "failed"
    assert "Task-path scope violation" in record.error
