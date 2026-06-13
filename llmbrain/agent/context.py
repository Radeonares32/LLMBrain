"""Execution context and security helper components for Phase 3 tools."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmbrain.agent.safety import SafetyMode


class ToolExecutionContext(BaseModel):
    """Scoped runtime execution context for a single tool call."""

    task_id: str
    session_id: str
    workspace_root: Path
    cwd: Path
    agent_mode: str = "build"
    permission_mode: SafetyMode = SafetyMode.ASK_BEFORE_WRITE
    env_allowlist: list[str] = Field(
        default_factory=lambda: ["PATH", "LANG", "LC_ALL", "HOME", "TERM"]
    )
    command_allowlist: list[str] = Field(default_factory=list)
    timeout_limits: dict[str, float] = Field(default_factory=dict)
    output_limits: dict[str, int] = Field(default_factory=dict)
    cancellation_token: Any = None
    audit_sink: Any = None
    event_bus: Any = None


class PathResolver:
    """Safely resolves and canonicalizes paths within the active workspace root."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def resolve(self, relative_path: str) -> Path:
        """Resolve relative path to canonical path. Rejects traversals and escapes."""
        raw_path = Path(relative_path)
        if raw_path.is_absolute():
            resolved = raw_path
        else:
            resolved = self.workspace_root / raw_path

        try:
            resolved = resolved.resolve(strict=False)
        except Exception as e:
            raise ValueError(f"Path resolution error: {e}")

        if not str(resolved).startswith(str(self.workspace_root)):
            raise ValueError("Path escapes the workspace boundary")

        # Symlink safety check
        for parent in list(resolved.parents) + [resolved]:
            if parent.exists() and parent.is_symlink():
                real_target = parent.resolve()
                if not str(real_target).startswith(str(self.workspace_root)):
                    raise ValueError("Symbolic link escapes the workspace boundary")

        # Prohibit secrets & credential path access
        resolved_str = str(resolved).lower()
        if resolved.name == ".env" or ".env." in resolved.name:
            raise PermissionError("Access to .env file is denied by default")

        prohibited_keywords = [
            ".ssh",
            "id_rsa",
            "id_dsa",
            "id_ed25519",
            "credentials",
            "config.json",
        ]
        for kw in prohibited_keywords:
            if kw in resolved_str:
                raise PermissionError("Access to sensitive credential path is denied")

        return resolved


class OutputLimiter:
    """Enforces byte and line truncation constraints on tool output."""

    def __init__(self, max_bytes: int = 50_000, max_lines: int = 1000) -> None:
        self.max_bytes = max_bytes
        self.max_lines = max_lines

    def limit(self, text: str) -> tuple[str, bool, int, int]:
        """Truncate text to fit within limits, retaining head and tail structure."""
        original_bytes = len(text.encode("utf-8", errors="replace"))
        lines = text.splitlines()
        original_lines = len(lines)
        truncated = False

        # Limit lines
        if len(lines) > self.max_lines:
            half = self.max_lines // 2
            head = lines[:half]
            tail = lines[-half:]
            trunc_msg = f"\n... [Truncated {original_lines - self.max_lines} lines] ...\n"
            lines = head + [trunc_msg] + tail
            text = "\n".join(lines)
            truncated = True

        # Limit bytes
        text_bytes = text.encode("utf-8", errors="replace")
        if len(text_bytes) > self.max_bytes:
            half_bytes = self.max_bytes // 2
            head_part = text_bytes[:half_bytes].decode("utf-8", errors="ignore")
            tail_part = text_bytes[-half_bytes:].decode("utf-8", errors="ignore")
            diff_bytes = original_bytes - len(text_bytes)
            trunc_msg = f"\n\n... [Truncated {diff_bytes} bytes] ...\n\n"
            text = head_part + trunc_msg + tail_part
            truncated = True

        returned_bytes = len(text.encode("utf-8", errors="replace"))
        return text, truncated, original_bytes, returned_bytes


class SecretRedactor:
    """Redacts secret patterns like credentials and private keys from outputs."""

    def __init__(self) -> None:
        self.patterns = [
            re.compile(
                r"(?:api_key|password|secret|token|passwd|key|db_url|database_url)\s*[:=]\s*['\"]([^'\"]+)['\"]",
                re.IGNORECASE,
            ),
            re.compile(r"bearer\s+[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE),
            re.compile(
                r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?"
                r"-----END [A-Z ]+ PRIVATE KEY-----"
            ),
            re.compile(
                r"(?:db|database|connection|conn)?url\s*[:=]\s*['\"]([^'\"]+)['\"]",
                re.IGNORECASE,
            ),
        ]

    def redact(self, text: str) -> tuple[str, bool]:
        """Replace detected secrets with [REDACTED]."""
        redacted = False
        result = text
        for pattern in self.patterns:
            matches = pattern.findall(result)
            if matches:
                redacted = True
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    if len(match) > 4:
                        result = result.replace(match, "[REDACTED]")
        return result, redacted


class CommandPolicy:
    """Evaluates program commands against safety rules and agent modes."""

    def __init__(self, mode: SafetyMode) -> None:
        self.mode = mode
        self.safe_programs = {
            "python",
            "python3",
            "node",
            "pytest",
            "npm",
            "cargo",
            "go",
            "ruff",
            "mypy",
            "git",
        }
        self.prohibited_programs = {
            "rm",
            "rmdir",
            "shred",
            "sudo",
            "su",
            "chmod",
            "chown",
            "mkfs",
            "dd",
            "shutdown",
            "reboot",
            "kill",
            "pkill",
            "curl",
            "wget",
            "docker",
            "kubectl",
            "terraform",
        }

    def evaluate(self, program: str, args: list[str]) -> str:
        """Return 'allow', 'require_approval', or 'deny'."""
        if self.mode == SafetyMode.DENY_SHELL:
            return "deny"

        prog_name = Path(program).name.lower()
        if prog_name in self.prohibited_programs:
            return "deny"

        if prog_name == "git":
            prohibited_git_args = {
                "push",
                "reset",
                "clean",
                "rebase",
                "commit",
                "checkout",
                "force",
            }
            if any(arg in prohibited_git_args for arg in args):
                return "deny"

        for arg in args:
            if "rm " in arg or "rmdir " in arg or "-rf" in arg or "--force" in arg:
                return "deny"

        if prog_name in self.safe_programs:
            if self.mode == SafetyMode.TRUSTED_PROJECT:
                return "allow"
            return "require_approval"

        return "require_approval"


class ProcessRunner:
    """Safely executes system commands as child processes with timeouts and cancellation."""

    def __init__(self, env_allowlist: list[str] = None) -> None:
        self.env_allowlist = env_allowlist or ["PATH", "LANG", "LC_ALL", "HOME", "TERM"]

    async def run(
        self,
        program: str,
        args: list[str],
        cwd: Path,
        timeout: float = 60.0,
        cancellation_token: Any = None,
    ) -> tuple[int, str, str, float]:
        """Run process group and return (exit_code, stdout, stderr, duration_ms)."""
        start_time = time.time()
        minimal_env = {}
        for var in self.env_allowlist:
            if var in os.environ:
                minimal_env[var] = os.environ[var]

        try:
            proc = subprocess_popen_helper(program, args, cwd, minimal_env)
        except FileNotFoundError:
            return -1, "", f"Command not found: {program}", 0.0
        except Exception as e:
            return -1, "", f"Process spawn error: {e}", 0.0

        elapsed = 0.0
        poll_interval = 0.1
        while proc.poll() is None:
            if cancellation_token and cancellation_token.is_cancelled:
                terminate_proc_group(proc)
                duration = (time.time() - start_time) * 1000
                return -1, "", "Command cancelled by token.", duration

            if elapsed >= timeout:
                terminate_proc_group(proc)
                duration = (time.time() - start_time) * 1000
                return -1, "", "Command timed out.", duration

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        stdout, stderr = proc.communicate()
        duration = (time.time() - start_time) * 1000
        return proc.returncode, stdout or "", stderr or "", duration


def subprocess_popen_helper(program: str, args: list[str], cwd: Path, env: dict) -> Any:
    import subprocess

    if os.name != "nt":
        return subprocess.Popen(
            [program] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            start_new_session=True,
        )
    else:
        return subprocess.Popen(
            [program] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )


def terminate_proc_group(proc: Any) -> None:
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        proc.kill()
    proc.wait()
