"""Built-in production-grade tools for the LLMBrain coding agent."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmbrain.agent.context import (
    CommandPolicy,
    OutputLimiter,
    PathResolver,
    ProcessRunner,
    SecretRedactor,
    ToolExecutionContext,
)
from llmbrain.agent.safety import PermissionLevel


class AuditRecord(BaseModel):
    """Execution audit record for a tool call."""

    tool_name: str
    arguments: dict[str, Any]
    permission_level: str
    status: str  # 'allowed', 'denied', 'executed', 'failed'
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    duration_ms: float = 0.0
    output_size: int = 0
    error: str | None = None


class ToolResult(BaseModel):
    """Standardized output result of a tool execution matching Phase 3 contract."""

    tool_call_id: str | None = None
    tool_name: str
    status: str  # 'success', 'failed'
    summary: str
    data: dict[str, Any] | None = None
    stdout: str | None = None
    stderr: str | None = None
    truncated: bool = False
    duration_ms: float = 0.0
    affected_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None

    def __init__(self, **data: Any) -> None:
        # Backward compatibility translation:
        if "success" in data:
            success_val = data.pop("success")
            data["status"] = "success" if success_val else "failed"
        if "output" in data:
            output_val = data.pop("output")
            if "summary" not in data:
                data["summary"] = output_val
            if "data" not in data:
                data["data"] = {"output": output_val}
            elif isinstance(data["data"], dict) and "output" not in data["data"]:
                data["data"]["output"] = output_val
        if "error" in data and isinstance(data["error"], str):
            error_str = data.pop("error")
            data["error"] = {"code": "unknown_error", "message": error_str, "recoverable": True}
        if "tool_name" not in data:
            data["tool_name"] = "legacy_tool"
        super().__init__(**data)

    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def output(self) -> str:
        if self.status == "failed" and self.error:
            return self.error.get("message", self.summary)
        if self.stdout is not None:
            if self.stderr:
                return f"{self.stdout}\n[STDERR]\n{self.stderr}"
            return self.stdout
        if self.data and "content" in self.data:
            return str(self.data["content"])
        if self.data and "output" in self.data:
            return str(self.data["output"])
        return self.summary


class AgentTool(ABC):
    """Abstract base class for all built-in agent tools."""

    name: str
    description: str
    permission_level: PermissionLevel
    timeout: float = 30.0
    output_size_limit: int = 50_000

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    @abstractmethod
    def get_input_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for the tool's input arguments."""
        pass

    @abstractmethod
    def get_output_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for the tool's output."""
        pass

    @abstractmethod
    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """Execute the tool logic with validated arguments."""
        pass

    def get_context(self, context: ToolExecutionContext | None) -> ToolExecutionContext:
        """Retrieve or construct a default execution context."""
        if context is not None:
            return context
        return ToolExecutionContext(
            task_id="task_default",
            session_id="session_default",
            workspace_root=self.project_root,
            cwd=self.project_root,
        )

    def resolve_path(self, relative_path: str, context: ToolExecutionContext | None = None) -> Path:
        """Safely resolve paths using the workspace boundary resolver."""
        ctx = self.get_context(context)
        resolver = PathResolver(ctx.workspace_root)
        return resolver.resolve(relative_path)

    def limit_output(
        self, text: str, context: ToolExecutionContext | None = None
    ) -> tuple[str, bool]:
        """Truncate tool output if it exceeds size limits."""
        ctx = self.get_context(context)
        max_bytes = ctx.output_limits.get("max_bytes", self.output_size_limit)
        limiter = OutputLimiter(max_bytes=max_bytes)
        limited_text, truncated, _, _ = limiter.limit(text)
        return limited_text, truncated

    def redact_secrets(self, text: str) -> str:
        """Redact sensitive keys/secrets from stdout or command outputs."""
        redactor = SecretRedactor()
        redacted_text, _ = redactor.redact(text)
        return redacted_text


# ── File and Repository Tools ─────────────────────────────────────────


class ReadFileTool(AgentTool):
    """Reads a file's content with optional line slicing and binary checks."""

    name = "read_file"
    description = "Reads a file's content. Supports line range constraints and size limiting."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace relative path of the file"},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read (inclusive, 1-indexed)",
                },
                "max_bytes": {"type": "integer", "description": "Maximum bytes to load"},
            },
            "required": ["path"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "total_lines": {"type": "integer"},
                        "sha256": {"type": "string"},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        path_str = arguments["path"]
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")
        max_bytes = arguments.get("max_bytes")

        try:
            file_path = self.resolve_path(path_str, context)
            if not file_path.is_file():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"File not found: {path_str}",
                    error={
                        "code": "file_not_found",
                        "message": f"File not found: {path_str}",
                        "recoverable": True,
                    },
                )

            # Check binary
            with open(file_path, "rb") as f:
                head = f.read(8192)
                if b"\x00" in head:
                    return ToolResult(
                        tool_name=self.name,
                        status="failed",
                        summary=f"Binary file unsupported: {path_str}",
                        error={
                            "code": "binary_file_unsupported",
                            "message": f"Binary file unsupported: {path_str}",
                            "recoverable": False,
                        },
                    )

            # Read content up to max_bytes if specified
            if max_bytes is not None:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    content = f.read(max_bytes)
            else:
                content = file_path.read_text(encoding="utf-8", errors="replace")

            # Redact secrets
            content = self.redact_secrets(content)

            lines = content.splitlines()
            total_lines = len(lines)

            if start_line is not None or end_line is not None:
                start = max(0, (start_line or 1) - 1)
                end = min(total_lines, end_line or total_lines)
                sliced = lines[start:end]
                output_content = "\n".join(
                    f"{idx + start + 1}: {line}" for idx, line in enumerate(sliced)
                )
                start_l = start + 1
                end_l = end
            else:
                output_content = content
                start_l = 1
                end_l = total_lines

            # Limit output size
            output_content, truncated = self.limit_output(output_content, context)
            sha256_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Read {end_l - start_l + 1} lines from {path_str}",
                data={
                    "path": path_str,
                    "content": output_content,
                    "start_line": start_l,
                    "end_line": end_l,
                    "total_lines": total_lines,
                    "sha256": sha256_hash,
                },
                truncated=truncated,
                duration_ms=duration,
                affected_paths=[path_str],
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "read_file_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class ReadFilesTool(AgentTool):
    """Batch reads multiple files with safety constraints and partial success handling."""

    name = "read_files"
    description = (
        "Batch reads contents of multiple files. Fails files individually if "
        "boundary violation occurs."
    )
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of workspace relative file paths to read",
                }
            },
            "required": ["paths"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                    "success": {"type": "boolean"},
                                    "error": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        paths = arguments["paths"]
        results = []
        warnings = []

        # Bounded batch reading
        if len(paths) > 10:
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary="Too many files. Batch size is limited to 10 files.",
                error={
                    "code": "batch_limit_exceeded",
                    "message": "Batch size limit (10) exceeded",
                    "recoverable": True,
                },
            )

        for path_str in paths:
            try:
                file_path = self.resolve_path(path_str, context)
                if not file_path.is_file():
                    results.append({"path": path_str, "success": False, "error": "File not found"})
                    warnings.append(f"File not found: {path_str}")
                    continue

                with open(file_path, "rb") as f:
                    if b"\x00" in f.read(8192):
                        results.append(
                            {"path": path_str, "success": False, "error": "Binary file unsupported"}
                        )
                        warnings.append(f"Binary file unsupported: {path_str}")
                        continue

                content = file_path.read_text(encoding="utf-8", errors="replace")
                content = self.redact_secrets(content)
                # limit single file content inside batch to 20,000 chars
                if len(content) > 20000:
                    content = content[:20000] + "\n... [Truncated in batch read] ..."

                results.append({"path": path_str, "success": True, "content": content})
            except Exception as e:
                results.append({"path": path_str, "success": False, "error": str(e)})
                warnings.append(f"Failed to read {path_str}: {e}")

        duration = (time.time() - start_time) * 1000
        return ToolResult(
            tool_name=self.name,
            status="success",
            summary=f"Successfully batch read {len([r for r in results if r['success']])} files.",
            data={"files": results},
            duration_ms=duration,
            warnings=warnings,
            affected_paths=[r["path"] for r in results if r["success"]],
        )


class ListFilesTool(AgentTool):
    """Lists repository files, honoring gitignore and exclusion filters."""

    name = "list_files"
    description = "Lists repository files in workspace, honoring gitignore and depth constraints."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Relative directory path (default: '.')",
                },
                "recursive": {"type": "boolean", "description": "List subdirectories recursively"},
                "depth": {"type": "integer", "description": "Depth limit for directory traversal"},
                "show_hidden": {
                    "type": "boolean",
                    "description": "Show hidden files starting with '.'",
                },
                "limit": {"type": "integer", "description": "Result count limit (default: 1000)"},
            },
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        dir_rel = arguments.get("directory_path", ".")
        recursive = arguments.get("recursive", True)
        depth_limit = arguments.get("depth")
        show_hidden = arguments.get("show_hidden", False)
        limit = arguments.get("limit", 1000)

        try:
            target_dir = self.resolve_path(dir_rel, context)
            if not target_dir.is_dir():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"Directory not found: {dir_rel}",
                    error={
                        "code": "directory_not_found",
                        "message": f"Directory not found: {dir_rel}",
                        "recoverable": True,
                    },
                )

            skip_dirs = {
                "node_modules",
                ".git",
                "dist",
                "build",
                ".venv",
                "venv",
                "__pycache__",
                ".llmbrain",
                ".pytest_cache",
                ".agents",
                ".codex",
            }

            files = []
            ctx = self.get_context(context)

            # Check if inside git repo to respect gitignore
            git_tracked = set()
            git_check = subprocess.run(
                ["git", "ls-files"],
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
            )
            if git_check.returncode == 0:
                for line in git_check.stdout.splitlines():
                    git_tracked.add(str(ctx.workspace_root / line))

            # Python fallback traversal
            def traverse(path: Path, current_depth: int) -> None:
                if len(files) >= limit:
                    return
                if depth_limit is not None and current_depth > depth_limit:
                    return

                try:
                    for entry in sorted(path.iterdir()):
                        if len(files) >= limit:
                            break
                        if entry.name in skip_dirs:
                            continue
                        if not show_hidden and entry.name.startswith("."):
                            continue

                        # Respect gitignore lookup if git was successful
                        if git_tracked and str(entry) not in git_tracked and entry.is_file():
                            continue

                        if entry.is_file():
                            rel_p = entry.relative_to(ctx.workspace_root)
                            files.append(str(rel_p))
                        elif entry.is_dir() and recursive:
                            traverse(entry, current_depth + 1)
                except Exception:
                    pass

            traverse(target_dir, 1)
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Listed {len(files)} files in {dir_rel}",
                data={"files": files},
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "list_files_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class GlobTool(AgentTool):
    """Finds files matching specific patterns with deduplication and sorting."""

    name = "glob"
    description = "Searches for files matching standard repository patterns."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns",
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude patterns",
                },
                "limit": {"type": "integer", "description": "Result count limit (default: 500)"},
            },
            "required": ["patterns"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "matches": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        patterns = arguments["patterns"]
        excludes = arguments.get("exclude") or []
        limit = arguments.get("limit", 500)
        ctx = self.get_context(context)

        try:
            matches = []
            skip_dirs = {".git", "node_modules", ".llmbrain", ".venv", "venv", "__pycache__"}

            for root, dirnames, filenames in os.walk(ctx.workspace_root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]

                for pat in patterns:
                    for filename in fnmatch.filter(filenames, pat):
                        full_path = Path(root) / filename
                        rel_path = str(full_path.relative_to(ctx.workspace_root))
                        matches.append(rel_path)

                    if "**" in pat:
                        for filename in filenames:
                            full_path = Path(root) / filename
                            rel_p = str(full_path.relative_to(ctx.workspace_root))
                            if fnmatch.fnmatch(rel_p, pat):
                                matches.append(rel_p)

            # Apply excludes
            filtered_matches = []
            for m in matches:
                excluded = False
                for excl in excludes:
                    if fnmatch.fnmatch(m, excl):
                        excluded = True
                        break
                if not excluded:
                    filtered_matches.append(m)

            filtered_matches = sorted(list(set(filtered_matches)))[:limit]
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Found {len(filtered_matches)} glob matches.",
                data={"matches": filtered_matches},
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "glob_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class GrepTool(AgentTool):
    """Executes high-performance search in files, utilizing ripgrep when available."""

    name = "grep"
    description = "Searches for matches of regex/literal patterns in files."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path_filter": {"type": "string", "description": "Glob filter for target paths"},
                "case_sensitive": {"type": "boolean", "description": "Case sensitive search"},
                "max_matches": {
                    "type": "integer",
                    "description": "Limit of matches (default: 250)",
                },
            },
            "required": ["pattern"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "matches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "line": {"type": "integer"},
                                    "content": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        pattern_str = arguments["pattern"]
        path_filter = arguments.get("path_filter")
        case_sensitive = arguments.get("case_sensitive", True)
        max_matches = arguments.get("max_matches", 250)
        ctx = self.get_context(context)

        try:
            # Attempt native ripgrep search first
            rg_args = ["-n", "--no-heading", "-M", "200"]
            if not case_sensitive:
                rg_args.append("-i")
            if path_filter:
                rg_args.extend(["-g", path_filter])
            rg_args.extend([pattern_str, "."])

            # Run ripgrep securely if available
            result = subprocess.run(
                ["rg"] + rg_args,
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=15.0,
            )
            if result.returncode in (0, 1):
                # parse output
                matches = []
                lines = result.stdout.splitlines()
                for line in lines[:max_matches]:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        matches.append(
                            {"path": parts[0], "line": int(parts[1]), "content": parts[2]}
                        )

                # Redact secrets from matches
                for m in matches:
                    m["content"] = self.redact_secrets(m["content"])

                duration = (time.time() - start_time) * 1000
                return ToolResult(
                    tool_name=self.name,
                    status="success",
                    summary=f"Found {len(matches)} matches via ripgrep.",
                    data={"matches": matches},
                    duration_ms=duration,
                )
        except Exception:
            pass

        # Fallback to pure Python search
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern_str, flags)
            matches = []

            skip_dirs = {".git", "node_modules", ".llmbrain", ".venv", "venv", "__pycache__"}

            for root, dirnames, filenames in os.walk(ctx.workspace_root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]

                for filename in filenames:
                    if len(matches) >= max_matches:
                        break

                    full_path = Path(root) / filename
                    rel_path = full_path.relative_to(ctx.workspace_root)

                    if path_filter and not fnmatch.fnmatch(str(rel_path), path_filter):
                        continue

                    # Check binary
                    with open(full_path, "rb") as f:
                        if b"\x00" in f.read(8192):
                            continue

                    try:
                        content = full_path.read_text(encoding="utf-8", errors="replace")
                        for idx, line in enumerate(content.splitlines()):
                            if pattern.search(line):
                                line_redacted = self.redact_secrets(line.strip())
                                matches.append(
                                    {
                                        "path": str(rel_path),
                                        "line": idx + 1,
                                        "content": line_redacted,
                                    }
                                )
                                if len(matches) >= max_matches:
                                    break
                    except Exception:
                        continue

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Found {len(matches)} matches (Python fallback).",
                data={"matches": matches},
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "grep_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class FileMetadataTool(AgentTool):
    """Retrieves standard file metadata, timestamps, size, and hash."""

    name = "file_metadata"
    description = "Provides metadata (size, hash, type, modified time) for a target path."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace relative path to examine"},
                "include_hash": {"type": "boolean", "description": "Generate SHA-256 hash"},
            },
            "required": ["path"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "type": {"type": "string"},
                        "size": {"type": "integer"},
                        "modified": {"type": "string"},
                        "is_symlink": {"type": "boolean"},
                        "sha256": {"type": "string"},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        path_str = arguments["path"]
        include_hash = arguments.get("include_hash", False)

        try:
            file_path = self.resolve_path(path_str, context)
            if not file_path.exists() and not file_path.is_symlink():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"Path not found: {path_str}",
                    error={
                        "code": "path_not_found",
                        "message": f"Path not found: {path_str}",
                        "recoverable": True,
                    },
                )

            stat = file_path.lstat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
            is_sym = file_path.is_symlink()

            path_type = "file"
            if file_path.is_dir():
                path_type = "directory"
            elif is_sym:
                path_type = "symlink"

            sha256 = None
            if include_hash and file_path.is_file() and not is_sym:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Obtained metadata for {path_str}",
                data={
                    "path": path_str,
                    "type": path_type,
                    "size": size,
                    "modified": mtime,
                    "is_symlink": is_sym,
                    "sha256": sha256,
                },
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "metadata_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


# ── File Editing Tools ────────────────────────────────────────────────


class CreateFileTool(AgentTool):
    """Creates a new file securely, preventing duplicates or parent escapes."""

    name = "create_file"
    description = "Creates a new file with optional default contents. Fails if file already exists."
    permission_level = PermissionLevel.WRITE

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace relative path of new file"},
                "content": {"type": "string", "description": "Initial text content"},
            },
            "required": ["path"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "sha256": {"type": "string"},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        path_str = arguments["path"]
        content = arguments.get("content", "")

        try:
            file_path = self.resolve_path(path_str, context)
            if file_path.exists():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"File already exists: {path_str}",
                    error={
                        "code": "file_already_exists",
                        "message": f"File already exists: {path_str}",
                        "recoverable": True,
                    },
                )

            # Atomic write implementation
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = file_path.with_name(f".{file_path.name}.tmp")
            temp_file.write_text(content, encoding="utf-8")
            temp_file.replace(file_path)

            sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Created file: {path_str}",
                data={"path": path_str, "sha256": sha256},
                duration_ms=duration,
                affected_paths=[path_str],
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "create_file_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class WriteFileTool(AgentTool):
    """Writes or overwrites complete file content using optimistic locking (hash checking)."""

    name = "write_file"
    description = "Writes complete content to a file, verifying hash to avoid stale writes."
    permission_level = PermissionLevel.WRITE

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace relative path"},
                "content": {"type": "string", "description": "Full file content"},
                "expected_sha256": {
                    "type": "string",
                    "description": "Expected SHA-256 before modification",
                },
            },
            "required": ["path", "content"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "sha256": {"type": "string"},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        path_str = arguments["path"]
        content = arguments["content"]
        expected_sha = arguments.get("expected_sha256")

        try:
            file_path = self.resolve_path(path_str, context)

            # Optimistic concurrency check
            if expected_sha and file_path.is_file():
                current_text = file_path.read_text(encoding="utf-8", errors="replace")
                current_sha = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
                if current_sha != expected_sha:
                    return ToolResult(
                        tool_name=self.name,
                        status="failed",
                        summary="Stale write rejected. Expected hash does not match current state.",
                        error={
                            "code": "stale_hash",
                            "message": "Expected file hash did not match",
                            "recoverable": True,
                        },
                    )

            # Atomic write
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = file_path.with_name(f".{file_path.name}.tmp")
            temp_file.write_text(content, encoding="utf-8")
            temp_file.replace(file_path)

            sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Wrote file content successfully to {path_str}",
                data={"path": path_str, "sha256": sha256},
                duration_ms=duration,
                affected_paths=[path_str],
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "write_file_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class ReplaceTextTool(AgentTool):
    """Performs exact text replacements in a file with occurrence limits."""

    name = "replace_text"
    description = "Performs exact-match text replacements with safe occurrence constraints."
    permission_level = PermissionLevel.WRITE

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace relative path"},
                "old_text": {"type": "string", "description": "Exact text substring to replace"},
                "new_text": {"type": "string", "description": "Replacement string"},
                "occurrence_limit": {
                    "type": "integer",
                    "description": "Allowed count of occurrences (default: 1)",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "occurrences_replaced": {"type": "integer"},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        path_str = arguments["path"]
        old_text = arguments["old_text"]
        new_text = arguments["new_text"]
        occurrence_limit = arguments.get("occurrence_limit", 1)

        try:
            file_path = self.resolve_path(path_str, context)
            if not file_path.is_file():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"File not found: {path_str}",
                    error={
                        "code": "file_not_found",
                        "message": f"File not found: {path_str}",
                        "recoverable": True,
                    },
                )

            content = file_path.read_text(encoding="utf-8", errors="replace")
            matches = content.count(old_text)

            if matches == 0:
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"No occurrences of old_text found in {path_str}",
                    error={
                        "code": "no_match",
                        "message": "Text to replace not found",
                        "recoverable": True,
                    },
                )

            if matches > occurrence_limit:
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=(
                        f"Ambiguous replacement. Found {matches} occurrences, "
                        f"limit is {occurrence_limit}"
                    ),
                    error={
                        "code": "ambiguous_replacement",
                        "message": "Ambiguous replacement matches",
                        "recoverable": True,
                    },
                )

            new_content = content.replace(old_text, new_text, occurrence_limit)

            # Atomic write
            temp_file = file_path.with_name(f".{file_path.name}.tmp")
            temp_file.write_text(new_content, encoding="utf-8")
            temp_file.replace(file_path)

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Replaced {matches} occurrences of text in {path_str}",
                data={"path": path_str, "occurrences_replaced": matches},
                duration_ms=duration,
                affected_paths=[path_str],
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "replace_text_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class ApplyPatchTool(AgentTool):
    """Validates and applies unified diff format patches securely."""

    name = "apply_patch"
    description = "Applies a unified diff patch to a target file. Dry-run checks are run first."
    permission_level = PermissionLevel.WRITE

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path"},
                "patch": {"type": "string", "description": "Unified diff patch body"},
            },
            "required": ["path", "patch"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        path_str = arguments["path"]
        patch_content = arguments["patch"]
        ctx = self.get_context(context)

        try:
            file_path = self.resolve_path(path_str, context)
            if not file_path.is_file():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"File not found: {path_str}",
                    error={
                        "code": "file_not_found",
                        "message": f"File not found: {path_str}",
                        "recoverable": True,
                    },
                )

            patch_file = ctx.workspace_root / ".llmbrain_temp.patch"
            patch_file.write_text(patch_content, encoding="utf-8")

            try:
                # Dry run first
                dry_run = subprocess.run(
                    ["git", "apply", "--check", str(patch_file)],
                    cwd=ctx.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if dry_run.returncode != 0:
                    # Fallback dry run via patch utility
                    dry_run2 = subprocess.run(
                        ["patch", "--dry-run", "-p1", "-i", str(patch_file)],
                        cwd=ctx.workspace_root,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )
                    if dry_run2.returncode != 0:
                        return ToolResult(
                            tool_name=self.name,
                            status="failed",
                            summary="Patch validation (dry-run) failed.",
                            error={
                                "code": "patch_validation_failed",
                                "message": f"Dry-run errors: {dry_run2.stderr}",
                                "recoverable": True,
                            },
                        )

                # Real application
                result = subprocess.run(
                    ["git", "apply", str(patch_file)],
                    cwd=ctx.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode == 0:
                    duration = (time.time() - start_time) * 1000
                    return ToolResult(
                        tool_name=self.name,
                        status="success",
                        summary=f"Successfully applied patch to {path_str}",
                        duration_ms=duration,
                        affected_paths=[path_str],
                    )

                result2 = subprocess.run(
                    ["patch", "-p1", "-i", str(patch_file)],
                    cwd=ctx.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result2.returncode == 0:
                    duration = (time.time() - start_time) * 1000
                    return ToolResult(
                        tool_name=self.name,
                        status="success",
                        summary=f"Successfully applied patch to {path_str}",
                        duration_ms=duration,
                        affected_paths=[path_str],
                    )

                duration = (time.time() - start_time) * 1000
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary="Failed to apply patch.",
                    error={
                        "code": "patch_apply_failed",
                        "message": result2.stderr,
                        "recoverable": True,
                    },
                    duration_ms=duration,
                )
            finally:
                if patch_file.exists():
                    patch_file.unlink()
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "patch_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class DeleteFileTool(AgentTool):
    """Deletes a file securely. Prevents directory deletions in this phase."""

    name = "delete_file"
    description = "Deletes a single file. Directories are rejected."
    permission_level = PermissionLevel.WRITE

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace relative path of file to delete",
                },
            },
            "required": ["path"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        path_str = arguments["path"]

        try:
            file_path = self.resolve_path(path_str, context)
            if not file_path.exists():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"File not found: {path_str}",
                    error={
                        "code": "file_not_found",
                        "message": f"File not found: {path_str}",
                        "recoverable": True,
                    },
                )

            if file_path.is_dir():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"Directory deletion unsupported: {path_str}",
                    error={
                        "code": "directory_deletion_unsupported",
                        "message": "Cannot delete directories",
                        "recoverable": False,
                    },
                )

            file_path.unlink()
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Deleted file: {path_str}",
                duration_ms=duration,
                affected_paths=[path_str],
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "delete_file_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


# ── Git Tools ─────────────────────────────────────────────────────────


class GitStatusTool(AgentTool):
    """Porcelain structured Git status."""

    name = "git_status"
    description = "Provides current Git status of the repository."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "branch": {"type": "string"},
                        "staged": {"type": "array", "items": {"type": "string"}},
                        "unstaged": {"type": "array", "items": {"type": "string"}},
                        "untracked": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        ctx = self.get_context(context)

        try:
            # Check git repo
            if not (ctx.workspace_root / ".git").exists():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary="Not a Git repository",
                    error={
                        "code": "not_git_repository",
                        "message": "Not a Git repository",
                        "recoverable": True,
                    },
                )

            result = subprocess.run(
                ["git", "status", "--porcelain", "-b"],
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary="Git command failed.",
                    error={
                        "code": "git_status_failed",
                        "message": result.stderr,
                        "recoverable": True,
                    },
                )

            lines = result.stdout.splitlines()
            branch = "unknown"
            staged = []
            unstaged = []
            untracked = []

            for line in lines:
                if line.startswith("## "):
                    branch = line[3:].split("...")[0].strip()
                elif line.startswith(" M ") or line.startswith(" A ") or line.startswith(" D "):
                    staged.append(line[3:].strip())
                elif line.startswith("M  ") or line.startswith("D  "):
                    unstaged.append(line[3:].strip())
                elif line.startswith("?? "):
                    untracked.append(line[3:].strip())

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Git branch: {branch}. Changed files: {len(staged) + len(unstaged)}",
                data={
                    "branch": branch,
                    "staged": staged,
                    "unstaged": unstaged,
                    "untracked": untracked,
                },
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "git_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class GitDiffTool(AgentTool):
    """Provides staged or working tree Git diff outputs."""

    name = "git_diff"
    description = "Provides repository diff summaries and bounded patch text."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Optional file path filter"},
                "staged": {"type": "boolean", "description": "Compare staged changes"},
            },
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "stdout": {"type": "string"},
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        file_path = arguments.get("file_path")
        staged = arguments.get("staged", False)
        ctx = self.get_context(context)

        try:
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--cached")
            if file_path:
                cmd.append(str(self.resolve_path(file_path, context)))

            result = subprocess.run(
                cmd,
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = self.redact_secrets(result.stdout)
            stdout, truncated = self.limit_output(stdout, context)

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary="Diff generated successfully.",
                stdout=stdout,
                truncated=truncated,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "git_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class GitLogTool(AgentTool):
    """Provides bounded Git commit log lists."""

    name = "git_log"
    description = "Provides commit log summaries."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum commits (default: 10)"},
            },
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "stdout": {"type": "string"},
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        limit = arguments.get("limit", 10)
        ctx = self.get_context(context)

        try:
            result = subprocess.run(
                ["git", "log", "-n", str(limit), "--oneline"],
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout, truncated = self.limit_output(result.stdout, context)
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Git log retrieved: {limit} commits limit.",
                stdout=stdout,
                truncated=truncated,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "git_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class GitShowTool(AgentTool):
    """Inspects a single commit or Git object safely."""

    name = "git_show"
    description = "Inspects Git commits or objects safely."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "commit_hash": {"type": "string", "description": "Target commit hash"},
            },
            "required": ["commit_hash"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "stdout": {"type": "string"},
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        commit = arguments["commit_hash"]
        ctx = self.get_context(context)

        # Basic revision validation
        if not re.match(r"^[a-zA-Z0-9\-\^~_./]+$", commit):
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary="Invalid commit hash pattern.",
                error={
                    "code": "invalid_commit_pattern",
                    "message": "Invalid commit hash pattern.",
                    "recoverable": True,
                },
            )

        try:
            result = subprocess.run(
                ["git", "show", commit],
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = self.redact_secrets(result.stdout)
            stdout, truncated = self.limit_output(stdout, context)

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Inspected object {commit}",
                stdout=stdout,
                truncated=truncated,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "git_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class GitBlameTool(AgentTool):
    """Provides Git blame output for a file."""

    name = "git_blame"
    description = "Provides blame metadata annotation for a file."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Target path"},
            },
            "required": ["file_path"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "stdout": {"type": "string"},
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        file_path = arguments["file_path"]
        ctx = self.get_context(context)

        try:
            target = self.resolve_path(file_path, context)
            result = subprocess.run(
                ["git", "blame", str(target)],
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout, truncated = self.limit_output(result.stdout, context)

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Git blame generated for {file_path}",
                stdout=stdout,
                truncated=truncated,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "git_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class GitBranchTool(AgentTool):
    """Lists local branch configurations."""

    name = "git_branch"
    description = "Lists repository branch settings."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "stdout": {"type": "string"},
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        ctx = self.get_context(context)

        try:
            result = subprocess.run(
                ["git", "branch", "-a"],
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout, truncated = self.limit_output(result.stdout, context)

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary="Listed Git branches",
                stdout=stdout,
                truncated=truncated,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "git_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class GitChangedFilesTool(AgentTool):
    """Identifies changed paths comparison."""

    name = "git_changed_files"
    description = "Provides lists of files changed between git boundaries."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "changed_files": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        ctx = self.get_context(context)

        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            files = result.stdout.splitlines()

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                summary=f"Found {len(files)} changed files.",
                data={"changed_files": files},
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "git_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


# ── Command and Verification Tools ───────────────────────────────────


class ShellTool(AgentTool):
    """Spawns structured commands securely, enforcing policies and minimal environments."""

    name = "shell"
    description = (
        "Executes process commands safely, denying shell operators and forbidden commands."
    )
    permission_level = PermissionLevel.SHELL

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "Executable program to spawn"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argument list",
                },
                "cwd": {"type": "string", "description": "Relative directory"},
                "timeout_seconds": {"type": "integer", "description": "Execution timeout"},
            },
            "required": ["program"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        program = arguments["program"]
        args = arguments.get("args") or []
        cwd_rel = arguments.get("cwd", ".")
        timeout = float(arguments.get("timeout_seconds") or self.timeout)
        ctx = self.get_context(context)

        # Enforce command policy
        policy = CommandPolicy(ctx.permission_mode)
        decision = policy.evaluate(program, args)
        if decision == "deny":
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=f"Command execution prohibited: {program}",
                error={
                    "code": "command_denied",
                    "message": f"Execution of {program} is denied by policy.",
                    "recoverable": False,
                },
            )

        try:
            cwd_path = self.resolve_path(cwd_rel, context)
            if not cwd_path.is_dir():
                return ToolResult(
                    tool_name=self.name,
                    status="failed",
                    summary=f"Directory not found: {cwd_rel}",
                    error={
                        "code": "directory_not_found",
                        "message": f"Cwd directory not found: {cwd_rel}",
                        "recoverable": True,
                    },
                )

            # Spawn process using ProcessRunner
            runner = ProcessRunner(env_allowlist=ctx.env_allowlist)
            ret, stdout, stderr, run_duration = await runner.run(
                program, args, cwd_path, timeout, ctx.cancellation_token
            )

            # Redact secrets
            stdout = self.redact_secrets(stdout)
            stderr = self.redact_secrets(stderr)

            # Truncate output
            stdout, tr_out = self.limit_output(stdout, context)
            stderr, tr_err = self.limit_output(stderr, context)

            status = "success" if ret == 0 else "failed"
            summary = f"Command exited with code {ret} in {run_duration:.1f}ms."
            error_data = None
            if ret != 0:
                error_data = {"code": "non_zero_exit", "message": summary, "recoverable": True}

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status=status,
                summary=summary,
                stdout=stdout,
                stderr=stderr,
                truncated=tr_out or tr_err,
                duration_ms=duration,
                error=error_data,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "shell_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class RunTestsTool(AgentTool):
    """Executes high-level tests, autodetecting project dependencies."""

    name = "run_tests"
    description = "Executes the test suite inside the workspace."
    permission_level = PermissionLevel.SHELL

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Specific test file or module path"},
            },
        }

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "framework": {"type": "string"},
                        "command": {"type": "array", "items": {"type": "string"}},
                        "passed": {"type": "integer"},
                        "failed": {"type": "integer"},
                        "skipped": {"type": "integer"},
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        target = arguments.get("target")
        ctx = self.get_context(context)

        # Autodetect runner commands
        cmd = ["pytest"]
        if (ctx.workspace_root / "flake.nix").exists():
            cmd = ["nix", "develop", "--command", "pytest"]
        elif (ctx.workspace_root / "poetry.lock").exists():
            cmd = ["poetry", "run", "pytest"]

        if target:
            cmd.append(target)

        try:
            runner = ProcessRunner(env_allowlist=ctx.env_allowlist)
            ret, stdout, stderr, run_duration = await runner.run(
                cmd[0], cmd[1:], ctx.workspace_root, 60.0, ctx.cancellation_token
            )

            # Basic structured parser
            passed = 0
            failed = 0
            skipped = 0

            # e.g., "37 passed, 1 skipped in 2.76s" or "3 failed, 15 passed"
            match = re.search(r"(\d+)\s+passed", stdout)
            if match:
                passed = int(match.group(1))
            match_fail = re.search(r"(\d+)\s+failed", stdout)
            if match_fail:
                failed = int(match_fail.group(1))
            match_skip = re.search(r"(\d+)\s+skipped", stdout)
            if match_skip:
                skipped = int(match_skip.group(1))

            status = "success" if ret == 0 else "failed"
            summary = f"Test execution status: {status}. Passed: {passed}, Failed: {failed}."

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status=status,
                summary=summary,
                stdout=self.redact_secrets(stdout),
                stderr=self.redact_secrets(stderr),
                data={
                    "framework": "pytest",
                    "command": cmd,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                },
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "test_runner_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


class DiagnosticsTool(AgentTool):
    """Executes code analysis and compilation checks in the repository."""

    name = "diagnostics"
    description = "Executes linters and static checks to return project-level diagnostics."
    permission_level = PermissionLevel.READ

    def get_input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "diagnostics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "line": {"type": "integer"},
                                    "severity": {"type": "string"},
                                    "message": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        start_time = time.time()
        ctx = self.get_context(context)

        # Autodetect ruff or pyproject.toml
        cmd = ["ruff", "check", "."]
        if (ctx.workspace_root / "flake.nix").exists():
            cmd = ["nix", "develop", "--command", "ruff", "check", "."]

        try:
            runner = ProcessRunner(env_allowlist=ctx.env_allowlist)
            ret, stdout, stderr, run_duration = await runner.run(
                cmd[0], cmd[1:], ctx.workspace_root, 30.0, ctx.cancellation_token
            )

            # parse diagnostics best-effort
            diagnostics = []
            lines = stdout.splitlines()
            for line in lines:
                # e.g., "path/to/file.py:12:4: error: message"
                m = re.match(r"^([^:]+):(\d+):(\d+):\s+([^\s]+)\s+(.*)$", line)
                if m:
                    diagnostics.append(
                        {
                            "path": m.group(1).strip(),
                            "line": int(m.group(2)),
                            "severity": "error" if "error" in m.group(4).lower() else "warning",
                            "message": m.group(5).strip(),
                        }
                    )

            status = "success" if ret == 0 or len(diagnostics) == 0 else "failed"
            summary = f"Code diagnostics executed. Found {len(diagnostics)} issues."

            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status=status,
                summary=summary,
                data={"diagnostics": diagnostics},
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=self.name,
                status="failed",
                summary=str(e),
                error={"code": "diagnostics_error", "message": str(e), "recoverable": True},
                duration_ms=duration,
            )


# ── Registry ─────────────────────────────────────────────────────────


class ToolRegistry:
    """Registry to manage and look up available agent tools."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.tools: dict[str, AgentTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            ReadFileTool,
            ReadFilesTool,
            ListFilesTool,
            GlobTool,
            GrepTool,
            FileMetadataTool,
            CreateFileTool,
            WriteFileTool,
            ReplaceTextTool,
            ApplyPatchTool,
            DeleteFileTool,
            GitStatusTool,
            GitDiffTool,
            GitLogTool,
            GitShowTool,
            GitBlameTool,
            GitBranchTool,
            GitChangedFilesTool,
            ShellTool,
            RunTestsTool,
            DiagnosticsTool,
        ]
        for tool_class in defaults:
            tool_instance = tool_class(self.project_root)
            self.tools[tool_instance.name] = tool_instance

    def register(self, tool: AgentTool) -> None:
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> AgentTool | None:
        return self.tools.get(name)

    def get_all_tools(self) -> list[AgentTool]:
        return list(self.tools.values())
