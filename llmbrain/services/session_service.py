"""Session and Application Controller service for LLMBrain."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from llmbrain.core.identity import get_project_storage_dir, load_or_create_project_identity
from llmbrain.core.lock import WorkspaceLock
from llmbrain.storage.cache import BrainCache
from llmbrain.storage.sqlite import EventStore, SessionStore, SQLiteStore


class SessionService:
    """Orchestrates persistent project brain, memory, sessions, and cache lifecycles."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.identity = load_or_create_project_identity(self.project_root)
        self.project_id = self.identity["project_id"]
        self.storage_dir = get_project_storage_dir(self.project_id)

        # Database connections
        self.store = SQLiteStore(self.storage_dir / "brain.db")
        self.session_store = SessionStore(self.storage_dir / "sessions.db")
        self.event_store = EventStore(self.storage_dir / "events.db")

        # Workspace lock
        self.lock = WorkspaceLock(self.project_id, self.storage_dir / "locks")

        # Configurable RAM Cache limits
        self.cache = BrainCache(
            max_items=500,
            max_bytes=134217728,  # 128 MB
            ttl_seconds=1800,
        )

        # Configurable storage quotas
        self.project_max_bytes = 2147483648  # 2 GB
        self.session_retention_days = 90
        self.raw_tool_output_retention_days = 14
        self.event_retention_days = 30
        self.auto_compact = True

    # ── session operations ──────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """List all saved sessions for this project."""
        cache_key = "sessions_list"
        cached = self.cache.get(self.project_id, cache_key)
        if cached is not None:
            return cached

        sessions = self.session_store.get_sessions(self.project_id)
        self.cache.set(self.project_id, cache_key, sessions)
        return sessions

    def create_session(
        self,
        title: str,
        active_agent: str,
        model_config: dict,
        permission_mode: str,
        parent_session_id: str | None = None,
        branch: str | None = None,
        commit_hash: str | None = None,
    ) -> dict:
        """Create a new session."""
        session_id = f"sess_{uuid4().hex[:12]}"
        session = self.session_store.create_session(
            id=session_id,
            project_id=self.project_id,
            title=title,
            active_agent=active_agent,
            model_config=model_config,
            permission_mode=permission_mode,
            status="active",
            parent_session_id=parent_session_id,
            branch=branch,
            commit_hash=commit_hash,
        )
        self.cache.invalidate(self.project_id, "sessions_list")
        return session

    def get_session(self, session_id: str) -> dict | None:
        """Retrieve a specific session by ID (cached)."""
        cache_key = f"session_{session_id}"
        cached = self.cache.get(self.project_id, cache_key)
        if cached is not None:
            return cached

        session = self.session_store.get_session(session_id)
        if session:
            self.cache.set(self.project_id, cache_key, session)
        return session

    def update_session(self, session_id: str, **kwargs: Any) -> None:
        """Update session settings or state."""
        self.session_store.update_session(session_id, **kwargs)
        self.cache.invalidate(self.project_id, f"session_{session_id}")
        self.cache.invalidate(self.project_id, "sessions_list")

    def rename_session(self, session_id: str, new_title: str) -> None:
        """Rename a session."""
        self.update_session(session_id, title=new_title)

    def archive_session(self, session_id: str) -> None:
        """Archive a session."""
        self.update_session(session_id, status="archived")

    def delete_session(self, session_id: str) -> None:
        """Delete a session from database."""
        self.session_store.delete_session(session_id)
        self.cache.invalidate(self.project_id, f"session_{session_id}")
        self.cache.invalidate(self.project_id, "sessions_list")

    def get_messages(self, session_id: str) -> list[dict]:
        """Get paginated conversation messages."""
        return self.session_store.get_messages(session_id, limit=200)

    def add_message(
        self, session_id: str, role: str, content: str, token_estimate: int = 0
    ) -> None:
        """Add a message to a session."""
        msg_id = f"msg_{int(time.time() * 1000)}_{role}"
        self.session_store.insert_message(msg_id, session_id, role, content, token_estimate)
        self.cache.invalidate(self.project_id, f"session_{session_id}")

    def add_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict | str,
        result: str | None = None,
        status: str = "requested",
        duration_ms: int = 0,
    ) -> None:
        """Record a tool execution log in the session history."""
        tc_id = f"tc_{uuid4().hex[:12]}"
        self.session_store.insert_tool_call(
            id=tc_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            status=status,
            duration_ms=duration_ms,
        )

    def get_tool_calls(self, session_id: str) -> list[dict]:
        """Retrieve tool call logs for a session."""
        return self.session_store.get_tool_calls(session_id, limit=100)

    def export_session(self, session_id: str) -> str:
        """Export session transcript to readable markdown text."""
        session = self.get_session(session_id)
        if not session:
            return ""

        messages = self.get_messages(session_id)
        tool_calls = self.get_tool_calls(session_id)

        lines = [
            f"# Session: {session['title']}",
            f"- Project ID: {session['project_id']}",
            f"- Agent: {session['active_agent']}",
            f"- Model: {session['model_config'].get('model', 'default')}",
            f"- Status: {session['status']}",
            f"- Created At: {session['created_at']}",
            "",
            "## Conversation",
            "",
        ]

        for msg in messages:
            lines.append(f"### {msg['role'].upper()} ({msg['timestamp']})")
            lines.append(msg["content"])
            lines.append("")

        if tool_calls:
            lines.append("## Tool Executions")
            lines.append("")
            for tc in tool_calls:
                lines.append(f"### `{tc['tool_name']}` ({tc['status']})")
                lines.append(f"- Duration: {tc['duration_ms']} ms")
                lines.append(f"- Timestamp: {tc['timestamp']}")
                lines.append("- Arguments:")
                lines.append("```json")
                lines.append(tc["arguments"])
                lines.append("```")
                if tc.get("result"):
                    lines.append("- Output:")
                    lines.append("```")
                    lines.append(str(tc["result"])[:1000])  # Bounded export
                    lines.append("```")
                lines.append("")

        return "\n".join(lines)

    # ── session compaction ──────────────────────────────────────────────

    async def compact_session(self, session_id: str, provider: Any = None) -> dict:
        """Compact older chat history into structured engineering summary."""
        session = self.get_session(session_id)
        if not session:
            return {}

        messages = self.get_messages(session_id)
        tool_calls = self.get_tool_calls(session_id)

        history_str = ""
        for msg in messages:
            history_str += f"{msg['role'].upper()}: {msg['content']}\n"
        for tc in tool_calls:
            history_str += f"TOOL CALL: {tc['tool_name']}({tc['arguments']}) -> {tc['status']}\n"

        summary = "Compacted conversation history"
        completed = []
        unresolved = []
        decisions = []
        files_changed = []
        commands_run = []
        test_outcomes = []
        failures = []
        user_constraints = []

        if provider:
            try:
                # Build compaction summary using model provider
                prompt = (
                    "You are compacting a coding session history. Summarize the following "
                    "conversation, decisions made, completed and unresolved objectives, "
                    "files changed, failures/resolutions, and user constraints.\n\n"
                    f"{history_str}\n\n"
                    "Output a valid JSON object with keys: conversation_summary, "
                    "completed_objectives, unresolved_objectives, decisions, files_changed, "
                    "commands_run, test_outcomes, failures_resolutions, user_constraints"
                )
                from llmbrain.agent.runtime import Message as AgentMessage
                from llmbrain.llm.base import ModelRequest

                res = await provider.complete(
                    ModelRequest(
                        messages=[AgentMessage(role="user", content=prompt)],
                        system_prompt="Return ONLY a JSON response.",
                    )
                )
                data = json.loads(res.message)
                summary = data.get("conversation_summary", summary)
                completed = data.get("completed_objectives", [])
                unresolved = data.get("unresolved_objectives", [])
                decisions = data.get("decisions", [])
                files_changed = data.get("files_changed", [])
                commands_run = data.get("commands_run", [])
                test_outcomes = data.get("test_outcomes", [])
                failures = data.get("failures_resolutions", [])
                user_constraints = data.get("user_constraints", [])
            except Exception:
                pass  # Fallback programmatically if offline/tests

        # Programmatic parsing from task runs history to enrich summary
        runs = self.store.get_task_runs(self.project_id)
        for r in runs:
            details = self.store.get_task_details(r["id"])
            if details:
                for d in details.get("decisions", []):
                    decisions.append(d["decision"])
                for f in details.get("failures", []):
                    failures.append(f"{f['failure']} -> {f.get('resolution')}")
                for c in details.get("commands", []):
                    commands_run.append(c["command"])

        # Deduplicate
        decisions = list(set(decisions))
        failures = list(set(failures))
        commands_run = list(set(commands_run))

        compaction_state = {
            "conversation_summary": summary,
            "completed_objectives": completed,
            "unresolved_objectives": unresolved,
            "decisions": decisions,
            "files_changed": files_changed,
            "commands_run": commands_run,
            "test_outcomes": test_outcomes,
            "failures_resolutions": failures,
            "user_constraints": user_constraints,
            "compacted_at": datetime.now(UTC).isoformat(),
        }

        self.update_session(session_id, compaction_state=compaction_state)
        return compaction_state

    # ── quotas & cleanups ───────────────────────────────────────────────

    def enforce_quotas(self) -> dict[str, Any]:
        """Prune database entries exceeding retention constraints. Report pressure."""
        # Calculate current total database files size
        db_files = ["brain.db", "sessions.db", "events.db"]
        total_bytes = 0
        for db in db_files:
            p = self.storage_dir / db
            if p.exists():
                total_bytes += p.stat().st_size

        # Delete expired logs if quota approaching
        pressure_ratio = total_bytes / self.project_max_bytes
        if pressure_ratio > 0.8:
            # High storage pressure, execute vacuum and eviction
            self.cache.clear()

            # Clean expired session tool calls (raw process outputs)
            # Safe vacuum to clean DB pages
            with self.session_store._cursor() as cur:
                cur.execute(
                    "DELETE FROM session_tool_calls WHERE timestamp < ?",
                    (
                        datetime.fromtimestamp(
                            time.time() - self.raw_tool_output_retention_days * 86400, UTC
                        ).isoformat(),
                    ),
                )

            with self.event_store._cursor() as cur:
                cur.execute(
                    "DELETE FROM events WHERE timestamp < ?",
                    (
                        datetime.fromtimestamp(
                            time.time() - self.event_retention_days * 86400, UTC
                        ).isoformat(),
                    ),
                )

        return {
            "total_bytes": total_bytes,
            "max_bytes": self.project_max_bytes,
            "pressure_ratio": pressure_ratio,
            "warning": pressure_ratio > 0.9,
        }

    # ── crash recovery status ───────────────────────────────────────────

    def check_crash_recovery(self) -> dict[str, Any] | None:
        """Inspect if the last session tasks completed cleanly."""
        # Look at the active locks and session tasks
        lock_file = self.storage_dir / "locks" / "workspace.lock"
        if not lock_file.exists():
            return None

        try:
            lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
            pid = lock_data.get("process_id")
            # If the process is dead, we have a crash!
            try:
                import os

                os.kill(pid, 0)
                # Process alive, not crashed
                return None
            except OSError:
                # Process is dead! Interrupted state found.
                return {
                    "session_id": lock_data.get("session_id"),
                    "task_id": lock_data.get("task_id"),
                    "process_id": pid,
                    "acquisition_time": lock_data.get("acquisition_time"),
                }
        except Exception:
            return None
