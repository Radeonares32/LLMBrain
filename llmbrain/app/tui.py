"""Interactive Terminal User Interface for LLMBrain."""

from __future__ import annotations

import asyncio
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from llmbrain.agent.runtime import AgentRuntime
from llmbrain.agent.runtime import Message as AgentMessage
from llmbrain.agent.safety import SafetyMode
from llmbrain.llm.providers import create_provider
from llmbrain.services.session_service import SessionService


class TuiState:
    """Explicit UI projection and event state."""

    def __init__(self) -> None:
        self.project_name = ""
        self.branch_name = "main"
        self.last_indexed_commit = "unknown"
        self.active_agent = "Build"
        self.active_model = "configured/default"
        self.permission_mode = "ask-before-write"
        self.is_running = True

        # Sessions
        self.sessions: list[dict] = []
        self.selected_session_id = ""
        self.messages: list[dict] = []
        self.tool_calls: list[dict] = []

        # Input Editor
        self.input_buffer = ""
        self.input_cursor = 0
        self.command_history: list[str] = []
        self.history_index = -1

        # Panels & focus
        self.focused_panel = (
            "conversation"  # 'conversation' | 'sessions' | 'brain' | 'help' | 'diff' | 'tests'
        )
        self.scroll_offset = 0
        self.leader_mode = False

        # System status
        self.status_message = "Ready"
        self.status_time = time.time()
        self.indexing_progress = 0
        self.indexing_active = False

        # Streaming and Execution
        self.running_task = False
        self.streaming_text = ""

        # Modal State
        self.modal_active = False  # True when approval or prompt is open
        self.approval_request: dict | None = None
        self.approval_event: asyncio.Event | None = None
        self.approval_decision: str | None = None

        # Views data
        self.diff_content = ""
        self.test_content = ""

        # Phase 6 — Observability
        self.queue_stats: dict = {}       # {status: count}
        self.resource_cpu: float = 0.0
        self.resource_mem: float = 0.0
        self.resource_state: str = "unknown"


class LLMBrainTUI:
    """Local terminal-based user interface using Rich."""

    def __init__(self, project_root: str | Path, provider_name: str | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.session_service = SessionService(self.project_root)
        self.provider_name = provider_name or "openai"

        self.state = TuiState()
        self.state.project_name = self.session_service.identity.get("name", self.project_root.name)
        self.state.active_model = self.provider_name
        self.state.permission_mode = (
            self.session_service.session_store.get_sessions(self.session_service.project_id)[0].get(
                "permission_mode", "ask-before-write"
            )
            if self.session_service.session_store.get_sessions(self.session_service.project_id)
            else "ask-before-write"
        )

        # Detect Git state
        self._detect_git_state()

        self.console = Console()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.input_reader = RawTerminalInput()

        # Load custom commands
        self.custom_commands = self._load_custom_commands()

    def _detect_git_state(self) -> None:
        try:
            import subprocess

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            self.state.branch_name = branch
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            self.state.last_indexed_commit = commit
        except Exception:
            self.state.branch_name = "no-git"
            self.state.last_indexed_commit = "none"

    def _load_custom_commands(self) -> dict[str, dict]:
        commands = {}
        cmd_dir = self.project_root / ".llmbrain" / "commands"
        if cmd_dir.exists():
            for f in cmd_dir.glob("*.md"):
                try:
                    content = f.read_text(encoding="utf-8")
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 3:
                            import yaml

                            meta = yaml.safe_load(parts[1])
                            name = meta.get("name", f.stem)
                            commands[name] = {
                                "description": meta.get("description", ""),
                                "agent": meta.get("agent", "build"),
                                "template": parts[2].strip(),
                            }
                except Exception:
                    pass
        return commands

    # ── view layout generation ──────────────────────────────────────────

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=1), Layout(name="body"), Layout(name="footer", size=3)
        )

        layout["body"].split_row(Layout(name="sidebar", ratio=1), Layout(name="main", ratio=3))

        layout["sidebar"].split(
            Layout(name="sessions", ratio=1),
            Layout(name="brain", ratio=1),
            Layout(name="observe", ratio=1),
        )

        return layout

    def _render_header(self) -> Panel:
        status_txt = f"Status: [cyan]{self.state.status_message}[/cyan]"
        if self.state.indexing_active:
            status_txt = f"Indexing: [yellow]{self.state.indexing_progress}%[/yellow]"

        header = Text.assemble(
            (" LLMBrain ", "bold reverse cyan"),
            f" │ Project: [green]{self.state.project_name}[/green]",
            f" │ Branch: [magenta]{self.state.branch_name}[/magenta]",
            f" │ Agent: [yellow]{self.state.active_agent}[/yellow]",
            f" │ Model: [blue]{self.state.active_model}[/blue]",
            f" │ Permissions: [red]{self.state.permission_mode}[/red]  ",
            Text.from_markup(status_txt, justify="right"),
        )
        return Panel(header, style="white on blue", box=None)

    def _render_sessions(self) -> Panel:
        session_lines = []
        for s in self.state.sessions:
            is_sel = s["id"] == self.state.selected_session_id
            prefix = "> " if is_sel else "  "
            title = s["title"]
            if len(title) > 22:
                title = title[:19] + "..."
            style = "bold cyan" if is_sel else "white"
            session_lines.append(Text(f"{prefix}{title}", style=style))

        if not session_lines:
            session_lines.append(Text("  No sessions found.", style="dim italic"))

        border_style = "green" if self.state.focused_panel == "sessions" else "dim"
        return Panel(Group(*session_lines), title="Sessions (Ctrl+X S)", border_style=border_style)

    def _render_brain(self) -> Panel:
        stats = self.session_service.cache.stats()
        quota = self.session_service.enforce_quotas()

        # Get active memory facts count
        facts_count = len(self.session_service.store.get_facts(self.session_service.project_id))

        lines = [
            f"Memories: [green]{facts_count}[/green]",
            f"Cache hit/miss: [cyan]{stats.hits}/{stats.misses}[/cyan]",
            f"Cache size: [blue]{stats.current_bytes // 1024} KB[/blue]",
            f"Storage used: [yellow]{quota['total_bytes'] // 1024} KB[/yellow]",
            f"Quota pressure: [magenta]{int(quota['pressure_ratio'] * 100)}%[/magenta]",
        ]

        border_style = "green" if self.state.focused_panel == "brain" else "dim"
        return Panel(
            Group(*[Text.from_markup(line) for line in lines]),
            title="Project Brain (Ctrl+X B)",
            border_style=border_style,
        )

    def _render_observe(self) -> Panel:
        """Render Phase 6 live observability panel (queue + resource stats)."""
        state_color = {"normal": "green", "degraded": "yellow", "critical": "red"}.get(
            self.state.resource_state, "dim"
        )
        lines = [
            (
                f"CPU: [cyan]{self.state.resource_cpu:.1f}%[/cyan]  "
                f"RAM: [cyan]{self.state.resource_mem:.1f}%[/cyan]  "
                f"State: [{state_color}]{self.state.resource_state}[/{state_color}]"
            ),
        ]

        q = self.state.queue_stats
        if q:
            pending = q.get("PENDING", 0)
            running = q.get("RUNNING", 0)
            completed = q.get("COMPLETED", 0)
            failed = q.get("FAILED", 0)
            lines.append(
                f"Q: [yellow]{pending}[/yellow] wait  "
                f"[cyan]{running}[/cyan] run  "
                f"[green]{completed}[/green] done  "
                f"[red]{failed}[/red] err"
            )
        else:
            lines.append("Queue: [dim]boş[/dim]")

        border_style = "green" if self.state.focused_panel == "observe" else "dim"
        return Panel(
            Group(*[Text.from_markup(line) for line in lines]),
            title="Gözlem (Ctrl+X O)",
            border_style=border_style,
        )

    def _render_main(self) -> Panel:
        if self.state.focused_panel == "diff":
            # Render diff view
            content = self.state.diff_content or "No active git diff changes."
            return Panel(
                Text(content, style="green"), title="Git Diff View (Ctrl+X D)", border_style="green"
            )

        if self.state.focused_panel == "tests":
            # Render tests panel
            content = self.state.test_content or "No test execution history."
            return Panel(
                Text(content), title="Tests & Diagnostics (Ctrl+X T)", border_style="green"
            )

        if self.state.focused_panel == "help":
            help_lines = [
                "LLMBrain TUI Commands & Shortcuts:",
                "",
                "Ctrl+X N : Yeni oturum",
                "Ctrl+X S : Oturum paneli",
                "Ctrl+X B : Brain istatistikleri",
                "Ctrl+X O : Gözlem paneli (CPU/Kuyruk)",
                "Ctrl+X D : Diff görünümü",
                "Ctrl+X T : Test/Diagnostics",
                "Ctrl+X P : İzin modu değiştir",
                "Ctrl+X ? : Yardım paneli",
                "Esc      : Konuşmaya dön / Modal kapat",
                "Ctrl+C   : Güvenli çıkış",
                "",
                "Slash commands:",
                "  /new, /rename <title>, /archive, /delete",
                "  /agent <ask|plan|build|review>, /model <name>",
                "  /index, /refresh, /cache, /compact, /diff, /tests, /exit",
            ]
            return Panel(
                Group(*[Text(line) for line in help_lines]),
                title="Help Panel (Ctrl+X ?)",
                border_style="green",
            )

        # Otherwise render conversation view
        elements = []
        for msg in self.state.messages:
            role = msg["role"].upper()
            content = msg["content"]
            color = "green" if role == "USER" else "cyan" if role == "ASSISTANT" else "yellow"
            elements.append(Text(f"{role}:", style=f"bold {color}"))
            elements.append(Text(content))
            elements.append(Text("-" * 40, style="dim"))

        if self.state.streaming_text:
            elements.append(Text("BUILD AGENT (streaming):", style="bold yellow"))
            elements.append(Text(self.state.streaming_text))

        if not elements:
            elements.append(
                Text("Welcome to LLMBrain TUI! Ask a question or type /help.", style="dim italic")
            )

        # Render floating modal overlay if active
        if self.state.modal_active and self.state.approval_request:
            req = self.state.approval_request
            modal_lines = [
                "┌────────────────────────────────────────────────────────────┐",
                "│             ⚠️  SECURITY APPROVAL REQUIRED                   │",
                "├────────────────────────────────────────────────────────────┤",
                f"│ Agent wishes to execute: [bold yellow]{req.get('tool_name')}[/bold yellow]",
                f"│ Arguments: {req.get('arguments')}",
                "│                                                            │",
                "│ [A] Approve once  [S] Approve session  [D] Deny            │",
                "└────────────────────────────────────────────────────────────┘",
            ]
            elements.append(Text(""))
            elements.append(
                Group(*[Text.from_markup(line, justify="center") for line in modal_lines])
            )

        border_style = "green" if self.state.focused_panel == "conversation" else "dim"
        return Panel(Group(*elements), title="Conversation", border_style=border_style)

    def _render_footer(self) -> Panel:
        cursor_buffer = list(self.state.input_buffer)
        if self.state.input_cursor <= len(cursor_buffer):
            cursor_buffer.insert(self.state.input_cursor, "█")
        input_line = "".join(cursor_buffer)

        prompt = Text.assemble(("> ", "bold green"), (input_line, "white"))
        shortcuts = "Ctrl+X ? Help │ Ctrl+X N New │ Ctrl+X S Sessions │ Ctrl+X D Diff │ Ctrl+C Exit"
        footer_group = Group(prompt, Text(shortcuts, style="dim", justify="center"))
        return Panel(footer_group, box=None)

    def draw(self, live: Live) -> None:
        layout = self._make_layout()
        layout["header"].update(self._render_header())
        layout["sessions"].update(self._render_sessions())
        layout["brain"].update(self._render_brain())
        layout["observe"].update(self._render_observe())
        layout["main"].update(self._render_main())
        layout["footer"].update(self._render_footer())
        live.update(layout)

    # ── TUI Event Loop ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize TUI layout and start asynchronous input reader loop."""
        self._loop = asyncio.get_running_loop()

        # Load initial sessions
        self.state.sessions = self.session_service.list_sessions()
        if self.state.sessions:
            self.state.selected_session_id = self.state.sessions[0]["id"]
            self.state.messages = self.session_service.get_messages(self.state.selected_session_id)
        else:
            # Create default first session
            sess = self.session_service.create_session(
                "Default Session", "build", {}, "ask-before-write"
            )
            self.state.sessions = [sess]
            self.state.selected_session_id = sess["id"]
            self.state.messages = []

        # Run fast incremental indexing on start
        self._loop.create_task(self._startup_indexing())
        # Phase 6: live observability refresh
        self._loop.create_task(self._refresh_observability())

        self.input_reader.enable()
        try:
            with Live(self._make_layout(), console=self.console, refresh_per_second=10) as live:
                while self.state.is_running:
                    self.draw(live)
                    char = self.input_reader.read_char()
                    if char:
                        await self._handle_input(char)
                    await asyncio.sleep(0.05)
        finally:
            self.input_reader.disable()

    async def _startup_indexing(self) -> None:
        self.state.indexing_active = True
        self.state.status_message = "Indexing project..."
        try:
            # Async indexing simulation
            for i in range(1, 11):
                self.state.indexing_progress = i * 10
                await asyncio.sleep(0.1)
            # Actually run incremental build
            self.session_service.session_store.get_sessions(self.session_service.project_id)
            self.session_service.enforce_quotas()
            self.state.status_message = "Project indexed successfully"
        except Exception as e:
            self.state.status_message = f"Indexing failed: {e}"
        finally:
            self.state.indexing_active = False
            await asyncio.sleep(2.0)
            self.state.status_message = "Hazır"

    async def _refresh_observability(self) -> None:
        """Background task: refresh queue and resource stats every 3 seconds."""
        from llmbrain.core.resource_manager import ResourceManager

        rm = ResourceManager()
        while self.state.is_running:
            try:
                rm.sample()
                stats = rm.get_stats()
                snap = rm.snapshots[-1] if rm.snapshots else None
                self.state.resource_state = stats.get("state", "unknown")
                self.state.resource_cpu = snap.cpu_percent if snap else 0.0
                self.state.resource_mem = snap.memory_percent if snap else 0.0
            except Exception:
                pass

            try:
                from llmbrain.core.identity import (
                    get_project_storage_dir,
                    load_or_create_project_identity,
                )
                from llmbrain.core.queue import IndexQueue

                identity = load_or_create_project_identity(self.project_root)
                pid = identity["project_id"]
                db_path = get_project_storage_dir(pid) / "queue.db"
                if db_path.exists():
                    q = IndexQueue(db_path)
                    self.state.queue_stats = q.stats(pid)
            except Exception:
                pass

            await asyncio.sleep(3.0)

    async def _handle_input(self, char: str) -> None:
        # Leader Mode Handling
        if self.state.leader_mode:
            self.state.leader_mode = False
            c = char.lower()
            if c == "n":
                # New session
                sess = self.session_service.create_session(
                    f"Session {len(self.state.sessions) + 1}",
                    "build",
                    {},
                    self.state.permission_mode,
                )
                self.state.sessions = self.session_service.list_sessions()
                self.state.selected_session_id = sess["id"]
                self.state.messages = []
                self.state.focused_panel = "conversation"
            elif c == "s":
                self.state.focused_panel = "sessions"
            elif c == "b":
                self.state.focused_panel = "brain"
            elif c == "d":
                self.state.focused_panel = "diff"
                # Load diff
                try:
                    from llmbrain.services.git_diff import analyze_git_diff

                    diffs = analyze_git_diff(str(self.project_root))
                    self.state.diff_content = (
                        "\n".join(f"{d.status} : {d.path}" for d in diffs)
                        if diffs
                        else "No modified files."
                    )
                except Exception:
                    self.state.diff_content = "Git diff inspection unavailable."
            elif c == "t":
                self.state.focused_panel = "tests"
                self.state.test_content = "Running pytest...\n"
                self._loop.create_task(self._run_tests_background())
            elif c == "p":
                # Toggle permissions
                modes = ["read-only", "ask-before-write", "trusted-project"]
                idx = (modes.index(self.state.permission_mode) + 1) % len(modes)
                self.state.permission_mode = modes[idx]
                if self.state.selected_session_id:
                    self.session_service.update_session(
                        self.state.selected_session_id, permission_mode=self.state.permission_mode
                    )
            elif c == "?":
                self.state.focused_panel = "help"
            elif c == "o":
                self.state.focused_panel = "observe"
            return

        # Handle Modal Inputs
        if self.state.modal_active and self.state.approval_event:
            c = char.lower()
            if c == "a":
                self.state.approval_decision = "approve_once"
                self.state.modal_active = False
                self.state.approval_event.set()
            elif c == "s":
                self.state.approval_decision = "approve_session"
                self.state.modal_active = False
                self.state.approval_event.set()
            elif c == "d":
                self.state.approval_decision = "deny"
                self.state.modal_active = False
                self.state.approval_event.set()
            return

        # Standard inputs
        if char == "\x03":  # Ctrl+C
            self.state.is_running = False
            return
        elif char == "\x18":  # Ctrl+X
            self.state.leader_mode = True
            return
        elif char == "\x1b":  # Esc
            self.state.focused_panel = "conversation"
            return
        elif char in ("\r", "\n"):
            # Submit input buffer
            cmd = self.state.input_buffer.strip()
            if cmd:
                self.state.command_history.append(self.state.input_buffer)
                self.state.history_index = -1
                self.state.input_buffer = ""
                self.state.input_cursor = 0
                await self._execute_command(cmd)
        elif char == "\x7f" or char == "\x08":  # Backspace
            if self.state.input_cursor > 0:
                self.state.input_buffer = (
                    self.state.input_buffer[: self.state.input_cursor - 1]
                    + self.state.input_buffer[self.state.input_cursor :]
                )
                self.state.input_cursor -= 1
        elif char == "\x1b[A":  # Arrow Up
            if self.state.focused_panel == "sessions":
                # Move selected session
                if self.state.sessions:
                    idx = next(
                        i
                        for i, s in enumerate(self.state.sessions)
                        if s["id"] == self.state.selected_session_id
                    )
                    idx = (idx - 1) % len(self.state.sessions)
                    self.state.selected_session_id = self.state.sessions[idx]["id"]
                    self.state.messages = self.session_service.get_messages(
                        self.state.selected_session_id
                    )
            else:
                # History recall
                if self.state.command_history:
                    if self.state.history_index == -1:
                        self.state.history_index = len(self.state.command_history) - 1
                    else:
                        self.state.history_index = max(0, self.state.history_index - 1)
                    self.state.input_buffer = self.state.command_history[self.state.history_index]
                    self.state.input_cursor = len(self.state.input_buffer)
        elif char == "\x1b[B":  # Arrow Down
            if self.state.focused_panel == "sessions":
                if self.state.sessions:
                    idx = next(
                        i
                        for i, s in enumerate(self.state.sessions)
                        if s["id"] == self.state.selected_session_id
                    )
                    idx = (idx + 1) % len(self.state.sessions)
                    self.state.selected_session_id = self.state.sessions[idx]["id"]
                    self.state.messages = self.session_service.get_messages(
                        self.state.selected_session_id
                    )
            else:
                if self.state.command_history:
                    if self.state.history_index != -1:
                        if self.state.history_index == len(self.state.command_history) - 1:
                            self.state.history_index = -1
                            self.state.input_buffer = ""
                        else:
                            self.state.history_index = min(
                                len(self.state.command_history) - 1, self.state.history_index + 1
                            )
                            self.state.input_buffer = self.state.command_history[
                                self.state.history_index
                            ]
                        self.state.input_cursor = len(self.state.input_buffer)
        elif char == "\x1b[C":  # Arrow Right
            self.state.input_cursor = min(len(self.state.input_buffer), self.state.input_cursor + 1)
        elif char == "\x1b[D":  # Arrow Left
            self.state.input_cursor = max(0, self.state.input_cursor - 1)
        else:
            # Printable chars
            if len(char) == 1 and char.isprintable():
                self.state.input_buffer = (
                    self.state.input_buffer[: self.state.input_cursor]
                    + char
                    + self.state.input_buffer[self.state.input_cursor :]
                )
                self.state.input_cursor += 1

    async def _run_tests_background(self) -> None:
        try:
            import subprocess

            res = await self._loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["pytest", "-q"], cwd=str(self.project_root), capture_output=True, text=True
                ),
            )
            self.state.test_content = res.stdout or res.stderr or "pytest finished with no output."
        except Exception as e:
            self.state.test_content = f"Test execution failed: {e}"

    async def _execute_command(self, cmd: str) -> None:
        if cmd.startswith("/"):
            # Command parsing
            parts = cmd.split(" ", 1)
            slash_cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if slash_cmd == "/exit":
                self.state.is_running = False
            elif slash_cmd == "/help":
                self.state.focused_panel = "help"
            elif slash_cmd == "/new":
                sess = self.session_service.create_session(
                    "New Session", self.state.active_agent, {}, self.state.permission_mode
                )
                self.state.sessions = self.session_service.list_sessions()
                self.state.selected_session_id = sess["id"]
                self.state.messages = []
            elif slash_cmd == "/rename":
                if args and self.state.selected_session_id:
                    self.session_service.rename_session(self.state.selected_session_id, args)
                    self.state.sessions = self.session_service.list_sessions()
            elif slash_cmd == "/delete":
                if self.state.selected_session_id:
                    self.session_service.delete_session(self.state.selected_session_id)
                    self.state.sessions = self.session_service.list_sessions()
                    if self.state.sessions:
                        self.state.selected_session_id = self.state.sessions[0]["id"]
                        self.state.messages = self.session_service.get_messages(
                            self.state.selected_session_id
                        )
                    else:
                        self.state.selected_session_id = ""
                        self.state.messages = []
            elif slash_cmd == "/agent":
                if args:
                    self.state.active_agent = args
                    if self.state.selected_session_id:
                        self.session_service.update_session(
                            self.state.selected_session_id, active_agent=args
                        )
            elif slash_cmd == "/model":
                if args:
                    self.state.active_model = args
                    if self.state.selected_session_id:
                        self.session_service.update_session(
                            self.state.selected_session_id, model_config={"model": args}
                        )
            elif slash_cmd == "/permissions":
                if args in ("read-only", "ask-before-write", "trusted-project"):
                    self.state.permission_mode = args
                    if self.state.selected_session_id:
                        self.session_service.update_session(
                            self.state.selected_session_id, permission_mode=args
                        )
            elif slash_cmd == "/index":
                self._loop.create_task(self._startup_indexing())
            elif slash_cmd == "/compact":
                if self.state.selected_session_id:
                    self.state.status_message = "Compacting session context..."
                    await self.session_service.compact_session(self.state.selected_session_id)
                    self.state.messages = self.session_service.get_messages(
                        self.state.selected_session_id
                    )
                    self.state.status_message = "Session compacted successfully."
            elif slash_cmd == "/diff":
                self.state.focused_panel = "diff"
                # Load diff
                try:
                    from llmbrain.services.git_diff import analyze_git_diff

                    diffs = analyze_git_diff(str(self.project_root))
                    self.state.diff_content = (
                        "\n".join(f"{d.status} : {d.path}" for d in diffs)
                        if diffs
                        else "No modified files."
                    )
                except Exception:
                    self.state.diff_content = "Git diff inspection unavailable."
            elif slash_cmd == "/tests":
                self.state.focused_panel = "tests"
                self.state.test_content = "Running pytest...\n"
                self._loop.create_task(self._run_tests_background())
            elif slash_cmd in self.custom_commands:
                # Custom Command Expansion
                custom = self.custom_commands[slash_cmd]
                prompt = custom["template"].replace("{{ scope }}", args)
                self.state.focused_panel = "conversation"
                self._loop.create_task(self._run_agent_task(prompt))
            else:
                self.state.status_message = f"Unknown command: {slash_cmd}"
        else:
            # Run task in background
            self.state.focused_panel = "conversation"
            self._loop.create_task(self._run_agent_task(cmd))

    async def _run_agent_task(self, prompt: str) -> None:
        if self.state.running_task:
            self.state.status_message = "A task is already running!"
            return

        self.state.running_task = True
        self.state.status_message = "Agent running..."

        # Save user message to database
        self.session_service.add_message(self.state.selected_session_id, "user", prompt)
        self.state.messages = self.session_service.get_messages(self.state.selected_session_id)

        try:
            # 1. Acquire workspace lock
            if not self.session_service.lock.acquire(
                self.state.selected_session_id, self.state.selected_session_id, timeout_seconds=2.0
            ):
                self.session_service.add_message(
                    self.state.selected_session_id,
                    "assistant",
                    "Error: Workspace is locked by another LLMBrain process.",
                )
                self.state.messages = self.session_service.get_messages(
                    self.state.selected_session_id
                )
                return

            # 2. Build prior messages from database/compaction state
            prior_messages = []
            session = self.session_service.get_session(self.state.selected_session_id)
            compaction = session.get("compaction_state") if session else None

            if compaction and compaction.get("conversation_summary"):
                summary_content = (
                    f"=== PRIOR CONVERSATION SUMMARY ===\n"
                    f"Summary: {compaction['conversation_summary']}\n"
                    f"Completed: {', '.join(compaction.get('completed_objectives', []))}\n"
                    f"Decisions: {', '.join(compaction.get('decisions', []))}\n"
                    f"Unresolved: {', '.join(compaction.get('unresolved_objectives', []))}\n"
                    f"===================================="
                )
                prior_messages.append(AgentMessage(role="system", content=summary_content))

                # Fetch recent messages since compaction
                compacted_at = compaction.get("compacted_at")
                for m in self.state.messages:
                    if m["timestamp"] > compacted_at:
                        prior_messages.append(AgentMessage(role=m["role"], content=m["content"]))
            else:
                for m in self.state.messages:
                    prior_messages.append(AgentMessage(role=m["role"], content=m["content"]))

            # 3. Setup Agent Runtime
            llm = create_provider(self.state.active_model)

            # Custom prompt interceptor for TUI approval workflow
            def intercept_prompt(msg: str) -> bool:
                # Capture info
                self.state.approval_request = {"tool_name": "apply_patch/shell", "arguments": msg}
                self.state.approval_event = asyncio.Event()
                self.state.modal_active = True

                # Wait for user key choice in the input loop
                # Run sync wait on event in the loop executor
                async def wait_decision():
                    await self.state.approval_event.wait()
                    return self.state.approval_decision

                fut = asyncio.run_coroutine_threadsafe(wait_decision(), self._loop)
                decision = fut.result()

                self.state.modal_active = False
                self.state.approval_request = None

                if decision == "approve_once":
                    return True
                elif decision == "approve_session":
                    # Elevate safety policy
                    self.state.permission_mode = "trusted-project"
                    self.session_service.update_session(
                        self.state.selected_session_id, permission_mode="trusted-project"
                    )
                    return True
                else:
                    return False

            # Custom event subscriber to project streaming responses
            def event_listener(event: Any) -> None:
                if event.event_type == "model_response_received":
                    # Capture streaming fragments or token updates
                    pass
                elif event.event_type == "tool_execution_started":
                    self.session_service.add_tool_call(
                        self.state.selected_session_id,
                        event.payload.get("tool_name", "unknown"),
                        event.payload.get("arguments", {}),
                        status="requested",
                    )

            agent = AgentRuntime(
                project_root=self.project_root,
                provider=llm,
                safety_mode=SafetyMode(self.state.permission_mode),
                prompt_func=intercept_prompt,
                event_listener=event_listener,
                agent_name=self.state.active_agent.lower(),
            )

            # Execute task
            record = await agent.execute_task(user_request=prompt, prior_messages=prior_messages)

            # Persist response
            self.session_service.add_message(
                self.state.selected_session_id, "assistant", record.summary
            )
            self.state.messages = self.session_service.get_messages(self.state.selected_session_id)
            self.state.status_message = "Ready"
        except Exception as e:
            self.session_service.add_message(
                self.state.selected_session_id, "assistant", f"Error: {e}"
            )
            self.state.messages = self.session_service.get_messages(self.state.selected_session_id)
            self.state.status_message = "Failed"
        finally:
            self.session_service.lock.release(self.state.selected_session_id)
            self.state.running_task = False


# ── Raw Terminal Input Parser ────────────────────────────────────────


class RawTerminalInput:
    """Read keys from standard input using raw mode fileno flags."""

    def __init__(self) -> None:
        self.old_settings = None

    def enable(self) -> None:
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())

    def disable(self) -> None:
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def read_char(self) -> str | None:
        """Non-blocking check for single char or escape sequence."""
        if not select.select([sys.stdin], [], [], 0.02)[0]:
            return None
        char = sys.stdin.read(1)
        if char == "\x1b":
            # Check for escape sequence
            if select.select([sys.stdin], [], [], 0.01)[0]:
                char += sys.stdin.read(2)
        return char
