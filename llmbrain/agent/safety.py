"""Safety management for the LLMBrain agent runtime."""

from collections.abc import Callable
from enum import StrEnum


class PermissionLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    DESTRUCTIVE = "destructive"
    EXECUTE_SAFE = "execute_safe"
    EXECUTE_NETWORK = "execute_network"
    PROHIBITED = "prohibited"


class SafetyMode(StrEnum):
    READ_ONLY = "read-only"
    ASK_BEFORE_WRITE = "ask-before-write"
    TRUSTED_PROJECT = "trusted-project"
    DENY_SHELL = "deny-shell"


class SafetyManager:
    """Manages permissions and prompts for tool execution."""

    def __init__(
        self,
        mode: SafetyMode = SafetyMode.ASK_BEFORE_WRITE,
        prompt_func: Callable[[str], bool] | None = None,
    ) -> None:
        self.mode = mode
        self.prompt_func = prompt_func or self._default_prompt

    def _default_prompt(self, msg: str) -> bool:
        """Standard interactive terminal prompt."""
        print(f"\n⚠️  [GÜVENLİK UYARISI] {msg}")
        try:
            ans = input("Bu işleme izin veriyor musunuz? (y/N): ").strip().lower()
            return ans in ("y", "yes", "evet", "e")
        except (KeyboardInterrupt, EOFError):
            return False

    def check_permission(self, tool_name: str, level: PermissionLevel, details: str) -> bool:
        """Validate if the agent has permission to execute a tool with the given arguments."""
        if level == PermissionLevel.PROHIBITED:
            print(f"❌ '{tool_name}' aracı engellendi (Seviye: prohibited)")
            return False

        if self.mode == SafetyMode.READ_ONLY:
            if level == PermissionLevel.READ:
                return True
            print(f"❌ '{tool_name}' aracı engellendi (Mod: {self.mode.value})")
            return False

        msg = f"Araç: {tool_name} (İzin: {level.value})\nDetay: {details}"

        if self.mode == SafetyMode.DENY_SHELL:
            if level in (
                PermissionLevel.SHELL,
                PermissionLevel.EXECUTE_SAFE,
                PermissionLevel.EXECUTE_NETWORK,
                PermissionLevel.DESTRUCTIVE,
            ):
                print(f"❌ '{tool_name}' aracı engellendi (Mod: {self.mode.value})")
                return False
            if level == PermissionLevel.READ:
                return True
            return self.prompt_func(msg)

        if self.mode == SafetyMode.TRUSTED_PROJECT:
            if level in (PermissionLevel.READ, PermissionLevel.WRITE):
                return True
            if level in (
                PermissionLevel.SHELL,
                PermissionLevel.EXECUTE_SAFE,
                PermissionLevel.EXECUTE_NETWORK,
                PermissionLevel.DESTRUCTIVE,
            ):
                return self.prompt_func(msg)
            return False

        # SafetyMode.ASK_BEFORE_WRITE (Default)
        if level == PermissionLevel.READ:
            return True
        if level in (
            PermissionLevel.WRITE,
            PermissionLevel.SHELL,
            PermissionLevel.EXECUTE_SAFE,
            PermissionLevel.EXECUTE_NETWORK,
            PermissionLevel.DESTRUCTIVE,
        ):
            return self.prompt_func(msg)
        return False


def is_destructive_command(command: str) -> bool:
    """Detect destructive shell commands or git commands."""
    cmd = command.strip().lower()
    destructive_patterns = [
        "rm -rf /",
        "rm -rf *",
        "force",
        "push -f",
        "push --force",
        "reset --hard",
        "rebase",
        "publish",
    ]
    return any(p in cmd for p in destructive_patterns)
