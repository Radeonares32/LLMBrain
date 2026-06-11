"""Persistent cache for structured LLM extraction results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmbrain.services.redactor import redact_model_strings

SCHEMA_VERSION = "2026-06-11-redacted"


class LLMExtractionCache:
    """JSONL-backed cache keyed by provider, model, task, and content hash."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._items: dict[str, dict[str, Any]] = {}
        if enabled:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("key")
                parsed = row.get("parsed")
                if (
                    isinstance(key, str)
                    and key.startswith(SCHEMA_VERSION + "|")
                    and isinstance(parsed, dict)
                ):
                    self._items[key] = parsed

    @staticmethod
    def key(
        *,
        provider_name: str,
        model: str,
        task: str,
        content_hash: str,
    ) -> str:
        return "|".join([SCHEMA_VERSION, provider_name, model, task, content_hash])

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return self._items.get(key)

    def set(self, key: str, parsed: dict[str, Any]) -> None:
        if self.enabled:
            self._items[key] = redact_model_strings(parsed)

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for key in sorted(self._items):
                handle.write(json.dumps({"key": key, "parsed": self._items[key]}) + "\n")
