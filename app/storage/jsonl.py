"""JSONL export — writes model lists as newline-delimited JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_jsonl(path: str | Path, items: list[BaseModel | dict]) -> int:
    """Write *items* as JSONL and return the number of lines written."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            if isinstance(item, BaseModel):
                line = item.model_dump_json()
            else:
                line = json.dumps(item, ensure_ascii=False, default=str)
            fh.write(line + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""

    path = Path(path)
    if not path.exists():
        return []

    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
