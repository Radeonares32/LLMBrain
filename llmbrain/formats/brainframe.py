"""BrainFrame compact LLM context renderer.

BrainFrame is a TOON/JTON-style table format for LLM input. It keeps SQLite
and JSONL as canonical storage while avoiding repeated JSON keys in prompts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from llmbrain.services.token_budget import estimate_tokens

DEFAULT_CELL_MAX_CHARS = 240
DEFAULT_CONTEXT_MAX_CHARS = 120_000


def _get(value: Any, key: str, default: Any = "") -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def escape_cell(value: Any, max_chars: int = DEFAULT_CELL_MAX_CHARS) -> str:
    """Escape a BrainFrame table cell."""

    if value is None:
        return ""
    text = str(value).replace("|", r"\|").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def format_evidence(path: str, start_line: int, end_line: int) -> str:
    """Format source evidence as a single compact cell."""

    if not path:
        return ""
    if start_line and end_line:
        return f"{path}:L{start_line}-L{end_line}"
    return path


def _fact_evidence(fact: Any) -> str:
    evidence = _get(fact, "evidence", []) or []
    first = evidence[0] if evidence else None
    if first is None:
        return ""
    return format_evidence(
        str(_get(first, "path", "")),
        int(_get(first, "start_line", 0) or 0),
        int(_get(first, "end_line", 0) or 0),
    )


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [" | ".join(headers)]
    for row in rows:
        lines.append(" | ".join(escape_cell(cell) for cell in row))
    return "\n".join(lines)


def _relation_rows(relations: list[Any], entity_index: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for relation in relations:
        source_id = str(_get(relation, "source_entity_id", ""))
        target_id = str(_get(relation, "target_entity_id", ""))
        source = entity_index.get(source_id)
        target = entity_index.get(target_id)
        rows.append(
            [
                _get(source, "name", source_id[:8]) if source else source_id[:8],
                _get(relation, "relation", ""),
                _get(target, "name", target_id[:8]) if target else target_id[:8],
                _get(relation, "evidence", ""),
                _get(relation, "confidence", ""),
            ]
        )
    return rows


def _fact_priority(fact: Any) -> int:
    confidence = str(_get(fact, "confidence", "")).lower()
    has_evidence = bool(_get(fact, "evidence", []) or [])
    if confidence == "high" and has_evidence:
        return 0
    if confidence in {"high", "medium"}:
        return 1
    return 2


def _fact_rows(facts: list[Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for fact in sorted(facts, key=_fact_priority):
        rows.append(
            [
                str(_get(fact, "id", ""))[:8],
                _get(fact, "subject", ""),
                _get(fact, "predicate", ""),
                _get(fact, "object", ""),
                _fact_evidence(fact),
                _get(fact, "confidence", ""),
            ]
        )
    return rows


def _section(title: str, headers: list[str], rows: list[list[Any]]) -> str:
    return f"#{title}\n" + _table(headers, rows)


def _fits(parts: list[str], candidate: str, max_chars: int, max_tokens: int | None = None, truncated: bool = False) -> bool:
    suffix = "\n@truncated true\n" if truncated else "\n"
    content = "\n\n".join([*parts, candidate]) + suffix
    if max_tokens is not None:
        return estimate_tokens(content) <= max_tokens
    return len(content) <= max_chars


def _append_section(
    parts: list[str],
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    max_chars: int,
    max_tokens: int | None = None,
) -> bool:
    accepted: list[list[Any]] = []
    truncated = False
    for row in rows:
        candidate = _section(title, headers, [*accepted, row])
        if _fits(parts, candidate, max_chars, max_tokens):
            accepted.append(row)
            continue
        truncated = True
        break
    candidate = _section(title, headers, accepted)
    if not _fits(parts, candidate, max_chars, max_tokens, truncated=truncated):
        candidate = _section(title, headers, [])
        truncated = truncated or bool(rows)
    parts.append(candidate)
    return truncated


def build_brainframe_context(
    project_name: str,
    project_id: str,
    entities: list[Any],
    relations: list[Any],
    facts: list[Any],
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    max_tokens: int | None = None,
) -> str:
    """Build compact BrainFrame context without JSON or repeated keys."""

    entity_index = {str(_get(entity, "id", "")): entity for entity in entities}
    parts = [
        f"@project {escape_cell(project_name)}",
        f"@project_id {escape_cell(project_id)}",
        "@type engineering_knowledge_compiler",
    ]

    truncated = False
    entity_rows = [
        [
            str(_get(entity, "id", ""))[:8],
            _get(entity, "type", ""),
            _get(entity, "name", ""),
            _get(entity, "path", ""),
            _get(entity, "confidence", ""),
        ]
        for entity in entities
    ]
    truncated |= _append_section(
        parts,
        "entities",
        ["id", "type", "name", "path", "confidence"],
        entity_rows,
        max_chars,
        max_tokens,
    )
    truncated |= _append_section(
        parts,
        "relations",
        ["from", "relation", "to", "evidence", "confidence"],
        _relation_rows(relations, entity_index),
        max_chars,
        max_tokens,
    )
    truncated |= _append_section(
        parts,
        "facts",
        ["id", "subject", "predicate", "object", "evidence", "confidence"],
        _fact_rows(facts),
        max_chars,
        max_tokens,
    )

    output = "\n\n".join(parts)
    if truncated:
        output += "\n@truncated true"
    return output + "\n"


def build_compact_context(
    project_name: str,
    entities: list[Any],
    relations: list[Any],
    facts: list[Any],
    *,
    entity_index: dict[str, Any] | None = None,
    project_id: str = "",
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    max_tokens: int | None = None,
) -> str:
    """Backward-compatible alias for older callers."""

    _ = entity_index
    return build_brainframe_context(project_name, project_id, entities, relations, facts, max_chars, max_tokens)


__all__ = [
    "build_brainframe_context",
    "build_compact_context",
    "escape_cell",
    "format_evidence",
]
