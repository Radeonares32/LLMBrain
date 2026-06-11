"""Compact LLM-context builder (TOON / JTON–style brainframe format).

Produces a token-efficient text representation of the knowledge base
that avoids the key-repetition overhead of JSON.
"""

from __future__ import annotations

from llmbrain.models.entity import Entity
from llmbrain.models.fact import Fact
from llmbrain.models.relation import Relation


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a compact pipe-delimited table."""
    lines = [" | ".join(headers)]
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def build_compact_context(
    project_name: str,
    entities: list[Entity],
    relations: list[Relation],
    facts: list[Fact],
    *,
    entity_index: dict[str, Entity] | None = None,
) -> str:
    """Return the full compact context string."""

    if entity_index is None:
        entity_index = {e.id: e for e in entities}

    sections: list[str] = [
        f"@project {project_name}",
        "@type engineering_knowledge_compiler",
    ]

    # ── entities ────────────────────────────────────────────────────────
    if entities:
        headers = ["id", "type", "name", "path", "confidence"]
        rows = [
            [e.id[:8], e.type, e.name, e.path, e.confidence]
            for e in entities
        ]
        sections.append("\n#entities\n" + _table(headers, rows))

    # ── relations ───────────────────────────────────────────────────────
    if relations:
        headers = ["from", "relation", "to", "evidence", "confidence"]
        rows = []
        for r in relations:
            src = entity_index.get(r.source_entity_id)
            tgt = entity_index.get(r.target_entity_id)
            rows.append([
                src.name if src else r.source_entity_id[:8],
                r.relation,
                tgt.name if tgt else r.target_entity_id[:8],
                r.evidence,
                r.confidence,
            ])
        sections.append("\n#relations\n" + _table(headers, rows))

    # ── facts ───────────────────────────────────────────────────────────
    if facts:
        headers = ["id", "subject", "predicate", "object", "evidence", "confidence"]
        rows = []
        for f in facts:
            ev_str = ""
            if f.evidence:
                e = f.evidence[0]
                ev_str = f"{e.path}:L{e.start_line}-L{e.end_line}"
            rows.append([f.id[:8], f.subject, f.predicate, f.object, ev_str, f.confidence])
        sections.append("\n#facts\n" + _table(headers, rows))

    return "\n\n".join(sections) + "\n"
