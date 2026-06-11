"""Fact extractor — extracts subject-predicate-object claims from chunks.

MVP uses a heuristic extractor.  The interface is designed so that a
real LLM adapter can be swapped in later.
"""

from __future__ import annotations

import re
from uuid import uuid4

from llmbrain.models.chunk import Chunk
from llmbrain.models.fact import Fact, FactEvidence

# ── Heuristic patterns for extraction ───────────────────────────────────────

_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+([\w.]+)",
    re.MULTILINE,
)
_FUNC_DEF_RE = re.compile(
    r"^\s*(?:def|func|function|fn|public\s+\w+|private\s+\w+)\s+(\w+)",
    re.MULTILINE,
)
_CLASS_RE = re.compile(
    r"^\s*(?:class|struct|interface|type)\s+(\w+)",
    re.MULTILINE,
)
_ENV_VAR_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]{2,})\b",
)


def extract_facts_from_chunk(chunk: Chunk, project_id: str) -> list[Fact]:
    """Return heuristic facts from a single chunk."""

    facts: list[Fact] = []
    evidence = FactEvidence(
        id=uuid4().hex,
        fact_id="",  # will be set below
        document_id=chunk.document_id,
        path=chunk.path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
    )

    # imports → "file depends_on module"
    for m in _IMPORT_RE.finditer(chunk.content):
        fid = uuid4().hex
        ev = evidence.model_copy(update={"id": uuid4().hex, "fact_id": fid})
        facts.append(
            Fact(
                id=fid,
                project_id=project_id,
                subject=chunk.path,
                predicate="imports",
                object=m.group(1),
                claim=f"{chunk.path} imports {m.group(1)}",
                confidence="high",
                evidence=[ev],
            )
        )

    # function defs → "file defines function"
    for m in _FUNC_DEF_RE.finditer(chunk.content):
        fid = uuid4().hex
        ev = evidence.model_copy(update={"id": uuid4().hex, "fact_id": fid})
        facts.append(
            Fact(
                id=fid,
                project_id=project_id,
                subject=chunk.path,
                predicate="defines_function",
                object=m.group(1),
                claim=f"{chunk.path} defines function {m.group(1)}",
                confidence="high",
                evidence=[ev],
            )
        )

    # class / struct defs
    for m in _CLASS_RE.finditer(chunk.content):
        fid = uuid4().hex
        ev = evidence.model_copy(update={"id": uuid4().hex, "fact_id": fid})
        facts.append(
            Fact(
                id=fid,
                project_id=project_id,
                subject=chunk.path,
                predicate="defines_class",
                object=m.group(1),
                claim=f"{chunk.path} defines class {m.group(1)}",
                confidence="high",
                evidence=[ev],
            )
        )

    return facts


def extract_facts(chunks: list[Chunk], project_id: str) -> list[Fact]:
    """Extract facts from all chunks."""
    facts: list[Fact] = []
    for chunk in chunks:
        facts.extend(extract_facts_from_chunk(chunk, project_id))
    return facts
