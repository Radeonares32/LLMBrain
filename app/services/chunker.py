"""Document chunker — splits a document into overlapping line-range chunks."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from app.core.config import settings
from app.models.chunk import Chunk
from app.models.document import Document


def _content_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8", errors="replace")).hexdigest()


def chunk_document(
    doc: Document,
    max_lines: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split *doc* into line-range chunks.

    Each chunk preserves `start_line` / `end_line` (1-indexed) so that every
    extracted fact can point back to exact source evidence.
    """

    if doc.content is None:
        return []

    max_lines = max_lines or settings.chunk_max_lines
    overlap = overlap or settings.chunk_overlap_lines

    lines = doc.content.splitlines(keepends=True)
    total = len(lines)

    if total == 0:
        return []

    chunks: list[Chunk] = []
    start = 0

    while start < total:
        end = min(start + max_lines, total)
        chunk_lines = lines[start:end]
        content = "".join(chunk_lines)

        chunks.append(
            Chunk(
                id=uuid4().hex,
                project_id=doc.project_id,
                document_id=doc.id,
                path=doc.relative_path,
                start_line=start + 1,  # 1-indexed
                end_line=end,           # inclusive
                content=content,
                content_hash=_content_hash(content),
            )
        )

        if end >= total:
            break
        start = end - overlap

    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Chunk a list of documents."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks
