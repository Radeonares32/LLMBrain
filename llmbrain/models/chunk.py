"""Chunk model — a contiguous slice of a document."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A line-range slice of a source document."""

    id: str
    project_id: str
    document_id: str
    path: str = Field(..., description="Relative file path.")
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    content: str
    content_hash: str


class ChunkSummary(BaseModel):
    """Lightweight chunk projection."""

    id: str
    document_id: str
    path: str
    start_line: int
    end_line: int
