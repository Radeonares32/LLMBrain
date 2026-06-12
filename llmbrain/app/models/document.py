"""Document model — represents a single parsed source file."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(BaseModel):
    """A source file discovered and parsed by the scanner."""

    id: str = Field(..., description="Unique document identifier (uuid4 hex).")
    project_id: str
    path: str = Field(..., description="Absolute path on disk.")
    relative_path: str = Field(..., description="Path relative to project root.")
    content_hash: str = Field(..., description="SHA-256 hex digest of raw content.")
    file_type: str = Field(
        ...,
        description="Extension or special filename, e.g. '.py', 'Dockerfile'.",
    )
    language: str = Field(default="unknown", description="Detected programming language.")
    line_count: int = Field(default=0, ge=0)
    size_bytes: int = Field(default=0, ge=0)
    content: str | None = Field(
        default=None,
        description="Raw content (may be omitted for large files).",
    )
    created_at: datetime = Field(default_factory=_utcnow)


class DocumentSummary(BaseModel):
    """Lightweight projection returned by list endpoints."""

    id: str
    relative_path: str
    file_type: str
    language: str
    line_count: int
    size_bytes: int
