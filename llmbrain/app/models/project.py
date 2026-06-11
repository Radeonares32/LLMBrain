"""Project model — top-level grouping of all knowledge artefacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProjectCreate(BaseModel):
    """Payload for scanning / building a project."""

    path: str = Field(..., description="Absolute path to the project root.")
    name: Optional[str] = Field(default=None, description="Optional human-friendly name.")


class Project(BaseModel):
    """Persisted project record."""

    id: str
    name: str
    root_path: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ProjectStats(BaseModel):
    """Aggregated counts for a project."""

    documents: int = 0
    chunks: int = 0
    facts: int = 0
    entities: int = 0
    relations: int = 0
    wiki_pages: int = 0


class ScanResult(BaseModel):
    """Returned by POST /projects/scan."""

    project: Project
    stats: ProjectStats
    documents: list = Field(default_factory=list)


class BuildResult(BaseModel):
    """Returned by POST /projects/build."""

    project: Project
    stats: ProjectStats
    output_path: str = Field(default="", description="Path to .llmbrain output.")
