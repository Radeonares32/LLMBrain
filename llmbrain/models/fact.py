"""Fact model — a subject-predicate-object claim extracted from source."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FactEvidence(BaseModel):
    """Points back to the source lines that support a fact."""

    id: str
    fact_id: str
    document_id: str
    path: str
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)


class Fact(BaseModel):
    """A single extracted knowledge claim."""

    id: str
    project_id: str
    subject: str
    predicate: str
    object: str
    claim: str = Field(default="", description="Human-readable sentence.")
    confidence: str = Field(default="medium", description="low | medium | high")
    evidence: list[FactEvidence] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class FactSummary(BaseModel):
    """Lightweight fact projection."""

    id: str
    subject: str
    predicate: str
    object: str
    confidence: str
