"""Relation model — a directed edge between two entities."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Relation(BaseModel):
    """A directed relationship between two entities."""

    id: str
    project_id: str
    source_entity_id: str
    relation: str = Field(
        ...,
        description=(
            "Relation type: depends_on | calls | exposes | reads_from | "
            "writes_to | configured_by | documented_by | related_to"
        ),
    )
    target_entity_id: str
    evidence: str = Field(default="", description="Source reference, e.g. path:L10-L20.")
    confidence: str = Field(default="medium")


class RelationSummary(BaseModel):
    """Lightweight relation projection."""

    id: str
    source_entity_id: str
    relation: str
    target_entity_id: str
    confidence: str
