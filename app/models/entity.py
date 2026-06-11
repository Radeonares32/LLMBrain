"""Entity model — a named engineering artefact (service, module, endpoint …)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """An extracted engineering entity."""

    id: str
    project_id: str
    name: str
    type: str = Field(
        ...,
        description=(
            "Entity category: service | module | api_endpoint | database | "
            "queue | config | env_var | dependency | file | package"
        ),
    )
    path: str = Field(default="", description="Primary source path.")
    confidence: str = Field(default="medium")
    metadata: dict = Field(default_factory=dict)


class EntitySummary(BaseModel):
    """Lightweight entity projection."""

    id: str
    name: str
    type: str
    confidence: str
