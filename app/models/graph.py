"""Graph output models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    id: str
    label: str
    type: str
    metadata: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A directed edge in the knowledge graph."""

    source: str
    target: str
    relation: str
    confidence: str = Field(default="medium")
    evidence: str = Field(default="")


class KnowledgeGraph(BaseModel):
    """Full graph payload."""

    project_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
