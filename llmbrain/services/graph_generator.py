"""Graph generator — builds graph.json and GraphML from entities + relations."""

from __future__ import annotations

from llmbrain.models.entity import Entity
from llmbrain.models.graph import GraphEdge, GraphNode, KnowledgeGraph
from llmbrain.models.relation import Relation


def build_knowledge_graph(
    entities: list[Entity],
    relations: list[Relation],
    project_id: str,
) -> KnowledgeGraph:
    """Build the in-memory knowledge graph."""

    nodes = [
        GraphNode(
            id=e.id,
            label=e.name,
            type=e.type,
            metadata={"path": e.path, "confidence": e.confidence},
        )
        for e in entities
    ]

    entity_index = {e.id: e for e in entities}

    edges = []
    for r in relations:
        if r.source_entity_id in entity_index and r.target_entity_id in entity_index:
            edges.append(
                GraphEdge(
                    source=r.source_entity_id,
                    target=r.target_entity_id,
                    relation=r.relation,
                    confidence=r.confidence,
                    evidence=r.evidence,
                )
            )

    return KnowledgeGraph(project_id=project_id, nodes=nodes, edges=edges)


def graph_to_graphml(graph: KnowledgeGraph) -> str:
    """Render the knowledge graph as GraphML XML."""

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphstruct.org/graphml"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xsi:schemaLocation="http://graphml.graphstruct.org/graphml">',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]

    for node in graph.nodes:
        lines.append(f'    <node id="{node.id}">')
        lines.append(f'      <data key="label">{node.label}</data>')
        lines.append(f'      <data key="type">{node.type}</data>')
        lines.append("    </node>")

    for i, edge in enumerate(graph.edges):
        lines.append(f'    <edge id="e{i}" source="{edge.source}" target="{edge.target}">')
        lines.append(f'      <data key="relation">{edge.relation}</data>')
        lines.append("    </edge>")

    lines.append("  </graph>")
    lines.append("</graphml>")

    return "\n".join(lines)


def graph_to_mermaid(graph: KnowledgeGraph) -> str:
    """Render the knowledge graph as a Mermaid JS diagram."""
    lines = ["graph TD"]
    
    # Optional: Group by type if needed, but for simplicity just nodes
    for node in graph.nodes:
        # Mermaid doesn't like quotes or special chars in IDs sometimes, but standard alphanumerics are fine
        safe_id = "".join(c if c.isalnum() else "_" for c in node.id)
        safe_label = str(node.label).replace('"', "'")
        lines.append(f'    {safe_id}["{safe_label} ({node.type})"]')

    for i, edge in enumerate(graph.edges):
        safe_source = "".join(c if c.isalnum() else "_" for c in edge.source)
        safe_target = "".join(c if c.isalnum() else "_" for c in edge.target)
        safe_relation = str(edge.relation).replace('"', "'")
        lines.append(f'    {safe_source} -->|{safe_relation}| {safe_target}')

    return "\n".join(lines)
