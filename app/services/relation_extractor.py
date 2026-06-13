"""Relation extractor — discovers edges between entities.

MVP builds relations from import/dependency facts and entity co-occurrence.
Future versions will use LLM extraction.
"""

from __future__ import annotations

from uuid import uuid4

from app.models.entity import Entity
from app.models.fact import Fact
from app.models.relation import Relation


def extract_relations(
    entities: list[Entity],
    facts: list[Fact],
    project_id: str,
) -> list[Relation]:
    """Build relations by cross-referencing entities and facts."""

    relations: list[Relation] = []

    # build entity index by name (lowercase)
    entity_by_name: dict[str, Entity] = {}
    for ent in entities:
        entity_by_name[ent.name.lower()] = ent

    # file entity index by path
    file_entities: dict[str, Entity] = {ent.path: ent for ent in entities if ent.type == "file"}

    # from import facts: file --depends_on--> imported_package
    for fact in facts:
        if fact.predicate != "imports":
            continue

        source_ent = file_entities.get(fact.subject)
        target_ent = entity_by_name.get(fact.object.lower())

        if source_ent and target_ent and source_ent.id != target_ent.id:
            ev_str = ""
            if fact.evidence:
                e = fact.evidence[0]
                ev_str = f"{e.path}:L{e.start_line}-L{e.end_line}"

            relations.append(
                Relation(
                    id=uuid4().hex,
                    project_id=project_id,
                    source_entity_id=source_ent.id,
                    relation="depends_on",
                    target_entity_id=target_ent.id,
                    evidence=ev_str,
                    confidence=fact.confidence,
                )
            )

    return relations
