"""LLM-backed relation extractor."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from llmbrain.formats.brainframe import build_brainframe_context
from llmbrain.llm.base import BaseLLMProvider
from llmbrain.llm.cache import LLMExtractionCache
from llmbrain.llm.providers import LLMProviderError
from llmbrain.llm.schemas import SCHEMA_RELATION_LIST
from llmbrain.models.entity import Entity
from llmbrain.models.fact import Fact
from llmbrain.models.llm import LLMRequest
from llmbrain.models.relation import Relation


def _prompt(entities: list[Entity], facts: list[Fact], project_id: str) -> str:
    selected_entities = entities[:120]
    selected_facts = facts[:180]
    brainframe = build_brainframe_context(
        "relation_extraction",
        project_id,
        selected_entities,
        [],
        selected_facts,
        max_chars=32_000,
    )
    return (
        "Infer source-grounded relationships between the listed entities using only "
        "the supplied facts. The relation source and target must exactly match an "
        "entity name from the entity list. Return an empty relations array if no "
        "relationship is supported. Return at most 40 high-value relations. Prefer "
        "architecture, dependency, API, persistence, and configuration relationships.\n\n"
        "Here is the compact BrainFrame context:\n"
        "<brainframe>\n"
        f"{brainframe}"
        "</brainframe>\n\n"
        "Output:\n"
        "Return JSON matching the relations schema only."
    )


def _index_entities(entities: list[Entity]) -> dict[str, Entity]:
    index: dict[str, Entity] = {}
    for entity in entities:
        index.setdefault(entity.name.lower(), entity)
        index.setdefault(entity.path.lower(), entity)
    return index


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


async def extract_relations(
    entities: list[Entity],
    facts: list[Fact],
    project_id: str,
    provider: BaseLLMProvider,
    *,
    cache: LLMExtractionCache | None = None,
) -> list[Relation]:
    """Build relations from entities and facts using the configured LLM provider."""

    prompt = _prompt(entities, facts, project_id)
    provider_name = getattr(provider, "name", provider.__class__.__name__)
    model = getattr(provider, "model", "")
    cache_key = LLMExtractionCache.key(
        provider_name=provider_name,
        model=model,
        task="relations",
        content_hash=_hash_text(prompt),
    )
    parsed = cache.get(cache_key) if cache else None

    if parsed is None:
        request = LLMRequest(
            prompt=prompt,
            schema_name="relations",
            max_tokens=4096,
        )
        response = await provider.generate_structured(request, SCHEMA_RELATION_LIST)
        if not response.is_valid:
            retry = request.model_copy(
                update={
                    "prompt": (
                        request.prompt
                        + "\n\nYour previous response was invalid JSON. Return only a valid JSON "
                        "object with a relations array and no Markdown."
                    )
                }
            )
            response = await provider.generate_structured(retry, SCHEMA_RELATION_LIST)
        if not response.is_valid or response.parsed is None:
            raise LLMProviderError(f"Relation extraction failed: {response.errors}")
        parsed = response.parsed
        if cache:
            cache.set(cache_key, parsed)

    entity_index = _index_entities(entities)
    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in parsed.get("relations", []):
        source = entity_index.get(item["source"].lower())
        target = entity_index.get(item["target"].lower())
        if source is None or target is None or source.id == target.id:
            continue
        key = (source.id, item["relation"], target.id)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            Relation(
                id=uuid4().hex,
                project_id=project_id,
                source_entity_id=source.id,
                relation=item["relation"],
                target_entity_id=target.id,
                evidence=item.get("evidence", ""),
                confidence=item["confidence"],
            )
        )
    return relations
