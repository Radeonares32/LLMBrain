"""LLM-backed entity extractor."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from llmbrain.llm.base import BaseLLMProvider
from llmbrain.llm.cache import LLMExtractionCache
from llmbrain.llm.providers import LLMProviderError
from llmbrain.llm.schemas import SCHEMA_ENTITY_LIST
from llmbrain.models.document import Document
from llmbrain.models.entity import Entity
from llmbrain.models.llm import LLMRequest


def _prompt(doc: Document) -> str:
    return (
        "Extract named engineering entities from this source document. Include "
        "services, modules, API endpoints, databases, queues, config values, "
        "environment variables, dependencies, and packages that are explicitly "
        "present. Return at most 12 high-value entities. Do not invent entities. "
        "Return an empty entities array when there are no notable entities beyond "
        "the file itself.\n\n"
        f"Path: {doc.relative_path}\n"
        f"Language: {doc.language}\n\n"
        "Source:\n"
        f"{doc.content}"
    )


async def extract_entities_from_document(
    doc: Document,
    project_id: str,
    provider: BaseLLMProvider,
    cache: LLMExtractionCache | None = None,
) -> list[Entity]:
    """Extract entities from one document using the configured LLM provider."""

    entities: list[Entity] = [
        Entity(
            id=uuid4().hex,
            project_id=project_id,
            name=doc.relative_path,
            type="file",
            path=doc.relative_path,
            confidence="high",
        )
    ]

    provider_name = getattr(provider, "name", provider.__class__.__name__)
    model = getattr(provider, "model", "")
    cache_key = LLMExtractionCache.key(
        provider_name=provider_name,
        model=model,
        task="entities",
        content_hash=doc.content_hash,
    )
    parsed = cache.get(cache_key) if cache else None

    if parsed is None:
        request = LLMRequest(
            prompt=_prompt(doc),
            schema_name="entities",
            max_tokens=4096,
        )
        response = await provider.generate_structured(request, SCHEMA_ENTITY_LIST)
        if not response.is_valid:
            retry = request.model_copy(
                update={
                    "prompt": (
                        request.prompt
                        + "\n\nYour previous response was invalid JSON. Return only a valid JSON "
                        "object with an entities array and no Markdown."
                    )
                }
            )
            response = await provider.generate_structured(retry, SCHEMA_ENTITY_LIST)
        if not response.is_valid or response.parsed is None:
            raise LLMProviderError(
                f"Entity extraction failed for {doc.relative_path}: {response.errors}"
            )
        parsed = response.parsed
        if cache:
            cache.set(cache_key, parsed)

    seen = {(entities[0].name.lower(), entities[0].type, entities[0].path)}
    for item in parsed.get("entities", []):
        path = item.get("path") or doc.relative_path
        key = (item["name"].lower(), item["type"], path)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            Entity(
                id=uuid4().hex,
                project_id=project_id,
                name=item["name"],
                type=item["type"],
                path=path,
                confidence=item["confidence"],
                metadata=item.get("metadata") or {},
            )
        )
    return entities


async def extract_entities(
    docs: list[Document],
    project_id: str,
    provider: BaseLLMProvider,
    *,
    cache: LLMExtractionCache | None = None,
    concurrency: int = 4,
) -> list[Entity]:
    """Extract entities from all documents."""

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(doc: Document) -> list[Entity]:
        async with semaphore:
            return await extract_entities_from_document(doc, project_id, provider, cache)

    results = await asyncio.gather(*(run(doc) for doc in docs))
    return [entity for doc_entities in results for entity in doc_entities]
