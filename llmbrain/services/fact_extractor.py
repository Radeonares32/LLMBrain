"""LLM-backed fact extractor."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from llmbrain.llm.base import BaseLLMProvider
from llmbrain.llm.cache import LLMExtractionCache
from llmbrain.llm.providers import LLMProviderError
from llmbrain.llm.schemas import SCHEMA_FACT_LIST
from llmbrain.models.chunk import Chunk
from llmbrain.models.fact import Fact, FactEvidence
from llmbrain.models.llm import LLMRequest


def _prompt(chunk: Chunk) -> str:
    return (
        "Extract durable engineering facts from this source chunk. "
        "Facts must be directly supported by the source. Use concise predicates "
        "such as imports, defines_function, defines_class, exposes_endpoint, "
        "reads_config, depends_on, calls, documents, or configures. "
        "Return at most 12 high-value facts. Prefer facts that explain behavior, "
        "architecture, public APIs, configuration, dependencies, and persistence. "
        "Return an empty facts array if nothing source-grounded is present.\n\n"
        f"Path: {chunk.path}\n"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n\n"
        "Source:\n"
        f"{chunk.content}"
    )


async def extract_facts_from_chunk(
    chunk: Chunk,
    project_id: str,
    provider: BaseLLMProvider,
    cache: LLMExtractionCache | None = None,
) -> list[Fact]:
    """Extract facts from a single chunk using the configured LLM provider."""

    provider_name = getattr(provider, "name", provider.__class__.__name__)
    model = getattr(provider, "model", "")
    cache_key = LLMExtractionCache.key(
        provider_name=provider_name,
        model=model,
        task="facts",
        content_hash=chunk.content_hash,
    )
    parsed = cache.get(cache_key) if cache else None

    if parsed is None:
        request = LLMRequest(
            prompt=_prompt(chunk),
            schema_name="facts",
            max_tokens=4096,
        )
        response = await provider.generate_structured(request, SCHEMA_FACT_LIST)
        if not response.is_valid:
            retry = request.model_copy(
                update={
                    "prompt": (
                        request.prompt
                        + "\n\nYour previous response was invalid JSON. Return only a valid JSON "
                        "object with a facts array and no Markdown."
                    )
                }
            )
            response = await provider.generate_structured(retry, SCHEMA_FACT_LIST)
        if not response.is_valid or response.parsed is None:
            raise LLMProviderError(f"Fact extraction failed for {chunk.path}: {response.errors}")
        parsed = response.parsed
        if cache:
            cache.set(cache_key, parsed)

    facts: list[Fact] = []
    for item in parsed.get("facts", []):
        fid = uuid4().hex
        evidence = FactEvidence(
            id=uuid4().hex,
            fact_id=fid,
            document_id=chunk.document_id,
            path=chunk.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
        facts.append(
            Fact(
                id=fid,
                project_id=project_id,
                subject=item["subject"],
                predicate=item["predicate"],
                object=item["object"],
                claim=item["claim"],
                confidence=item["confidence"],
                evidence=[evidence],
            )
        )
    return facts


async def extract_facts(
    chunks: list[Chunk],
    project_id: str,
    provider: BaseLLMProvider,
    *,
    cache: LLMExtractionCache | None = None,
    concurrency: int = 4,
) -> list[Fact]:
    """Extract facts from all chunks."""

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(chunk: Chunk) -> list[Fact]:
        async with semaphore:
            return await extract_facts_from_chunk(chunk, project_id, provider, cache)

    results = await asyncio.gather(*(run(chunk) for chunk in chunks))
    return [fact for chunk_facts in results for fact in chunk_facts]
