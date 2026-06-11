import time
from pathlib import Path

import pytest

from llmbrain.llm.base import BaseLLMProvider
from llmbrain.llm.cache import LLMExtractionCache
from llmbrain.models.chunk import Chunk
from llmbrain.models.llm import LLMRequest, LLMResponse
from llmbrain.services.fact_extractor import extract_facts


class SlowProvider(BaseLLMProvider):
    name = "test"
    model = "slow"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(raw="{}")

    async def generate_structured(self, request: LLMRequest, schema: dict) -> LLMResponse:
        import asyncio

        self.calls += 1
        await asyncio.sleep(0.05)
        return LLMResponse(
            parsed={
                "facts": [
                    {
                        "subject": "file.py",
                        "predicate": "defines_function",
                        "object": "f",
                        "claim": "file.py defines f",
                        "confidence": "high",
                    }
                ]
            },
            is_valid=True,
        )


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id=str(index),
            project_id="p",
            document_id=f"d{index}",
            path=f"file{index}.py",
            start_line=1,
            end_line=1,
            content=f"def f{index}(): pass",
            content_hash=f"h{index}",
        )
        for index in range(8)
    ]


@pytest.mark.asyncio
async def test_fact_extraction_uses_parallelism_and_cache(tmp_path: Path):
    cache = LLMExtractionCache(tmp_path / "cache.jsonl")
    provider = SlowProvider()

    start = time.perf_counter()
    facts = await extract_facts(_chunks(), "p", provider, cache=cache, concurrency=4)
    duration = time.perf_counter() - start
    cache.save()

    assert len(facts) == 8
    assert provider.calls == 8
    assert duration < 0.25

    warm_cache = LLMExtractionCache(tmp_path / "cache.jsonl")
    warm_provider = SlowProvider()
    cached_facts = await extract_facts(
        _chunks(),
        "p",
        warm_provider,
        cache=warm_cache,
        concurrency=4,
    )

    assert len(cached_facts) == 8
    assert warm_provider.calls == 0
