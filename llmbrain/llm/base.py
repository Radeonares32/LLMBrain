"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable

from llmbrain.models.llm import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    """Interface that all LLM adapters must implement."""

    @abstractmethod
    async def generate(
        self, request: LLMRequest, stream_callback: Callable[[str], None] | None = None
    ) -> LLMResponse:
        """Send a prompt and return a validated response."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        request: LLMRequest,
        schema: dict,
        stream_callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a prompt with a JSON Schema constraint."""
        ...

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Send a prompt and stream the text response."""
        ...
