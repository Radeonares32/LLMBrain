"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmbrain.models.llm import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    """Interface that all LLM adapters must implement."""

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Send a prompt and return a validated response."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        request: LLMRequest,
        schema: dict,
    ) -> LLMResponse:
        """Send a prompt with a JSON Schema constraint."""
        ...
