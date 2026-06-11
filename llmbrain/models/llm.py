"""Models for LLM adapter interface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """A request sent to an LLM provider."""

    prompt: str
    system_prompt: str = ""
    schema_name: str | None = Field(default=None, description="JSON Schema name to enforce.")
    model: str | None = Field(default=None, description="Optional provider model override.")
    temperature: float = 0.0
    max_tokens: int = 4096


class LLMResponse(BaseModel):
    """A validated response from an LLM provider."""

    raw: str = ""
    parsed: dict[str, Any] | None = None
    model: str = ""
    usage_tokens: int = 0
    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)
