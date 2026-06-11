"""Models for LLM adapter interface."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """A request sent to an LLM provider."""

    prompt: str
    system_prompt: str = ""
    schema_name: Optional[str] = Field(default=None, description="JSON Schema name to enforce.")
    temperature: float = 0.0
    max_tokens: int = 4096


class LLMResponse(BaseModel):
    """A validated response from an LLM provider."""

    raw: str = ""
    parsed: Optional[dict[str, Any]] = None
    model: str = ""
    usage_tokens: int = 0
    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)
