"""Production LLM provider adapters."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv
from jsonschema import ValidationError, validate

from llmbrain.llm.base import BaseLLMProvider
from llmbrain.models.llm import LLMRequest, LLMResponse


class LLMProviderError(RuntimeError):
    """Raised when a production LLM provider cannot be configured or called."""


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Provider returned JSON that is not an object.")
    return parsed


def _system_prompt(schema: dict[str, Any]) -> str:
    return (
        "You extract source-grounded engineering knowledge. Return only valid JSON "
        "matching this JSON Schema. Do not include Markdown or prose.\n\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


class OpenAICompatibleProvider(BaseLLMProvider):
    """Adapter for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        model: str,
        base_url: str,
        use_json_schema: bool,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.use_json_schema = use_json_schema

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        raw, usage = await self._post_chat(payload)
        return LLMResponse(raw=raw, model=payload["model"], usage_tokens=usage)

    async def generate_structured(self, request: LLMRequest, schema: dict) -> LLMResponse:
        model = request.model or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt or _system_prompt(schema)},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if self.use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name or "llmbrain_schema",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        raw, usage = await self._post_chat(payload)
        return _structured_response(raw, model, usage, schema)

    async def _post_chat(self, payload: dict[str, Any]) -> tuple[str, int]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"{self.name} request failed: {exc}") from exc

        data = response.json()
        try:
            raw = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"{self.name} returned an unexpected response shape.") from exc
        usage = int((data.get("usage") or {}).get("total_tokens") or 0)
        return raw or "", usage


class AnthropicProvider(BaseLLMProvider):
    """Adapter for Anthropic Messages API."""

    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        self.name = "anthropic"
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raw, usage = await self._post_messages(request, request.system_prompt)
        return LLMResponse(raw=raw, model=request.model or self.model, usage_tokens=usage)

    async def generate_structured(self, request: LLMRequest, schema: dict) -> LLMResponse:
        system_prompt = request.system_prompt or _system_prompt(schema)
        raw, usage = await self._post_messages(request, system_prompt)
        return _structured_response(raw, request.model or self.model, usage, schema)

    async def _post_messages(self, request: LLMRequest, system_prompt: str) -> tuple[str, int]:
        payload = {
            "model": request.model or self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"anthropic request failed: {exc}") from exc

        data = response.json()
        blocks = data.get("content") or []
        raw = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        usage_data = data.get("usage") or {}
        usage = int(usage_data.get("input_tokens") or 0) + int(usage_data.get("output_tokens") or 0)
        return raw, usage


class OllamaProvider(BaseLLMProvider):
    """Adapter for local Ollama chat models."""

    def __init__(self, *, model: str, base_url: str) -> None:
        self.name = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = self._payload(request, None)
        raw, usage = await self._post_chat(payload)
        return LLMResponse(raw=raw, model=payload["model"], usage_tokens=usage)

    async def generate_structured(self, request: LLMRequest, schema: dict) -> LLMResponse:
        payload = self._payload(request, schema)
        raw, usage = await self._post_chat(payload)
        return _structured_response(raw, payload["model"], usage, schema)

    def _payload(self, request: LLMRequest, schema: dict | None) -> dict[str, Any]:
        system_prompt = request.system_prompt
        if schema is not None:
            system_prompt = system_prompt or _system_prompt(schema)
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if schema is not None:
            payload["format"] = schema
        return payload

    async def _post_chat(self, payload: dict[str, Any]) -> tuple[str, int]:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"ollama request failed: {exc}") from exc

        data = response.json()
        raw = ((data.get("message") or {}).get("content")) or ""
        usage = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
        return raw, usage


def _structured_response(raw: str, model: str, usage: int, schema: dict[str, Any]) -> LLMResponse:
    errors: list[str] = []
    parsed: dict[str, Any] | None = None
    try:
        parsed = _extract_json(raw)
        validate(instance=parsed, schema=schema)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        errors.append(str(exc))
    return LLMResponse(
        raw=raw,
        parsed=parsed,
        model=model,
        usage_tokens=usage,
        is_valid=not errors,
        errors=errors,
    )


def create_provider(provider_name: str | None = None) -> BaseLLMProvider:
    """Create a configured production LLM provider."""

    load_dotenv()
    configured_provider = provider_name or _env("LLMBRAIN_DEFAULT_PROVIDER", default="openai")
    name = (configured_provider or "openai").lower()

    if name == "openai":
        api_key = _env("OPENAI_API_KEY", "LLMBRAIN_OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError("OPENAI_API_KEY is required for provider 'openai'.")
        return OpenAICompatibleProvider(
            name="openai",
            api_key=api_key,
            model=_env(
                "OPENAI_MODEL",
                "LLMBRAIN_OPENAI_MODEL",
                default="gpt-4o-mini",
            )
            or "gpt-4o-mini",
            base_url=_env(
                "OPENAI_BASE_URL",
                "LLMBRAIN_OPENAI_BASE_URL",
                default="https://api.openai.com/v1",
            )
            or "https://api.openai.com/v1",
            use_json_schema=True,
        )

    if name == "deepseek":
        api_key = _env("DEEPSEEK_API_KEY", "LLMBRAIN_DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMProviderError("DEEPSEEK_API_KEY is required for provider 'deepseek'.")
        return OpenAICompatibleProvider(
            name="deepseek",
            api_key=api_key,
            model=_env(
                "DEEPSEEK_MODEL",
                "LLMBRAIN_DEEPSEEK_MODEL",
                default="deepseek-chat",
            )
            or "deepseek-chat",
            base_url=_env(
                "DEEPSEEK_BASE_URL",
                "LLMBRAIN_DEEPSEEK_BASE_URL",
                default="https://api.deepseek.com/v1",
            )
            or "https://api.deepseek.com/v1",
            use_json_schema=False,
        )

    if name == "anthropic":
        api_key = _env("ANTHROPIC_API_KEY", "LLMBRAIN_ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY is required for provider 'anthropic'.")
        return AnthropicProvider(
            api_key=api_key,
            model=_env(
                "ANTHROPIC_MODEL",
                "LLMBRAIN_ANTHROPIC_MODEL",
                default="claude-3-5-sonnet-latest",
            )
            or "claude-3-5-sonnet-latest",
            base_url=_env(
                "ANTHROPIC_BASE_URL",
                "LLMBRAIN_ANTHROPIC_BASE_URL",
                default="https://api.anthropic.com/v1",
            )
            or "https://api.anthropic.com/v1",
        )

    if name == "ollama":
        model = _env("OLLAMA_MODEL", "LLMBRAIN_OLLAMA_MODEL")
        if not model:
            raise LLMProviderError("OLLAMA_MODEL is required for provider 'ollama'.")
        return OllamaProvider(
            model=model,
            base_url=_env(
                "OLLAMA_BASE_URL", "LLMBRAIN_OLLAMA_BASE_URL", default="http://localhost:11434"
            )
            or "http://localhost:11434",
        )

    raise LLMProviderError(
        f"Unsupported provider '{provider_name}'. "
        "Supported providers: openai, deepseek, anthropic, ollama."
    )
