"""Secret redaction utilities for LLM inputs and generated artifacts."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

REDACTION_TOKEN = "<REDACTED:SECRET>"

_SECRET_KEYWORDS = (
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "token",
    "password",
    "passwd",
    "pwd",
    "private_key",
    "client_secret",
    "auth",
    "credential",
)

_KEY_VALUE_RE = re.compile(
    r"(?P<prefix>\b[A-Za-z_][A-Za-z0-9_-]*\b[ \t]*(?::|=(?!=))[ \t]*)"
    r"(?P<quote>['\"]?)"
    r"(?P<value>(?!<REDACTED:SECRET>)[^'\"\s,#}]+)"
    r"(?P=quote)",
    re.IGNORECASE,
)

_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>\b[A-Za-z_][A-Za-z0-9_]*(?:secret|token|password|passwd|pwd|"
    r"api_key|access_key|private_key|client_secret|auth|credential)[A-Za-z0-9_]*"
    r"\b\s*=\s*)"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)

_PASSWORD_COMPARISON_RE = re.compile(
    r"(?P<prefix>\b(?:password|passwd|pwd|secret|token)\b\s*(?:==|!=|is)\s*)"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)

_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_deepseek_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
)


@dataclass
class RedactionReport:
    """Counts and kinds of redactions applied to a text payload."""

    counts: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def add(self, kind: str, count: int = 1) -> None:
        if count:
            self.counts[kind] += count

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass
class RedactionResult:
    """Redacted text plus metadata for observability."""

    text: str
    report: RedactionReport

    @property
    def changed(self) -> bool:
        return self.report.total > 0


def _redact_key_value(match: re.Match[str], report: RedactionReport) -> str:
    key = match.group("prefix")
    value = match.group("value")
    key_name = re.split(r"[:=]", key, maxsplit=1)[0].strip().lower().replace("-", "_")
    if not any(keyword in key_name for keyword in _SECRET_KEYWORDS):
        return match.group(0)
    if value in {"(", "[", "{"} or value.endswith("("):
        return match.group(0)
    if not value or value == REDACTION_TOKEN:
        return match.group(0)
    report.add("key_value")
    quote = match.group("quote") or ""
    return f"{key}{quote}{REDACTION_TOKEN}{quote}"


def _redact_assignment(match: re.Match[str], report: RedactionReport) -> str:
    value = match.group("value")
    if value == REDACTION_TOKEN:
        return match.group(0)
    report.add("assignment")
    return f"{match.group('prefix')}{match.group('quote')}{REDACTION_TOKEN}{match.group('quote')}"


def _redact_password_comparison(match: re.Match[str], report: RedactionReport) -> str:
    value = match.group("value")
    if value == REDACTION_TOKEN:
        return match.group(0)
    report.add("password_comparison")
    return f"{match.group('prefix')}{match.group('quote')}{REDACTION_TOKEN}{match.group('quote')}"


def redact_text(text: str) -> RedactionResult:
    """Redact common secret values while preserving code shape and line numbers."""

    report = RedactionReport()

    redacted = _KEY_VALUE_RE.sub(lambda m: _redact_key_value(m, report), text)
    redacted = _ASSIGNMENT_RE.sub(lambda m: _redact_assignment(m, report), redacted)
    redacted = _PASSWORD_COMPARISON_RE.sub(
        lambda m: _redact_password_comparison(m, report),
        redacted,
    )

    for kind, pattern in _VALUE_PATTERNS:
        redacted, count = pattern.subn(REDACTION_TOKEN, redacted)
        report.add(kind, count)

    return RedactionResult(text=redacted, report=report)


def looks_secretish(value: str) -> bool:
    """Return True when a string is likely to contain a credential or secret."""

    lower = value.lower()
    if REDACTION_TOKEN.lower() in lower:
        return False
    if any(keyword in lower for keyword in _SECRET_KEYWORDS):
        return True
    return any(pattern.search(value) for _, pattern in _VALUE_PATTERNS)


def redact_model_strings(value: Any) -> Any:
    """Recursively redact string values in pydantic models or plain structures."""

    if isinstance(value, BaseModel):
        return value.__class__.model_validate(redact_model_strings(value.model_dump()))
    if isinstance(value, dict):
        return {key: redact_model_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_model_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_model_strings(item) for item in value)
    if isinstance(value, str):
        return redact_text(value).text
    return value
