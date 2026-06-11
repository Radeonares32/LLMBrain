"""Approximate token budget helpers for context format comparisons."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate token count without a provider-specific tokenizer."""

    if not text:
        return 0
    return max(1, len(text) // 4)


def compare_context_size(json_text: str, brainframe_text: str) -> dict:
    """Compare JSON-style context against compact BrainFrame context."""

    json_chars = len(json_text)
    brainframe_chars = len(brainframe_text)
    json_tokens = estimate_tokens(json_text)
    brainframe_tokens = estimate_tokens(brainframe_text)
    saved_chars = json_chars - brainframe_chars
    saved_tokens = json_tokens - brainframe_tokens
    saved_percent = round((saved_tokens / json_tokens) * 100, 2) if json_tokens else 0.0
    return {
        "json_chars": json_chars,
        "brainframe_chars": brainframe_chars,
        "json_estimated_tokens": json_tokens,
        "brainframe_estimated_tokens": brainframe_tokens,
        "saved_chars": saved_chars,
        "saved_tokens": saved_tokens,
        "saved_percent": saved_percent,
    }
