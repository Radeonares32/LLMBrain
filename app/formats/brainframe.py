"""BrainFrame compact context renderer re-export."""

from llmbrain.formats.brainframe import (
    build_brainframe_context,
    build_compact_context,
    escape_cell,
    format_evidence,
)

__all__ = [
    "build_brainframe_context",
    "build_compact_context",
    "escape_cell",
    "format_evidence",
]
