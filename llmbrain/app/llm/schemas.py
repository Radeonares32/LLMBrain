"""JSON Schema definitions for structured LLM outputs."""

from __future__ import annotations

SCHEMA_WIKI_PAGE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WikiPage",
    "type": "object",
    "required": ["title", "slug", "type", "summary", "key_facts"],
    "properties": {
        "title": {"type": "string"},
        "slug": {"type": "string"},
        "type": {"type": "string"},
        "summary": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "dependencies": {"type": "array", "items": {"type": "string"}},
    },
}

SCHEMA_FACT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Fact",
    "type": "object",
    "required": ["subject", "predicate", "object", "claim", "confidence"],
    "properties": {
        "subject": {"type": "string"},
        "predicate": {"type": "string"},
        "object": {"type": "string"},
        "claim": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

SCHEMA_ENTITY = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Entity",
    "type": "object",
    "required": ["name", "type", "confidence"],
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string"},
        "path": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

SCHEMA_RELATION = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Relation",
    "type": "object",
    "required": ["source", "relation", "target", "confidence"],
    "properties": {
        "source": {"type": "string"},
        "relation": {"type": "string"},
        "target": {"type": "string"},
        "evidence": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

SCHEMA_REGISTRY: dict[str, dict] = {
    "wiki_page": SCHEMA_WIKI_PAGE,
    "fact": SCHEMA_FACT,
    "entity": SCHEMA_ENTITY,
    "relation": SCHEMA_RELATION,
}
