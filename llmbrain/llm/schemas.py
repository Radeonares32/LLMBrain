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

SCHEMA_FACT_LIST = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FactList",
    "type": "object",
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["subject", "predicate", "object", "claim", "confidence"],
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "claim": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

SCHEMA_ENTITY_LIST = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EntityList",
    "type": "object",
    "required": ["entities"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type", "path", "confidence"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "path": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "metadata": {"type": "object"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

SCHEMA_RELATION_LIST = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RelationList",
    "type": "object",
    "required": ["relations"],
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "relation", "target", "evidence", "confidence"],
                "properties": {
                    "source": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": [
                            "depends_on",
                            "calls",
                            "exposes",
                            "reads_from",
                            "writes_to",
                            "configured_by",
                            "documented_by",
                            "related_to",
                        ],
                    },
                    "target": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

SCHEMA_REGISTRY: dict[str, dict] = {
    "wiki_page": SCHEMA_WIKI_PAGE,
    "facts": SCHEMA_FACT_LIST,
    "entities": SCHEMA_ENTITY_LIST,
    "relations": SCHEMA_RELATION_LIST,
}
