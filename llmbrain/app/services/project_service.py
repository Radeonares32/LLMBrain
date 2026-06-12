"""Project service — orchestrates the full build pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from llmbrain.core.config import settings
from llmbrain.models.project import BuildResult, Project, ProjectCreate, ProjectStats, ScanResult
from llmbrain.services.chunker import chunk_documents
from llmbrain.services.context_builder import build_compact_context
from llmbrain.services.entity_extractor import extract_entities
from llmbrain.services.fact_extractor import extract_facts
from llmbrain.services.graph_generator import build_knowledge_graph, graph_to_graphml
from llmbrain.services.relation_extractor import extract_relations
from llmbrain.services.scanner import scan_project
from llmbrain.services.wiki_generator import generate_wiki_pages
from llmbrain.storage.filesystem import (
    ensure_output_dirs,
    output_root,
    write_manifest,
    write_text_file,
)
from llmbrain.storage.jsonl import write_jsonl
from llmbrain.storage.sqlite import SQLiteStore


def _now() -> datetime:
    return datetime.now(UTC)


class ProjectService:
    """High-level service that ties together scanning, extraction, and output."""

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_path(raw: str) -> Path:
        """Resolve and validate a user-supplied path."""
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
        return path

    @staticmethod
    def _project_id_from_path(path: Path) -> str:
        """Deterministic project id so re-scanning the same path reuses state."""
        import hashlib

        return hashlib.sha256(str(path).encode()).hexdigest()[:16]

    def _store(self, project_root: Path) -> SQLiteStore:
        db_path = output_root(project_root) / settings.db_filename
        return SQLiteStore(db_path)

    # ── scan ────────────────────────────────────────────────────────────

    def scan(self, req: ProjectCreate) -> ScanResult:
        root = self._normalize_path(req.path)
        project_id = self._project_id_from_path(root)
        name = req.name or root.name

        project = Project(
            id=project_id,
            name=name,
            root_path=str(root),
            created_at=_now(),
            updated_at=_now(),
        )

        documents = scan_project(root, project_id)

        # persist
        store = self._store(root)
        store.upsert_project(project)
        store.delete_project_data(project_id)
        store.insert_documents(documents)

        stats = ProjectStats(documents=len(documents))

        return ScanResult(
            project=project,
            stats=stats,
            documents=[
                {
                    "id": d.id,
                    "relative_path": d.relative_path,
                    "file_type": d.file_type,
                    "language": d.language,
                    "line_count": d.line_count,
                }
                for d in documents
            ],
        )

    # ── full build ──────────────────────────────────────────────────────

    def build(self, req: ProjectCreate) -> BuildResult:
        root = self._normalize_path(req.path)
        project_id = self._project_id_from_path(root)
        name = req.name or root.name

        project = Project(
            id=project_id,
            name=name,
            root_path=str(root),
            created_at=_now(),
            updated_at=_now(),
        )

        # 1. scan
        documents = scan_project(root, project_id)

        # 2. chunk
        chunks = chunk_documents(documents)

        # 3. extract
        facts = extract_facts(chunks, project_id)
        entities = extract_entities(documents, project_id)
        relations = extract_relations(entities, facts, project_id)

        # 4. generate
        wiki_pages = generate_wiki_pages(entities, facts, relations, project_id)
        graph = build_knowledge_graph(entities, relations, project_id)
        compact_ctx = build_compact_context(name, entities, relations, facts)

        # 5. persist to SQLite
        store = self._store(root)
        store.upsert_project(project)
        store.delete_project_data(project_id)
        store.insert_documents(documents)
        store.insert_chunks(chunks)
        store.insert_facts(facts)
        store.insert_entities(entities)
        store.insert_relations(relations)

        # 6. write output files
        dirs = ensure_output_dirs(root)

        # JSONL exports
        write_jsonl(dirs["root"] / "documents.jsonl", documents)
        write_jsonl(dirs["root"] / "chunks.jsonl", chunks)
        write_jsonl(dirs["root"] / "facts.jsonl", facts)
        write_jsonl(dirs["root"] / "entities.jsonl", entities)
        write_jsonl(dirs["root"] / "relations.jsonl", relations)

        # LLM compact context
        write_text_file(dirs["llm_context"], "brainframe.bf", compact_ctx)

        # Wiki pages
        store.insert_wiki_pages(wiki_pages, str(dirs["wiki"]), str(dirs["wiki_mdx"]))
        for page in wiki_pages:
            write_text_file(dirs["wiki"], f"{page.slug}.md", page.markdown_content)
            write_text_file(dirs["wiki_mdx"], f"{page.slug}.mdx", page.mdx_content)

        # Graph
        graph_json = graph.model_dump_json(indent=2)
        write_text_file(dirs["graph"], "graph.json", graph_json)
        write_text_file(dirs["graph"], "graph.graphml", graph_to_graphml(graph))

        # JSON Schemas
        self._write_schemas(dirs["schemas"])

        # Manifest
        stats = ProjectStats(
            documents=len(documents),
            chunks=len(chunks),
            facts=len(facts),
            entities=len(entities),
            relations=len(relations),
            wiki_pages=len(wiki_pages),
        )
        write_manifest(
            root,
            {
                "project_id": project_id,
                "project_name": name,
                "root_path": str(root),
                "stats": stats.model_dump(),
            },
        )

        return BuildResult(
            project=project,
            stats=stats,
            output_path=str(dirs["root"]),
        )

    # ── query helpers ───────────────────────────────────────────────────

    def get_project(self, project_id: str) -> Project | None:
        """Look up a project by ID across known stores.

        Since we derive the DB path from the project root, and we don't keep
        a global index in the MVP, we search for any .llmbrain/brain.db that
        matches.  A production version would use a central registry.
        """
        # MVP shortcut — look in the store cache tracked via recent builds.
        return self._find_project(project_id)

    def _find_project(self, project_id: str) -> Project | None:
        """Search for a project across known databases."""
        # In MVP, we check a simple in-memory or tmp index.
        # For now, iterate brain.db files under common locations.
        import glob
        import os

        search_roots = [
            os.path.expanduser("~"),
            "/tmp",
        ]

        for sr in search_roots:
            pattern = os.path.join(sr, "**", ".llmbrain", "brain.db")
            for db_path in glob.iglob(pattern, recursive=True):
                try:
                    store = SQLiteStore(db_path)
                    project = store.get_project(project_id)
                    if project:
                        return project
                except Exception:
                    continue
        return None

    def _store_for_project(self, project_id: str) -> SQLiteStore | None:
        project = self._find_project(project_id)
        if project is None:
            return None
        root = Path(project.root_path)
        return self._store(root)

    def get_documents(self, project_id: str) -> list[dict]:
        store = self._store_for_project(project_id)
        return store.get_documents(project_id) if store else []

    def get_chunks(self, project_id: str) -> list[dict]:
        store = self._store_for_project(project_id)
        return store.get_chunks(project_id) if store else []

    def get_facts(self, project_id: str) -> list[dict]:
        store = self._store_for_project(project_id)
        return store.get_facts(project_id) if store else []

    def get_entities(self, project_id: str) -> list[dict]:
        store = self._store_for_project(project_id)
        return store.get_entities(project_id) if store else []

    def get_relations(self, project_id: str) -> list[dict]:
        store = self._store_for_project(project_id)
        return store.get_relations(project_id) if store else []

    def get_wiki_pages(self, project_id: str) -> list[dict]:
        store = self._store_for_project(project_id)
        return store.get_wiki_pages(project_id) if store else []

    def get_graph(self, project_id: str) -> dict:
        """Return the graph.json content."""
        project = self._find_project(project_id)
        if project is None:
            return {"nodes": [], "edges": []}
        graph_path = output_root(project.root_path) / "graph" / "graph.json"
        if graph_path.exists():
            return json.loads(graph_path.read_text(encoding="utf-8"))
        return {"nodes": [], "edges": []}

    def get_compact_context(self, project_id: str) -> dict:
        """Return the brainframe compact context."""
        project = self._find_project(project_id)
        if project is None:
            return {"context": ""}
        bf_path = output_root(project.root_path) / "llm-context" / "brainframe.bf"
        if bf_path.exists():
            return {"context": bf_path.read_text(encoding="utf-8")}
        return {"context": ""}

    # ── schema helpers ──────────────────────────────────────────────────

    @staticmethod
    def _write_schemas(schema_dir: Path) -> None:
        """Write JSON Schema files for structured LLM output."""

        schemas = {
            "wiki_page.schema.json": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "WikiPage",
                "type": "object",
                "required": ["title", "slug", "type", "summary", "key_facts"],
                "properties": {
                    "title": {"type": "string"},
                    "slug": {"type": "string"},
                    "type": {"type": "string", "enum": ["service", "module", "overview", "page"]},
                    "summary": {"type": "string"},
                    "key_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "start_line": {"type": "integer"},
                                "end_line": {"type": "integer"},
                            },
                        },
                    },
                },
            },
            "fact.schema.json": {
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
            },
            "entity.schema.json": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "Entity",
                "type": "object",
                "required": ["name", "type", "confidence"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "service",
                            "module",
                            "api_endpoint",
                            "database",
                            "queue",
                            "config",
                            "env_var",
                            "dependency",
                            "file",
                            "package",
                        ],
                    },
                    "path": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
            "relation.schema.json": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "Relation",
                "type": "object",
                "required": ["source", "relation", "target", "confidence"],
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
            },
        }

        for filename, schema in schemas.items():
            path = schema_dir / filename
            path.write_text(
                json.dumps(schema, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
