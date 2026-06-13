"""Project service — orchestrates the full build pipeline."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from llmbrain.core.config import settings
from llmbrain.llm.cache import LLMExtractionCache
from llmbrain.llm.providers import create_provider
from llmbrain.models.chunk import Chunk
from llmbrain.models.document import Document
from llmbrain.models.entity import Entity
from llmbrain.models.fact import Fact, FactEvidence
from llmbrain.models.project import BuildResult, Project, ProjectCreate, ProjectStats, ScanResult
from llmbrain.models.relation import Relation
from llmbrain.services.chunker import chunk_documents
from llmbrain.services.context_builder import build_brainframe_context
from llmbrain.services.entity_extractor import extract_entities
from llmbrain.services.fact_extractor import extract_facts
from llmbrain.services.graph_generator import build_knowledge_graph, graph_to_graphml
from llmbrain.services.redactor import redact_model_strings
from llmbrain.services.relation_extractor import extract_relations
from llmbrain.services.scanner import scan_project
from llmbrain.services.token_budget import compare_context_size
from llmbrain.services.wiki_generator import generate_wiki_pages
from llmbrain.storage.filesystem import (
    ensure_output_dirs,
    output_root,
    write_manifest,
    write_text_file,
)
from llmbrain.storage.jsonl import read_jsonl, write_jsonl
from llmbrain.storage.sqlite import SQLiteStore


def _now() -> datetime:
    return datetime.now(UTC)


def _clear_directory(directory: Path, suffixes: tuple[str, ...]) -> None:
    """Remove previously generated files so stale artifacts cannot survive a build."""
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file() and path.suffix in suffixes:
            path.unlink()


def _chunk_key_from_values(path: str, start_line: int, end_line: int, content_hash: str) -> tuple:
    return (path, int(start_line), int(end_line), content_hash)


def _chunk_key(chunk: Chunk | dict) -> tuple:
    if isinstance(chunk, Chunk):
        return _chunk_key_from_values(
            chunk.path,
            chunk.start_line,
            chunk.end_line,
            chunk.content_hash,
        )
    return _chunk_key_from_values(
        str(chunk.get("path", "")),
        int(chunk.get("start_line", 0) or 0),
        int(chunk.get("end_line", 0) or 0),
        str(chunk.get("content_hash", "")),
    )


def _document_hash(document: Document | dict) -> str:
    if isinstance(document, Document):
        return document.raw_content_hash or document.content_hash
    return str(document.get("raw_content_hash") or document.get("content_hash") or "")


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
        """Deterministic project id using load_or_create_project_identity."""
        from llmbrain.core.identity import load_or_create_project_identity

        identity = load_or_create_project_identity(path)
        return identity["project_id"]

    def _store(self, project_root: Path) -> SQLiteStore:
        from llmbrain.core.identity import get_project_storage_dir, load_or_create_project_identity

        identity = load_or_create_project_identity(project_root)
        db_path = get_project_storage_dir(identity["project_id"]) / "brain.db"
        return SQLiteStore(db_path)

    def _reuse_incremental_memory(
        self,
        dirs: dict[str, Path],
        documents: list[Document],
        chunks: list[Chunk],
        project_id: str,
    ) -> tuple[list[Fact], list[Chunk], list[Entity], list[Document]]:
        """Reuse unchanged facts/entities from previous JSONL artifacts."""

        previous_documents = read_jsonl(dirs["root"] / "documents.jsonl")
        previous_chunks = read_jsonl(dirs["root"] / "chunks.jsonl")
        previous_facts = read_jsonl(dirs["root"] / "facts.jsonl")
        previous_entities = read_jsonl(dirs["root"] / "entities.jsonl")
        if not previous_documents or not previous_chunks:
            return [], chunks, [], documents

        previous_doc_hashes = {
            str(row.get("relative_path", "")): _document_hash(row) for row in previous_documents
        }
        unchanged_doc_paths = {
            doc.relative_path
            for doc in documents
            if previous_doc_hashes.get(doc.relative_path) == _document_hash(doc)
        }

        previous_chunk_keys = {
            _chunk_key(row)
            for row in previous_chunks
            if str(row.get("path", "")) in unchanged_doc_paths
        }
        current_chunks_by_evidence = {
            (chunk.path, chunk.start_line, chunk.end_line): chunk for chunk in chunks
        }

        reusable_facts: list[Fact] = []
        for row in previous_facts:
            evidence_rows = row.get("evidence") or []
            if not evidence_rows:
                continue
            remapped_evidence: list[FactEvidence] = []
            can_reuse = True
            for evidence in evidence_rows:
                chunk = current_chunks_by_evidence.get(
                    (
                        str(evidence.get("path", "")),
                        int(evidence.get("start_line", 0) or 0),
                        int(evidence.get("end_line", 0) or 0),
                    )
                )
                if chunk is None or _chunk_key(chunk) not in previous_chunk_keys:
                    can_reuse = False
                    break
                remapped_evidence.append(
                    FactEvidence(
                        id=str(evidence.get("id", "")),
                        fact_id=str(row.get("id", "")),
                        document_id=chunk.document_id,
                        path=chunk.path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                    )
                )
            if not can_reuse:
                continue
            fact = Fact.model_validate(
                {
                    **row,
                    "project_id": project_id,
                    "evidence": [ev.model_dump() for ev in remapped_evidence],
                }
            )
            reusable_facts.append(fact)

        chunks_to_extract = [
            chunk for chunk in chunks if _chunk_key(chunk) not in previous_chunk_keys
        ]

        reusable_entities: list[Entity] = []
        for row in previous_entities:
            path = str(row.get("path", ""))
            if path and path not in unchanged_doc_paths:
                continue
            reusable_entities.append(Entity.model_validate({**row, "project_id": project_id}))

        docs_to_extract = [doc for doc in documents if doc.relative_path not in unchanged_doc_paths]
        return reusable_facts, chunks_to_extract, reusable_entities, docs_to_extract

    def _reuse_relations(self, dirs: dict[str, Path], project_id: str) -> list[Relation]:
        """Reuse previous relations when the graph inputs are unchanged."""

        return [
            Relation.model_validate({**row, "project_id": project_id})
            for row in read_jsonl(dirs["root"] / "relations.jsonl")
        ]

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

    def scan_project(self, path: str, name: str | None = None) -> ScanResult:
        """Compatibility wrapper used by the CLI."""
        return self.scan(ProjectCreate(path=path, name=name))

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
        dirs = ensure_output_dirs(root)
        reusable_facts: list[Fact] = []
        reusable_entities: list[Entity] = []
        chunks_to_extract = chunks
        docs_to_extract = documents
        if req.incremental:
            (
                reusable_facts,
                chunks_to_extract,
                reusable_entities,
                docs_to_extract,
            ) = self._reuse_incremental_memory(dirs, documents, chunks, project_id)

        cache = LLMExtractionCache(
            dirs["cache"] / "llm-cache.jsonl",
            enabled=settings.llm_cache_enabled,
        )
        should_reuse_relations = req.incremental and not chunks_to_extract and not docs_to_extract
        provider = None if should_reuse_relations else create_provider(req.llm_provider)
        extracted_facts = (
            asyncio.run(
                extract_facts(
                    chunks_to_extract,
                    project_id,
                    provider,
                    cache=cache,
                    concurrency=settings.llm_concurrency,
                )
            )
            if provider
            else []
        )
        facts = redact_model_strings([*reusable_facts, *extracted_facts])
        extracted_entities = (
            asyncio.run(
                extract_entities(
                    docs_to_extract,
                    project_id,
                    provider,
                    cache=cache,
                    concurrency=settings.llm_concurrency,
                )
            )
            if provider
            else []
        )
        entities = redact_model_strings([*reusable_entities, *extracted_entities])
        if should_reuse_relations:
            relations = self._reuse_relations(dirs, project_id)
        else:
            relations = asyncio.run(
                extract_relations(entities, facts, project_id, provider, cache=cache)
            )
        relations = redact_model_strings(relations)
        cache.save()

        # 4. generate
        wiki_pages = generate_wiki_pages(entities, facts, relations, project_id)
        graph = build_knowledge_graph(entities, relations, project_id)
        compact_ctx = build_brainframe_context(name, project_id, entities, relations, facts)

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
        # JSONL exports
        write_jsonl(dirs["root"] / "documents.jsonl", documents)
        write_jsonl(dirs["root"] / "chunks.jsonl", chunks)
        write_jsonl(dirs["root"] / "facts.jsonl", facts)
        write_jsonl(dirs["root"] / "entities.jsonl", entities)
        write_jsonl(dirs["root"] / "relations.jsonl", relations)

        # LLM compact context
        write_text_file(dirs["llm_context"], "brainframe.bf", compact_ctx)

        # Wiki pages
        _clear_directory(dirs["wiki"], (".md",))
        _clear_directory(dirs["wiki_mdx"], (".mdx",))
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

    def build_project(
        self,
        path: str,
        name: str | None = None,
        mode: str | None = None,
        llm_provider: str | None = None,
        incremental: bool = True,
    ) -> BuildResult:
        """Compatibility wrapper used by the CLI and older tests.

        The production pipeline always uses a configured provider; mode is kept
        only for older callers and is ignored.
        """
        _ = mode
        return self.build(
            ProjectCreate(
                path=path,
                name=name,
                llm_provider=llm_provider,
                incremental=incremental,
            )
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
        from llmbrain.core.identity import get_project_storage_dir

        db_path = get_project_storage_dir(project_id) / "brain.db"
        if db_path.exists():
            try:
                store = SQLiteStore(db_path)
                return store.get_project(project_id)
            except Exception:
                pass
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

    def token_report(self, project_id: str, max_chars: int = 120_000) -> dict:
        """Compare JSON-style context with compact BrainFrame context."""

        project = self._find_project(project_id)
        if project is None:
            raise FileNotFoundError(f"Project {project_id} not found.")

        facts = self.get_facts(project_id)
        entities = self.get_entities(project_id)
        relations = self.get_relations(project_id)
        json_context = json.dumps(
            {
                "project": project.name,
                "project_id": project.id,
                "entities": entities,
                "relations": relations,
                "facts": facts,
            },
            ensure_ascii=False,
            default=str,
        )
        brainframe_context = build_brainframe_context(
            project.name,
            project.id,
            entities,
            relations,
            facts,
            max_chars=max_chars,
        )
        result = compare_context_size(json_context, brainframe_context)
        result["max_chars"] = max_chars
        result["brainframe_truncated"] = "@truncated true" in brainframe_context
        result["entities"] = len(entities)
        result["relations"] = len(relations)
        result["facts"] = len(facts)
        return result

    def token_report_for_path(self, path: str, max_chars: int = 120_000) -> dict:
        """Resolve a project path and return its token efficiency report."""

        root = self._normalize_path(path)
        project_id = self._project_id_from_path(root)
        return self.token_report(project_id, max_chars=max_chars)

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
