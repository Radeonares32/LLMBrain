"""SQLite storage engine — canonical persistent store for all artefacts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from llmbrain.models.chunk import Chunk
from llmbrain.models.document import Document
from llmbrain.models.entity import Entity
from llmbrain.models.fact import Fact, FactEvidence
from llmbrain.models.project import Project
from llmbrain.models.relation import Relation
from llmbrain.models.wiki import WikiPage

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    root_path   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id),
    path          TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    language      TEXT NOT NULL DEFAULT 'unknown',
    line_count    INTEGER NOT NULL DEFAULT 0,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id),
    document_id   TEXT NOT NULL REFERENCES documents(id),
    path          TEXT NOT NULL,
    start_line    INTEGER NOT NULL,
    end_line      INTEGER NOT NULL,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    claim       TEXT NOT NULL DEFAULT '',
    confidence  TEXT NOT NULL DEFAULT 'medium',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_evidence (
    id          TEXT PRIMARY KEY,
    fact_id     TEXT NOT NULL REFERENCES facts(id),
    document_id TEXT NOT NULL REFERENCES documents(id),
    path        TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    path        TEXT NOT NULL DEFAULT '',
    confidence  TEXT NOT NULL DEFAULT 'medium',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS relations (
    id                TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL REFERENCES projects(id),
    source_entity_id  TEXT NOT NULL REFERENCES entities(id),
    relation          TEXT NOT NULL,
    target_entity_id  TEXT NOT NULL REFERENCES entities(id),
    evidence          TEXT NOT NULL DEFAULT '',
    confidence        TEXT NOT NULL DEFAULT 'medium'
);

CREATE TABLE IF NOT EXISTS wiki_pages (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL REFERENCES projects(id),
    title           TEXT NOT NULL,
    slug            TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'page',
    markdown_path   TEXT NOT NULL DEFAULT '',
    mdx_path        TEXT NOT NULL DEFAULT '',
    confidence      TEXT NOT NULL DEFAULT 'medium'
);
"""


class SQLiteStore:
    """Thin wrapper around a per-project SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── connection helpers ──────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(_SCHEMA_SQL)
        conn.close()

    # ── projects ────────────────────────────────────────────────────────

    def upsert_project(self, project: Project) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO projects (id, name, root_path, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (project.id, project.name, project.root_path,
                 project.created_at.isoformat(), project.updated_at.isoformat()),
            )

    def get_project(self, project_id: str) -> Project | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return Project(
                id=row["id"], name=row["name"], root_path=row["root_path"],
                created_at=row["created_at"], updated_at=row["updated_at"],
            )

    # ── documents ───────────────────────────────────────────────────────

    def insert_documents(self, docs: list[Document]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO documents "
                "(id, project_id, path, relative_path, content_hash, file_type, language, line_count, size_bytes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (d.id, d.project_id, d.path, d.relative_path, d.content_hash,
                     d.file_type, d.language, d.line_count, d.size_bytes, d.created_at.isoformat())
                    for d in docs
                ],
            )

    def get_documents(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE project_id = ? ORDER BY relative_path", (project_id,))
            return [dict(row) for row in cur.fetchall()]

    # ── chunks ──────────────────────────────────────────────────────────

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(id, project_id, document_id, path, start_line, end_line, content, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, c.project_id, c.document_id, c.path,
                     c.start_line, c.end_line, c.content, c.content_hash)
                    for c in chunks
                ],
            )

    def get_chunks(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT id, project_id, document_id, path, start_line, end_line, content_hash FROM chunks WHERE project_id = ? ORDER BY path, start_line", (project_id,))
            return [dict(row) for row in cur.fetchall()]

    # ── facts ───────────────────────────────────────────────────────────

    def insert_facts(self, facts: list[Fact]) -> None:
        with self._cursor() as cur:
            for f in facts:
                cur.execute(
                    "INSERT OR REPLACE INTO facts "
                    "(id, project_id, subject, predicate, object, claim, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (f.id, f.project_id, f.subject, f.predicate, f.object,
                     f.claim, f.confidence, f.created_at.isoformat()),
                )
                for ev in f.evidence:
                    cur.execute(
                        "INSERT OR REPLACE INTO fact_evidence "
                        "(id, fact_id, document_id, path, start_line, end_line) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (ev.id, ev.fact_id, ev.document_id, ev.path, ev.start_line, ev.end_line),
                    )

    def get_facts(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM facts WHERE project_id = ? ORDER BY subject, predicate", (project_id,))
            return [dict(row) for row in cur.fetchall()]

    # ── entities ────────────────────────────────────────────────────────

    def insert_entities(self, entities: list[Entity]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO entities "
                "(id, project_id, name, type, path, confidence, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (e.id, e.project_id, e.name, e.type, e.path, e.confidence,
                     json.dumps(e.metadata))
                    for e in entities
                ],
            )

    def get_entities(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM entities WHERE project_id = ? ORDER BY type, name", (project_id,))
            return [dict(row) for row in cur.fetchall()]

    # ── relations ───────────────────────────────────────────────────────

    def insert_relations(self, relations: list[Relation]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO relations "
                "(id, project_id, source_entity_id, relation, target_entity_id, evidence, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (r.id, r.project_id, r.source_entity_id, r.relation,
                     r.target_entity_id, r.evidence, r.confidence)
                    for r in relations
                ],
            )

    def get_relations(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM relations WHERE project_id = ? ORDER BY relation", (project_id,))
            return [dict(row) for row in cur.fetchall()]

    # ── wiki pages ──────────────────────────────────────────────────────

    def insert_wiki_pages(self, pages: list[WikiPage], wiki_dir: str = "", mdx_dir: str = "") -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO wiki_pages "
                "(id, project_id, title, slug, type, markdown_path, mdx_path, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (p.id, p.project_id, p.title, p.slug, p.type,
                     f"{wiki_dir}/{p.slug}.md" if wiki_dir else "",
                     f"{mdx_dir}/{p.slug}.mdx" if mdx_dir else "",
                     p.confidence)
                    for p in pages
                ],
            )

    def get_wiki_pages(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM wiki_pages WHERE project_id = ? ORDER BY title", (project_id,))
            return [dict(row) for row in cur.fetchall()]

    # ── cleanup ─────────────────────────────────────────────────────────

    def delete_project_data(self, project_id: str) -> None:
        """Remove all data for a project (for full rebuild)."""
        with self._cursor() as cur:
            # fact_evidence has no project_id — delete via fact_id FK
            cur.execute(
                "DELETE FROM fact_evidence WHERE fact_id IN "
                "(SELECT id FROM facts WHERE project_id = ?)",
                (project_id,),
            )
            tables = [
                "wiki_pages", "relations", "entities",
                "facts", "chunks", "documents",
            ]
            for table in tables:
                cur.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))

