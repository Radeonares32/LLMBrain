"""SQLite storage engine — canonical persistent store for all artefacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmbrain.models.chunk import Chunk
from llmbrain.models.document import Document
from llmbrain.models.entity import Entity
from llmbrain.models.fact import Fact
from llmbrain.models.project import Project
from llmbrain.models.relation import Relation
from llmbrain.models.wiki import WikiPage


def _wiki_sources_from_markdown(content: str) -> list[dict]:
    """Read source references from generated wiki frontmatter."""
    sources: list[dict] = []
    current: dict | None = None
    for line in content.splitlines():
        path_match = re.match(r"\s*-\s+path:\s*(.+)\s*$", line)
        if path_match:
            current = {"path": path_match.group(1).strip(), "start_line": 0, "end_line": 0}
            sources.append(current)
            continue
        if current is None:
            continue
        start_match = re.match(r"\s*start_line:\s*(\d+)\s*$", line)
        if start_match:
            current["start_line"] = int(start_match.group(1))
            continue
        end_match = re.match(r"\s*end_line:\s*(\d+)\s*$", line)
        if end_match:
            current["end_line"] = int(end_match.group(1))
    return sources


class DatabaseMigrator:
    """Manages transactional schema migrations for SQLite databases."""

    def __init__(self, conn: sqlite3.Connection, db_name: str) -> None:
        self.conn = conn
        self.db_name = db_name
        self._init_meta()

    def _init_meta(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)"  # noqa: E501
        )
        self.conn.commit()

    def get_current_version(self) -> int:
        cur = self.conn.execute("SELECT max(version) FROM schema_version")
        val = cur.fetchone()[0]
        return val if val is not None else 0

    def apply_migrations(self, migrations: dict[int, list[str]]) -> None:
        current = self.get_current_version()
        target = max(migrations.keys()) if migrations else 0
        if current >= target:
            return

        for ver in sorted(migrations.keys()):
            if ver <= current:
                continue
            # Set isolation_level=None to control transaction manually
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for sql in migrations[ver]:
                    self.conn.execute(sql)
                self.conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (ver, datetime.now(UTC).isoformat()),
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise RuntimeError(
                    f"Migration of {self.db_name} to version {ver} failed: {e}. Rollback executed."
                ) from e


BRAIN_MIGRATIONS = {
    1: [
        """CREATE TABLE IF NOT EXISTS projects (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            root_path   TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS documents (
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
        )""",
        """CREATE TABLE IF NOT EXISTS chunks (
            id            TEXT PRIMARY KEY,
            project_id    TEXT NOT NULL REFERENCES projects(id),
            document_id   TEXT NOT NULL REFERENCES documents(id),
            path          TEXT NOT NULL,
            start_line    INTEGER NOT NULL,
            end_line      INTEGER NOT NULL,
            content       TEXT NOT NULL,
            content_hash  TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS facts (
            id          TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL REFERENCES projects(id),
            subject     TEXT NOT NULL,
            predicate   TEXT NOT NULL,
            object      TEXT NOT NULL,
            claim       TEXT NOT NULL DEFAULT '',
            confidence  TEXT NOT NULL DEFAULT 'medium',
            created_at  TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS fact_evidence (
            id          TEXT PRIMARY KEY,
            fact_id     TEXT NOT NULL REFERENCES facts(id),
            document_id TEXT NOT NULL REFERENCES documents(id),
            path        TEXT NOT NULL,
            start_line  INTEGER NOT NULL,
            end_line    INTEGER NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS entities (
            id          TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL REFERENCES projects(id),
            name        TEXT NOT NULL,
            type        TEXT NOT NULL,
            path        TEXT NOT NULL DEFAULT '',
            confidence  TEXT NOT NULL DEFAULT 'medium',
            metadata    TEXT NOT NULL DEFAULT '{}'
        )""",
        """CREATE TABLE IF NOT EXISTS relations (
            id                TEXT PRIMARY KEY,
            project_id        TEXT NOT NULL REFERENCES projects(id),
            source_entity_id  TEXT NOT NULL REFERENCES entities(id),
            relation          TEXT NOT NULL,
            target_entity_id  TEXT NOT NULL REFERENCES entities(id),
            evidence          TEXT NOT NULL DEFAULT '',
            confidence        TEXT NOT NULL DEFAULT 'medium'
        )""",
        """CREATE TABLE IF NOT EXISTS wiki_pages (
            id              TEXT PRIMARY KEY,
            project_id      TEXT NOT NULL REFERENCES projects(id),
            title           TEXT NOT NULL,
            slug            TEXT NOT NULL,
            type            TEXT NOT NULL DEFAULT 'page',
            markdown_path   TEXT NOT NULL DEFAULT '',
            mdx_path        TEXT NOT NULL DEFAULT '',
            confidence      TEXT NOT NULL DEFAULT 'medium'
        )""",
        """CREATE TABLE IF NOT EXISTS task_runs (
            id            TEXT PRIMARY KEY,
            project_id    TEXT NOT NULL REFERENCES projects(id),
            request       TEXT NOT NULL,
            summary       TEXT,
            commit_hash   TEXT,
            status        TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS task_decisions (
            id            TEXT PRIMARY KEY,
            task_id       TEXT NOT NULL REFERENCES task_runs(id),
            decision      TEXT NOT NULL,
            rationale     TEXT,
            created_at    TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS task_commands (
            id            TEXT PRIMARY KEY,
            task_id       TEXT NOT NULL REFERENCES task_runs(id),
            command       TEXT NOT NULL,
            output        TEXT,
            status        TEXT NOT NULL,
            created_at    TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS task_failures (
            id            TEXT PRIMARY KEY,
            task_id       TEXT NOT NULL REFERENCES task_runs(id),
            failure       TEXT NOT NULL,
            resolution    TEXT,
            created_at    TEXT NOT NULL
        )""",
    ],
    2: [
        # Migration test example: Add state column to facts for bounded memory management
        "ALTER TABLE facts ADD COLUMN state TEXT NOT NULL DEFAULT 'active'"
    ],
    3: [
        # Phase 6: Async indexing queue persistence
        """CREATE TABLE IF NOT EXISTS indexing_queue (
            id            TEXT PRIMARY KEY,
            project_id    TEXT NOT NULL,
            job_type      TEXT NOT NULL,
            priority      INTEGER NOT NULL DEFAULT 3,
            status        TEXT NOT NULL DEFAULT 'pending',
            payload       TEXT NOT NULL DEFAULT '{}',
            created_at    TEXT NOT NULL,
            started_at    TEXT,
            completed_at  TEXT,
            error         TEXT,
            retry_count   INTEGER NOT NULL DEFAULT 0,
            max_retries   INTEGER NOT NULL DEFAULT 2,
            progress      REAL NOT NULL DEFAULT 0.0
        )""",
        """CREATE INDEX IF NOT EXISTS idx_queue_project_status
            ON indexing_queue (project_id, status, priority, created_at)""",
        # Phase 6: Indexing checkpoints for crash recovery
        """CREATE TABLE IF NOT EXISTS indexing_checkpoints (
            id            TEXT PRIMARY KEY,
            project_id    TEXT NOT NULL UNIQUE,
            last_job_id   TEXT,
            scanned_files INTEGER NOT NULL DEFAULT 0,
            indexed_files INTEGER NOT NULL DEFAULT 0,
            updated_at    TEXT NOT NULL
        )""",
    ],
}


class SQLiteStore:
    """Thin wrapper around a per-project SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── connection helpers ──────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
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
        try:
            migrator = DatabaseMigrator(conn, "brain.db")
            migrator.apply_migrations(BRAIN_MIGRATIONS)
        finally:
            conn.close()

    # ── projects ────────────────────────────────────────────────────────

    def upsert_project(self, project: Project) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO projects (id, name, root_path, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    project.id,
                    project.name,
                    project.root_path,
                    project.created_at.isoformat()
                    if hasattr(project.created_at, "isoformat")
                    else str(project.created_at),
                    project.updated_at.isoformat()
                    if hasattr(project.updated_at, "isoformat")
                    else str(project.updated_at),
                ),
            )

    def get_project(self, project_id: str) -> Project | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return Project(
                id=row["id"],
                name=row["name"],
                root_path=row["root_path"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    # ── documents ───────────────────────────────────────────────────────

    def insert_documents(self, docs: list[Document]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO documents "
                "(id, project_id, path, relative_path, content_hash, file_type, "
                "language, line_count, size_bytes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        d.id,
                        d.project_id,
                        d.path,
                        d.relative_path,
                        d.content_hash,
                        d.file_type,
                        d.language,
                        d.line_count,
                        d.size_bytes,
                        d.created_at.isoformat()
                        if hasattr(d.created_at, "isoformat")
                        else str(d.created_at),
                    )
                    for d in docs
                ],
            )

    def get_documents(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM documents WHERE project_id = ? ORDER BY relative_path",
                (project_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    # ── chunks ──────────────────────────────────────────────────────────

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(id, project_id, document_id, path, start_line, end_line, "
                "content, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.id,
                        c.project_id,
                        c.document_id,
                        c.path,
                        c.start_line,
                        c.end_line,
                        c.content,
                        c.content_hash,
                    )
                    for c in chunks
                ],
            )

    def get_chunks(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, project_id, document_id, path, start_line, "
                "end_line, content_hash FROM chunks WHERE project_id = ? "
                "ORDER BY path, start_line",
                (project_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    # ── facts ───────────────────────────────────────────────────────────

    def insert_facts(self, facts: list[Fact]) -> None:
        with self._cursor() as cur:
            for f in facts:
                cur.execute(
                    "INSERT OR REPLACE INTO facts "
                    "(id, project_id, subject, predicate, object, claim, confidence, created_at, state) "  # noqa: E501
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f.id,
                        f.project_id,
                        f.subject,
                        f.predicate,
                        f.object,
                        f.claim,
                        f.confidence,
                        f.created_at.isoformat()
                        if hasattr(f.created_at, "isoformat")
                        else str(f.created_at),
                        getattr(f, "state", "active"),
                    ),
                )
                for ev in f.evidence:
                    cur.execute(
                        "INSERT OR REPLACE INTO fact_evidence "
                        "(id, fact_id, document_id, path, start_line, end_line) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            ev.id,
                            ev.fact_id,
                            ev.document_id,
                            ev.path,
                            ev.start_line,
                            ev.end_line,
                        ),
                    )

    def get_facts(self, project_id: str, state: str | None = None) -> list[dict]:
        with self._cursor() as cur:
            if state:
                cur.execute(
                    "SELECT * FROM facts WHERE project_id = ? AND state = ? ORDER BY subject, predicate",  # noqa: E501
                    (project_id, state),
                )
            else:
                cur.execute(
                    "SELECT * FROM facts WHERE project_id = ? ORDER BY subject, predicate",
                    (project_id,),
                )
            facts = [dict(row) for row in cur.fetchall()]
            for fact in facts:
                cur.execute(
                    "SELECT * FROM fact_evidence WHERE fact_id = ? ORDER BY start_line",
                    (fact["id"],),
                )
                fact["evidence"] = [dict(row) for row in cur.fetchall()]
            return facts

    def update_fact_state(self, fact_id: str, state: str) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE facts SET state = ? WHERE id = ?", (state, fact_id))

    # ── entities ────────────────────────────────────────────────────────

    def insert_entities(self, entities: list[Entity]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO entities "
                "(id, project_id, name, type, path, confidence, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        e.id,
                        e.project_id,
                        e.name,
                        e.type,
                        e.path,
                        e.confidence,
                        json.dumps(e.metadata),
                    )
                    for e in entities
                ],
            )

    def get_entities(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM entities WHERE project_id = ? ORDER BY type, name",
                (project_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    # ── relations ───────────────────────────────────────────────────────

    def insert_relations(self, relations: list[Relation]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO relations "
                "(id, project_id, source_entity_id, relation, target_entity_id, "
                "evidence, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        r.id,
                        r.project_id,
                        r.source_entity_id,
                        r.relation,
                        r.target_entity_id,
                        r.evidence,
                        r.confidence,
                    )
                    for r in relations
                ],
            )

    def get_relations(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM relations WHERE project_id = ? ORDER BY relation",
                (project_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    # ── wiki pages ──────────────────────────────────────────────────────

    def insert_wiki_pages(
        self,
        pages: list[WikiPage],
        wiki_dir: str = "",
        mdx_dir: str = "",
    ) -> None:
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO wiki_pages "
                "(id, project_id, title, slug, type, markdown_path, mdx_path, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        p.id,
                        p.project_id,
                        p.title,
                        p.slug,
                        p.type,
                        f"{wiki_dir}/{p.slug}.md" if wiki_dir else "",
                        f"{mdx_dir}/{p.slug}.mdx" if mdx_dir else "",
                        p.confidence,
                    )
                    for p in pages
                ],
            )

    def get_wiki_pages(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM wiki_pages WHERE project_id = ? ORDER BY title",
                (project_id,),
            )
            pages = [dict(row) for row in cur.fetchall()]
            for page in pages:
                markdown_path = page.get("markdown_path")
                if not markdown_path:
                    continue
                path = Path(markdown_path)
                if not path.exists():
                    continue
                content = path.read_text(encoding="utf-8")
                page["markdown_content"] = content
                page["sources"] = _wiki_sources_from_markdown(content)
            return pages

    # ── cleanup ─────────────────────────────────────────────────────────

    def delete_project_data(self, project_id: str) -> None:
        """Remove all data for a project (for full rebuild)."""
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM fact_evidence WHERE fact_id IN "
                "(SELECT id FROM facts WHERE project_id = ?)",
                (project_id,),
            )
            cur.execute(
                "DELETE FROM task_failures WHERE task_id IN "
                "(SELECT id FROM task_runs WHERE project_id = ?)",
                (project_id,),
            )
            cur.execute(
                "DELETE FROM task_commands WHERE task_id IN "
                "(SELECT id FROM task_runs WHERE project_id = ?)",
                (project_id,),
            )
            cur.execute(
                "DELETE FROM task_decisions WHERE task_id IN "
                "(SELECT id FROM task_runs WHERE project_id = ?)",
                (project_id,),
            )
            cur.execute("DELETE FROM task_runs WHERE project_id = ?", (project_id,))

            tables = [
                "wiki_pages",
                "relations",
                "entities",
                "facts",
                "chunks",
                "documents",
            ]
            for table in tables:
                cur.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))

    # ── task persistence ───────────────────────────────────────────────

    def insert_task_run(
        self,
        id: str,
        project_id: str,
        request: str,
        summary: str | None,
        commit_hash: str | None,
        status: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO task_runs "
                "(id, project_id, request, summary, commit_hash, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (id, project_id, request, summary, commit_hash, status, created_at, updated_at),
            )

    def insert_task_decision(
        self,
        id: str,
        task_id: str,
        decision: str,
        rationale: str | None,
        created_at: str,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO task_decisions "
                "(id, task_id, decision, rationale, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (id, task_id, decision, rationale, created_at),
            )

    def insert_task_command(
        self,
        id: str,
        task_id: str,
        command: str,
        output: str | None,
        status: str,
        created_at: str,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO task_commands "
                "(id, task_id, command, output, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (id, task_id, command, output, status, created_at),
            )

    def insert_task_failure(
        self,
        id: str,
        task_id: str,
        failure: str,
        resolution: str | None,
        created_at: str,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO task_failures "
                "(id, task_id, failure, resolution, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (id, task_id, failure, resolution, created_at),
            )

    def get_task_runs(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM task_runs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_task_details(self, task_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM task_runs WHERE id = ?", (task_id,))
            row = cur.fetchone()
            if not row:
                return None
            run = dict(row)

            cur.execute(
                "SELECT * FROM task_decisions WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            )
            run["decisions"] = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT * FROM task_commands WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            )
            run["commands"] = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT * FROM task_failures WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            )
            run["failures"] = [dict(r) for r in cur.fetchall()]

            return run


# ── Sessions Store ──────────────────────────────────────────────────

SESSION_MIGRATIONS = {
    1: [
        """CREATE TABLE IF NOT EXISTS sessions (
            id                  TEXT PRIMARY KEY,
            project_id          TEXT NOT NULL,
            title               TEXT NOT NULL,
            active_agent        TEXT NOT NULL,
            model_config        TEXT NOT NULL,
            permission_mode     TEXT NOT NULL,
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL,
            status              TEXT NOT NULL,
            parent_session_id   TEXT,
            branch              TEXT,
            commit_hash         TEXT,
            token_usage         TEXT NOT NULL DEFAULT '{}',
            compaction_state    TEXT NOT NULL DEFAULT '{}'
        )""",
        """CREATE TABLE IF NOT EXISTS session_messages (
            id                  TEXT PRIMARY KEY,
            session_id          TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role                TEXT NOT NULL,
            content             TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            token_estimate      INTEGER DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS session_tool_calls (
            id                  TEXT PRIMARY KEY,
            session_id          TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            tool_name           TEXT NOT NULL,
            arguments           TEXT NOT NULL,
            result              TEXT,
            status              TEXT NOT NULL,
            duration_ms         INTEGER DEFAULT 0,
            timestamp           TEXT NOT NULL
        )""",
    ]
}


class SessionStore:
    """SQLite-backed session repository."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
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
        try:
            migrator = DatabaseMigrator(conn, "sessions.db")
            migrator.apply_migrations(SESSION_MIGRATIONS)
        finally:
            conn.close()

    def create_session(
        self,
        id: str,
        project_id: str,
        title: str,
        active_agent: str,
        model_config: dict,
        permission_mode: str,
        status: str = "active",
        parent_session_id: str | None = None,
        branch: str | None = None,
        commit_hash: str | None = None,
    ) -> dict:
        now_str = datetime.now(UTC).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, project_id, title, active_agent, model_config, "
                "permission_mode, created_at, updated_at, status, parent_session_id, branch, commit_hash) "  # noqa: E501
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    id,
                    project_id,
                    title,
                    active_agent,
                    json.dumps(model_config),
                    permission_mode,
                    now_str,
                    now_str,
                    status,
                    parent_session_id,
                    branch,
                    commit_hash,
                ),
            )
        return self.get_session(id)

    def update_session(self, id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        kwargs["updated_at"] = datetime.now(UTC).isoformat()

        # Serialize dict columns
        if "model_config" in kwargs:
            kwargs["model_config"] = json.dumps(kwargs["model_config"])
        if "token_usage" in kwargs:
            kwargs["token_usage"] = json.dumps(kwargs["token_usage"])
        if "compaction_state" in kwargs:
            kwargs["compaction_state"] = json.dumps(kwargs["compaction_state"])

        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values())
        values.append(id)

        with self._cursor() as cur:
            cur.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", tuple(values))

    def get_session(self, id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE id = ?", (id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            res["model_config"] = json.loads(res["model_config"])
            res["token_usage"] = json.loads(res["token_usage"])
            res["compaction_state"] = json.loads(res["compaction_state"])
            return res

    def get_sessions(self, project_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM sessions WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            )
            rows = cur.fetchall()
            res = []
            for row in rows:
                d = dict(row)
                d["model_config"] = json.loads(d["model_config"])
                d["token_usage"] = json.loads(d["token_usage"])
                d["compaction_state"] = json.loads(d["compaction_state"])
                res.append(d)
            return res

    def delete_session(self, id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE id = ?", (id,))

    # ── messages ────────────────────────────────────────────────────────

    def insert_message(
        self,
        id: str,
        session_id: str,
        role: str,
        content: str,
        token_estimate: int = 0,
    ) -> None:
        now_str = datetime.now(UTC).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO session_messages (id, session_id, role, content, timestamp, token_estimate) "  # noqa: E501
                "VALUES (?, ?, ?, ?, ?, ?)",
                (id, session_id, role, content, now_str, token_estimate),
            )

    def get_messages(self, session_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM session_messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ? OFFSET ?",  # noqa: E501
                (session_id, limit, offset),
            )
            return [dict(row) for row in cur.fetchall()]

    # ── tool calls ──────────────────────────────────────────────────────

    def insert_tool_call(
        self,
        id: str,
        session_id: str,
        tool_name: str,
        arguments: dict | str,
        result: str | None = None,
        status: str = "requested",
        duration_ms: int = 0,
    ) -> None:
        now_str = datetime.now(UTC).isoformat()
        args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO session_tool_calls "
                "(id, session_id, tool_name, arguments, result, status, duration_ms, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (id, session_id, tool_name, args_str, result, status, duration_ms, now_str),
            )

    def get_tool_calls(self, session_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM session_tool_calls WHERE session_id = ? ORDER BY timestamp ASC LIMIT ? OFFSET ?",  # noqa: E501
                (session_id, limit, offset),
            )
            return [dict(row) for row in cur.fetchall()]


# ── Events Store ────────────────────────────────────────────────────

EVENT_MIGRATIONS = {
    1: [
        """CREATE TABLE IF NOT EXISTS events (
            id                  TEXT PRIMARY KEY,
            project_id          TEXT NOT NULL,
            session_id          TEXT,
            event_type          TEXT NOT NULL,
            payload             TEXT NOT NULL,
            timestamp           TEXT NOT NULL
        )"""
    ]
}


class EventStore:
    """SQLite-backed event and audit logger."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
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
        try:
            migrator = DatabaseMigrator(conn, "events.db")
            migrator.apply_migrations(EVENT_MIGRATIONS)
        finally:
            conn.close()

    def insert_event(
        self,
        id: str,
        project_id: str,
        session_id: str | None,
        event_type: str,
        payload: dict | str,
    ) -> None:
        now_str = datetime.now(UTC).isoformat()
        payload_str = payload if isinstance(payload, str) else json.dumps(payload)
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO events (id, project_id, session_id, event_type, payload, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (id, project_id, session_id, event_type, payload_str, now_str),
            )

    def get_events(
        self,
        project_id: str,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        with self._cursor() as cur:
            if session_id:
                cur.execute(
                    "SELECT * FROM events WHERE project_id = ? AND session_id = ? ORDER BY timestamp ASC LIMIT ? OFFSET ?",  # noqa: E501
                    (project_id, session_id, limit, offset),
                )
            else:
                cur.execute(
                    "SELECT * FROM events WHERE project_id = ? ORDER BY timestamp ASC LIMIT ? OFFSET ?",  # noqa: E501
                    (project_id, limit, offset),
                )
            return [dict(row) for row in cur.fetchall()]


# ── Backup & Restore ────────────────────────────────────────────────


def backup_project_db(project_id: str, backup_zip_path: str | Path) -> None:
    """Create a consistent zip backup of all project database files."""
    from llmbrain.core.identity import get_project_storage_dir

    storage_dir = get_project_storage_dir(project_id)
    backup_zip_path = Path(backup_zip_path)
    backup_zip_path.parent.mkdir(parents=True, exist_ok=True)

    db_files = ["brain.db", "sessions.db", "events.db"]
    manifest = {
        "backup_version": 1,
        "project_id": project_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "schema_version": 2,  # brain.db max schema version
        "files": {},
    }

    # Backup under a transaction lock or copying
    temp_dir = storage_dir / "backup_temp"
    temp_dir.mkdir(exist_ok=True)
    try:
        for db in db_files:
            p = storage_dir / db
            if p.exists():
                temp_p = temp_dir / db
                # Use sqlite3 VACUUM INTO or copy for simple backup
                conn = sqlite3.connect(str(p))
                try:
                    conn.execute(f"VACUUM INTO '{temp_p}'")
                finally:
                    conn.close()

                # Calculate hash of copy
                h = hashlib.sha256()
                h.update(temp_p.read_bytes())
                manifest["files"][db] = h.hexdigest()

        # Write manifest
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Zip files
        with zipfile.ZipFile(backup_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for f in temp_dir.iterdir():
                zipf.write(f, f.name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def restore_project_db(project_id: str, backup_zip_path: str | Path) -> None:
    """Restore project databases from a zip backup after validation."""
    from llmbrain.core.identity import get_project_storage_dir

    storage_dir = get_project_storage_dir(project_id)
    backup_zip_path = Path(backup_zip_path)
    if not backup_zip_path.exists():
        raise FileNotFoundError(f"Backup zip not found: {backup_zip_path}")

    # Extract to temp directory to validate
    temp_dir = storage_dir / "restore_temp"
    temp_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(backup_zip_path, "r") as zipf:
            zipf.extractall(temp_dir)

        manifest_path = temp_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Invalid backup: manifest.json is missing.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("project_id") != project_id:
            raise ValueError(
                f"Backup project_id mismatch. Expected {project_id}, got {manifest.get('project_id')}"  # noqa: E501
            )

        # Validate hashes
        for db, expected_hash in manifest.get("files", {}).items():
            db_file = temp_dir / db
            if not db_file.exists():
                raise ValueError(f"Backup missing expected file: {db}")
            h = hashlib.sha256()
            h.update(db_file.read_bytes())
            if h.hexdigest() != expected_hash:
                raise ValueError(f"Integrity check failed for {db} in backup.")

        # Create safety backup of existing files
        safety_zip = storage_dir / f"safety_backup_{int(datetime.now(UTC).timestamp())}.zip"
        backup_project_db(project_id, safety_zip)

        # Overwrite database files
        for db in manifest.get("files", {}).keys():
            src = temp_dir / db
            dst = storage_dir / db
            shutil.copy2(src, dst)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
