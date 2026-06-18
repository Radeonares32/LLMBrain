"""SQLite-backed vector store — stores and retrieves embedding vectors per project."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

# ── Model ─────────────────────────────────────────────────────────────


class VectorRecord(BaseModel):
    """A stored embedding vector with provenance metadata."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    project_id: str
    source_type: str
    source_id: str
    text_preview: str = ""
    vector: list[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Schema ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    text_preview TEXT NOT NULL DEFAULT '',
    vector       TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_source
    ON embeddings (project_id, source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_type
    ON embeddings (project_id, source_type);
"""


def _row_to_record(row: sqlite3.Row) -> VectorRecord:
    d = dict(row)
    d["vector"] = json.loads(d["vector"])
    return VectorRecord.model_validate(d)


# ── VectorStore ───────────────────────────────────────────────────────


class VectorStore:
    """Thread-safe SQLite vector store for semantic embedding lookup."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── write ─────────────────────────────────────────────────────────

    def upsert(
        self,
        project_id: str,
        source_type: str,
        source_id: str,
        text_preview: str,
        vector: list[float],
    ) -> VectorRecord:
        """Insert or replace a vector record (unique on project+type+source)."""
        now = datetime.now(UTC).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "SELECT id FROM embeddings "
                "WHERE project_id = ? AND source_type = ? AND source_id = ?",
                (project_id, source_type, source_id),
            )
            existing = cur.fetchone()
            rec_id = existing["id"] if existing else uuid4().hex
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(id, project_id, source_type, source_id, text_preview, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    rec_id,
                    project_id,
                    source_type,
                    source_id,
                    text_preview[:500],
                    json.dumps(vector),
                    now,
                ),
            )
            self._conn.commit()
        return VectorRecord(
            id=rec_id,
            project_id=project_id,
            source_type=source_type,
            source_id=source_id,
            text_preview=text_preview[:500],
            vector=vector,
            created_at=datetime.fromisoformat(now),
        )

    def upsert_batch(self, records: list[dict]) -> int:
        """Bulk upsert.

        Each dict must have: project_id, source_type, source_id, text_preview, vector.
        """
        now = datetime.now(UTC).isoformat()
        count = 0
        with self._lock:
            for rec in records:
                self._conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(id, project_id, source_type, source_id, text_preview, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        uuid4().hex,
                        rec["project_id"],
                        rec["source_type"],
                        rec["source_id"],
                        rec.get("text_preview", "")[:500],
                        json.dumps(rec["vector"]),
                        now,
                    ),
                )
                count += 1
            self._conn.commit()
        return count

    # ── read ──────────────────────────────────────────────────────────

    def get_by_source(
        self, project_id: str, source_type: str, source_id: str
    ) -> VectorRecord | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM embeddings "
                "WHERE project_id = ? AND source_type = ? AND source_id = ?",
                (project_id, source_type, source_id),
            )
            row = cur.fetchone()
        return _row_to_record(row) if row else None

    def get_all(self, project_id: str, source_type: str | None = None) -> list[VectorRecord]:
        with self._lock:
            if source_type:
                cur = self._conn.execute(
                    "SELECT * FROM embeddings WHERE project_id = ? AND source_type = ?",
                    (project_id, source_type),
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM embeddings WHERE project_id = ?",
                    (project_id,),
                )
            rows = cur.fetchall()
        return [_row_to_record(row) for row in rows]

    # ── export ────────────────────────────────────────────────────────

    def export_jsonl(self, project_id: str, output_path: str | Path) -> int:
        """Export all vector records for a project to a JSONL file."""
        records = self.get_all(project_id)
        from llmbrain.storage.jsonl import write_jsonl
        return write_jsonl(output_path, records)

    # ── delete ────────────────────────────────────────────────────────

    def delete_project(self, project_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM embeddings WHERE project_id = ?", (project_id,))
            self._conn.commit()
        return cur.rowcount

    # ── stats ─────────────────────────────────────────────────────────

    def stats(self, project_id: str) -> dict:
        with self._lock:
            cur = self._conn.execute(
                "SELECT source_type, COUNT(*) as cnt FROM embeddings "
                "WHERE project_id = ? GROUP BY source_type",
                (project_id,),
            )
            rows = cur.fetchall()
        by_type = {row["source_type"]: row["cnt"] for row in rows}
        return {"total": sum(by_type.values()), "by_type": by_type}

    # ── search ────────────────────────────────────────────────────────

    def search(
        self,
        project_id: str,
        query_vec: list[float],
        source_type: str | None = None,
        k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[VectorRecord, float]]:
        """Brute-force cosine similarity search. Returns (record, score) sorted desc."""
        records = self.get_all(project_id, source_type=source_type)
        if not records:
            return []

        scored: list[tuple[VectorRecord, float]] = []
        for rec in records:
            score = _cosine(query_vec, rec.vector)
            if score >= threshold:
                scored.append((rec, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── helpers ───────────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
