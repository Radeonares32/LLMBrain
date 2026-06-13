"""Async indexing job queue backed by a thread-safe SQLite database."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class JobPriority(IntEnum):
    """Lower value = higher priority (processed first)."""

    CRITICAL = 0
    INTERACTIVE = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class JobStatus(str):
    """Valid job lifecycle states (kept as plain strings for SQLite affinity)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"


class JobType(str):
    """Types of indexing operations that can be queued."""

    SCAN = "SCAN"
    CHUNK = "CHUNK"
    EXTRACT_FACTS = "EXTRACT_FACTS"
    EXTRACT_ENTITIES = "EXTRACT_ENTITIES"
    EXTRACT_RELATIONS = "EXTRACT_RELATIONS"
    FULL_BUILD = "FULL_BUILD"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class IndexJob(BaseModel):
    """A single indexing job stored in the queue."""

    id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:16]}")
    project_id: str
    job_type: str  # JobType value
    priority: int = int(JobPriority.NORMAL)  # JobPriority int value
    status: str = JobStatus.PENDING
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 2
    progress: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS index_jobs (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    job_type        TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 3,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    payload         TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    error           TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 2,
    progress        REAL NOT NULL DEFAULT 0.0
);
"""

_CREATE_IDX_PROJECT_STATUS = """
CREATE INDEX IF NOT EXISTS idx_jobs_project_status
ON index_jobs (project_id, status);
"""

_CREATE_IDX_PRIORITY_CREATED = """
CREATE INDEX IF NOT EXISTS idx_jobs_priority_created
ON index_jobs (priority ASC, created_at ASC);
"""


def _row_to_job(row: sqlite3.Row) -> IndexJob:
    """Convert a sqlite3.Row to an IndexJob model."""

    def _dt(val: str | None) -> datetime | None:
        if val is None:
            return None
        return datetime.fromisoformat(val)

    return IndexJob(
        id=row["id"],
        project_id=row["project_id"],
        job_type=row["job_type"],
        priority=row["priority"],
        status=row["status"],
        payload=json.loads(row["payload"]),
        created_at=_dt(row["created_at"]),  # type: ignore[arg-type]
        started_at=_dt(row["started_at"]),
        completed_at=_dt(row["completed_at"]),
        error=row["error"],
        retry_count=row["retry_count"],
        max_retries=row["max_retries"],
        progress=row["progress"],
    )


def _dt_iso(val: datetime | None) -> str | None:
    """Serialize a datetime to ISO-8601 string, preserving None."""
    if val is None:
        return None
    return val.isoformat()


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class IndexQueue:
    """Thread-safe, SQLite-backed indexing job queue.

    The queue uses WAL journal mode for better concurrent read performance.
    All public methods acquire an internal threading.Lock to ensure
    only one operation runs against the connection at a time.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialise the queue and ensure the database schema exists.

        Parameters
        ----------
        db_path:
            Absolute or relative path to the SQLite database file.
            Parent directories are created if necessary.
        """
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_CREATE_TABLE)
            self._conn.execute(_CREATE_IDX_PROJECT_STATUS)
            self._conn.execute(_CREATE_IDX_PRIORITY_CREATED)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        project_id: str,
        job_type: str,
        payload: dict[str, Any],
        priority: int = int(JobPriority.NORMAL),
        max_retries: int = 2,
    ) -> IndexJob:
        """Add a new job to the queue and return the created IndexJob.

        Parameters
        ----------
        project_id:
            Owning project identifier.
        job_type:
            One of the :class:`JobType` string constants.
        payload:
            Arbitrary JSON-serialisable data for the worker.
        priority:
            Integer priority value from :class:`JobPriority` (lower = sooner).
        max_retries:
            Maximum automatic retry attempts before the job is marked FAILED.
        """
        job = IndexJob(
            project_id=project_id,
            job_type=job_type,
            priority=priority,
            payload=payload,
            max_retries=max_retries,
        )
        sql = """
            INSERT INTO index_jobs
                (id, project_id, job_type, priority, status, payload,
                 created_at, started_at, completed_at, error,
                 retry_count, max_retries, progress)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute(
                sql,
                (
                    job.id,
                    job.project_id,
                    job.job_type,
                    job.priority,
                    job.status,
                    json.dumps(job.payload),
                    _dt_iso(job.created_at),
                    _dt_iso(job.started_at),
                    _dt_iso(job.completed_at),
                    job.error,
                    job.retry_count,
                    job.max_retries,
                    job.progress,
                ),
            )
            self._conn.commit()
        return job

    def dequeue_next(self) -> IndexJob | None:
        """Atomically pop the highest-priority oldest pending job.

        Returns
        -------
        IndexJob | None
            The dequeued job (now in RUNNING status), or *None* if the queue
            is empty.
        """
        sql_select = """
            SELECT * FROM index_jobs
            WHERE status IN ('PENDING', 'RETRYING')
            ORDER BY priority ASC, created_at ASC
            LIMIT 1
        """
        sql_update = """
            UPDATE index_jobs
            SET status = 'RUNNING', started_at = ?
            WHERE id = ? AND status IN ('PENDING', 'RETRYING')
        """
        now_iso = _dt_iso(datetime.now(UTC))
        with self._lock:
            row = self._conn.execute(sql_select).fetchone()
            if row is None:
                return None
            job_id = row["id"]
            affected = self._conn.execute(sql_update, (now_iso, job_id)).rowcount
            self._conn.commit()
            if affected == 0:
                # Lost the race — try again (recursive call inside lock is safe
                # because our lock is non-reentrant; return None for safety).
                return None
            # Re-fetch to get updated row
            updated_row = self._conn.execute(
                "SELECT * FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(updated_row)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update arbitrary fields on an existing job.

        Only columns present in the ``index_jobs`` table are accepted.
        Datetime fields may be passed as :class:`datetime` objects or ISO
        strings.  The ``payload`` field may be a :class:`dict`.

        Parameters
        ----------
        job_id:
            Target job identifier.
        **kwargs:
            Column names and new values.
        """
        _allowed = {
            "status",
            "priority",
            "payload",
            "started_at",
            "completed_at",
            "error",
            "retry_count",
            "max_retries",
            "progress",
        }
        updates: dict[str, Any] = {}
        for key, val in kwargs.items():
            if key not in _allowed:
                raise ValueError(f"Unknown job field: {key!r}")
            if isinstance(val, datetime):
                updates[key] = _dt_iso(val)
            elif key == "payload" and isinstance(val, dict):
                updates[key] = json.dumps(val)
            else:
                updates[key] = val

        if not updates:
            return

        set_clause = ", ".join(f"{col} = ?" for col in updates)
        values = list(updates.values()) + [job_id]
        sql = f"UPDATE index_jobs SET {set_clause} WHERE id = ?"  # noqa: S608
        with self._lock:
            self._conn.execute(sql, values)
            self._conn.commit()

    def get_job(self, job_id: str) -> IndexJob | None:
        """Return a single job by ID, or *None* if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(row) if row else None

    def get_jobs(
        self,
        project_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[IndexJob]:
        """Return jobs for a project, optionally filtered by status.

        Results are ordered by priority (ascending) then creation time
        (ascending).

        Parameters
        ----------
        project_id:
            Filter by this project identifier.
        status:
            Optional :class:`JobStatus` string to filter on.
        limit:
            Maximum number of rows to return.
        """
        if status is not None:
            sql = """
                SELECT * FROM index_jobs
                WHERE project_id = ? AND status = ?
                ORDER BY priority ASC, created_at ASC
                LIMIT ?
            """
            params: tuple[Any, ...] = (project_id, status, limit)
        else:
            sql = """
                SELECT * FROM index_jobs
                WHERE project_id = ?
                ORDER BY priority ASC, created_at ASC
                LIMIT ?
            """
            params = (project_id, limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_job(row) for row in rows]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job.

        Returns
        -------
        bool
            *True* if the job was found and cancelled; *False* otherwise.
        """
        sql = """
            UPDATE index_jobs
            SET status = 'CANCELLED'
            WHERE id = ? AND status IN ('PENDING', 'RUNNING', 'RETRYING')
        """
        with self._lock:
            affected = self._conn.execute(sql, (job_id,)).rowcount
            self._conn.commit()
        return affected > 0

    def retry_failed(self, project_id: str) -> int:
        """Re-queue all FAILED jobs for the given project that still have
        retry budget remaining.

        Each qualifying job has its status set to RETRYING and its
        ``retry_count`` incremented by one.

        Returns
        -------
        int
            Number of jobs re-queued.
        """
        sql_select = """
            SELECT id, retry_count, max_retries FROM index_jobs
            WHERE project_id = ? AND status = 'FAILED'
        """
        sql_update = """
            UPDATE index_jobs
            SET status = 'RETRYING', retry_count = retry_count + 1,
                error = NULL, started_at = NULL
            WHERE id = ?
        """
        retried = 0
        with self._lock:
            rows = self._conn.execute(sql_select, (project_id,)).fetchall()
            for row in rows:
                if row["retry_count"] < row["max_retries"]:
                    self._conn.execute(sql_update, (row["id"],))
                    retried += 1
            self._conn.commit()
        return retried

    def stats(self, project_id: str) -> dict[str, int]:
        """Return a mapping of status → count for the given project.

        Parameters
        ----------
        project_id:
            Target project identifier.
        """
        sql = """
            SELECT status, COUNT(*) AS cnt
            FROM index_jobs
            WHERE project_id = ?
            GROUP BY status
        """
        with self._lock:
            rows = self._conn.execute(sql, (project_id,)).fetchall()
        result: dict[str, int] = {
            JobStatus.PENDING: 0,
            JobStatus.RUNNING: 0,
            JobStatus.COMPLETED: 0,
            JobStatus.FAILED: 0,
            JobStatus.CANCELLED: 0,
            JobStatus.RETRYING: 0,
        }
        for row in rows:
            result[row["status"]] = row["cnt"]
        return result

    def purge_completed(self, project_id: str, older_than_hours: float = 24.0) -> int:
        """Delete completed (and cancelled) jobs older than the given threshold.

        Parameters
        ----------
        project_id:
            Target project identifier.
        older_than_hours:
            Jobs whose ``completed_at`` is older than this many hours are
            removed.

        Returns
        -------
        int
            Number of rows deleted.
        """
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(hours=older_than_hours)
        cutoff_iso = _dt_iso(cutoff)
        sql = """
            DELETE FROM index_jobs
            WHERE project_id = ?
              AND status IN ('COMPLETED', 'CANCELLED')
              AND completed_at IS NOT NULL
              AND completed_at < ?
        """
        with self._lock:
            affected = self._conn.execute(sql, (project_id, cutoff_iso)).rowcount
            self._conn.commit()
        return affected

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
