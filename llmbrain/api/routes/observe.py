"""Observability endpoints — queue stats, resource status and service health."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/observe", tags=["observe"])


def _get_project_storage(path: str) -> tuple[str, Path]:
    """Resolve project_id and storage dir from a path string."""
    from llmbrain.core.identity import get_project_storage_dir, load_or_create_project_identity

    p = Path(path).expanduser().resolve()
    identity = load_or_create_project_identity(p)
    project_id = identity["project_id"]
    storage_dir = get_project_storage_dir(project_id)
    return project_id, storage_dir


# ── /observe/queue ────────────────────────────────────────────────────


@router.get("/queue")
async def queue_stats(path: str = Query(".", description="Project root path")) -> dict[str, Any]:
    """Return indexing queue statistics for a project."""
    try:
        project_id, storage_dir = _get_project_storage(path)
        db_path = storage_dir / "queue.db"
        if not db_path.exists():
            return {"project_id": project_id, "queue_db": str(db_path), "stats": {}}
        from llmbrain.core.queue import IndexQueue

        q = IndexQueue(db_path)
        stats = q.stats(project_id)
        return {"project_id": project_id, "queue_db": str(db_path), "stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/queue/{project_id}/jobs")
async def queue_jobs(
    project_id: str,
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Return jobs for a project from the indexing queue."""
    try:
        from llmbrain.core.identity import get_project_storage_dir
        from llmbrain.core.queue import IndexQueue

        db_path = get_project_storage_dir(project_id) / "queue.db"
        if not db_path.exists():
            return []
        q = IndexQueue(db_path)
        jobs = q.get_jobs(project_id, status=status, limit=limit)
        return [j.model_dump(mode="json") for j in jobs]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── /observe/resource ────────────────────────────────────────────────


@router.get("/resource")
async def resource_status() -> dict[str, Any]:
    """Return current CPU and memory resource statistics."""
    try:
        from llmbrain.core.resource_manager import ResourceManager

        rm = ResourceManager()
        for _ in range(3):
            rm.sample()
            await asyncio.sleep(0.05)
        stats = rm.get_stats()
        snapshots = [
            {
                "cpu_percent": s.cpu_percent,
                "memory_percent": s.memory_percent,
                "memory_mb": s.memory_mb,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in list(rm.snapshots)[-3:]
        ]
        return {**stats, "snapshots": snapshots}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── /observe/health ───────────────────────────────────────────────────


@router.get("/health")
async def service_health(
    endpoint: list[str] = Query([], description="Endpoints as name=url"),
) -> dict[str, Any]:
    """Check remote service health endpoints and return aggregated state."""
    try:
        from llmbrain.services.remote import (
            RemoteServiceMonitor,
            ServiceEndpoint,
        )

        endpoints = []
        for ep_str in endpoint:
            if "=" not in ep_str:
                continue
            name, url = ep_str.split("=", 1)
            endpoints.append(ServiceEndpoint(name=name.strip(), base_url=url.strip()))

        monitor = RemoteServiceMonitor(endpoints)
        results = await monitor.check_all()
        overall = monitor.get_overall_state()
        return {
            "overall": overall.value,
            "services": [
                {
                    "service_name": r.service_name,
                    "state": r.state.value,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "last_checked": r.last_checked.isoformat() if r.last_checked else None,
                }
                for r in results
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── /observe/profiler ─────────────────────────────────────────────────


@router.get("/profiler")
async def profiler_report(
    top: int = Query(10, ge=1, le=100, description="Number of slowest operations"),
) -> dict[str, Any]:
    """Return operation profiler report for the current server process."""
    try:
        from llmbrain.services.profiler import default_profiler

        data = default_profiler.as_dict()
        slowest = [e.model_dump(mode="json") for e in default_profiler.get_slowest(top)]
        return {**data, "slowest": slowest}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── /observe/semantic-search ──────────────────────────────────────────


@router.get("/semantic-search")
async def semantic_search(
    query: str = Query(..., description="Search query text"),
    path: str = Query(".", description="Project root path"),
    k: int = Query(10, ge=1, le=100),
    threshold: float = Query(0.25, ge=0.0, le=1.0),
    source_types: str | None = Query(None, description="Comma-separated: chunk,fact,entity"),
) -> dict[str, Any]:
    """Run semantic search over the indexed project memory."""
    try:
        from llmbrain.services.semantic_search import SemanticSearchService
        from llmbrain.storage.sqlite import SQLiteStore
        from llmbrain.storage.vector_store import VectorStore

        project_id, storage_dir = _get_project_storage(path)
        store = SQLiteStore(storage_dir / "brain.db")
        vs = VectorStore(storage_dir / "vectors.db")
        svc = SemanticSearchService(project_id, store, vs)

        types = [t.strip() for t in source_types.split(",")] if source_types else None
        results = svc.search(query, source_types=types, k=k, threshold=threshold)
        return {
            "query": query,
            "project_id": project_id,
            "results": [r.model_dump() for r in results],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
