"""Project endpoints — scan, build, and query knowledge artefacts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from llmbrain.models.project import BuildResult, ProjectCreate, ScanResult
from llmbrain.services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["projects"])

# A single service instance shared across requests (stateless, uses SQLite).
_svc = ProjectService()


# ── scan & build ────────────────────────────────────────────────────────────

@router.post("/scan", response_model=ScanResult)
def scan_project(body: ProjectCreate):
    """Scan a local project directory and index its documents."""
    try:
        return _svc.scan(body)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/build", response_model=BuildResult)
def build_project(body: ProjectCreate):
    """Run the full build pipeline and write .llmbrain output."""
    try:
        return _svc.build(body)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── query endpoints ─────────────────────────────────────────────────────────

def _resolve_project(project_id: str):
    project = _svc.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")
    return project


@router.get("/{project_id}/documents")
def list_documents(project_id: str):
    _resolve_project(project_id)
    return _svc.get_documents(project_id)


@router.get("/{project_id}/chunks")
def list_chunks(project_id: str):
    _resolve_project(project_id)
    return _svc.get_chunks(project_id)


@router.get("/{project_id}/facts")
def list_facts(project_id: str):
    _resolve_project(project_id)
    return _svc.get_facts(project_id)


@router.get("/{project_id}/entities")
def list_entities(project_id: str):
    _resolve_project(project_id)
    return _svc.get_entities(project_id)


@router.get("/{project_id}/relations")
def list_relations(project_id: str):
    _resolve_project(project_id)
    return _svc.get_relations(project_id)


@router.get("/{project_id}/wiki")
def get_wiki(project_id: str):
    _resolve_project(project_id)
    return _svc.get_wiki_pages(project_id)


@router.get("/{project_id}/graph")
def get_graph(project_id: str):
    _resolve_project(project_id)
    return _svc.get_graph(project_id)


@router.get("/{project_id}/context")
def get_context(project_id: str):
    _resolve_project(project_id)
    return _svc.get_compact_context(project_id)


@router.get("/{project_id}/token-report")
def get_token_report(
    project_id: str,
    max_chars: int = Query(120_000, ge=1),
):
    _resolve_project(project_id)
    return _svc.token_report(project_id, max_chars=max_chars)
