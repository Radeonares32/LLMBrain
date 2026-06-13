"""FastAPI application entry-point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llmbrain.api.routes.health import router as health_router
from llmbrain.api.routes.observe import router as observe_router
from llmbrain.api.routes.projects import router as projects_router
from llmbrain.core.config import settings


def create_app() -> FastAPI:
    """Application factory."""

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Engineering knowledge compiler — turns repos, docs, and source code "
            "into a structured, queryable knowledge base."
        ),
    )

    # ── middleware ───────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── routers ─────────────────────────────────────────────────────────
    application.include_router(health_router)
    application.include_router(projects_router)
    application.include_router(observe_router)

    return application


app = create_app()
