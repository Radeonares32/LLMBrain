"""Entity extractor — identifies engineering artefacts in documents.

MVP uses heuristics; future versions will delegate to an LLM adapter.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from uuid import uuid4

from llmbrain.models.document import Document
from llmbrain.models.entity import Entity

# patterns
_ENV_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
_ENDPOINT_RE = re.compile(
    r"""(?:@(?:app|router)\.(get|post|put|patch|delete)\(['"]([^'"]+)['"])""",
    re.IGNORECASE,
)
_DOCKERFILE_FROM_RE = re.compile(r"^FROM\s+(\S+)", re.MULTILINE)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def extract_entities_from_document(doc: Document, project_id: str) -> list[Entity]:
    """Extract entities from a single document via heuristics."""

    entities: list[Entity] = []
    content = doc.content or ""

    # ── file entity (always) ────────────────────────────────────────────
    entities.append(
        Entity(
            id=uuid4().hex,
            project_id=project_id,
            name=doc.relative_path,
            type="file",
            path=doc.relative_path,
            confidence="high",
        )
    )

    # ── detect API endpoints ────────────────────────────────────────────
    for m in _ENDPOINT_RE.finditer(content):
        method = m.group(1).upper()
        path = m.group(2)
        entities.append(
            Entity(
                id=uuid4().hex,
                project_id=project_id,
                name=f"{method} {path}",
                type="api_endpoint",
                path=doc.relative_path,
                confidence="high",
                metadata={"method": method, "route": path},
            )
        )

    # ── detect environment variables ────────────────────────────────────
    if doc.file_type in (".env.example", ".yaml", ".yml", ".toml"):
        seen: set[str] = set()
        for m in _ENV_VAR_RE.finditer(content):
            var = m.group(1)
            if var not in seen and len(var) > 3:
                seen.add(var)
                entities.append(
                    Entity(
                        id=uuid4().hex,
                        project_id=project_id,
                        name=var,
                        type="env_var",
                        path=doc.relative_path,
                        confidence="medium",
                    )
                )

    # ── detect Docker base images ───────────────────────────────────────
    if doc.file_type in ("Dockerfile",) or doc.relative_path.startswith("Dockerfile"):
        for m in _DOCKERFILE_FROM_RE.finditer(content):
            entities.append(
                Entity(
                    id=uuid4().hex,
                    project_id=project_id,
                    name=m.group(1),
                    type="dependency",
                    path=doc.relative_path,
                    confidence="high",
                    metadata={"kind": "docker_base_image"},
                )
            )

    # ── detect packages (Python imports as dependencies) ────────────────
    if doc.language == "python":
        import_re = re.compile(r"^(?:from|import)\s+([\w]+)", re.MULTILINE)
        pkgs: set[str] = set()
        for m in import_re.finditer(content):
            pkg = m.group(1)
            if pkg not in pkgs and pkg not in ("__future__",):
                pkgs.add(pkg)
                entities.append(
                    Entity(
                        id=uuid4().hex,
                        project_id=project_id,
                        name=pkg,
                        type="package",
                        path=doc.relative_path,
                        confidence="medium",
                    )
                )

    return entities


def extract_entities(docs: list[Document], project_id: str) -> list[Entity]:
    """Extract entities from all documents."""
    entities: list[Entity] = []
    for doc in docs:
        entities.extend(extract_entities_from_document(doc, project_id))
    return entities
