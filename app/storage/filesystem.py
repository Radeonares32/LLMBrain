"""Filesystem helpers — manages the .llmbrain output directory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings


def output_root(project_root: str | Path) -> Path:
    """Return the resolved .llmbrain directory for a project."""
    return Path(project_root).resolve() / settings.output_dir_name


def ensure_output_dirs(project_root: str | Path) -> dict[str, Path]:
    """Create the full output directory tree and return a dict of key paths."""

    root = output_root(project_root)
    dirs = {
        "root": root,
        "llm_context": root / "llm-context",
        "schemas": root / "schemas",
        "graph": root / "graph",
        "wiki": root / "wiki",
        "wiki_mdx": root / "wiki-mdx",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_manifest(project_root: str | Path, manifest: dict) -> Path:
    """Write manifest.json to the output root."""

    root = output_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    manifest.setdefault("generated_at", datetime.now(UTC).isoformat())
    manifest.setdefault("version", settings.app_version)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def write_text_file(directory: Path, filename: str, content: str) -> Path:
    """Write a text file inside *directory*."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path
