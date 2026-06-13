"""Multi-repository registry — manages multiple project roots in a single brain."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from llmbrain.core.identity import load_or_create_project_identity

# ── Default registry path ─────────────────────────────────────────────

_DEFAULT_REGISTRY = Path.home() / ".local" / "share" / "llmbrain" / "registry.json"


# ── Model ─────────────────────────────────────────────────────────────


class RepoEntry(BaseModel):
    """A registered repository in the multi-repo registry."""

    project_id: str
    name: str
    root_path: str
    added_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_indexed: datetime | None = None
    tags: list[str] = Field(default_factory=list)


# ── Registry ──────────────────────────────────────────────────────────


class MultiRepoRegistry:
    """JSON file-backed registry for managing multiple project repositories."""

    def __init__(self, registry_path: Path | None = None) -> None:
        self._path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> list[RepoEntry]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [RepoEntry.model_validate(item) for item in data]
        except (json.JSONDecodeError, ValueError):
            return []

    def _save(self, entries: list[RepoEntry]) -> None:
        self._path.write_text(
            json.dumps(
                [e.model_dump(mode="json") for e in entries],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ── CRUD ──────────────────────────────────────────────────────────

    def add(
        self,
        root_path: str,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> RepoEntry:
        """Register a new repository. Raises ValueError if already registered."""
        resolved = str(Path(root_path).expanduser().resolve())
        entries = self._load()

        # Check if already registered by path
        for entry in entries:
            if entry.root_path == resolved:
                raise ValueError(
                    f"Repository already registered: {resolved} "
                    f"(project_id={entry.project_id})"
                )

        identity = load_or_create_project_identity(Path(resolved))
        project_id = identity["project_id"]

        # Check if already registered by project_id
        for entry in entries:
            if entry.project_id == project_id:
                raise ValueError(
                    f"Project already registered with a different path: {entry.root_path}"
                )

        repo_name = name or Path(resolved).name
        new_entry = RepoEntry(
            project_id=project_id,
            name=repo_name,
            root_path=resolved,
            tags=tags or [],
        )
        entries.append(new_entry)
        self._save(entries)
        return new_entry

    def remove(self, project_id: str) -> bool:
        """Remove a registered repository. Returns True if removed."""
        entries = self._load()
        new_entries = [e for e in entries if e.project_id != project_id]
        if len(new_entries) == len(entries):
            return False
        self._save(new_entries)
        return True

    def get(self, project_id: str) -> RepoEntry | None:
        for entry in self._load():
            if entry.project_id == project_id:
                return entry
        return None

    def list_repos(self) -> list[RepoEntry]:
        return self._load()

    def update_last_indexed(self, project_id: str) -> None:
        """Mark a project as freshly indexed."""
        entries = self._load()
        for entry in entries:
            if entry.project_id == project_id:
                entry.last_indexed = datetime.now(UTC)
                break
        self._save(entries)

    def search_by_tag(self, tag: str) -> list[RepoEntry]:
        return [e for e in self._load() if tag in e.tags]

    def find_by_path(self, root_path: str) -> RepoEntry | None:
        resolved = str(Path(root_path).expanduser().resolve())
        for entry in self._load():
            if entry.root_path == resolved:
                return entry
        return None

    def add_tag(self, project_id: str, tag: str) -> bool:
        """Add a tag to a registered repository."""
        entries = self._load()
        for entry in entries:
            if entry.project_id == project_id:
                if tag not in entry.tags:
                    entry.tags.append(tag)
                self._save(entries)
                return True
        return False

    def summary(self) -> dict:
        """Return a summary of the registry."""
        entries = self._load()
        return {
            "total": len(entries),
            "repos": [
                {
                    "project_id": e.project_id,
                    "name": e.name,
                    "root_path": e.root_path,
                    "last_indexed": e.last_indexed.isoformat() if e.last_indexed else None,
                    "tags": e.tags,
                }
                for e in entries
            ],
        }
