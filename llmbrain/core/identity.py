"""Project identity and storage layout helper."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def get_user_data_dir() -> Path:
    """Return platform-appropriate application data directory."""
    env_override = os.environ.get("LLMBRAIN_DATA_DIR")
    if env_override:
        return Path(env_override)

    home = Path.home()
    if sys.platform == "win32":
        appdata = os.environ.get("LOCALAPPDATA")
        if appdata:
            return Path(appdata) / "llmbrain"
        return home / "AppData" / "Local" / "llmbrain"
    elif sys.platform == "darwin":
        return home / "Library" / "Application Support" / "llmbrain"
    else:
        xdg_data = os.environ.get("XDG_DATA_HOME")
        if xdg_data:
            return Path(xdg_data) / "llmbrain"
        return home / ".local" / "share" / "llmbrain"


def find_repository_root(start_path: Path | str = ".") -> Path:
    """Traverse up to find repository root containing .git or .llmbrain."""
    current = Path(start_path).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".llmbrain").exists():
            return parent
    return current


def calculate_fingerprint(root_path: Path) -> str:
    """Calculate a stable fingerprint for a repository (initial git commit or absolute path)."""
    try:
        res = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=str(root_path),
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip().split("\n")[0]
        if commit:
            return f"git:{commit}"
    except Exception:
        pass
    return "path:" + hashlib.sha256(str(root_path.resolve()).encode()).hexdigest()[:32]


def load_or_create_project_identity(project_root: Path) -> dict:
    """Load or create project identity file inside .llmbrain/project.json."""
    project_root = Path(project_root).resolve()
    dot_llmbrain = project_root / ".llmbrain"
    dot_llmbrain.mkdir(parents=True, exist_ok=True)

    # Generate .gitignore recommendations
    gitignore = dot_llmbrain / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "# LLMBrain local storage/caches\n"
            "state.json\n"
            "*.db\n"
            "cache/\n"
            "locks/\n"
            "indexes/\n"
            "snapshots/\n"
            "logs/\n",
            encoding="utf-8",
        )

    project_json_path = dot_llmbrain / "project.json"
    fingerprint = calculate_fingerprint(project_root)
    actual_path = str(project_root)

    if project_json_path.exists():
        try:
            data = json.loads(project_json_path.read_text(encoding="utf-8"))
            stored_path = data.get("repository_root")

            # Check for copy vs move
            is_copy = False
            if stored_path and stored_path != actual_path:
                if Path(stored_path).exists():
                    is_copy = True

            if is_copy:
                # Copied repo detected: generate a new project ID for isolation
                import uuid

                data["project_id"] = f"prj_{uuid.uuid4().hex[:12]}"
                data["created_at"] = datetime.now(UTC).isoformat()

            data["repository_root"] = actual_path
            data["repository_root_fingerprint"] = fingerprint
            data["last_opened_at"] = datetime.now(UTC).isoformat()
            project_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data
        except Exception:
            pass

    # New project identity
    import uuid

    project_id = f"prj_{uuid.uuid4().hex[:12]}"
    data = {
        "schema_version": 1,
        "project_id": project_id,
        "name": project_root.name,
        "created_at": datetime.now(UTC).isoformat(),
        "repository_root": actual_path,
        "repository_root_fingerprint": fingerprint,
        "brain_location": "local",
        "last_opened_at": datetime.now(UTC).isoformat(),
    }
    project_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def get_project_storage_dir(project_id: str) -> Path:
    """Return the data directory for the given project_id in the user data dir."""
    p_dir = get_user_data_dir() / "projects" / project_id
    p_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["locks", "indexes", "snapshots", "logs"]:
        (p_dir / sub).mkdir(exist_ok=True)
    return p_dir
