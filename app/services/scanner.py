"""File-system scanner — discovers supported files in a project tree."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.models.document import Document

# Map extensions to language names
_EXT_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".md": "markdown",
    ".mdx": "mdx",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}


def _content_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8", errors="replace")).hexdigest()


def _detect_language(path: Path) -> str:
    return _EXT_LANGUAGE.get(path.suffix.lower(), "unknown")


def _is_binary(filepath: Path, sample_size: int = 8192) -> bool:
    """Quick heuristic: if the first N bytes contain a null byte it's binary."""
    try:
        with open(filepath, "rb") as fh:
            chunk = fh.read(sample_size)
        return b"\x00" in chunk
    except OSError:
        return True


def _should_skip_dir(name: str) -> bool:
    return name in settings.skip_dirs or name.startswith(".")


def _is_supported(path: Path) -> bool:
    if path.name in settings.supported_filenames:
        return True
    # .env.example has compound suffix
    if path.name.endswith(".env.example"):
        return True
    return path.suffix.lower() in settings.supported_extensions


def scan_project(root: str | Path, project_id: str) -> list[Document]:
    """Walk *root* and return a `Document` for every supported file."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project root not found: {root}")

    documents: list[Document] = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # prune skipped directories in-place
        dirnames[:] = [
            d for d in dirnames
            if not _should_skip_dir(d)
        ]

        for fname in filenames:
            filepath = Path(dirpath) / fname

            if not _is_supported(filepath):
                continue

            # size guard
            try:
                stat = filepath.stat()
            except OSError:
                continue
            if stat.st_size > settings.max_file_size_bytes:
                continue
            if stat.st_size == 0:
                continue

            # binary guard
            if _is_binary(filepath):
                continue

            # read content
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            line_count = content.count("\n") + 1
            if line_count > settings.max_line_count:
                continue

            relative = str(filepath.relative_to(root))
            ext = filepath.suffix.lower() if filepath.suffix else filepath.name

            documents.append(
                Document(
                    id=uuid4().hex,
                    project_id=project_id,
                    path=str(filepath),
                    relative_path=relative,
                    content_hash=_content_hash(content),
                    file_type=ext,
                    language=_detect_language(filepath),
                    line_count=line_count,
                    size_bytes=stat.st_size,
                    content=content,
                )
            )

    # deterministic order
    documents.sort(key=lambda d: d.relative_path)
    return documents
