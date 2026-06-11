"""Application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(env_prefix="LLMBRAIN_", env_file=".env")

    # ── server ──────────────────────────────────────────────────────────
    app_name: str = "LLM Brain"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── output ──────────────────────────────────────────────────────────
    output_dir_name: str = ".llmbrain"
    db_filename: str = "brain.db"

    # ── scanner limits ──────────────────────────────────────────────────
    max_file_size_bytes: int = 1_048_576  # 1 MB
    max_line_count: int = 10_000

    # ── chunker ─────────────────────────────────────────────────────────
    chunk_max_lines: int = 80
    chunk_overlap_lines: int = 10

    # ── supported extensions ────────────────────────────────────────────
    supported_extensions: list[str] = [
        ".md", ".mdx", ".txt",
        ".py", ".js", ".ts", ".tsx",
        ".go", ".rs", ".java",
        ".yaml", ".yml", ".json", ".toml",
    ]

    # filenames accepted regardless of extension
    supported_filenames: list[str] = [
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env.example",
    ]

    # directories to always skip
    skip_dirs: list[str] = [
        "node_modules", ".git", "dist", "build",
        ".venv", "venv", "__pycache__", ".llmbrain",
        ".tox", ".mypy_cache", ".pytest_cache",
        "vendor", "target",
    ]


settings = Settings()
