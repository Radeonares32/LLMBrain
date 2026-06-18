"""Deterministic tests for Phase 5 - Persistent Project Brain, Memory lifecycle and TUI."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from llmbrain.tui import LLMBrainTUI
from llmbrain.core.identity import (
    load_or_create_project_identity,
)
from llmbrain.services.session_service import SessionService
from llmbrain.storage.cache import BrainCache
from llmbrain.storage.sqlite import (
    backup_project_db,
    restore_project_db,
)


@pytest.fixture
def temp_env(monkeysession=None):
    """Fixture to redirect LLM data directories during tests."""
    temp_dir = tempfile.mkdtemp()
    os.environ["LLMBRAIN_DATA_DIR"] = temp_dir
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ── 1. Project Identity & Isolation Tests ─────────────────────────────


def test_project_identity_isolation(temp_env):
    # Distinct folders with same name must get distinct project IDs
    root_a = temp_env / "project_a"
    root_b = temp_env / "project_b"
    root_a.mkdir()
    root_b.mkdir()

    id_a = load_or_create_project_identity(root_a)
    id_b = load_or_create_project_identity(root_b)

    assert id_a["project_id"] != id_b["project_id"]
    assert id_a["name"] == "project_a"
    assert id_b["name"] == "project_b"


def test_project_copy_vs_move(temp_env):
    root_a = temp_env / "original"
    root_a.mkdir()

    id_a = load_or_create_project_identity(root_a)
    proj_id_a = id_a["project_id"]

    # Scenario: Copy repository (original still exists)
    root_copy = temp_env / "copy"
    shutil.copytree(root_a, root_copy)

    id_copy = load_or_create_project_identity(root_copy)
    # Copied repository should be detected and get a new project ID for isolation
    assert id_copy["project_id"] != proj_id_a

    # Scenario: Move repository (original is deleted)
    root_moved = temp_env / "moved"
    shutil.move(str(root_a), str(root_moved))

    id_moved = load_or_create_project_identity(root_moved)
    # Moved repository should retain the original project ID
    assert id_moved["project_id"] == proj_id_a


# ── 2. Cache Tests ──────────────────────────────────────────────────


def test_cache_hit_miss_eviction():
    cache = BrainCache(max_items=3, max_bytes=1000, ttl_seconds=10)

    cache.set("proj1", "key1", "value1")
    cache.set("proj1", "key2", "value2")
    cache.set("proj1", "key3", "value3")

    # Hits & Misses
    assert cache.get("proj1", "key1") == "value1"
    assert cache.get("proj1", "key4") is None

    # Eviction due to item count
    cache.set("proj1", "key4", "value4")
    # key2 should be evicted (LRU ordering: key1 was accessed, key2 is oldest)
    assert cache.get("proj1", "key2") is None
    assert cache.get("proj1", "key3") == "value3"


def test_cache_byte_size_eviction():
    # Set small byte limit
    cache = BrainCache(max_items=10, max_bytes=20, ttl_seconds=10)

    cache.set("proj1", "key1", "a" * 8)
    cache.set("proj1", "key2", "b" * 8)

    stats = cache.stats()
    assert stats.current_bytes <= 20

    # Adding key3 (size 8) should evict key1
    cache.set("proj1", "key3", "c" * 8)
    assert cache.get("proj1", "key1") is None
    assert cache.get("proj1", "key2") == "b" * 8
    assert cache.get("proj1", "key3") == "c" * 8


def test_cache_ttl_and_invalidation():
    cache = BrainCache(max_items=5, max_bytes=1000, ttl_seconds=0.1)
    cache.set("proj1", "key1", "val1")

    time.sleep(0.15)
    assert cache.get("proj1", "key1") is None

    cache.set("proj1", "key2", "val2", source_version="v1")
    # Version mismatch invalidation
    assert cache.get("proj1", "key2", current_source_version="v2") is None


# ── 3. Session & Compaction Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_session_management_and_compaction(temp_env):
    root = temp_env / "my_project"
    root.mkdir()

    service = SessionService(root)
    session = service.create_session(
        "Auth Feature", "build", {"model": "openai"}, "ask-before-write"
    )
    session_id = session["id"]

    service.add_message(session_id, "user", "Add endpoint /login")
    service.add_message(session_id, "assistant", "Sure, I will add it.")

    messages = service.get_messages(session_id)
    assert len(messages) == 2
    assert messages[0]["content"] == "Add endpoint /login"

    # Run Compaction
    comp_state = await service.compact_session(session_id)
    assert "conversation_summary" in comp_state

    # Check updated session state
    sess = service.get_session(session_id)
    assert sess["compaction_state"]["conversation_summary"] == "Compacted conversation history"


# ── 4. Workspace Lock Tests ──────────────────────────────────────────


def test_workspace_locking(temp_env):
    root = temp_env / "locked_project"
    root.mkdir()

    service = SessionService(root)
    lock = service.lock

    # Acquire
    assert lock.acquire("task_1", "sess_1") is True
    # Duplicate acquisition fails
    assert lock.acquire("task_2", "sess_1", timeout_seconds=0.1) is False

    # Release
    lock.release("task_1")
    # Re-acquire works
    assert lock.acquire("task_2", "sess_1") is True


# ── 5. Backup & Restore Tests ─────────────────────────────────────────


def test_db_backup_and_restore(temp_env):
    root = temp_env / "db_project"
    root.mkdir()

    service = SessionService(root)
    proj_id = service.project_id

    # Add dummy session data
    service.create_session("Pre-Backup", "ask", {}, "read-only")

    backup_file = temp_env / "backup.zip"
    backup_project_db(proj_id, backup_file)
    assert backup_file.exists()

    # Modify DB (add another session)
    service.create_session("Post-Backup", "ask", {}, "read-only")
    assert len(service.list_sessions()) == 2

    # Restore DB
    restore_project_db(proj_id, backup_file)

    # Reset connection / check restored state
    restored_service = SessionService(root)
    sessions = restored_service.list_sessions()
    # Should only have 1 session (Pre-Backup)
    assert len(sessions) == 1
    assert sessions[0]["title"] == "Pre-Backup"


# ── 6. TUI State & Command Tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_tui_render_and_slash_commands(temp_env):
    root = temp_env / "tui_project"
    root.mkdir()

    # Initialize TUI
    tui = LLMBrainTUI(root, provider_name="openai")
    tui.state.sessions = tui.session_service.list_sessions()
    if not tui.state.sessions:
        sess = tui.session_service.create_session(
            "Default Session", "build", {}, "ask-before-write"
        )
        tui.state.sessions = [sess]
    tui.state.selected_session_id = tui.state.sessions[0]["id"]

    # Simulate typing commands
    await tui._handle_input("/")
    await tui._handle_input("h")
    await tui._handle_input("e")
    await tui._handle_input("l")
    await tui._handle_input("p")
    await tui._handle_input("\n")

    # Should navigate to help panel
    assert tui.state.focused_panel == "help"

    # ESC to focus back to conversation
    await tui._handle_input("\x1b")
    assert tui.state.focused_panel == "conversation"

    # Run exit slash command
    await tui._handle_input("/")
    await tui._handle_input("e")
    await tui._handle_input("x")
    await tui._handle_input("i")
    await tui._handle_input("t")
    await tui._handle_input("\n")
    assert tui.state.is_running is False
