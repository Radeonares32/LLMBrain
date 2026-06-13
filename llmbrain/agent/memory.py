"""Task memory integration for LLMBrain agent."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from llmbrain.formats.brainframe import build_brainframe_context
from llmbrain.storage.sqlite import SQLiteStore


class TaskMemoryManager:
    """Manages retrieving context for a task and persisting results to database."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        from llmbrain.core.identity import get_project_storage_dir, load_or_create_project_identity

        identity = load_or_create_project_identity(self.project_root)
        db_path = get_project_storage_dir(identity["project_id"]) / "brain.db"
        self.store = SQLiteStore(db_path)

        # Ensure project exists in SQLite database for Foreign Key constraints
        project_id = self._project_id()
        from llmbrain.models.project import Project

        now = datetime.now(UTC)
        p = Project(
            id=project_id,
            name=self.project_root.name,
            root_path=str(self.project_root),
            created_at=now,
            updated_at=now,
        )
        self.store.upsert_project(p)

    def retrieve_context(
        self, task_query: str, active_task_state: dict[str, Any] | None = None
    ) -> str:
        """Selectively compile task-relevant context.

        Combines:
        - Entity names / facts containing terms from the task query
        - Recency (recently changed files from git)
        - Symbol relationships
        """
        project_id = self._project_id()

        # Gather all project data
        entities = self.store.get_entities(project_id)
        facts = self.store.get_facts(project_id)
        relations = self.store.get_relations(project_id)

        # 1. Filter by keyword matching in query
        query_words = set(task_query.lower().split())

        # Find git changed files (recency & file changes)
        changed_files = self._get_git_changed_files()

        # Score entities
        scored_entities = []
        for entity in entities:
            score = 0
            name = str(entity.get("name", "")).lower()
            path = str(entity.get("path", "")).lower()

            # Word match
            for word in query_words:
                if word in name:
                    score += 10
                if word in path:
                    score += 5

            # Git change relevance
            if any(cf in path for cf in changed_files):
                score += 15

            # Confidence boost
            confidence = str(entity.get("confidence", "")).lower()
            if confidence == "high":
                score += 2
            elif confidence == "medium":
                score += 1

            scored_entities.append((entity, score))

        # Sort entities by score
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        # Keep top 30 relevant entities
        relevant_entities = [item[0] for item in scored_entities if item[1] > 0 or not query_words]
        if not relevant_entities:
            # Fallback to all if no filter matches
            relevant_entities = entities[:30]

        relevant_entity_ids = {str(e.get("id")) for e in relevant_entities}

        # Filter relations involving relevant entities
        relevant_relations = []
        for rel in relations:
            src = str(rel.get("source_entity_id"))
            tgt = str(rel.get("target_entity_id"))
            if src in relevant_entity_ids or tgt in relevant_entity_ids:
                relevant_relations.append(rel)

        # Filter facts linked to relevant entities/files
        relevant_facts = []
        for fact in facts:
            subject = str(fact.get("subject", "")).lower()
            obj = str(fact.get("object", "")).lower()
            claim = str(fact.get("claim", "")).lower()

            is_relevant = False
            for word in query_words:
                if word in subject or word in obj or word in claim:
                    is_relevant = True
                    break

            evidence_paths = [str(ev.get("path", "")).lower() for ev in fact.get("evidence", [])]
            if any(cf in ep for cf in changed_files for ep in evidence_paths):
                is_relevant = True

            if is_relevant or not query_words:
                relevant_facts.append(fact)

        # Fallback if list is too small
        if not relevant_facts:
            relevant_facts = facts[:20]

        # 2. Build final BrainFrame context format
        context_str = build_brainframe_context(
            project_name=self.project_root.name,
            project_id=project_id,
            entities=relevant_entities,
            relations=relevant_relations,
            facts=relevant_facts,
        )

        # Append recent task runs context (prior decisions)
        past_runs = self.store.get_task_runs(project_id)
        if past_runs:
            context_str += "\n# Prior Decisions\n"
            for pr in past_runs[:5]:
                if pr.get("summary"):
                    context_str += (
                        f"- Task: {pr.get('request')} | "
                        f"Summary: {pr.get('summary')} | "
                        f"Status: {pr.get('status')}\n"
                    )

        return context_str

    def persist_task_run(
        self,
        task_id: str,
        request: str,
        summary: str,
        status: str,
        decisions: list[dict[str, str]],
        commands: list[dict[str, Any]],
        failures: list[dict[str, str]],
        commit_hash: str | None = None,
    ) -> None:
        """Persist a completed task's details into the SQLite database."""
        project_id = self._project_id()
        now_str = datetime.now(UTC).isoformat()

        # Get latest git commit if not provided
        if not commit_hash:
            try:
                commit_hash = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.project_root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                commit_hash = None

        # Insert run
        self.store.insert_task_run(
            id=task_id,
            project_id=project_id,
            request=request,
            summary=summary,
            commit_hash=commit_hash,
            status=status,
            created_at=now_str,
            updated_at=now_str,
        )

        # Insert decisions
        for dec in decisions:
            self.store.insert_task_decision(
                id=uuid4().hex,
                task_id=task_id,
                decision=dec["decision"],
                rationale=dec.get("rationale"),
                created_at=now_str,
            )

        # Insert commands
        for cmd in commands:
            self.store.insert_task_command(
                id=uuid4().hex,
                task_id=task_id,
                command=cmd["command"],
                output=cmd.get("output"),
                status=cmd.get("status", "success"),
                created_at=now_str,
            )

        # Insert failures
        for fail in failures:
            self.store.insert_task_failure(
                id=uuid4().hex,
                task_id=task_id,
                failure=fail["failure"],
                resolution=fail.get("resolution"),
                created_at=now_str,
            )

    def _project_id(self) -> str:
        """Derive project ID deterministically from project root path."""
        from llmbrain.core.identity import load_or_create_project_identity

        identity = load_or_create_project_identity(self.project_root)
        return identity["project_id"]

    def _get_git_changed_files(self) -> list[str]:
        """Get relative paths of modified/untracked files from Git."""
        try:
            out = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            files = []
            for line in out.splitlines():
                if len(line) > 3:
                    files.append(line[3:].strip())
            return files
        except Exception:
            return []
