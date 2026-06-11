"""Benchmark source-to-memory compression and knowledge density."""

from __future__ import annotations

import json
import math
from pathlib import Path

from llmbrain.services.redactor import redact_text
from llmbrain.services.scanner import scan_project
from llmbrain.storage.filesystem import output_root


def estimate_tokens(text: str) -> int:
    """Estimate LLM tokens without depending on a provider-specific tokenizer."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_tree_text(root: Path, suffixes: tuple[str, ...]) -> str:
    if not root.exists():
        return ""
    parts: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            parts.append(_read_text(path))
    return "\n\n".join(parts)


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _artifact(name: str, content: str, baseline_tokens: int) -> dict:
    tokens = estimate_tokens(content)
    savings = 0.0
    if baseline_tokens:
        savings = round((1 - (tokens / baseline_tokens)) * 100, 2)
    return {
        "name": name,
        "chars": len(content),
        "estimated_tokens": tokens,
        "savings_vs_source_percent": savings,
        "compression_ratio": round((baseline_tokens / tokens), 2) if tokens else None,
    }


def _fact_context(fact_rows: list[dict], limit: int = 5) -> str:
    lines = ["#retrieval_context"]
    for fact in fact_rows[:limit]:
        evidence = ""
        for item in fact.get("evidence") or []:
            path = item.get("path", "")
            start = item.get("start_line", 0)
            end = item.get("end_line", 0)
            if path:
                evidence = f" @{path}:L{start}-L{end}"
                break
        claim = fact.get("claim") or " ".join(
            str(fact.get(key, "")) for key in ("subject", "predicate", "object")
        ).strip()
        if claim:
            lines.append(f"- {claim}{evidence}")
    return "\n".join(lines) + "\n"


def benchmark_project(path: str) -> dict:
    """Benchmark generated LLM Brain artifacts against raw scanned source."""

    root = Path(path).expanduser().resolve()
    project_id = "benchmark"
    docs = scan_project(root, project_id)
    source_text = "\n\n".join(doc.content or "" for doc in docs)
    source_tokens = estimate_tokens(source_text)
    redaction_counts: dict[str, int] = {}
    for doc in docs:
        for kind, count in doc.redactions.items():
            redaction_counts[kind] = redaction_counts.get(kind, 0) + count

    out = output_root(root)
    if not out.exists():
        raise FileNotFoundError(f"No .llmbrain output found at {out}. Run build first.")

    brainframe = _read_text(out / "llm-context" / "brainframe.bf")
    wiki = _read_tree_text(out / "wiki", (".md",))
    graph = _read_text(out / "graph" / "graph.json")
    facts_jsonl = _read_text(out / "facts.jsonl")
    entities_jsonl = _read_text(out / "entities.jsonl")
    relations_jsonl = _read_text(out / "relations.jsonl")
    memory_parts = [brainframe, wiki, graph, facts_jsonl, entities_jsonl, relations_jsonl]
    memory_bundle = "\n\n".join(part for part in memory_parts if part)

    facts = _jsonl_count(out / "facts.jsonl")
    entities = _jsonl_count(out / "entities.jsonl")
    relations = _jsonl_count(out / "relations.jsonl")
    wiki_pages = len(list((out / "wiki").glob("*.md"))) if (out / "wiki").exists() else 0
    fact_rows = _jsonl_rows(out / "facts.jsonl")
    evidenced_facts = sum(1 for fact in fact_rows if fact.get("evidence"))
    retrieval_context = _fact_context(fact_rows)

    required_files = {
        "sqlite": out / "brain.db",
        "documents_jsonl": out / "documents.jsonl",
        "facts_jsonl": out / "facts.jsonl",
        "entities_jsonl": out / "entities.jsonl",
        "relations_jsonl": out / "relations.jsonl",
        "brainframe": out / "llm-context" / "brainframe.bf",
        "graph_json": out / "graph" / "graph.json",
        "wiki_index": out / "wiki" / "index.md",
        "schemas": out / "schemas" / "fact.schema.json",
    }
    goal_checks = {
        "durable_memory_files": all(path.exists() for path in required_files.values()),
        "brainframe_context": bool(brainframe.strip()),
        "wiki_pages": wiki_pages > 0,
        "knowledge_graph": bool(graph.strip()),
        "source_grounded_facts": facts > 0 and evidenced_facts == facts,
        "strict_schemas": (out / "schemas").exists()
        and len(list((out / "schemas").glob("*.schema.json"))) >= 3,
    }

    artifacts = [
        _artifact("retrieval_context_top5", retrieval_context, source_tokens),
        _artifact("brainframe", brainframe, source_tokens),
        _artifact("wiki_markdown", wiki, source_tokens),
        _artifact("graph_json", graph, source_tokens),
        _artifact("memory_bundle", memory_bundle, source_tokens),
    ]
    artifact_texts = {
        "brainframe": brainframe,
        "wiki_markdown": wiki,
        "graph_json": graph,
        "facts_jsonl": facts_jsonl,
        "entities_jsonl": entities_jsonl,
        "relations_jsonl": relations_jsonl,
    }
    artifact_secret_findings: dict[str, int] = {}
    for name, content in artifact_texts.items():
        finding_count = redact_text(content).report.total
        if finding_count:
            artifact_secret_findings[name] = finding_count
    goal_checks["secrets_redacted"] = not artifact_secret_findings

    knowledge_items = facts + entities + relations
    return {
        "project": root.name,
        "documents": len(docs),
        "source": {
            "chars": len(source_text),
            "estimated_tokens": source_tokens,
        },
        "artifacts": artifacts,
        "knowledge": {
            "facts": facts,
            "facts_with_evidence": evidenced_facts,
            "entities": entities,
            "relations": relations,
            "wiki_pages": wiki_pages,
            "items_total": knowledge_items,
            "items_per_1k_source_tokens": (
                round((knowledge_items / source_tokens) * 1000, 2) if source_tokens else 0
            ),
        },
        "redaction": {
            "documents_with_redactions": sum(1 for doc in docs if doc.redactions),
            "total_redactions": sum(redaction_counts.values()),
            "redaction_counts": redaction_counts,
            "artifact_secret_findings": artifact_secret_findings,
            "artifacts_clean": not artifact_secret_findings,
        },
        "goal_checks": goal_checks,
        "brain_readiness_score": round(
            sum(1 for passed in goal_checks.values() if passed) / len(goal_checks) * 100
        ),
    }
