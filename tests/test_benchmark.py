import json

from llmbrain.services.benchmark import benchmark_project


def test_benchmark_reports_memory_and_brain_readiness(tmp_path):
    (tmp_path / "app.py").write_text("def login(user):\n    return user\n", encoding="utf-8")

    out = tmp_path / ".llmbrain"
    (out / "llm-context").mkdir(parents=True)
    (out / "graph").mkdir()
    (out / "wiki").mkdir()
    (out / "schemas").mkdir()
    (out / "brain.db").write_text("", encoding="utf-8")
    (out / "documents.jsonl").write_text("{}", encoding="utf-8")
    (out / "chunks.jsonl").write_text("{}", encoding="utf-8")
    (out / "facts.jsonl").write_text(
        json.dumps(
            {
                "claim": "app.py defines login",
                "evidence": [{"path": "app.py", "start_line": 1, "end_line": 2}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "entities.jsonl").write_text("{}\n", encoding="utf-8")
    (out / "relations.jsonl").write_text("{}\n", encoding="utf-8")
    (out / "llm-context" / "brainframe.bf").write_text("#facts\nlogin\n", encoding="utf-8")
    (out / "graph" / "graph.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    (out / "wiki" / "index.md").write_text("# Project Overview\n", encoding="utf-8")
    (out / "schemas" / "fact.schema.json").write_text("{}", encoding="utf-8")
    (out / "schemas" / "entity.schema.json").write_text("{}", encoding="utf-8")
    (out / "schemas" / "relation.schema.json").write_text("{}", encoding="utf-8")

    result = benchmark_project(str(tmp_path))

    assert result["knowledge"]["facts_with_evidence"] == 1
    assert result["artifacts"][0]["name"] == "retrieval_context_top5"
    assert result["redaction"]["artifacts_clean"] is True
    assert result["goal_checks"]["durable_memory_files"] is True
    assert result["goal_checks"]["source_grounded_facts"] is True
    assert result["goal_checks"]["secrets_redacted"] is True
    assert result["brain_readiness_score"] == 100
