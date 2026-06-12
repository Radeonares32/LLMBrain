from llmbrain.formats.brainframe import (
    build_brainframe_context,
    build_compact_context,
    escape_cell,
    format_evidence,
)


def test_brainframe_basic():
    bf = build_compact_context(project_name="test_project", entities=[], relations=[], facts=[])
    assert "@project test_project" in bf


def test_brainframe_escapes_cells_and_writes_empty_headers():
    bf = build_brainframe_context("test|project", "p1", [], [], [])

    assert "@project test\\|project" in bf
    assert "#entities\nid | type | name | path | confidence" in bf
    assert "#relations\nfrom | relation | to | evidence | confidence" in bf
    assert "#facts\nid | subject | predicate | object | evidence | confidence" in bf
    assert escape_cell("a|b\nc   d") == "a\\|b c d"
    assert format_evidence("app.py", 1, 3) == "app.py:L1-L3"


def test_brainframe_truncates_by_priority():
    facts = [
        {
            "id": "low123456",
            "subject": "low",
            "predicate": "documents",
            "object": "x" * 400,
            "confidence": "low",
            "evidence": [],
        },
        {
            "id": "high123456",
            "subject": "high",
            "predicate": "defines",
            "object": "service",
            "confidence": "high",
            "evidence": [{"path": "app.py", "start_line": 1, "end_line": 2}],
        },
    ]

    bf = build_brainframe_context("p", "pid", [], [], facts, max_chars=380)

    assert "@truncated true" in bf
    assert "high123" in bf
    assert "low123" not in bf
