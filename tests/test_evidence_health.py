"""Tests for evidence_health.py."""

from llmbrain.models.fact import Fact, FactEvidence
from llmbrain.models.wiki import WikiPage, WikiSource
from llmbrain.services.evidence_health import calculate_evidence_health, _rating


def test_rating_scale():
    assert _rating(95) == "excellent"
    assert _rating(90) == "excellent"
    assert _rating(80) == "good"
    assert _rating(50) == "fair"
    assert _rating(30) == "weak"
    assert _rating(10) == "poor"

def test_calculate_evidence_health_empty():
    res = calculate_evidence_health([], [])
    assert res["score"] == 100
    assert res["rating"] == "excellent"

def test_calculate_evidence_health_all_valid():
    # 1 fact with evidence
    f = Fact(
        id="f1", project_id="p1", subject="s", predicate="p", object="o", claim="c",
        evidence=[FactEvidence(id="e1", fact_id="f1", document_id="d1", path="test.py", start_line=1, end_line=5)]
    )
    # 1 page with source
    w = WikiPage(
        id="w1", project_id="p1", title="t", slug="s", type="page",
        sources=[WikiSource(path="test.py")]
    )
    res = calculate_evidence_health([f], [w])
    assert res["score"] == 100
    assert res["rating"] == "excellent"

def test_calculate_evidence_health_no_evidence():
    f = Fact(
        id="f1", project_id="p1", subject="s", predicate="p", object="o", claim="c",
        evidence=[]
    )
    w = WikiPage(
        id="w1", project_id="p1", title="t", slug="s", type="page",
        sources=[]
    )
    res = calculate_evidence_health([f], [w])
    assert res["score"] == 0
    assert res["rating"] == "poor"
    assert res["facts"]["with_evidence"] == 0
    assert res["wiki_pages"]["with_sources"] == 0

def test_calculate_evidence_health_partial():
    # 2 facts, 1 with evidence
    f1 = Fact(id="f1", project_id="p1", subject="s", predicate="p", object="o", claim="c", evidence=[FactEvidence(id="e1", fact_id="f1", document_id="d1", path="test.py", start_line=1, end_line=5)])
    f2 = Fact(id="f2", project_id="p1", subject="s", predicate="p", object="o", claim="c", evidence=[])
    
    # 4 pages, 1 with source, 1 is overview
    w1 = WikiPage(id="w1", project_id="p1", title="t", slug="s", type="page", sources=[WikiSource(path="test.py")])
    w2 = WikiPage(id="w2", project_id="p1", title="t", slug="s", type="page", sources=[])
    w3 = WikiPage(id="w3", project_id="p1", title="t", slug="s", type="page", sources=[])
    w4 = WikiPage(id="w4", project_id="p1", title="t", slug="s", type="overview", sources=[]) # Overview is always valid

    res = calculate_evidence_health([f1, f2], [w1, w2, w3, w4])
    # facts score: 1/2 = 50%
    # wiki score: 2/4 = 50%
    # total score: 50 * 0.75 + 50 * 0.25 = 50
    assert res["score"] == 50
    assert res["rating"] == "fair"
    assert res["facts"]["with_evidence"] == 1
    assert res["facts"]["coverage"] == 50.0
    assert res["wiki_pages"]["with_sources"] == 2
    assert res["wiki_pages"]["coverage"] == 50.0
