"""Evidence health scoring for generated knowledge artifacts."""

from __future__ import annotations

from llmbrain.models.fact import Fact
from llmbrain.models.wiki import WikiPage


def _rating(score: int) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 50:
        return "fair"
    if score >= 25:
        return "weak"
    return "poor"


def _has_valid_fact_evidence(fact: Fact) -> bool:
    return any(
        evidence.path
        and evidence.start_line > 0
        and evidence.end_line >= evidence.start_line
        for evidence in fact.evidence
    )


def _has_wiki_source(page: WikiPage) -> bool:
    if page.type == "overview":
        return True
    return any(source.path for source in page.sources)


def calculate_evidence_health(facts: list[Fact], pages: list[WikiPage]) -> dict:
    """Calculate a compact evidence-health report.

    Facts are weighted more heavily because they are the durable claims that
    downstream wiki, graph, and BrainFrame artifacts depend on.
    """

    evidenced_facts = sum(1 for fact in facts if _has_valid_fact_evidence(fact))
    sourced_pages = sum(1 for page in pages if _has_wiki_source(page))

    fact_score = (evidenced_facts / len(facts) * 100) if facts else 100.0
    wiki_score = (sourced_pages / len(pages) * 100) if pages else 100.0
    score = round((fact_score * 0.75) + (wiki_score * 0.25))

    return {
        "score": score,
        "rating": _rating(score),
        "facts": {
            "total": len(facts),
            "with_evidence": evidenced_facts,
            "coverage": round(fact_score, 2),
        },
        "wiki_pages": {
            "total": len(pages),
            "with_sources": sourced_pages,
            "coverage": round(wiki_score, 2),
        },
    }
