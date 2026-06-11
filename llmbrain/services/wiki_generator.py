"""Wiki generator — produces source-grounded Markdown and MDX wiki pages."""

from __future__ import annotations

import re
from uuid import uuid4

from llmbrain.models.entity import Entity
from llmbrain.models.fact import Fact
from llmbrain.models.relation import Relation
from llmbrain.models.wiki import WikiPage, WikiSource


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _build_frontmatter(page: WikiPage) -> str:
    lines = [
        "---",
        f"id: {page.slug}",
        f"type: {page.type}",
        f"confidence: {page.confidence}",
        "sources:",
    ]
    for s in page.sources:
        lines.append(f"  - path: {s.path}")
        if s.start_line:
            lines.append(f"    start_line: {s.start_line}")
        if s.end_line:
            lines.append(f"    end_line: {s.end_line}")
    if page.dependencies:
        lines.append("dependencies:")
        for d in page.dependencies:
            lines.append(f"  - {d}")
    if page.tags:
        lines.append("tags:")
        for t in page.tags:
            lines.append(f"  - {t}")
    lines.append("---")
    return "\n".join(lines)


def _evidence_label(fact: Fact) -> str:
    if not fact.evidence:
        return ""
    evidence = fact.evidence[0]
    return f" (`{evidence.path}:L{evidence.start_line}-L{evidence.end_line}`)"


def _summary_sentence(page: WikiPage, facts: list[Fact]) -> str:
    location = f" in `{page.sources[0].path}`" if page.sources else ""
    if facts:
        return (
            f"{page.title} is tracked as a `{page.type}`{location}. "
            f"The memory contains {len(facts)} source-grounded facts about it."
        )
    return f"{page.title} is tracked as a `{page.type}`{location}."


def _build_markdown_body(page: WikiPage, facts: list[Fact]) -> str:
    sections: list[str] = [f"# {page.title}", ""]

    sections.append("## Summary")
    sections.append(_summary_sentence(page, facts))
    sections.append("")

    if facts:
        sections.append("## What It Knows")
        for f in facts:
            sections.append(f"- {f.claim}{_evidence_label(f)}")
        sections.append("")

    if page.sources:
        sections.append("## Source Evidence")
        for s in page.sources:
            sections.append(f"- `{s.path}:L{s.start_line}-L{s.end_line}`")
        sections.append("")

    if page.dependencies:
        sections.append("## Related Memory")
        for d in page.dependencies:
            sections.append(f"- [[{d}]]")
        sections.append("")

    return "\n".join(sections)


def _fact_matches_entity(entity: Entity, fact: Fact) -> bool:
    name = entity.name.lower()
    path = entity.path.lower()
    subject = fact.subject.lower()
    obj = fact.object.lower()
    claim = fact.claim.lower()
    evidence_paths = {e.path.lower() for e in fact.evidence}
    return (
        subject in {name, path}
        or obj == name
        or name in claim
        or bool(path and path in evidence_paths)
    )


def _sources_from_facts(facts: list[Fact]) -> list[WikiSource]:
    seen: set[tuple[str, int, int]] = set()
    sources: list[WikiSource] = []
    for fact in facts:
        for evidence in fact.evidence:
            key = (evidence.path, evidence.start_line, evidence.end_line)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                WikiSource(
                    path=evidence.path,
                    start_line=evidence.start_line,
                    end_line=evidence.end_line,
                )
            )
    return sources


def _build_overview_body(
    page: WikiPage,
    facts: list[Fact],
    entities: list[Entity],
    relations: list[Relation],
) -> str:
    sections = ["# Project Overview", ""]
    sections.append("## Summary")
    sections.append(
        "This repository has been compiled into durable engineering memory: "
        f"{len(facts)} facts, {len(entities)} entities, {len(relations)} relations, "
        f"and {len(page.dependencies)} wiki pages."
    )
    sections.append("")

    if facts:
        sections.append("## Representative Facts")
        for fact in facts[:8]:
            sections.append(f"- {fact.claim}{_evidence_label(fact)}")
        sections.append("")

    if page.dependencies:
        sections.append("## Wiki Index")
        for dependency in page.dependencies:
            sections.append(f"- [[{dependency}]]")
        sections.append("")

    return "\n".join(sections)


def generate_wiki_pages(
    entities: list[Entity],
    facts: list[Fact],
    relations: list[Relation],
    project_id: str,
) -> list[WikiPage]:
    """Generate wiki pages for notable entities."""

    # entity index
    entity_index = {e.id: e for e in entities}

    # relation targets per entity
    deps_by_entity: dict[str, list[str]] = {}
    for r in relations:
        tgt = entity_index.get(r.target_entity_id)
        if tgt:
            deps_by_entity.setdefault(r.source_entity_id, []).append(tgt.name)

    pages: list[WikiPage] = []

    # create a page for each non-file entity (services, endpoints, etc.)
    notable_types = {
        "service",
        "module",
        "api_endpoint",
        "database",
        "queue",
        "package",
        "env_var",
        "config",
        "class",
        "function",
        "format",
        "artifact",
        "schema",
        "command",
        "provider",
        "workflow",
    }
    seen_slugs: set[str] = set()

    for ent in entities:
        if ent.type in notable_types:
            slug = _slugify(ent.name)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            ent_facts = [fact for fact in facts if _fact_matches_entity(ent, fact)]
            sources = _sources_from_facts(ent_facts)

            page = WikiPage(
                id=uuid4().hex,
                project_id=project_id,
                title=ent.name,
                slug=slug,
                type=ent.type,
                confidence=ent.confidence,
                sources=sources,
                dependencies=deps_by_entity.get(ent.id, []),
                tags=[ent.type],
            )

            frontmatter = _build_frontmatter(page)
            body = _build_markdown_body(page, ent_facts)
            page.markdown_content = frontmatter + "\n\n" + body

            # simple MDX variant
            page.mdx_content = page.markdown_content  # MVP: identical

            pages.append(page)

    # ── project overview / index page ───────────────────────────────────
    index_page = WikiPage(
        id=uuid4().hex,
        project_id=project_id,
        title="Project Overview",
        slug="index",
        type="overview",
        confidence="high",
        sources=[],
        dependencies=[p.title for p in pages],
        tags=["overview"],
    )
    index_body = _build_overview_body(index_page, facts, entities, relations)
    index_page.markdown_content = _build_frontmatter(index_page) + "\n\n" + index_body
    index_page.mdx_content = index_page.markdown_content
    pages.insert(0, index_page)

    return pages
