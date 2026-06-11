"""Wiki generator — produces Markdown and MDX wiki pages.

MVP generates pages from entities with attached facts.
Future versions will use structured LLM output.
"""

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


def _build_markdown_body(page: WikiPage, facts: list[Fact]) -> str:
    sections: list[str] = [f"# {page.title}", ""]

    sections.append("## Summary")
    sections.append(f"{page.title} — auto-generated from source analysis.")
    sections.append("")

    if facts:
        sections.append("## Key Facts")
        for f in facts:
            ev_str = ""
            if f.evidence:
                e = f.evidence[0]
                ev_str = f" (`{e.path}:L{e.start_line}-L{e.end_line}`)"
            sections.append(f"- {f.claim}{ev_str}")
        sections.append("")

    if page.sources:
        sections.append("## Source Evidence")
        for s in page.sources:
            sections.append(f"- `{s.path}:L{s.start_line}-L{s.end_line}`")
        sections.append("")

    if page.dependencies:
        sections.append("## Related Pages")
        for d in page.dependencies:
            sections.append(f"- [[{d}]]")
        sections.append("")

    return "\n".join(sections)


def generate_wiki_pages(
    entities: list[Entity],
    facts: list[Fact],
    relations: list[Relation],
    project_id: str,
) -> list[WikiPage]:
    """Generate wiki pages for notable entities."""

    # group facts by subject path
    facts_by_subject: dict[str, list[Fact]] = {}
    for f in facts:
        facts_by_subject.setdefault(f.subject, []).append(f)

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
    notable_types = {"service", "module", "api_endpoint", "database", "queue", "package"}
    seen_slugs: set[str] = set()

    for ent in entities:
        if ent.type in notable_types:
            slug = _slugify(ent.name)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            ent_facts = facts_by_subject.get(ent.path, [])
            sources = [
                WikiSource(path=ent.path, start_line=0, end_line=0)
            ] if ent.path else []

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
    index_body = _build_markdown_body(index_page, [])
    index_page.markdown_content = _build_frontmatter(index_page) + "\n\n" + index_body
    index_page.mdx_content = index_page.markdown_content
    pages.insert(0, index_page)

    return pages
