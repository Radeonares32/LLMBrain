"""Markdown rendering utilities."""

from __future__ import annotations


def render_markdown_page(
    title: str,
    frontmatter: str,
    body_sections: list[tuple[str, str]],
) -> str:
    """Render a full Markdown page with YAML frontmatter.

    Parameters
    ----------
    title:
        Page title (used for the H1 heading).
    frontmatter:
        Already-formatted YAML frontmatter block (without ---  delimiters).
    body_sections:
        List of (heading, content) pairs.
    """

    parts = [f"---\n{frontmatter}\n---", "", f"# {title}", ""]
    for heading, content in body_sections:
        parts.append(f"## {heading}")
        parts.append(content)
        parts.append("")
    return "\n".join(parts)
