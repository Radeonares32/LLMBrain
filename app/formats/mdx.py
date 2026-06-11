"""MDX rendering utilities."""

from __future__ import annotations


def markdown_to_mdx(markdown: str) -> str:
    """Convert a Markdown page to a basic MDX variant.

    In the MVP the conversion is identity — MDX is a superset of Markdown.
    Future versions can inject React components, import statements, etc.
    """
    return markdown
