"""Wiki page model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WikiSource(BaseModel):
    """A source reference inside a wiki page."""

    path: str
    start_line: int = 0
    end_line: int = 0


class WikiPage(BaseModel):
    """A generated Markdown / MDX wiki page."""

    id: str
    project_id: str
    title: str
    slug: str
    type: str = Field(default="page", description="page | service | module | overview")
    markdown_content: str = Field(default="")
    mdx_content: str = Field(default="")
    confidence: str = Field(default="medium")
    sources: list[WikiSource] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class WikiIndex(BaseModel):
    """Summary returned by the wiki index endpoint."""

    project_id: str
    pages: list[WikiPageSummary] = Field(default_factory=list)


class WikiPageSummary(BaseModel):
    """Lightweight wiki page entry."""

    id: str
    title: str
    slug: str
    type: str


# Fix forward reference
WikiIndex.model_rebuild()
