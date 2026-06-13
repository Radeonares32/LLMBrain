"""Semantic search service — indexes project memory and retrieves by embedding similarity."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llmbrain.services.embeddings import TfIdfEmbedder, default_embedder
from llmbrain.storage.sqlite import SQLiteStore
from llmbrain.storage.vector_store import VectorStore

# ── Result model ──────────────────────────────────────────────────────


class SearchResult(BaseModel):
    """Single semantic search result."""

    source_type: str
    source_id: str
    score: float
    text_preview: str
    metadata: dict = Field(default_factory=dict)


# ── Service ───────────────────────────────────────────────────────────


class SemanticSearchService:
    """Indexes SQLiteStore memory into a VectorStore and runs semantic queries."""

    def __init__(
        self,
        project_id: str,
        store: SQLiteStore,
        vector_store: VectorStore,
        embedder: TfIdfEmbedder | None = None,
    ) -> None:
        self.project_id = project_id
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder or default_embedder

    # ── indexing ──────────────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> int:
        """Embed and store code/text chunks."""
        if not chunks:
            return 0
        texts = [c.get("content", "") for c in chunks]
        if not self.embedder.is_fitted:
            self.embedder.fit(texts)
        count = 0
        for chunk in chunks:
            text = chunk.get("content", "")
            if not text.strip():
                continue
            vec = self.embedder.embed(text)
            preview = text[:200].replace("\n", " ")
            self.vector_store.upsert(
                self.project_id, "chunk", chunk["id"], preview, vec
            )
            count += 1
        return count

    def index_facts(self, facts: list[dict]) -> int:
        """Embed and store extracted facts."""
        if not facts:
            return 0
        texts = [
            f"{f.get('subject', '')} {f.get('predicate', '')} "
            f"{f.get('object', '')} {f.get('claim', '')}"
            for f in facts
        ]
        if not self.embedder.is_fitted:
            self.embedder.fit(texts)
        count = 0
        for fact, text in zip(facts, texts):
            if not text.strip():
                continue
            vec = self.embedder.embed(text)
            self.vector_store.upsert(
                self.project_id, "fact", fact["id"], text[:200], vec
            )
            count += 1
        return count

    def index_entities(self, entities: list[dict]) -> int:
        """Embed and store extracted entities."""
        if not entities:
            return 0
        texts = [
            f"{e.get('name', '')} {e.get('kind', '')} {e.get('path', '')}"
            for e in entities
        ]
        if not self.embedder.is_fitted:
            self.embedder.fit(texts)
        count = 0
        for entity, text in zip(entities, texts):
            if not text.strip():
                continue
            vec = self.embedder.embed(text)
            self.vector_store.upsert(
                self.project_id, "entity", entity["id"], text[:200], vec
            )
            count += 1
        return count

    def index_all(self) -> dict[str, int]:
        """Fit embedder on all text, then index chunks, facts and entities."""
        chunks = self.store.get_chunks(self.project_id) or []
        facts = self.store.get_facts(self.project_id) or []
        entities = self.store.get_entities(self.project_id) or []

        # Collect all text for fitting
        all_texts: list[str] = []
        all_texts.extend(c.get("content", "") for c in chunks)
        all_texts.extend(
            f"{f.get('subject', '')} {f.get('predicate', '')} "
            f"{f.get('object', '')} {f.get('claim', '')}"
            for f in facts
        )
        all_texts.extend(
            f"{e.get('name', '')} {e.get('kind', '')} {e.get('path', '')}"
            for e in entities
        )
        corpus = [t for t in all_texts if t.strip()]
        if corpus:
            self.embedder.fit(corpus)

        return {
            "chunks": self.index_chunks(chunks),
            "facts": self.index_facts(facts),
            "entities": self.index_entities(entities),
        }

    # ── search ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        source_types: list[str] | None = None,
        k: int = 10,
        threshold: float = 0.25,
    ) -> list[SearchResult]:
        """Embed query, search each source type, merge and return top-k results."""
        if not self.embedder.is_fitted:
            # Nothing indexed yet
            return []

        query_vec = self.embedder.embed(query)
        types = source_types or ["chunk", "fact", "entity"]
        all_results: list[SearchResult] = []

        for stype in types:
            hits = self.vector_store.search(
                self.project_id,
                query_vec,
                source_type=stype,
                k=k,
                threshold=threshold,
            )
            for rec, score in hits:
                all_results.append(
                    SearchResult(
                        source_type=stype,
                        source_id=rec.source_id,
                        score=round(score, 4),
                        text_preview=rec.text_preview,
                    )
                )

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:k]

    def get_stats(self) -> dict:
        """Return indexing stats from the vector store."""
        vs_stats = self.vector_store.stats(self.project_id)
        return {
            "project_id": self.project_id,
            "vector_store": vs_stats,
            "embedder_fitted": self.embedder.is_fitted,
            "vocabulary_size": self.embedder.vocabulary_size,
        }


# ── Convenience factory ───────────────────────────────────────────────


def create_search_service(project_id: str, storage_dir) -> SemanticSearchService:
    """Create a SemanticSearchService for a project from its storage directory."""
    from pathlib import Path

    storage_dir = Path(storage_dir)
    store = SQLiteStore(storage_dir / "brain.db")
    vs = VectorStore(storage_dir / "vectors.db")
    return SemanticSearchService(project_id, store, vs)
