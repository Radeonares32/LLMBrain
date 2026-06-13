"""Semantic embedding service — TF-IDF based pure-Python vector representations."""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel

# Type alias
EmbeddingVector = list[float]

# Stopwords to filter out
_STOPWORDS = frozenset(
    {
        "the",
        "is",
        "in",
        "it",
        "of",
        "and",
        "or",
        "to",
        "a",
        "an",
        "for",
        "on",
        "with",
        "as",
        "at",
        "be",
        "this",
        "that",
        "are",
        "was",
        "were",
        "by",
        "from",
        "not",
        "but",
        "if",
        "so",
        "do",
        "we",
        "he",
        "she",
        "they",
        "you",
        "i",
        "my",
        "its",
        "our",
        "has",
        "have",
        "had",
        "will",
        "would",
        "can",
        "could",
        "may",
        "should",
        "all",
        "no",
        "than",
        "when",
        "then",
    }
)


class EmbeddingConfig(BaseModel):
    """Configuration for the TF-IDF embedder."""

    dimensions: int = 256
    tfidf_max_terms: int = 5000
    similarity_threshold: float = 0.3


class TfIdfEmbedder:
    """Pure-Python TF-IDF embedder — no external dependencies."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._vocab: list[str] = []  # ordered term list (size <= max_terms)
        self._vocab_index: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._fitted = False

    # ── tokenization ──────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, filter stopwords and short tokens."""
        tokens = re.split(r"[^a-z0-9]+", text.lower())
        return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]

    # ── fit ───────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary and IDF weights from a corpus of texts."""
        n_docs = len(texts)
        if n_docs == 0:
            return

        doc_freq: Counter[str] = Counter()
        for text in texts:
            tokens = set(self._tokenize(text))
            for tok in tokens:
                doc_freq[tok] += 1

        # Keep top-N by document frequency
        top_terms = [term for term, _ in doc_freq.most_common(self.config.tfidf_max_terms)]

        # Truncate / pad to desired dimensions
        vocab = top_terms[: self.config.dimensions]

        self._vocab = vocab
        self._vocab_index = {term: i for i, term in enumerate(vocab)}
        self._idf = {term: math.log((n_docs + 1) / (doc_freq[term] + 1)) + 1.0 for term in vocab}
        self._fitted = True

    # ── embed ─────────────────────────────────────────────────────────

    def embed(self, text: str) -> EmbeddingVector:
        """Return a unit-normalized TF-IDF vector for *text*."""
        if not self._fitted:
            self.fit([text])

        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.config.dimensions

        tf: Counter[str] = Counter(tokens)
        total = len(tokens)
        dim = self.config.dimensions
        vec = [0.0] * dim

        for term, count in tf.items():
            idx = self._vocab_index.get(term)
            if idx is not None:
                tfidf = (count / total) * self._idf.get(term, 1.0)
                vec[idx] = tfidf

        return self._unit_normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed multiple texts. Fits on them if not already fitted."""
        if not self._fitted:
            self.fit(texts)
        return [self.embed(t) for t in texts]

    # ── similarity ────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: EmbeddingVector, b: EmbeddingVector) -> float:
        """Pure-Python cosine similarity. Returns 0.0 for zero vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def top_k_similar(
        self,
        query_vec: EmbeddingVector,
        corpus: list[tuple[str, EmbeddingVector]],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return top-k (id, score) pairs sorted by cosine similarity descending."""
        scored = [(item_id, self.cosine_similarity(query_vec, vec)) for item_id, vec in corpus]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _unit_normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# Module-level singleton
default_embedder = TfIdfEmbedder()
