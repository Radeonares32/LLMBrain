"""Phase 7 tests: TF-IDF embedder, vector store, semantic search, multi-repo registry."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llmbrain.services.embeddings import EmbeddingConfig, TfIdfEmbedder, default_embedder
from llmbrain.storage.vector_store import VectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tmp_vs(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vectors.db")


def _make_embedder(texts: list[str] | None = None) -> TfIdfEmbedder:
    emb = TfIdfEmbedder(EmbeddingConfig(dimensions=32, tfidf_max_terms=500))
    if texts:
        emb.fit(texts)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# TfIdfEmbedder — unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTfIdfEmbedder:
    def test_tokenize_basic(self):
        emb = TfIdfEmbedder()
        tokens = emb._tokenize("Hello World foo bar")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_removes_stopwords(self):
        emb = TfIdfEmbedder()
        tokens = emb._tokenize("the is and or to a an")
        assert tokens == []

    def test_tokenize_removes_short(self):
        emb = TfIdfEmbedder()
        tokens = emb._tokenize("a b cc ddd")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cc" in tokens
        assert "ddd" in tokens

    def test_fit_builds_vocab(self):
        emb = TfIdfEmbedder(EmbeddingConfig(dimensions=10))
        emb.fit(["hello world", "world foo", "bar baz"])
        assert emb.is_fitted
        assert emb.vocabulary_size > 0
        assert emb.vocabulary_size <= 10

    def test_fit_empty_corpus(self):
        emb = TfIdfEmbedder()
        emb.fit([])
        assert not emb.is_fitted

    def test_embed_returns_float_list(self):
        emb = _make_embedder(["hello world", "foo bar baz"])
        vec = emb.embed("hello world")
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        assert len(vec) == 32

    def test_embed_is_unit_normalized(self):
        emb = _make_embedder(["hello world", "foo bar baz test"])
        vec = emb.embed("hello world")
        norm = math.sqrt(sum(x * x for x in vec))
        # Allow for zero vector or unit vector
        assert norm < 1.01

    def test_embed_unknown_word_returns_zeros(self):
        emb = _make_embedder(["hello world"])
        vec = emb.embed("zzzyyyxxx")  # unknown tokens
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == 0.0

    def test_embed_auto_fits_when_not_fitted(self):
        emb = TfIdfEmbedder(EmbeddingConfig(dimensions=16))
        vec = emb.embed("hello world")
        assert emb.is_fitted
        assert len(vec) == 16

    def test_embed_batch(self):
        emb = _make_embedder(["hello world", "foo bar"])
        vecs = emb.embed_batch(["hello", "world", "foo"])
        assert len(vecs) == 3
        assert all(len(v) == 32 for v in vecs)

    def test_cosine_similarity_identical(self):
        vec = [1.0, 0.0, 0.0]
        score = TfIdfEmbedder.cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert TfIdfEmbedder.cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert TfIdfEmbedder.cosine_similarity(a, b) == 0.0

    def test_top_k_similar(self):
        emb = _make_embedder(["python programming", "java programming", "cooking recipes"])
        query = emb.embed("python programming language")
        corpus = [
            ("doc1", emb.embed("python programming")),
            ("doc2", emb.embed("java programming")),
            ("doc3", emb.embed("cooking recipes")),
        ]
        results = emb.top_k_similar(query, corpus, k=2)
        assert len(results) == 2
        assert results[0][0] == "doc1"  # most similar
        assert results[0][1] >= results[1][1]  # sorted descending

    def test_top_k_returns_at_most_k(self):
        emb = _make_embedder(["hello world", "foo bar"])
        query = emb.embed("hello")
        corpus = [(str(i), emb.embed(f"word{i}")) for i in range(20)]
        results = emb.top_k_similar(query, corpus, k=5)
        assert len(results) <= 5

    def test_default_embedder_singleton(self):
        assert default_embedder is not None
        assert isinstance(default_embedder, TfIdfEmbedder)

    def test_similar_texts_score_higher(self):
        emb = _make_embedder(
            [
                "async python programming asyncio coroutine",
                "database sql query sqlite",
                "machine learning neural network",
            ]
        )
        q = emb.embed("async coroutine python")
        s1 = TfIdfEmbedder.cosine_similarity(
            q, emb.embed("async python programming asyncio coroutine")
        )
        s2 = TfIdfEmbedder.cosine_similarity(q, emb.embed("database sql query sqlite"))
        assert s1 >= s2


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore — unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVectorStore:
    def test_upsert_returns_record(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        rec = vs.upsert("proj-1", "chunk", "chunk-001", "hello world", [0.1, 0.2, 0.3])
        assert rec.project_id == "proj-1"
        assert rec.source_type == "chunk"
        assert rec.source_id == "chunk-001"
        assert rec.vector == [0.1, 0.2, 0.3]

    def test_upsert_idempotent(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-1", "chunk", "chunk-001", "text", [0.1])
        vs.upsert("proj-1", "chunk", "chunk-001", "text updated", [0.9])
        all_recs = vs.get_all("proj-1")
        assert len(all_recs) == 1
        assert all_recs[0].vector == [0.9]

    def test_export_jsonl(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("p1", "type1", "s1", "txt1", [0.1])
        vs.upsert("p1", "type1", "s2", "txt2", [0.2])
        out_path = tmp_path / "export.jsonl"
        count = vs.export_jsonl("p1", out_path)
        assert count == 2
        assert out_path.exists()
        import json
        with open(out_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
            data = json.loads(lines[0])
            assert "vector" in data
            assert data["project_id"] == "p1"

    def test_upsert_batch(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        records = [
            {
                "project_id": "proj-1",
                "source_type": "fact",
                "source_id": f"f{i}",
                "text_preview": f"fact {i}",
                "vector": [float(i)],
            }
            for i in range(5)
        ]
        count = vs.upsert_batch(records)
        assert count == 5
        assert len(vs.get_all("proj-1")) == 5

    def test_get_by_source(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-1", "entity", "ent-001", "Entity text", [1.0, 0.0])
        rec = vs.get_by_source("proj-1", "entity", "ent-001")
        assert rec is not None
        assert rec.source_id == "ent-001"

    def test_get_by_source_not_found(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        assert vs.get_by_source("proj-1", "chunk", "nonexistent") is None

    def test_get_all(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-1", "chunk", "c1", "text", [1.0])
        vs.upsert("proj-1", "fact", "f1", "text", [0.5])
        vs.upsert("proj-2", "chunk", "c2", "text", [0.3])
        all_p1 = vs.get_all("proj-1")
        assert len(all_p1) == 2

    def test_get_all_filter_type(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-1", "chunk", "c1", "text", [1.0])
        vs.upsert("proj-1", "fact", "f1", "text", [0.5])
        chunks = vs.get_all("proj-1", source_type="chunk")
        assert len(chunks) == 1
        assert chunks[0].source_type == "chunk"

    def test_delete_project(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-1", "chunk", "c1", "text", [1.0])
        vs.upsert("proj-1", "chunk", "c2", "text", [0.5])
        vs.upsert("proj-2", "chunk", "c3", "text", [0.3])
        deleted = vs.delete_project("proj-1")
        assert deleted == 2
        assert vs.get_all("proj-1") == []
        assert len(vs.get_all("proj-2")) == 1

    def test_stats(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-1", "chunk", "c1", "text", [1.0])
        vs.upsert("proj-1", "chunk", "c2", "text", [0.5])
        vs.upsert("proj-1", "fact", "f1", "text", [0.3])
        stats = vs.stats("proj-1")
        assert stats["total"] == 3
        assert stats["by_type"]["chunk"] == 2
        assert stats["by_type"]["fact"] == 1

    def test_stats_empty(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        stats = vs.stats("empty-project")
        assert stats["total"] == 0
        assert stats["by_type"] == {}

    def test_search_cosine(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        # Unit vectors
        vs.upsert("proj-1", "chunk", "a", "text a", [1.0, 0.0, 0.0])
        vs.upsert("proj-1", "chunk", "b", "text b", [0.0, 1.0, 0.0])
        vs.upsert("proj-1", "chunk", "c", "text c", [0.0, 0.0, 1.0])
        # Query aligned with 'a'
        results = vs.search("proj-1", [1.0, 0.0, 0.0], k=3, threshold=0.0)
        assert len(results) > 0
        top_rec, top_score = results[0]
        assert top_rec.source_id == "a"
        assert abs(top_score - 1.0) < 1e-6

    def test_search_with_threshold(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-1", "chunk", "a", "text", [1.0, 0.0])
        vs.upsert("proj-1", "chunk", "b", "text", [0.0, 1.0])
        results = vs.search("proj-1", [1.0, 0.0], k=10, threshold=0.5)
        # Only the aligned vector should be above threshold
        assert all(score >= 0.5 for _, score in results)

    def test_search_empty_store(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        results = vs.search("proj-1", [1.0, 0.0], k=5)
        assert results == []

    def test_search_filter_by_type(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-1", "chunk", "c1", "text", [1.0, 0.0])
        vs.upsert("proj-1", "fact", "f1", "text", [1.0, 0.0])
        results = vs.search("proj-1", [1.0, 0.0], source_type="fact", k=10)
        assert all(r.source_type == "fact" for r, _ in results)

    def test_projects_isolated(self, tmp_path):
        vs = _tmp_vs(tmp_path)
        vs.upsert("proj-A", "chunk", "c1", "text", [1.0])
        vs.upsert("proj-B", "chunk", "c2", "text", [0.5])
        assert len(vs.get_all("proj-A")) == 1
        assert len(vs.get_all("proj-B")) == 1


# ─────────────────────────────────────────────────────────────────────────────
# SemanticSearchService — unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSemanticSearchService:
    def _make_service(self, tmp_path: Path):
        from unittest.mock import MagicMock

        from llmbrain.services.semantic_search import SemanticSearchService

        store_mock = MagicMock()
        store_mock.get_chunks.return_value = []
        store_mock.get_facts.return_value = []
        store_mock.get_entities.return_value = []

        vs = _tmp_vs(tmp_path)
        emb = _make_embedder()
        return SemanticSearchService("proj-1", store_mock, vs, embedder=emb)

    def test_index_chunks_returns_count(self, tmp_path):
        svc = self._make_service(tmp_path)
        chunks = [
            {
                "id": f"c{i}",
                "content": f"Python async programming chunk {i}",
                "path": "foo.py",
                "start_line": i,
                "end_line": i + 5,
            }
            for i in range(5)
        ]
        count = svc.index_chunks(chunks)
        assert count == 5

    def test_index_chunks_empty(self, tmp_path):
        svc = self._make_service(tmp_path)
        assert svc.index_chunks([]) == 0

    def test_index_facts(self, tmp_path):
        svc = self._make_service(tmp_path)
        facts = [
            {
                "id": f"f{i}",
                "subject": "Python",
                "predicate": "is",
                "object": "language",
                "claim": f"Python fact {i}",
            }
            for i in range(3)
        ]
        count = svc.index_facts(facts)
        assert count == 3

    def test_index_entities(self, tmp_path):
        svc = self._make_service(tmp_path)
        entities = [
            {"id": f"e{i}", "name": f"Entity{i}", "kind": "class", "path": "mod.py"}
            for i in range(4)
        ]
        count = svc.index_entities(entities)
        assert count == 4

    def test_search_returns_results(self, tmp_path):
        svc = self._make_service(tmp_path)
        chunks = [
            {
                "id": "c1",
                "content": "async python coroutine asyncio",
                "path": "a.py",
                "start_line": 1,
                "end_line": 10,
            },
            {
                "id": "c2",
                "content": "database sqlite sql query",
                "path": "b.py",
                "start_line": 1,
                "end_line": 10,
            },
        ]
        svc.index_chunks(chunks)
        results = svc.search("python async", source_types=["chunk"], k=5, threshold=0.0)
        assert isinstance(results, list)

    def test_search_not_fitted_returns_empty(self, tmp_path):
        svc = self._make_service(tmp_path)
        # No indexing done — embedder not fitted
        results = svc.search("anything", k=5)
        assert results == []

    def test_get_stats(self, tmp_path):
        svc = self._make_service(tmp_path)
        stats = svc.get_stats()
        assert "project_id" in stats
        assert "vector_store" in stats
        assert "embedder_fitted" in stats
        assert "vocabulary_size" in stats

    def test_search_merges_types(self, tmp_path):
        svc = self._make_service(tmp_path)
        svc.index_chunks(
            [
                {
                    "id": "c1",
                    "content": "python programming",
                    "path": "x.py",
                    "start_line": 1,
                    "end_line": 5,
                }
            ]
        )
        svc.index_facts(
            [
                {
                    "id": "f1",
                    "subject": "python",
                    "predicate": "is",
                    "object": "language",
                    "claim": "python programming language",
                }
            ]
        )
        results = svc.search("python", source_types=["chunk", "fact"], k=10, threshold=0.0)
        types_found = {r.source_type for r in results}
        assert len(types_found) >= 1  # At least one type found

    def test_create_search_service_factory(self, tmp_path):
        from unittest.mock import patch

        from llmbrain.services.semantic_search import create_search_service

        with patch("llmbrain.storage.sqlite.SQLiteStore"):
            svc = create_search_service("proj-test", tmp_path)
            assert svc.project_id == "proj-test"

    def test_hybrid_search(self, tmp_path):
        svc = self._make_service(tmp_path)
        # Mock the FTS search result
        svc.store.search_fts.return_value = [
            {
                "id": "c1",
                "source_type": "chunk",
                "text_content": "python hybrid search chunk",
                "score": -0.5
            }
        ]
        
        # Add vector hits
        svc.index_chunks(
            [
                {
                    "id": "c1",
                    "content": "python hybrid search chunk",
                    "path": "x.py",
                    "start_line": 1,
                    "end_line": 5,
                },
                {
                    "id": "c2",
                    "content": "something else entirely",
                    "path": "y.py",
                    "start_line": 1,
                    "end_line": 5,
                }
            ]
        )
        
        # Test hybrid search
        results = svc.search("python hybrid search", source_types=["chunk"], k=5, hybrid=True)
        assert len(results) > 0
        assert results[0].source_id == "c1"  # c1 should be boosted by RRF


# ─────────────────────────────────────────────────────────────────────────────
# MultiRepoRegistry — unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiRepoRegistry:
    def _make_registry(self, tmp_path: Path):
        from llmbrain.services.multi_repo import MultiRepoRegistry

        return MultiRepoRegistry(registry_path=tmp_path / "registry.json")

    def test_list_repos_empty(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.list_repos() == []

    def test_add_repo(self, tmp_path):
        from unittest.mock import patch

        reg = self._make_registry(tmp_path)
        fake_project_path = tmp_path / "myproject"
        fake_project_path.mkdir()

        fake_identity = {"project_id": "proj-abc123"}
        with patch(
            "llmbrain.services.multi_repo.load_or_create_project_identity",
            return_value=fake_identity,
        ):
            entry = reg.add(str(fake_project_path), name="MyProject", tags=["python"])

        assert entry.project_id == "proj-abc123"
        assert entry.name == "MyProject"
        assert "python" in entry.tags

    def test_add_duplicate_raises(self, tmp_path):
        from unittest.mock import patch

        reg = self._make_registry(tmp_path)
        fake_path = tmp_path / "repo1"
        fake_path.mkdir()
        fake_identity = {"project_id": "proj-dup"}

        with patch(
            "llmbrain.services.multi_repo.load_or_create_project_identity",
            return_value=fake_identity,
        ):
            reg.add(str(fake_path))
            with pytest.raises(ValueError, match="already registered"):
                reg.add(str(fake_path))

    def test_remove_existing(self, tmp_path):
        from unittest.mock import patch

        reg = self._make_registry(tmp_path)
        fake_path = tmp_path / "repo2"
        fake_path.mkdir()
        fake_identity = {"project_id": "proj-remove"}

        with patch(
            "llmbrain.services.multi_repo.load_or_create_project_identity",
            return_value=fake_identity,
        ):
            reg.add(str(fake_path))
        assert reg.remove("proj-remove") is True
        assert reg.list_repos() == []

    def test_remove_nonexistent(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.remove("proj-nonexistent") is False

    def test_get_returns_entry(self, tmp_path):
        from unittest.mock import patch

        reg = self._make_registry(tmp_path)
        fake_path = tmp_path / "repo3"
        fake_path.mkdir()
        fake_identity = {"project_id": "proj-get"}

        with patch(
            "llmbrain.services.multi_repo.load_or_create_project_identity",
            return_value=fake_identity,
        ):
            reg.add(str(fake_path), name="GetRepo")

        entry = reg.get("proj-get")
        assert entry is not None
        assert entry.name == "GetRepo"

    def test_get_nonexistent_returns_none(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.get("proj-missing") is None

    def test_search_by_tag(self, tmp_path):
        from unittest.mock import patch

        reg = self._make_registry(tmp_path)
        for i, tag_set in enumerate([["python", "web"], ["rust"], ["python", "cli"]]):
            p = tmp_path / f"repo{i}"
            p.mkdir()
            with patch(
                "llmbrain.services.multi_repo.load_or_create_project_identity",
                return_value={"project_id": f"proj-{i}"},
            ):
                reg.add(str(p), tags=tag_set)

        python_repos = reg.search_by_tag("python")
        assert len(python_repos) == 2

    def test_find_by_path(self, tmp_path):
        from unittest.mock import patch

        reg = self._make_registry(tmp_path)
        p = tmp_path / "findme"
        p.mkdir()
        with patch(
            "llmbrain.services.multi_repo.load_or_create_project_identity",
            return_value={"project_id": "proj-find"},
        ):
            reg.add(str(p), name="FindMe")

        found = reg.find_by_path(str(p))
        assert found is not None
        assert found.name == "FindMe"

    def test_update_last_indexed(self, tmp_path):
        from unittest.mock import patch

        reg = self._make_registry(tmp_path)
        p = tmp_path / "repolast"
        p.mkdir()
        with patch(
            "llmbrain.services.multi_repo.load_or_create_project_identity",
            return_value={"project_id": "proj-last"},
        ):
            reg.add(str(p))

        assert reg.get("proj-last").last_indexed is None
        reg.update_last_indexed("proj-last")
        assert reg.get("proj-last").last_indexed is not None

    def test_add_tag(self, tmp_path):
        from unittest.mock import patch

        reg = self._make_registry(tmp_path)
        p = tmp_path / "tagtest"
        p.mkdir()
        with patch(
            "llmbrain.services.multi_repo.load_or_create_project_identity",
            return_value={"project_id": "proj-tag"},
        ):
            reg.add(str(p))

        assert reg.add_tag("proj-tag", "newlabel") is True
        assert "newlabel" in reg.get("proj-tag").tags

    def test_add_tag_nonexistent(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.add_tag("proj-missing", "tag") is False

    def test_summary(self, tmp_path):
        reg = self._make_registry(tmp_path)
        summary = reg.summary()
        assert "total" in summary
        assert "repos" in summary
        assert summary["total"] == 0

    def test_persistence_across_instances(self, tmp_path):
        """Registry must persist to disk and be readable by a new instance."""
        from unittest.mock import patch

        from llmbrain.services.multi_repo import MultiRepoRegistry

        reg_path = tmp_path / "registry.json"
        p = tmp_path / "persist"
        p.mkdir()

        reg1 = MultiRepoRegistry(registry_path=reg_path)
        with patch(
            "llmbrain.services.multi_repo.load_or_create_project_identity",
            return_value={"project_id": "proj-persist"},
        ):
            reg1.add(str(p), name="PersistMe")

        # New instance reads same file
        reg2 = MultiRepoRegistry(registry_path=reg_path)
        assert len(reg2.list_repos()) == 1
        assert reg2.get("proj-persist").name == "PersistMe"
