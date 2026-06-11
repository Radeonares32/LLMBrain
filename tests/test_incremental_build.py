from uuid import uuid4

from llmbrain.models.entity import Entity
from llmbrain.models.fact import Fact, FactEvidence
from llmbrain.services.chunker import chunk_documents
from llmbrain.services.project_service import ProjectService
from llmbrain.services.scanner import scan_project
from llmbrain.storage.filesystem import ensure_output_dirs
from llmbrain.storage.jsonl import write_jsonl


def test_incremental_reuses_unchanged_facts_and_entities(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("def login(user):\n    return user\n", encoding="utf-8")

    project_id = "project"
    docs = scan_project(tmp_path, project_id)
    chunks = chunk_documents(docs)
    dirs = ensure_output_dirs(tmp_path)

    fact_id = uuid4().hex
    fact = Fact(
        id=fact_id,
        project_id=project_id,
        subject="app.py",
        predicate="defines_function",
        object="login",
        claim="app.py defines login.",
        confidence="high",
        evidence=[
            FactEvidence(
                id=uuid4().hex,
                fact_id=fact_id,
                document_id=docs[0].id,
                path=chunks[0].path,
                start_line=chunks[0].start_line,
                end_line=chunks[0].end_line,
            )
        ],
    )
    entity = Entity(
        id=uuid4().hex,
        project_id=project_id,
        name="login",
        type="function",
        path="app.py",
        confidence="high",
    )
    write_jsonl(dirs["root"] / "documents.jsonl", docs)
    write_jsonl(dirs["root"] / "chunks.jsonl", chunks)
    write_jsonl(dirs["root"] / "facts.jsonl", [fact])
    write_jsonl(dirs["root"] / "entities.jsonl", [entity])

    new_docs = scan_project(tmp_path, project_id)
    new_chunks = chunk_documents(new_docs)
    service = ProjectService()
    facts, chunks_to_extract, entities, docs_to_extract = service._reuse_incremental_memory(
        dirs,
        new_docs,
        new_chunks,
        project_id,
    )

    assert len(facts) == 1
    assert facts[0].evidence[0].document_id == new_docs[0].id
    assert chunks_to_extract == []
    assert len(entities) == 1
    assert docs_to_extract == []


def test_incremental_no_change_build_does_not_create_provider(tmp_path, monkeypatch):
    source = tmp_path / "app.py"
    source.write_text("def login(user):\n    return user\n", encoding="utf-8")

    project_id = ProjectService._project_id_from_path(tmp_path.resolve())
    docs = scan_project(tmp_path, project_id)
    chunks = chunk_documents(docs)
    dirs = ensure_output_dirs(tmp_path)

    fact_id = uuid4().hex
    write_jsonl(dirs["root"] / "documents.jsonl", docs)
    write_jsonl(dirs["root"] / "chunks.jsonl", chunks)
    write_jsonl(
        dirs["root"] / "facts.jsonl",
        [
            Fact(
                id=fact_id,
                project_id=project_id,
                subject="app.py",
                predicate="defines_function",
                object="login",
                claim="app.py defines login.",
                confidence="high",
                evidence=[
                    FactEvidence(
                        id=uuid4().hex,
                        fact_id=fact_id,
                        document_id=docs[0].id,
                        path=chunks[0].path,
                        start_line=chunks[0].start_line,
                        end_line=chunks[0].end_line,
                    )
                ],
            )
        ],
    )
    write_jsonl(
        dirs["root"] / "entities.jsonl",
        [
            Entity(
                id=uuid4().hex,
                project_id=project_id,
                name="login",
                type="function",
                path="app.py",
                confidence="high",
            )
        ],
    )
    write_jsonl(dirs["root"] / "relations.jsonl", [])

    def fail_provider(_provider_name):
        raise AssertionError("provider should not be created for unchanged incremental build")

    monkeypatch.setattr("llmbrain.services.project_service.create_provider", fail_provider)

    result = ProjectService().build_project(str(tmp_path), llm_provider="openai", incremental=True)

    assert result.stats.facts == 1
    assert result.stats.entities == 1
