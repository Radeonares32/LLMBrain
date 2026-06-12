from llmbrain.models.document import Document
from llmbrain.services.chunker import chunk_document


def test_chunker_basic():
    content = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\n"
    doc = Document(
        id="doc_id",
        project_id="proj_id",
        path="/some/path.py",
        relative_path="path.py",
        content_hash="hash",
        file_type=".py",
        language="python",
        line_count=8,
        size_bytes=100,
        content=content,
    )
    chunks = chunk_document(doc, max_lines=4, overlap=2)
    assert len(chunks) > 0
    assert chunks[0].start_line == 1
    assert chunks[1].start_line == 3  # 4 end_line - 2 overlap + 1 = 3
