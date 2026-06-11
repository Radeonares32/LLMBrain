from llmbrain.services.chunker import chunk_documents
from llmbrain.services.redactor import REDACTION_TOKEN, redact_text
from llmbrain.services.scanner import scan_project


def test_redactor_masks_env_values_and_api_keys():
    result = redact_text(
        "DEEPSEEK_API_KEY=sk-abc12345678901234567890\n"
        "SECRET_KEY='my_secret_key'\n"
        'if password == "secret":\n'
    )

    assert result.changed
    assert "sk-abc" not in result.text
    assert "my_secret_key" not in result.text
    assert '"secret"' not in result.text
    assert result.text.count(REDACTION_TOKEN) >= 3


def test_scanner_redacts_before_chunking(tmp_path):
    (tmp_path / ".env.example").write_text(
        "SECRET_KEY=my_secret_key\nDATABASE_URL=sqlite:///app.db\n",
        encoding="utf-8",
    )

    docs = scan_project(tmp_path, "project")
    chunks = chunk_documents(docs)

    assert docs[0].raw_content_hash != docs[0].content_hash
    assert docs[0].redactions
    assert "my_secret_key" not in (docs[0].content or "")
    assert "my_secret_key" not in chunks[0].content
    assert REDACTION_TOKEN in chunks[0].content
