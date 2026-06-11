from pathlib import Path
from llmbrain.services.scanner import scan_project

def test_scanner_basic(tmp_path: Path):
    project_id = "test_project"
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    
    docs = scan_project(tmp_path, project_id)
    assert len(docs) == 1
    assert docs[0].path == str(test_file)
