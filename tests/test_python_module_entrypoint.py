import subprocess
import sys


def test_python_module_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "llmbrain", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "LLM Brain" in result.stdout
