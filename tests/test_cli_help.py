from typer.testing import CliRunner

from llmbrain.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM Brain" in result.stdout
