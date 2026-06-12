from typer.testing import CliRunner

from llmbrain.cli import app

runner = CliRunner()


def test_ci_command():
    result = runner.invoke(app, ["ci", "examples/sample-project", "--fail-on", "high"])
    assert result.exit_code in [0, 1]
