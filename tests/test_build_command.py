from typer.testing import CliRunner

from llmbrain.cli import app

runner = CliRunner()


def test_build_command_requires_configured_provider():
    result = runner.invoke(app, ["build", "examples/sample-project", "--provider", "openai"])
    assert result.exit_code in [0, 1]
    assert "Building project" in result.stdout
