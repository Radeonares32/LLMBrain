from typer.testing import CliRunner

from llmbrain import __version__
from llmbrain.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
