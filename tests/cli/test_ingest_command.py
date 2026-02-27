"""Test ingest CLI command."""
from click.testing import CliRunner


def test_ingest_command_exists():
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "evidence" in result.output.lower()
    assert "training" in result.output.lower()
