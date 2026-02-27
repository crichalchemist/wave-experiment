"""Test scrape-foia CLI command."""
from click.testing import CliRunner


def test_scrape_foia_command_exists():
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["scrape-foia", "--help"])
    assert result.exit_code == 0
    assert "portal" in result.output.lower()


def test_scrape_foia_shows_portal_choices():
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["scrape-foia", "--help"])
    assert "fbi_vault" in result.output
    assert "nara" in result.output
    assert "state_dept" in result.output
