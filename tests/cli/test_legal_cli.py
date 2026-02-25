from click.testing import CliRunner
from unittest.mock import patch, MagicMock


def test_legal_warmup_cli_exists():
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["legal-warmup", "--help"])
    assert result.exit_code == 0
    assert "legal" in result.output.lower()


def test_legal_warmup_cli_invokes_pipeline():
    from src.cli.main import cli
    runner = CliRunner()

    mock_cfg_cls = MagicMock()
    mock_run = MagicMock(return_value=42)

    with patch("src.cli.main.provider_from_env", return_value=MagicMock()), \
         patch("src.core.providers.critic_provider_from_env", return_value=MagicMock()), \
         patch("src.training.legal_warmup.run_legal_warmup", mock_run), \
         patch("src.training.legal_warmup.LegalWarmupConfig", mock_cfg_cls):

        mock_cfg_cls.return_value = MagicMock()
        result = runner.invoke(cli, [
            "legal-warmup",
            "--domains", "criminal_justice",
            "--examples-per-domain", "10",
        ])

    assert result.exit_code == 0
    assert "42" in result.output
