"""Test extract-scenarios CLI command."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.cli.main import cli


def test_extract_scenarios_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["extract-scenarios", "--help"])
    assert result.exit_code == 0
    assert "corpus" in result.output.lower() or "extract" in result.output.lower()


def test_extract_scenarios_runs_pipeline(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Some investigative text about entities and events.", encoding="utf-8")
    output = tmp_path / "out.json"

    fake_result = {
        "profiles": [{"id": "p1"}],
        "patterns": [{"name": "decline", "shape": "monotonic_down"}],
        "scenarios": [{"name": "s1"}],
    }

    runner = CliRunner()
    with patch("src.inference.scenario_extraction.run_extraction_pipeline", return_value=fake_result):
        result = runner.invoke(cli, [
            "extract-scenarios", str(corpus),
            "--output", str(output),
            "--length", "100",
        ])

    assert result.exit_code == 0, result.output
    assert "Profiles: 1" in result.output
    assert "Patterns: 1" in result.output
    assert "Scenarios: 1" in result.output
    assert output.exists()

    saved = json.loads(output.read_text())
    assert len(saved) == 1
    assert saved[0]["name"] == "decline"


def test_extract_scenarios_default_output(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Text.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["extract-scenarios", str(corpus), "--help"])
    assert "extracted_scenarios.json" in result.output


def test_extract_scenarios_missing_corpus() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["extract-scenarios", "/nonexistent/path.txt"])
    assert result.exit_code != 0
