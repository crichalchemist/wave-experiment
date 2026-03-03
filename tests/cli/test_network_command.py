"""Tests for detective network and detective ingest-epstein CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from src.cli.main import cli
from src.core.types import RelationType
from src.data.graph_store import InMemoryGraph


def _populated_graph() -> InMemoryGraph:
    """Build a small graph for network tests."""
    g = InMemoryGraph()
    g.add_edge("Jeffrey Epstein", "Ghislaine Maxwell", RelationType.CO_MENTIONED, 0.5)
    g.add_edge("Ghislaine Maxwell", "Jeffrey Epstein", RelationType.CO_MENTIONED, 0.5)
    g.add_edge("Jeffrey Epstein", "Jane Doe", RelationType.ASSOCIATED, 0.8)
    g.add_edge("Jane Doe", "Jeffrey Epstein", RelationType.ASSOCIATED, 0.8)
    g.add_edge("Ghislaine Maxwell", "Jane Doe", RelationType.CO_MENTIONED, 0.5)
    return g


# ---------------------------------------------------------------------------
# detective network
# ---------------------------------------------------------------------------


class TestNetworkCommand:
    def test_text_output(self) -> None:
        runner = CliRunner()
        graph = _populated_graph()
        with patch("src.data.graph_store.graph_store_from_env", return_value=graph):
            result = runner.invoke(cli, ["network", "--entity", "Jeffrey Epstein"])
        assert result.exit_code == 0
        assert "Jeffrey Epstein" in result.output
        assert "Ghislaine Maxwell" in result.output
        assert "co_mentioned" in result.output

    def test_json_output(self) -> None:
        runner = CliRunner()
        graph = _populated_graph()
        with patch("src.data.graph_store.graph_store_from_env", return_value=graph):
            result = runner.invoke(
                cli, ["network", "--entity", "Jeffrey Epstein", "--format", "json"]
            )
        assert result.exit_code == 0
        data = json.loads(result.output.split("\n\nPaths")[0])  # JSON before paths section
        assert isinstance(data, list)
        assert any(r["target"] == "Ghislaine Maxwell" for r in data)

    def test_no_connections(self) -> None:
        runner = CliRunner()
        graph = InMemoryGraph()
        with patch("src.data.graph_store.graph_store_from_env", return_value=graph):
            result = runner.invoke(cli, ["network", "--entity", "Unknown"])
        assert result.exit_code == 0
        assert "No connections found" in result.output

    def test_hops_parameter(self) -> None:
        runner = CliRunner()
        graph = _populated_graph()
        with patch("src.data.graph_store.graph_store_from_env", return_value=graph):
            result = runner.invoke(
                cli, ["network", "--entity", "Jeffrey Epstein", "--hops", "3"]
            )
        assert result.exit_code == 0
        # Should find multi-hop paths
        assert "Paths" in result.output

    def test_entity_required(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["network"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


# ---------------------------------------------------------------------------
# detective ingest-epstein
# ---------------------------------------------------------------------------


def _epstein_fixture(tmp_path: Path) -> Path:
    """Minimal epstein-docs structure for CLI testing."""
    dedupe = {"people": {}, "organizations": {}, "locations": {}}
    (tmp_path / "dedupe.json").write_text(json.dumps(dedupe))
    (tmp_path / "analyses.json").write_text(json.dumps({"total": 0, "analyses": []}))

    results_dir = tmp_path / "results" / "IMAGES001"
    results_dir.mkdir(parents=True)
    page = {
        "document_metadata": {"page_number": "1", "document_type": "Court Document"},
        "full_text": "Epstein and Maxwell appeared before the court.",
        "entities": {
            "people": ["Jeffrey Epstein", "Ghislaine Maxwell"],
            "organizations": ["DOJ"],
            "locations": [],
            "dates": [],
        },
    }
    (results_dir / "PAGE-001.json").write_text(json.dumps(page))
    return tmp_path


class TestIngestEpsteinCommand:
    def test_ingest_runs(self, tmp_path: Path) -> None:
        root = _epstein_fixture(tmp_path)
        runner = CliRunner()
        with patch("src.data.graph_store.graph_store_from_env") as mock_gsfe:
            mock_gsfe.return_value = InMemoryGraph()
            result = runner.invoke(cli, ["ingest-epstein", "--root", str(root)])
        assert result.exit_code == 0
        assert "Pages processed:" in result.output
        assert "Edges created:" in result.output

    def test_ingest_with_max_pages(self, tmp_path: Path) -> None:
        root = _epstein_fixture(tmp_path)
        runner = CliRunner()
        with patch("src.data.graph_store.graph_store_from_env") as mock_gsfe:
            mock_gsfe.return_value = InMemoryGraph()
            result = runner.invoke(
                cli, ["ingest-epstein", "--root", str(root), "--max-pages", "0"]
            )
        assert result.exit_code == 0
        assert "Pages processed:" in result.output

    def test_help_text(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest-epstein", "--help"])
        assert result.exit_code == 0
        assert "epstein-docs" in result.output

    def test_appears_in_cli_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ingest-epstein" in result.output
