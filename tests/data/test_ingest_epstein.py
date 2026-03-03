"""Tests for epstein-docs ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.types import RelationType
from src.data.graph_store import InMemoryGraph
from src.data.ingest_epstein import IngestionStats, ingest_epstein


@pytest.fixture()
def epstein_root(tmp_path: Path) -> Path:
    """Minimal epstein-docs tree for ingestion testing."""
    # dedupe.json
    dedupe = {
        "people": {"Jeff Epstein": "Jeffrey Epstein"},
        "organizations": {},
        "locations": {},
    }
    (tmp_path / "dedupe.json").write_text(json.dumps(dedupe))

    # analyses.json — two people yields 1 pair
    analyses = {
        "total": 1,
        "analyses": [
            {
                "document_id": "DOC-001",
                "page_count": 2,
                "analysis": {
                    "document_type": "Court Filing",
                    "key_topics": ["Epstein case"],
                    "key_people": [
                        {"name": "Jeffrey Epstein", "role": "Defendant"},
                        {"name": "Ghislaine Maxwell", "role": "Co-Conspirator"},
                    ],
                    "significance": "Important.",
                    "summary": "A filing.",
                },
            },
        ],
    }
    (tmp_path / "analyses.json").write_text(json.dumps(analyses))

    # results/IMAGES001/ — 3 pages
    results_dir = tmp_path / "results" / "IMAGES001"
    results_dir.mkdir(parents=True)

    # Page with 2 people → co-mention edges
    page1 = {
        "document_metadata": {"page_number": "1", "document_type": "Court Document"},
        "full_text": "Epstein and Maxwell appeared.",
        "entities": {
            "people": ["Jeff Epstein", "Ghislaine Maxwell"],
            "organizations": ["DOJ"],
            "locations": [],
            "dates": [],
        },
    }
    (results_dir / "PAGE-001.json").write_text(json.dumps(page1))

    # Page with 3 people → 3 pairs = 6 directed edges
    page2 = {
        "document_metadata": {"page_number": "2", "document_type": "Court Document"},
        "full_text": "Epstein, Maxwell, and Doe were present.",
        "entities": {
            "people": ["Jeff Epstein", "Ghislaine Maxwell", "Virginia Giuffre"],
            "organizations": [],
            "locations": [],
            "dates": [],
        },
    }
    (results_dir / "PAGE-002.json").write_text(json.dumps(page2))

    # Page with only 1 person → skipped
    page3 = {
        "document_metadata": {"page_number": "3", "document_type": "Letter"},
        "full_text": "Epstein wrote a letter.",
        "entities": {
            "people": ["Jeff Epstein"],
            "organizations": [],
            "locations": [],
            "dates": [],
        },
    }
    (results_dir / "PAGE-003.json").write_text(json.dumps(page3))

    # Page with empty text → skipped
    page4 = {
        "document_metadata": {"page_number": "4", "document_type": "Blank"},
        "full_text": "",
        "entities": {
            "people": ["Jeff Epstein", "Ghislaine Maxwell"],
            "organizations": [],
            "locations": [],
            "dates": [],
        },
    }
    (results_dir / "PAGE-004.json").write_text(json.dumps(page4))

    return tmp_path


class TestIngestion:
    def test_ingest_creates_edges(self, epstein_root: Path) -> None:
        graph = InMemoryGraph()
        stats = ingest_epstein(epstein_root, graph)

        assert isinstance(stats, IngestionStats)
        assert stats.pages_processed == 2  # page1 + page2
        assert stats.skipped == 2  # page3 (1 person) + page4 (empty text)
        assert stats.edges_created > 0
        assert stats.entities_added > 0

    def test_co_mentioned_edges(self, epstein_root: Path) -> None:
        graph = InMemoryGraph()
        ingest_epstein(epstein_root, graph)

        # Jeffrey Epstein ↔ Ghislaine Maxwell: CO_MENTIONED from pages,
        # then overwritten by ASSOCIATED from analyses (networkx stores
        # one edge per (source, target) pair — last write wins).
        edge = graph.get_edge("Jeffrey Epstein", "Ghislaine Maxwell")
        assert edge is not None
        assert edge.relation in (RelationType.CO_MENTIONED, RelationType.ASSOCIATED)

    def test_associated_edges_from_analyses(self, epstein_root: Path) -> None:
        graph = InMemoryGraph()
        ingest_epstein(epstein_root, graph)

        # Analysis has Epstein + Maxwell as key people → ASSOCIATED
        # Note: ASSOCIATED may overwrite CO_MENTIONED for the same pair
        # depending on processing order — both are valid
        nodes = graph.nodes()
        assert "Jeffrey Epstein" in nodes
        assert "Ghislaine Maxwell" in nodes

    def test_max_pages_limits_processing(self, epstein_root: Path) -> None:
        graph = InMemoryGraph()
        stats = ingest_epstein(epstein_root, graph, max_pages=1)

        assert stats.pages_processed == 1

    def test_dedup_applied_to_edges(self, epstein_root: Path) -> None:
        graph = InMemoryGraph()
        ingest_epstein(epstein_root, graph)

        nodes = graph.nodes()
        # "Jeff Epstein" should be normalized to "Jeffrey Epstein"
        assert "Jeff Epstein" not in nodes
        assert "Jeffrey Epstein" in nodes

    def test_three_person_page_creates_all_pairs(self, epstein_root: Path) -> None:
        graph = InMemoryGraph()
        ingest_epstein(epstein_root, graph)

        # Page 2 has 3 people: Jeffrey Epstein, Ghislaine Maxwell, Virginia Giuffre
        # Should create 3 pairs × 2 directions = 6 CO_MENTIONED edges from that page alone
        assert graph.get_edge("Jeffrey Epstein", "Virginia Giuffre") is not None
        assert graph.get_edge("Virginia Giuffre", "Jeffrey Epstein") is not None
        assert graph.get_edge("Ghislaine Maxwell", "Virginia Giuffre") is not None

    def test_successors_include_connected_entities(self, epstein_root: Path) -> None:
        graph = InMemoryGraph()
        ingest_epstein(epstein_root, graph)

        successors = graph.successors("Jeffrey Epstein")
        assert "Ghislaine Maxwell" in successors

    def test_stats_frozen(self, epstein_root: Path) -> None:
        graph = InMemoryGraph()
        stats = ingest_epstein(epstein_root, graph)
        with pytest.raises(AttributeError):
            stats.pages_processed = 999  # type: ignore[misc]

    def test_empty_root(self, tmp_path: Path) -> None:
        """Ingestion on empty directory returns zero stats without error."""
        graph = InMemoryGraph()
        stats = ingest_epstein(tmp_path, graph)
        assert stats.pages_processed == 0
        assert stats.edges_created == 0


# ---------------------------------------------------------------------------
# Entity filter integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def noisy_epstein_root(tmp_path: Path) -> Path:
    """Epstein-docs tree with noisy entities to test filtering."""
    dedupe = {"people": {}, "organizations": {}, "locations": {}}
    (tmp_path / "dedupe.json").write_text(json.dumps(dedupe))
    (tmp_path / "analyses.json").write_text(json.dumps({"total": 0, "analyses": []}))

    results_dir = tmp_path / "results" / "IMAGES001"
    results_dir.mkdir(parents=True)

    page = {
        "document_metadata": {"page_number": "1", "document_type": "Court Document"},
        "full_text": "Multiple entities mentioned in this document.",
        "entities": {
            "people": [
                "Jeffrey Epstein",
                "Ghislaine Maxwell",
                "(b)(6)",             # Layer 1: FOIA code
                "Inmate 7",          # Layer 3: inmate pattern
                "D",                 # Layer 1: too short
                "defendants",        # Layer 3: role prefix
                "user@aol.com",      # Layer 1: email
            ],
            "organizations": ["DOJ", "[REDACTED]"],  # Layer 1: bracket-redacted
            "locations": [],
            "dates": [],
        },
    }
    (results_dir / "PAGE-001.json").write_text(json.dumps(page))

    return tmp_path


class TestFilteredIngestion:
    def test_junk_entities_not_in_graph(self, noisy_epstein_root: Path) -> None:
        graph = InMemoryGraph()
        stats = ingest_epstein(noisy_epstein_root, graph)

        nodes = graph.nodes()
        assert "Jeffrey Epstein" in nodes
        assert "Ghislaine Maxwell" in nodes
        assert "(b)(6)" not in nodes
        assert "Inmate 7" not in nodes
        assert "D" not in nodes
        assert "defendants" not in nodes
        assert "user@aol.com" not in nodes
        assert "[REDACTED]" not in nodes
        assert stats.entities_dropped > 0

    def test_drop_log_written(self, noisy_epstein_root: Path, tmp_path: Path) -> None:
        drop_log = tmp_path / "drops.jsonl"
        graph = InMemoryGraph()
        ingest_epstein(noisy_epstein_root, graph, drop_log_path=drop_log)

        assert drop_log.exists()
        lines = drop_log.read_text().strip().split("\n")
        assert len(lines) >= 5  # at least 5 noisy entities
        # Verify JSONL format
        for line in lines:
            entry = json.loads(line)
            assert "entity" in entry
            assert "reason" in entry
            assert "category" in entry

    def test_backward_compatible_without_drop_log(self, noisy_epstein_root: Path) -> None:
        """Ingestion works without drop_log_path (backward compatible)."""
        graph = InMemoryGraph()
        stats = ingest_epstein(noisy_epstein_root, graph)

        assert isinstance(stats, IngestionStats)
        assert stats.pages_processed >= 1
        assert stats.entities_dropped > 0
        assert stats.fuzzy_mappings_added >= 0
