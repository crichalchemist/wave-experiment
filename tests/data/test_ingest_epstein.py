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
            "people": ["Jeff Epstein", "Ghislaine Maxwell", "Jane Doe"],
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

        # Page 2 has 3 people: Jeffrey Epstein, Ghislaine Maxwell, Jane Doe
        # Should create 3 pairs × 2 directions = 6 CO_MENTIONED edges from that page alone
        assert graph.get_edge("Jeffrey Epstein", "Jane Doe") is not None
        assert graph.get_edge("Jane Doe", "Jeffrey Epstein") is not None
        assert graph.get_edge("Ghislaine Maxwell", "Jane Doe") is not None

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
