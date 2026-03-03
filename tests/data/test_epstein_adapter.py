"""Tests for epstein-docs adapter: parsing, normalization, iteration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.epstein_adapter import (
    EpsteinAnalysis,
    EpsteinPage,
    iter_pages,
    load_analyses,
    load_dedupe_mappings,
    normalize,
)


# ---------------------------------------------------------------------------
# Fixtures — tiny on-disk structures mimicking epstein-docs layout
# ---------------------------------------------------------------------------


@pytest.fixture()
def epstein_root(tmp_path: Path) -> Path:
    """Create a minimal epstein-docs directory tree."""
    # dedupe.json
    dedupe = {
        "people": {"Jeff Epstein": "Jeffrey Epstein", "J. Epstein": "Jeffrey Epstein"},
        "organizations": {"DOJ": "Department of Justice"},
        "locations": {"NYC": "New York City"},
    }
    (tmp_path / "dedupe.json").write_text(json.dumps(dedupe))

    # analyses.json
    analyses = {
        "total": 2,
        "analyses": [
            {
                "document_id": "DOC-001",
                "page_count": 2,
                "analysis": {
                    "document_type": "Court Filing",
                    "key_topics": ["Discovery", "Epstein case"],
                    "key_people": [
                        {"name": "Jeffrey Epstein", "role": "Defendant"},
                        {"name": "Jane Doe", "role": "Plaintiff"},
                    ],
                    "significance": "Significant court filing.",
                    "summary": "A filing in the Epstein case.",
                },
            },
            {
                "document_id": "DOC-002",
                "page_count": 1,
                "analysis": {
                    "document_type": "Letter",
                    "key_topics": ["Correspondence"],
                    "key_people": [],
                    "significance": "Minor.",
                    "summary": "A letter.",
                },
            },
        ],
    }
    (tmp_path / "analyses.json").write_text(json.dumps(analyses))

    # results/IMAGES001/ with two page JSONs
    results_dir = tmp_path / "results" / "IMAGES001"
    results_dir.mkdir(parents=True)

    page1 = {
        "document_metadata": {
            "page_number": "1",
            "document_type": "Court Document",
        },
        "full_text": "Jeffrey Epstein appeared before the court.",
        "entities": {
            "people": ["Jeff Epstein", "Ghislaine Maxwell"],
            "organizations": ["DOJ"],
            "locations": ["NYC"],
            "dates": ["2019-07-06"],
        },
    }
    (results_dir / "DOC-001.json").write_text(json.dumps(page1))

    page2 = {
        "document_metadata": {
            "page_number": "2",
            "document_type": "Court Document",
        },
        "full_text": "The defendant was represented by counsel.",
        "entities": {
            "people": ["J. Epstein"],
            "organizations": [],
            "locations": [],
            "dates": [],
        },
    }
    (results_dir / "DOC-002.json").write_text(json.dumps(page2))

    # A bad JSON file to test error handling
    (results_dir / "BAD-001.json").write_text("{not valid json")

    return tmp_path


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_load_dedupe_mappings(self, epstein_root: Path) -> None:
        mappings = load_dedupe_mappings(epstein_root)
        assert "people" in mappings
        assert "organizations" in mappings
        assert "locations" in mappings
        assert mappings["people"]["Jeff Epstein"] == "Jeffrey Epstein"

    def test_load_dedupe_missing_file(self, tmp_path: Path) -> None:
        mappings = load_dedupe_mappings(tmp_path)
        assert mappings == {}

    def test_normalize_with_match(self) -> None:
        assert normalize("Jeff Epstein", {"Jeff Epstein": "Jeffrey Epstein"}) == "Jeffrey Epstein"

    def test_normalize_without_match(self) -> None:
        assert normalize("Unknown Person", {"Jeff Epstein": "Jeffrey Epstein"}) == "Unknown Person"


# ---------------------------------------------------------------------------
# Page iteration
# ---------------------------------------------------------------------------


class TestIterPages:
    def test_iter_pages_yields_pages(self, epstein_root: Path) -> None:
        mappings = load_dedupe_mappings(epstein_root)
        pages = list(iter_pages(epstein_root, mappings))
        # BAD-001.json is skipped → 2 valid pages
        assert len(pages) == 2

    def test_page_structure(self, epstein_root: Path) -> None:
        mappings = load_dedupe_mappings(epstein_root)
        pages = list(iter_pages(epstein_root, mappings))
        page = pages[0]  # DOC-001.json (sorted)

        assert isinstance(page, EpsteinPage)
        assert page.doc_id == "DOC-001"
        assert page.page_number == "1"
        assert "Jeffrey Epstein" in page.full_text

    def test_entity_normalization_applied(self, epstein_root: Path) -> None:
        mappings = load_dedupe_mappings(epstein_root)
        pages = list(iter_pages(epstein_root, mappings))

        # Page 1: "Jeff Epstein" → "Jeffrey Epstein", "DOJ" → "Department of Justice"
        page1 = pages[0]
        assert "Jeffrey Epstein" in page1.people
        assert "Ghislaine Maxwell" in page1.people
        assert "Department of Justice" in page1.organizations
        assert "New York City" in page1.locations

        # Page 2: "J. Epstein" → "Jeffrey Epstein"
        page2 = pages[1]
        assert "Jeffrey Epstein" in page2.people

    def test_iter_pages_without_mappings(self, epstein_root: Path) -> None:
        pages = list(iter_pages(epstein_root))
        assert len(pages) == 2
        # Without mappings, original names preserved
        assert "Jeff Epstein" in pages[0].people

    def test_frozen_page(self, epstein_root: Path) -> None:
        pages = list(iter_pages(epstein_root))
        with pytest.raises(AttributeError):
            pages[0].doc_id = "mutated"  # type: ignore[misc]

    def test_missing_results_dir(self, tmp_path: Path) -> None:
        pages = list(iter_pages(tmp_path))
        assert pages == []


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


class TestAnalyses:
    def test_load_analyses(self, epstein_root: Path) -> None:
        analyses = load_analyses(epstein_root)
        assert len(analyses) == 2
        assert "DOC-001" in analyses
        assert "DOC-002" in analyses

    def test_analysis_structure(self, epstein_root: Path) -> None:
        analyses = load_analyses(epstein_root)
        a = analyses["DOC-001"]

        assert isinstance(a, EpsteinAnalysis)
        assert a.document_id == "DOC-001"
        assert a.page_count == 2
        assert a.document_type == "Court Filing"
        assert len(a.key_people) == 2
        assert a.key_people[0] == ("Jeffrey Epstein", "Defendant")
        assert "Discovery" in a.key_topics

    def test_frozen_analysis(self, epstein_root: Path) -> None:
        analyses = load_analyses(epstein_root)
        with pytest.raises(AttributeError):
            analyses["DOC-001"].document_id = "mutated"  # type: ignore[misc]

    def test_load_analyses_missing_file(self, tmp_path: Path) -> None:
        analyses = load_analyses(tmp_path)
        assert analyses == {}
