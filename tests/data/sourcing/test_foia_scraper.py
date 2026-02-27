"""Tests for the FOIA portal scraper module."""

import dataclasses
import pytest


def test_foia_document_dataclass():
    """FOIADocument can be created with all required fields."""
    from src.data.sourcing.foia_scraper import FOIADocument

    doc = FOIADocument(
        source_portal="fbi_vault",
        title="Jeffrey Epstein Part 01 of 22",
        url="https://vault.fbi.gov/jeffrey-epstein/jeffrey-epstein-part-01-of-22",
        date="2019-07-10",
        collection="jeffrey-epstein",
        text="Extracted OCR text from page 1...",
        pdf_path=None,
    )
    assert doc.source_portal == "fbi_vault"
    assert doc.title == "Jeffrey Epstein Part 01 of 22"
    assert doc.url.startswith("https://vault.fbi.gov")
    assert doc.date == "2019-07-10"
    assert doc.collection == "jeffrey-epstein"
    assert "Extracted OCR text" in doc.text
    assert doc.pdf_path is None


def test_foia_document_is_frozen():
    """FOIADocument is immutable (frozen dataclass)."""
    from src.data.sourcing.foia_scraper import FOIADocument

    doc = FOIADocument(
        source_portal="nara",
        title="Declassified Cable",
        url="https://www.archives.gov/example",
        date=None,
        collection=None,
        text="Some text",
        pdf_path=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.title = "Modified title"  # type: ignore[misc]


def test_portal_configs_exist():
    """All three FOIA portals are defined in PORTAL_CONFIGS."""
    from src.data.sourcing.foia_scraper import PORTAL_CONFIGS

    assert "fbi_vault" in PORTAL_CONFIGS
    assert "nara" in PORTAL_CONFIGS
    assert "state_dept" in PORTAL_CONFIGS
    assert len(PORTAL_CONFIGS) == 3


def test_portal_config_has_required_keys():
    """Each portal config has at least base_url and description."""
    from src.data.sourcing.foia_scraper import PORTAL_CONFIGS

    for portal, config in PORTAL_CONFIGS.items():
        assert "base_url" in config, f"{portal} missing base_url"
        assert "description" in config, f"{portal} missing description"
        assert config["base_url"].startswith("https://"), (
            f"{portal} base_url must be HTTPS"
        )


def test_foia_scraper_class_exists():
    """FOIAScraper is importable and callable."""
    from src.data.sourcing.foia_scraper import FOIAScraper

    assert callable(FOIAScraper)


def test_scraper_rejects_unknown_portal(tmp_path):
    """FOIAScraper raises ValueError for an unknown portal name."""
    from src.data.sourcing.foia_scraper import FOIAScraper

    with pytest.raises(ValueError, match="Unknown portal"):
        FOIAScraper(portal="nonexistent_portal", output_dir=tmp_path)
