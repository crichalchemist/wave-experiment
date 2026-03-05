"""Tests for international stakeholder denouncement loader."""
from unittest.mock import MagicMock, patch

from src.data.sourcing.international_loader import (
    load_github_public_foia,
    load_iicsa_reports,
    load_occrp_batch,
)
from src.data.sourcing.types import SourceDocument


def _mock_response() -> MagicMock:
    """Build a generic successful mock httpx response."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    return resp


def _make_github_repo(full_name: str = "org/repo", description: str = "FOIA docs",
                      html_url: str = "https://github.com/org/repo",
                      private: bool = False) -> dict:
    return {
        "full_name": full_name,
        "description": description,
        "html_url": html_url,
        "private": private,
        "updated_at": "2024-01-01T00:00:00Z",
        "topics": ["foia"],
    }


# --------------------------------------------------------------------------- #
# OCCRP tests
# --------------------------------------------------------------------------- #

def test_occrp_batch_returns_source_documents():
    """Mock _httpx_get, verify returns list[SourceDocument] (currently returns empty)."""
    mock_resp = _mock_response()

    with patch("src.data.sourcing.international_loader._httpx_get", return_value=mock_resp):
        results = load_occrp_batch()

    assert isinstance(results, list)
    assert results == []  # Stub: returns empty until HTML parsing is implemented


def test_occrp_batch_handles_error():
    """Mock _httpx_get to raise, verify returns empty list."""
    with patch("src.data.sourcing.international_loader._httpx_get", side_effect=Exception("network error")):
        results = load_occrp_batch()

    assert results == []


# --------------------------------------------------------------------------- #
# IICSA tests
# --------------------------------------------------------------------------- #

def test_iicsa_reports_returns_source_documents():
    """Mock _httpx_get, verify returns list[SourceDocument] (currently returns empty)."""
    mock_resp = _mock_response()

    with patch("src.data.sourcing.international_loader._httpx_get", return_value=mock_resp):
        results = load_iicsa_reports()

    assert isinstance(results, list)
    assert results == []  # Stub: returns empty until PDF parsing is implemented


def test_iicsa_reports_handles_error():
    """Mock _httpx_get to raise, verify returns empty list."""
    with patch("src.data.sourcing.international_loader._httpx_get", side_effect=Exception("timeout")):
        results = load_iicsa_reports()

    assert results == []


# --------------------------------------------------------------------------- #
# GitHub public FOIA tests
# --------------------------------------------------------------------------- #

def test_github_foia_returns_source_documents():
    """Mock httpx.get with items, verify returns SourceDocument instances."""
    repos = [
        _make_github_repo("muckrock/foia-docs", "FOIA document archive"),
        _make_github_repo("journo/epstein-data", "Epstein investigation data"),
    ]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"items": repos}

    with patch("httpx.get", return_value=mock_resp):
        results = load_github_public_foia()

    assert len(results) == 2
    for doc in results:
        assert isinstance(doc, SourceDocument)
        assert doc.source.startswith("github:")
        assert "repo" in doc.metadata
        assert "url" in doc.metadata


def test_github_foia_skips_private_repos():
    """Include a private repo in mock results, verify it's skipped."""
    repos = [
        _make_github_repo("org/public-repo", private=False),
        _make_github_repo("org/private-repo", private=True),
        _make_github_repo("org/another-public", private=False),
    ]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"items": repos}

    with patch("httpx.get", return_value=mock_resp):
        results = load_github_public_foia()

    assert len(results) == 2
    sources = [doc.source for doc in results]
    assert "github:org/private-repo" not in sources
    assert "github:org/public-repo" in sources
    assert "github:org/another-public" in sources


def test_github_foia_max_documents():
    """Mock with 10 repos, call with max_documents=3, verify truncation via per_page param."""
    repos = [_make_github_repo(f"org/repo-{i}") for i in range(10)]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"items": repos}

    with patch("httpx.get", return_value=mock_resp) as mock_get:
        load_github_public_foia(max_documents=3)

    # max_documents is passed as per_page param to the API
    call_kwargs = mock_get.call_args
    assert call_kwargs[1]["params"]["per_page"] == 3
