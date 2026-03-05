"""Tests for DOJ / CourtListener public records loader."""
from unittest.mock import MagicMock, patch

from src.data.sourcing.doj_loader import (
    load_courtlistener_batch,
    load_fbi_vault_epstein,
)
from src.data.sourcing.types import SourceDocument


def _mock_courtlistener_response(items: list[dict]) -> MagicMock:
    """Build a mock httpx response containing CourtListener-shaped results."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"results": items}
    return resp


def _make_item(docket_id: str = "12345", plain_text: str = "opinion text",
               date_filed: str = "2023-01-01", description: str = "filing") -> dict:
    return {
        "docket_id": docket_id,
        "plain_text": plain_text,
        "date_filed": date_filed,
        "description": description,
    }


# --------------------------------------------------------------------------- #
# CourtListener tests
# --------------------------------------------------------------------------- #

def test_courtlistener_returns_source_documents():
    """Mock _httpx_get to return results with plain_text, verify SourceDocument instances."""
    items = [_make_item(), _make_item(docket_id="67890", plain_text="another opinion")]
    mock_resp = _mock_courtlistener_response(items)

    with patch("src.data.sourcing.doj_loader._httpx_get", return_value=mock_resp):
        results = load_courtlistener_batch()

    assert len(results) == 2
    for doc in results:
        assert isinstance(doc, SourceDocument)
        assert doc.text
        assert doc.source.startswith("courtlistener:")
        assert "case_name" in doc.metadata
        assert "jurisdiction" in doc.metadata


def test_courtlistener_max_documents_limiting():
    """Mock with 10 results, call with max_documents=3, verify only 3 returned."""
    items = [_make_item(docket_id=str(i), plain_text=f"text {i}") for i in range(10)]
    mock_resp = _mock_courtlistener_response(items)

    with patch("src.data.sourcing.doj_loader._httpx_get", return_value=mock_resp):
        results = load_courtlistener_batch(max_documents=3)

    assert len(results) == 3


def test_courtlistener_http_error_returns_empty():
    """Mock _httpx_get to raise Exception, verify empty list returned."""
    with patch("src.data.sourcing.doj_loader._httpx_get", side_effect=Exception("connection error")):
        results = load_courtlistener_batch()

    assert results == []


def test_courtlistener_skips_empty_text():
    """Mock with results where some have empty plain_text, verify those are skipped."""
    items = [
        _make_item(docket_id="1", plain_text="valid text"),
        _make_item(docket_id="2", plain_text=""),
        _make_item(docket_id="3", plain_text="   "),
        _make_item(docket_id="4", plain_text="also valid"),
    ]
    mock_resp = _mock_courtlistener_response(items)

    with patch("src.data.sourcing.doj_loader._httpx_get", return_value=mock_resp):
        results = load_courtlistener_batch()

    assert len(results) == 2
    assert results[0].text == "valid text"
    assert results[1].text == "also valid"


# --------------------------------------------------------------------------- #
# FBI Vault tests
# --------------------------------------------------------------------------- #

def test_fbi_vault_stub_returns_empty():
    """Call load_fbi_vault_epstein with mock, verify returns empty list."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("src.data.sourcing.doj_loader._httpx_get", return_value=mock_resp):
        results = load_fbi_vault_epstein()

    assert results == []


def test_fbi_vault_error_returns_empty():
    """Mock _httpx_get to raise, verify returns empty list."""
    with patch("src.data.sourcing.doj_loader._httpx_get", side_effect=Exception("timeout")):
        results = load_fbi_vault_epstein()

    assert results == []
