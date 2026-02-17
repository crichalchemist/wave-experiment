from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.memory.mcp_vault import (
    MCPVaultClient,
    _DEFAULT_HOST,
    _build_headers,
    _prefix_parts,
    mcp_vault_from_env,
)


# ---------------------------------------------------------------------------
# Helper / pure function tests
# ---------------------------------------------------------------------------


def test_build_headers_includes_bearer() -> None:
    headers = _build_headers("tok")
    assert headers == {"Authorization": "Bearer tok"}


def test_prefix_parts_empty() -> None:
    assert _prefix_parts("") == ()


def test_prefix_parts_with_value() -> None:
    assert _prefix_parts("decisions") == ("decisions",)


# ---------------------------------------------------------------------------
# Construction / import-guard tests
# ---------------------------------------------------------------------------


def test_missing_httpx_raises_import_error() -> None:
    with patch("src.memory.mcp_vault._httpx", None):
        with pytest.raises(ImportError, match="httpx is required"):
            MCPVaultClient(host="http://localhost:27123", token="tok")


# ---------------------------------------------------------------------------
# write_note tests
# ---------------------------------------------------------------------------


@patch("src.memory.mcp_vault._httpx")
def test_write_note_calls_put(mock_httpx: MagicMock) -> None:
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.put.return_value = mock_response

    vault = MCPVaultClient(host="http://localhost:27123", token="test-token")
    vault.write_note("decisions/ADR-001.md", "# ADR-001")

    mock_client.put.assert_called_once_with(
        "http://localhost:27123/vault/decisions/ADR-001.md",
        content="# ADR-001",
        headers={"Authorization": "Bearer test-token"},
    )


@patch("src.memory.mcp_vault._httpx")
def test_write_note_raises_on_non_2xx(mock_httpx: MagicMock) -> None:
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_client.put.return_value = mock_response

    vault = MCPVaultClient(host="http://localhost:27123", token="test-token")

    with pytest.raises(RuntimeError, match="500"):
        vault.write_note("decisions/ADR-001.md", "# ADR-001")


# ---------------------------------------------------------------------------
# read_note tests
# ---------------------------------------------------------------------------


@patch("src.memory.mcp_vault._httpx")
def test_read_note_calls_get(mock_httpx: MagicMock) -> None:
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "# ADR-001 content"
    mock_client.get.return_value = mock_response

    vault = MCPVaultClient(host="http://localhost:27123", token="test-token")
    result = vault.read_note("decisions/ADR-001.md")

    assert result == "# ADR-001 content"
    mock_client.get.assert_called_once_with(
        "http://localhost:27123/vault/decisions/ADR-001.md",
        headers={"Authorization": "Bearer test-token"},
    )


@patch("src.memory.mcp_vault._httpx")
def test_read_note_raises_file_not_found_on_404(mock_httpx: MagicMock) -> None:
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_client.get.return_value = mock_response

    vault = MCPVaultClient(host="http://localhost:27123", token="test-token")

    with pytest.raises(FileNotFoundError, match="decisions/ADR-001.md"):
        vault.read_note("decisions/ADR-001.md")


@patch("src.memory.mcp_vault._httpx")
def test_read_note_raises_runtime_error_on_5xx(mock_httpx: MagicMock) -> None:
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_client.get.return_value = mock_response

    vault = MCPVaultClient(host="http://localhost:27123", token="test-token")

    with pytest.raises(RuntimeError, match="503"):
        vault.read_note("decisions/ADR-001.md")


# ---------------------------------------------------------------------------
# search_notes tests
# ---------------------------------------------------------------------------


@patch("src.memory.mcp_vault._httpx")
def test_search_notes_calls_post(mock_httpx: MagicMock) -> None:
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"files": ["a.md", "b.md"]}
    mock_client.post.return_value = mock_response

    vault = MCPVaultClient(host="http://localhost:27123", token="test-token")
    result = vault.search_notes("ADR")

    assert result == ("a.md", "b.md")
    mock_client.post.assert_called_once_with(
        "http://localhost:27123/search/simple/",
        json={"query": "ADR"},
        headers={"Authorization": "Bearer test-token"},
    )


@patch("src.memory.mcp_vault._httpx")
def test_search_notes_empty_result(mock_httpx: MagicMock) -> None:
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"files": []}
    mock_client.post.return_value = mock_response

    vault = MCPVaultClient(host="http://localhost:27123", token="test-token")
    result = vault.search_notes("nonexistent")

    assert result == ()


# ---------------------------------------------------------------------------
# list_notes tests
# ---------------------------------------------------------------------------


@patch("src.memory.mcp_vault._httpx")
def test_list_notes_filters_by_prefix(mock_httpx: MagicMock) -> None:
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "files": ["decisions/ADR-001.md", "hypothesis-traces/trace.md"]
    }
    mock_client.get.return_value = mock_response

    vault = MCPVaultClient(host="http://localhost:27123", token="test-token")
    result = vault.list_notes(prefix="decisions")

    assert result == ("decisions/ADR-001.md",)


# ---------------------------------------------------------------------------
# mcp_vault_from_env tests
# ---------------------------------------------------------------------------


def test_mcp_vault_from_env_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OBSIDIAN_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="OBSIDIAN_API_TOKEN"):
        mcp_vault_from_env()


@patch("src.memory.mcp_vault._httpx")
def test_mcp_vault_from_env_default_host(
    mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_httpx.Client.return_value = MagicMock()
    monkeypatch.setenv("OBSIDIAN_API_TOKEN", "my-token")
    monkeypatch.delenv("OBSIDIAN_HOST", raising=False)

    client = mcp_vault_from_env()

    assert client.host == _DEFAULT_HOST
    assert client.token == "my-token"


@patch("src.memory.mcp_vault._httpx")
def test_mcp_vault_from_env_custom_host(
    mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_httpx.Client.return_value = MagicMock()
    monkeypatch.setenv("OBSIDIAN_API_TOKEN", "my-token")
    monkeypatch.setenv("OBSIDIAN_HOST", "http://obsidian.local:27124")

    client = mcp_vault_from_env()

    assert client.host == "http://obsidian.local:27124"
    assert client.token == "my-token"
