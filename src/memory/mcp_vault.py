from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import httpx as _httpx
except ImportError:
    _httpx = None  # type: ignore[assignment]

_DEFAULT_HOST = "http://localhost:27123"
_API_TOKEN_ENV = "OBSIDIAN_API_TOKEN"
_OBSIDIAN_HOST_ENV = "OBSIDIAN_HOST"

_HTTP_OK_MIN = 200
_HTTP_OK_MAX = 299
_HTTP_NOT_FOUND = 404

_SEARCH_QUERY_KEY = "query"
_RESPONSE_FILES_KEY = "files"


def _build_headers(token: str) -> dict[str, str]:
    """Bearer auth header — extracted so tests can verify headers without I/O."""
    return {"Authorization": f"Bearer {token}"}


def _prefix_parts(prefix: str) -> tuple[str, ...]:
    """Decompose a vault path prefix into its path components.

    Empty string returns () so callers can test truthiness before filtering.
    Mirrors FileVaultClient's list_notes prefix logic to avoid duplication.
    """
    if not prefix:
        return ()
    return Path(prefix.rstrip("/")).parts


@dataclass(frozen=True)
class MCPVaultClient:
    """Obsidian Local REST API client implementing the VaultClient protocol.

    Requires the Obsidian Local REST API community plugin running at `host`.
    Uses httpx for HTTP — fails loud at construction if httpx is absent rather
    than at first I/O, preventing misconfiguration from surfacing late.
    """

    host: str
    token: str

    def __post_init__(self) -> None:
        if _httpx is None:
            raise ImportError(
                "httpx is required for MCPVaultClient. Run: pip install httpx"
            )
        # Frozen dataclass — use object.__setattr__ to store the lazy client.
        object.__setattr__(self, "_client", _httpx.Client())

    @property
    def _http(self):  # noqa: ANN201 — internal accessor, type varies by mock
        return object.__getattribute__(self, "_client")

    def close(self) -> None:
        """Release the underlying httpx connection pool."""
        self._http.close()

    def write_note(self, path: str, content: str) -> None:
        url = f"{self.host}/vault/{path}"
        response = self._http.put(url, content=content, headers=_build_headers(self.token))
        if not (_HTTP_OK_MIN <= response.status_code <= _HTTP_OK_MAX):
            raise RuntimeError(
                f"write_note failed for {path!r}: HTTP {response.status_code}"
            )

    def read_note(self, path: str) -> str:
        url = f"{self.host}/vault/{path}"
        response = self._http.get(url, headers=_build_headers(self.token))
        if response.status_code == _HTTP_NOT_FOUND:
            raise FileNotFoundError(f"Note not found in vault: {path}")
        if not (_HTTP_OK_MIN <= response.status_code <= _HTTP_OK_MAX):
            raise RuntimeError(
                f"read_note failed for {path!r}: HTTP {response.status_code}"
            )
        return response.text

    def search_notes(self, query: str) -> tuple[str, ...]:
        url = f"{self.host}/search/simple/"
        response = self._http.post(
            url, json={_SEARCH_QUERY_KEY: query}, headers=_build_headers(self.token)
        )
        if not (_HTTP_OK_MIN <= response.status_code <= _HTTP_OK_MAX):
            raise RuntimeError(
                f"search_notes failed for query {query!r}: HTTP {response.status_code}"
            )
        files: list[str] = response.json().get(_RESPONSE_FILES_KEY, [])
        return tuple(files)

    def list_notes(self, prefix: str = "") -> tuple[str, ...]:
        url = f"{self.host}/vault/"
        response = self._http.get(url, headers=_build_headers(self.token))
        if not (_HTTP_OK_MIN <= response.status_code <= _HTTP_OK_MAX):
            raise RuntimeError(
                f"list_notes failed: HTTP {response.status_code}"
            )
        files: list[str] = response.json().get(_RESPONSE_FILES_KEY, [])
        sorted_files = sorted(files)
        if not prefix:
            return tuple(sorted_files)
        parts = _prefix_parts(prefix)
        return tuple(
            p for p in sorted_files
            if Path(p).parts[: len(parts)] == parts
        )


def mcp_vault_from_env() -> MCPVaultClient:
    """Construct an MCPVaultClient from environment variables.

    Fails fast when OBSIDIAN_API_TOKEN is absent — an unconfigured token would
    silently produce 401s on every request, masking the real configuration error.
    """
    token = os.environ.get(_API_TOKEN_ENV)
    if token is None:
        raise ValueError(
            f"Environment variable {_API_TOKEN_ENV!r} is not set. "
            "Set it to your Obsidian Local REST API bearer token."
        )
    host = os.environ.get(_OBSIDIAN_HOST_ENV, _DEFAULT_HOST)
    return MCPVaultClient(host=host, token=token)
