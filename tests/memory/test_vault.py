from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.memory.vault import FileVaultClient, VaultClient, vault_from_env


def test_write_read_roundtrip(tmp_path: Path) -> None:
    client = FileVaultClient(tmp_path)
    content = "# Test Note\n\nSome content here."
    client.write_note("decisions/ADR-001.md", content)
    result = client.read_note("decisions/ADR-001.md")
    assert result == content


def test_read_missing_raises(tmp_path: Path) -> None:
    client = FileVaultClient(tmp_path)
    missing_path = "decisions/does-not-exist.md"
    with pytest.raises(FileNotFoundError, match=missing_path):
        client.read_note(missing_path)


def test_search_finds_matching_notes(tmp_path: Path) -> None:
    client = FileVaultClient(tmp_path)
    client.write_note("notes/alpha.md", "This note mentions quantum entanglement.")
    client.write_note("notes/beta.md", "This note is about classical mechanics.")
    results = client.search_notes("quantum entanglement")
    assert results == ("notes/alpha.md",)


def test_search_case_insensitive(tmp_path: Path) -> None:
    client = FileVaultClient(tmp_path)
    client.write_note("notes/lower.md", "hypothesis trace: network analysis")
    results = client.search_notes("HYPOTHESIS TRACE")
    assert results == ("notes/lower.md",)


def test_list_notes_all(tmp_path: Path) -> None:
    client = FileVaultClient(tmp_path)
    client.write_note("decisions/ADR-001.md", "content a")
    client.write_note("hypotheses/H-001.md", "content b")
    results = client.list_notes("")
    assert results == ("decisions/ADR-001.md", "hypotheses/H-001.md")


def test_list_notes_prefix(tmp_path: Path) -> None:
    client = FileVaultClient(tmp_path)
    client.write_note("decisions/ADR-001.md", "content a")
    client.write_note("decisions/ADR-002.md", "content b")
    client.write_note("hypotheses/H-001.md", "content c")
    results = client.list_notes("decisions/")
    assert results == ("decisions/ADR-001.md", "decisions/ADR-002.md")


def test_vault_client_protocol_satisfied(tmp_path: Path) -> None:
    client = FileVaultClient(tmp_path)
    assert isinstance(client, VaultClient)


def test_vault_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DETECTIVE_VAULT_PATH", raising=False)
    with pytest.raises(ValueError, match="DETECTIVE_VAULT_PATH"):
        vault_from_env()


def test_vault_from_env_returns_file_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DETECTIVE_VAULT_PATH", str(tmp_path))
    client = vault_from_env()
    assert isinstance(client, FileVaultClient)


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    client = FileVaultClient(tmp_path)
    client.write_note("a/b/c.md", "nested content")
    assert (tmp_path / "a" / "b" / "c.md").exists()
    result = client.read_note("a/b/c.md")
    assert result == "nested content"
