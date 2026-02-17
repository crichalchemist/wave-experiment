from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

_VAULT_PATH_ENV = "DETECTIVE_VAULT_PATH"


@runtime_checkable
class VaultClient(Protocol):
    """Protocol defining the interface for reading and writing vault notes.

    Vault paths are always relative to the vault root (e.g. "decisions/ADR-001.md").
    """

    def write_note(self, path: str, content: str) -> None:
        """Vault-relative paths allow notes to be meaningful both in-repo and in Obsidian."""
        ...

    def read_note(self, path: str) -> str:
        """Raising rather than returning None makes missing notes fail loud — callers must not silently skip absent decisions."""
        ...

    def search_notes(self, query: str) -> tuple[str, ...]:
        """Substring match is intentional — queries here are human analytical terms, not structured queries."""
        ...

    def list_notes(self, prefix: str = "") -> tuple[str, ...]:
        """Prefix scoping lets callers work within a vault subtree (decisions/, hypothesis-traces/) without scanning all notes."""
        ...


@dataclass(frozen=True)
class FileVaultClient:
    """Obsidian-compatible vault client backed by the local filesystem.

    Stores notes as plain .md files under a root directory. All paths passed to
    the public methods are relative to that root.
    """

    root: Path

    def write_note(self, path: str, content: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_note(self, path: str) -> str:
        target = self.root / path
        if not target.exists():
            raise FileNotFoundError(f"Note not found: {path}")
        return target.read_text(encoding="utf-8")

    def search_notes(self, query: str) -> tuple[str, ...]:
        if not self.root.is_dir():
            return ()
        lower_query = query.lower()
        matching = (
            str(note.relative_to(self.root))
            for note in self.root.glob("**/*.md")
            if lower_query in note.read_text(encoding="utf-8").lower()
        )
        return tuple(sorted(matching))

    def list_notes(self, prefix: str = "") -> tuple[str, ...]:
        if not self.root.is_dir():
            return ()
        all_relative = sorted(
            str(p.relative_to(self.root))
            for p in self.root.glob("**/*.md")
        )
        if not prefix:
            return tuple(all_relative)
        prefix_parts = Path(prefix.rstrip("/")).parts
        return tuple(
            p for p in all_relative
            if Path(p).parts[:len(prefix_parts)] == prefix_parts
        )


def vault_from_env() -> VaultClient:
    """Construct a FileVaultClient from the DETECTIVE_VAULT_PATH environment variable.

    Fails fast with a clear message when the variable is absent — prevents silent
    misconfiguration that would only surface at first I/O call.
    """
    value = os.environ.get(_VAULT_PATH_ENV)
    if value is None:
        raise ValueError(
            f"Environment variable {_VAULT_PATH_ENV!r} is not set. "
            "Set it to the absolute path of the Obsidian vault directory."
        )
    return FileVaultClient(Path(value))
