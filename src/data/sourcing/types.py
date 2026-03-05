"""
Shared types for the data sourcing pipeline.

SourceDocument is the canonical return type for all document loaders.
DocumentLoader is the structural Protocol that loaders should satisfy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SourceDocument:
    """A single sourced document with provenance metadata."""

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(Protocol):
    """Structural protocol for document loaders."""

    def __call__(self, *, max_documents: int) -> list[SourceDocument]: ...


def limit_results(
    results: list[SourceDocument], max_documents: int
) -> list[SourceDocument]:
    """Truncate results to max_documents. Shared by all loaders."""
    return results[:max_documents]
