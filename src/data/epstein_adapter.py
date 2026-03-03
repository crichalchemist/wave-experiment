"""Adapter for the epstein-docs dataset (29K page JSONs + 8K analyses).

Parses page-level entity extractions and document-level analyses into
frozen dataclasses, applying entity deduplication from ``dedupe.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes — frozen for immutability (project convention)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpsteinPage:
    """A single page from the epstein-docs dataset with extracted entities."""

    doc_id: str
    page_number: str
    full_text: str
    people: tuple[str, ...]
    organizations: tuple[str, ...]
    locations: tuple[str, ...]
    dates: tuple[str, ...]
    document_type: str


@dataclass(frozen=True)
class EpsteinAnalysis:
    """Document-level analysis with key people, topics, and significance."""

    document_id: str
    page_count: int
    document_type: str
    key_people: tuple[tuple[str, str], ...]  # (name, role)
    key_topics: tuple[str, ...]
    significance: str
    summary: str


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def load_dedupe_mappings(root: Path) -> dict[str, dict[str, str]]:
    """Load ``dedupe.json`` — maps variant names to canonical forms.

    Returns a dict keyed by entity type (``"people"``, ``"organizations"``,
    ``"locations"``) whose values are ``{variant: canonical}`` mappings.
    """
    path = root / "dedupe.json"
    if not path.exists():
        logger.warning("dedupe.json not found at %s — skipping normalization", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize(name: str, mappings: dict[str, str]) -> str:
    """Apply dedup mapping, falling back to original if no match."""
    return mappings.get(name, name)


# ---------------------------------------------------------------------------
# Page iteration
# ---------------------------------------------------------------------------


def iter_pages(
    root: Path,
    mappings: dict[str, dict[str, str]] | None = None,
) -> Iterator[EpsteinPage]:
    """Iterate all page JSONs under ``root/results/``, yielding parsed pages.

    Entity names are normalized through the dedup mappings when provided.
    Pages with unparseable JSON are logged and skipped.
    """
    if mappings is None:
        mappings = {}

    people_map = mappings.get("people", {})
    org_map = mappings.get("organizations", {})
    loc_map = mappings.get("locations", {})

    results_dir = root / "results"
    if not results_dir.exists():
        logger.warning("results directory not found at %s", results_dir)
        return

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for json_path in sorted(subdir.glob("*.json")):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Skipping %s: %s", json_path, exc)
                continue

            meta = data.get("document_metadata", {})
            entities = data.get("entities", {})

            yield EpsteinPage(
                doc_id=json_path.stem,
                page_number=meta.get("page_number", ""),
                full_text=data.get("full_text", ""),
                people=tuple(
                    normalize(p, people_map) for p in entities.get("people", [])
                ),
                organizations=tuple(
                    normalize(o, org_map) for o in entities.get("organizations", [])
                ),
                locations=tuple(
                    normalize(loc, loc_map) for loc in entities.get("locations", [])
                ),
                dates=tuple(entities.get("dates", [])),
                document_type=meta.get("document_type", ""),
            )


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def load_analyses(root: Path) -> dict[str, EpsteinAnalysis]:
    """Load ``analyses.json`` and return analyses keyed by ``document_id``."""
    path = root / "analyses.json"
    if not path.exists():
        logger.warning("analyses.json not found at %s", path)
        return {}

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items: list[dict] = raw.get("analyses", [])
    result: dict[str, EpsteinAnalysis] = {}

    for entry in items:
        doc_id = entry.get("document_id", "")
        analysis = entry.get("analysis", {})

        key_people = tuple(
            (kp.get("name", ""), kp.get("role", ""))
            for kp in analysis.get("key_people", [])
        )

        result[doc_id] = EpsteinAnalysis(
            document_id=doc_id,
            page_count=entry.get("page_count", 0),
            document_type=analysis.get("document_type", ""),
            key_people=key_people,
            key_topics=tuple(analysis.get("key_topics", [])),
            significance=analysis.get("significance", ""),
            summary=analysis.get("summary", ""),
        )

    return result
