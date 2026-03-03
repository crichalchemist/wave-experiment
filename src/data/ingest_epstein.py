"""Ingest epstein-docs entities into a GraphStore.

Reads page-level entity co-occurrences and document-level role associations,
creating CO_MENTIONED and ASSOCIATED edges in the knowledge graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from src.core.types import RelationType
from src.data.epstein_adapter import (
    EpsteinAnalysis,
    iter_pages,
    load_analyses,
    load_dedupe_mappings,
    normalize,
)
from src.data.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionStats:
    """Summary of an ingestion run."""

    pages_processed: int
    entities_added: int
    edges_created: int
    skipped: int


def ingest_epstein(
    root: Path,
    graph: GraphStore,
    max_pages: int | None = None,
) -> IngestionStats:
    """Populate *graph* with entity relationships from epstein-docs.

    Strategy:
    1. Load dedup mappings and analyses.
    2. Iterate pages; for each page with >= 2 people, create
       **CO_MENTIONED** edges between all person-person pairs (confidence 0.5).
    3. For each analysis with key_people, create **ASSOCIATED** edges between
       all key-person pairs (confidence 0.8).
    4. Skip pages with empty full_text or no people.

    Returns an :class:`IngestionStats` summary.
    """
    mappings = load_dedupe_mappings(root)
    people_map = mappings.get("people", {})
    analyses = load_analyses(root)

    entities_seen: set[str] = set()
    edges_created = 0
    pages_processed = 0
    skipped = 0

    # --- Phase 1: page-level co-mentions ---
    for page in iter_pages(root, mappings):
        if max_pages is not None and pages_processed >= max_pages:
            break

        if not page.full_text.strip() or len(page.people) < 2:
            skipped += 1
            continue

        pages_processed += 1
        entities_seen.update(page.people)
        entities_seen.update(page.organizations)
        entities_seen.update(page.locations)

        # Person-person co-mention edges (all unique pairs)
        for a, b in combinations(sorted(set(page.people)), 2):
            graph.add_edge(a, b, RelationType.CO_MENTIONED, confidence=0.5)
            graph.add_edge(b, a, RelationType.CO_MENTIONED, confidence=0.5)
            edges_created += 2

    # --- Phase 2: analysis-level role associations ---
    _ingest_analysis_edges(analyses, people_map, graph, entities_seen)
    # Count edges from analyses separately
    analysis_edges = _count_analysis_edges(analyses, people_map)
    edges_created += analysis_edges

    return IngestionStats(
        pages_processed=pages_processed,
        entities_added=len(entities_seen),
        edges_created=edges_created,
        skipped=skipped,
    )


def _ingest_analysis_edges(
    analyses: dict[str, EpsteinAnalysis],
    people_map: dict[str, str],
    graph: GraphStore,
    entities_seen: set[str],
) -> None:
    """Create ASSOCIATED edges between key people in each analysis."""
    for analysis in analyses.values():
        if len(analysis.key_people) < 2:
            continue

        names = [normalize(name, people_map) for name, _role in analysis.key_people]
        unique_names = sorted(set(names))
        entities_seen.update(unique_names)

        for a, b in combinations(unique_names, 2):
            graph.add_edge(a, b, RelationType.ASSOCIATED, confidence=0.8)
            graph.add_edge(b, a, RelationType.ASSOCIATED, confidence=0.8)


def _count_analysis_edges(
    analyses: dict[str, EpsteinAnalysis],
    people_map: dict[str, str],
) -> int:
    """Count edges that would be created from analyses (for stats)."""
    total = 0
    for analysis in analyses.values():
        if len(analysis.key_people) < 2:
            continue
        names = [normalize(name, people_map) for name, _role in analysis.key_people]
        n = len(set(names))
        total += n * (n - 1)  # bidirectional pairs
    return total
