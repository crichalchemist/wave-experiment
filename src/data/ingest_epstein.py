"""Ingest epstein-docs entities into a GraphStore.

Reads page-level entity co-occurrences and document-level role associations,
creating CO_MENTIONED and ASSOCIATED edges in the knowledge graph.

When the backend is KuzuGraph, edges are collected in memory and flushed in
bulk for dramatically faster ingestion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from src.core.types import RelationType
from src.data.entity_filter import DropLog, build_fuzzy_mappings, filter_entities
from src.data.epstein_adapter import (
    EpsteinAnalysis,
    iter_pages,
    load_analyses,
    load_dedupe_mappings,
    normalize,
)
from src.data.graph_store import GraphStore

logger = logging.getLogger(__name__)

# Flush every N edges to bound memory during bulk ingestion
_BULK_FLUSH_SIZE: int = 5_000


@dataclass(frozen=True)
class IngestionStats:
    """Summary of an ingestion run."""

    pages_processed: int
    entities_added: int
    edges_created: int
    skipped: int
    entities_dropped: int = 0
    fuzzy_mappings_added: int = 0


def _has_bulk(graph: GraphStore) -> bool:
    """Check if the graph backend supports bulk_add_edges."""
    return hasattr(graph, "bulk_add_edges") and callable(graph.bulk_add_edges)


def _flush_bulk(graph: GraphStore, buffer: list[tuple[str, str, RelationType, float]]) -> None:
    """Flush the edge buffer via bulk_add_edges if available."""
    if buffer and _has_bulk(graph):
        graph.bulk_add_edges(buffer)  # type: ignore[attr-defined]
        buffer.clear()


def ingest_epstein(
    root: Path,
    graph: GraphStore,
    max_pages: int | None = None,
    drop_log_path: Path | None = None,
) -> IngestionStats:
    """Populate *graph* with entity relationships from epstein-docs.

    Strategy:
    0. (Layer 2) Fuzzy dedup pre-pass — collect raw entity names, build
       new variant→canonical mappings, merge into ``people_map``.
    1. Load dedup mappings and analyses.
    2. Iterate pages; apply Layer 1+3 entity filter, then for each page
       with >= 2 clean people, create **CO_MENTIONED** edges (confidence 0.5).
    3. For each analysis with key_people, create **ASSOCIATED** edges between
       all key-person pairs (confidence 0.8).
    4. Skip pages with empty full_text or no people.

    When the graph backend supports ``bulk_add_edges`` (e.g. KuzuGraph),
    edges are buffered and flushed in batches for faster ingestion.

    Returns an :class:`IngestionStats` summary.
    """
    mappings = load_dedupe_mappings(root)
    people_map = mappings.get("people", {})
    analyses = load_analyses(root)

    # --- Layer 2: fuzzy dedup pre-pass ---
    raw_entities: list[str] = []
    for page in iter_pages(root, mappings):
        raw_entities.extend(page.people)
    unique_raw = list(set(raw_entities))
    fuzzy_new = build_fuzzy_mappings(unique_raw, people_map)
    people_map.update(fuzzy_new)
    fuzzy_mappings_added = len(fuzzy_new)
    if fuzzy_new:
        logger.info("Fuzzy dedup added %d new mappings", fuzzy_mappings_added)

    # Rebuild mappings dict with augmented people_map for iter_pages
    augmented_mappings = {**mappings, "people": people_map}

    # Set up drop log for Layer 1+3
    drop_log = DropLog(drop_log_path) if drop_log_path else None

    use_bulk = _has_bulk(graph)
    edge_buffer: list[tuple[str, str, RelationType, float]] = []

    entities_seen: set[str] = set()
    edges_created = 0
    pages_processed = 0
    skipped = 0
    entities_dropped = 0

    # --- Phase 1: page-level co-mentions ---
    for page in iter_pages(root, augmented_mappings):
        if max_pages is not None and pages_processed >= max_pages:
            break

        if not page.full_text.strip() or len(page.people) < 2:
            skipped += 1
            continue

        pages_processed += 1

        # Apply Layer 1 + Layer 3 entity filter
        clean_people = filter_entities(list(page.people), drop_log=drop_log)
        clean_orgs = filter_entities(list(page.organizations), drop_log=drop_log)
        clean_locs = filter_entities(list(page.locations), drop_log=drop_log)

        pre_filter_count = len(page.people) + len(page.organizations) + len(page.locations)
        post_filter_count = len(clean_people) + len(clean_orgs) + len(clean_locs)
        entities_dropped += pre_filter_count - post_filter_count

        entities_seen.update(clean_people)
        entities_seen.update(clean_orgs)
        entities_seen.update(clean_locs)

        # Person-person co-mention edges (all unique pairs)
        for a, b in combinations(sorted(p for p in set(clean_people) if p), 2):
            if use_bulk:
                edge_buffer.append((a, b, RelationType.CO_MENTIONED, 0.5))
                edge_buffer.append((b, a, RelationType.CO_MENTIONED, 0.5))
            else:
                graph.add_edge(a, b, RelationType.CO_MENTIONED, confidence=0.5)
                graph.add_edge(b, a, RelationType.CO_MENTIONED, confidence=0.5)
            edges_created += 2

        if use_bulk and len(edge_buffer) >= _BULK_FLUSH_SIZE:
            _flush_bulk(graph, edge_buffer)

    # Flush remaining page edges
    _flush_bulk(graph, edge_buffer)

    # --- Phase 2: analysis-level role associations ---
    for analysis in analyses.values():
        if len(analysis.key_people) < 2:
            continue

        names = [normalize(name, people_map) for name, _role in analysis.key_people]
        clean_names = filter_entities(names, drop_log=drop_log)
        unique_names = sorted(n for n in set(clean_names) if n)
        entities_seen.update(unique_names)

        for a, b in combinations(unique_names, 2):
            if use_bulk:
                edge_buffer.append((a, b, RelationType.ASSOCIATED, 0.8))
                edge_buffer.append((b, a, RelationType.ASSOCIATED, 0.8))
            else:
                graph.add_edge(a, b, RelationType.ASSOCIATED, confidence=0.8)
                graph.add_edge(b, a, RelationType.ASSOCIATED, confidence=0.8)
            edges_created += 2

    # Final flush
    _flush_bulk(graph, edge_buffer)

    return IngestionStats(
        pages_processed=pages_processed,
        entities_added=len(entities_seen),
        edges_created=edges_created,
        skipped=skipped,
        entities_dropped=entities_dropped,
        fuzzy_mappings_added=fuzzy_mappings_added,
    )
