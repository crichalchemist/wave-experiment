---
id: ADR-015
title: Epstein-docs entity ingestion into knowledge graph
status: accepted
date: 2026-03-03
tags: [data, graph, ingestion, entities, cli]
---

# ADR-015: Epstein-Docs Entity Ingestion into Knowledge Graph

## Decision

Ingest pre-extracted entities from the epstein-docs dataset (29,439 page JSONs + 8,186 document analyses) into the existing GraphStore infrastructure. Entity names are normalized through the dataset's deduplication mappings. Two new RelationType values (CO_MENTIONED, ASSOCIATED) capture the relationship signals. A dedicated CLI command (`detective ingest-epstein`) drives the pipeline; `detective network` is upgraded from stub to real graph explorer.

## Context

The epstein-docs dataset (MIT license, 303 stars) provides pre-extracted entities (people, organizations, locations, dates) at the page level, plus document-level analyses with key_people (name + role), summaries, and significance ratings. Entity deduplication mappings (11,299 people, 5,590 orgs, 2,956 locations) are provided in `dedupe.json`.

The GraphStore Protocol and backends (InMemoryGraph, KuzuGraph) exist since ADR-008, but had no real data flowing through them. This ingestion adapter populates the graph for the first time, enabling `detective network` to map relationships and `detective analyze` to retrieve graph-backed evidence.

## Implementation

### RelationType extensions

Two new enum values in `src/core/types.py`:
- `CO_MENTIONED` — entities appearing in the same document page (weak signal)
- `ASSOCIATED` — person linked to another person via role-based analysis (moderate signal)

Hop decay: CO_MENTIONED 0.6, ASSOCIATED 0.8. Co-mention decays fast because same-page appearance is correlative, not causal. Association is stronger because it derives from structured analysis with explicit roles.

### Adapter layer

`src/data/epstein_adapter.py` provides frozen dataclasses (`EpsteinPage`, `EpsteinAnalysis`) and functions for parsing, normalization, and iteration. `iter_pages()` walks `results/IMAGES001-12/` yielding normalized pages; `load_analyses()` parses the document-level analysis dict.

### Ingestion pipeline

`src/data/ingest_epstein.py` drives the two-phase ingestion:

1. **Page-level co-mentions** — for each page with ≥2 people, create bidirectional CO_MENTIONED edges between all person-person pairs (confidence 0.5).
2. **Analysis-level associations** — for each document analysis with ≥2 key_people, create bidirectional ASSOCIATED edges (confidence 0.8).

Pages with empty text or fewer than 2 people are skipped. Returns `IngestionStats` (frozen dataclass) with counts.

### CLI commands

- `detective ingest-epstein [--root data/epstein-docs] [--max-pages N]` — load graph backend, run ingestion, print stats.
- `detective network --entity NAME [--hops N] [--format text|json]` — query successors and n-hop paths from the graph store. Replaces the previous stub that only printed document metadata.

## Consequences

- CO_MENTIONED edges are weak (0.5 confidence, 0.6 decay) — many per page with multiple names. The graph will be dense for frequently-mentioned entities.
- ASSOCIATED edges (0.8 confidence, 0.8 decay) provide higher-quality signal from structured analyses.
- InMemoryGraph stores one edge per (source, target) pair — if both CO_MENTIONED and ASSOCIATED exist for the same pair, the last write wins. ASSOCIATED overwrites CO_MENTIONED, which is the desired behavior (stronger signal replaces weaker).
- The `network` command signature changed from positional `doc_path` to `--entity` option. This is a breaking change to the CLI interface.
- `max_pages` parameter allows incremental testing without processing all 29K pages.

## Files

- `src/core/types.py` — CO_MENTIONED, ASSOCIATED enum values
- `src/data/knowledge_graph.py` — hop decay entries for new types
- `src/data/epstein_adapter.py` — parser, normalizer, iterator
- `src/data/ingest_epstein.py` — ingestion pipeline + IngestionStats
- `src/cli/main.py` — ingest-epstein command + network rewrite
- `tests/data/test_epstein_adapter.py` — 14 tests
- `tests/data/test_ingest_epstein.py` — 9 tests
- `tests/cli/test_network_command.py` — 9 tests
- `tests/cli/test_main.py` — updated network test
