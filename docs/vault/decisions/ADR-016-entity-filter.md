---
id: ADR-016
title: 3-layer entity filter for epstein-docs ingestion
status: accepted
date: 2026-03-03
tags: [data, graph, ingestion, entities, filtering, dedup]
---

# ADR-016: 3-Layer Entity Filter for Epstein-Docs Ingestion

## Decision

Add a 3-layer entity filter to the epstein-docs ingestion pipeline that removes noise entities (FOIA codes, role descriptions, anonymized identifiers, variant spellings) before they enter the knowledge graph. Dropped entities are logged to JSONL for investigative audit.

## Context

The epstein-docs ingestion (ADR-015) produces ~3,772 entity nodes, of which ~862 (22.9%) are noise. The noise breaks down into three categories:

- **Layer 1 (junk):** FOIA redaction codes (`(b)(6)`), entities ≤2 chars, emails/handles, numeric refs, bracket-redacted entries (~305 nodes)
- **Layer 2 (fuzzy dedup):** Variant spellings missed by the static `dedupe.json` — e.g., `"EPSTEIN, JEFFREY EDWARD"` vs `"Jeffrey Epstein (Mr. Epstein)"` — requiring fuzzy string matching
- **Layer 3 (role descriptions):** Possessives (`Epstein's attorneys`), inmate patterns (`Inmate 7`), anonymized officers (`*D - CBP OFFCR-C`), lowercase informal strings (`autograph`), role prefix keywords (`defendants`, `co-conspirator`) (~436 nodes)

Without filtering, these noise entities create spurious graph edges that degrade relationship queries and waste storage.

## Implementation

### Filter module

`src/data/entity_filter.py` implements all three layers plus a `DropLog` JSONL writer:

- `is_junk(entity) -> str | None` — Layer 1 regex-based junk detection (6 pattern types)
- `is_role_description(entity) -> str | None` — Layer 3 heuristic role detection (6 pattern types)
- `build_fuzzy_mappings(entities, existing_mappings, threshold=0.75) -> dict` -- Layer 2 pre-pass using `difflib.SequenceMatcher` (now called indirectly via `build_entity_mappings_minhash()` from `src/data/dedup.py` (ADR-026), which delegates to `build_fuzzy_mappings()` for sets < 500 entities)
- `filter_entities(entities, drop_log=None) -> list[str]` — orchestrator applying Layers 1+3

### Pipeline integration

Layer 2 runs as a one-time pre-pass before the main ingestion loop, collecting all raw entity names and building fuzzy variant→canonical mappings that augment the `people_map` from `dedupe.json`. Layers 1+3 apply per-page inside the loop via `filter_entities()`.

### Threshold choices

- **Fuzzy dedup threshold: 0.75** — chosen via testing against the real entity corpus. Below 0.7, false positives appear (e.g., matching unrelated short surnames). Above 0.8, legitimate variants like `"EPSTEIN, JEFFREY"` vs `"Jeffrey Epstein"` are missed.
- **Lowercase informal threshold: >5 chars** — entities ≤5 chars in all-lowercase are kept to avoid dropping legitimate short surnames.

### Drop logging

`DropLog` writes each filtered entity as a JSON line with `entity`, `reason`, and `category` fields. This preserves investigative provenance — "noise" entities may be forensically relevant later.

### CLI

`detective ingest-epstein` gains `--drop-log PATH` option. `IngestionStats` adds `entities_dropped` and `fuzzy_mappings_added` fields.

## Consequences

- Entity count drops by ~22% (862 of 3,772), with corresponding reduction in spurious graph edges
- Fuzzy dedup pre-pass adds ~26s to ingestion time with the `difflib` path; the MinHash path (ADR-026) may differ in timing depending on set size. `datasketch` is an optional dependency for MinHash/LSH acceleration.
- The filter is conservative — false positives (dropping real entities) are minimized by narrow regex patterns and explicit role prefix lists
- Drop log enables post-hoc review: `wc -l` for volume, `jq` queries for category breakdown
- No changes to the `epstein_adapter.py` normalize/iter_pages interface — filter is additive

## Files

- `src/data/entity_filter.py` — filter module (DropLog, 3 layers, orchestrator)
- `src/data/ingest_epstein.py` — pipeline integration, extended IngestionStats
- `src/cli/main.py` — `--drop-log` CLI option
- `tests/data/test_entity_filter.py` — 42 tests
- `tests/data/test_ingest_epstein.py` — 3 new integration tests (12 total)
