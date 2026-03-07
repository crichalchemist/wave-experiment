---
id: ADR-026
title: MinHash/LSH near-duplicate detection
status: accepted
date: 2026-03-07
tags: [dedup, minhash, lsh, entities, documents, investigation]
---

# ADR-026: MinHash/LSH Near-Duplicate Detection

## Decision

Add a standalone dedup module (`src/data/dedup.py`) using MinHash signatures + Locality-Sensitive Hashing (LSH) for O(n) near-duplicate detection of both entities and documents. Falls back to `difflib.SequenceMatcher` for small sets (< 500 items) where the O(n^2) cost is acceptable and MinHash overhead isn't worth it.

## Context

Two dedup pain points exist in the current codebase:

1. **Entity dedup in ingestion** — `build_fuzzy_mappings()` in `entity_filter.py` uses O(n^2) pairwise `SequenceMatcher`. Works for small datasets but unacceptable for the 29K-page epstein-docs corpus where unique entity counts reach thousands.

2. **Document dedup in investigation** — the `_gather_phase()` collects documents from multiple sources without checking for near-duplicates. The same content can appear via different portals (e.g., an FBI vault document also appearing on a news site).

Three approaches were considered:

1. **SimHash** — single hash per document, fast but only catches near-identical content
2. **MinHash/LSH** — probabilistic Jaccard similarity via locality-sensitive hashing, O(n) indexing with tunable threshold
3. **TF-IDF + cosine** — more accurate but requires vectorization infrastructure

Approach 2 was chosen because:
- O(n) insertion and query time via LSH index
- Tunable similarity threshold (0.5 default)
- `datasketch` library is lightweight and well-maintained
- Natural fallback to difflib for small sets preserves existing behavior

## Architecture

```
src/data/dedup.py
    shingle_text(text, k=3) -> frozenset[str]
    compute_minhash(shingles, num_perm=128) -> MinHash | frozenset
    estimate_similarity(sig1, sig2) -> float

    DedupIndex
        .add(doc_id, text) -> DocumentFingerprint
        .find_duplicates(doc_id) -> list[DuplicateMatch]
        .deduplicate() -> list[DedupResult]

    build_entity_mappings_minhash(entities, existing, threshold)
        → delegates to difflib for len < 500
        → uses MinHash/LSH for larger sets

Frozen types:
    DedupResult(canonical, variants, similarity)
    DocumentFingerprint(doc_id, shingle_count, minhash)
    DuplicateMatch(doc_id, similarity)
```

### Integration points

1. `ingest_epstein.py` Layer 2: replace `build_fuzzy_mappings()` with `build_entity_mappings_minhash()`
2. `agent.py:_gather_phase()`: add `DedupIndex` to skip near-duplicate evidence

### Optional dependency

```toml
[project.optional-dependencies]
dedup = ["datasketch>=1.6.0"]
```

Without datasketch, falls back to pairwise Jaccard on raw shingle sets — correct but slower for large inputs.

## Consequences

- **Scalable** — O(n) dedup for 29K+ page datasets
- **Backward compatible** — small sets still use difflib, same results
- **No hard dependency** — works without datasketch via shingle-set Jaccard
- **Tunable** — threshold parameter controls sensitivity (0.5 default)
- **Dual use** — same module handles entity dedup and document dedup

## Files

- `src/data/dedup.py` — new module
- `tests/data/test_dedup.py` — new tests
- `src/data/ingest_epstein.py` — upgrade Layer 2
- `src/detective/investigation/agent.py` — document dedup in `_gather_phase()`
- `pyproject.toml` — add `dedup` optional dep group
