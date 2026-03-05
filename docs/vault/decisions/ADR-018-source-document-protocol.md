---
id: ADR-018
title: SourceDocument type and DocumentLoader Protocol
status: accepted
date: 2026-03-05
tags: [data, sourcing, types, protocol, consolidation]
---

# ADR-018: SourceDocument Type and DocumentLoader Protocol

## Decision

Introduce a `SourceDocument` frozen dataclass and `DocumentLoader` structural Protocol as the canonical types for all data sourcing loaders. Standardize the limiting parameter to `max_documents` across all loaders.

## Context

The data sourcing pipeline had four loaders (`hf_loader`, `doj_loader`, `international_loader`, `legal_sources`) that all returned `list[dict[str, Any]]` with inconsistent parameter naming (`max_examples`, `max_results`, `max_documents`). Each loader independently constructed dicts with `text`, `source`, and `metadata` keys. Callers used `.get("text")` accessor patterns with no type safety and no guarantee of field presence.

This created three problems:
1. **No compile-time guarantees** — misspelled keys (`"souce"` vs `"source"`) silently produced `None`
2. **Inconsistent limiting** — each loader reimplemented truncation with different parameter names
3. **Opaque return types** — new contributors couldn't discover the expected dict shape without reading source

## Implementation

- `src/data/sourcing/types.py` defines `SourceDocument(text, source, metadata)` as `@dataclass(frozen=True)` and `DocumentLoader` as a `Protocol` with `__call__(*, max_documents: int) -> list[SourceDocument]`
- All four loaders now return `list[SourceDocument]` and accept `max_documents`
- Callers (`constitutional_warmup.py`, `legal_warmup.py`) use attribute access (`.text`, `.source`, `.metadata`)
- `limit_results()` shared helper for truncation

## Consequences

- Type checkers catch field access errors at analysis time
- All loaders share a uniform interface — easy to add new loaders
- Frozen dataclass prevents accidental mutation of sourced documents
- `DocumentLoader` Protocol enables structural typing — any callable with the right signature satisfies it without inheritance

## Files

- `src/data/sourcing/types.py` (new)
- `src/data/sourcing/hf_loader.py` (modified)
- `src/data/sourcing/doj_loader.py` (modified)
- `src/data/sourcing/international_loader.py` (modified)
- `src/data/sourcing/legal_sources.py` (modified)
- `src/training/constitutional_warmup.py` (modified)
- `src/training/legal_warmup.py` (modified)
- `tests/data/test_sourcing.py` (modified)
- `tests/data/sourcing/test_legal_sources.py` (modified)
