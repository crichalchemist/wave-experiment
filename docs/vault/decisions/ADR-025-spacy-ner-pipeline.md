---
id: ADR-025
title: spaCy NER pipeline with heuristic fallback
status: accepted
date: 2026-03-07
tags: [ner, entities, spacy, investigation, enrich]
---

# ADR-025: spaCy NER Pipeline with Heuristic Fallback

## Decision

Add a standalone NER module (`src/data/ner.py`) using spaCy as the primary entity extraction backend with an improved heuristic fallback. The module returns frozen `NerResult` dataclasses and replaces the word-splitting heuristic in the investigation agent's `_enrich_phase()`.

## Context

The investigation agent's `_enrich_phase()` extracts entities from gathered documents using a simple heuristic: split on whitespace, keep words that start with an uppercase letter and are purely alphabetic. This approach has significant blind spots:

1. **Multi-word entities** — "Jeffrey Epstein", "Southern District of New York" are split into separate words
2. **Organizations** — "FBI", "SEC", "DOJ" are missed (all-caps or too short)
3. **Locations and dates** — no distinction between entity types
4. **False positives** — sentence-starting words ("The", "However") pass the uppercase filter

spaCy provides production-quality NER with pre-trained models that handle all these cases. The `en_core_web_trf` (transformer) model offers highest accuracy; `en_core_web_sm` (statistical) is a lighter alternative.

Three approaches were considered:

1. **spaCy only** — hard dependency, fails without spaCy installed
2. **spaCy primary + heuristic fallback** — graceful degradation when spaCy unavailable
3. **Custom NER model** — too much effort for marginal gains over spaCy

Approach 2 was chosen to maintain the project's graceful degradation pattern (see `ocr_provider.py`, `foia_scraper.py`).

## Architecture

```
src/data/ner.py
    extract_entities(text: str) -> NerResult
        ├─ _load_spacy()  [lazy: en_core_web_trf → en_core_web_sm → None]
        ├─ _spacy_extract(text, nlp) -> NerResult  [if spaCy available]
        └─ _heuristic_extract(text) -> NerResult   [fallback]

NerEntity(frozen): text, label, start, end
NerResult(frozen): entities, backend, text_length
    .persons -> tuple[NerEntity, ...]
    .organizations -> tuple[NerEntity, ...]
    .unique_texts(label?) -> tuple[str, ...]
```

### Integration point

`agent.py:_enrich_phase()` replaces word-splitting with:
```python
ner_result = extract_entities(doc.text)
for ent in ner_result.entities:
    if ent.label in ("PERSON", "ORG"):
        all_entity_names.append(ent.text)
```

### Optional dependency

```toml
[project.optional-dependencies]
ner = ["spacy>=3.7.0"]
```

spaCy models are downloaded separately (`python -m spacy download en_core_web_sm`). The module detects available models at first call and caches the result.

## Consequences

- **Better entity quality** — multi-word entities, organizations, locations correctly identified
- **Type-aware extraction** — PERSON vs ORG vs GPE enables smarter graph edges
- **No hard dependency** — works without spaCy via improved heuristic
- **Lazy loading** — spaCy model loaded once on first call, not at import time
- **Testable** — pure function, no side effects, tests work with or without spaCy

## Files

- `src/data/ner.py` — new module
- `tests/data/test_ner.py` — new tests
- `src/detective/investigation/agent.py` — rewire `_enrich_phase()`
- `pyproject.toml` — add `ner` optional dep group
