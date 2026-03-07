---
id: ADR-024
title: OCR fallback chain with confidence scoring
status: accepted
date: 2026-03-07
tags: [ocr, fallback, confidence, document-ingestion]
---

# ADR-024: OCR Fallback Chain with Confidence Scoring

## Decision

Extend the OCR provider module (`src/data/sourcing/ocr_provider.py`) with an `OcrResult` frozen dataclass carrying confidence scores, an `estimate_ocr_confidence()` heuristic, and an `OcrFallbackChain` that tries backends in priority order and returns the highest-confidence result.

## Context

The existing OCR module selects a single backend at import time via `_select_backend()`. If that backend fails (e.g., Tesseract returns garbled text from a scanned PDF, or DeepSeek model loading fails), there is no recovery path. Additionally, there is no way to assess OCR output quality — the caller gets raw text with no signal about whether it's usable.

Three approaches were considered:

1. **Retry with same backend** — simple but doesn't help when the backend consistently produces poor results for certain document types
2. **Fallback chain with confidence scoring** — try backends in order, score each result, return the best one or stop early when confidence exceeds threshold
3. **Ensemble voting** — run all backends, vote on character-level agreement — accurate but expensive (runs every backend on every image)

Approach 2 was chosen because:
- Graceful degradation: if the primary backend fails or produces garbage, the next backend gets a chance
- Confidence scoring provides a quality signal to downstream consumers (e.g., `DocumentRecord.ocr_confidence`)
- Early stopping via `confidence_threshold` avoids running slow backends when the fast one succeeds
- The chain itself satisfies the `OcrBackend` Protocol, so existing call sites work unchanged

## Architecture

```
OcrResult(frozen)
    text: str
    confidence: float  # [0.0, 1.0]
    backend_name: str

estimate_ocr_confidence(text) -> float
    Heuristic: 40% alpha ratio + 30% word density + 30% length score

OcrFallbackChain
    backends: list[OcrBackend]
    confidence_threshold: float = 0.6

    extract_text(image) -> str           # OcrBackend Protocol
    extract_text_with_confidence(image) -> OcrResult

    For each backend:
        1. Try extract_text()
        2. Score with estimate_ocr_confidence()
        3. Keep if higher confidence than current best
        4. Stop early if confidence >= threshold
```

### Integration

- `OcrFallbackChain` satisfies `OcrBackend` Protocol (has `.name` and `.extract_text()`)
- `document_ingestion.py` uses `extract_text_with_confidence()` to get both text and confidence
- `DocumentRecord` gains `ocr_confidence: float` field for downstream quality filtering

## Consequences

- **Resilient** — OCR failures no longer produce empty results when another backend is available
- **Quality signal** — confidence score enables downstream filtering of low-quality OCR output
- **Backward compatible** — `OcrFallbackChain` satisfies `OcrBackend` Protocol; existing `ocr_image()` API unchanged
- **Minimal overhead** — early stopping avoids running unnecessary backends
- **Heuristic confidence** — the scoring is approximate (no ground truth), but sufficient for ranking backends and flagging garbage output

## Files

- `src/data/sourcing/ocr_provider.py` — add `OcrResult`, `estimate_ocr_confidence()`, `OcrFallbackChain`
- `tests/data/sourcing/test_ocr_provider.py` — add tests for new types
- `src/data/sourcing/document_ingestion.py` — wire chain, add `ocr_confidence` to `DocumentRecord`
