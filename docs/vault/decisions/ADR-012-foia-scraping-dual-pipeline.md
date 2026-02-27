---
id: ADR-012
title: Scrapling FOIA scraping with dual evidence/training pipeline
status: accepted
date: 2026-02-27
tags: [architecture, scraping, foia, evidence, training, flywheel]
---

# ADR-012: FOIA Scraping with Dual Evidence/Training Pipeline

## Decision

Use Scrapling's adaptive web scraping framework to collect documents from FOIA portals (FBI Vault, NARA, State Dept). Every scraped document feeds two pipelines simultaneously: investigation evidence (welfare scoring + knowledge graph) and model training data (welfare-scored examples for classifier and DPO pair generation).

## Context

The detective needs evidence from declassified government documents. FOIA portals are the primary source but present challenges: layouts change, rate limiting, PDF-heavy content, inconsistent structure across portals. Scrapling's adaptive parser handles layout changes, StealthyFetcher handles rate limiting, and the Spider framework handles paginated crawls with checkpoints.

The key insight: evidence collection IS training data collection. The same document that feeds the detective's investigation also produces welfare-scored training examples for the classifier and DPO preference pairs for model improvement. This eliminates the need for separate data collection pipelines.

## Pipeline

```
FOIA portal → Scrapling Spider → HTML/PDFs
  → document_ingestion.py → MIME detection → OCR → DocumentRecord
  → DUAL OUTPUT:
    1. Evidence: welfare construct scores + threatened constructs + Phi
    2. Training: welfare-scored JSONL for classifier + DPO pair generation
```

## Consequences

- `src/data/sourcing/foia_scraper.py` — FOIAScraper with per-portal Spider configs
- `src/data/sourcing/dual_pipeline.py` — process_for_evidence() + process_for_training()
- `detective scrape-foia` CLI command for portal crawling
- `detective ingest` CLI command for dual pipeline processing
- Scrapling added as optional dependency (`pip install -e ".[scraping]"`)
- Existing `document_ingestion.py` and `ocr_provider.py` handle PDF→text conversion
- V1 limitation: no automated retraining — flywheel is manual (run ingest, then retrain)

## Files

- `src/data/sourcing/foia_scraper.py` — FOIAScraper class
- `src/data/sourcing/dual_pipeline.py` — dual output functions
- `src/data/sourcing/document_ingestion.py` — OCR pipeline (pre-existing)
- `src/data/sourcing/ocr_provider.py` — Tesseract/DeepSeek backends (pre-existing)
- `src/cli/main.py` — scrape-foia + ingest commands
