---
id: ADR-003
title: DistilBERT independent classifiers for Modules A/B/C
status: accepted
date: 2026-02-16
tags: [architecture, classification, distilbert, modules]
---

# ADR-003: Independent DistilBERT Classifiers for Modules A/B/C

## Decision

Modules A (cognitive bias), B (historical determinism), C (geopolitical presumption) are independent DistilBERT classifiers loaded via `@lru_cache(maxsize=1)`, NOT heads on the generative model.

## Context

The original plan attached bias/assumption detection as heads on DetectiveGPT. The Cyberbullying Detection paper showed DistilBERT achieves 95% accuracy at 0.053s/35MB RAM for text classification. Three benefits drove the change:

1. **Separable training** — classifiers can be updated without retraining the generative backbone
2. **Faster inference** — 0.053s vs. a full generative forward pass
3. **Independent deployment** — each classifier can be swapped for a domain-specific fine-tune

## Implementation notes

- Lazy loading via `@lru_cache(maxsize=1)` — model loads only on first call, not at import time
- Module-level `_MODEL_NAME` constant marks the placeholder checkpoint; replace with fine-tuned checkpoint when available
- `BiasDetection.source_text` stores the full input (not an extracted span — DistilBERT text-classification does not return offsets)
- `_CONFIDENCE_THRESHOLD = 0.7` — detections below this are discarded

## Files

- `src/detective/module_a.py` — implemented
- `src/detective/module_b.py` — planned (historical determinism)
- `src/detective/module_c.py` — planned (geopolitical presumption)
