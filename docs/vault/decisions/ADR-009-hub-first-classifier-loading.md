---
id: ADR-009
title: Hub-first welfare classifier loading with local fallback
status: accepted
date: 2026-02-27
tags: [architecture, classifier, hub, deployment]
---

# ADR-009: Hub-First Welfare Classifier Loading

## Decision

The welfare construct classifier (`src/inference/welfare_classifier.py`) loads from Hugging Face Hub first (`crichalchemist/welfare-constructs-distilbert`), falling back to a local directory (`models/welfare-constructs-distilbert/`) if the Hub is unreachable. If both fail, `get_construct_scores()` returns zero scores — the keyword fallback in `welfare_scoring.py` handles the rest.

## Context

Layer 1 of the Detective-Forecaster bridge trains the welfare classifier on HF Jobs and pushes the trained model to the Hub. Both the detective (`src/`) and the forecaster Space (`spaces/maninagarden/`) need to load it. A Hub-first approach means:

1. **No manual model distribution** — trained model is immediately available everywhere
2. **Offline development works** — local path fallback means you can work without internet
3. **Graceful degradation** — zero scores + keyword fallback means the system never crashes, just loses semantic precision

The `@lru_cache(maxsize=1)` pattern means the Hub/local decision happens exactly once per process. Each test must `cache_clear()` to avoid state leakage.

## Training pipeline

Training data lives on Hub at `crichalchemist/welfare-training-data` (791 train, 198 val examples). Training runs via `scripts/train_welfare_classifier_hf_job.py` which generates a self-contained PEP 723 script and launches it on HF Jobs (t4-small). The trained model is pushed to `crichalchemist/welfare-constructs-distilbert`.

Achieved metrics: **MAE 0.164** (target < 0.20). Per-construct MAE: c=0.190, kappa=0.129, j=0.139, p=0.131, eps=0.134, lam_L=0.186, lam_P=0.199, xi=0.189.

## Consequences

- `HUB_MODEL_ID = "crichalchemist/welfare-constructs-distilbert"` is the single source of truth for the model location
- First call to `get_construct_scores()` incurs a ~2s Hub download (cached locally by huggingface_hub after first download)
- `welfare_scoring.py:infer_threatened_constructs()` and `get_construct_scores()` both delegate to the classifier, with keyword fallback if it returns all zeros
- Tests mock `pipeline()` to avoid real model downloads

## Files

- `src/inference/welfare_classifier.py` — Hub-first loading logic
- `scripts/train_welfare_classifier_hf_job.py` — HF Jobs training script generator + launcher
- `tests/inference/test_welfare_classifier.py` — 10 tests including Hub loading
