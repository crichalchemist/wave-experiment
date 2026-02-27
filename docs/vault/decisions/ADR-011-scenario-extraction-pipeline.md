---
id: ADR-011
title: Detective-generated scenario extraction for forecaster training
status: accepted
date: 2026-02-27
tags: [architecture, bridge, scenarios, training, flywheel]
---

# ADR-011: Detective-Generated Scenario Extraction Pipeline

## Decision

Real text corpora are processed through the welfare classifier and A/B/C modules to extract welfare construct trajectory patterns. These patterns become synthetic training scenarios for the Phi Forecaster, extending beyond the 8 hand-designed scenarios. This closes the data flywheel: detective findings enrich the forecaster, whose predictions inform the detective.

## Context

Layer 3 of the Detective-Forecaster bridge. The Phi Forecaster trained on 8 hand-designed scenarios (stable_community, capitalism_suppresses_love, surveillance_state, etc.). These capture archetypal patterns but miss the nuance in real-world text. The detective's A/B/C modules and welfare classifier can extract construct-level patterns from actual corpora — patterns like "love declining while truth stays high" (surveillance emerging) that the hand-designed scenarios may not cover.

## Pipeline

```
Corpus text
  → chunk into 500-word segments
  → welfare classifier scores each chunk (8 constructs)
  → identify trajectory patterns (consecutive chunks where a construct changes >= 0.15)
  → deduplicate by construct+direction (one "declining_lam_L" pattern)
  → generate synthetic trajectories (interpolate start→end with noise)
  → compute Phi at each timestep
  → feed into forecaster training alongside original 8 scenarios
```

## Key design choices

- **500-word chunks**: matches DistilBERT's 512-token limit while capturing enough context for welfare scoring
- **Trailing chunk cutoff**: chunks < 25% of chunk_size are discarded (too short for reliable scoring)
- **Min run length = 3**: a pattern needs at least 3 consecutive chunks to be real, not noise
- **Change threshold = 0.15**: construct must change by at least 0.15 total across the run
- **Deduplication by label**: only one pattern per construct-direction pair (e.g., one "declining_lam_L")
- **No derivatives in src compute_phi**: `welfare_scoring.py:compute_phi()` doesn't accept derivatives. The Space's `welfare.py` does. Scenarios generated in src use the simpler formula; recovery-aware floors activate during forecaster training in the Space.

## The data flywheel

Once all three layers run:
1. Real text → classifier → construct scores
2. A/B/C modules → assumption detections
3. Scenario extraction → new training scenarios
4. Forecaster retrains on richer data
5. Forecaster predicts trajectories → trajectory urgency
6. Detective uses urgency to prioritize hypotheses
7. Detective finds new patterns → back to step 1

## Consequences

- `src/inference/scenario_extraction.py` — three core functions + `run_extraction_pipeline()`
- `detective extract-scenarios <corpus>` CLI command writes templates to JSON
- `spaces/maninagarden/scenarios.py` gains `load_extracted_scenarios()` and `generate_extracted_scenario()`
- Space training script (`training.py`) includes extracted scenarios alongside hand-designed ones
- Extracted scenario JSON lives at `spaces/maninagarden/extracted_scenarios.json` (empty until first extraction run)
- V1 single-corpus limitation acknowledged: only `smiles_and_cries_extracted.txt` for now

## Files

- `src/inference/scenario_extraction.py` — extraction pipeline
- `src/cli/main.py` — `extract-scenarios` command
- `spaces/maninagarden/scenarios.py` — extracted scenario loading + generation
- `spaces/maninagarden/training.py` — f-string template includes extracted scenarios
- `tests/inference/test_scenario_extraction.py` — 10 tests
- `tests/cli/test_extract_command.py` — 4 tests
- `tests/spaces/test_extracted_scenarios.py` — 4 tests
- `tests/spaces/test_training_with_extracted.py` — 2 tests
