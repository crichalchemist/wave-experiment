---
id: ADR-010
title: Forecast-informed trajectory urgency in hypothesis scoring
status: accepted
date: 2026-02-27
tags: [architecture, hypothesis, forecaster, bridge, scoring]
---

# ADR-010: Forecast-Informed Trajectory Urgency

## Decision

Hypothesis scoring gains a 4th component: `trajectory_urgency`. The Phi Forecaster predicts whether current welfare construct levels lead to decline. Declining trajectories increase hypothesis urgency. The `combined_score` formula becomes:

```
combined_score = 0.45*confidence + 0.25*welfare_relevance
               + 0.15*curiosity_relevance + 0.15*trajectory_urgency
```

Default `delta=0.0` preserves backward compatibility — the 4-weight formula only activates when `phi_metrics` is provided to `evolve_parallel()`.

## Context

Layer 2 of the Detective-Forecaster bridge. Previously, hypothesis scoring used 3 weights (confidence, welfare, curiosity). The forecaster existed independently — it predicted Phi trajectories but never informed the detective's prioritization.

The key insight: a hypothesis about declining care (c) is more urgent if the forecaster predicts care will *continue* declining. Trajectory urgency answers "where is this heading?" rather than "how bad is it now?"

## How it works

1. `score_hypothesis_trajectory(hypothesis, phi_metrics)` builds a 200-step constant-level scenario from current metrics
2. Runs the `PhiTrajectoryForecaster` to predict 10-step Phi trajectory
3. Computes slope: `(phi[-1] - phi[0]) / len(phi)`
4. Normalizes: `urgency = decline / (decline + k)` where `k=0.02`
5. Rising/stable trajectories return 0.0 — only decline creates urgency

The forecaster loads lazily on first call (`_forecaster_cache` singleton). CPU inference only, <100ms.

## Weight rebalancing

Old weights (3-component): `alpha=0.55, beta=0.30, gamma=0.15`
New weights (4-component): `alpha=0.45, beta=0.25, gamma=0.15, delta=0.15`

Confidence stays dominant (Constitution Principle 1: epistemic honesty). The 0.10 taken from confidence and welfare goes to trajectory urgency.

## Curiosity scoring wired in

As part of drift resolution, `score_hypothesis_curiosity()` is now called in the same `evolve_parallel()` loop as welfare and trajectory scoring. When `phi_metrics` is provided, each evolved hypothesis receives a `curiosity_relevance` value computed from the geometric mean of the lam_L and xi gradients. This completes the 4-weight `combined_score` formula: all four components (confidence, welfare_relevance, curiosity_relevance, trajectory_urgency) are now populated when `phi_metrics` is supplied.

The curiosity call sits between `score_hypothesis_trajectory()` and the `replace()` call, keeping the same pattern as the other welfare scoring calls. No new lazy imports needed since `score_hypothesis_curiosity` is in the same `welfare_scoring` module.

## Consequences

- `Hypothesis.trajectory_urgency: float = 0.0` — new frozen field, validated [0, 1]
- `combined_score(delta=0.0)` is backward-compatible default
- `evolve_parallel()` computes welfare relevance, curiosity relevance, and trajectory urgency when `phi_metrics` provided, uses 4-weight sort
- All tests mock `_get_trajectory_prediction` — no real forecaster in test suite
- Graceful degradation: if forecaster fails, `score_hypothesis_trajectory` returns 0.0

## Files

- `src/detective/hypothesis.py` — `trajectory_urgency` field + `delta` parameter
- `src/inference/welfare_scoring.py` — `score_hypothesis_trajectory()`, `score_hypothesis_curiosity()`, `_get_forecaster()`, `_forecast_from_metrics()`
- `src/detective/parallel_evolution.py` — 4-weight scoring in `evolve_parallel()` (welfare, curiosity, trajectory all wired)
- `tests/inference/test_trajectory_scoring.py` — 5 tests
- `tests/detective/test_parallel_evolution.py` — 3 new tests (trajectory urgency, 4-weight scoring, curiosity populated)
- `tests/integration/test_bridge_pipeline.py` — end-to-end bridge tests
