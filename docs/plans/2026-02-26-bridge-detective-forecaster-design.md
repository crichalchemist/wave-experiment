# Bridge Design: Detective LLM + Phi Forecaster Integration

**Goal:** Connect the two independent training tracks — Detective LLM (gap detection, hypothesis evolution) and Phi Forecasting (welfare trajectory prediction) — through a three-layer sequential pipeline that creates a data flywheel between them.

**Problem:** Both tracks use the Phi welfare function but independently. The detective uses Phi gradients to prioritize hypotheses. The forecaster predicts Phi trajectories from synthetic scenarios. Neither informs the other. The welfare classifier (DistilBERT) that both need has training data ready but was never trained.

**Architecture:** Three layers, each independently useful, each building on the previous:

```
Layer 1: Welfare Classifier        → shared foundation
Layer 2: Forecast-Informed Scoring → forecaster informs detective
Layer 3: Detective-Generated Scenarios → detective informs forecaster
```

---

## Layer 1: Welfare Classifier

Train `distilbert-base-uncased` on 991 welfare-labeled examples for 8-construct multi-label regression. This replaces the keyword fallback in both tracks with semantic understanding.

**Training:**
- Data: `data/training/welfare_training_split_train.jsonl` (791 examples), `welfare_training_split_val.jsonl` (198 examples)
- Task: Multi-label regression — predict [0,1] scores for c, kappa, j, p, eps, lam_L, lam_P, xi
- Config: 3 epochs, lr=2e-5, batch=16, FP16 on GPU
- Metric: Per-construct MAE, target < 0.20
- Compute: HF Jobs (t4-small, ~5 min)
- Output: Push to `crichalchemist/welfare-constructs-distilbert` on Hub

**Wiring into welfare_scoring.py:**
- `get_construct_scores(text)` already has try/except that falls back to keywords when the classifier is unavailable
- `infer_threatened_constructs(text)` uses scores >= 0.3 threshold — stays the same, fed by real predictions
- `src/inference/welfare_classifier.py` updated to load from Hub instead of local path

**Success criterion:** MAE < 0.20 on val set. If higher, need more training data before Layer 2.

---

## Layer 2: Forecast-Informed Hypothesis Scoring

When the detective finds a gap, the forecaster predicts what happens if current construct levels persist. Declining trajectories increase urgency.

**Mechanism:**
1. Detective finds gap → `infer_threatened_constructs()` returns threatened constructs
2. Current Phi metrics (from classifier) give construct levels
3. Build 200-step constant-level scenario from those levels
4. PhiForecasterGPU predicts 10-step Phi trajectory
5. Compute trajectory urgency: `slope = (phi[-1] - phi[0]) / len(phi)`, normalized to [0,1] where 1.0 = steepest decline

**Updated combined_score:**
```
combined_score = 0.45*confidence + 0.25*welfare_relevance
               + 0.15*curiosity_relevance + 0.15*trajectory_urgency
```
Weights sum to 1.0. Confidence stays dominant. Trajectory urgency is the new 4th component.

**Implementation:**
- New function: `score_hypothesis_trajectory(hypothesis, phi_metrics)` in `welfare_scoring.py`
- New field: `Hypothesis.trajectory_urgency: float = 0.0`
- `combined_score()` gains delta parameter (default 0.15), alpha/beta/gamma rebalanced
- Forecaster loaded lazily on first call, cached. CPU inference only (<100ms).

**Key design decision:** Synergy and penalty in compute_phi operate on raw metrics (detect actual state). Trajectory urgency also operates on raw metrics — it forecasts from the real situation, not the recovery-adjusted view.

---

## Layer 3: Detective-Generated Scenarios

Run A/B/C modules on real text to extract welfare construct patterns, convert them into new training scenarios for the forecaster.

**Source:** `data/training/smiles_and_cries_extracted.txt` (2.3MB, ~5,774 lines)

**Pipeline:**
1. **Extract construct profiles:** Split corpus into ~500-word chunks → welfare classifier scores each → A/B/C modules tag assumptions → result: time-ordered sequence of construct snapshots with assumption types
2. **Identify trajectory patterns:** Group consecutive chunks by dominant pattern (e.g., "lam_L declining while xi stays high" = surveillance). Each group becomes a named scenario template with start/end construct levels, dominant assumption type, and a descriptive label.
3. **Generate synthetic trajectories:** Interpolate between start/end levels over 200 steps, add noise calibrated to observed variance, compute Phi with recovery-aware floors + derivatives. These extend beyond the 8 hand-designed scenarios.
4. **Retrain forecaster:** Original 8 scenarios + N extracted scenarios → HF Jobs → push checkpoint to Hub → Space loads richer model.

**New module:** `src/inference/scenario_extraction.py`
- `extract_construct_profiles(corpus_path, classifier, modules)` → list of construct profiles
- `identify_trajectory_patterns(profiles)` → list of scenario templates
- `generate_from_template(template, length=200, rng=None)` → DataFrame

**Updated Space:**
- `scenarios.py` imports extracted scenarios alongside hand-designed ones
- Scenario Explorer dropdown expands from 8 to 8+N
- Data Workshop shows extracted patterns and their source text

**Success criterion:** Retrained forecaster val_loss <= 0.000196 (current). If higher, filter extracted scenarios by confidence.

**Risk:** Single source corpus (memoirs) produces narrow patterns. Acceptable for v1 — the pipeline infrastructure is the deliverable, more corpora can be added.

---

## The Data Flywheel

Once all three layers are running:

```
Real text → Welfare classifier → construct scores
         → A/B/C modules → assumption detections
         → Scenario extraction → new training scenarios
         → Forecaster retrains on richer data
         → Forecaster predicts trajectories
         → Detective uses predictions for hypothesis urgency
         → Detective finds new patterns in new documents
         → Back to scenario extraction
```

Each cycle enriches both tracks. The forecaster learns from increasingly realistic patterns. The detective prioritizes increasingly well-informed by trajectory predictions.

---

## Space Integration

The maninagarden Space remains the visual interface for both tracks:

| Tab | Layer 1 | Layer 2 | Layer 3 |
|-----|---------|---------|---------|
| Scenario Explorer | No change | No change | Dropdown expands with extracted scenarios |
| Custom Forecast | No change | No change | No change |
| Experiment Lab | No change | No change | Compare pre/post-extraction checkpoints |
| Training | Classifier training button | No change | Expanded training with extracted scenarios |
| Data Workshop | Classifier predictions vs keywords | No change | Extracted patterns + source text |
| Research | No change | Trajectory urgency explained | Flywheel diagram |

---

## Dependencies

Layer 1 has no dependencies — it uses existing data and existing code patterns.

Layer 2 depends on Layer 1 (classifier provides construct scores for scenario generation).

Layer 3 depends on Layer 1 (classifier scores text chunks) and Layer 2 (forecaster is the training target).

---

## What This Does NOT Include

- Completing Stages 2-5 of the Detective training curriculum (legal DPO, SFT, GRPO, DetectiveGPT). Those remain independent work.
- New model architectures. PhiForecasterGPU and DistilBERT stay as-is.
- Real-time integration. The flywheel is batch-oriented (retrain periodically, not continuously).
- Multi-corpus support. V1 uses smiles_and_cries only. Extending to legal/FOIA corpora is future work.
