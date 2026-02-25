# Phi Forecasting Integration Design

**Date:** 2026-02-25
**Status:** Approved
**Approach:** Layered Forecast Stack (Approach A)
**Integration Mode:** Git submodule (Dignity-Model)

---

## 1. Goal

Integrate Dignity-Model's forecasting toolkit (CNN+LSTM+Attention backbone, signal processor, data pipeline, training engine) into wave-experiment to enable **Phi trajectory forecasting** — predicting how community welfare scores evolve over time.

### Four forecast targets (layered, bottom-up)

1. **Phi trajectory**: Predict scalar Phi(t+1...t+k) from construct time-series
2. **Construct-level**: Predict individual construct trajectories (8 channels)
3. **Hypothesis urgency**: Predict which investigation hunches become high-priority
4. **Document gap prediction**: Predict where information gaps will emerge

### Data strategy

- **Phase 1 (now):** Synthetic Phi trajectory generator for training and validation
- **Phase 2 (future):** Document-derived time-series once document ingestion pipeline is wired

---

## 2. Architecture

### Directory structure

```
src/forecasting/                    ← NEW package
├── __init__.py
├── synthetic.py                    ← Synthetic Phi trajectory generator
├── signals.py                      ← Adapter: Dignity SignalProcessor → construct signals
├── pipeline.py                     ← PhiPipeline: scaling + windowing
├── model.py                        ← PhiForecaster: backbone + forecast heads
├── engine.py                       ← Training loop wrapper
├── phi_trajectory.py               ← Layer 1: Scalar Phi forecast
├── construct_forecast.py           ← Layer 2: 8-channel construct predictions
├── urgency_forecast.py             ← Layer 3: Hypothesis urgency prediction
└── gap_prediction.py               ← Layer 4: Document gap prediction

Dignity-Model/                      ← Git submodule (read-only)
├── core/signals.py                 ← imported by src/forecasting/signals.py
├── data/pipeline.py                ← patterns reused by src/forecasting/pipeline.py
├── models/backbone/hybrid.py       ← imported by src/forecasting/model.py
├── models/head/forecast.py         ← imported by src/forecasting/model.py
└── train/engine.py                 ← imported by src/forecasting/engine.py
```

### Data flow

```
Synthetic scenarios (or future document-derived data)
  → PhiPipeline: compute signals → scale → window → [batch, seq_len, 34]
    → PhiForecaster: DignityBackbone → ForecastHead(s)
      → Layer 1: Phi(t+1...t+k) scalar predictions
      → Layer 2: construct_i(t+1...t+k) per-construct predictions
      → Layer 3: urgency(hypothesis_j, t+1) via forecasted construct gradients
      → Layer 4: gap_probability(type, t+1) via forecasted construct trajectories
```

### Import pattern

`src/forecasting/` imports from `Dignity-Model/` via the git submodule. All adaptation (renaming features, adding Phi-specific logic) happens in `src/forecasting/`, keeping Dignity untouched.

---

## 3. Component Details

### 3.1 Synthetic Data Generator (`synthetic.py`)

Generates community scenarios as 8-channel time-series. Each scenario is a named archetype:

| Scenario | Trajectory Pattern | What it tests |
|---|---|---|
| `stable_community` | All 8 constructs ~0.5 with low noise | Baseline — Phi stays flat |
| `capitalism_suppresses_love` | `lam_L` declines 0.6→0.1 over 200 steps | Love collapse → curiosity collapse → Phi drops |
| `surveillance_state` | `xi` rises 0.3→0.9 while `lam_L` drops 0.6→0.1 | Truth without love = surveillance; divergence penalty fires |
| `willful_ignorance` | `lam_L` rises while `xi` drops | Love without truth; curiosity collapses differently |
| `recovery_arc` | All drop to floors, then `lam_L` recovers first | Tests recovery-aware floor + community multiplier |
| `sudden_crisis` | `kappa` and `lam_P` spike down at t=100 | Crisis detection and recovery forecasting |
| `slow_decay` | All 8 decline at different rates | Multi-construct collapse |
| `random_walk` | Correlated random walks with mean-reversion | General forecasting difficulty |

Output: DataFrame with columns `[t, c, kappa, j, p, eps, lam_L, lam_P, xi]`, 200-500 timesteps per scenario. Generator also computes ground-truth Phi using `compute_phi()`.

### 3.2 Signal Processing (`signals.py`)

Adapter over Dignity's `SignalProcessor`:

| Dignity signal | Phi construct signal | Purpose |
|---|---|---|
| `volatility(prices)` | `volatility(construct_values)` | Rolling std per construct |
| `price_momentum(prices)` | `construct_momentum(values)` | Rate of change per construct |
| `directional_change(prices)` | `construct_direction(values)` | Significant shift detection |
| `regime_detection(vol)` | `welfare_regime(construct_vol)` | Calm/normal/crisis classification |

Phi-specific signals (not in Dignity):
- `synergy_signal(pair_a, pair_b)` — rolling geometric mean for each construct pair
- `divergence_signal(pair_a, pair_b)` — rolling squared difference for penalty pairs
- `phi_derivative(phi_series)` — rate of change of Phi itself

### 3.3 Pipeline (`pipeline.py`)

`PhiPipeline` adapting Dignity's `TransactionPipeline`:

**Input features (34 total):**
- 8 raw construct values
- 8 construct volatilities (rolling std)
- 8 construct momentums (rate of change)
- 5 synergy signals (4 primary pairs + 1 curiosity cross-pair)
- 4 divergence signals (4 penalty pairs, curiosity included in synergy)
- 1 Phi value (computed from constructs)

**Methods:** `fit()` → `transform()` → `create_sequences()` — same API as Dignity's pipeline, with Phi-specific feature computation.

### 3.4 Model (`model.py`)

`PhiForecaster` composing Dignity's backbone with forecast heads:

```python
class PhiForecaster(nn.Module):
    def __init__(self, input_size=34, hidden_size=256, n_layers=2, pred_len=10):
        self.backbone = DignityBackbone(input_size, hidden_size, n_layers)
        self.phi_head = ForecastHead(hidden_size, pred_len, num_features=1)
        self.construct_head = ForecastHead(hidden_size, pred_len, num_features=8)
```

**Multi-task loss:** `L_total = L_phi + alpha * L_construct` (both MSE). Uses Dignity's `train_epoch()` with AMP.

### 3.5 Layers 3-4 (Symbolic Composition)

These layers don't train new models. They compose numerical forecasts with existing welfare scoring:

**Layer 3 (`urgency_forecast.py`):**
```python
def forecast_hypothesis_urgency(hypothesis, model, current_series):
    future_constructs = model.predict_constructs(current_series)
    future_metrics = {c: future_constructs[i] for i, c in enumerate(ALL_CONSTRUCTS)}
    return score_hypothesis_welfare(hypothesis, future_metrics)
```

**Layer 4 (`gap_prediction.py`):**
```python
def predict_gap_emergence(model, current_series, threshold=0.3):
    future_constructs = model.predict_constructs(current_series)
    at_risk = [c for c, val in future_constructs.items() if val < CONSTRUCT_FLOORS[c]]
    return at_risk
```

---

## 4. Testing Strategy

| Component | Test type | Validates |
|---|---|---|
| Synthetic | Unit | Shape, Phi computable, construct bounds [0,1] |
| Pipeline | Unit | Signals correct, scaling invertible, sequence shapes |
| Model | Unit | Forward pass shapes, gradient flow, multi-task loss |
| Layer 1 | Integration | Learns constant Phi, then linear trend |
| Layer 2 | Integration | Recovers construct trajectories from noisy input |
| Layer 3 | Integration | Forecast urgency > current urgency when declining |
| Layer 4 | Integration | At-risk constructs flagged below floors |
| Regression | Full suite | Existing 421+ tests still pass |

---

## 5. Deferred

- Document-derived time-series (requires document ingestion pipeline)
- Dual-backbone fusion (Approach C — wait for document ingestion)
- ONNX export (deployment, not needed now)
- Real-world calibration (requires actual community data)

---

## 6. Dignity Modules Used

| Module | Used for | Modified? |
|---|---|---|
| `core/signals.py` | Volatility, momentum, regime detection | Yes — add Phi-specific signals (synergy, divergence, phi_derivative) |
| `data/pipeline.py` | Scaling + windowing patterns | Yes — add PhiPipeline subclass or config mode |
| `models/backbone/hybrid.py` | CNN+LSTM+Attention encoder | No — works as-is for 34-feature input |
| `models/head/forecast.py` | Multi-step prediction head | No — works as-is for multi-channel output |
| `train/engine.py` | AMP training loop | Yes — add multi-task loss support (L_phi + alpha * L_construct) |

**Both repos updated in parallel.** Dignity-Model is not read-only — Phi-specific extensions are added directly to Dignity's modules where it makes sense, keeping the adapter layer in `src/forecasting/` thin.

Not used: `core/privacy.py`, `data/source/crypto.py`, `models/head/risk.py`, `models/head/policy.py`, `export/to_onnx.py`.

---

## 7. Success Criteria

1. Synthetic generator produces all 8 scenarios with correct construct trajectories
2. PhiPipeline transforms raw constructs into 34-feature windowed sequences
3. PhiForecaster trains on synthetic data and converges (loss decreasing)
4. Layer 1: Phi forecast MSE < 0.05 on stable_community (trivial baseline)
5. Layer 2: Construct forecast per-channel MSE < 0.1 on slow_decay
6. Layer 3: Urgency forecast correctly predicts rising urgency for declining scenarios
7. Layer 4: Gap prediction flags constructs approaching floors 5+ timesteps ahead
8. All existing tests pass (0 regressions)
