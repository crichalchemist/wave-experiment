# Phi Research Workbench: Evolving maninagarden

**Goal:** Transform the `crichalchemist/maninagarden` HF Space from a static forecast demo into a full research workbench — training orchestration, experiment comparison, data curation, and visual research showcase — while fixing the formula drift between three divergent copies of `compute_phi()`.

**Architecture:** Modular Space (6 files) on `cpu-upgrade`, launching GPU training via HF Jobs, with password-gated admin tabs and versioned checkpoints on Hub.

---

## 1. The Drift Problem

Three files implement the Phi welfare formula independently:

| Component | `welfare_scoring.py` | `train_phi_hf_job.py` | `app.py` (Space) |
|-----------|:---:|:---:|:---:|
| `recovery_aware_input()` | Defined, **NOT called** by compute_phi | Missing | Missing |
| `equity_weights()` | Modular function | Inlined in compute_phi | Inlined in compute_phi |
| `community_multiplier()` | Modular function | Inlined | Inlined |
| `ubuntu_synergy()` | Modular function | Inlined | Inlined |
| `divergence_penalty()` | Modular function | Inlined | Inlined |
| `compute_phi()` | Full, but skips recovery floors | Flattened inline (~20 lines) | Verbatim copy of training script |
| Constants (ETA, MU, etc.) | All present | All present (duplicated) | All present (duplicated) |

**Functional drift:** All three compute the same result today. But `recovery_aware_input()` — the theoretical advance where community capacity (lam_L) mediates recovery for constructs below their floors — exists only as dead infrastructure. It was written, tested, and documented, but never wired into the actual Phi computation.

**Structural drift:** The training script and Space flatten everything into a single `compute_phi()` function. When the theory evolves (as it has twice — formula revision, curiosity coupling), each copy must be updated independently. This is how drift accumulates.

### Refocusing Strategy

1. **`welfare.py` in the Space** becomes the single inline source of truth for both inference and data generation
2. The training script reads its formula from `welfare.py` passed inline (or duplicates it with a version tag)
3. Each checkpoint's `training_metadata.json` records the formula version and constants, so the Experiment Lab can show what formula produced each model
4. `src/inference/welfare_scoring.py` remains the production module for the detective-llm system — it has additional functions (gradient computation, hypothesis scoring, gap urgency) that the Space doesn't need

---

## 2. File Structure

```
spaces/maninagarden/
├── app.py              # UI shell — 6 tabs, event wiring (~350 lines)
├── model.py            # PhiForecasterGPU architecture + checkpoint loading (~120 lines)
├── welfare.py          # Phi formula — recovery floors, curiosity, equity weights (~130 lines)
├── scenarios.py        # 8 scenarios + signal processing + data generation (~200 lines)
├── training.py         # HF Jobs launcher + Trackio fetcher + checkpoint comparison (~200 lines)
├── requirements.txt    # torch, numpy, pandas, scikit-learn, plotly, huggingface-hub, gradio, trackio
└── README.md           # sdk: gradio, sdk_version: 5.12.0, hardware: cpu-upgrade
```

### What goes where

**`welfare.py`** — Pure functions, no dependencies beyond `math`. Contains:
- Constants: ALL_CONSTRUCTS, CONSTRUCT_FLOORS, ETA, MU, ETA_CURIOSITY, GAMMA, CONSTRUCT_PAIRS, CURIOSITY_CROSS_PAIR, PENALTY_PAIRS
- `_sigmoid()`, `recovery_aware_input()`
- `equity_weights()`, `community_multiplier()`
- `ubuntu_synergy()`, `divergence_penalty()`
- `compute_phi()` — **now calls `recovery_aware_input()`** for each construct before the weighted geometric mean
- Display helpers: CONSTRUCT_DISPLAY_NAMES mapping symbols to human labels

**`model.py`** — PyTorch model classes copied verbatim from training script (required for `load_state_dict(strict=True)`):
- `CNN1D`, `StackedLSTM`, `AdditiveAttention`, `PhiForecasterGPU`
- `load_model(repo_id, filename)` — downloads checkpoint, instantiates model, loads weights
- `load_model_version(repo_id, revision)` — loads a specific commit/tag for comparison

**`scenarios.py`** — Data generation pipeline:
- `SCENARIOS` dict (8 scenario definitions)
- `generate_scenario()` — produces 200-step construct trajectories
- Signal processing: `volatility()`, `price_momentum()`, `synergy_signal()`, `divergence_signal()`, `compute_all_signals()`
- `prepare_input()` — full pipeline: generate → signals → scale → window → tensor

**`training.py`** — HF Jobs integration (password-gated):
- `launch_training_job(config)` — calls `huggingface_hub.run_uv_job()` with updated training script
- `get_training_script(config)` — generates the training script string with configurable hyperparams
- `list_checkpoints(repo_id)` — lists available checkpoints from Hub
- `load_training_metadata(repo_id, revision)` — fetches training_metadata.json for a checkpoint
- `fetch_trackio_metrics(project)` — retrieves metrics from Trackio for display

**`app.py`** — Gradio UI shell:
- Imports from the four modules above
- 6 tabs wired to handler functions
- Password gate for Training and Data Workshop tabs
- `demo.load()` auto-runs default scenario on page load

---

## 3. Tab Architecture

### Tab 1: Scenario Explorer (existing, updated)
- **Input:** Dropdown (8 scenarios) + seed slider
- **Action:** `scenarios.prepare_input()` → `model.forward()` → plot
- **Output:** Phi trajectory (historical + forecast), 8 construct forecasts, attention heatmap
- **Update:** Uses `welfare.compute_phi()` for the historical Phi line (now includes recovery-aware floors)

### Tab 2: Custom Forecast (existing, updated)
- **Input:** 8 sliders (minimums from CONSTRUCT_FLOORS)
- **Action:** Generate constant-level history + noise → same pipeline
- **Output:** Same plots as Tab 1
- **Update:** Slider minimums still from CONSTRUCT_FLOORS

### Tab 3: Experiment Lab (new)
- **Input:** Two dropdowns listing available checkpoints (by date/commit), scenario selector
- **Action:** Load both checkpoints via `model.load_model_version()`, run same scenario through each
- **Output:**
  - Overlaid Phi forecast lines (Checkpoint A vs B)
  - Construct-by-construct comparison (8 small subplots)
  - Metadata comparison table (epochs, lr, val_loss, formula version)
- **Purpose:** Answer "did retraining improve predictions?" and "what changed?"

### Tab 4: Training (new, password-gated)
- **Input:** Password field + hyperparameter controls:
  - Epochs slider (10-500, default 100)
  - Learning rate slider (1e-5 to 1e-2, log scale, default 5e-4)
  - Hidden size dropdown (64, 128, 256, 512)
  - Batch size dropdown (16, 32, 64, 128)
  - Scenarios per type slider (10-200, default 50)
  - Hardware dropdown (t4-small, a10g-small, a10g-large)
- **Action:** `training.launch_training_job()` → returns job ID + monitoring URL
- **Output:** Job status display, Trackio metrics chart (train/val loss over epochs)
- **Gate:** Controls hidden until correct ADMIN_KEY entered

### Tab 5: Data Workshop (new, password-gated)
- **Input:** Scenario selector + noise/parameter adjustments
- **Output:**
  - Raw 200-step trajectory plot (8 construct lines)
  - Signal processing output (36 features heatmap)
  - RobustScaler transform visualization (before/after)
  - Phi trajectory computed from `welfare.compute_phi()` with recovery-aware floors visible
- **Purpose:** Inspect what the model trains on. See how recovery floors affect the training signal.
- **Gate:** Same ADMIN_KEY as Training tab

### Tab 6: Research (existing, updated)
- **Content:** Markdown blocks with:
  - The full Phi formula (updated to show recovery-aware floors)
  - 8 construct definitions with floors
  - Construct pairs including curiosity cross-pair (lam_L x xi)
  - Divergence penalty interpretations (paternalism, surveillance, willful ignorance)
  - Key citations (hooks, Sen, Fricker, Collins, Metz, Ramose)
- **Update:** Formula text now includes recovery-aware floor explanation and curiosity coupling

---

## 4. Retraining Pipeline

### Step 1: Update training script formula

The training script's inline `compute_phi()` must be updated to include `recovery_aware_input()`. This means:

```python
# Before computing weighted geometric mean, apply recovery floors:
for c in ALL_CONSTRUCTS:
    floor_i = CONSTRUCT_FLOORS.get(c, 0.10)
    dx_dt_i = ... # compute from trajectory
    lam_L_val = metrics.get("lam_L", 0.5)
    x_tilde_i = recovery_aware_input(metrics[c], floor_i, dx_dt_i, lam_L_val)
    # use x_tilde_i instead of raw metrics[c] in the product
```

The `dx_dt_i` for each construct comes from the finite difference of the generated trajectory — the same data we already have. This is not a new input to the model; it changes how the *target* Phi values are computed during data generation.

### Step 2: Launch training job

From the Space's Training tab (or CLI):
- Hardware: `t4-small` for validation (~30 min), `a10g-small` for production (~1-2 hrs)
- Timeout: 2h with buffer
- Secrets: `HF_TOKEN` for checkpoint push
- The script is passed inline via `run_uv_job()` with all code self-contained

### Step 3: Checkpoint versioning

Each training run pushes to `crichalchemist/phi-forecaster`:
- `phi_forecaster_best.pt` — model weights
- `training_metadata.json` — hyperparams + formula version:
  ```json
  {
    "formula_version": "2.1-recovery-floors",
    "constants": {"ETA": 0.10, "MU": 0.15, "ETA_CURIOSITY": 0.08, "GAMMA": 0.5},
    "has_recovery_floors": true,
    "epochs": 100,
    "best_val_loss": 0.0042,
    "timestamp": "2026-02-26T..."
  }
  ```
- Git commit message includes formula version for Experiment Lab lookup

### Step 4: Space auto-updates

After retraining, the Space loads the latest checkpoint by default. The Experiment Lab can compare old (pre-recovery) and new (post-recovery) checkpoints side by side.

---

## 5. Authentication

| Secret | Purpose | Where used |
|--------|---------|------------|
| `HF_TOKEN` | Launch HF Jobs, push checkpoints, list Hub revisions | `training.py` (server-side only) |
| `ADMIN_KEY` | Unlock Training + Data Workshop tabs | `app.py` (password input comparison) |

- Scenario Explorer, Custom Forecast, Experiment Lab, Research: **public**
- Training, Data Workshop: **password-gated**
- HF_TOKEN never exposed to UI — read from `os.environ["HF_TOKEN"]`
- ADMIN_KEY stored as Space secret, compared server-side

---

## 6. Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.0.0
huggingface-hub>=0.25.0
gradio>=4.0.0
trackio
```

Only addition from current requirements.txt: `trackio` for fetching training metrics.

---

## 7. README.md Changes

```yaml
---
title: Phi Forecaster
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
models:
  - crichalchemist/phi-forecaster
short_description: "Welfare trajectory forecasting with Φ(humanity)"
---
```

Hardware upgrade from `cpu-basic` to `cpu-upgrade` configured in Space settings (not in README).

---

## 8. What This Design Does NOT Include

- **OAuth login** — kept simple with Space secret passphrase
- **Multi-user support** — single-researcher workbench
- **Automated retraining** — no scheduled jobs or webhooks (can add later)
- **Model architecture changes** — PhiForecasterGPU stays the same, only training data changes
- **New constructs** — 8-construct taxonomy is stable
- **Production welfare_scoring.py changes** — that module has additional detective-llm functions (gradient, hypothesis scoring, gap urgency) that are orthogonal to this Space. The Space's `welfare.py` is a focused subset.
