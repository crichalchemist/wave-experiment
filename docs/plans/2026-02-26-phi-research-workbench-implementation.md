# Phi Research Workbench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evolve the maninagarden HF Space from a monolithic 718-line demo into a modular 6-tab research workbench with training orchestration, experiment comparison, and the latest Phi formula (recovery-aware floors).

**Architecture:** Extract the current `app.py` into 5 focused modules (`welfare.py`, `model.py`, `scenarios.py`, `training.py`, `app.py` shell). Wire `recovery_aware_input()` into `compute_phi()` for the first time. Add 3 new Gradio tabs (Experiment Lab, Training, Data Workshop). Password-gate admin tabs via Space Secret.

**Tech Stack:** Python 3.12, PyTorch, Gradio 4+, Plotly, huggingface-hub, trackio, scikit-learn

**Design doc:** `docs/plans/2026-02-26-phi-research-workbench-design.md`

---

### Task 1: Create `welfare.py` — the single source of truth for the Phi formula

**Files:**
- Create: `spaces/maninagarden/welfare.py`

**Context:** This module contains ALL welfare scoring logic. It is the single source of truth for the Space. It includes `recovery_aware_input()` from `src/inference/welfare_scoring.py` (lines 106-143) — but with one critical change: `compute_phi()` now CALLS `recovery_aware_input()` for each construct. This has never been done before (it exists in welfare_scoring.py but isn't wired in).

The function needs a `dx_dt` (trajectory derivative) for each construct. When called without trajectory context (e.g., from the Research tab for a static snapshot), we default `dx_dt=0.0` which means recovery potential depends only on community capacity.

**Step 1: Create welfare.py**

```python
"""
Phi(humanity) welfare formula — single source of truth.

Contains the full formula with recovery-aware floors, equity weights,
community multiplier, curiosity coupling, and divergence penalty.

This module has NO external dependencies (only stdlib math).
"""
import math
from typing import Dict, Optional

# ============================================================================
# Constants
# ============================================================================

ALL_CONSTRUCTS = ("c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi")

CONSTRUCT_FLOORS: Dict[str, float] = {
    "c": 0.20,       # Basic needs non-negotiable
    "kappa": 0.20,    # Crisis response minimum
    "lam_P": 0.20,    # Safety non-negotiable
    "lam_L": 0.15,    # Community minimum
    "xi": 0.30,       # Epistemic integrity highest floor
    "j": 0.10,        # Lower but present
    "p": 0.10,
    "eps": 0.10,
}

ETA = 0.10            # Ubuntu synergy coupling strength
MU = 0.15             # Divergence penalty coefficient
ETA_CURIOSITY = 0.08  # Cross-pair curiosity coupling (love x truth)
GAMMA = 0.5           # Community solidarity exponent

CONSTRUCT_PAIRS = [
    ("c", "lam_L"),      # Care x Love
    ("kappa", "lam_P"),   # Compassion x Protection
    ("j", "p"),           # Joy x Purpose
    ("eps", "xi"),        # Empathy x Truth
]
CURIOSITY_CROSS_PAIR = ("lam_L", "xi")
PENALTY_PAIRS = CONSTRUCT_PAIRS + [CURIOSITY_CROSS_PAIR]

CONSTRUCT_DISPLAY = {
    "c": "Care (c)",
    "kappa": "Compassion (\u03ba)",
    "j": "Joy (j)",
    "p": "Purpose (p)",
    "eps": "Empathy (\u03b5)",
    "lam_L": "Love (\u03bb_L)",
    "lam_P": "Protection (\u03bb_P)",
    "xi": "Truth (\u03be)",
}

FORMULA_VERSION = "2.1-recovery-floors"

# ============================================================================
# Recovery-aware floor function
# ============================================================================

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def recovery_aware_input(
    x_i: float,
    floor_i: float,
    dx_dt_i: float,
    lam_L: float,
) -> float:
    """
    Recovery-aware effective input for a construct.

    When above floor, pass through unchanged.
    When below, recovery potential comes from trajectory + community capacity.
    """
    if x_i >= floor_i:
        return x_i
    trajectory = _sigmoid(10.0 * dx_dt_i - 3.0)
    community_capacity = max(0.01, lam_L) ** 0.5
    recovery_potential = max(trajectory, community_capacity * 0.5)
    return x_i + (floor_i - x_i) * recovery_potential


# ============================================================================
# Component functions
# ============================================================================

def equity_weights(metrics: Dict[str, float]) -> Dict[str, float]:
    """Inverse-deprivation weights: w_i = (1/x_i) / sum(1/x_j)."""
    inv = {c: 1.0 / max(0.01, metrics.get(c, 0.5)) for c in ALL_CONSTRUCTS}
    inv_sum = sum(inv.values())
    return {c: inv[c] / inv_sum for c in ALL_CONSTRUCTS}


def community_multiplier(lam_L: float) -> float:
    """f(lam_L) = lam_L^gamma. Ubuntu substrate."""
    return max(0.01, lam_L) ** GAMMA


def ubuntu_synergy(metrics: Dict[str, float]) -> float:
    """Psi_ubuntu: paired construct synergy + curiosity cross-pair."""
    pair_sum = sum(
        math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5)))
        for a, b in CONSTRUCT_PAIRS
    )
    a, b = CURIOSITY_CROSS_PAIR
    curiosity = math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5)))
    return 1.0 + ETA * pair_sum + ETA_CURIOSITY * curiosity


def divergence_penalty(metrics: Dict[str, float]) -> float:
    """Psi_penalty: penalize paired construct mismatches."""
    sq_sum = sum(
        (metrics.get(a, 0.5) - metrics.get(b, 0.5)) ** 2
        for a, b in PENALTY_PAIRS
    )
    return MU * sq_sum / len(PENALTY_PAIRS)


# ============================================================================
# Full Phi computation
# ============================================================================

def compute_phi(
    metrics: Dict[str, float],
    derivatives: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute Phi(humanity) with recovery-aware floors.

    Phi = f(lam_L) * product(x_tilde_i^w_i) * Psi_ubuntu * (1 - Psi_penalty)

    Args:
        metrics: Current construct values, each in [0, 1].
        derivatives: Optional dx/dt for each construct. If None, defaults to 0.0
                     (recovery depends only on community capacity).
    """
    if derivatives is None:
        derivatives = {}

    lam_L_val = max(0.01, metrics.get("lam_L", 0.5))

    # Apply recovery-aware floors to get effective inputs
    effective = {}
    for c in ALL_CONSTRUCTS:
        x_i = max(0.01, metrics.get(c, 0.5))
        floor_i = CONSTRUCT_FLOORS.get(c, 0.10)
        dx_dt_i = derivatives.get(c, 0.0)
        effective[c] = recovery_aware_input(x_i, floor_i, dx_dt_i, lam_L_val)

    # Community multiplier
    f_lam = community_multiplier(lam_L_val)

    # Equity weights on effective inputs
    weights = equity_weights(effective)

    # Weighted geometric mean
    product = 1.0
    for c in ALL_CONSTRUCTS:
        product *= effective[c] ** weights[c]

    # Synergy and penalty (on raw metrics — synergy/penalty detect actual state)
    synergy = ubuntu_synergy(metrics)
    penalty = divergence_penalty(metrics)

    return max(0.0, f_lam * product * synergy * (1.0 - penalty))
```

**Step 2: Verify welfare.py works standalone**

Run from the repo root:

```bash
cd /home/crichalchemist/wave-experiment && python -c "
import sys; sys.path.insert(0, 'spaces/maninagarden')
from welfare import compute_phi, FORMULA_VERSION, ALL_CONSTRUCTS, CONSTRUCT_FLOORS
# Basic sanity
m = {c: 0.5 for c in ALL_CONSTRUCTS}
phi = compute_phi(m)
print(f'Version: {FORMULA_VERSION}')
print(f'Phi(all=0.5): {phi:.4f}')
assert 0.3 < phi < 0.8, f'Phi out of range: {phi}'
# Recovery floors active: below-floor construct with high community should recover
m_low = dict(m); m_low['c'] = 0.05
phi_low = compute_phi(m_low)
phi_low_deriv = compute_phi(m_low, derivatives={'c': 0.5})
print(f'Phi(c=0.05, no deriv): {phi_low:.4f}')
print(f'Phi(c=0.05, dx/dt=0.5): {phi_low_deriv:.4f}')
assert phi_low_deriv > phi_low, 'Recovery with positive trajectory should help'
print('welfare.py: ALL CHECKS PASSED')
"
```

Expected: All checks pass, Phi(all=0.5) in moderate range, recovery derivative effect visible.

**Step 3: Commit**

```bash
git add spaces/maninagarden/welfare.py
git commit -m "feat(space): extract welfare.py with recovery-aware floors wired into compute_phi"
```

---

### Task 2: Create `model.py` — model architecture + checkpoint loading

**Files:**
- Create: `spaces/maninagarden/model.py`

**Context:** Model classes are copied VERBATIM from `train_phi_hf_job.py:217-293` (required for `load_state_dict(strict=True)`). The loading logic is extracted from current `app.py:351-371` with an added `load_model_version()` for the Experiment Lab's checkpoint comparison feature.

**Step 1: Create model.py**

```python
"""
PhiForecasterGPU model architecture and checkpoint loading.

Architecture classes are VERBATIM copies from the training script
(scripts/train_phi_hf_job.py:217-293). Any change here will break
load_state_dict(strict=True).
"""
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_commits

MODEL_REPO = "crichalchemist/phi-forecaster"
CHECKPOINT_FILE = "phi_forecaster_best.pt"
SEQ_LEN = 50
PRED_LEN = 10
INPUT_SIZE = 36
HIDDEN_SIZE = 256


# ============================================================================
# Architecture (VERBATIM from training script — do NOT modify)
# ============================================================================

class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size // 2),
                  nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2),
                       nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, x):
        return self.lstm(x)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        q = self.query(x[:, -1:, :])
        k = self.key(x)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * x).sum(dim=1)
        return context, weights


class PhiForecasterGPU(nn.Module):
    def __init__(self, input_size=36, hidden_size=256, n_layers=2,
                 pred_len=10, dropout=0.1, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.cnn = CNN1D(input_size, hidden_size, kernel_size=3,
                         num_layers=2, dropout=dropout)
        self.lstm = StackedLSTM(hidden_size, hidden_size, n_layers, dropout)
        self.attn = AdditiveAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_len * 1),
        )
        self.construct_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_len * 8),
        )
        self.pred_len = pred_len
        self._loss_fn = nn.MSELoss()

    def forward(self, x):
        h = self.cnn(x)
        h, _ = self.lstm(h)
        h = self.dropout(h)
        context, attn = self.attn(h)
        phi = self.phi_head(context).view(-1, self.pred_len, 1)
        constructs = self.construct_head(context).view(-1, self.pred_len, 8)
        return phi, constructs, attn

    def compute_loss(self, phi_pred, phi_target, construct_pred, construct_target):
        return self._loss_fn(phi_pred, phi_target) + self.alpha * self._loss_fn(
            construct_pred, construct_target
        )

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Loading functions
# ============================================================================

def load_model(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE, revision=None):
    """Download checkpoint from Hub and load into PhiForecasterGPU."""
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    model = PhiForecasterGPU(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
        n_layers=2, pred_len=PRED_LEN,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def list_checkpoint_versions(repo_id=MODEL_REPO):
    """List available checkpoint versions (commits) from Hub."""
    try:
        commits = list_repo_commits(repo_id)
        versions = []
        for commit in commits[:20]:  # limit to recent 20
            versions.append({
                "sha": commit.commit_id[:8],
                "date": str(commit.created_at)[:10],
                "message": commit.title,
            })
        return versions
    except Exception:
        return []


def load_training_metadata(repo_id=MODEL_REPO, revision=None):
    """Load training_metadata.json from a checkpoint version."""
    import json
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename="training_metadata.json", revision=revision
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}
```

**Step 2: Verify model.py loads the checkpoint**

```bash
cd /home/crichalchemist/wave-experiment && python -c "
import sys; sys.path.insert(0, 'spaces/maninagarden')
from model import load_model, PhiForecasterGPU, list_checkpoint_versions
m = load_model()
print(f'Model loaded: {m.num_parameters:,} parameters')
import torch
x = torch.randn(1, 50, 36)
phi, constructs, attn = m(x)
print(f'phi shape: {phi.shape}')
print(f'constructs shape: {constructs.shape}')
print(f'attention sums to: {attn.sum().item():.6f}')
assert phi.shape == (1, 10, 1)
assert constructs.shape == (1, 10, 8)
print('model.py: ALL CHECKS PASSED')
"
```

**Step 3: Commit**

```bash
git add spaces/maninagarden/model.py
git commit -m "feat(space): extract model.py with checkpoint loading + version listing"
```

---

### Task 3: Create `scenarios.py` — data generation + signal processing

**Files:**
- Create: `spaces/maninagarden/scenarios.py`

**Context:** Signal processing and scenario generation extracted from current `app.py:69-205`. The key change: `compute_phi` is now imported from `welfare.py` instead of being defined inline. The `compute_all_signals` function also needs to use the new `compute_phi` with derivatives support — but since signal processing operates on DataFrames where we have the full trajectory, we can compute finite-difference `dx/dt` for each construct and pass it to `compute_phi`.

**Step 1: Create scenarios.py**

```python
"""
Scenario generation and signal processing for Phi forecasting.

Generates 8 archetypal welfare trajectories and computes 36 features:
8 raw constructs + 8 volatility + 8 momentum + 5 synergy + 5 divergence + phi + dphi_dt
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from welfare import compute_phi, ALL_CONSTRUCTS, PENALTY_PAIRS

# ============================================================================
# Signal processing
# ============================================================================

def volatility(values, window=20):
    if len(values) < window:
        return np.zeros_like(values)
    result = np.zeros_like(values, dtype=np.float64)
    for i in range(window, len(values)):
        result[i] = np.std(values[i - window:i])
    if window < len(values):
        result[:window] = result[window]
    return result


def price_momentum(prices, window=10):
    if len(prices) < window:
        return np.zeros_like(prices)
    result = np.zeros_like(prices, dtype=np.float64)
    for i in range(window, len(prices)):
        result[i] = (prices[i] - prices[i - window]) / max(1e-10, abs(prices[i - window]))
    return result


def synergy_signal(a, b, window=20):
    geo = np.sqrt(np.maximum(0, a) * np.maximum(0, b))
    if len(geo) < window:
        return geo
    result = np.zeros_like(geo, dtype=np.float64)
    for i in range(window, len(geo)):
        result[i] = np.mean(geo[i - window:i])
    result[:window] = result[window] if window < len(geo) else 0
    return result


def divergence_signal(a, b, window=20):
    sq_diff = (a - b) ** 2
    if len(sq_diff) < window:
        return sq_diff
    result = np.zeros_like(sq_diff, dtype=np.float64)
    for i in range(window, len(sq_diff)):
        result[i] = np.mean(sq_diff[i - window:i])
    result[:window] = result[window] if window < len(sq_diff) else 0
    return result


def compute_all_signals(df, window=20):
    """Compute all 36 features from raw construct trajectories."""
    out = df[list(ALL_CONSTRUCTS)].copy()
    for c in ALL_CONSTRUCTS:
        vals = out[c].values.astype(np.float64)
        out[f"{c}_vol"] = volatility(vals, window)
        out[f"{c}_mom"] = price_momentum(vals, window)
    for a, b in PENALTY_PAIRS:
        out[f"syn_{a}_{b}"] = synergy_signal(df[a].values, df[b].values, window)
        out[f"div_{a}_{b}"] = divergence_signal(df[a].values, df[b].values, window)

    # Compute Phi with derivatives (finite difference of each construct)
    derivatives_list = []
    for idx in range(len(out)):
        derivs = {}
        for c in ALL_CONSTRUCTS:
            vals = out[c].values
            if idx > 0:
                derivs[c] = float(vals[idx] - vals[idx - 1])
            else:
                derivs[c] = 0.0
        derivatives_list.append(derivs)

    phi_vals = np.array([
        compute_phi(
            {c: row[c] for c in ALL_CONSTRUCTS},
            derivatives=derivatives_list[idx],
        )
        for idx, (_, row) in enumerate(out[list(ALL_CONSTRUCTS)].iterrows())
    ])
    out["phi"] = phi_vals
    out["dphi_dt"] = np.gradient(phi_vals)
    return out


# ============================================================================
# Scenarios
# ============================================================================

SCENARIOS = (
    "stable_community", "capitalism_suppresses_love", "surveillance_state",
    "willful_ignorance", "recovery_arc", "sudden_crisis", "slow_decay", "random_walk",
)

SCENARIO_DESCRIPTIONS = {
    "stable_community": "All constructs hover around 0.5 with low noise. "
                        "Phi remains stable. Baseline scenario.",
    "capitalism_suppresses_love": "Love (\u03bb_L) declines 0.6\u21920.1 as purpose erodes. "
                                  "Phi drops as community solidarity degrades. "
                                  "Material care persists but developmental support vanishes.",
    "surveillance_state": "Truth (\u03be) rises 0.3\u21920.9 while love drops 0.6\u21920.1. "
                          "Truth without love = surveillance. "
                          "Divergence penalty fires on (\u03bb_L \u2212 \u03be)\u00b2.",
    "willful_ignorance": "Love rises 0.3\u21920.8 while truth drops 0.7\u21920.15. "
                         "Love without truth = willful ignorance. "
                         "Community solidarity high but epistemic integrity collapses.",
    "recovery_arc": "All constructs drop to floor, then \u03bb_L recovers first. "
                    "Community leads recovery (Ubuntu). "
                    "Shows how love is the substrate for rebuilding.",
    "sudden_crisis": "Compassion (\u03ba) and protection (\u03bb_P) crash mid-scenario. "
                     "Crisis event disrupts emergency response and safeguarding.",
    "slow_decay": "All 8 constructs decline at different rates from 0.7. "
                  "Gradual institutional erosion \u2014 the boiling frog.",
    "random_walk": "Correlated random walks with mean-reversion around 0.5. "
                   "Tests general forecasting under uncertainty.",
}


def generate_scenario(scenario, length=200, rng=None):
    """Generate a synthetic welfare trajectory for the given scenario."""
    if rng is None:
        rng = np.random.default_rng(42)

    def base(noise=0.01):
        return {c: 0.5 + rng.normal(0, noise, length).cumsum().clip(-0.1, 0.1)
                for c in ALL_CONSTRUCTS}

    if scenario == "stable_community":
        d = {c: 0.5 + rng.normal(0, 0.02, length).cumsum().clip(-0.1, 0.1)
             for c in ALL_CONSTRUCTS}
    elif scenario == "capitalism_suppresses_love":
        d = base()
        d["lam_L"] = np.linspace(0.6, 0.1, length) + rng.normal(0, 0.01, length)
        d["p"] = np.linspace(0.5, 0.25, length) + rng.normal(0, 0.01, length)
    elif scenario == "surveillance_state":
        d = base()
        d["xi"] = np.linspace(0.3, 0.9, length) + rng.normal(0, 0.01, length)
        d["lam_L"] = np.linspace(0.6, 0.1, length) + rng.normal(0, 0.01, length)
        d["eps"] = np.linspace(0.5, 0.15, length) + rng.normal(0, 0.01, length)
    elif scenario == "willful_ignorance":
        d = base()
        d["lam_L"] = np.linspace(0.3, 0.8, length) + rng.normal(0, 0.01, length)
        d["xi"] = np.linspace(0.7, 0.15, length) + rng.normal(0, 0.01, length)
    elif scenario == "recovery_arc":
        third = length // 3
        d = {}
        for c in ALL_CONSTRUCTS:
            d[c] = np.concatenate([
                np.linspace(0.5, 0.15, third),
                np.full(third, 0.15),
                np.linspace(0.15, 0.45, length - 2 * third),
            ]) + rng.normal(0, 0.01, length)
        d["lam_L"] = np.concatenate([
            np.linspace(0.5, 0.15, third),
            np.full(third, 0.15),
            np.linspace(0.15, 0.7, length - 2 * third),
        ]) + rng.normal(0, 0.01, length)
    elif scenario == "sudden_crisis":
        d = base()
        cs, ce = length // 3, 2 * length // 3
        for c in ("kappa", "lam_P"):
            d[c][cs:ce] = np.linspace(0.5, 0.1, ce - cs) + rng.normal(0, 0.01, ce - cs)
    elif scenario == "slow_decay":
        rates = {"c": 0.002, "kappa": 0.001, "j": 0.003, "p": 0.001,
                 "eps": 0.002, "lam_L": 0.003, "lam_P": 0.001, "xi": 0.002}
        d = {c: 0.7 - rates[c] * np.arange(length) + rng.normal(0, 0.01, length)
             for c in ALL_CONSTRUCTS}
    elif scenario == "random_walk":
        d = {}
        for c in ALL_CONSTRUCTS:
            walk = np.zeros(length)
            walk[0] = 0.5
            for i in range(1, length):
                walk[i] = walk[i - 1] + rng.normal(0, 0.02)
                walk[i] += 0.01 * (0.5 - walk[i])
            d[c] = walk
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    df = pd.DataFrame(d)
    for c in ALL_CONSTRUCTS:
        df[c] = df[c].clip(0.0, 1.0)

    # Compute phi using derivatives from the trajectory
    phi_vals = []
    for idx in range(len(df)):
        metrics = {c: df[c].iloc[idx] for c in ALL_CONSTRUCTS}
        derivs = {}
        if idx > 0:
            for c in ALL_CONSTRUCTS:
                derivs[c] = float(df[c].iloc[idx] - df[c].iloc[idx - 1])
        phi_vals.append(compute_phi(metrics, derivatives=derivs))
    df["phi"] = phi_vals
    return df


def build_reference_scaler(seed=42):
    """Build RobustScaler from the reference scenario (must match training)."""
    rng = np.random.default_rng(seed)
    df_ref = generate_scenario("stable_community", length=200, rng=rng)
    features_ref = compute_all_signals(df_ref, window=20)
    scaler = RobustScaler()
    scaler.fit(features_ref.values)
    return scaler, list(features_ref.columns)
```

**Step 2: Verify scenarios.py generates valid data**

```bash
cd /home/crichalchemist/wave-experiment && python -c "
import sys; sys.path.insert(0, 'spaces/maninagarden')
from scenarios import generate_scenario, compute_all_signals, build_reference_scaler, SCENARIOS
import numpy as np

for s in SCENARIOS:
    df = generate_scenario(s, length=200, rng=np.random.default_rng(42))
    assert len(df) == 200, f'{s}: wrong length {len(df)}'
    assert not df['phi'].isna().any(), f'{s}: NaN in phi'
    assert (df['phi'] >= 0).all(), f'{s}: negative phi'
    print(f'{s}: phi range [{df[\"phi\"].min():.3f}, {df[\"phi\"].max():.3f}]')

# Test signal processing
df = generate_scenario('stable_community', length=200)
features = compute_all_signals(df)
assert features.shape[1] == 36, f'Expected 36 features, got {features.shape[1]}'
assert not np.any(np.isnan(features.values)), 'NaN in features'

# Test scaler
scaler, names = build_reference_scaler()
assert len(names) == 36
print(f'Features: {len(names)} columns')
print('scenarios.py: ALL CHECKS PASSED')
"
```

**Step 3: Commit**

```bash
git add spaces/maninagarden/scenarios.py
git commit -m "feat(space): extract scenarios.py with recovery-aware Phi in data generation"
```

---

### Task 4: Create `training.py` — HF Jobs integration + experiment comparison

**Files:**
- Create: `spaces/maninagarden/training.py`

**Context:** Entirely new module. Provides: (1) a function to build the training script string with configurable hyperparams, (2) a function to launch the job via `huggingface_hub.run_uv_job()`, (3) checkpoint comparison helpers. The training script passed to HF Jobs must be fully self-contained (all code inline), so `get_training_script()` generates the complete Python script as a string with the latest formula baked in.

**Step 1: Create training.py**

```python
"""
Training orchestration: launch HF Jobs, compare checkpoints.

The training script is generated as a self-contained string with all
welfare/signal/scenario/model code inline (HF Jobs has no access to
this repo). The formula version is tagged in training_metadata.json.
"""
import os
import json

from welfare import FORMULA_VERSION


def get_training_script(
    epochs=100,
    lr=5e-4,
    hidden_size=256,
    batch_size=64,
    scenarios_per_type=50,
    pred_len=10,
    seq_len=50,
) -> str:
    """Generate the full training script as a string for HF Jobs.

    The script is self-contained: all welfare scoring, signal processing,
    scenario generation, and model architecture are inlined. It reads
    HF_TOKEN from the environment and pushes the checkpoint to Hub.
    """
    # Read the current training script as the base
    # (it already has all inline code; we just parametrize it)
    script = f'''# /// script
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "pandas>=2.0.0",
#     "scikit-learn>=1.3.0",
#     "scipy>=1.10.0",
#     "tqdm>=4.65.0",
#     "huggingface-hub>=0.25.0",
#     "trackio",
# ]
# ///
"""PhiForecaster training — launched from Space.
Formula version: {FORMULA_VERSION}
Config: epochs={epochs}, lr={lr}, hidden={hidden_size}, batch={batch_size}, scenarios={scenarios_per_type}
"""
import math, os, json, logging
from typing import Dict, Tuple
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import RobustScaler
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_CONSTRUCTS = ("c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi")
CONSTRUCT_FLOORS = {{"c": 0.20, "kappa": 0.20, "lam_P": 0.20, "lam_L": 0.15, "xi": 0.30, "j": 0.10, "p": 0.10, "eps": 0.10}}
ETA = 0.10; MU = 0.15; ETA_CURIOSITY = 0.08; GAMMA = 0.5
CONSTRUCT_PAIRS = [("c", "lam_L"), ("kappa", "lam_P"), ("j", "p"), ("eps", "xi")]
CURIOSITY_CROSS_PAIR = ("lam_L", "xi")
PENALTY_PAIRS = CONSTRUCT_PAIRS + [CURIOSITY_CROSS_PAIR]

def _sigmoid(x):
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x); return ex / (1.0 + ex)

def recovery_aware_input(x_i, floor_i, dx_dt_i, lam_L):
    if x_i >= floor_i: return x_i
    trajectory = _sigmoid(10.0 * dx_dt_i - 3.0)
    community_capacity = max(0.01, lam_L) ** 0.5
    recovery_potential = max(trajectory, community_capacity * 0.5)
    return x_i + (floor_i - x_i) * recovery_potential

def compute_phi(metrics, derivatives=None):
    if derivatives is None: derivatives = {{}}
    lam_L_val = max(0.01, metrics.get("lam_L", 0.5))
    effective = {{}}
    for c in ALL_CONSTRUCTS:
        x_i = max(0.01, metrics.get(c, 0.5))
        floor_i = CONSTRUCT_FLOORS.get(c, 0.10)
        dx_dt_i = derivatives.get(c, 0.0)
        effective[c] = recovery_aware_input(x_i, floor_i, dx_dt_i, lam_L_val)
    f_lam = max(0.01, lam_L_val) ** GAMMA
    inv = {{c: 1.0 / max(0.01, effective[c]) for c in ALL_CONSTRUCTS}}
    inv_sum = sum(inv.values())
    weights = {{c: inv[c] / inv_sum for c in ALL_CONSTRUCTS}}
    product = 1.0
    for c in ALL_CONSTRUCTS: product *= effective[c] ** weights[c]
    pair_sum = sum(math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5))) for a, b in CONSTRUCT_PAIRS)
    a, b = CURIOSITY_CROSS_PAIR
    curiosity = math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5)))
    synergy = 1.0 + ETA * pair_sum + ETA_CURIOSITY * curiosity
    sq_sum = sum((metrics.get(a, 0.5) - metrics.get(b, 0.5)) ** 2 for a, b in PENALTY_PAIRS)
    penalty = MU * sq_sum / len(PENALTY_PAIRS)
    return max(0.0, f_lam * product * synergy * (1.0 - penalty))

def volatility(values, window=20):
    if len(values) < window: return np.zeros_like(values)
    result = np.zeros_like(values, dtype=np.float64)
    for i in range(window, len(values)): result[i] = np.std(values[i - window:i])
    if window < len(values): result[:window] = result[window]
    return result

def price_momentum(prices, window=10):
    if len(prices) < window: return np.zeros_like(prices)
    result = np.zeros_like(prices, dtype=np.float64)
    for i in range(window, len(prices)): result[i] = (prices[i] - prices[i - window]) / max(1e-10, abs(prices[i - window]))
    return result

def synergy_signal(a, b, window=20):
    geo = np.sqrt(np.maximum(0, a) * np.maximum(0, b))
    if len(geo) < window: return geo
    result = np.zeros_like(geo, dtype=np.float64)
    for i in range(window, len(geo)): result[i] = np.mean(geo[i - window:i])
    result[:window] = result[window] if window < len(geo) else 0
    return result

def divergence_signal(a, b, window=20):
    sq_diff = (a - b) ** 2
    if len(sq_diff) < window: return sq_diff
    result = np.zeros_like(sq_diff, dtype=np.float64)
    for i in range(window, len(sq_diff)): result[i] = np.mean(sq_diff[i - window:i])
    result[:window] = result[window] if window < len(sq_diff) else 0
    return result

def compute_all_signals(df, window=20):
    out = df[list(ALL_CONSTRUCTS)].copy()
    for c in ALL_CONSTRUCTS:
        vals = out[c].values.astype(np.float64)
        out[f"{{c}}_vol"] = volatility(vals, window)
        out[f"{{c}}_mom"] = price_momentum(vals, window)
    for a, b in PENALTY_PAIRS:
        out[f"syn_{{a}}_{{b}}"] = synergy_signal(df[a].values, df[b].values, window)
        out[f"div_{{a}}_{{b}}"] = divergence_signal(df[a].values, df[b].values, window)
    phi_vals = np.array([compute_phi({{c: row[c] for c in ALL_CONSTRUCTS}}) for _, row in out[list(ALL_CONSTRUCTS)].iterrows()])
    out["phi"] = phi_vals
    out["dphi_dt"] = np.gradient(phi_vals)
    return out

SCENARIOS = ("stable_community", "capitalism_suppresses_love", "surveillance_state",
    "willful_ignorance", "recovery_arc", "sudden_crisis", "slow_decay", "random_walk")

def generate_scenario(scenario, length=200, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    def base(noise=0.01):
        return {{c: 0.5 + rng.normal(0, noise, length).cumsum().clip(-0.1, 0.1) for c in ALL_CONSTRUCTS}}
    if scenario == "stable_community":
        d = {{c: 0.5 + rng.normal(0, 0.02, length).cumsum().clip(-0.1, 0.1) for c in ALL_CONSTRUCTS}}
    elif scenario == "capitalism_suppresses_love":
        d = base(); d["lam_L"] = np.linspace(0.6, 0.1, length) + rng.normal(0, 0.01, length)
        d["p"] = np.linspace(0.5, 0.25, length) + rng.normal(0, 0.01, length)
    elif scenario == "surveillance_state":
        d = base(); d["xi"] = np.linspace(0.3, 0.9, length) + rng.normal(0, 0.01, length)
        d["lam_L"] = np.linspace(0.6, 0.1, length) + rng.normal(0, 0.01, length)
        d["eps"] = np.linspace(0.5, 0.15, length) + rng.normal(0, 0.01, length)
    elif scenario == "willful_ignorance":
        d = base(); d["lam_L"] = np.linspace(0.3, 0.8, length) + rng.normal(0, 0.01, length)
        d["xi"] = np.linspace(0.7, 0.15, length) + rng.normal(0, 0.01, length)
    elif scenario == "recovery_arc":
        third = length // 3; d = {{}}
        for c in ALL_CONSTRUCTS:
            d[c] = np.concatenate([np.linspace(0.5, 0.15, third), np.full(third, 0.15),
                np.linspace(0.15, 0.45, length - 2 * third)]) + rng.normal(0, 0.01, length)
        d["lam_L"] = np.concatenate([np.linspace(0.5, 0.15, third), np.full(third, 0.15),
            np.linspace(0.15, 0.7, length - 2 * third)]) + rng.normal(0, 0.01, length)
    elif scenario == "sudden_crisis":
        d = base(); cs, ce = length // 3, 2 * length // 3
        for c in ("kappa", "lam_P"):
            d[c][cs:ce] = np.linspace(0.5, 0.1, ce - cs) + rng.normal(0, 0.01, ce - cs)
    elif scenario == "slow_decay":
        rates = {{"c": 0.002, "kappa": 0.001, "j": 0.003, "p": 0.001, "eps": 0.002, "lam_L": 0.003, "lam_P": 0.001, "xi": 0.002}}
        d = {{c: 0.7 - rates[c] * np.arange(length) + rng.normal(0, 0.01, length) for c in ALL_CONSTRUCTS}}
    elif scenario == "random_walk":
        d = {{}}
        for c in ALL_CONSTRUCTS:
            walk = np.zeros(length); walk[0] = 0.5
            for i in range(1, length): walk[i] = walk[i-1] + rng.normal(0, 0.02); walk[i] += 0.01*(0.5-walk[i])
            d[c] = walk
    else: raise ValueError(f"Unknown scenario: {{scenario}}")
    df = pd.DataFrame(d)
    for c in ALL_CONSTRUCTS: df[c] = df[c].clip(0.0, 1.0)
    df["phi"] = df.apply(lambda row: compute_phi({{c: row[c] for c in ALL_CONSTRUCTS}}), axis=1)
    return df

class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size//2), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers-1): layers += [nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x.transpose(1, 2)).transpose(1, 2)

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    def forward(self, x): return self.lstm(x)

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    def forward(self, x, mask=None):
        q = self.query(x[:, -1:, :]); k = self.key(x)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        if mask is not None: scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * x).sum(dim=1)
        return context, weights

class PhiForecasterGPU(nn.Module):
    def __init__(self, input_size=36, hidden_size=256, n_layers=2, pred_len=10, dropout=0.1, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.cnn = CNN1D(input_size, hidden_size, kernel_size=3, num_layers=2, dropout=dropout)
        self.lstm = StackedLSTM(hidden_size, hidden_size, n_layers, dropout)
        self.attn = AdditiveAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.phi_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size//2, hidden_size//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size//2, pred_len*1))
        self.construct_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size//2, hidden_size//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size//2, pred_len*8))
        self.pred_len = pred_len; self._loss_fn = nn.MSELoss()
    def forward(self, x):
        h = self.cnn(x); h, _ = self.lstm(h); h = self.dropout(h)
        context, attn = self.attn(h)
        phi = self.phi_head(context).view(-1, self.pred_len, 1)
        constructs = self.construct_head(context).view(-1, self.pred_len, 8)
        return phi, constructs, attn
    def compute_loss(self, phi_pred, phi_target, construct_pred, construct_target):
        return self._loss_fn(phi_pred, phi_target) + self.alpha * self._loss_fn(construct_pred, construct_target)
    @property
    def num_parameters(self): return sum(p.numel() for p in self.parameters())

def main():
    import trackio
    seed = 42; pred_len = {pred_len}; seq_len = {seq_len}; hidden_size = {hidden_size}
    epochs = {epochs}; batch_size = {batch_size}; lr = {lr}; scenarios_per_type = {scenarios_per_type}
    length = 200; window = 20
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {{device}}")
    if device.type == "cuda": logger.info(f"GPU: {{torch.cuda.get_device_name()}}")
    token = os.environ.get("HF_TOKEN")
    trackio.init(project="phi-forecaster-training", name=f"v{FORMULA_VERSION}-{{epochs}}ep",
        config={{"hidden_size": hidden_size, "n_layers": 2, "pred_len": pred_len, "seq_len": seq_len,
                "lr": lr, "batch_size": batch_size, "epochs": epochs, "scenarios_per_type": scenarios_per_type,
                "formula_version": "{FORMULA_VERSION}"}})
    logger.info(f"Generating {{len(SCENARIOS)}} x {{scenarios_per_type}} scenarios...")
    all_X, all_y_phi, all_y_construct = [], [], []
    scaler = RobustScaler(); feature_names = None
    for s_idx, scenario in enumerate(SCENARIOS):
        for i in range(scenarios_per_type):
            rng = np.random.default_rng(seed + s_idx * 1000 + i)
            df = generate_scenario(scenario, length=length, rng=rng)
            features = compute_all_signals(df, window=window)
            if feature_names is None: feature_names = list(features.columns); scaler.fit(features.values)
            X = scaler.transform(features[feature_names].values)
            n_features = X.shape[1]
            for j in range(len(X) - seq_len - pred_len):
                all_X.append(X[j:j + seq_len])
                phi_t = df["phi"].values[j + seq_len:j + seq_len + pred_len]
                if len(phi_t) < pred_len: continue
                all_y_phi.append(phi_t.reshape(-1, 1))
                ct = np.stack([df[c].values[j + seq_len:j + seq_len + pred_len] for c in ALL_CONSTRUCTS], axis=-1)
                all_y_construct.append(ct)
    X_all = torch.tensor(np.array(all_X), dtype=torch.float32)
    y_phi = torch.tensor(np.array(all_y_phi), dtype=torch.float32)
    y_construct = torch.tensor(np.array(all_y_construct), dtype=torch.float32)
    logger.info(f"Data: {{X_all.shape[0]}} sequences, {{n_features}} features")
    n = len(X_all); perm = torch.randperm(n); split = int(0.8 * n)
    train_ds = torch.utils.data.TensorDataset(X_all[perm[:split]], y_phi[perm[:split]], y_construct[perm[:split]])
    val_ds = torch.utils.data.TensorDataset(X_all[perm[split:]], y_phi[perm[split:]], y_construct[perm[split:]])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True)
    model = PhiForecasterGPU(input_size=n_features, hidden_size=hidden_size, n_layers=2, pred_len=pred_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    logger.info(f"Model: {{model.num_parameters:,}} parameters")
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {{epoch}}", leave=False):
            x, yp, yc = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad(); pp, cp, _ = model(x)
            loss = model.compute_loss(pp, yp, cp, yc); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, yp, yc = [b.to(device, non_blocking=True) for b in batch]
                pp, cp, _ = model(x); val_loss += model.compute_loss(pp, yp, cp, yc).item()
        val_loss /= len(val_loader); scheduler.step()
        trackio.log({{"train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]}})
        logger.info(f"Epoch {{epoch}}/{{epochs}} train={{train_loss:.6f}} val={{val_loss:.6f}}")
        if val_loss < best_val_loss: best_val_loss = val_loss; torch.save(model.state_dict(), "/tmp/phi_forecaster_best.pt")
    logger.info(f"Best val_loss: {{best_val_loss:.6f}}")
    from huggingface_hub import HfApi
    api = HfApi(token=token); api.create_repo("crichalchemist/phi-forecaster", exist_ok=True)
    api.upload_file(path_or_fileobj="/tmp/phi_forecaster_best.pt", path_in_repo="phi_forecaster_best.pt",
        repo_id="crichalchemist/phi-forecaster", commit_message=f"PhiForecaster {FORMULA_VERSION} — val_loss={{best_val_loss:.6f}}")
    metadata = {{"model": "PhiForecasterGPU", "formula_version": "{FORMULA_VERSION}",
        "has_recovery_floors": True,
        "constants": {{"ETA": 0.10, "MU": 0.15, "ETA_CURIOSITY": 0.08, "GAMMA": 0.5}},
        "hidden_size": hidden_size, "n_layers": 2, "pred_len": pred_len,
        "input_features": n_features, "epochs": epochs, "best_val_loss": best_val_loss,
        "train_sequences": len(train_ds), "scenarios": list(SCENARIOS), "scenarios_per_type": scenarios_per_type}}
    import tempfile; meta_path = os.path.join(tempfile.gettempdir(), "training_metadata.json")
    with open(meta_path, "w") as f: json.dump(metadata, f, indent=2)
    api.upload_file(path_or_fileobj=meta_path, path_in_repo="training_metadata.json",
        repo_id="crichalchemist/phi-forecaster")
    logger.info("Checkpoint + metadata pushed to crichalchemist/phi-forecaster")

if __name__ == "__main__": main()
'''
    return script


HARDWARE_OPTIONS = {
    "t4-small": "T4 Small — quick experiments (~$0.06/hr)",
    "a10g-small": "A10G Small — production training (~$0.75/hr)",
    "a10g-large": "A10G Large — large batch training (~$1.50/hr)",
}


def launch_training_job(
    epochs=100,
    lr=5e-4,
    hidden_size=256,
    batch_size=64,
    scenarios_per_type=50,
    hardware="t4-small",
    timeout="2h",
):
    """Launch a training job on HF Infrastructure.

    Returns (job_id, message) tuple.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None, "HF_TOKEN not found. Add it as a Space secret."

    script = get_training_script(
        epochs=epochs, lr=lr, hidden_size=hidden_size,
        batch_size=batch_size, scenarios_per_type=scenarios_per_type,
    )

    try:
        from huggingface_hub import run_uv_job
        job = run_uv_job(
            script,
            flavor=hardware,
            timeout=timeout,
            secrets={"HF_TOKEN": token},
        )
        job_id = getattr(job, "id", str(job))
        return job_id, f"Job launched: {job_id}\nHardware: {hardware}\nTimeout: {timeout}"
    except Exception as e:
        return None, f"Failed to launch job: {e}"


def compare_checkpoints(repo_id, revision_a, revision_b):
    """Load metadata for two checkpoint revisions for comparison."""
    from model import load_training_metadata
    meta_a = load_training_metadata(repo_id, revision=revision_a if revision_a != "latest" else None)
    meta_b = load_training_metadata(repo_id, revision=revision_b if revision_b != "latest" else None)
    return meta_a, meta_b
```

**Step 2: Verify training.py generates a valid script**

```bash
cd /home/crichalchemist/wave-experiment && python -c "
import sys; sys.path.insert(0, 'spaces/maninagarden')
from training import get_training_script, HARDWARE_OPTIONS
script = get_training_script(epochs=5, lr=1e-3, hidden_size=64, batch_size=16, scenarios_per_type=5)
assert 'recovery_aware_input' in script, 'Recovery floors missing from training script'
assert 'formula_version' in script, 'Formula version missing from training script'
assert 'trackio.init' in script, 'Trackio missing from training script'
assert '2.1-recovery-floors' in script, 'Formula version tag missing'
print(f'Script length: {len(script)} chars')
print(f'Hardware options: {list(HARDWARE_OPTIONS.keys())}')
print('training.py: ALL CHECKS PASSED')
"
```

**Step 3: Commit**

```bash
git add spaces/maninagarden/training.py
git commit -m "feat(space): add training.py — HF Jobs orchestration with recovery-aware formula"
```

---

### Task 5: Rewrite `app.py` as UI shell with 6 tabs

**Files:**
- Rewrite: `spaces/maninagarden/app.py`

**Context:** The current 718-line monolith becomes a ~400-line UI shell that imports from the 4 modules. Three existing tabs are preserved (Scenario Explorer, Custom Forecast, Research). Three new tabs are added (Experiment Lab, Training, Data Workshop). Training and Data Workshop are password-gated via ADMIN_KEY Space secret.

**Step 1: Rewrite app.py**

Write the complete new `app.py`. This is the full file — replace everything:

```python
"""
Phi Research Workbench — Interactive welfare trajectory forecasting + training.

6 tabs: Scenario Explorer, Custom Forecast, Experiment Lab,
Training (gated), Data Workshop (gated), Research.
"""
import os
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import gradio as gr

from welfare import (
    compute_phi, ALL_CONSTRUCTS, CONSTRUCT_FLOORS, CONSTRUCT_DISPLAY,
    PENALTY_PAIRS, FORMULA_VERSION,
)
from model import load_model, list_checkpoint_versions, load_training_metadata, SEQ_LEN, PRED_LEN, MODEL_REPO
from scenarios import (
    generate_scenario, compute_all_signals, build_reference_scaler,
    SCENARIOS, SCENARIO_DESCRIPTIONS,
)
from training import get_training_script, launch_training_job, HARDWARE_OPTIONS

# ============================================================================
# Startup
# ============================================================================

print(f"Formula version: {FORMULA_VERSION}")
print("Loading PhiForecaster checkpoint...")
MODEL = load_model()
print(f"Model loaded: {MODEL.num_parameters:,} parameters")

print("Building reference scaler...")
REFERENCE_SCALER, FEATURE_NAMES = build_reference_scaler()
print(f"Scaler ready: {len(FEATURE_NAMES)} features")

CONSTRUCT_COLORS = [
    "#E91E63", "#9C27B0", "#FF9800", "#4CAF50",
    "#00BCD4", "#F44336", "#3F51B5", "#795548",
]


# ============================================================================
# Inference pipeline
# ============================================================================

def run_inference(df, model=None):
    """Full pipeline: DataFrame -> (phi_pred, construct_pred, attention)."""
    if model is None:
        model = MODEL
    features = compute_all_signals(df, window=20)
    X_scaled = REFERENCE_SCALER.transform(features[FEATURE_NAMES].values)
    X_window = X_scaled[-SEQ_LEN:]
    X_tensor = torch.tensor(X_window[np.newaxis], dtype=torch.float32)

    with torch.no_grad():
        phi_pred, construct_pred, attn_weights = model(X_tensor)

    return (
        phi_pred[0, :, 0].numpy(),
        construct_pred[0].numpy(),
        attn_weights[0].numpy(),
    )


# ============================================================================
# Plotting helpers
# ============================================================================

def make_phi_plot(historical_phi, phi_forecast, title, length=200):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(length)), y=historical_phi,
        mode="lines", name="Historical \u03a6",
        line=dict(color="#2196F3", width=2),
    ))
    forecast_x = list(range(length, length + PRED_LEN))
    fig.add_trace(go.Scatter(
        x=forecast_x, y=phi_forecast,
        mode="lines+markers", name="Forecasted \u03a6",
        line=dict(color="#FF5722", width=3, dash="dash"),
        marker=dict(size=6),
    ))
    fig.add_vline(x=length - 0.5, line_dash="dot", line_color="gray",
                  annotation_text="Forecast horizon")
    fig.update_layout(
        title=title, xaxis_title="Time Step", yaxis_title="\u03a6 Score",
        yaxis_range=[0, 1], template="plotly_white", height=400,
    )
    return fig


def make_construct_plot(df, construct_forecast, title, length=200):
    fig = go.Figure()
    forecast_x = list(range(length, length + PRED_LEN))
    for i, c in enumerate(ALL_CONSTRUCTS):
        fig.add_trace(go.Scatter(
            x=list(range(length)), y=df[c].values,
            mode="lines", name=CONSTRUCT_DISPLAY[c],
            line=dict(color=CONSTRUCT_COLORS[i], width=1.5),
            legendgroup=c,
        ))
        fig.add_trace(go.Scatter(
            x=forecast_x, y=construct_forecast[:, i],
            mode="lines", name=f"{CONSTRUCT_DISPLAY[c]} (forecast)",
            line=dict(color=CONSTRUCT_COLORS[i], width=2.5, dash="dash"),
            legendgroup=c, showlegend=False,
        ))
    fig.add_vline(x=length - 0.5, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=title, xaxis_title="Time Step",
        yaxis_title="Construct Value [0, 1]",
        yaxis_range=[0, 1], template="plotly_white", height=500,
    )
    return fig


def make_attention_plot(attention):
    fig = go.Figure(go.Bar(
        x=list(range(len(attention))), y=attention,
        marker_color="#7C4DFF",
    ))
    fig.update_layout(
        title="Attention Weights (last 50 input steps)",
        xaxis_title="Input Step (within window)",
        yaxis_title="Attention Weight",
        template="plotly_white", height=300,
    )
    return fig


# ============================================================================
# Tab handlers
# ============================================================================

def explore_scenario(scenario, seed):
    rng = np.random.default_rng(int(seed))
    df = generate_scenario(scenario, length=200, rng=rng)
    phi_traj, construct_traj, attn = run_inference(df)
    title = scenario.replace("_", " ").title()
    return (
        SCENARIO_DESCRIPTIONS.get(scenario, ""),
        make_phi_plot(df["phi"].values, phi_traj, f"\u03a6(humanity) \u2014 {title}"),
        make_construct_plot(df, construct_traj, f"8 Welfare Constructs \u2014 {title}"),
        make_attention_plot(attn),
    )


def custom_forecast(c_val, kappa_val, j_val, p_val, eps_val, lam_L_val, lam_P_val, xi_val):
    levels = {
        "c": c_val, "kappa": kappa_val, "j": j_val, "p": p_val,
        "eps": eps_val, "lam_L": lam_L_val, "lam_P": lam_P_val, "xi": xi_val,
    }
    current_phi = compute_phi(levels)
    rng = np.random.default_rng(42)
    data = {}
    for c in ALL_CONSTRUCTS:
        center = levels[c]
        values = np.zeros(200)
        values[0] = center
        for i in range(1, 200):
            values[i] = values[i-1] + 0.05*(center - values[i-1]) + rng.normal(0, 0.015)
        data[c] = np.clip(values, 0.0, 1.0)
    df = pd.DataFrame(data)
    phi_vals = []
    for idx in range(len(df)):
        m = {c: df[c].iloc[idx] for c in ALL_CONSTRUCTS}
        derivs = {}
        if idx > 0:
            derivs = {c: float(df[c].iloc[idx] - df[c].iloc[idx-1]) for c in ALL_CONSTRUCTS}
        phi_vals.append(compute_phi(m, derivatives=derivs))
    df["phi"] = phi_vals

    phi_traj, construct_traj, _ = run_inference(df)
    return (
        f"**Current \u03a6:** {current_phi:.3f}",
        make_phi_plot(df["phi"].values, phi_traj, "Custom \u03a6 Trajectory"),
        make_construct_plot(df, construct_traj, "Custom Construct Trajectories"),
    )


def check_admin_key(key):
    """Verify admin key and return visibility updates."""
    expected = os.environ.get("ADMIN_KEY", "")
    if key == expected and expected:
        return gr.update(visible=True), "Unlocked."
    return gr.update(visible=False), "Invalid key."


def do_launch_training(key, epochs, lr, hidden_size, batch_size, scenarios_per_type, hardware):
    expected = os.environ.get("ADMIN_KEY", "")
    if key != expected or not expected:
        return "Not authenticated."
    job_id, msg = launch_training_job(
        epochs=int(epochs), lr=float(lr), hidden_size=int(hidden_size),
        batch_size=int(batch_size), scenarios_per_type=int(scenarios_per_type),
        hardware=hardware,
    )
    return msg


def inspect_data(scenario, seed):
    """Data Workshop: show raw trajectory + signal features."""
    rng = np.random.default_rng(int(seed))
    df = generate_scenario(scenario, length=200, rng=rng)
    features = compute_all_signals(df, window=20)

    # Raw trajectory plot
    fig_raw = go.Figure()
    for i, c in enumerate(ALL_CONSTRUCTS):
        fig_raw.add_trace(go.Scatter(
            x=list(range(200)), y=df[c].values,
            mode="lines", name=CONSTRUCT_DISPLAY[c],
            line=dict(color=CONSTRUCT_COLORS[i]),
        ))
    fig_raw.update_layout(
        title=f"Raw Trajectory \u2014 {scenario.replace('_', ' ').title()}",
        xaxis_title="Time Step", yaxis_title="Value [0, 1]",
        yaxis_range=[0, 1], template="plotly_white", height=400,
    )

    # Feature heatmap (sampled columns for readability)
    sample_cols = [f"{c}_vol" for c in ALL_CONSTRUCTS[:4]] + [f"{c}_mom" for c in ALL_CONSTRUCTS[:4]] + ["phi", "dphi_dt"]
    fig_heat = go.Figure(go.Heatmap(
        z=features[sample_cols].values.T,
        x=list(range(200)),
        y=sample_cols,
        colorscale="RdBu_r",
    ))
    fig_heat.update_layout(
        title="Signal Features (sample)", xaxis_title="Time Step",
        template="plotly_white", height=400,
    )

    # Phi with recovery floors visible
    fig_phi = go.Figure()
    fig_phi.add_trace(go.Scatter(
        x=list(range(200)), y=df["phi"].values,
        mode="lines", name="\u03a6(humanity)",
        line=dict(color="#2196F3", width=2),
    ))
    fig_phi.update_layout(
        title="\u03a6 Trajectory (with recovery-aware floors)",
        xaxis_title="Time Step", yaxis_title="\u03a6",
        yaxis_range=[0, 1], template="plotly_white", height=300,
    )

    stats = f"**Features:** {features.shape[1]} columns, {features.shape[0]} rows\n"
    stats += f"**Phi range:** [{df['phi'].min():.3f}, {df['phi'].max():.3f}]\n"
    stats += f"**NaN count:** {features.isna().sum().sum()}"

    return fig_raw, fig_heat, fig_phi, stats


def compare_experiments(scenario, seed, revision_a, revision_b):
    """Experiment Lab: run same scenario through two checkpoints."""
    rng_a = np.random.default_rng(int(seed))
    df = generate_scenario(scenario, length=200, rng=rng_a)

    try:
        model_a = load_model(revision=revision_a if revision_a != "latest" else None)
    except Exception as e:
        return None, None, f"Failed to load checkpoint A: {e}"
    try:
        model_b = load_model(revision=revision_b if revision_b != "latest" else None)
    except Exception as e:
        return None, None, f"Failed to load checkpoint B: {e}"

    phi_a, _, _ = run_inference(df, model=model_a)
    phi_b, _, _ = run_inference(df, model=model_b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(200)), y=df["phi"].values,
        mode="lines", name="Historical \u03a6", line=dict(color="#2196F3", width=2),
    ))
    forecast_x = list(range(200, 200 + PRED_LEN))
    fig.add_trace(go.Scatter(
        x=forecast_x, y=phi_a,
        mode="lines+markers", name=f"Checkpoint A ({revision_a[:8]})",
        line=dict(color="#FF5722", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_x, y=phi_b,
        mode="lines+markers", name=f"Checkpoint B ({revision_b[:8]})",
        line=dict(color="#4CAF50", width=2, dash="dot"),
    ))
    fig.add_vline(x=199.5, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"Checkpoint Comparison \u2014 {scenario.replace('_', ' ').title()}",
        xaxis_title="Time Step", yaxis_title="\u03a6",
        yaxis_range=[0, 1], template="plotly_white", height=450,
    )

    meta_a = load_training_metadata(revision=revision_a if revision_a != "latest" else None)
    meta_b = load_training_metadata(revision=revision_b if revision_b != "latest" else None)

    def fmt(m):
        if not m:
            return "No metadata available"
        return (f"**Epochs:** {m.get('epochs', '?')} | "
                f"**Val Loss:** {m.get('best_val_loss', '?'):.6f} | "
                f"**Formula:** {m.get('formula_version', 'unknown')}")

    meta_text = f"**A:** {fmt(meta_a)}\n\n**B:** {fmt(meta_b)}"
    return fig, meta_text, ""


# ============================================================================
# Gradio UI
# ============================================================================

with gr.Blocks(
    title="\u03a6 Research Workbench",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(f"# \u03a6(humanity) Research Workbench")
    gr.Markdown(
        f"Formula **{FORMULA_VERSION}** | "
        "CNN+LSTM+Attention forecaster | "
        "Grounded in care ethics (hooks 2000), capability theory (Sen 1999), Ubuntu philosophy"
    )

    with gr.Tabs():
        # ==================== TAB 1: Scenario Explorer ====================
        with gr.Tab("Scenario Explorer"):
            with gr.Row():
                scenario_dropdown = gr.Dropdown(
                    choices=list(SCENARIOS), value="stable_community",
                    label="Welfare Scenario",
                )
                seed_slider = gr.Slider(0, 999, step=1, value=42, label="Random Seed")
                forecast_btn = gr.Button("Run Forecast", variant="primary")
            scenario_description = gr.Markdown(SCENARIO_DESCRIPTIONS["stable_community"])
            phi_plot = gr.Plot(label="\u03a6 Trajectory")
            construct_plot = gr.Plot(label="Construct Trajectories")
            attention_plot = gr.Plot(label="Attention Weights")

            forecast_btn.click(
                fn=explore_scenario,
                inputs=[scenario_dropdown, seed_slider],
                outputs=[scenario_description, phi_plot, construct_plot, attention_plot],
            )
            scenario_dropdown.change(
                fn=lambda s: SCENARIO_DESCRIPTIONS.get(s, ""),
                inputs=[scenario_dropdown], outputs=[scenario_description],
            )

        # ==================== TAB 2: Custom Forecast ====================
        with gr.Tab("Custom Forecast"):
            gr.Markdown("### Set construct levels and forecast")
            with gr.Row():
                c_slider = gr.Slider(CONSTRUCT_FLOORS["c"], 0.95, value=0.5, step=0.05, label="Care (c)")
                kappa_slider = gr.Slider(CONSTRUCT_FLOORS["kappa"], 0.95, value=0.5, step=0.05, label="Compassion (\u03ba)")
                j_slider = gr.Slider(CONSTRUCT_FLOORS["j"], 0.95, value=0.5, step=0.05, label="Joy (j)")
                p_slider = gr.Slider(CONSTRUCT_FLOORS["p"], 0.95, value=0.5, step=0.05, label="Purpose (p)")
            with gr.Row():
                eps_slider = gr.Slider(CONSTRUCT_FLOORS["eps"], 0.95, value=0.5, step=0.05, label="Empathy (\u03b5)")
                lam_L_slider = gr.Slider(CONSTRUCT_FLOORS["lam_L"], 0.95, value=0.5, step=0.05, label="Love (\u03bb_L)")
                lam_P_slider = gr.Slider(CONSTRUCT_FLOORS["lam_P"], 0.95, value=0.5, step=0.05, label="Protection (\u03bb_P)")
                xi_slider = gr.Slider(CONSTRUCT_FLOORS["xi"], 0.95, value=0.5, step=0.05, label="Truth (\u03be)")
            custom_btn = gr.Button("Generate & Forecast", variant="primary")
            phi_current_md = gr.Markdown("**Current \u03a6:** 0.500")
            custom_phi_plot = gr.Plot(label="\u03a6 Trajectory")
            custom_construct_plot = gr.Plot(label="Construct Trajectories")

            custom_btn.click(
                fn=custom_forecast,
                inputs=[c_slider, kappa_slider, j_slider, p_slider,
                        eps_slider, lam_L_slider, lam_P_slider, xi_slider],
                outputs=[phi_current_md, custom_phi_plot, custom_construct_plot],
            )

        # ==================== TAB 3: Experiment Lab ====================
        with gr.Tab("Experiment Lab"):
            gr.Markdown("### Compare two checkpoints on the same scenario")
            with gr.Row():
                exp_scenario = gr.Dropdown(choices=list(SCENARIOS), value="stable_community", label="Scenario")
                exp_seed = gr.Slider(0, 999, step=1, value=42, label="Seed")
            with gr.Row():
                exp_rev_a = gr.Textbox(value="latest", label="Checkpoint A (commit SHA or 'latest')")
                exp_rev_b = gr.Textbox(value="latest", label="Checkpoint B (commit SHA or 'latest')")
            exp_btn = gr.Button("Compare", variant="primary")
            exp_plot = gr.Plot(label="Forecast Comparison")
            exp_meta = gr.Markdown("")
            exp_error = gr.Markdown("")

            exp_btn.click(
                fn=compare_experiments,
                inputs=[exp_scenario, exp_seed, exp_rev_a, exp_rev_b],
                outputs=[exp_plot, exp_meta, exp_error],
            )

        # ==================== TAB 4: Training (gated) ====================
        with gr.Tab("Training"):
            gr.Markdown("### Launch training on HF Jobs")
            with gr.Row():
                train_key = gr.Textbox(type="password", label="Admin Key", scale=2)
                train_unlock_btn = gr.Button("Unlock", scale=1)
            train_status = gr.Markdown("")

            with gr.Group(visible=False) as train_controls:
                with gr.Row():
                    train_epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
                    train_lr = gr.Number(value=5e-4, label="Learning Rate")
                with gr.Row():
                    train_hidden = gr.Dropdown(choices=["64", "128", "256", "512"], value="256", label="Hidden Size")
                    train_batch = gr.Dropdown(choices=["16", "32", "64", "128"], value="64", label="Batch Size")
                with gr.Row():
                    train_scenarios = gr.Slider(10, 200, value=50, step=10, label="Scenarios per Type")
                    train_hardware = gr.Dropdown(choices=list(HARDWARE_OPTIONS.keys()), value="t4-small", label="Hardware")
                train_launch_btn = gr.Button("Launch Training Job", variant="primary")
                train_result = gr.Markdown("")

            train_unlock_btn.click(
                fn=check_admin_key, inputs=[train_key],
                outputs=[train_controls, train_status],
            )
            train_launch_btn.click(
                fn=do_launch_training,
                inputs=[train_key, train_epochs, train_lr, train_hidden,
                        train_batch, train_scenarios, train_hardware],
                outputs=[train_result],
            )

        # ==================== TAB 5: Data Workshop (gated) ====================
        with gr.Tab("Data Workshop"):
            gr.Markdown("### Inspect training data: scenarios, signals, features")
            with gr.Row():
                dw_key = gr.Textbox(type="password", label="Admin Key", scale=2)
                dw_unlock_btn = gr.Button("Unlock", scale=1)
            dw_status = gr.Markdown("")

            with gr.Group(visible=False) as dw_controls:
                with gr.Row():
                    dw_scenario = gr.Dropdown(choices=list(SCENARIOS), value="stable_community", label="Scenario")
                    dw_seed = gr.Slider(0, 999, step=1, value=42, label="Seed")
                dw_btn = gr.Button("Inspect", variant="primary")
                dw_raw = gr.Plot(label="Raw Trajectory")
                dw_heat = gr.Plot(label="Signal Features")
                dw_phi = gr.Plot(label="\u03a6 with Recovery Floors")
                dw_stats = gr.Markdown("")

            dw_unlock_btn.click(
                fn=check_admin_key, inputs=[dw_key],
                outputs=[dw_controls, dw_status],
            )
            dw_btn.click(
                fn=inspect_data, inputs=[dw_scenario, dw_seed],
                outputs=[dw_raw, dw_heat, dw_phi, dw_stats],
            )

        # ==================== TAB 6: Research ====================
        with gr.Tab("Research"):
            gr.Markdown("""
## \u03a6(humanity): A Rigorous Ethical-Affective Objective Function

### The Formula (v""" + FORMULA_VERSION + """)

```
\u03a6(humanity) = f(\u03bb_L) \u00b7 [\u220f(x\u0303_i)^(w_i)] \u00b7 \u03a8_ubuntu \u00b7 (1 \u2212 \u03a8_penalty)
```

**Components:**
- **f(\u03bb_L) = \u03bb_L^0.5** \u2014 Community solidarity multiplier (Ubuntu substrate)
- **x\u0303_i = recovery_aware_input(x_i)** \u2014 Effective input with recovery-aware floors. Below-floor constructs receive community-mediated recovery potential.
- **w_i = (1/x\u0303_i) / \u2211(1/x\u0303_j)** \u2014 Inverse-deprivation weights (Rawlsian maximin)
- **\u03a8_ubuntu = 1 + 0.10\u00b7[\u221a(c\u00b7\u03bb_L) + \u221a(\u03ba\u00b7\u03bb_P) + \u221a(j\u00b7p) + \u221a(\u03b5\u00b7\u03be)] + 0.08\u00b7\u221a(\u03bb_L\u00b7\u03be)** \u2014 Relational synergy + curiosity
- **\u03a8_penalty = 0.15\u00b7[(c\u2212\u03bb_L)\u00b2 + (\u03ba\u2212\u03bb_P)\u00b2 + (j\u2212p)\u00b2 + (\u03b5\u2212\u03be)\u00b2 + (\u03bb_L\u2212\u03be)\u00b2] / 5** \u2014 Structural distortion penalty

### Recovery-Aware Floors (New)

When a construct falls below its hard floor, recovery depends on both **trajectory** (dx/dt) and **community capacity** (\u03bb_L):
- **Healing trajectory + strong community** \u2192 rapid recovery toward floor
- **Stagnant + strong community** \u2192 partial recovery (community compensates)
- **Stagnant + no community** \u2192 true collapse ("white supremacy signature")

*Key insight: care doesn't begin the uptick without community intervention.*

### The Eight Constructs

| Symbol | Name | Definition | Floor | Citation |
|--------|------|-----------|-------|----------|
| c | Care | Resource allocation meeting basic needs | 0.20 | Tronto 1993 |
| \u03ba | Compassion | Responsive support to acute distress | 0.20 | Nussbaum 1996 |
| j | Joy | Positive affect above subsistence | 0.10 | Csikszentmihalyi 1990 |
| p | Purpose | Alignment of actions with chosen goals | 0.10 | Frankfurt 1971 |
| \u03b5 | Empathy | Accuracy of perspective-taking across groups | 0.10 | Batson 1991 |
| \u03bb_L | Love | Active extension for another's growth | 0.15 | hooks 2000 |
| \u03bb_P | Protection | Risk-weighted safeguarding from harm | 0.20 | Berlin 1969 |
| \u03be | Truth | Accuracy and transparency of records | 0.30 | Fricker 2007 |

### Synergy Pairs

| Pair | Constructs | Meaning | Distortion When Mismatched |
|------|-----------|---------|---------------------------|
| Care \u00d7 Love | c \u00b7 \u03bb_L | Material provision + developmental extension | Care without love = paternalistic control |
| Compassion \u00d7 Protection | \u03ba \u00b7 \u03bb_P | Emergency response + safeguarding | Compassion without protection = vulnerable support |
| Joy \u00d7 Purpose | j \u00b7 p | Positive affect + goal-alignment | Joy without purpose = hedonic treadmill |
| Empathy \u00d7 Truth | \u03b5 \u00b7 \u03be | Perspective-taking + epistemic integrity | Empathy without truth = manipulated solidarity |
| Love \u00d7 Truth | \u03bb_L \u00b7 \u03be | Investigative drive (curiosity) | Truth without love = surveillance; love without truth = willful ignorance |

### Key References

- hooks, b. (2000). *All About Love: New Visions*. William Morrow.
- Sen, A. (1999). *Development as Freedom*. Oxford University Press.
- Fricker, M. (2007). *Epistemic Injustice*. Oxford University Press.
- Collins, P. H. (1990). *Black Feminist Thought*. Routledge.
- Metz, T. (2007). Toward an African moral theory. *Journal of Political Philosophy*.
- Ramose, M. B. (1999). *African Philosophy Through Ubuntu*. Mond Books.

### Limitations

- **Diagnostic tool**, not an optimization target (Goodhart's Law)
- 8-construct taxonomy is Western-situated; requires adaptation for Ubuntu, Confucian, Buddhist, and Indigenous frameworks
- Trained on synthetic data; real-world calibration pending
- Recovery-aware floors add theoretical richness but require retraining to reflect in model predictions

*\u03a6(humanity) is not a turnkey moral oracle. It is a disciplined framework forcing transparency about normative commitments.*
""")

    # Auto-run default scenario on page load
    demo.load(
        fn=explore_scenario,
        inputs=[scenario_dropdown, seed_slider],
        outputs=[scenario_description, phi_plot, construct_plot, attention_plot],
    )

    demo.launch(show_error=True)
```

**Step 2: Verify the app imports correctly**

```bash
cd /home/crichalchemist/wave-experiment && python -c "
import sys; sys.path.insert(0, 'spaces/maninagarden')
# Just verify imports work — don't launch Gradio
from welfare import compute_phi, FORMULA_VERSION
from model import PhiForecasterGPU
from scenarios import SCENARIOS, generate_scenario
from training import get_training_script
print(f'Formula: {FORMULA_VERSION}')
print(f'Scenarios: {len(SCENARIOS)}')
print(f'Model class: {PhiForecasterGPU.__name__}')
print('All imports successful')
"
```

**Step 3: Commit**

```bash
git add spaces/maninagarden/app.py
git commit -m "feat(space): rewrite app.py as modular 6-tab research workbench"
```

---

### Task 6: Update `requirements.txt` and `README.md`

**Files:**
- Modify: `spaces/maninagarden/requirements.txt`
- Modify: `spaces/maninagarden/README.md`

**Step 1: Update requirements.txt**

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

**Step 2: Update README.md**

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

# Phi Research Workbench

Predict how community welfare evolves using the **Φ(humanity)** function — an ethical-affective objective function grounded in care ethics (hooks 2000), capability theory (Sen 1999), and Ubuntu philosophy.

## Tabs

1. **Scenario Explorer** — Pick from 8 archetypal welfare trajectories and see forecasts
2. **Custom Forecast** — Set construct levels with sliders and forecast from there
3. **Experiment Lab** — Compare checkpoints side-by-side on the same scenario
4. **Training** — Launch GPU training jobs on HF infrastructure (admin-gated)
5. **Data Workshop** — Inspect training data: raw trajectories, signal features, Φ computation (admin-gated)
6. **Research** — Full theoretical grounding, formula, citations

## Formula v2.1

```
Φ(humanity) = f(λ_L) · [∏(x̃_i)^(w_i)] · Ψ_ubuntu · (1 − Ψ_penalty)
```

Where x̃_i includes recovery-aware floors: below-floor constructs receive community-mediated recovery potential.

## Model

**PhiForecasterGPU**: CNN1D → Stacked LSTM → Additive Attention → Dual Heads

Checkpoint: [`crichalchemist/phi-forecaster`](https://huggingface.co/crichalchemist/phi-forecaster)

## Key References

- hooks, b. (2000). *All About Love: New Visions*
- Sen, A. (1999). *Development as Freedom*
- Fricker, M. (2007). *Epistemic Injustice*
- Collins, P. H. (1990). *Black Feminist Thought*
```

**Step 3: Commit**

```bash
git add spaces/maninagarden/requirements.txt spaces/maninagarden/README.md
git commit -m "feat(space): update requirements (add trackio) and README for research workbench"
```

---

### Task 7: Push to HF Space

**Files:**
- No local changes — pushing to Hub

**Step 1: Delete old app.py from Space (it will be replaced)**

The old Space has a single `app.py`. We need to push all 6 files. Use the HfApi to upload the entire directory.

```bash
cd /home/crichalchemist/wave-experiment && python -c "
from huggingface_hub import HfApi
api = HfApi()
repo = 'crichalchemist/maninagarden'

# Upload all Space files
files = [
    'spaces/maninagarden/app.py',
    'spaces/maninagarden/welfare.py',
    'spaces/maninagarden/model.py',
    'spaces/maninagarden/scenarios.py',
    'spaces/maninagarden/training.py',
    'spaces/maninagarden/requirements.txt',
    'spaces/maninagarden/README.md',
]
for f in files:
    name = f.split('/')[-1]
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=name,
        repo_id=repo,
        repo_type='space',
    )
    print(f'Pushed {name}')
print('All files pushed to Space')
"
```

**Step 2: Add ADMIN_KEY and HF_TOKEN as Space secrets**

This must be done via the HF web UI or CLI:

```bash
# Set ADMIN_KEY (choose your own passphrase)
# Go to: https://huggingface.co/spaces/crichalchemist/maninagarden/settings
# Add secret: ADMIN_KEY = <your chosen passphrase>
# Add secret: HF_TOKEN = <your HF write token>
```

**Step 3: Verify Space rebuilds**

```bash
cd /home/crichalchemist/wave-experiment && python -c "
from huggingface_hub import HfApi
import time
api = HfApi()
for i in range(30):
    info = api.space_info('crichalchemist/maninagarden')
    status = info.runtime.get('stage', 'UNKNOWN') if info.runtime else 'NO_RUNTIME'
    print(f'{i*10}s: {status}')
    if status == 'RUNNING':
        print('Space is RUNNING')
        break
    time.sleep(10)
"
```

Expected: Status progresses BUILDING → APP_STARTING → RUNNING within ~5 minutes.

**Step 4: Commit the push confirmation**

No local file changes needed — the Space is deployed.

---

### Task 8: Verify all 6 tabs work

**Step 1: Test Tab 1 (Scenario Explorer) via gradio_client**

```bash
cd /home/crichalchemist/wave-experiment && python -c "
from gradio_client import Client
client = Client('crichalchemist/maninagarden')
result = client.predict('stable_community', 42, api_name='/explore_scenario')
print(f'Tab 1: {type(result)}')
print(f'Description: {result[0][:60]}...')
print('Tab 1: PASSED')
"
```

**Step 2: Test Tab 2 (Custom Forecast)**

```bash
cd /home/crichalchemist/wave-experiment && python -c "
from gradio_client import Client
client = Client('crichalchemist/maninagarden')
result = client.predict(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, api_name='/custom_forecast')
print(f'Tab 2: {result[0]}')
print('Tab 2: PASSED')
"
```

**Step 3: Visual verification of all tabs**

Open https://huggingface.co/spaces/crichalchemist/maninagarden in browser and check:
- Tab 1: Plots auto-populate on load
- Tab 2: Sliders work, forecast updates
- Tab 3: Experiment Lab shows comparison (both "latest" by default)
- Tab 4: Training tab shows password field, controls hidden until authenticated
- Tab 5: Data Workshop same pattern
- Tab 6: Research text includes recovery-aware floors section

---

### Task 9: Retrain with recovery-aware formula (separate HF Job)

**Context:** This is the final step — launch a training job using the new formula with recovery-aware floors. This can be done either from the Space's Training tab (after setting secrets) or from the CLI.

**Step 1: Launch training job from CLI**

```bash
cd /home/crichalchemist/wave-experiment && python -c "
import sys; sys.path.insert(0, 'spaces/maninagarden')
from training import get_training_script
from huggingface_hub import run_uv_job
import os

script = get_training_script(epochs=100, lr=5e-4, hidden_size=256, batch_size=64, scenarios_per_type=50)
print(f'Script length: {len(script)} chars')
print('Launching training job on t4-small...')

job = run_uv_job(
    script,
    flavor='t4-small',
    timeout='2h',
    secrets={'HF_TOKEN': os.environ.get('HF_TOKEN', '')},
)
print(f'Job launched: {job}')
print('Monitor at: https://huggingface.co/jobs')
"
```

**Step 2: Monitor job**

```bash
hf jobs ps
# When complete, check logs:
# hf jobs logs <job-id>
```

**Step 3: Verify new checkpoint**

After the job completes, verify the new checkpoint has recovery-aware metadata:

```bash
cd /home/crichalchemist/wave-experiment && python -c "
import sys; sys.path.insert(0, 'spaces/maninagarden')
from model import load_training_metadata
meta = load_training_metadata()
print(f'Formula version: {meta.get(\"formula_version\", \"unknown\")}')
print(f'Has recovery floors: {meta.get(\"has_recovery_floors\", False)}')
print(f'Val loss: {meta.get(\"best_val_loss\", \"?\")}')
print(f'Epochs: {meta.get(\"epochs\", \"?\")}')
"
```

Expected: `formula_version: 2.1-recovery-floors`, `has_recovery_floors: True`
