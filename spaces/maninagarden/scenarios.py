"""
Scenario generation and signal processing for Phi Forecaster.

Extracted from the monolithic app.py so that welfare scoring lives in
welfare.py (single source of truth) and scenarios import compute_phi
from there.  compute_phi calls now pass finite-difference derivatives
so the v2.1 recovery-aware floor logic activates during data generation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from welfare import compute_phi, ALL_CONSTRUCTS, PENALTY_PAIRS


# ============================================================================
# Signal processing (volatility, momentum, synergy, divergence)
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
    """Compute 36 features: 8 raw + 8 vol + 8 mom + 5 synergy + 5 divergence + phi + dphi_dt."""
    out = df[list(ALL_CONSTRUCTS)].copy()
    for c in ALL_CONSTRUCTS:
        vals = out[c].values.astype(np.float64)
        out[f"{c}_vol"] = volatility(vals, window)
        out[f"{c}_mom"] = price_momentum(vals, window)
    for a, b in PENALTY_PAIRS:
        out[f"syn_{a}_{b}"] = synergy_signal(df[a].values, df[b].values, window)
        out[f"div_{a}_{b}"] = divergence_signal(df[a].values, df[b].values, window)

    # Phi with finite-difference derivatives
    n = len(out)
    phi_vals = np.empty(n, dtype=np.float64)
    prev_metrics = None
    for idx in range(n):
        row = out.iloc[idx]
        metrics = {c: row[c] for c in ALL_CONSTRUCTS}
        if idx == 0 or prev_metrics is None:
            derivs = {}
        else:
            derivs = {c: metrics[c] - prev_metrics[c] for c in ALL_CONSTRUCTS}
        phi_vals[idx] = compute_phi(metrics, derivatives=derivs)
        prev_metrics = metrics

    out["phi"] = phi_vals
    out["dphi_dt"] = np.gradient(phi_vals)
    return out


# ============================================================================
# Scenario definitions
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


# ============================================================================
# Scenario generators
# ============================================================================

def generate_scenario(scenario, length=200, rng=None):
    """Generate a synthetic welfare trajectory for the given scenario.

    Returns a DataFrame with columns for each construct plus 'phi'.
    Phi is computed using recovery-aware welfare.compute_phi with
    finite-difference derivatives passed at each timestep.
    """
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

    # Compute phi with finite-difference derivatives at each timestep
    n = len(df)
    phi_vals = np.empty(n, dtype=np.float64)
    prev_metrics = None
    for idx in range(n):
        metrics = {c: df.at[idx, c] for c in ALL_CONSTRUCTS}
        if idx == 0 or prev_metrics is None:
            derivs = {}
        else:
            derivs = {c: metrics[c] - prev_metrics[c] for c in ALL_CONSTRUCTS}
        phi_vals[idx] = compute_phi(metrics, derivatives=derivs)
        prev_metrics = metrics

    df["phi"] = phi_vals
    return df


# ============================================================================
# Reference scaler
# ============================================================================

def build_reference_scaler(seed=42):
    """Build RobustScaler from the stable_community reference scenario."""
    rng = np.random.default_rng(seed)
    df_ref = generate_scenario("stable_community", length=200, rng=rng)
    features_ref = compute_all_signals(df_ref, window=20)
    scaler = RobustScaler()
    scaler.fit(features_ref.values)
    return scaler, list(features_ref.columns)
