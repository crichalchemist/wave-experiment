"""
Phi signal processor — 34+ feature construct signal extraction.

Extends Dignity-Model's SignalProcessor with welfare-specific signals:
volatility, momentum, synergy, divergence, and Phi trajectory for all
8 Phi(humanity) constructs.

Feature layout (34+ columns):
    8 raw constructs
    8 volatility (_vol)
    8 momentum (_mom)
    5 synergy (syn_a_b for each pair + curiosity cross-pair)
    5 divergence (div_a_b for each pair + curiosity cross-pair)
    1 phi (composite welfare score)
    1 dphi_dt (rate of change of Phi)
"""

import importlib.util
import os

import numpy as np
import pandas as pd

# Import Dignity-Model's SignalProcessor by absolute path to avoid
# namespace collision with wave-experiment's own tests/core/ package.
_signals_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "Dignity-Model", "core", "signals.py"
)
_signals_path = os.path.abspath(_signals_path)
_spec = importlib.util.spec_from_file_location("dignity_core_signals", _signals_path)
_dignity_signals_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dignity_signals_mod)
_DignitySignals = _dignity_signals_mod.SignalProcessor
from src.inference.welfare_scoring import (
    ALL_CONSTRUCTS,
    CONSTRUCT_PAIRS,
    CURIOSITY_CROSS_PAIR,
    PENALTY_PAIRS,
    compute_phi,
)


class PhiSignalProcessor(_DignitySignals):
    """
    Welfare-aware signal processor for Phi construct time-series.

    Inherits Dignity's volatility, momentum, directional_change, regime,
    synergy_signal, divergence_signal, and phi_derivative.  Adds
    ``compute_all_signals`` which produces a 34+ column feature DataFrame
    from raw 8-construct input.
    """

    @classmethod
    def compute_all_signals(
        cls,
        df: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Compute all signal features from a construct-level DataFrame.

        Args:
            df: DataFrame with columns matching ALL_CONSTRUCTS (8 cols).
            window: Rolling window for volatility, momentum, synergy, divergence.

        Returns:
            DataFrame with 34+ columns:
                - 8 raw constructs
                - 8 {c}_vol  (rolling volatility)
                - 8 {c}_mom  (rolling momentum / rate-of-change)
                - 5 syn_{a}_{b}  (rolling geometric-mean synergy per pair)
                - 5 div_{a}_{b}  (rolling squared-divergence per pair)
                - 1 phi  (composite welfare score per row)
                - 1 dphi_dt  (rate of change of phi)
        """
        missing = [c for c in ALL_CONSTRUCTS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing construct columns: {missing}")

        out = df[list(ALL_CONSTRUCTS)].copy()

        # --- Per-construct signals -------------------------------------------
        for c in ALL_CONSTRUCTS:
            vals = out[c].values.astype(np.float64)
            out[f"{c}_vol"] = cls.volatility(vals, window=window)
            out[f"{c}_mom"] = cls.price_momentum(vals, window=window)

        # --- Paired synergy signals ------------------------------------------
        for a, b in PENALTY_PAIRS:
            a_vals = out[a].values.astype(np.float64)
            b_vals = out[b].values.astype(np.float64)
            out[f"syn_{a}_{b}"] = cls.synergy_signal(a_vals, b_vals, window=window)
            out[f"div_{a}_{b}"] = cls.divergence_signal(a_vals, b_vals, window=window)

        # --- Composite Phi per row -------------------------------------------
        phi_vals = np.array(
            [
                compute_phi({c: row[c] for c in ALL_CONSTRUCTS})
                for _, row in out[list(ALL_CONSTRUCTS)].iterrows()
            ],
            dtype=np.float64,
        )
        out["phi"] = phi_vals

        # --- Phi derivative --------------------------------------------------
        out["dphi_dt"] = cls.phi_derivative(phi_vals)

        return out
