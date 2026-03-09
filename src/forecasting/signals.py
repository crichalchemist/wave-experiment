"""
Phi signal processor — 34+ feature construct signal extraction.

Computes welfare-specific signals: volatility, momentum, synergy,
divergence, and Phi trajectory for all 8 Phi(humanity) constructs.

Feature layout (34+ columns):
    8 raw constructs
    8 volatility (_vol)
    8 momentum (_mom)
    5 synergy (syn_a_b for each pair + curiosity cross-pair)
    5 divergence (div_a_b for each pair + curiosity cross-pair)
    1 phi (composite welfare score)
    1 dphi_dt (rate of change of Phi)
"""

import numpy as np
import pandas as pd

from src.inference.welfare_scoring import (
    ALL_CONSTRUCTS,
    PENALTY_PAIRS,
    compute_phi,
)


class PhiSignalProcessor:
    """
    Welfare-aware signal processor for Phi construct time-series.

    Provides volatility, momentum, synergy_signal, divergence_signal,
    and phi_derivative as static methods, plus ``compute_all_signals``
    which produces a 34+ column feature DataFrame from raw 8-construct
    input.
    """

    @staticmethod
    def volatility(values: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate rolling volatility (standard deviation)."""
        if len(values) < window:
            return np.zeros_like(values)

        result = np.zeros_like(values)
        for i in range(window, len(values)):
            result[i] = np.std(values[i - window : i])

        # Fill initial values with first valid volatility
        if window < len(values):
            result[:window] = result[window]

        return result

    @staticmethod
    def price_momentum(prices: np.ndarray, window: int = 10) -> np.ndarray:
        """Calculate price momentum (rate of change)."""
        if len(prices) < window:
            return np.zeros_like(prices)

        result = np.zeros_like(prices)
        for i in range(window, len(prices)):
            result[i] = (prices[i] - prices[i - window]) / prices[i - window]

        return result

    @staticmethod
    def synergy_signal(a: np.ndarray, b: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling geometric mean synergy between two construct time-series."""
        geo = np.sqrt(np.maximum(0, a) * np.maximum(0, b))
        if len(geo) < window:
            return geo
        result = np.zeros_like(geo)
        for i in range(window, len(geo)):
            result[i] = np.mean(geo[i - window : i])
        result[:window] = result[window] if window < len(geo) else 0
        return result

    @staticmethod
    def divergence_signal(a: np.ndarray, b: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling squared divergence between two construct time-series."""
        sq_diff = (a - b) ** 2
        if len(sq_diff) < window:
            return sq_diff
        result = np.zeros_like(sq_diff)
        for i in range(window, len(sq_diff)):
            result[i] = np.mean(sq_diff[i - window : i])
        result[:window] = result[window] if window < len(sq_diff) else 0
        return result

    @staticmethod
    def phi_derivative(phi: np.ndarray) -> np.ndarray:
        """Rate of change of Phi over time."""
        return np.gradient(phi)

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
