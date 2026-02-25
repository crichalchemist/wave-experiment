"""Tests for PhiSignalProcessor — 34-feature construct signal extraction."""

import numpy as np
import pandas as pd
import pytest

from src.inference.welfare_scoring import ALL_CONSTRUCTS, compute_phi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_construct_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic DataFrame with all 8 construct columns."""
    rng = np.random.RandomState(seed)
    data = {col: rng.uniform(0.2, 0.9, size=n) for col in ALL_CONSTRUCTS}
    return pd.DataFrame(data)


def _uptrend(n: int = 100) -> np.ndarray:
    """Monotonically increasing series from 0.1 to 0.9."""
    return np.linspace(0.1, 0.9, n)


# ---------------------------------------------------------------------------
# Dignity-Model signal method tests
# ---------------------------------------------------------------------------

class TestDignitySignalMethods:
    """Test the 3 new methods added to Dignity's SignalProcessor."""

    def test_construct_volatility(self):
        """Output len matches input; non-zero after window."""
        from src.forecasting.signals import PhiSignalProcessor as SignalProcessor

        arr = np.random.RandomState(0).uniform(0.2, 0.8, size=60)
        vol = SignalProcessor.volatility(arr, window=20)
        assert len(vol) == len(arr)
        # After the window, volatility should be non-zero for random data
        assert np.any(vol[20:] > 0)

    def test_construct_momentum(self):
        """Positive for an uptrend series."""
        from src.forecasting.signals import PhiSignalProcessor as SignalProcessor

        arr = _uptrend(60)
        mom = SignalProcessor.price_momentum(arr, window=10)
        # After window, momentum should be positive for uptrend
        assert np.all(mom[10:] > 0)

    def test_synergy_signal(self):
        """Correct length and non-negative output."""
        from src.forecasting.signals import PhiSignalProcessor as SignalProcessor

        a = np.random.RandomState(1).uniform(0.3, 0.8, size=50)
        b = np.random.RandomState(2).uniform(0.3, 0.8, size=50)
        syn = SignalProcessor.synergy_signal(a, b, window=10)
        assert len(syn) == 50
        assert np.all(syn >= 0)

    def test_synergy_penalizes_imbalance(self):
        """Balanced pair produces higher synergy than imbalanced pair."""
        from src.forecasting.signals import PhiSignalProcessor as SignalProcessor

        n = 60
        window = 10
        # Balanced: both at 0.6
        balanced_a = np.full(n, 0.6)
        balanced_b = np.full(n, 0.6)
        # Imbalanced: one high, one low (same arithmetic mean)
        imbalanced_a = np.full(n, 0.9)
        imbalanced_b = np.full(n, 0.1)

        syn_bal = SignalProcessor.synergy_signal(balanced_a, balanced_b, window=window)
        syn_imb = SignalProcessor.synergy_signal(imbalanced_a, imbalanced_b, window=window)

        # Geometric mean favors balance: sqrt(0.6*0.6) > sqrt(0.9*0.1)
        assert syn_bal[window + 1] > syn_imb[window + 1]

    def test_divergence_signal(self):
        """Large divergence for a divergent pair."""
        from src.forecasting.signals import PhiSignalProcessor as SignalProcessor

        n = 60
        window = 10
        a = np.full(n, 0.8)
        b = np.full(n, 0.2)
        div = SignalProcessor.divergence_signal(a, b, window=window)
        # (0.8 - 0.2)^2 = 0.36
        assert div[window + 1] == pytest.approx(0.36, abs=1e-6)

    def test_phi_derivative(self):
        """Positive derivative for increasing Phi series."""
        from src.forecasting.signals import PhiSignalProcessor as SignalProcessor

        phi = np.linspace(0.3, 0.9, 50)
        dphi = SignalProcessor.phi_derivative(phi)
        assert len(dphi) == 50
        # gradient of a linearly increasing series should be positive everywhere
        assert np.all(dphi > 0)


# ---------------------------------------------------------------------------
# PhiSignalProcessor adapter tests
# ---------------------------------------------------------------------------

class TestPhiSignalProcessor:
    """Test the wave-experiment adapter that produces 34+ features."""

    def test_compute_all_signals(self):
        """Result has 34+ columns and correct row count."""
        from src.forecasting.signals import PhiSignalProcessor

        df = _make_construct_df(100)
        result = PhiSignalProcessor.compute_all_signals(df, window=20)

        assert len(result) == 100
        # 8 raw + 8 volatility + 8 momentum + 5 synergy + 5 divergence + 1 phi = 35
        assert len(result.columns) >= 34

    def test_raw_columns_present(self):
        """All 8 raw construct columns present in output."""
        from src.forecasting.signals import PhiSignalProcessor

        df = _make_construct_df(50)
        result = PhiSignalProcessor.compute_all_signals(df, window=10)
        for c in ALL_CONSTRUCTS:
            assert c in result.columns

    def test_volatility_columns(self):
        """All 8 volatility columns present."""
        from src.forecasting.signals import PhiSignalProcessor

        df = _make_construct_df(50)
        result = PhiSignalProcessor.compute_all_signals(df, window=10)
        for c in ALL_CONSTRUCTS:
            assert f"{c}_vol" in result.columns

    def test_momentum_columns(self):
        """All 8 momentum columns present."""
        from src.forecasting.signals import PhiSignalProcessor

        df = _make_construct_df(50)
        result = PhiSignalProcessor.compute_all_signals(df, window=10)
        for c in ALL_CONSTRUCTS:
            assert f"{c}_mom" in result.columns

    def test_synergy_columns(self):
        """5 synergy columns (4 pairs + 1 curiosity cross-pair)."""
        from src.forecasting.signals import PhiSignalProcessor

        df = _make_construct_df(50)
        result = PhiSignalProcessor.compute_all_signals(df, window=10)
        expected_synergy = [
            "syn_c_lam_L", "syn_kappa_lam_P", "syn_j_p", "syn_eps_xi",
            "syn_lam_L_xi",
        ]
        for col in expected_synergy:
            assert col in result.columns

    def test_divergence_columns(self):
        """5 divergence columns (4 pairs + 1 curiosity cross-pair)."""
        from src.forecasting.signals import PhiSignalProcessor

        df = _make_construct_df(50)
        result = PhiSignalProcessor.compute_all_signals(df, window=10)
        expected_div = [
            "div_c_lam_L", "div_kappa_lam_P", "div_j_p", "div_eps_xi",
            "div_lam_L_xi",
        ]
        for col in expected_div:
            assert col in result.columns

    def test_phi_column(self):
        """Phi column present and positive."""
        from src.forecasting.signals import PhiSignalProcessor

        df = _make_construct_df(50)
        result = PhiSignalProcessor.compute_all_signals(df, window=10)
        assert "phi" in result.columns
        assert np.all(result["phi"].values >= 0)

    def test_phi_derivative_column(self):
        """dphi_dt column present in output."""
        from src.forecasting.signals import PhiSignalProcessor

        df = _make_construct_df(50)
        result = PhiSignalProcessor.compute_all_signals(df, window=10)
        assert "dphi_dt" in result.columns
