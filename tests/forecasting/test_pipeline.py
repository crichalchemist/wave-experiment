"""Tests for PhiPipeline — scaling + windowing for 36-feature construct signals."""

import numpy as np
import pytest

from src.forecasting.synthetic import PhiScenarioGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_df(length: int = 200):
    """Generate a stable_community trajectory for testing."""
    return PhiScenarioGenerator(seed=42).generate("stable_community", length=length)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPhiPipeline:
    """PhiPipeline: fit, transform, sequence creation, and end-to-end process."""

    def test_fit_transform_shape(self):
        """200-row DataFrame → [200, 36] scaled array."""
        from src.forecasting.pipeline import PhiPipeline

        df = _stable_df(200)
        pipeline = PhiPipeline(seq_len=100, window=20)
        X = pipeline.fit_transform(df)

        assert X.shape == (200, 36)
        assert pipeline.fitted is True
        assert len(pipeline.feature_names) == 36

    def test_create_sequences_shape(self):
        """With seq_len=50, [200, 36] → X_seq [151, 50, 36], y_seq [151]."""
        from src.forecasting.pipeline import PhiPipeline

        pipeline = PhiPipeline(seq_len=50, window=20)
        df = _stable_df(200)
        X = pipeline.fit_transform(df)

        y = np.arange(200, dtype=np.float64)
        X_seq, y_seq = pipeline.create_sequences(X, y, stride=1)

        # n_seq = (200 - 50) // 1 + 1 = 151
        assert X_seq.shape == (151, 50, 36)
        assert y_seq.shape == (151,)
        # y_seq[i] should be y[i * stride + seq_len - 1] = i + 49
        assert y_seq[0] == pytest.approx(49.0)
        assert y_seq[-1] == pytest.approx(199.0)

    def test_scaling_reversible(self):
        """Median of RobustScaler-transformed data should be near 0."""
        from src.forecasting.pipeline import PhiPipeline

        df = _stable_df(200)
        pipeline = PhiPipeline(seq_len=100, window=20)
        X = pipeline.fit_transform(df)

        # RobustScaler centres on median: median of each column should be ~0
        col_medians = np.median(X, axis=0)
        assert np.allclose(col_medians, 0.0, atol=1e-10)

    def test_pipeline_process(self):
        """End-to-end process() returns 3D X_seq and 1D y_seq."""
        from src.forecasting.pipeline import PhiPipeline

        df = _stable_df(200)
        labels = np.random.default_rng(42).random(200)

        pipeline = PhiPipeline(seq_len=100, window=20)
        X_seq, y_seq = pipeline.process(df, labels=labels, fit=True, stride=1)

        # n_seq = (200 - 100) // 1 + 1 = 101
        assert X_seq.ndim == 3
        assert X_seq.shape == (101, 100, 36)
        assert y_seq.shape == (101,)
