"""Tests for synthetic Phi trajectory generator."""

import pytest
import pandas as pd

from src.forecasting.synthetic import PhiScenarioGenerator
from src.inference.welfare_scoring import ALL_CONSTRUCTS, compute_phi


@pytest.fixture
def gen():
    """Create a generator with fixed seed for reproducibility."""
    return PhiScenarioGenerator(seed=42)


# ── Shape and schema ──────────────────────────────────────────────────────

def test_stable_community_shape(gen):
    """Stable community scenario returns 200 rows with all 8 construct columns."""
    df = gen.generate("stable_community", length=200)
    assert len(df) == 200
    for c in ALL_CONSTRUCTS:
        assert c in df.columns, f"Missing column: {c}"
    assert "phi" in df.columns


# ── Bounds ────────────────────────────────────────────────────────────────

def test_all_constructs_in_bounds(gen):
    """All construct values must be in [0, 1] across every scenario."""
    for scenario in PhiScenarioGenerator.SCENARIOS:
        df = gen.generate(scenario, length=200)
        for c in ALL_CONSTRUCTS:
            assert df[c].min() >= 0.0, f"{scenario}/{c} below 0"
            assert df[c].max() <= 1.0, f"{scenario}/{c} above 1"


# ── Phi computation ───────────────────────────────────────────────────────

def test_phi_column_computed(gen):
    """Phi column must match compute_phi() applied to the first row."""
    df = gen.generate("stable_community", length=50)
    row = df.iloc[0]
    metrics = {c: float(row[c]) for c in ALL_CONSTRUCTS}
    expected = compute_phi(metrics)
    assert abs(df["phi"].iloc[0] - expected) < 1e-9


# ── Scenario-specific invariants ──────────────────────────────────────────

def test_capitalism_suppresses_love(gen):
    """In capitalism_suppresses_love, lam_L must decline and end below 0.2."""
    df = gen.generate("capitalism_suppresses_love", length=200)
    first_quarter = df["lam_L"].iloc[:50].mean()
    last_quarter = df["lam_L"].iloc[-50:].mean()
    assert last_quarter < first_quarter, "lam_L should decline"
    assert df["lam_L"].iloc[-1] < 0.2, "lam_L should end below 0.2"


def test_surveillance_state_divergence(gen):
    """In surveillance_state, xi rises while lam_L falls."""
    df = gen.generate("surveillance_state", length=200)
    xi_start = df["xi"].iloc[:20].mean()
    xi_end = df["xi"].iloc[-20:].mean()
    lam_start = df["lam_L"].iloc[:20].mean()
    lam_end = df["lam_L"].iloc[-20:].mean()
    assert xi_end > xi_start, "xi should rise"
    assert lam_end < lam_start, "lam_L should fall"


def test_recovery_arc(gen):
    """In recovery_arc, phi dips then recovers."""
    df = gen.generate("recovery_arc", length=200)
    early = df["phi"].iloc[:30].mean()
    mid = df["phi"].iloc[80:120].mean()
    late = df["phi"].iloc[-30:].mean()
    assert mid < early, "phi should dip in the middle"
    assert late > mid, "phi should recover after the dip"


# ── Dataset generation ────────────────────────────────────────────────────

def test_generate_dataset(gen):
    """generate_dataset returns correct count of DataFrames."""
    dataset = gen.generate_dataset(scenarios_per_type=3, length=100)
    expected_count = len(PhiScenarioGenerator.SCENARIOS) * 3
    assert len(dataset) == expected_count
    assert all(isinstance(df, pd.DataFrame) for df in dataset)


# ── Error handling ────────────────────────────────────────────────────────

def test_unknown_scenario_raises(gen):
    """Unknown scenario name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown scenario"):
        gen.generate("nonexistent_scenario")
