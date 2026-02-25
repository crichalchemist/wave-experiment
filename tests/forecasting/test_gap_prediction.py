"""Tests for Layer 4 — gap emergence prediction from forecasted construct levels."""

from src.forecasting.gap_prediction import predict_gap_emergence
from src.inference.welfare_scoring import ALL_CONSTRUCTS


def test_below_floor_flagged():
    """xi=0.1 is below its floor of 0.3, so it should be flagged as at-risk."""
    metrics = {c: 0.5 for c in ALL_CONSTRUCTS}
    metrics["xi"] = 0.1  # floor for xi is 0.30

    at_risk = predict_gap_emergence(metrics)

    assert "xi" in at_risk, f"xi=0.1 below floor=0.3 should be flagged, got {at_risk}"


def test_above_floor_not_flagged():
    """All constructs at 0.5 should be above all floors, so nothing flagged."""
    metrics = {c: 0.5 for c in ALL_CONSTRUCTS}

    at_risk = predict_gap_emergence(metrics)

    assert at_risk == [], f"All at 0.5 should produce empty list, got {at_risk}"


def test_multiple_at_risk():
    """All constructs at 0.05 should flag all 8 constructs."""
    metrics = {c: 0.05 for c in ALL_CONSTRUCTS}

    at_risk = predict_gap_emergence(metrics)

    assert len(at_risk) == 8, (
        f"All at 0.05 should flag all 8 constructs, got {len(at_risk)}: {at_risk}"
    )
    for c in ALL_CONSTRUCTS:
        assert c in at_risk, f"Construct {c} should be flagged at 0.05"
