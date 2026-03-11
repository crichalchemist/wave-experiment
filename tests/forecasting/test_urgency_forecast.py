"""Tests for Layer 3 — urgency forecasting via future construct levels."""

from src.detective.hypothesis import Hypothesis
from src.forecasting.urgency_forecast import forecast_hypothesis_urgency
from src.inference.welfare_scoring import ALL_CONSTRUCTS


def test_declining_constructs_increase_urgency():
    """Declining the threatened construct (care) should increase urgency.

    When all constructs decline uniformly, equity weights redistribute
    evenly and the urgency can actually drop. The meaningful signal is
    a *targeted* decline in the construct the hypothesis threatens.
    """
    h = Hypothesis.create("Resource allocation gap in financial records", 0.8)
    current_metrics = {c: 0.5 for c in ALL_CONSTRUCTS}
    # Only the threatened construct (care — "c") declines
    future_metrics = {c: 0.5 for c in ALL_CONSTRUCTS}
    future_metrics["c"] = 0.1  # care becomes scarce

    current_urgency = forecast_hypothesis_urgency(h, current_metrics)
    future_urgency = forecast_hypothesis_urgency(h, future_metrics)

    assert future_urgency >= current_urgency, (
        f"Declining threatened construct should increase urgency: "
        f"future={future_urgency} < current={current_urgency}"
    )


def test_stable_constructs_stable_urgency():
    """Urgency should be in [0, 1] for stable construct levels."""
    h = Hypothesis.create("Temporal gap in financial records 2013-2017", 0.8)
    stable_metrics = {c: 0.5 for c in ALL_CONSTRUCTS}

    result = forecast_hypothesis_urgency(h, stable_metrics)

    assert 0.0 <= result <= 1.0, f"Urgency should be in [0, 1], got {result}"
