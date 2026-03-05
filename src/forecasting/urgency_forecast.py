"""
Layer 3 — Urgency forecasting.

Predicts future urgency by scoring a hypothesis against forecasted
construct levels. Uses the existing welfare scoring machinery to
evaluate how urgent a hypothesis becomes under projected future conditions.
"""

from src.detective.hypothesis import Hypothesis
from src.inference.welfare_scoring import score_hypothesis_welfare


def forecast_hypothesis_urgency(
    hypothesis: Hypothesis,
    future_metrics: dict[str, float],
) -> float:
    """Predict future urgency by scoring hypothesis against forecasted construct levels."""
    return score_hypothesis_welfare(hypothesis, future_metrics)
