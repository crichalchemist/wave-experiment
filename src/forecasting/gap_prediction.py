"""
Layer 4 — Gap emergence prediction.

Predicts which Phi constructs will produce information gaps by comparing
forecasted construct levels against their hard floors. Constructs projected
to fall below their floor are flagged as at-risk for gap emergence.
"""

from typing import Dict, List

from src.inference.welfare_scoring import ALL_CONSTRUCTS, CONSTRUCT_FLOORS


def predict_gap_emergence(future_metrics: Dict[str, float]) -> List[str]:
    """Predict which constructs will produce information gaps."""
    return [
        c
        for c in ALL_CONSTRUCTS
        if future_metrics.get(c, 0.5) < CONSTRUCT_FLOORS.get(c, 0.1)
    ]
