"""
Layer 2 — Construct forecast helper.

Converts a raw [8] array of forecasted construct predictions into a named
dictionary keyed by the Phi construct symbols defined in welfare_scoring.
"""

import logging

from src.inference.welfare_scoring import ALL_CONSTRUCTS

_logger = logging.getLogger(__name__)


def forecasted_metrics_to_dict(construct_array) -> dict[str, float]:
    """Convert a [8] array of construct predictions to a named dict."""
    return {c: float(construct_array[i]) for i, c in enumerate(ALL_CONSTRUCTS)}
