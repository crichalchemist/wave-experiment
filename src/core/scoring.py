"""Shared scoring utilities for LLM response parsing.

Consolidates the _parse_score() pattern used across detective modules A, B, C
and parallel_evolution into a single source of truth.
"""
from __future__ import annotations

import re

# Accepts "score:", "confidence:", with ":" or "=" separator.
SCORE_RE = re.compile(
    r"(?:score|confidence)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)


def clamp_confidence(value: float) -> float:
    """Clamp a confidence/score value to [0.0, 1.0]."""
    return min(1.0, max(0.0, value))


def parse_score(response: str, default: float = 0.0) -> float:
    """Extract and clamp a float score from an LLM response.

    Handles formats: "score: 0.85", "confidence = 0.7", "score:0.9"
    Returns default if no match found. Always returns a value in [0.0, 1.0].
    """
    match = SCORE_RE.search(response)
    if not match:
        return default
    try:
        return clamp_confidence(float(match.group(1)))
    except ValueError:
        return default
