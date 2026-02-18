"""Independent-discovery evaluation harness for gap detection quality.

Compares system-detected gaps against expert-independently-discovered gaps,
yielding standard IR metrics (precision, recall, F1) plus a discovery rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Named constants — no magic numbers
# ---------------------------------------------------------------------------

MIN_PRECISION_THRESHOLD: float = 0.0
MAX_PRECISION_THRESHOLD: float = 1.0
EMPTY_EVAL_DISCOVERY_RATE: float = 0.0

_ZERO_DENOMINATOR_FALLBACK: float = 0.0
_F1_MULTIPLIER: float = 2.0


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveryEvaluation:
    """One gap assessment record.

    independently_discovered: True when a human expert found this gap
        without seeing the system output first (ground-truth label).
    matches_system: True when the system's output contained this gap.
    path_taken: Free-text narrative of how the expert reached their conclusion
        (audit trail, not used in metric computation).
    """

    gap_id: str
    independently_discovered: bool
    path_taken: str
    matches_system: bool


# ---------------------------------------------------------------------------
# Metric functions — all pure, no side effects
# ---------------------------------------------------------------------------

def discovery_rate(items: Sequence[DiscoveryEvaluation]) -> float:
    """Fraction of records where the system found the gap (matches_system=True).

    Returns EMPTY_EVAL_DISCOVERY_RATE when the sequence is empty so callers
    never receive a ZeroDivisionError.
    """
    if not items:
        return EMPTY_EVAL_DISCOVERY_RATE
    matched = sum(1 for item in items if item.matches_system)
    return matched / len(items)


def precision(items: Sequence[DiscoveryEvaluation]) -> float:
    """Among gaps the system found, the fraction that were truly real.

    Numerator:   matches_system=True AND independently_discovered=True
    Denominator: matches_system=True

    Returns 0.0 when no system matches exist (undefined → pessimistic default).
    """
    system_found = [item for item in items if item.matches_system]
    if not system_found:
        return _ZERO_DENOMINATOR_FALLBACK
    true_positives = sum(1 for item in system_found if item.independently_discovered)
    return true_positives / len(system_found)


def recall(items: Sequence[DiscoveryEvaluation]) -> float:
    """Among all real gaps (expert-discovered), the fraction the system found.

    Numerator:   matches_system=True AND independently_discovered=True
    Denominator: independently_discovered=True

    Returns 0.0 when no expert discoveries exist (undefined → pessimistic default).
    """
    expert_found = [item for item in items if item.independently_discovered]
    if not expert_found:
        return _ZERO_DENOMINATOR_FALLBACK
    true_positives = sum(1 for item in expert_found if item.matches_system)
    return true_positives / len(expert_found)


def f1_score(items: Sequence[DiscoveryEvaluation]) -> float:
    """Harmonic mean of precision and recall.

    Returns 0.0 when both precision and recall are zero to avoid division by zero.
    """
    p = precision(items)
    r = recall(items)
    denominator = p + r
    if denominator == _ZERO_DENOMINATOR_FALLBACK:
        return _ZERO_DENOMINATOR_FALLBACK
    return _F1_MULTIPLIER * (p * r) / denominator


# ---------------------------------------------------------------------------
# Summary container + aggregation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationSummary:
    """Immutable snapshot of all gap-detection metrics for a run."""

    total: int
    discovery_rate: float
    precision: float
    recall: float
    f1: float


def summarise(items: Sequence[DiscoveryEvaluation]) -> EvaluationSummary:
    """Compute all metrics in one pass and return as an immutable summary."""
    return EvaluationSummary(
        total=len(items),
        discovery_rate=discovery_rate(items),
        precision=precision(items),
        recall=recall(items),
        f1=f1_score(items),
    )
