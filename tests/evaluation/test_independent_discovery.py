"""Tests for the independent-discovery evaluation harness."""
import pytest

from evaluation.independent_discovery import (
    EMPTY_EVAL_DISCOVERY_RATE,
    MAX_PRECISION_THRESHOLD,
    MIN_PRECISION_THRESHOLD,
    DiscoveryEvaluation,
    EvaluationSummary,
    discovery_rate,
    f1_score,
    precision,
    recall,
    summarise,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make(
    gap_id: str,
    independently_discovered: bool,
    matches_system: bool,
    path_taken: str = "standard path",
) -> DiscoveryEvaluation:
    return DiscoveryEvaluation(
        gap_id=gap_id,
        independently_discovered=independently_discovered,
        path_taken=path_taken,
        matches_system=matches_system,
    )


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

def test_evaluation_constants() -> None:
    assert isinstance(MIN_PRECISION_THRESHOLD, float)
    assert isinstance(MAX_PRECISION_THRESHOLD, float)
    assert isinstance(EMPTY_EVAL_DISCOVERY_RATE, float)
    assert MIN_PRECISION_THRESHOLD == 0.0
    assert MAX_PRECISION_THRESHOLD == 1.0
    assert EMPTY_EVAL_DISCOVERY_RATE == 0.0


# ---------------------------------------------------------------------------
# 2. DiscoveryEvaluation frozen dataclass
# ---------------------------------------------------------------------------

def test_discovery_evaluation_is_frozen() -> None:
    item = _make("gap-1", independently_discovered=True, matches_system=True)
    with pytest.raises((AttributeError, TypeError)):
        item.gap_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3-6. discovery_rate
# ---------------------------------------------------------------------------

def test_discovery_rate_all_match() -> None:
    items = [
        _make("g1", independently_discovered=True, matches_system=True),
        _make("g2", independently_discovered=True, matches_system=True),
        _make("g3", independently_discovered=False, matches_system=True),
    ]
    assert discovery_rate(items) == pytest.approx(1.0)


def test_discovery_rate_none_match() -> None:
    items = [
        _make("g1", independently_discovered=False, matches_system=False),
        _make("g2", independently_discovered=True, matches_system=False),
        _make("g3", independently_discovered=True, matches_system=False),
    ]
    assert discovery_rate(items) == pytest.approx(0.0)


def test_discovery_rate_partial() -> None:
    items = [
        _make("g1", independently_discovered=True, matches_system=True),
        _make("g2", independently_discovered=False, matches_system=True),
        _make("g3", independently_discovered=True, matches_system=False),
    ]
    assert discovery_rate(items) == pytest.approx(2 / 3)


def test_discovery_rate_empty() -> None:
    assert discovery_rate([]) == EMPTY_EVAL_DISCOVERY_RATE


# ---------------------------------------------------------------------------
# 7-8. precision
# ---------------------------------------------------------------------------

def test_precision_perfect() -> None:
    items = [
        _make("g1", independently_discovered=True, matches_system=True),
        _make("g2", independently_discovered=True, matches_system=True),
    ]
    assert precision(items) == pytest.approx(1.0)


def test_precision_none_correct() -> None:
    # System found them (matches_system=True) but expert did not independently discover them.
    items = [
        _make("g1", independently_discovered=False, matches_system=True),
        _make("g2", independently_discovered=False, matches_system=True),
    ]
    assert precision(items) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. recall
# ---------------------------------------------------------------------------

def test_recall_full() -> None:
    # Expert found 2 gaps; system found both.
    items = [
        _make("g1", independently_discovered=True, matches_system=True),
        _make("g2", independently_discovered=True, matches_system=True),
    ]
    assert recall(items) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10-11. f1_score
# ---------------------------------------------------------------------------

def test_f1_perfect() -> None:
    items = [
        _make("g1", independently_discovered=True, matches_system=True),
        _make("g2", independently_discovered=True, matches_system=True),
    ]
    assert f1_score(items) == pytest.approx(1.0)


def test_f1_zero_when_both_zero() -> None:
    # No system matches, no expert discoveries — precision=0, recall=0 → f1=0.
    items = [
        _make("g1", independently_discovered=False, matches_system=False),
        _make("g2", independently_discovered=False, matches_system=False),
    ]
    assert f1_score(items) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 12-13. summarise
# ---------------------------------------------------------------------------

def test_summarise_returns_evaluation_summary() -> None:
    items = [_make("g1", independently_discovered=True, matches_system=True)]
    result = summarise(items)
    assert isinstance(result, EvaluationSummary)
    assert hasattr(result, "total")
    assert hasattr(result, "discovery_rate")
    assert hasattr(result, "precision")
    assert hasattr(result, "recall")
    assert hasattr(result, "f1")


def test_summarise_fields_match_individual_functions() -> None:
    items = [
        _make("g1", independently_discovered=True, matches_system=True),
        _make("g2", independently_discovered=False, matches_system=True),
        _make("g3", independently_discovered=True, matches_system=False),
    ]
    summary = summarise(items)
    assert summary.total == len(items)
    assert summary.discovery_rate == pytest.approx(discovery_rate(items))
    assert summary.precision == pytest.approx(precision(items))
    assert summary.recall == pytest.approx(recall(items))
    assert summary.f1 == pytest.approx(f1_score(items))
