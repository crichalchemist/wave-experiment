"""Tests for Module B: historical determinism detection."""
import pytest
from unittest.mock import MagicMock


def test_import():
    from src.detective.module_b import DeterminismDetection, detect_historical_determinism
    assert DeterminismDetection is not None
    assert callable(detect_historical_determinism)


def test_returns_list():
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.1"
    result = detect_historical_determinism("Entity A always operated in New York.", provider)
    assert isinstance(result, list)


def test_detects_always_language():
    """'always' is a canonical historical determinism marker."""
    from src.detective.module_b import detect_historical_determinism, DeterminismDetection
    provider = MagicMock()
    provider.complete.return_value = "score: 0.85"
    results = detect_historical_determinism(
        "The organization always maintained these records.", provider
    )
    assert len(results) >= 1
    assert all(isinstance(r, DeterminismDetection) for r in results)
    assert results[0].score >= 0.7


def test_detects_continues_to_language():
    """'continues to' implies the past determines the present."""
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.80"
    results = detect_historical_determinism(
        "The board continues to deny any involvement.", provider
    )
    assert len(results) >= 1


def test_low_score_filtered():
    """Spans below threshold (0.5) must not be returned."""
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.2"
    results = detect_historical_determinism(
        "This report documents events in 2019.", provider
    )
    assert results == []


def test_detects_timestamp_as_causation():
    """Treating document date as event date is the core failure mode."""
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.9"
    results = detect_historical_determinism(
        "Because the memo was dated June 1, the meeting therefore occurred on June 1.", provider
    )
    assert len(results) >= 1
    assert results[0].score >= 0.7


def test_dataclass_is_frozen():
    from src.detective.module_b import DeterminismDetection
    from src.core.types import AssumptionType
    d = DeterminismDetection(
        assumption_type=AssumptionType.HISTORICAL_DETERMINISM,
        score=0.8,
        source_text="always maintained",
        trigger_phrase="always",
    )
    with pytest.raises(Exception):
        d.score = 0.5  # type: ignore[misc]


def test_multiple_triggers_in_one_text():
    """Multiple deterministic phrases → multiple detections."""
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.75"
    results = detect_historical_determinism(
        "The foundation always funded the program and continues to do so invariably.", provider
    )
    assert len(results) >= 2


def test_assumption_type_is_historical_determinism():
    from src.detective.module_b import detect_historical_determinism
    from src.core.types import AssumptionType
    provider = MagicMock()
    provider.complete.return_value = "score: 0.8"
    results = detect_historical_determinism("The records always showed this.", provider)
    assert all(r.assumption_type == AssumptionType.HISTORICAL_DETERMINISM for r in results)
