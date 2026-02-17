from unittest.mock import MagicMock, patch

import pytest

from src.core.types import AssumptionType
from src.detective.module_a import (
    BiasDetection,
    _CONFIDENCE_THRESHOLD,
    detect_cognitive_biases,
)


def _make_classifier_result(label: str, score: float) -> dict:
    return {"label": label, "score": score}


def test_detect_returns_empty_when_no_results_above_threshold() -> None:
    mock_pipeline = MagicMock(return_value=[
        _make_classifier_result("cognitive_bias", 0.3),
    ])
    with patch("src.detective.module_a._get_classifier", return_value=mock_pipeline):
        result = detect_cognitive_biases("some text")
    assert result == []


def test_detect_returns_detection_above_threshold() -> None:
    mock_pipeline = MagicMock(return_value=[
        _make_classifier_result("cognitive_bias", 0.85),
    ])
    with patch("src.detective.module_a._get_classifier", return_value=mock_pipeline):
        result = detect_cognitive_biases("some text")
    assert len(result) == 1
    assert result[0].assumption_type == AssumptionType.COGNITIVE_BIAS
    assert result[0].score == 0.85


def test_detect_filters_exactly_at_threshold() -> None:
    # score == threshold should be INCLUDED (>= not >)
    mock_pipeline = MagicMock(return_value=[
        _make_classifier_result("historical_determinism", _CONFIDENCE_THRESHOLD),
    ])
    with patch("src.detective.module_a._get_classifier", return_value=mock_pipeline):
        result = detect_cognitive_biases("some text")
    assert len(result) == 1
    assert result[0].assumption_type == AssumptionType.HISTORICAL_DETERMINISM


def test_detect_returns_multiple_detections() -> None:
    mock_pipeline = MagicMock(return_value=[
        _make_classifier_result("cognitive_bias", 0.9),
        _make_classifier_result("geopolitical_presumption", 0.75),
        _make_classifier_result("historical_determinism", 0.4),  # below threshold
    ])
    with patch("src.detective.module_a._get_classifier", return_value=mock_pipeline):
        result = detect_cognitive_biases("some text")
    assert len(result) == 2
    types = {d.assumption_type for d in result}
    assert AssumptionType.COGNITIVE_BIAS in types
    assert AssumptionType.GEOPOLITICAL_PRESUMPTION in types
