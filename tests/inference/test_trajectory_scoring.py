"""Test forecast-informed trajectory urgency scoring."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_score_hypothesis_trajectory_returns_float():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Resource allocation gap 2013-2017", 0.8)
    metrics = {"c": 0.3, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    mock_predictions = np.array([0.5, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_predictions):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert isinstance(urgency, float)
    assert 0.0 <= urgency <= 1.0


def test_declining_trajectory_gives_high_urgency():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Truth suppression in oversight records", 0.8)
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.2}

    mock_pred = np.array([0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_pred):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert urgency > 0.5


def test_stable_trajectory_gives_low_urgency():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Minor record discrepancy", 0.8)
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    mock_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_pred):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert urgency < 0.2


def test_rising_trajectory_gives_zero_urgency():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Recovery underway", 0.8)
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    mock_pred = np.array([0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_pred):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert urgency == 0.0


def test_prediction_failure_returns_zero():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Test", 0.8)
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    with patch("src.inference.welfare_scoring._get_trajectory_prediction", side_effect=RuntimeError("model unavailable")):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert urgency == 0.0
