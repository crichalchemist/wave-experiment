"""Integration test: full bridge pipeline with mocks at boundaries."""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import replace
import numpy as np


def test_layer1_classifier_scores_flow_to_welfare_scoring():
    """Classifier scores -> infer_threatened_constructs -> welfare_relevance."""
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import (
        score_hypothesis_welfare, infer_threatened_constructs,
    )

    mock_scores = {
        "c": 0.8, "kappa": 0.1, "j": 0.1, "p": 0.1,
        "eps": 0.1, "lam_L": 0.1, "lam_P": 0.1, "xi": 0.7,
    }

    with patch("src.inference.welfare_classifier.get_construct_scores", return_value=mock_scores):
        constructs = infer_threatened_constructs("Resource gap in records")
        assert "c" in constructs
        assert "xi" in constructs

    h = Hypothesis.create("Resource gap in records", 0.8)
    h = replace(h, threatened_constructs=constructs)
    phi_metrics = {"c": 0.2, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.3}
    welfare = score_hypothesis_welfare(h, phi_metrics)
    assert welfare > 0.3


def test_layer2_trajectory_urgency_into_hypothesis():
    """Trajectory urgency flows through combined_score."""
    from src.detective.hypothesis import Hypothesis

    h = Hypothesis.create("Declining welfare detected", 0.7)
    h = replace(h, welfare_relevance=0.5, curiosity_relevance=0.3, trajectory_urgency=0.8)

    score = h.combined_score(alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)
    assert score > 0.5
    score_no_traj = h.combined_score(alpha=0.45, beta=0.25, gamma=0.15, delta=0.0)
    assert score > score_no_traj


def test_layer3_extraction_produces_valid_scenarios():
    """Extraction pipeline -> templates -> generateable scenarios."""
    from src.inference.scenario_extraction import (
        identify_trajectory_patterns, generate_from_template,
    )

    profiles = [
        {"chunk_index": i, "scores": {
            "c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
            "eps": 0.5, "lam_L": max(0.1, 0.7 - i * 0.15), "lam_P": 0.5, "xi": 0.5,
        }}
        for i in range(6)
    ]

    patterns = identify_trajectory_patterns(profiles)
    assert len(patterns) >= 1

    for pattern in patterns:
        df = generate_from_template(pattern, length=200)
        assert len(df) == 200
        assert "phi" in df.columns
        assert all(df["phi"] > 0)


def test_trajectory_urgency_math():
    """Verify the urgency normalization formula."""
    from src.inference.welfare_scoring import score_hypothesis_trajectory
    from src.detective.hypothesis import Hypothesis

    h = Hypothesis.create("Test", 0.8)
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    # Steep decline: 0.5 -> 0.0 over 10 steps = slope -0.05/step
    mock_pred = np.linspace(0.5, 0.0, 10)
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_pred):
        urgency = score_hypothesis_trajectory(h, metrics)

    # decline = 0.05, k = 0.02, urgency = 0.05/(0.05+0.02) ~ 0.714
    expected = 0.05 / (0.05 + 0.02)  # ~ 0.714
    assert abs(urgency - expected) < 0.05  # within tolerance


def test_full_bridge_flow_with_mocked_boundaries():
    """End-to-end: hypothesis -> welfare score -> trajectory urgency -> combined score."""
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import (
        score_hypothesis_welfare, score_hypothesis_trajectory,
        infer_threatened_constructs,
    )

    # Create hypothesis about resource gaps
    h = Hypothesis.create("Temporal gap in financial oversight 2013-2017", 0.75)

    # Mock classifier to return high care + truth scores
    mock_scores = {"c": 0.8, "kappa": 0.2, "j": 0.1, "p": 0.1,
                   "eps": 0.2, "lam_L": 0.1, "lam_P": 0.3, "xi": 0.7}

    with patch("src.inference.welfare_classifier.get_construct_scores", return_value=mock_scores):
        constructs = infer_threatened_constructs(h.text)

    h = replace(h, threatened_constructs=constructs)

    # Score welfare (care and truth are scarce)
    phi_metrics = {"c": 0.15, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.2}
    welfare = score_hypothesis_welfare(h, phi_metrics)
    assert welfare > 0.4  # Both c and xi are scarce -> high relevance

    # Mock declining trajectory
    mock_pred = np.array([0.4, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_pred):
        urgency = score_hypothesis_trajectory(h, phi_metrics)
    assert urgency > 0.3  # Declining -> urgent

    # Set all scores on hypothesis
    h = replace(h, welfare_relevance=welfare, trajectory_urgency=urgency, curiosity_relevance=0.2)

    # Bridge-mode combined score
    score = h.combined_score(alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)
    assert 0.0 < score < 1.0
    assert score > h.confidence * 0.45  # Welfare and trajectory add value
