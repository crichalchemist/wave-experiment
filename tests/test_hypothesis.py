"""Tests for hypothesis evolution."""

import pytest
from dataclasses import replace
from datetime import datetime
from src.detective.hypothesis import Hypothesis


def test_hypothesis_creation():
    """Test creating a hypothesis."""
    h = Hypothesis.create("Test hypothesis", 0.8)
    assert h.confidence == 0.8
    assert h.text == "Test hypothesis"
    assert h.parent_id is None


def test_hypothesis_update():
    """Test updating hypothesis confidence."""
    h1 = Hypothesis.create("Test", 0.8)
    h2 = h1.update_confidence(0.6)

    assert h2.confidence == 0.6
    assert h2.parent_id == h1.id
    assert h1.confidence == 0.8  # Original unchanged


def test_hypothesis_invalid_confidence_raises() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        Hypothesis.create(text="test", confidence=1.5)


def test_hypothesis_with_welfare_fields():
    """Hypothesis accepts welfare_relevance and threatened_constructs."""
    h = Hypothesis.create("Resource gap found", 0.8)
    h = replace(
        h,
        welfare_relevance=0.75,
        threatened_constructs=("c", "lam_P"),
    )
    assert h.welfare_relevance == 0.75
    assert h.threatened_constructs == ("c", "lam_P")


def test_hypothesis_welfare_fields_default():
    """Welfare fields default to 0.0 and () for backward compatibility."""
    h = Hypothesis.create("Test hypothesis", 0.6)
    assert h.welfare_relevance == 0.0
    assert h.threatened_constructs == ()


def test_hypothesis_welfare_relevance_validation():
    """Hypothesis.welfare_relevance must be in [0, 1]."""
    h = Hypothesis.create("Test", 0.5)

    with pytest.raises(ValueError, match="welfare_relevance must be in"):
        replace(h, welfare_relevance=1.5)

    with pytest.raises(ValueError, match="welfare_relevance must be in"):
        replace(h, welfare_relevance=-0.1)


def test_hypothesis_combined_score():
    """combined_score() computes alpha*confidence + beta*welfare + gamma*curiosity."""
    h = Hypothesis.create("Test", 0.8)
    h = replace(h, welfare_relevance=0.6)

    # Default alpha=0.55, beta=0.30, gamma=0.15, curiosity_relevance=0.0
    # 0.55*0.8 + 0.30*0.6 + 0.15*0.0 = 0.44 + 0.18 + 0.0 = 0.62
    assert h.combined_score() == pytest.approx(0.62)

    # Explicit old-style weights (backward compat: set gamma=0)
    assert h.combined_score(alpha=0.7, beta=0.3, gamma=0.0) == pytest.approx(0.74)


def test_hypothesis_curiosity_field_default():
    """Curiosity relevance defaults to 0.0 for backward compatibility."""
    h = Hypothesis.create("Test hypothesis", 0.6)
    assert h.curiosity_relevance == 0.0


def test_hypothesis_curiosity_field_accepts_value():
    """Curiosity relevance can be set via replace."""
    h = Hypothesis.create("Something doesn't add up", 0.5)
    h = replace(h, curiosity_relevance=0.8)
    assert h.curiosity_relevance == 0.8


def test_hypothesis_curiosity_relevance_validation():
    """Curiosity relevance must be in [0, 1]."""
    h = Hypothesis.create("Test", 0.5)
    with pytest.raises(ValueError, match="curiosity_relevance must be in"):
        replace(h, curiosity_relevance=1.5)
    with pytest.raises(ValueError, match="curiosity_relevance must be in"):
        replace(h, curiosity_relevance=-0.1)


def test_hypothesis_combined_score_with_curiosity():
    """combined_score includes curiosity term."""
    h = Hypothesis.create("Test", 0.8)
    h = replace(h, welfare_relevance=0.6, curiosity_relevance=1.0)

    # Default: alpha=0.55, beta=0.30, gamma=0.15
    # 0.55*0.8 + 0.30*0.6 + 0.15*1.0 = 0.44 + 0.18 + 0.15 = 0.77
    assert h.combined_score() == pytest.approx(0.77)


def test_hypothesis_curiosity_surfaces_hunches():
    """A low-confidence hunch with high curiosity outranks a boring high-confidence finding."""
    hunch = Hypothesis.create("Something doesn't add up in the financial records", 0.4)
    hunch = replace(hunch, welfare_relevance=0.3, curiosity_relevance=0.9)

    boring = Hypothesis.create("Meeting minutes confirm standard procedure", 0.6)
    boring = replace(boring, welfare_relevance=0.1, curiosity_relevance=0.0)

    # hunch: 0.55*0.4 + 0.30*0.3 + 0.15*0.9 = 0.22 + 0.09 + 0.135 = 0.445
    # boring: 0.55*0.6 + 0.30*0.1 + 0.15*0.0 = 0.33 + 0.03 + 0.00  = 0.36
    assert hunch.combined_score() > boring.combined_score()


def test_hypothesis_has_trajectory_urgency():
    h = Hypothesis.create("Test", 0.8)
    assert h.trajectory_urgency == 0.0


def test_trajectory_urgency_validation():
    with pytest.raises(ValueError, match="trajectory_urgency"):
        Hypothesis(
            id="x", text="t", confidence=0.5,
            timestamp=datetime.now(), trajectory_urgency=-0.1,
        )


def test_combined_score_with_trajectory_urgency():
    h = Hypothesis.create("Test", 0.8)
    h = replace(h, welfare_relevance=0.6, curiosity_relevance=0.4, trajectory_urgency=1.0)
    score = h.combined_score(alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)
    expected = 0.45 * 0.8 + 0.25 * 0.6 + 0.15 * 0.4 + 0.15 * 1.0
    assert abs(score - expected) < 1e-9


def test_combined_score_backward_compatible():
    h = Hypothesis.create("Test", 0.8)
    score_new = h.combined_score(alpha=0.55, beta=0.30, gamma=0.15, delta=0.0)
    score_old = 0.55 * 0.8 + 0.30 * 0.0 + 0.15 * 0.0
    assert abs(score_new - score_old) < 1e-9
