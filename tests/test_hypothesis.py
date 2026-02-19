"""Tests for hypothesis evolution."""

import pytest
from dataclasses import replace
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
        threatened_constructs=("c", "lam"),
    )
    assert h.welfare_relevance == 0.75
    assert h.threatened_constructs == ("c", "lam")


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
    """combined_score() computes α·confidence + β·welfare."""
    h = Hypothesis.create("Test", 0.8)
    h = replace(h, welfare_relevance=0.6)

    # Default α=0.7, β=0.3
    # 0.7*0.8 + 0.3*0.6 = 0.56 + 0.18 = 0.74
    assert h.combined_score() == pytest.approx(0.74)

    # Custom weights
    assert h.combined_score(alpha=0.5, beta=0.5) == pytest.approx(0.7)
