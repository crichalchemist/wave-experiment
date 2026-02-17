import pytest

from src.core.providers import MockProvider
from src.detective.evolution import (
    BRANCHING_THRESHOLD,
    BranchingStrategy,
    _classify_action,
    _parse_confidence,
    branching_rule,
    evolve_hypothesis,
)
from src.detective.experience import EMPTY_LIBRARY
from src.detective.hypothesis import Hypothesis


def _make_hypothesis(text: str = "Money was laundered", confidence: float = 0.6) -> Hypothesis:
    return Hypothesis.create(text=text, confidence=confidence)


def test_branching_rule_returns_breadth_below_threshold() -> None:
    assert branching_rule(0.0) == "breadth"
    assert branching_rule(BRANCHING_THRESHOLD - 0.01) == "breadth"


def test_branching_rule_returns_depth_at_and_above_threshold() -> None:
    assert branching_rule(BRANCHING_THRESHOLD) == "depth"
    assert branching_rule(1.0) == "depth"


def test_branching_rule_raises_on_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        branching_rule(1.5)


def test_classify_action_confirmed_on_positive_delta() -> None:
    assert _classify_action(0.5, 0.7) == "confirmed"


def test_classify_action_refuted_on_negative_delta() -> None:
    assert _classify_action(0.7, 0.5) == "refuted"


def test_classify_action_spawned_on_zero_delta() -> None:
    assert _classify_action(0.5, 0.5) == "spawned_alternative"


def test_parse_confidence_valid_float() -> None:
    assert _parse_confidence("0.75") == pytest.approx(0.75)
    assert _parse_confidence("  0.0  ") == pytest.approx(0.0)
    assert _parse_confidence("1.0") == pytest.approx(1.0)


def test_parse_confidence_returns_none_for_invalid() -> None:
    assert _parse_confidence("I cannot determine") is None
    assert _parse_confidence("1.5") is None
    assert _parse_confidence("") is None


def test_evolve_hypothesis_returns_evolved_and_experience() -> None:
    h = _make_hypothesis(confidence=0.6)
    provider = MockProvider(response="0.8")
    evolved, experience = evolve_hypothesis(h, "Strong corroborating evidence", EMPTY_LIBRARY, provider)
    assert evolved.confidence == pytest.approx(0.8)
    assert evolved.parent_id == h.id
    assert experience.action == "confirmed"
    assert experience.hypothesis_id == h.id


def test_evolve_hypothesis_unparseable_response_applies_decay() -> None:
    h = _make_hypothesis(confidence=0.6)
    provider = MockProvider(response="I cannot determine confidence")
    evolved, experience = evolve_hypothesis(h, "Ambiguous evidence", EMPTY_LIBRARY, provider)
    assert evolved.confidence < h.confidence
    assert experience.action == "refuted"


def test_evolve_hypothesis_does_not_mutate_original() -> None:
    h = _make_hypothesis(confidence=0.6)
    provider = MockProvider(response="0.9")
    _, _ = evolve_hypothesis(h, "Evidence", EMPTY_LIBRARY, provider)
    assert h.confidence == pytest.approx(0.6)  # original unchanged
