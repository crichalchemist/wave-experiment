from typing import Literal

import pytest
from src.detective.experience import (
    EMPTY_LIBRARY,
    Experience,
    ExperienceLibrary,
    add_experience,
    query_similar,
)


def _make_experience(
    hypothesis_id: str = "h1",
    hypothesis_text: str = "the hypothesis text",
    evidence: str = "doc evidence",
    action: Literal["confirmed", "refuted", "spawned_alternative"] = "confirmed",
    confidence_delta: float = 0.1,
    outcome_quality: float = 0.8,
) -> Experience:
    return Experience(
        hypothesis_id=hypothesis_id,
        hypothesis_text=hypothesis_text,
        evidence=evidence,
        action=action,
        confidence_delta=confidence_delta,
        outcome_quality=outcome_quality,
    )


def test_experience_is_frozen() -> None:
    from dataclasses import FrozenInstanceError
    exp = _make_experience()
    with pytest.raises(FrozenInstanceError):
        exp.outcome_quality = 0.5  # type: ignore[misc]


def test_empty_library_is_empty_tuple() -> None:
    assert EMPTY_LIBRARY == ()
    assert isinstance(EMPTY_LIBRARY, tuple)


def test_add_experience_returns_new_library() -> None:
    exp = _make_experience()
    new_lib = add_experience(EMPTY_LIBRARY, exp)
    assert len(new_lib) == 1
    assert new_lib[0] is exp
    assert EMPTY_LIBRARY == ()  # original unchanged


def test_add_experience_accumulates() -> None:
    lib = EMPTY_LIBRARY
    for i in range(5):
        lib = add_experience(lib, _make_experience(hypothesis_id=f"h{i}"))
    assert len(lib) == 5


def test_query_similar_returns_top_k() -> None:
    lib = (
        _make_experience(hypothesis_text="money laundering scheme", evidence="wire transfer", hypothesis_id="h1"),
        _make_experience(hypothesis_text="money laundering via shell company", evidence="offshore account", hypothesis_id="h2"),
        _make_experience(hypothesis_text="foreign influence network", evidence="lobbying records", hypothesis_id="h3"),
    )
    results = query_similar(lib, "money laundering scheme", "wire transfer pattern", top_k=2)
    assert len(results) == 2
    texts = {r.hypothesis_text for r in results}
    assert "foreign influence network" not in texts


def test_query_similar_empty_library_returns_empty() -> None:
    results = query_similar(EMPTY_LIBRARY, "anything", "anything")
    assert results == ()


def test_experience_invalid_action_raises() -> None:
    with pytest.raises(ValueError, match="action must be one of"):
        Experience(
            hypothesis_id="h1",
            hypothesis_text="some hypothesis",
            evidence="e1",
            action="invalid",  # type: ignore[arg-type]
            confidence_delta=0.1,
            outcome_quality=0.5,
        )


def test_experience_invalid_outcome_quality_raises() -> None:
    with pytest.raises(ValueError, match="outcome_quality must be in"):
        _make_experience(outcome_quality=1.5)
