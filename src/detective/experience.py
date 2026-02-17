from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Experience:
    """A recorded outcome from a single hypothesis evolution step."""

    hypothesis_id: str
    evidence: str
    action: Literal["confirmed", "refuted", "spawned_alternative"]
    confidence_delta: float  # signed: positive = hypothesis strengthened
    outcome_quality: float   # scored post-hoc in [0.0, 1.0]

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("Experience.hypothesis_id must not be empty")
        if not self.evidence:
            raise ValueError("Experience.evidence must not be empty")
        if self.action not in ("confirmed", "refuted", "spawned_alternative"):
            raise ValueError(
                f"Experience.action must be one of confirmed/refuted/spawned_alternative, "
                f"got {self.action!r}"
            )
        if not (0.0 <= self.outcome_quality <= 1.0):
            raise ValueError(
                f"Experience.outcome_quality must be in [0.0, 1.0], got {self.outcome_quality!r}"
            )


# Immutable library — append always produces a new tuple
ExperienceLibrary = tuple[Experience, ...]

EMPTY_LIBRARY: ExperienceLibrary = ()


def add_experience(library: ExperienceLibrary, experience: Experience) -> ExperienceLibrary:
    """Return new library with experience appended. Does not mutate."""
    return library + (experience,)


def query_similar(
    library: ExperienceLibrary,
    hypothesis_text: str,
    evidence: str,
    top_k: int = 3,
) -> ExperienceLibrary:
    """Return the top_k most similar past experiences using Jaccard similarity on word sets."""
    if not library:
        return EMPTY_LIBRARY

    query_words = set((hypothesis_text + " " + evidence).lower().split())

    def jaccard(exp: Experience) -> float:
        exp_words = set((exp.hypothesis_id + " " + exp.evidence).lower().split())
        intersection = len(query_words & exp_words)
        union = len(query_words | exp_words)
        return intersection / union if union else 0.0

    scored = sorted(library, key=jaccard, reverse=True)
    return tuple(scored[:top_k])
