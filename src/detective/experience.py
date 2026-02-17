from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Experience:
    """Records a single evolution step so future reasoning can learn from prior trajectories."""

    hypothesis_id: str
    hypothesis_text: str  # the natural-language hypothesis text
    evidence: str
    action: Literal["confirmed", "refuted", "spawned_alternative"]
    confidence_delta: float  # signed: positive = hypothesis strengthened
    outcome_quality: float

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("Experience.hypothesis_id must not be empty")
        if not self.hypothesis_text:
            raise ValueError("Experience.hypothesis_text must not be empty")
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
    """Immutable append — caller always receives a new value; prior library states are preserved for audit."""
    return library + (experience,)


def query_similar(
    library: ExperienceLibrary,
    hypothesis_text: str,
    evidence: str,
    top_k: int = 3,
) -> ExperienceLibrary:
    """Consult prior trajectories before evolving a hypothesis — similar past contexts inform the branching decision."""
    if not library:
        return EMPTY_LIBRARY

    query_words = set((hypothesis_text + " " + evidence).lower().split())

    def jaccard(exp: Experience) -> float:
        exp_words = set((exp.hypothesis_text + " " + exp.evidence).lower().split())
        intersection = len(query_words & exp_words)
        union = len(query_words | exp_words)
        return intersection / union if union else 0.0

    scored = sorted(library, key=jaccard, reverse=True)
    return tuple(scored[:top_k])
