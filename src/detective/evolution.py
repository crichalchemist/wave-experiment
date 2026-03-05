from __future__ import annotations

import logging
from typing import Literal

from src.core.providers import ModelProvider
from src.detective.experience import (
    Experience,
    ExperienceLibrary,
    add_experience,
    query_similar,
)
from src.detective.hypothesis import Hypothesis

_logger = logging.getLogger(__name__)

BRANCHING_THRESHOLD: float = 0.5
# Confidence delta applied when evidence refutes the hypothesis
_REFUTATION_DECAY: float = 0.2
# Number of similar past experiences to retrieve for context
_LIBRARY_TOP_K: int = 3


BranchingStrategy = Literal["breadth", "depth"]


def branching_rule(confidence: float) -> BranchingStrategy:
    """
    Below threshold: generate competing hypotheses (breadth — Self-Consistency).
    Above threshold: invest in verification passes (depth).
    Maps to the SC-first compute allocation from When To Solve/When To Verify.
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence!r}")
    return "breadth" if confidence < BRANCHING_THRESHOLD else "depth"


def _classify_action(
    original_confidence: float,
    updated_confidence: float,
) -> Literal["confirmed", "refuted", "spawned_alternative"]:
    """Zero delta maps to spawned_alternative rather than no-op — unchanged confidence after new evidence suggests the hypothesis has forked, not confirmed."""
    delta = updated_confidence - original_confidence
    if delta > 0:
        return "confirmed"
    if delta < 0:
        return "refuted"
    return "spawned_alternative"


def _build_evolution_prompt(
    hypothesis: Hypothesis,
    new_evidence: str,
    similar: ExperienceLibrary,
) -> str:
    """Build the prompt sent to the model for hypothesis evolution."""
    prior_context = ""
    if similar:
        examples = "\n".join(
            f"- [{e.action}] {e.hypothesis_text}: {e.evidence}"
            for e in similar
        )
        prior_context = f"\n\nSimilar past trajectories:\n{examples}"

    return (
        f"Hypothesis: {hypothesis.text}\n"
        f"Current confidence: {hypothesis.confidence:.2f}\n"
        f"New evidence: {new_evidence}"
        f"{prior_context}\n\n"
        f"How does this evidence affect the hypothesis? "
        f"Reply with ONLY a float between 0.0 and 1.0 representing the updated confidence."
    )


def _parse_confidence(response: str) -> float | None:
    """Extract a float in [0.0, 1.0] from model response, or None if unparseable."""
    try:
        value = float(response.strip())
        if 0.0 <= value <= 1.0:
            return value
        return None
    except ValueError:
        return None


def evolve_hypothesis(
    hypothesis: Hypothesis,
    new_evidence: str,
    library: ExperienceLibrary,
    provider: ModelProvider,
) -> tuple[Hypothesis, Experience]:
    """
    Consult the experience library before evolving — prior trajectories inform the update.
    Returns the evolved hypothesis and the experience record for this step.
    Confidence decay is bounded at 0.0; gain is bounded at 1.0.
    """
    similar = query_similar(library, hypothesis.text, new_evidence, top_k=_LIBRARY_TOP_K)
    prompt = _build_evolution_prompt(hypothesis, new_evidence, similar)

    response = provider.complete(prompt)
    parsed = _parse_confidence(response)

    if parsed is not None:
        new_confidence = parsed
    else:
        # Unparseable response — treat as evidence of uncertainty, apply small decay
        new_confidence = max(0.0, hypothesis.confidence - _REFUTATION_DECAY)

    evolved = hypothesis.update_confidence(new_confidence)
    action = _classify_action(hypothesis.confidence, new_confidence)
    confidence_delta = new_confidence - hypothesis.confidence

    experience = Experience(
        hypothesis_id=hypothesis.id,
        hypothesis_text=hypothesis.text,
        evidence=new_evidence,
        action=action,
        confidence_delta=confidence_delta,
        outcome_quality=0.0,  # scored post-hoc; 0.0 is placeholder
    )

    return evolved, experience
