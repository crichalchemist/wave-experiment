"""Cognitive bias detection — Module A.

Uses regex triggers for known bias patterns + LLM-scored confirmation,
matching the architecture of Modules B and C.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.core.types import AssumptionType
from src.core.providers import ModelProvider
from src.core.scoring import parse_score as _parse_score

_logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD: float = 0.5

# Bias pattern triggers
_BIAS_PATTERNS: dict[str, re.Pattern] = {
    "confirmation": re.compile(
        r"confirm|consistent with|supports? (the|our|my) (view|hypothesis|belief)",
        re.IGNORECASE,
    ),
    "anchoring": re.compile(
        r"initial|first (report|estimate)|anchor|starting point|baseline assumption",
        re.IGNORECASE,
    ),
    "survivorship": re.compile(
        r"surviv|success stor|only (the|those) (who|that) (made|succeeded)|ignoring fail",
        re.IGNORECASE,
    ),
    "ingroup": re.compile(
        r"our (group|side|team)|they (always|never)|us vs\.? them|in-?group|out-?group",
        re.IGNORECASE,
    ),
}

_SCORING_PROMPT = (
    "Rate the cognitive bias in the following text on a scale of 0.0 to 1.0.\n"
    "Bias type: {bias_type}\n"
    "Text: {text}\n\n"
    "Reply with: score: <float between 0 and 1>"
)


@dataclass(frozen=True)
class BiasDetection:
    """A detected cognitive bias."""
    assumption_type: AssumptionType
    score: float
    source_text: str
    bias_type: str  # which bias pattern triggered

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"BiasDetection.score must be in [0.0, 1.0], got {self.score!r}")


def detect_cognitive_biases(
    text: str,
    provider: ModelProvider | None = None,
    threshold: float = _CONFIDENCE_THRESHOLD,
) -> list[BiasDetection]:
    """Detect cognitive biases via regex triggers + optional LLM scoring.

    When provider is None, returns detections with score=1.0 for any
    matched pattern (keyword-only mode). When provider is given, each
    trigger is scored by the LLM.
    """
    detections: list[BiasDetection] = []

    for bias_type, pattern in _BIAS_PATTERNS.items():
        if not pattern.search(text):
            continue

        if provider is not None:
            prompt = _SCORING_PROMPT.format(bias_type=bias_type, text=text[:500])
            response = provider.complete(prompt)
            score = _parse_score(response)
        else:
            score = 1.0  # keyword match without LLM confirmation

        if score >= threshold:
            detections.append(BiasDetection(
                assumption_type=AssumptionType.COGNITIVE_BIAS,
                score=score,
                source_text=text,
                bias_type=bias_type,
            ))

    return detections
