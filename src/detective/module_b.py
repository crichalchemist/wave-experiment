"""
Module B: Historical Determinism Detector.

Detects language that treats documentary sequence as causal sequence,
document timestamps as event timestamps, or assumes the past fully
determines the present — the second failure mode in docs/constitution.md.

Uses a provider to score matched spans rather than a static classifier,
because determinism markers are context-dependent: "always" in a personal
memoir differs from "always" in a regulatory filing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.providers import ModelProvider
from src.core.types import AssumptionType

# Phrases that trigger determinism scoring — ordered from strongest signal to weakest.
_DETERMINISM_TRIGGERS: tuple[str, ...] = (
    r"\balways\b",
    r"\bcontinues?\s+to\b",
    r"\binvariably\b",
    r"\bconsistently\b",
    r"\bhas\s+(?:always|ever)\b",
    r"\bnever\s+(?:changed|deviated|altered)\b",
    r"\bsince\s+(?:the\s+beginning|its\s+founding|inception)\b",
    r"\bbecause\s+(?:the\s+)?(?:document|memo|record|filing)\s+(?:was\s+)?dated\b",
    r"\bthe\s+(?:document|record|memo)\s+therefore\b",
    r"\bhistorically\b",
    r"\btraditionally\b",
    r"\bby\s+(?:its\s+)?nature\b",
    r"\bde\s+facto\b",
)

_SCORE_THRESHOLD: float = 0.5

_SCORE_PROMPT = (
    "You are evaluating whether the following text span exhibits 'historical determinism' — "
    "the assumption that documentary sequence reflects causal sequence, or that the past "
    "fully determines the present without examining whether the documentary record is itself "
    "a strategic artifact.\n\n"
    "Rate on a scale from 0.0 (no determinism) to 1.0 (strong determinism).\n"
    "Reply with ONLY: score: <float>\n\n"
    "Text span: {span}\n"
    "Full context: {context}"
)


@dataclass(frozen=True)
class DeterminismDetection:
    """
    A detected instance of historical determinism language.

    Frozen because detections form an immutable audit record — mutating a
    detection after the fact would undermine the chain of evidence.
    """
    assumption_type: AssumptionType
    score: float          # 0.0–1.0; how strongly deterministic this span is
    source_text: str      # the broader context sentence
    trigger_phrase: str   # which regex pattern fired

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")


def _parse_score(response: str) -> float:
    """Extract float from 'score: 0.85' style response. Returns 0.0 on failure."""
    match = re.search(r"score\s*:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if not match:
        return 0.0
    try:
        return min(1.0, max(0.0, float(match.group(1))))
    except ValueError:
        return 0.0


def detect_historical_determinism(
    text: str,
    provider: ModelProvider,
    threshold: float = _SCORE_THRESHOLD,
) -> list[DeterminismDetection]:
    """
    Scan text for historical determinism language and score each match.

    Args:
        text: The document text to analyze.
        provider: LLM provider for context-sensitive scoring.
        threshold: Minimum score to include in results (default 0.5).

    Returns:
        List of DeterminismDetection instances, ordered by score descending.
        Empty list if no spans exceed threshold.
    """
    detections: list[DeterminismDetection] = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for pattern in _DETERMINISM_TRIGGERS:
        for sentence in sentences:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if not match:
                continue
            span = match.group(0)
            prompt = _SCORE_PROMPT.format(span=span, context=sentence)
            raw = provider.complete(prompt)
            score = _parse_score(raw)
            if score >= threshold:
                detections.append(DeterminismDetection(
                    assumption_type=AssumptionType.HISTORICAL_DETERMINISM,
                    score=score,
                    source_text=sentence,
                    trigger_phrase=pattern,
                ))

    return sorted(detections, key=lambda d: d.score, reverse=True)
