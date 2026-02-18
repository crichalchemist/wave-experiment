"""
VaryBalance annotation consistency scorer.

Measures annotation reliability by asking a provider to rewrite two
independently-produced analyses of the same claim. If the rewrites are
similar in length, the annotation is consistent (low MSD = reliable).
Low MSD = consistent = reliable annotation.
"""
from __future__ import annotations

from src.core.providers import ModelProvider

# Threshold below which an annotation is considered consistent/reliable
CONSISTENCY_THRESHOLD: float = 0.1

# Prompt template for rewrite — asking for a rewrite tests semantic stability
_REWRITE_PROMPT: str = (
    "Rewrite the following analysis concisely in your own words, "
    "preserving all factual claims:\n\n{analysis}"
)


def _rewrite(analysis: str, provider: ModelProvider) -> str:
    """Ask provider to rewrite the analysis. Pure function (no side effects)."""
    return provider.complete(_REWRITE_PROMPT.format(analysis=analysis))


def annotation_consistency_score(
    analysis_a: str,
    analysis_b: str,
    provider: ModelProvider,
) -> float:
    """
    VaryBalance consistency proxy: rewrite both analyses and compute MSD of
    rewrite lengths. Low MSD = consistent = reliable annotation.

    MSD = ((len(rewrite_a) - len(rewrite_b)) ** 2 + (len(rewrite_b) - len(rewrite_a)) ** 2) / 2
        = (len(rewrite_a) - len(rewrite_b)) ** 2

    The 2-sample MSD simplifies to the squared length difference.
    Returns a non-negative float.
    """
    rewrite_a = _rewrite(analysis_a, provider)
    rewrite_b = _rewrite(analysis_b, provider)
    diff = len(rewrite_a) - len(rewrite_b)
    return float(diff ** 2)


def is_consistent(
    analysis_a: str,
    analysis_b: str,
    provider: ModelProvider,
    threshold: float = CONSISTENCY_THRESHOLD,
) -> bool:
    """Return True if the annotation pair scores below the consistency threshold."""
    return annotation_consistency_score(analysis_a, analysis_b, provider) <= threshold
