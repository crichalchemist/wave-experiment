"""
Welfare impact scoring for hypotheses and gaps.

Maps investigative findings to Phi(humanity) constructs and computes
welfare relevance via Phi gradients.

Uses semantic classifier (DistilBERT) as primary method with keyword
fallback when the model is not yet trained.
"""
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword patterns for construct threat inference (fallback)
# Based on humanity.md definitions and constitution.md usage
# ---------------------------------------------------------------------------

_CARE_PATTERNS = frozenset({
    "resource", "allocation", "funding", "provision", "basic needs",
    "poverty", "deprivation", "access", "material", "sustenance",
    "shelter", "food", "water", "healthcare", "education"
})

_COMPASSION_PATTERNS = frozenset({
    "distress", "suffering", "crisis", "emergency", "relief",
    "support", "assistance", "response", "aid"
})

_JOY_PATTERNS = frozenset({
    "wellbeing", "happiness", "positive affect", "quality of life",
    "flourishing", "life satisfaction"
})

_PURPOSE_PATTERNS = frozenset({
    "autonomy", "agency", "self-determination", "goals", "meaning",
    "purpose", "fulfillment", "chosen", "voluntary"
})

_EMPATHY_PATTERNS = frozenset({
    "perspective", "understanding", "intergroup", "discrimination",
    "bias", "prejudice", "othering", "dehumanization", "stereotyping",
    "outgroup", "marginalized", "excluded"
})

_LOVE_PATTERNS = frozenset({
    "growth", "mutual aid", "nurture", "solidarity", "community",
    "capacity", "bonding", "collective", "cooperative", "fellowship",
    "kinship", "togetherness", "belonging"
})

_PROTECTION_PATTERNS = frozenset({
    "safeguard", "protect", "safety", "security", "violence", "harm",
    "abuse", "exploitation", "vulnerability", "risk", "threat",
    "danger", "integrity", "dignity", "rights"
})

_TRUTH_PATTERNS = frozenset({
    "suppress", "conceal", "redact", "withhold", "falsify", "fabricate",
    "misinform", "disinform", "contradiction", "inconsistency",
    "omission", "cover-up", "distortion", "manipulation"
})

# Map patterns to Phi construct names (matching humanity.md symbols)
# 8 constructs: split former "lam" into "lam_L" (love) and "lam_P" (protection)
_CONSTRUCT_PATTERNS = {
    "c": _CARE_PATTERNS,
    "kappa": _COMPASSION_PATTERNS,
    "j": _JOY_PATTERNS,
    "p": _PURPOSE_PATTERNS,
    "eps": _EMPATHY_PATTERNS,
    "lam_L": _LOVE_PATTERNS,
    "lam_P": _PROTECTION_PATTERNS,
    "xi": _TRUTH_PATTERNS,
}


def _keyword_fallback(text: str) -> Tuple[str, ...]:
    """
    Infer threatened constructs via keyword matching (fallback method).

    Returns construct symbols: e.g., ("c", "lam_P") for care + protection.
    """
    lower_text = text.lower()
    threatened = []

    for construct, patterns in _CONSTRUCT_PATTERNS.items():
        if any(pattern in lower_text for pattern in patterns):
            threatened.append(construct)

    return tuple(sorted(threatened))


def get_construct_scores(text: str) -> Dict[str, float]:
    """
    Get welfare construct scores in [0, 1] for all 8 constructs.

    Delegates to the semantic classifier (welfare_classifier.py).
    Falls back to keyword-based binary scores if the model is unavailable.

    Args:
        text: Input text to analyse.

    Returns:
        Dict mapping each of the 8 constructs to a score in [0, 1].
    """
    try:
        from src.inference.welfare_classifier import get_construct_scores as _semantic_scores
        scores = _semantic_scores(text)
        # If all scores are zero, the model wasn't loaded; fall back to keywords
        if any(score > 0.0 for score in scores.values()):
            return scores
    except Exception:
        logger.debug("Semantic classifier unavailable, using keyword fallback")

    # Keyword fallback: binary 0.0 or 1.0 per construct
    keyword_constructs = _keyword_fallback(text)
    return {
        construct: (1.0 if construct in keyword_constructs else 0.0)
        for construct in _CONSTRUCT_PATTERNS
    }


def infer_threatened_constructs(text: str) -> Tuple[str, ...]:
    """
    Infer which Phi constructs a hypothesis/gap threatens.

    Uses semantic classifier as primary method, keyword matching as fallback.

    Returns construct symbols: e.g., ("c", "lam_P") for care + protection.

    Examples:
        >>> infer_threatened_constructs("Resource allocation gap in 2013-2017")
        ('c',)
        >>> infer_threatened_constructs("Redacted correspondence about safeguarding")
        ('lam_P', 'xi')
    """
    try:
        from src.inference.welfare_classifier import get_construct_scores as _semantic_scores
        scores = _semantic_scores(text)
        # If any semantic score is non-zero, the model is loaded
        if any(score > 0.0 for score in scores.values()):
            threatened = [
                construct
                for construct, score in scores.items()
                if score >= 0.3
            ]
            return tuple(sorted(threatened))
    except Exception:
        logger.debug("Semantic classifier unavailable, using keyword fallback")

    # Keyword fallback
    return _keyword_fallback(text)


def phi_gradient_wrt(construct: str, metrics: Dict[str, float]) -> float:
    """
    Compute dPhi/dx for construct x, given current metric levels.

    Simplified gradient approximation using Nash SWF structure from humanity.md.
    For construct with current value x_i and Nash weight theta_i:
        dPhi/dx ~ (theta / x) at current level

    Low values -> high gradients -> high priority (Rawlsian maximin intuition).

    Args:
        construct: Symbol from humanity.md ("c", "kappa", "j", "p", "eps",
                   "lam_L", "lam_P", "xi")
        metrics: Current Phi metric levels, each in [0, 1]

    Returns:
        Gradient value (unbounded, but typically in [0.1, 100] range)

    Examples:
        >>> phi_gradient_wrt("c", {"c": 0.1})  # care is very scarce
        1.25  # high gradient -> high priority
        >>> phi_gradient_wrt("c", {"c": 0.9})  # care is abundant
        0.14  # low gradient -> low priority
    """
    x = metrics.get(construct, 0.5)  # default to mid-level if unknown
    theta = 1.0 / 8.0  # equal Nash weights across 8 constructs

    # Floor to prevent division by zero and extreme gradients
    # Using 0.01 floor -> max gradient = theta/0.01 = 12.5 for single construct
    x_clamped = max(0.01, min(1.0, x))

    return theta / x_clamped


def score_hypothesis_welfare(
    hypothesis: "Hypothesis",  # type: ignore - import at runtime to avoid circular
    phi_metrics: Dict[str, float],
) -> float:
    """
    Compute welfare relevance score for a hypothesis.

    Score = sum(Phi_gradient) for each threatened construct, normalized to [0, 1].

    A hypothesis about resource allocation gaps when care (c) is scarce
    gets high welfare relevance due to high dPhi/dc gradient.

    Args:
        hypothesis: Hypothesis to score
        phi_metrics: Current Phi construct levels

    Returns:
        Welfare relevance in [0, 1]

    Examples:
        >>> h = Hypothesis.create("Temporal gap in financial records 2013-2017", 0.8)
        >>> h = replace(h, threatened_constructs=("c",))
        >>> score_hypothesis_welfare(h, {"c": 0.1})  # care is scarce
        0.56  # high welfare relevance
        >>> score_hypothesis_welfare(h, {"c": 0.9})  # care is abundant
        0.12  # low welfare relevance
    """
    if not hypothesis.threatened_constructs:
        # Infer on first call if not already set
        constructs = infer_threatened_constructs(hypothesis.text)
    else:
        constructs = hypothesis.threatened_constructs

    if not constructs:
        return 0.0  # No welfare threat detected

    gradient_sum = sum(
        phi_gradient_wrt(construct, phi_metrics)
        for construct in constructs
    )

    # Normalize to [0, 1] using soft saturation
    # score = gradient_sum / (gradient_sum + k)
    # k=1.0 means score->0.5 when gradient_sum=1.0
    k = 1.0
    normalized = gradient_sum / (gradient_sum + k)

    return min(1.0, max(0.0, normalized))


def compute_gap_urgency(gap: "Gap", phi_metrics: Dict[str, float]) -> float:  # type: ignore
    """
    Compute investigative urgency for a detected gap.

    Urgency = sum(Phi_gradients) x epistemic_confidence

    Gaps threatening scarce constructs with high epistemic confidence
    are most urgent.

    Args:
        gap: Detected information gap
        phi_metrics: Current Phi construct levels

    Returns:
        Urgency score (unbounded, but typically in [0, 20] range)

    Examples:
        >>> gap = Gap(
        ...     type=GapType.TEMPORAL,
        ...     description="Resource gap 2013-2017",
        ...     confidence=0.9,
        ...     location="doc.pdf",
        ...     threatened_constructs=("c",),
        ... )
        >>> compute_gap_urgency(gap, {"c": 0.1})  # scarce
        11.25  # high urgency
        >>> compute_gap_urgency(gap, {"c": 0.9})  # abundant
        0.13  # low urgency
    """
    if not gap.threatened_constructs:
        constructs = infer_threatened_constructs(gap.description)
    else:
        constructs = gap.threatened_constructs

    if not constructs:
        return 0.0

    gradient_sum = sum(
        phi_gradient_wrt(construct, phi_metrics)
        for construct in constructs
    )

    return gradient_sum * gap.confidence
