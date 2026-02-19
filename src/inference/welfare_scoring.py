"""
Welfare impact scoring for hypotheses and gaps.

Maps investigative findings to Φ(humanity) constructs and computes
welfare relevance via Φ gradients.
"""
from typing import Dict, Tuple

# Keyword patterns for construct threat inference
# Based on humanity.md definitions and constitution.md usage

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

# Map patterns to Φ construct names (matching humanity.md symbols)
_CONSTRUCT_PATTERNS = {
    "c": _CARE_PATTERNS,
    "kappa": _COMPASSION_PATTERNS,
    "j": _JOY_PATTERNS,
    "p": _PURPOSE_PATTERNS,
    "eps": _EMPATHY_PATTERNS,
    "lam": _PROTECTION_PATTERNS,
    "xi": _TRUTH_PATTERNS,
}


def infer_threatened_constructs(text: str) -> Tuple[str, ...]:
    """
    Infer which Φ constructs a hypothesis/gap threatens based on keyword matching.

    Returns construct symbols: e.g., ("c", "lam") for care + protection.

    Examples:
        >>> infer_threatened_constructs("Resource allocation gap in 2013-2017")
        ('c',)
        >>> infer_threatened_constructs("Redacted correspondence about safeguarding")
        ('lam', 'xi')
    """
    lower_text = text.lower()
    threatened = []

    for construct, patterns in _CONSTRUCT_PATTERNS.items():
        if any(pattern in lower_text for pattern in patterns):
            threatened.append(construct)

    return tuple(sorted(threatened))


def phi_gradient_wrt(construct: str, metrics: Dict[str, float]) -> float:
    """
    Compute ∂Φ/∂x for construct x, given current metric levels.

    Simplified gradient approximation using Nash SWF structure from humanity.md.
    For construct with current value x_i and Nash weight θ_i:
        ∂Φ/∂x ≈ (θ / x) at current level

    Low values → high gradients → high priority (Rawlsian maximin intuition).

    Args:
        construct: Symbol from humanity.md ("c", "kappa", "j", "p", "eps", "lam", "xi")
        metrics: Current Φ metric levels, each in [0, 1]

    Returns:
        Gradient value (unbounded, but typically in [0.1, 100] range)

    Examples:
        >>> phi_gradient_wrt("c", {"c": 0.1})  # care is very scarce
        1.43  # high gradient → high priority
        >>> phi_gradient_wrt("c", {"c": 0.9})  # care is abundant
        0.16  # low gradient → low priority
    """
    x = metrics.get(construct, 0.5)  # default to mid-level if unknown
    theta = 1.0 / 7.0  # equal Nash weights (default from humanity.md Section 2)

    # Floor to prevent division by zero and extreme gradients
    # Using 0.01 floor → max gradient = θ/0.01 = 14.3 for single construct
    x_clamped = max(0.01, min(1.0, x))

    return theta / x_clamped


def score_hypothesis_welfare(
    hypothesis: "Hypothesis",  # type: ignore - import at runtime to avoid circular
    phi_metrics: Dict[str, float],
) -> float:
    """
    Compute welfare relevance score for a hypothesis.

    Score = Σ(Φ_gradient) for each threatened construct, normalized to [0, 1].

    A hypothesis about resource allocation gaps when care (c) is scarce
    gets high welfare relevance due to high ∂Φ/∂c gradient.

    Args:
        hypothesis: Hypothesis to score
        phi_metrics: Current Φ construct levels

    Returns:
        Welfare relevance in [0, 1]

    Examples:
        >>> h = Hypothesis.create("Temporal gap in financial records 2013-2017", 0.8)
        >>> h = replace(h, threatened_constructs=("c",))
        >>> score_hypothesis_welfare(h, {"c": 0.2})  # care is scarce
        0.58  # high welfare relevance
        >>> score_hypothesis_welfare(h, {"c": 0.9})  # care is abundant
        0.14  # low welfare relevance
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
    # k=1.0 means score→0.5 when gradient_sum=1.0
    # With max single gradient ≈ 1.43 at x=0.1, score≈0.59
    # With gradient ≈ 0.71 at x=0.2, score≈0.42
    k = 1.0
    normalized = gradient_sum / (gradient_sum + k)

    return min(1.0, max(0.0, normalized))
