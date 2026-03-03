"""
Welfare impact scoring for hypotheses and gaps.

Maps investigative findings to Phi(humanity) constructs and computes
welfare relevance via Phi gradients.

Uses semantic classifier (DistilBERT) as primary method with keyword
fallback when the model is not yet trained.
"""
from typing import Dict, Optional, Tuple
import logging
import math

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

# Curiosity patterns: the investigative drive where love meets truth.
# hooks (2000): love is "the will to extend one's self for growth."
# Curiosity is that extension directed at understanding.
# These patterns fire BOTH lam_L and xi, because curiosity lives at their intersection.
_CURIOSITY_PATTERNS = frozenset({
    "inquiry", "investigate", "investigating", "hunch", "curiosity",
    "scrutiny", "discrepancy", "unanswered", "unexplained", "dig deeper",
    "follow the trail", "look closer", "something doesn't add up",
    "warrants further", "worth investigating", "pull on this thread",
})

# Map patterns to Phi construct names (matching humanity.md symbols)
# 8 constructs: split former "lam" into "lam_L" (love) and "lam_P" (protection)
# Curiosity patterns fire both lam_L and xi (love aimed at truth)
_CONSTRUCT_PATTERNS = {
    "c": _CARE_PATTERNS,
    "kappa": _COMPASSION_PATTERNS,
    "j": _JOY_PATTERNS,
    "p": _PURPOSE_PATTERNS,
    "eps": _EMPATHY_PATTERNS,
    "lam_L": _LOVE_PATTERNS | _CURIOSITY_PATTERNS,
    "lam_P": _PROTECTION_PATTERNS,
    "xi": _TRUTH_PATTERNS | _CURIOSITY_PATTERNS,
}

# ---------------------------------------------------------------------------
# Hard floors per construct (design doc §2)
# ---------------------------------------------------------------------------
CONSTRUCT_FLOORS: Dict[str, float] = {
    "c": 0.20,       # Basic needs non-negotiable
    "kappa": 0.20,    # Crisis response minimum
    "lam_P": 0.20,    # Safety non-negotiable
    "lam_L": 0.15,    # Community minimum
    "xi": 0.30,       # Epistemic integrity highest floor
    "j": 0.10,        # Lower but present
    "p": 0.10,
    "eps": 0.10,
}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def recovery_aware_input(
    x_i: float,
    floor_i: float,
    dx_dt_i: float,
    lam_L: float,
) -> float:
    """
    Recovery-aware effective input for a construct.

    When a construct is above its floor, pass through unchanged.
    When below floor, compute recovery potential from:
      - trajectory: sigmoid(10 * dx_dt) — own recovery trend
      - community_capacity: lam_L^0.5 — community can catalyze recovery
      - recovery_potential: max(trajectory, community_capacity * 0.5)

    Key insight: community can partially compensate for stagnant trajectory.
    Care doesn't begin the uptick without community intervention.
    """
    if x_i >= floor_i:
        return x_i

    # Bias of -3.0 shifts sigmoid so dx_dt=0 maps to ~0.047 (not 0.5).
    # Without this, sigmoid(0)=0.5 would dominate community_capacity*0.5
    # for all lam_L < 1.0, preventing community from compensating for
    # stagnant trajectory (contradicting the design intent).
    trajectory = _sigmoid(10.0 * dx_dt_i - 3.0)
    community_capacity = max(0.01, lam_L) ** 0.5  # guard against lam_L=0
    recovery_potential = max(trajectory, community_capacity * 0.5)

    return x_i + (floor_i - x_i) * recovery_potential


ALL_CONSTRUCTS = ("c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi")


def equity_weights(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compute equity-adjusted weights via inverse deprivation.

    w_i = (1/x_i) / sum(1/x_j) for all 8 constructs.
    Weights shift dynamically toward the most deprived construct.
    Replaces symmetric Nash theta=1/8 with Rawlsian maximin.
    """
    inv = {
        c: 1.0 / max(0.01, metrics.get(c, 0.5))
        for c in ALL_CONSTRUCTS
    }
    inv_sum = sum(inv.values())
    return {c: inv[c] / inv_sum for c in ALL_CONSTRUCTS}


GAMMA = 0.5  # Community solidarity exponent


def community_multiplier(lam_L: float) -> float:
    """
    Community solidarity multiplier.

    f(lam_L) = lam_L^gamma where gamma=0.5.
    When community solidarity is low, all welfare degrades.
    Ubuntu: welfare emerges from relational context.
    """
    return max(0.01, lam_L) ** GAMMA


ETA = 0.10   # Ubuntu synergy coupling strength
MU = 0.15    # Divergence penalty coefficient
ETA_CURIOSITY = 0.08  # Cross-pair curiosity coupling (love x truth)

# Paired constructs: (a, b) — welfare gains emerge from relationships
CONSTRUCT_PAIRS = [
    ("c", "lam_L"),      # Care x Love: material provision + developmental extension
    ("kappa", "lam_P"),   # Compassion x Protection: emergency response + safeguarding
    ("j", "p"),           # Joy x Purpose: positive affect + goal-alignment
    ("eps", "xi"),        # Empathy x Truth: perspective-taking + epistemic integrity
]

# Cross-pair coupling: love x truth = curiosity (investigative drive)
# Not a primary pair — a diagonal synergy that emerges when care meets epistemic integrity.
# hooks (2000): love as extension for growth + Fricker (2007): epistemic integrity
# = the will to investigate, the hunch that won't let go.
CURIOSITY_CROSS_PAIR = ("lam_L", "xi")


def ubuntu_synergy(metrics: Dict[str, float]) -> float:
    """
    Ubuntu synergy term.

    Psi_ubuntu = 1 + eta * [sqrt(c*lam_L) + sqrt(kappa*lam_P) + sqrt(j*p) + sqrt(eps*xi)]
                   + eta_curiosity * sqrt(lam_L * xi)

    Welfare gains emerge from relationships between constructs, not isolation.
    The curiosity cross-pair (love x truth) adds the investigative drive:
    you investigate because you care about truth, and that caring is love.
    """
    pair_sum = sum(
        math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5)))
        for a, b in CONSTRUCT_PAIRS
    )

    # Curiosity: love aimed at truth
    a, b = CURIOSITY_CROSS_PAIR
    curiosity = math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5)))

    return 1.0 + ETA * pair_sum + ETA_CURIOSITY * curiosity


# All penalty pairs: 4 primary + 1 curiosity cross-pair
PENALTY_PAIRS = CONSTRUCT_PAIRS + [CURIOSITY_CROSS_PAIR]


def divergence_penalty(metrics: Dict[str, float]) -> float:
    """
    Divergence penalty for paired construct mismatches.

    Psi_penalty = mu * [(c-lam_L)^2 + (kappa-lam_P)^2 + (j-p)^2 + (eps-xi)^2
                        + (lam_L-xi)^2] / 5

    Penalizes structural distortions:
      - care-without-love (paternalism)
      - compassion-without-protection (vulnerable support)
      - joy-without-purpose (hedonic treadmill)
      - empathy-without-truth (manipulated solidarity)
      - truth-without-love (surveillance) / love-without-truth (willful ignorance)
    """
    sq_sum = sum(
        (metrics.get(a, 0.5) - metrics.get(b, 0.5)) ** 2
        for a, b in PENALTY_PAIRS
    )
    return MU * sq_sum / len(PENALTY_PAIRS)


def compute_phi(
    metrics: Dict[str, float],
    derivatives: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute Phi(humanity) — the full welfare function (v2.1).

    Phi = f(lam_L) * product(x_tilde_i ^ w_i) * Psi_ubuntu * (1 - Psi_penalty)

    v2.1: recovery_aware_input() is called for each construct before the
    weighted geometric mean. Synergy and penalty still operate on raw
    metrics (they detect actual state).

    Args:
        metrics: Dict mapping each construct symbol to a value in [0, 1].
        derivatives: Optional dict of dx/dt per construct. Defaults to 0.0.
    """
    if derivatives is None:
        derivatives = {}

    lam_L_raw = max(0.01, metrics.get("lam_L", 0.5))
    f_lam = community_multiplier(lam_L_raw)

    # Recovery-aware effective values
    effective: Dict[str, float] = {}
    for c in ALL_CONSTRUCTS:
        x_raw = max(0.01, metrics.get(c, 0.5))
        floor_c = CONSTRUCT_FLOORS[c]
        dx_dt_c = derivatives.get(c, 0.0)
        effective[c] = recovery_aware_input(x_raw, floor_c, dx_dt_c, lam_L_raw)

    # Equity weights on effective values
    weights = equity_weights(effective)

    # Weighted geometric mean of effective values
    product = 1.0
    for c in ALL_CONSTRUCTS:
        x_eff = max(0.01, effective[c])
        product *= x_eff ** weights[c]

    # Synergy and penalty on RAW metrics
    synergy = ubuntu_synergy(metrics)
    penalty = divergence_penalty(metrics)

    phi = f_lam * product * synergy * (1.0 - penalty)
    return max(0.0, phi)


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

    Equity-weighted, community-mediated gradient:
        dPhi/dx ~ solidarity * w_i / x

    Where:
        - solidarity = lam_L^0.5 (community multiplier)
        - w_i = (1/x_i) / sum(1/x_j) (equity-adjusted weight)
        - x = clamped construct value

    Low values -> high weights -> high gradients -> high priority.
    Low community -> low solidarity -> all gradients reduced.

    Args:
        construct: Symbol from humanity.md ("c", "kappa", "j", "p", "eps",
                   "lam_L", "lam_P", "xi")
        metrics: Current Phi metric levels, each in [0, 1]

    Returns:
        Gradient value (unbounded, but typically in [0.1, 100] range)
    """
    x = max(0.01, min(1.0, metrics.get(construct, 0.5)))
    weights = equity_weights(metrics)
    w_i = weights.get(construct, 1.0 / 8.0)
    solidarity = community_multiplier(metrics.get("lam_L", 0.5))

    return solidarity * w_i / x


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


def score_hypothesis_curiosity(
    hypothesis: "Hypothesis",  # type: ignore
    phi_metrics: Dict[str, float],
) -> float:
    """
    Compute curiosity relevance for a hypothesis.

    Curiosity = sqrt(gradient_lam_L * gradient_xi), normalized to [0, 1].

    The geometric mean ensures both love AND truth must be scarce for
    curiosity to be high. If either is abundant, the score drops —
    you don't need investigative drive when the answers are already available,
    and you can't sustain investigation without the love that powers the hunch.

    Returns 0.0 if the hypothesis doesn't threaten lam_L or xi.
    """
    if not hypothesis.threatened_constructs:
        constructs = infer_threatened_constructs(hypothesis.text)
    else:
        constructs = hypothesis.threatened_constructs

    has_love = "lam_L" in constructs
    has_truth = "xi" in constructs

    if not (has_love or has_truth):
        return 0.0

    g_love = phi_gradient_wrt("lam_L", phi_metrics) if has_love else 0.0
    g_truth = phi_gradient_wrt("xi", phi_metrics) if has_truth else 0.0

    # Geometric mean: both must be present for curiosity to fire
    raw = math.sqrt(max(0.0, g_love) * max(0.0, g_truth))

    # Normalize to [0, 1]
    k = 1.0
    return min(1.0, max(0.0, raw / (raw + k)))


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


# ---------------------------------------------------------------------------
# Layer 2: Forecast-informed trajectory urgency
# ---------------------------------------------------------------------------

_forecaster_cache = None


def _get_forecaster():
    """Lazy-load PhiTrajectoryForecaster (cached singleton)."""
    global _forecaster_cache
    if _forecaster_cache is None:
        from src.forecasting.phi_trajectory import PhiTrajectoryForecaster
        _forecaster_cache = PhiTrajectoryForecaster()
    return _forecaster_cache


def _forecast_from_metrics(
    metrics: Dict[str, float],
    history_len: int = 200,
) -> "np.ndarray":
    """
    Build a constant-level scenario from current metrics and forecast Phi trajectory.

    Creates a 200-step DataFrame where all construct values are held constant
    at their current levels (with tiny noise for signal processing stability),
    then runs the forecaster to predict 10 future Phi values.
    """
    import numpy as np
    import pandas as pd
    import torch

    forecaster = _get_forecaster()
    rng = np.random.default_rng(42)

    # Build constant scenario: each construct stays at current level
    data = {}
    for c in ALL_CONSTRUCTS:
        level = max(0.01, min(1.0, metrics.get(c, 0.5)))
        data[c] = np.full(history_len, level) + rng.normal(0, 0.001, history_len)
        data[c] = np.clip(data[c], 0.0, 1.0)

    df = pd.DataFrame(data)

    # Compute Phi column (use local compute_phi — does NOT take derivatives)
    phi_vals = np.array([
        compute_phi({c: df.at[i, c] for c in ALL_CONSTRUCTS})
        for i in range(len(df))
    ])
    df["phi"] = phi_vals

    # Feature engineering via pipeline
    X = forecaster.pipeline.fit_transform(df)
    X_seq = X[np.newaxis, -forecaster.pipeline.seq_len:]  # [1, seq_len, 36]
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        phi_pred = forecaster.model.predict_phi(X_tensor)

    return phi_pred[0, :, 0].numpy()


def _get_trajectory_prediction(metrics: Dict[str, float]) -> "np.ndarray":
    """Get 10-step Phi trajectory prediction from current metrics."""
    return _forecast_from_metrics(metrics)


def score_hypothesis_trajectory(
    hypothesis: "Hypothesis",
    phi_metrics: Dict[str, float],
) -> float:
    """
    Compute trajectory urgency for a hypothesis.

    Runs the Phi forecaster on current construct levels to predict
    whether welfare is declining. Declining trajectories increase urgency.

    urgency = max(0, -slope) / (max(0, -slope) + k)

    Where slope = (phi[-1] - phi[0]) / len(phi), normalized to [0,1].
    Rising or stable trajectories -> 0.0 urgency.

    Args:
        hypothesis: Hypothesis to score (urgency depends on overall welfare state)
        phi_metrics: Current Phi construct levels

    Returns:
        Trajectory urgency in [0, 1], where 1.0 = steepest decline
    """
    try:
        predictions = _get_trajectory_prediction(phi_metrics)
    except Exception as e:
        logger.debug(f"Trajectory prediction failed: {e}")
        return 0.0

    if len(predictions) < 2:
        return 0.0

    slope = (float(predictions[-1]) - float(predictions[0])) / len(predictions)

    # Only declining trajectories create urgency
    if slope >= 0:
        return 0.0

    decline = -slope  # positive value
    k = 0.02  # normalize: decline of 0.02/step -> urgency ~0.5
    urgency = decline / (decline + k)

    return min(1.0, max(0.0, urgency))
