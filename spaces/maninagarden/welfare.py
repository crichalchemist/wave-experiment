"""
Phi(humanity) welfare scoring -- single source of truth.

Formula v2.1-recovery-floors: recovery_aware_input() is wired into
compute_phi() for the first time.  When a construct falls below its
hard floor, recovery potential (from trajectory and community capacity)
lifts the effective value used in the geometric mean.

NO external dependencies -- stdlib math only.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

# ── constructs ──────────────────────────────────────────────────────────
ALL_CONSTRUCTS = ("c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi")

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

CONSTRUCT_DISPLAY: Dict[str, str] = {
    "c": "Care (c)",
    "kappa": "Compassion (\u03ba)",
    "j": "Joy (j)",
    "p": "Purpose (p)",
    "eps": "Empathy (\u03b5)",
    "lam_L": "Love (\u03bb_L)",
    "lam_P": "Protection (\u03bb_P)",
    "xi": "Truth (\u03be)",
}

# ── hyperparameters ─────────────────────────────────────────────────────
ETA = 0.10                # Ubuntu synergy coupling strength
MU = 0.15                 # Divergence penalty coefficient
ETA_CURIOSITY = 0.08      # Cross-pair curiosity coupling (love x truth)
GAMMA = 0.5               # Community solidarity exponent

# ── construct pairs ─────────────────────────────────────────────────────
CONSTRUCT_PAIRS = [
    ("c", "lam_L"),       # Care x Love
    ("kappa", "lam_P"),   # Compassion x Protection
    ("j", "p"),           # Joy x Purpose
    ("eps", "xi"),        # Empathy x Truth
]
CURIOSITY_CROSS_PAIR = ("lam_L", "xi")
PENALTY_PAIRS = CONSTRUCT_PAIRS + [CURIOSITY_CROSS_PAIR]

# ── version ─────────────────────────────────────────────────────────────
FORMULA_VERSION = "2.1-recovery-floors"


# ── primitives ──────────────────────────────────────────────────────────

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
      - trajectory: sigmoid(10 * dx_dt - 3.0) -- own recovery trend
      - community_capacity: lam_L^0.5 -- community can catalyse recovery
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


# ── formula components ──────────────────────────────────────────────────

def equity_weights(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Inverse-deprivation weights: w_i = (1/x_i) / sum(1/x_j).

    Weights shift dynamically toward the most deprived construct.
    Replaces symmetric Nash theta=1/8 with Rawlsian maximin.
    """
    inv = {
        c: 1.0 / max(0.01, metrics.get(c, 0.5))
        for c in ALL_CONSTRUCTS
    }
    inv_sum = sum(inv.values())
    return {c: inv[c] / inv_sum for c in ALL_CONSTRUCTS}


def community_multiplier(lam_L: float) -> float:
    """
    Community solidarity multiplier: f(lam_L) = lam_L^gamma.

    When community solidarity is low, all welfare degrades.
    Ubuntu: welfare emerges from relational context.
    """
    return max(0.01, lam_L) ** GAMMA


def ubuntu_synergy(metrics: Dict[str, float]) -> float:
    """
    Ubuntu synergy term.

    Psi_ubuntu = 1 + eta * [sqrt(c*lam_L) + sqrt(kappa*lam_P)
                            + sqrt(j*p) + sqrt(eps*xi)]
                   + eta_curiosity * sqrt(lam_L * xi)

    Operates on RAW metrics (detects actual state, not recovery-adjusted).
    """
    pair_sum = sum(
        math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5)))
        for a, b in CONSTRUCT_PAIRS
    )
    a, b = CURIOSITY_CROSS_PAIR
    curiosity = math.sqrt(
        max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5))
    )
    return 1.0 + ETA * pair_sum + ETA_CURIOSITY * curiosity


def divergence_penalty(metrics: Dict[str, float]) -> float:
    """
    Divergence penalty for paired construct mismatches.

    Psi_penalty = mu * sum((a - b)^2) / len(PENALTY_PAIRS)

    Operates on RAW metrics (detects actual structural distortion).
    """
    sq_sum = sum(
        (metrics.get(a, 0.5) - metrics.get(b, 0.5)) ** 2
        for a, b in PENALTY_PAIRS
    )
    return MU * sq_sum / len(PENALTY_PAIRS)


# ── main formula ────────────────────────────────────────────────────────

def compute_phi(
    metrics: Dict[str, float],
    derivatives: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute Phi(humanity) -- the full welfare function (v2.1).

    Phi = f(lam_L) * product(x_tilde_i ^ w_i) * Psi_ubuntu * (1 - Psi_penalty)

    NEW in v2.1: recovery_aware_input() is called for each construct before
    the weighted geometric mean.  Synergy and penalty still operate on raw
    metrics (they detect actual state), but the geometric mean uses the
    effective (recovery-aware) values.

    Args:
        metrics: Dict mapping each construct symbol to a value in [0, 1].
        derivatives: Optional dict of dx/dt per construct. When omitted
                     (e.g. static snapshot from the Research tab), defaults
                     to 0.0 for all constructs.

    Returns:
        Phi score in [0, 1].
    """
    if derivatives is None:
        derivatives = {}

    lam_L_raw = max(0.01, metrics.get("lam_L", 0.5))

    # Community multiplier
    f_lam = community_multiplier(lam_L_raw)

    # Step 1: compute effective (recovery-aware) values for ALL constructs
    effective: Dict[str, float] = {}
    for c in ALL_CONSTRUCTS:
        x_raw = max(0.01, metrics.get(c, 0.5))
        floor_c = CONSTRUCT_FLOORS[c]
        dx_dt_c = derivatives.get(c, 0.0)
        effective[c] = recovery_aware_input(x_raw, floor_c, dx_dt_c, lam_L_raw)

    # Step 2: equity weights computed on EFFECTIVE values (not raw)
    weights = equity_weights(effective)

    # Step 3: weighted geometric mean of effective construct values
    product = 1.0
    for c in ALL_CONSTRUCTS:
        x_eff = max(0.01, effective[c])  # numerical guard
        product *= x_eff ** weights[c]

    # Synergy and penalty operate on RAW metrics
    synergy = ubuntu_synergy(metrics)
    penalty = divergence_penalty(metrics)

    phi = f_lam * product * synergy * (1.0 - penalty)
    return max(0.0, phi)
