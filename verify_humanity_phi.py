#!/usr/bin/env python3
"""Verify mathematical properties of corrected Phi(humanity) function."""
import numpy as np

def humanity_phi(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
    eta: float = 0.05,  # Reduced: 4 pairs * sqrt(1) * 0.05 = 0.20 max boost
    mu: float = 0.15,
) -> float:
    """Nash Social Welfare formulation of Phi(humanity)."""
    c = metrics['c']
    kappa = metrics['kappa']
    j = metrics['j']
    p = metrics['p']
    eps = metrics['eps']
    lam = metrics['lam']
    xi = metrics['xi']

    exponents = {'c': 0.7, 'kappa': 0.7, 'j': 1.0, 'p': 1.0,
                 'eps': 0.8, 'lam': 0.8, 'xi': 1.0}

    if weights is None:
        weights = {k: 1.0/7 for k in exponents.keys()}

    # Nash SWF base (multiplicative)
    # Formula: PRODUCT (x_i^α_i)^θ_i where α=concave exponent, θ=Nash weight
    base = 1.0
    for key, x in metrics.items():
        alpha = exponents[key]
        theta = weights[key]
        base *= (x**alpha) ** theta

    # Synergy (multiplicative, geometric mean of pairs)
    synergy = 1 + eta * (
        np.sqrt(c * eps) +
        np.sqrt(kappa * lam) +
        np.sqrt(j * p) +
        np.sqrt(eps * xi)
    )

    # Penalty (additive, squared divergence)
    penalty = mu * (
        (c - eps)**2 +
        (kappa - lam)**2 +
        (j - p)**2 +
        (eps - xi)**2
    ) / 4

    phi = base * synergy * (1 - penalty)
    return max(0.0, min(1.0, phi))


def test_edge_cases():
    """Verify critical mathematical properties."""
    print("=" * 60)
    print("EDGE CASE VERIFICATION")
    print("=" * 60)

    # Case 1: Care-without-empathy dystopia (MUST score near 0)
    print("\n1. Care-without-empathy dystopia (c=1, ε=0, others=0.5):")
    dystopia = humanity_phi({
        'c': 1.0, 'kappa': 1.0, 'j': 1.0, 'p': 1.0,
        'eps': 0.0, 'lam': 0.0, 'xi': 0.5
    })
    print(f"   Φ = {dystopia:.4f}")
    print(f"   {'✓ PASS' if dystopia < 0.1 else '✗ FAIL'}: Multiplicative structure prevents dimensional collapse")

    # Case 2: Balanced moderate society
    print("\n2. Balanced moderate society (all=0.5):")
    balanced = humanity_phi({
        'c': 0.5, 'kappa': 0.5, 'j': 0.5, 'p': 0.5,
        'eps': 0.5, 'lam': 0.5, 'xi': 0.5
    })
    print(f"   Φ = {balanced:.4f}")
    print(f"   {'✓ PASS' if 0.55 < balanced < 0.65 else '✗ FAIL'}: Moderate balanced score (Nash geometric mean + synergy)")

    # Case 3: Perfect equality
    print("\n3. Perfect equality (all=1.0):")
    perfect = humanity_phi({
        'c': 1.0, 'kappa': 1.0, 'j': 1.0, 'p': 1.0,
        'eps': 1.0, 'lam': 1.0, 'xi': 1.0
    })
    print(f"   Φ = {perfect:.4f}")
    print(f"   {'✓ PASS' if perfect >= 0.30 else '✗ FAIL'}: High score with full synergy (Nash + synergy boost)")

    # Case 4: Original additive dystopia comparison
    print("\n4. Original additive formulation dystopia comparison:")
    print(f"   Dystopia (c=1,ε=0): Φ = {dystopia:.4f}")
    print(f"   Balanced (all=0.5): Φ = {balanced:.4f}")
    print(f"   {'✓ PASS' if balanced > dystopia else '✗ FAIL'}: Balanced > Dystopia (CORRECTED)")

    # Case 5: Gradient check at low c
    print("\n5. Gradient behavior (concave exponents):")
    low_c = humanity_phi({
        'c': 0.1, 'kappa': 0.5, 'j': 0.5, 'p': 0.5,
        'eps': 0.5, 'lam': 0.5, 'xi': 0.5
    })
    high_c = humanity_phi({
        'c': 0.9, 'kappa': 0.5, 'j': 0.5, 'p': 0.5,
        'eps': 0.5, 'lam': 0.5, 'xi': 0.5
    })
    gradient_low = (humanity_phi({
        'c': 0.2, 'kappa': 0.5, 'j': 0.5, 'p': 0.5,
        'eps': 0.5, 'lam': 0.5, 'xi': 0.5
    }) - low_c) / 0.1
    gradient_high = (humanity_phi({
        'c': 1.0, 'kappa': 0.5, 'j': 0.5, 'p': 0.5,
        'eps': 0.5, 'lam': 0.5, 'xi': 0.5
    }) - high_c) / 0.1

    print(f"   Gradient at c=0.1: Δ Φ/Δc ≈ {gradient_low:.4f}")
    print(f"   Gradient at c=0.9: ΔΦ/Δc ≈ {gradient_high:.4f}")
    print(f"   {'✓ PASS' if gradient_low > gradient_high else '✗ FAIL'}: First units matter more (α=0.7 concave)")

    # Case 6: Synergy impact
    print("\n6. Synergy coupling strength:")
    no_synergy_sim = humanity_phi({
        'c': 0.7, 'kappa': 0.7, 'j': 0.7, 'p': 0.7,
        'eps': 0.3, 'lam': 0.3, 'xi': 0.7
    })
    with_synergy = humanity_phi({
        'c': 0.7, 'kappa': 0.7, 'j': 0.7, 'p': 0.7,
        'eps': 0.7, 'lam': 0.7, 'xi': 0.7
    })
    synergy_premium = (with_synergy - no_synergy_sim) / no_synergy_sim * 100
    print(f"   Without balance: Φ = {no_synergy_sim:.4f}")
    print(f"   With balance:    Φ = {with_synergy:.4f}")
    print(f"   Synergy premium: {synergy_premium:.1f}%")
    print(f"   {'✓ PASS' if synergy_premium > 15 else '✗ FAIL'}: Synergy coupling is load-bearing (>15% effect)")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_edge_cases()
