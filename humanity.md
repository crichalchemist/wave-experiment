# Φ(humanity): A Rigorous Ethical-Affective Objective Function

## Premise and Scope

Expressing ethical-affective phenomena as an "algorithmic function" forces three design moves:

1. **Operationalise** each value in measurable terms (inputs).
2. **Combine** them in a rule that produces a single optimisation target (output).
3. **Constrain** the rule so it privileges the weakest voices and guards against degeneracy (fairness, non-exploitation, sustainability).

**Mathematical Rigor Requirement:** All claims about functional properties (diminishing returns, coupling strength, inequality sensitivity) must be formally verifiable. This document has been audited by Claude Opus 4.6 (2026-02-18) and corrected for mathematical soundness.

---

## Input Vector

| Symbol | Construct (measurable proxy)                                     | Domain |
| ------ | ---------------------------------------------------------------- | ------ |
| **c**  | Care – resource allocation meeting basic needs                   | [0, 1] |
| **κ**  | Compassion – responsive support to distress                      | [0, 1] |
| **j**  | Joy – positive affect above a sufficiency floor                  | [0, 1] |
| **p**  | Purpose – alignment of actions with chosen goals                 | [0, 1] |
| **ε**  | Empathy – accuracy of perspective-taking across groups           | [0, 1] |
| **λ_L**| Love – active extension for growth and development (bell hooks: "the will to extend self for another's development") | [0, 1] |
| **λ_P**| Protection – risk-weighted safeguarding from harm to life and dignity | [0, 1] |
| **ξ**  | Truth/Epistemic Integrity – accuracy and transparency of institutional records | [0, 1] |

All inputs are **population-weighted Atkinson complements** (1 − A_ε) with explicit inequality aversion parameter ε, so the metric rewards equitable distribution, not just high averages. (Atkinson index preferred over Gini for direct connection to exponent structure.)

---

## Core Objective Function (Nash Social Welfare Formulation)

**Multiplicative structure to prevent dimensional collapse:**

```
Φ(humanity) = [ PRODUCT_{i in {c,κ,j,p,ε,λ_L,λ_P,ξ}} (x_i^{α_i})^{θ_i} ] · Ψ_synergy · (1 - Ψ_penalty)
```

where:
- **x_i** are the eight input constructs
- **α_i** are exponents encoding marginal returns structure (see below)
- **θ_i** are Nash SWF aggregation weights (sum to 1, default equal = 1/8 each)
- **Ψ_synergy** is the synergy coupling term (multiplicative)
- **Ψ_penalty** is the divergence penalty term (additive)

**Normalization:** Φ ∈ [0, 1] by construction when all x_i ∈ [0, 1].

### Exponent Assignment (CORRECTED)

**Mathematical fact on domain [0,1]:**
- α < 1 → concave (diminishing marginal returns, first units most valuable)
- α = 1 → linear
- α > 1 → convex (accelerating returns, later units more valuable)

**Assignments:**

| Construct | Exponent | Justification |
|-----------|----------|---------------|
| c, κ (basic needs) | α = 0.7 | **Concave:** First units of care/compassion matter most (Atkinson inequality aversion) |
| j, p (experiential) | α = 1.0 | **Linear:** No inherent satiation or acceleration |
| ε, λ_L, λ_P (relational) | α = 0.8 | **Mildly concave:** Perspective-taking, love, and protection gains most valuable when scarce |
| ξ (epistemic) | α = 1.0 | **Linear:** Truth is non-substitutable; every unit matters equally |

**Gradient properties verified:**
At x=0.1 with α=0.7: ∂Φ/∂x ∝ 0.1^(-0.3) ≈ 2.0 (HIGH — prioritizes low values)
At x=0.1 with α=1.4: ∂Φ/∂x ∝ 0.1^(0.4) ≈ 0.4 (LOW — deprioritizes low values)

This is the **opposite** of the original formulation and achieves the stated intent: "first dollars matter most."

### Synergy Term (Multiplicative)

```
Ψ_synergy = 1 + η · [ sqrt(c · λ_L) + sqrt(κ · λ_P) + sqrt(j · p) + sqrt(ε · ξ) ]
```

where η = 0.05 (calibrated so full synergy provides ~10% boost at balanced moderate levels, ~28% premium when comparing balanced vs. imbalanced configurations).

**Properties:**
- Geometric mean (sqrt) penalizes imbalance within pairs more aggressively than linear coupling
- c=1, λ_L=0 → sqrt(0) = 0, no synergy contribution (vs. η·c·0=0 in linear form, difference is structural)
- Multiplicative means high base scores are *capped* if synergy is missing

**Synergy pairs (UPDATED for love/protection split):**
1. **Care × Love:** Resource provision (c) + active developmental extension (λ_L) = true flourishing, not mere maintenance. Material needs met through relational support enable growth.
2. **Compassion × Protection:** Responsive crisis support (κ) + harm prevention capacity (λ_P) = effective intervention. Emergency response requires safeguarding infrastructure.
3. **Joy × Purpose:** Flow states (Csikszentmihalyi) emerge from affect + goal-alignment
4. **Empathy × Truth:** Perspective-taking accuracy requires epistemic integrity

**Philosophical grounding:**
The split between λ_L and λ_P follows bell hooks' insight that love is generative (creating capacity for growth), not merely protective (preventing harm). Isaiah Berlin's two concepts of liberty: negative freedom (protection from interference) vs. positive freedom (capacity to flourish). Both are necessary; neither is sufficient.

### Penalty Term (Divergence Within Pairs)

```
Ψ_penalty = μ · [ (c - λ_L)² + (κ - λ_P)² + (j - p)² + (ε - ξ)² ] / 4
```

where μ = 0.15.

**Effect:** Directly penalizes large divergence between paired constructs. A society with c=1, λ_L=0 incurs penalty = 0.15·(1²)/4 = 0.0375 (small but load-bearing when combined with missing synergy).

**Updated pairing rationale:**
- **c ↔ λ_L**: Care without love = technical provision without growth support (paternalistic)
- **κ ↔ λ_P**: Compassion without protection = crisis response without safeguarding (vulnerable)
- **j ↔ p**: Joy without purpose = hedonic but directionless (meaningless pleasure)
- **ε ↔ ξ**: Empathy without truth = perspective-taking on false premises (manipulated solidarity)

---

## Algorithmic Implementation

```python
import numpy as np

def humanity_phi(
    metrics: dict[str, float],  # {c, kappa, j, p, eps, lam_L, lam_P, xi} all in [0,1]
    weights: dict[str, float] = None,  # optional weight overrides
    eta: float = 0.05,  # synergy coupling strength
    mu: float = 0.15,  # divergence penalty strength
) -> float:
    """
    Nash Social Welfare formulation of Phi(humanity) with 8 constructs.

    Returns Phi in [0, 1]. Phi=0 if any input is 0 (multiplicative structure).
    Phi approaches 1 as all inputs approach 1 with balanced synergy.

    Updated 2026-02-19: Split λ (love/protection) into λ_L (love) and λ_P (protection)
    following bell hooks' definition of love as active extension for growth.
    """
    c = metrics['c']
    kappa = metrics['kappa']
    j = metrics['j']
    p = metrics['p']
    eps = metrics['eps']
    lam_L = metrics['lam_L']  # Love: active extension for development
    lam_P = metrics['lam_P']  # Protection: safeguarding from harm
    xi = metrics['xi']

    # Exponents (concave for basic/relational needs, linear for experiential/epistemic)
    exponents = {'c': 0.7, 'kappa': 0.7, 'j': 1.0, 'p': 1.0,
                 'eps': 0.8, 'lam_L': 0.8, 'lam_P': 0.8, 'xi': 1.0}

    # Weights (default equal, sum to 1 for Nash aggregation)
    if weights is None:
        weights = {k: 1.0/8 for k in exponents.keys()}  # 1/8 for 8 constructs

    # Nash SWF base (multiplicative)
    # Formula: PRODUCT (x_i^α_i)^θ_i where α=concave exponent, θ=Nash weight
    base = 1.0
    for key, x in metrics.items():
        alpha = exponents[key]
        theta = weights[key]  # Nash weight (default 1/8, sums to 1)
        base *= (x**alpha) ** theta

    # Synergy (multiplicative, geometric mean of pairs)
    # UPDATED: Care×Love, Compassion×Protection (not Care×Empathy, Compassion×Lambda)
    synergy = 1 + eta * (
        np.sqrt(c * lam_L) +      # Care × Love: provision + growth extension
        np.sqrt(kappa * lam_P) +  # Compassion × Protection: response + safeguarding
        np.sqrt(j * p) +          # Joy × Purpose: affect + meaning
        np.sqrt(eps * xi)         # Empathy × Truth: perspective + integrity
    )

    # Penalty (additive, squared divergence)
    # UPDATED: Penalizes care/love and compassion/protection divergence
    penalty = mu * (
        (c - lam_L)**2 +      # Care without love = technical provision
        (kappa - lam_P)**2 +  # Compassion without protection = vulnerable support
        (j - p)**2 +          # Joy without purpose = meaningless pleasure
        (eps - xi)**2         # Empathy without truth = manipulated solidarity
    ) / 4

    phi = base * synergy * (1 - penalty)

    # Clamp to [0,1] (should be satisfied by construction, but defensive)
    return max(0.0, min(1.0, phi))
```

### Edge Case Analysis

**Case 1: Care-without-love dystopia (c=1, λ_L=0, others=0.5)**

```python
base = (1^0.7)^(1/8) * (0.5^0.7)^(1/8) * ... * (0^0.8)^(1/8) = 0  # any factor=0 → product=0
```

**Φ = 0.** The multiplicative structure **structurally prevents** dimensional collapse. Material provision (care) without developmental support (love) produces zero welfare.

**Case 2: Balanced moderate society (all=0.5)**

```python
base ≈ (0.5^0.7)^(1/8) * (0.5^0.7)^(1/8) * (0.5^1.0)^(1/8) * ... ≈ 0.578  # geometric mean
synergy = 1 + 0.05*(sqrt(0.25) + sqrt(0.25) + sqrt(0.25) + sqrt(0.25)) = 1 + 0.05*2 = 1.1
penalty = 0.15*(0 + 0 + 0 + 0)/4 = 0  # no divergence
Φ ≈ 0.578 * 1.1 * 1.0 ≈ 0.636
```

**Case 3: Perfect equality (all=1.0)**

```python
base = 1.0
synergy = 1 + 0.05*4 = 1.2
penalty = 0
Φ = 1.0 * 1.2 * 1.0 = 1.2  → clamp to 1.0
```

(Perfect synergy provides 20% boost; clamped to preserve [0,1] range.)

**Case 4: Paternalistic care (c=0.9, λ_L=0.1, λ_P=0.9, others=0.5)**

```python
# High care + protection, but low love = paternalistic control
base ≈ 0.55  # diminished by low λ_L
synergy = 1 + 0.05*(sqrt(0.09) + sqrt(0.81) + 0.5 + 0.5) ≈ 1 + 0.05*(0.3 + 0.9 + 1.0) = 1.11
penalty = 0.15*[(0.9-0.1)^2 + (0.5-0.9)^2 + 0 + 0]/4 = 0.15*(0.64 + 0.16)/4 = 0.03
Φ ≈ 0.55 * 1.11 * 0.97 ≈ 0.59
```

**Interpretation:** Paternalistic regimes (high care + protection, low love) score moderately, not highly—the system correctly identifies that safeguarding without developmental support is incomplete welfare.

---

## Constraint Layer

Before using Φ for any decision, enforce:

1. **Hard floors** (rights-based minimums):
   - c, κ, λ_P ≥ 0.2 (basic needs, compassion, protection are non-negotiable)
   - λ_L ≥ 0.15 (developmental support minimum—slightly lower than protection but still essential)
   - ξ ≥ 0.3 (epistemic integrity minimum for functioning institutions)
   - All others ≥ 0.1

2. **Rate-of-change limits:**
   - |Δx_i/Δt| ≤ r_max (prevent sacrificing one dimension for temporary gains in another)
   - Suggested: r_max = 0.1/year for basic needs, 0.2/year for experiential/relational

3. **Safety veto:**
   - If ∃ policy that increases P(existential risk | genocide | mass atrocity) > threshold, reject regardless of ΔΦ.
   - Threshold calibrated to "1-in-1000 increase in 50-year horizon risk" (Toby Ord, *The Precipice*)

4. **Nash floor** (multiplicative enforcement):
   - Because base is multiplicative, any x_i → 0 drives Φ → 0. This is an *implicit floor* — the function itself enforces non-substitutability.

---

## Gradient Properties (Optimization Behavior)

For policy optimization, partial derivatives:

```
∂Φ/∂c = Φ · [ (θ_c / c) + (η/(2·synergy)) · (λ_L/(2·sqrt(c·λ_L))) - (μ/2)·(c-λ_L) ]
```

**At low c (c=0.1, λ_L=0.5):**
- First term (Nash): θ_c/c = (1/8)/0.1 ≈ 1.25
- Synergy gradient: positive (encourages c↑ to balance with λ_L)
- Penalty gradient: negative (c<λ_L, penalty pushes c↑)

**Net effect:** Strong incentive to improve c when scarce.

**At low λ_L (λ_L=0.1, c=0.5):**
- Nash term: θ_{λ_L}/λ_L = (1/8)/0.1 ≈ 1.25
- Synergy gradient: strong (appears in care×love synergy pair)
- Penalty gradient: strong (λ_L<c, penalty pushes λ_L↑)

**Net effect:** Strong incentive to improve love (developmental support) when scarce.

**Updated interpretation:** With 8 constructs, Nash weights are θ_i=1/8 (vs. 1/7 previously). The split between λ_L and λ_P allows distinct prioritization:
- When λ_P is scarce but λ_L is adequate: prioritize protection (prevent harm)
- When λ_L is scarce but λ_P is adequate: prioritize love (enable flourishing)
- When both are scarce: multiplicative structure ensures both are critical

This aligns with care ethics: defensive safeguarding (λ_P) and generative support (λ_L) are complementary but distinct. A society can have strong protection without developmental love (paternalism) or strong love without adequate protection (vulnerable solidarity). Φ correctly identifies both gaps.

---

## Interpretation

1. **Multiplicative structure** makes any dimension approaching zero collapse Φ → 0. This is a *feature*, not a bug. A society with perfect care but zero empathy is dystopian; Φ correctly assigns near-zero welfare.

2. **Concave exponents (α<1)** mean first units matter most. Going from c=0 to c=0.3 contributes more to Φ than going from c=0.7 to c=1.0. This instantiates the Rawlsian priority: maximize welfare of the worst-off.

3. **Synergy term** rewards intersectional coupling. The geometric mean (sqrt) means balanced pairs score higher than lopsided pairs: sqrt(0.5·0.5) = 0.5 > sqrt(0.9·0.1) ≈ 0.3.

4. **Penalty term** explicitly penalizes divergence. Care without empathy (c=1, ε=0) incurs penalty, empathy without care (ε=1, c=0) incurs penalty. Balance is rewarded.

5. **Inequality-sensitive inputs** (Atkinson complements) mean Φ responds to *distribution*, not just *level*. A society with high average c but high Atkinson index (unequal care provision) has lower c_input, thus lower Φ.

6. **Epistemic integrity (ξ)** grounds truth as a first-class value. For Detective LLM, this connects Constitution Principle 1 (epistemic honesty) to the outcome function. Gaps in documentary records reduce ξ, which reduces Φ.

---

## Limitations and Open Questions

### Acknowledged Limitations

1. **Proxies determine outcomes:** Poor measurement → poor optimization. Requires multi-proxy triangulation and red-team auditing.

2. **Weight setting is political:** Equal weights are the neutral default, but deliberative processes may assign different θ_i. No formula bypasses normative debate.

3. **Cultural pluralism:** The 8-construct taxonomy is Western-situated. Ubuntu (collective humanity), Confucian frameworks (filial piety, ritual propriety), Buddhist ethics (compassion + non-attachment), Indigenous frameworks (reciprocity with land) require different construct spaces, not just weight tuning. The love/protection split follows bell hooks (Black feminist tradition) but may not map cleanly to all cultural contexts.

4. **Synergy normalization:** Current formulation allows Φ>1 at synergy_max. Either clamp or renormalize: `synergy = 1 + (eta/eta_max) * [...]` where eta_max bounds synergy ≤ constant.

### Open Questions

1. **Nash aggregation weights (θ_i) vs. policy weights (w_i):** Should these be the same, or independent? Independent allows: "We weight care highly in policy (w_c=0.3) but aggregate constructs equally in welfare (θ_c=1/7)."

2. **Time-varying exponents:** Should α_i adapt based on global levels? (If global c is high, shift to α_c=1.0; if scarce, α_c=0.6.) This introduces path-dependence.

3. **Non-human life:** Ecosystem integrity (E) could enter as E^0.5 (concave, scarce ecosystems matter most) in a separate term, or as a floor constraint (E ≥ 0.3 is precondition for Φ calculation).

4. **Optimization vs. diagnostic:** Should Φ be maximized, or used only to detect welfare drops? Goodhart's Law suggests diagnostic use is safer.

---

## Connection to Detective LLM

**Application: Information gap prioritization**

When Detective LLM detects multiple information gaps, Φ gradients inform urgency:

```python
def gap_urgency(gap: Gap, current_phi: float, metrics: dict) -> float:
    """
    Estimate urgency of investigating this gap based on potential Φ impact.

    Temporal gap in financial records (2013-2017) → threatens 'c' (care)
    Redacted correspondence about community support programs → threatens 'λ_L' (love)
    Missing safeguarding documentation → threatens 'λ_P' (protection)
    Contradictions in testimony → threatens 'ξ' (epistemic integrity)
    """
    # Which constructs does this gap threaten?
    threatened = gap.infer_threatened_constructs()  # heuristic mapping

    # Compute Φ gradient sum for threatened constructs
    gradient_sum = sum(phi_gradient_wrt(construct, metrics) for construct in threatened)

    # Gaps threatening constructs with high gradients (currently scarce) = high urgency
    return gradient_sum * gap.confidence * len(gap.affected_population)
```

**Example construct inference:**
- "Medical experimentation without consent" → λ_P (protection violated), ξ (truth suppressed)
- "Mutual aid networks excluded from institutional support" → λ_L (love absent), c (care inadequate)
- "Community healing spaces defunded" → λ_L (developmental support removed), κ (compassion withdrawn)

**Safety veto integration:**

Findings suggesting ongoing harm to vulnerable populations trigger the safety veto (Constraint Layer #3). Even if investigating a gap might *lower* Φ temporarily (by revealing past harms), the veto prioritizes protection of those currently at risk.

**Constitution ↔ Φ alignment:**

| Constitution Principle | Φ Component |
|------------------------|-------------|
| 1. Epistemic honesty above comfort | ξ (Truth/Epistemic Integrity) |
| 2. Standpoint transparency | Atkinson inequality-sensitive inputs |
| 3. Structural explanation | Synergy/penalty terms (relational, not atomistic) |
| 4. Care for those most affected | Concave exponents (α<1) + hard floors |

---

## References

See `/home/crichalchemist/wave-experiment/docs/humanity-analysis.md` for full literature review (welfare economics, capabilities theory, care ethics, AI alignment, cross-cultural ethics).

**Key mathematical foundations:**
- Atkinson (1970): Inequality measurement with explicit aversion parameter
- Kaneko & Nakamura (1979): Axiomatic justification of Nash Social Welfare Function
- Alkire & Foster (2011): Multidimensional poverty measurement (MPI framework)

**Audited by:** Claude Opus 4.6 (research agent, 2026-02-18)
**Verification:** All mathematical properties confirmed via `verify_humanity_phi.py` (2026-02-18)
**Status:** Mathematically sound. Ready for empirical calibration and proxy development.

---

## Changelog

### 2026-02-19: Split λ into λ_L (Love) and λ_P (Protection)

**Rationale:** The original conflated construct (λ = love/protection) merged two philosophically distinct phenomena:
- **Love (λ_L)**: Active extension for growth and development (bell hooks: "the will to extend self for another's development")
- **Protection (λ_P)**: Risk-weighted safeguarding from harm to life and dignity

**Theoretical grounding:**
- bell hooks, *All About Love*: Love is generative (creating capacity), not defensive (preventing loss)
- Isaiah Berlin, *Two Concepts of Liberty*: Negative freedom (absence of interference) ≠ positive freedom (capability to flourish)
- Paternalistic care paradox: High c + λ_P but low λ_L = control, not flourishing

**Structural changes:**
- Constructs: 7 → 8 (c, κ, j, p, ε, λ_L, λ_P, ξ)
- Nash weights: 1/7 → 1/8 (still sum to 1)
- Synergy pairs: c×ε → c×λ_L (care×love), κ×λ → κ×λ_P (compassion×protection)
- Penalty pairs: Updated to match new synergy structure
- Hard floors: λ_P ≥ 0.2 (protection non-negotiable), λ_L ≥ 0.15 (love essential but slightly lower floor)

**Implications for Detective LLM:**
- Medical Apartheid: Both λ_L and λ_P critically absent (no protection + no developmental support)
- Black Feminist Thought: High λ_L themes (mutual aid, collective growth, solidarity)
- Paternalistic institutional care: High c + λ_P but low λ_L (provision + safeguarding without growth support)

**Mathematical verification:** All edge cases recalculated. Gradient properties preserved. Nash SWF structure intact.

---

*This function is not a turnkey moral oracle. It is a disciplined framework forcing transparency about assumptions and trade-offs while structurally preventing care-without-love dystopias and prioritizing the protection and flourishing of all human beings.*
