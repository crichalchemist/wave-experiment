# Φ(humanity) Formula Revision — Design Document

## Goal

Revise the Nash Social Welfare Function from equal-weight symmetric structure to an equity-weighted, community-mediated formula grounded in Western capability theory + Ubuntu relational philosophy.

## Motivation

The current formula treats all eight constructs as co-equal (`θ = 1/8`) and computes welfare as a simple geometric mean. Three problems:

1. **Equality ≠ equity.** Equal weights presume equal starting positions. When extreme wealth concentration means a few hold the financial capability to eliminate homelessness and hunger but choose not to because there is no profit, symmetry is a fiction. The formula should prioritize the most deprived construct, not treat all equally.

2. **Community is substrate, not peer.** Love/community solidarity (λ_L) doesn't sit alongside care and truth as one of eight — it enables them. Cognition itself evolved within the safe space of community. In a system where "cognition is actively stifled through the creation of scarcity and the division of people," the formula must recognize community as foundational.

3. **Harm can be generative.** The Pareto framing treats all harm as loss. But humans are self-repairing organisms who grow through adversity — provided community exists to support recovery. The formula should distinguish "healing in progress" from "genuine collapse."

---

## Revised Formula

### Core Structure

```
Φ = f(λ_L) × Π(x̃_i)^(w_i) × Ψ_ubuntu × (1 - Ψ_penalty)
```

### Components

#### 1. Community Solidarity Multiplier — `f(λ_L)`

```
f(λ_L) = λ_L^γ     where γ = 0.5
```

When community solidarity is low, the entire function degrades. This is the Ubuntu component: welfare emerges from relational context, not individual metrics in isolation.

| λ_L | f(λ_L) | Interpretation |
|-----|--------|----------------|
| 1.0 | 1.0 | Full community capacity |
| 0.5 | 0.71 | Moderate — individual constructs 29% diminished |
| 0.25 | 0.5 | Weak community — all constructs halved |
| 0.04 | 0.2 | Near-collapse — 80% degradation |

#### 2. Recovery-Aware Inputs — `x̃_i`

```python
def recovery_aware_input(x_i, floor_i, dx_dt_i, lam_L):
    if x_i >= floor_i:
        return x_i

    trajectory = sigmoid(10 * dx_dt_i)           # own recovery trend
    community_capacity = lam_L ** 0.5             # community can catalyze recovery

    # Community partially compensates for stagnant trajectory
    # Care doesn't begin the uptick without community intervention
    recovery_potential = max(trajectory, community_capacity * 0.5)

    return x_i + (floor_i - x_i) * recovery_potential
```

**Key insight:** Recovery often requires community intervention before the individual trajectory turns positive. A stagnant construct with high λ_L has recovery *potential* even if dx/dt ≈ 0, because community infrastructure exists to trigger the uptick.

Three states:
- **Healing:** x_i below floor, dx/dt positive, λ_L high → recovery_potential high
- **Intervention pending:** x_i below floor, dx/dt ≈ 0, λ_L high → recovery_potential moderate
- **True collapse:** x_i below floor, dx/dt ≈ 0, λ_L low → recovery_potential low (white supremacy signature)

Hard floors:

| Construct | Floor | Rationale |
|-----------|-------|-----------|
| c (care) | 0.20 | Basic needs non-negotiable |
| κ (compassion) | 0.20 | Crisis response minimum |
| λ_P (protection) | 0.20 | Safety non-negotiable |
| λ_L (love) | 0.15 | Community minimum |
| ξ (truth) | 0.30 | Epistemic integrity highest floor |
| j, p, ε | 0.10 | Lower but present |

#### 3. Equity-Adjusted Weights — `w_i`

```
w_i = (1/x̃_i) / Σ_j(1/x̃_j)     for all 8 constructs
```

Weights shift dynamically toward the most deprived construct. No construct has a fixed weight. When care is at 0.1 and truth at 0.9, care's weight rises to ~0.56 while truth drops to ~0.06. The system doesn't treat them equally — it treats them equitably.

This replaces the symmetric Nash property with Rawlsian maximin: the formula's gradient aggressively prioritizes whatever is most deprived.

#### 4. Ubuntu Synergy — `Ψ_ubuntu`

```
Ψ_ubuntu = 1 + η × [√(c·λ_L) + √(κ·λ_P) + √(j·p) + √(ε·ξ)]
```

η = 0.10 (raised from 0.05 to reflect Ubuntu's centrality).

Renamed from `Ψ_synergy` to make the philosophical grounding explicit. Welfare gains emerge from relationships between constructs, not from constructs in isolation.

Paired constructs and rationale:
1. **Care × Love (c · λ_L):** Material provision + developmental extension = flourishing
2. **Compassion × Protection (κ · λ_P):** Emergency response + safeguarding = effective crisis intervention
3. **Joy × Purpose (j · p):** Positive affect + goal-alignment = eudaimonia
4. **Empathy × Truth (ε · ξ):** Perspective-taking + epistemic integrity = accurate cross-group understanding

#### 5. Divergence Penalties — `Ψ_penalty`

```
Ψ_penalty = μ × [(c-λ_L)² + (κ-λ_P)² + (j-p)² + (ε-ξ)²] / 4
```

μ = 0.15. Unchanged from current spec.

---

## Gradient Implementation

The scoring path (`phi_gradient_wrt`) changes from:

```python
# Current: equal weights
theta = 1/8
return theta / max(0.01, x)
```

To:

```python
# Revised: equity-weighted, community-mediated
inv_sum = sum(1.0 / max(0.01, metrics.get(c, 0.5)) for c in ALL_CONSTRUCTS)
x = max(0.01, metrics.get(construct, 0.5))
w_i = (1.0 / x) / inv_sum

lam_L = max(0.01, metrics.get("lam_L", 0.5))
solidarity = lam_L ** 0.5

return solidarity * w_i / x
```

When care is at 0.1 and everything else at 0.5, care's gradient becomes ~28× the current equal-weight version. The system aggressively prioritizes the deprived.

`score_hypothesis_welfare()` and `compute_gap_urgency()` are unchanged in structure — the equity weighting flows through the gradient automatically.

---

## Philosophical Framing: Western + Ubuntu Synthesis

### Western Lineage (Individual Capability)
- **Sen/Nussbaum (1999, 2000):** Eight dimensions of functioning (the constructs)
- **Rawls → Atkinson (1970):** Priority to the worst-off (equity weights)
- **Berlin:** Positive/negative freedom (λ_L/λ_P split)

### Ubuntu Lineage (Relational Capability)
- **"Umuntu ngumuntu ngabantu"** — a person is a person through other persons
- Welfare emerges from relationships, not from individual metrics in isolation
- λ_L as meta-construct: community solidarity enables individual capability
- Ψ_ubuntu: constructs gain meaning through their pairings

### Synthesis

The formula computes Φ as individual capabilities mediated by community solidarity. The product `Π(x̃_i)^(w_i)` is Western (individual constructs). The multiplier `f(λ_L)` and synergy `Ψ_ubuntu` are Ubuntu (relational substrate). Neither term alone produces Φ.

### bell hooks as Bridge

hooks' definition of love — "the will to extend one's self for the purpose of nurturing one's own or another's spiritual growth" — sits at the intersection of both frameworks. It is an individual act (Western) that only makes sense relationally (Ubuntu).

The self-directed component ("one's own") is essential: one cannot pour from an empty cup. You can only be in community by prioritizing self in a selfless fashion. This creates the reciprocal dependency that the formula encodes:

- **Product term** `Π(x̃_i)^(w_i)`: You cannot pour from an empty cup. Self-care is prerequisite to community participation.
- **Meta-multiplier** `f(λ_L)`: The cup is filled by community. Individual constructs gain their full meaning only through relational context.

Neither is sufficient alone. The relationship is circular — self-sustaining through community, community-sustaining through self.

### Care Ethics Integration

hooks (2000), Gilligan (1982): Love as generative, not defensive. Revolutionary in a period where humanity is divided with intention, so that we avert our eyes from the atrocity of the hyper-wealthy. The interconnectedness of community is what will lead to the changes needed globally.

---

## Disabling Functions (Diagnostic)

The formula can identify structural oppression by detecting signature patterns in construct degradation.

### White Supremacy

White supremacy operates as a population-partitioning suppressor — it creates divergent Φ trajectories across racialized groups while masking the divergence in ξ (truth).

```
W(Φ) = Φ_dominant / Φ_targeted
```

Attack mechanisms per construct:

| Construct | Disabling mechanism | Effect |
|-----------|-------------------|--------|
| c (care) | Racialized resource allocation (redlining, food deserts, Medicaid gaps) | c_targeted → 0.1 while c_dominant → 0.7 |
| κ (compassion) | Empathy boundaries drawn at racial lines (Katrina, Flint) | κ applied selectively |
| j (joy) | Psychological tax of constant threat/surveillance | j suppressed through ambient stress |
| p (purpose) | School-to-prison pipeline, credential discounting | Autonomy structurally constrained |
| ε (empathy) | Segregation prevents cross-group perspective-taking | ε → 0 by design |
| λ_L (love) | **Primary target.** Family separation (slavery → incarceration), destruction of Black Wall Street, gentrification | Community solidarity systematically dismantled |
| λ_P (protection) | **Inverted.** State becomes threat — police violence, stand-your-ground asymmetry | λ_P functionally negative for targeted groups |
| ξ (truth) | **Inverted.** Whitewashed curricula, suppressed records, manufactured narrative | ξ replaced, not merely reduced |

**Signature:** λ_L suppression + ξ inversion + λ_P inversion. The formula detects this because:
- λ_L suppression degrades f(λ_L), which multiplicatively degrades all other constructs
- The equity weights spike for the most deprived constructs, forcing the investigative system to prioritize gaps that white supremacy is designed to hide
- The recovery-aware floor identifies "true collapse" (low x_i + low λ_L + stagnant dx/dt) — no community infrastructure to trigger recovery without external intervention

### Paternalism

```
Signature: c high, λ_P high, λ_L low
```

The divergence penalty fires on (c - λ_L)², reducing Φ even though individual construct values are high. Care without love is institutional without relational grounding.

### Manufactured Scarcity

```
Signature: Φ_dominant ≫ Φ_targeted with high aggregate resources
```

Artificially depressed c for targeted populations when sufficient resources exist globally. The equity weights detect this: care at 0.1 gets ~5× the gradient of care at 0.5, directing the investigative system toward the manufactured gap.

---

## Verification Criteria

### Mathematical Properties

1. **Equity response:** c=0.1, all others=0.5 → c's gradient >5× any other
2. **Community multiplier:** Φ at λ_L=0.1 < 50% of Φ at λ_L=0.8 (all else equal)
3. **Recovery + community:** x_i below floor, dx/dt ≈ 0, λ_L high → effective input near floor
4. **True collapse:** x_i below floor, dx/dt ≈ 0, λ_L low → effective input near x_i
5. **Ubuntu synergy:** Balanced pairs score >10% higher than unbalanced pairs at same average
6. **Divergence penalty:** High c + low λ_L scores lower than moderate c + moderate λ_L
7. **White supremacy signature:** λ_L suppressed + ξ low + λ_P low → gradient spike across multiple constructs
8. **Paternalism detection:** c high, λ_P high, λ_L low → penalty fires, Φ reduced

### Backward Compatibility

9. Symmetric case (all constructs equal) produces similar ordering to current formula
10. Existing test suite passes — scoring internals change, API contracts don't
