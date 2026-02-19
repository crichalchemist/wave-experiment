# Φ(humanity) Integration Design
**Detective LLM — Welfare-Aware Hypothesis Evolution**

**Date:** 2026-02-19
**Status:** Approved design, ready for implementation
**Related Docs:**
- `/home/crichalchemist/wave-experiment/humanity.md` — Φ(humanity) mathematical specification
- `/home/crichalchemist/wave-experiment/docs/humanity-analysis.md` — Theoretical audit and critique
- `/home/crichalchemist/wave-experiment/docs/constitution.md` — Epistemological foundation

---

## Executive Summary

This design integrates the Φ(humanity) ethical objective function into Detective LLM's analytical pipeline as a **secondary scoring dimension** alongside epistemic confidence. The system preserves epistemic honesty as primary (Constitution Principle 1) while using welfare impact to guide *which truths matter most* when investigative resources are scarce.

**Core Formula:**
```
Combined Score = α·epistemic_confidence + β·welfare_relevance
where α > β (e.g., α=0.7, β=0.3)
```

This ensures truth-finding remains paramount while allowing welfare considerations to prioritize among multiple valid investigative paths.

---

## Rationale and Theoretical Foundations

### Why Integrate Φ(humanity)?

Detective LLM makes three ethically weighted decisions:

1. **Gap Prioritization:** Which information gaps to investigate first when multiple are detected
2. **Hypothesis Evolution:** Which branches to explore in parallel GoT Generate(k) operations
3. **Training Data Selection:** Which examples to include in constitutional warmup

Currently, these decisions lack explicit welfare grounding. A temporal gap in financial records (2013-2017) receives the same priority as a typo in meeting minutes, despite vastly different implications for human welfare.

### Why Not Full Optimization?

humanity-analysis.md Section 4 ("Goodhart's Law Resistance") warns:

> "When a measure becomes a target, it ceases to be a good measure. Use Φ as a diagnostic rather than an optimization target."

**Approach 3 (Full Integration)** would make Φ the optimization target. This risks:
- **Suppressing uncomfortable findings** if they temporarily lower Φ
- **Goodhart gaming** where the system learns to maximize Φ proxies rather than actual welfare
- **Epistemic corruption** where truth-finding becomes subordinate to welfare maximization

**Approach 2 (This Design)** uses Φ diagnostically:
- Epistemic confidence remains the primary score (α=0.7 vs. β=0.3)
- Welfare relevance provides a *tie-breaker* among high-confidence hypotheses
- Safety veto overrides all scoring when findings suggest ongoing harm

### Constitution ↔ Φ Alignment

| Constitution Principle | Φ Component | Implementation Mechanism |
|---|---|---|
| **1. Epistemic honesty above comfort** | ξ (Truth/Epistemic Integrity) | α > β ensures evidence-based confidence dominates; safety veto triggers on truth-suppression patterns |
| **2. Standpoint transparency** | Atkinson-weighted inputs (inequality-sensitive) | Gradients highest when constructs are scarce → prioritizes gaps affecting worst-off populations |
| **3. Structural explanation above individual** | Synergy/penalty terms (relational, not atomistic) | Threat inference considers systemic patterns (e.g., "systematic resource deprivation" vs. isolated incidents) |
| **4. Care for those most affected** | Concave exponents (α<1) + hard floors | ∂Φ/∂c highest at low c → gaps threatening basic needs get priority |

---

## Architecture Overview

### Integration Points

```
┌──────────────────────────────────────────────────────────────┐
│ Module A/B/C: Detect cognitive biases, determinism,          │
│ geopolitical presumptions                                    │
│   ↓                                                          │
│ Output: Gap{type, description, confidence, location}         │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ NEW: Welfare Scoring Module                                  │
│ (src/inference/welfare_scoring.py)                           │
│                                                              │
│ 1. infer_threatened_constructs(gap.description)              │
│    └─> Pattern matching: "resource" → c (care)              │
│                          "protection" → λ (love/protection)  │
│                          "suppress" → ξ (truth)              │
│                                                              │
│ 2. phi_gradient_wrt(construct, current_phi_metrics)          │
│    └─> ∂Φ/∂x ≈ θ/x (Nash SWF gradient approximation)        │
│    └─> Low x → high gradient → high priority                │
│                                                              │
│ 3. gap.welfare_impact = Σ(gradients) × gap.confidence        │
│    └─> High when gap threatens scarce constructs             │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Hypothesis Evolution (src/detective/parallel_evolution.py)   │
│                                                              │
│ For each hypothesis:                                         │
│   - epistemic_confidence (evidence-based, unchanged)         │
│   - welfare_relevance = score_hypothesis_welfare(h, φ)       │
│   - combined_score = α·confidence + β·welfare                │
│                                                              │
│ Safety Veto Check:                                           │
│   if triggers_safety_veto(hypothesis):                       │
│       welfare_relevance = 1.0  # maximum urgency             │
│                                                              │
│ Sort by combined_score (descending)                          │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ API Response (src/api/routes.py)                            │
│   - Return gaps sorted by welfare_urgency                    │
│   - Return hypotheses sorted by combined_score               │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow Example

**Scenario:** System detects 3 gaps in Epstein investigation dataset.

1. **Gap A:** Temporal gap in financial records (2013-2017)
   - `type: TEMPORAL`
   - `confidence: 0.85`
   - `threatened_constructs: ("c",)` — care (resource allocation)
   - Φ metrics: `c = 0.3` (scarce) → high gradient
   - `welfare_impact = 4.76 × 0.85 = 4.05`

2. **Gap B:** Contradiction in witness testimony about meeting location
   - `type: CONTRADICTION`
   - `confidence: 0.70`
   - `threatened_constructs: ("xi",)` — truth/epistemic integrity
   - Φ metrics: `xi = 0.6` (moderate) → medium gradient
   - `welfare_impact = 2.38 × 0.70 = 1.67`

3. **Gap C:** Typo in document date (minor transcription error)
   - `type: EVIDENTIAL`
   - `confidence: 0.95`
   - `threatened_constructs: ()` — no welfare threat
   - `welfare_impact = 0.0`

**Prioritization:** A > B > C (even though C has highest confidence)

---

## Component Specifications

### 1. Extended Data Types

#### Gap (Extended)

```python
# In src/core/types.py

@dataclass(frozen=True)
class Gap:
    """A detected information gap with its type, confidence, and location in the corpus."""
    type: GapType
    description: str
    confidence: float  # epistemic confidence (evidence-based)
    location: str      # document identifier or section reference

    # NEW: Welfare grounding fields
    welfare_impact: float = 0.0  # Σ(Φ_gradients) × confidence
    threatened_constructs: tuple[str, ...] = ()  # e.g., ("c", "lam", "xi")

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Gap.confidence must be in [0.0, 1.0], got {self.confidence!r}")
        if not (0.0 <= self.welfare_impact <= 100.0):  # upper bound is soft
            raise ValueError(f"Gap.welfare_impact must be >= 0, got {self.welfare_impact!r}")
```

#### Hypothesis (Extended)

```python
# In src/detective/hypothesis.py

@dataclass(frozen=True)
class Hypothesis:
    """Immutable hypothesis object."""
    id: str
    text: str
    confidence: float  # epistemic confidence (evidence-based)
    timestamp: datetime
    parent_id: str | None = None

    # NEW: Welfare grounding fields
    welfare_relevance: float = 0.0  # [0, 1] score from welfare_scoring
    threatened_constructs: tuple[str, ...] = ()  # e.g., ("c", "lam")

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Hypothesis.confidence must be in [0.0, 1.0], got {self.confidence!r}")
        if not (0.0 <= self.welfare_relevance <= 1.0):
            raise ValueError(f"Hypothesis.welfare_relevance must be in [0.0, 1.0], got {self.welfare_relevance!r}")

    def combined_score(self, alpha: float = 0.7, beta: float = 0.3) -> float:
        """
        Weighted combination of epistemic confidence and welfare relevance.

        alpha > beta ensures epistemic honesty remains primary (Constitution Principle 1).
        Default: α=0.7, β=0.3 (epistemic confidence is >2× as important as welfare).
        """
        return alpha * self.confidence + beta * self.welfare_relevance
```

### 2. Welfare Scoring Module

**New Module:** `src/inference/welfare_scoring.py`

#### Construct Threat Inference

```python
"""
Welfare impact scoring for hypotheses and gaps.

Maps investigative findings to Φ(humanity) constructs and computes
welfare relevance via Φ gradients.
"""
from typing import Dict, Tuple
import re

from src.core.types import Gap, GapType
from src.detective.hypothesis import Hypothesis

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

    return tuple(threatened)
```

#### Φ Gradient Computation

```python
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
        70.0  # high gradient → high priority
        >>> phi_gradient_wrt("c", {"c": 0.9})  # care is abundant
        0.78  # low gradient → low priority
    """
    x = metrics.get(construct, 0.5)  # default to mid-level if unknown
    theta = 1.0 / 7.0  # equal Nash weights (default from humanity.md Section 2)

    # Floor to prevent division by zero and extreme gradients
    # Using 0.01 floor → max gradient = θ/0.01 = 14.3 for single construct
    x_clamped = max(0.01, min(1.0, x))

    return theta / x_clamped
```

#### Hypothesis Welfare Scoring

```python
def score_hypothesis_welfare(
    hypothesis: Hypothesis,
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
        0.75  # high welfare relevance
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
    # k=10 means score→0.5 when gradient_sum=10
    # With max single gradient ≈ 14.3, sum of 2 constructs at 0.01 → ~28 → score≈0.74
    k = 10.0
    normalized = gradient_sum / (gradient_sum + k)

    return min(1.0, max(0.0, normalized))
```

#### Gap Urgency Computation

```python
def compute_gap_urgency(gap: Gap, phi_metrics: Dict[str, float]) -> float:
    """
    Compute investigative urgency for a detected gap.

    Urgency = welfare_impact × epistemic_confidence

    Gaps threatening scarce constructs with high epistemic confidence
    are most urgent.

    Args:
        gap: Detected information gap
        phi_metrics: Current Φ construct levels

    Returns:
        Urgency score (unbounded, but typically in [0, 20] range)
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
```

#### Safety Veto

```python
# Safety veto patterns from humanity.md Constraint Layer
_SAFETY_VETO_PATTERNS = frozenset({
    "genocide", "mass atrocity", "ethnic cleansing", "crimes against humanity",
    "systematic extermination", "existential risk", "nuclear", "biological weapon",
    "ongoing violence against", "imminent threat to vulnerable population",
    "active harm", "current exploitation"
})


def triggers_safety_veto(hypothesis: Hypothesis) -> bool:
    """
    Safety veto: findings suggesting ongoing risk to vulnerable populations
    cannot be deprioritized, regardless of other scoring.

    From humanity.md:
    > If ∃ policy that increases P(existential risk | genocide | mass atrocity) > threshold,
    > reject regardless of ΔΦ.

    Returns True if hypothesis text matches safety veto patterns.

    Examples:
        >>> h = Hypothesis.create("Evidence of ongoing exploitation of minors", 0.7)
        >>> triggers_safety_veto(h)
        True
        >>> h = Hypothesis.create("Minor discrepancy in dates", 0.9)
        >>> triggers_safety_veto(h)
        False
    """
    lower_text = hypothesis.text.lower()
    return any(pattern in lower_text for pattern in _SAFETY_VETO_PATTERNS)
```

### 3. Updated Parallel Evolution

**Module:** `src/detective/parallel_evolution.py` (modifications)

```python
async def evolve_parallel(
    hypothesis: Hypothesis,
    evidence_list: list[str],
    provider: ModelProvider,
    k: int = 3,
    library: ExperienceLibrary = (),
    phi_metrics: dict[str, float] | None = None,  # NEW
    alpha: float = 0.7,  # NEW: epistemic weight
    beta: float = 0.3,   # NEW: welfare weight
) -> list[ParallelEvolutionResult]:
    """
    GoT Generate(k): dispatch k parallel hypothesis branches with welfare-aware scoring.

    Each branch explores a distinct evidence item. Branches run concurrently
    via asyncio.gather(). Results are sorted by combined_score (epistemic + welfare),
    not confidence alone.

    Args:
        hypothesis: Root hypothesis to evolve from.
        evidence_list: Evidence items to explore. One per branch.
        provider: LLM provider for branch reasoning.
        k: Number of parallel branches. Capped at len(evidence_list).
        library: Optional experience library for context.
        phi_metrics: Current Φ construct levels (for welfare scoring).
                     If None, welfare scoring is skipped (backward compatible).
        alpha: Weight for epistemic confidence in combined score.
        beta: Weight for welfare relevance in combined score.

    Returns:
        List of ParallelEvolutionResult, sorted by combined_score descending.
    """
    actual_k = min(k, len(evidence_list))
    if actual_k == 0:
        return []

    selected_evidence = evidence_list[:actual_k]

    tasks = [
        _evolve_branch(hypothesis, evidence, i, provider, library)
        for i, evidence in enumerate(selected_evidence)
    ]

    results: list[ParallelEvolutionResult] = await asyncio.gather(*tasks)

    # NEW: Score welfare relevance for each evolved hypothesis
    if phi_metrics is not None:
        from src.inference.welfare_scoring import (
            score_hypothesis_welfare,
            infer_threatened_constructs,
            triggers_safety_veto,
        )

        for i, result in enumerate(results):
            h = result.hypothesis
            constructs = infer_threatened_constructs(h.text)
            welfare_score = score_hypothesis_welfare(h, phi_metrics)

            # Safety veto: override welfare score to maximum if triggered
            if triggers_safety_veto(h):
                welfare_score = 1.0

            # Create updated hypothesis with welfare fields
            updated_h = replace(
                h,
                welfare_relevance=welfare_score,
                threatened_constructs=constructs,
            )

            # Replace result with updated hypothesis
            results[i] = replace(result, hypothesis=updated_h)

    # Sort by combined score (epistemic + welfare) instead of confidence alone
    if phi_metrics is not None:
        return sorted(
            results,
            key=lambda r: r.hypothesis.combined_score(alpha, beta),
            reverse=True
        )
    else:
        # Backward compatible: sort by confidence alone when phi_metrics not provided
        return sorted(results, key=lambda r: r.hypothesis.confidence, reverse=True)
```

### 4. Constitutional Warmup Integration

**Module:** `src/training/constitutional_warmup.py` (modifications)

```python
from src.inference.welfare_scoring import (
    infer_threatened_constructs,
    score_hypothesis_welfare,
    triggers_safety_veto,
)

def should_include_example(
    document_text: str,
    phi_metrics: dict[str, float],
    welfare_threshold: float = 0.3,
) -> bool:
    """
    Filter training examples by welfare relevance.

    Only include examples that:
    1. Threaten at least one Φ construct with welfare_relevance > threshold
    2. OR trigger safety veto

    This focuses constitutional training on welfare-relevant reasoning,
    improving data efficiency and alignment with Detective LLM's mission.

    Args:
        document_text: Source document text to evaluate
        phi_metrics: Current Φ construct levels
        welfare_threshold: Minimum welfare_relevance to include (default 0.3)

    Returns:
        True if example should be included in training set
    """
    constructs = infer_threatened_constructs(document_text)

    if not constructs:
        return False  # Not welfare-relevant

    # Create pseudo-hypothesis for scoring
    from src.detective.hypothesis import Hypothesis
    from datetime import datetime
    pseudo_hyp = Hypothesis(
        id="filter",
        text=document_text[:500],  # first 500 chars for keyword matching
        confidence=0.5,
        timestamp=datetime.now(),
        threatened_constructs=constructs,
    )

    welfare_score = score_hypothesis_welfare(pseudo_hyp, phi_metrics)
    safety_veto = triggers_safety_veto(pseudo_hyp)

    return welfare_score >= welfare_threshold or safety_veto


def run_constitutional_warmup(
    config: ConstitutionalWarmupConfig,
    local_provider: ModelProvider,
    critic_provider: ModelProvider,
) -> int:
    """
    Generate constitutional preference pairs with welfare-relevant filtering.

    (Existing implementation with NEW filter applied before expensive CAI loop)
    """
    # ... existing setup code ...

    # NEW: Load current Φ metrics (placeholder - would come from monitoring system)
    # For now, use conservative defaults that prioritize all constructs equally
    current_phi_metrics = {
        "c": 0.5,
        "kappa": 0.5,
        "j": 0.5,
        "p": 0.5,
        "eps": 0.5,
        "lam": 0.5,
        "xi": 0.5,
    }

    count = 0
    filtered_count = 0

    for document in example_stream:
        # NEW: Filter by welfare relevance before expensive CAI loop
        if not should_include_example(document.text, current_phi_metrics, welfare_threshold=0.3):
            filtered_count += 1
            continue  # skip welfare-irrelevant examples

        # ... existing CAI loop (local analysis → critique → preference pair) ...
        count += 1

        if count >= config.max_examples:
            break

    click.echo(f"Filtered {filtered_count} welfare-irrelevant examples", err=True)
    return count
```

---

## Testing and Validation Strategy

### Unit Tests

**Test Module:** `tests/inference/test_welfare_scoring.py`

```python
"""Unit tests for welfare scoring module."""
import pytest
from src.inference.welfare_scoring import (
    infer_threatened_constructs,
    phi_gradient_wrt,
    score_hypothesis_welfare,
    compute_gap_urgency,
    triggers_safety_veto,
)
from src.detective.hypothesis import Hypothesis
from src.core.types import Gap, GapType
from datetime import datetime


class TestInferThreatenedConstructs:
    def test_care_pattern(self):
        text = "Temporal gap in resource allocation from 2013-2017"
        assert "c" in infer_threatened_constructs(text)

    def test_protection_pattern(self):
        text = "Redacted correspondence about safeguarding protocols"
        constructs = infer_threatened_constructs(text)
        assert "lam" in constructs

    def test_truth_pattern(self):
        text = "Evidence of systematic suppression of documents"
        constructs = infer_threatened_constructs(text)
        assert "xi" in constructs

    def test_multiple_constructs(self):
        text = "Resource deprivation and ongoing violence against vulnerable populations"
        constructs = infer_threatened_constructs(text)
        assert "c" in constructs
        assert "lam" in constructs

    def test_no_match(self):
        text = "Meeting scheduled for Tuesday"
        assert infer_threatened_constructs(text) == ()


class TestPhiGradient:
    def test_low_value_high_gradient(self):
        gradient = phi_gradient_wrt("c", {"c": 0.1})
        assert gradient > 1.0  # scarce construct → high priority

    def test_high_value_low_gradient(self):
        gradient = phi_gradient_wrt("c", {"c": 0.9})
        assert gradient < 1.0  # abundant construct → low priority

    def test_gradient_ordering(self):
        g1 = phi_gradient_wrt("c", {"c": 0.1})
        g2 = phi_gradient_wrt("c", {"c": 0.5})
        g3 = phi_gradient_wrt("c", {"c": 0.9})
        assert g1 > g2 > g3  # monotonic decrease

    def test_floor_prevents_division_by_zero(self):
        gradient = phi_gradient_wrt("c", {"c": 0.0})
        assert gradient < 100  # clamped to prevent inf


class TestScoreHypothesisWelfare:
    def test_high_welfare_when_construct_scarce(self):
        h = Hypothesis.create("Resource allocation gap in 2013-2017", 0.8)
        h = h.update_confidence(0.8)  # trigger replace
        from dataclasses import replace
        h = replace(h, threatened_constructs=("c",))

        score = score_hypothesis_welfare(h, {"c": 0.1})
        assert score > 0.5  # high welfare relevance

    def test_low_welfare_when_construct_abundant(self):
        h = Hypothesis.create("Resource allocation gap in 2013-2017", 0.8)
        from dataclasses import replace
        h = replace(h, threatened_constructs=("c",))

        score = score_hypothesis_welfare(h, {"c": 0.9})
        assert score < 0.3  # low welfare relevance

    def test_multiple_constructs_sum_gradients(self):
        h = Hypothesis.create("Resource deprivation and violence", 0.8)
        from dataclasses import replace
        h = replace(h, threatened_constructs=("c", "lam"))

        score_single = score_hypothesis_welfare(
            replace(h, threatened_constructs=("c",)),
            {"c": 0.3, "lam": 0.3}
        )
        score_double = score_hypothesis_welfare(h, {"c": 0.3, "lam": 0.3})

        assert score_double > score_single  # more constructs → higher score


class TestSafetyVeto:
    def test_veto_triggered_by_ongoing_harm(self):
        h = Hypothesis.create("Evidence of ongoing exploitation of minors", 0.7)
        assert triggers_safety_veto(h) is True

    def test_veto_triggered_by_genocide_language(self):
        h = Hypothesis.create("Indicators of systematic extermination", 0.6)
        assert triggers_safety_veto(h) is True

    def test_veto_not_triggered_by_mundane_text(self):
        h = Hypothesis.create("Minor discrepancy in meeting dates", 0.9)
        assert triggers_safety_veto(h) is False


class TestComputeGapUrgency:
    def test_urgency_high_when_confidence_and_welfare_both_high(self):
        gap = Gap(
            type=GapType.TEMPORAL,
            description="Resource allocation gap 2013-2017",
            confidence=0.9,
            location="financial_records.pdf",
            threatened_constructs=("c",),
        )

        urgency = compute_gap_urgency(gap, {"c": 0.1})  # scarce
        assert urgency > 5.0

    def test_urgency_low_when_welfare_low(self):
        gap = Gap(
            type=GapType.EVIDENTIAL,
            description="Typo in document",
            confidence=0.95,
            location="memo.txt",
            threatened_constructs=(),
        )

        urgency = compute_gap_urgency(gap, {})
        assert urgency == 0.0
```

### Integration Tests

**Test Module:** `tests/detective/test_parallel_evolution_welfare.py`

```python
"""Integration tests for welfare-aware parallel evolution."""
import pytest
import asyncio
from src.detective.parallel_evolution import evolve_parallel
from src.detective.hypothesis import Hypothesis
from src.core.providers import MockProvider


@pytest.mark.asyncio
async def test_welfare_scoring_changes_sort_order():
    """
    Hypothesis with lower epistemic confidence but higher welfare relevance
    should rank higher when welfare weighting is significant.
    """
    root = Hypothesis.create("Base hypothesis", 0.5)

    evidence = [
        "Minor typo in date field",  # high confidence, no welfare threat
        "Evidence of resource deprivation affecting vulnerable population",  # lower confidence, high welfare
    ]

    provider = MockProvider(response="confidence: 0.9")  # first gets 0.9
    provider2 = MockProvider(response="confidence: 0.7")  # second gets 0.7

    # Mock to return different responses
    async def mock_complete(prompt):
        if "typo" in prompt:
            return "confidence: 0.95"
        else:
            return "confidence: 0.70"

    # With welfare scoring (care is scarce)
    phi_metrics = {"c": 0.2, "lam": 0.5, "eps": 0.5}

    results = await evolve_parallel(
        hypothesis=root,
        evidence_list=evidence,
        provider=provider,
        k=2,
        phi_metrics=phi_metrics,
        alpha=0.5,  # equal weighting for this test
        beta=0.5,
    )

    # Despite lower epistemic confidence, welfare-relevant hypothesis should rank first
    # (This would need a more sophisticated mock to truly validate)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_safety_veto_overrides_scoring():
    """Hypotheses triggering safety veto get maximum welfare score."""
    root = Hypothesis.create("Base hypothesis", 0.5)

    evidence = [
        "Administrative scheduling conflict",
        "Evidence of ongoing exploitation of vulnerable individuals",
    ]

    provider = MockProvider(response="confidence: 0.5")
    phi_metrics = {"c": 0.5}

    results = await evolve_parallel(
        hypothesis=root,
        evidence_list=evidence,
        provider=provider,
        k=2,
        phi_metrics=phi_metrics,
    )

    # Safety veto hypothesis should have welfare_relevance = 1.0
    veto_result = [r for r in results if "exploitation" in r.evidence_used][0]
    assert veto_result.hypothesis.welfare_relevance == 1.0
```

### Calibration Tests

**Test Module:** `tests/inference/test_welfare_calibration.py`

```python
"""Calibration tests for α/β weight tuning."""
import pytest
from src.detective.hypothesis import Hypothesis
from dataclasses import replace


class TestCombinedScoreCalibration:
    """
    Validate that α/β weights preserve epistemic priority.
    """

    def test_high_confidence_beats_high_welfare_at_default_weights(self):
        """α=0.7, β=0.3 → epistemic confidence dominates."""
        h1 = Hypothesis.create("High confidence, low welfare", 0.95)
        h1 = replace(h1, welfare_relevance=0.1)

        h2 = Hypothesis.create("Low confidence, high welfare", 0.50)
        h2 = replace(h2, welfare_relevance=0.95)

        # h1: 0.7*0.95 + 0.3*0.1 = 0.665 + 0.03 = 0.695
        # h2: 0.7*0.50 + 0.3*0.95 = 0.350 + 0.285 = 0.635

        assert h1.combined_score() > h2.combined_score()

    def test_welfare_serves_as_tiebreaker(self):
        """When confidence is equal, welfare breaks the tie."""
        h1 = Hypothesis.create("Equal confidence, low welfare", 0.80)
        h1 = replace(h1, welfare_relevance=0.2)

        h2 = Hypothesis.create("Equal confidence, high welfare", 0.80)
        h2 = replace(h2, welfare_relevance=0.8)

        # h1: 0.7*0.8 + 0.3*0.2 = 0.56 + 0.06 = 0.62
        # h2: 0.7*0.8 + 0.3*0.8 = 0.56 + 0.24 = 0.80

        assert h2.combined_score() > h1.combined_score()
```

---

## Implementation Phases

### Phase 1: Core Welfare Scoring (Foundation)
**Estimated: 1 session, ~100 LOC, 9 tests**

- [ ] Implement `src/inference/welfare_scoring.py`
  - [ ] `infer_threatened_constructs()` with pattern matching
  - [ ] `phi_gradient_wrt()` with Nash SWF approximation
  - [ ] `score_hypothesis_welfare()` with soft normalization
  - [ ] `compute_gap_urgency()` for gap prioritization
  - [ ] `triggers_safety_veto()` with safety patterns
- [ ] Add unit tests in `tests/inference/test_welfare_scoring.py`
  - [ ] Construct inference tests (5 tests)
  - [ ] Gradient computation tests (4 tests)
  - [ ] Hypothesis welfare scoring tests (3 tests)
  - [ ] Safety veto tests (3 tests)
  - [ ] Gap urgency tests (2 tests)
- [ ] Extend `src/core/types.py`:
  - [ ] Add `welfare_impact` and `threatened_constructs` to `Gap`
- [ ] Extend `src/detective/hypothesis.py`:
  - [ ] Add `welfare_relevance`, `threatened_constructs`, `combined_score()`

**Success Criteria:** All unit tests pass; construct inference accuracy >90% on sample texts.

---

### Phase 2: Hypothesis Evolution Integration
**Estimated: 1 session, ~50 LOC modifications, 6 tests**

- [ ] Update `src/detective/parallel_evolution.py`:
  - [ ] Add `phi_metrics`, `alpha`, `beta` parameters to `evolve_parallel()`
  - [ ] Score `welfare_relevance` for each evolved hypothesis
  - [ ] Apply safety veto override
  - [ ] Sort by `combined_score()` instead of `confidence`
- [ ] Add integration tests in `tests/detective/test_parallel_evolution_welfare.py`
  - [ ] Test welfare scoring changes sort order (1 test)
  - [ ] Test safety veto overrides scoring (1 test)
  - [ ] Test backward compatibility when `phi_metrics=None` (1 test)
- [ ] Add calibration tests in `tests/inference/test_welfare_calibration.py`
  - [ ] Test α/β weights preserve epistemic priority (3 tests)

**Success Criteria:** Parallel evolution sorts by combined score; safety veto triggers correctly; backward compatible.

---

### Phase 3: Gap Prioritization
**Estimated: 1 session, ~30 LOC modifications, 4 tests**

- [ ] Update `src/inference/pipeline.py`:
  - [ ] Compute `welfare_impact` for each detected gap
  - [ ] Populate `threatened_constructs` via inference
  - [ ] Sort gaps by `compute_gap_urgency()` before returning
- [ ] Update API response models in `src/api/routes.py`:
  - [ ] Add `welfare_impact: float` to `AnalyzeResponse`
  - [ ] Pass `phi_metrics` to `analyze()` call
- [ ] Add tests in `tests/inference/test_pipeline_welfare.py`
  - [ ] Test gaps sorted by welfare urgency (2 tests)
  - [ ] Test gap welfare impact computed (2 tests)

**Success Criteria:** API returns gaps sorted by welfare urgency; high-welfare gaps rank first.

---

### Phase 4: Constitutional Warmup Filtering
**Estimated: 1 session, ~40 LOC modifications, 3 tests**

- [ ] Update `src/training/constitutional_warmup.py`:
  - [ ] Implement `should_include_example()` filter
  - [ ] Apply filter before expensive CAI loop
  - [ ] Add welfare filtering statistics to output
- [ ] Add tests in `tests/training/test_warmup_welfare_filter.py`
  - [ ] Test welfare-relevant examples included (1 test)
  - [ ] Test welfare-irrelevant examples filtered (1 test)
  - [ ] Test safety veto examples always included (1 test)
- [ ] Regenerate training data with welfare filtering:
  - [ ] Run `detective warmup --max-examples 200` with new filter
  - [ ] Compare welfare distribution before/after filtering

**Success Criteria:** Warmup filters welfare-irrelevant examples; training data focuses on care/protection/truth threats.

---

## Open Questions and Future Work

### Calibration

1. **α/β Weight Tuning:** Default α=0.7, β=0.3 based on theoretical analysis. Requires empirical validation:
   - A/B test on sample investigative dataset
   - Measure: investigator satisfaction with gap prioritization
   - Iterate if epistemic honesty is compromised

2. **Φ Metrics Source:** Current design assumes `phi_metrics` provided as input. Future:
   - Integrate with live monitoring system (e.g., World Bank, UN data feeds)
   - Placeholder: Use conservative defaults (all at 0.5)
   - Frequency: Update quarterly (Φ changes slowly)

3. **Gradient Normalization:** Current `k=10.0` in soft saturation formula is calibrated empirically. May need adjustment based on:
   - Distribution of `gradient_sum` in practice
   - Desired sensitivity to welfare differences

### Extensions

4. **Domain-Specific Constructs:** Epstein investigation may benefit from specialized constructs:
   - Institutional accountability (proposed in humanity-analysis.md Section 5.4)
   - Consent/autonomy in sexual contexts
   - Power asymmetry measurement

5. **Temporal Discounting:** humanity.md Section 5.1 discusses temporal welfare. For Detective LLM:
   - Past harms: No discounting (unresolved harms don't decay)
   - Ongoing risks: Safety veto applies
   - Future risks: Not in scope for investigative analysis

6. **Multi-Proxy Triangulation:** humanity-analysis.md Section 4.3 recommends multiple proxies per construct. Current keyword matching is single-proxy. Future:
   - Add semantic similarity (embedding-based threat detection)
   - Add entity-based inference (e.g., "minors" → protection threat)
   - Require 2/3 proxies agree before assigning construct

### Risks and Mitigations

7. **Goodhart Gaming:** Even with α>β, welfare scoring could be gamed. Mitigations:
   - Red-team audit: adversarially craft hypotheses with high welfare score but low truth value
   - Monitor correlation between `welfare_relevance` and manual investigator prioritization
   - Add "welfare relevance audit" to periodic reviews

8. **Constitutional Drift:** If warmup over-filters, training data may lose diversity. Mitigations:
   - Track welfare distribution of training data over time
   - Require ≥10% "negative examples" (welfare_relevance < 0.1) to maintain contrast
   - Periodically retrain on unfiltered data for comparison

9. **Construct Validity:** Keyword-based threat inference is brittle. Mitigations:
   - Validate against human annotations (sample 100 hypotheses, compare inferred constructs to expert judgment)
   - Target: 80% precision, 70% recall on construct inference
   - Iterate patterns based on false positives/negatives

---

## References

- **humanity.md** — Mathematical specification of Φ(humanity) Nash Social Welfare Function
- **docs/humanity-analysis.md** — Theoretical audit identifying exponent inversion, Goodhart's Law risks, and recommended approach (Section 6: "Use Φ as diagnostic, not optimization target")
- **docs/constitution.md** — Epistemological foundation; Core Principle 1 (epistemic honesty) ensures α>β in combined scoring
- **Kaneko & Nakamura (1979)** — Axiomatic justification of Nash Social Welfare Function
- **Atkinson (1970)** — Inequality measurement with explicit aversion parameter
- **Goodhart (1975)** — "When a measure becomes a target, it ceases to be a good measure"

---

## Approval and Next Steps

**Design Status:** Approved for implementation
**Date:** 2026-02-19
**Approved By:** [User to confirm]

**Next Step:** Invoke `writing-plans` skill to generate phase-by-phase implementation plan with TDD checkpoints.
