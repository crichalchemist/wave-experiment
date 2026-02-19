# Φ(humanity) Integration Guide

## Overview

Detective LLM uses Φ(humanity) as a **secondary scoring dimension** to prioritize investigative work by welfare impact while preserving epistemic honesty as primary.

**Core Formula:**
```
Combined Score = α·epistemic_confidence + β·welfare_relevance
where α=0.7, β=0.3 (epistemic truth is >2× as important as welfare)
```

## Quick Start

### Hypothesis Evolution with Welfare Scoring

```python
import asyncio
from src.detective.hypothesis import Hypothesis
from src.detective.parallel_evolution import evolve_parallel
from src.core.providers import provider_from_env

# Define current Φ construct levels (from monitoring system)
phi_metrics = {
    "c": 0.3,      # care (resource allocation)
    "kappa": 0.5,  # compassion (responsive support)
    "j": 0.6,      # joy (positive affect)
    "p": 0.5,      # purpose (goal alignment)
    "eps": 0.4,    # empathy (perspective-taking)
    "lam": 0.3,    # protection (safeguarding)
    "xi": 0.4,     # truth (epistemic integrity)
}

# Evolve hypothesis with welfare-aware sorting
root = Hypothesis.create("Temporal gap in records", 0.5)
evidence = [
    "Evidence of resource deprivation 2013-2017",
    "Minor administrative note",
    "Testimony about ongoing harm",
]

provider = provider_from_env()
results = asyncio.run(evolve_parallel(
    hypothesis=root,
    evidence_list=evidence,
    provider=provider,
    k=3,
    phi_metrics=phi_metrics,  # Enable welfare scoring
    alpha=0.7,  # Epistemic weight
    beta=0.3,   # Welfare weight
))

# Results sorted by combined_score (epistemic + welfare)
best = results[0]
print(f"Best hypothesis: {best.hypothesis.text}")
print(f"Epistemic confidence: {best.hypothesis.confidence:.2f}")
print(f"Welfare relevance: {best.hypothesis.welfare_relevance:.2f}")
print(f"Threatened constructs: {best.hypothesis.threatened_constructs}")
```

### Gap Prioritization

```python
from src.core.types import Gap, GapType
from src.inference.pipeline import score_gaps_welfare

gaps = [
    Gap(
        type=GapType.TEMPORAL,
        description="Resource allocation gap 2013-2017",
        confidence=0.9,
        location="financial_records.pdf",
    ),
    Gap(
        type=GapType.EVIDENTIAL,
        description="Typo in meeting minutes",
        confidence=0.95,
        location="minutes.txt",
    ),
]

# Score and sort by welfare urgency
scored_gaps = score_gaps_welfare(gaps, phi_metrics)

for gap in scored_gaps:
    print(f"{gap.description}: urgency={gap.welfare_impact:.2f}")
```

### Constitutional Warmup Filtering

```python
from src.training.constitutional_warmup import (
    run_constitutional_warmup,
    ConstitutionalWarmupConfig,
)

config = ConstitutionalWarmupConfig(
    output_path="data/training/constitutional_pairs.jsonl",
    max_examples=200,
    constitution_path="docs/constitution.md",
)

# Warmup automatically filters welfare-irrelevant examples
count = run_constitutional_warmup(
    config=config,
    local_provider=local_provider,
    critic_provider=critic_provider,
)

print(f"Generated {count} welfare-relevant preference pairs")
```

## High-Urgency Patterns

Findings relating to **scarce constructs** receive high welfare relevance scores, helping prioritize investigation:

```python
from src.inference.welfare_scoring import score_hypothesis_welfare

h = Hypothesis.create("Evidence of ongoing resource deprivation", 0.8)
phi_metrics = {"c": 0.1, "lam": 0.2}  # care and protection are scarce

welfare_score = score_hypothesis_welfare(h, phi_metrics)
# High welfare_relevance due to scarce constructs
print(f"Welfare relevance: {welfare_score:.2f}")  # e.g., 0.74
```

**Note on Investigative Freedom:** Welfare scoring remains **advisory only** and never blocks investigation. Even hypotheses with low welfare_relevance remain accessible. In investigative and cyberdefense contexts, "dangerous" or low-welfare leads may be precisely what need investigation.

## Φ Construct Mapping

| Symbol | Construct | Keyword Patterns |
|--------|-----------|------------------|
| **c** | Care | resource, allocation, provision, basic needs, deprivation |
| **kappa** | Compassion | distress, crisis, support, relief |
| **j** | Joy | wellbeing, happiness, flourishing |
| **p** | Purpose | autonomy, agency, goals, meaning |
| **eps** | Empathy | perspective, discrimination, marginalized |
| **lam** | Protection | safeguard, violence, harm, exploitation, vulnerability |
| **xi** | Truth | suppress, conceal, falsify, contradiction |

## Default Φ Metrics

When real-time monitoring is not available, use conservative defaults:

```python
DEFAULT_PHI_METRICS = {
    "c": 0.5,
    "kappa": 0.5,
    "j": 0.5,
    "p": 0.5,
    "eps": 0.5,
    "lam": 0.5,
    "xi": 0.5,
}
```

These neutral values mean all constructs are treated equally until monitoring data is integrated.

## Weight Calibration

**Default: α=0.7, β=0.3**

This preserves epistemic honesty (Constitution Principle 1) while allowing welfare to guide priority.

**To adjust:**
```python
results = asyncio.run(evolve_parallel(
    ...,
    alpha=0.8,  # Increase epistemic weight
    beta=0.2,   # Decrease welfare weight
))
```

**When to adjust:**
- α→0.9, β→0.1: Pure investigative research (welfare as weak tie-breaker)
- α→0.6, β→0.4: Crisis response (welfare more important)
- **Never** α<β: Violates Constitution Principle 1

## API Integration

### POST /evolve with phi_metrics

```bash
curl -X POST http://localhost:8000/evolve \
  -H "Content-Type: application/json" \
  -d '{
    "evidence_path": "Evidence of resource deprivation",
    "phi_metrics": {
      "c": 0.2,
      "lam": 0.3
    }
  }'
```

Response includes hypothesis with welfare scoring applied.

## References

- **Design Doc:** `docs/plans/2026-02-19-humanity-phi-integration.md`
- **Φ Specification:** `humanity.md`
- **Theoretical Analysis:** `docs/humanity-analysis.md`
- **Constitution:** `docs/constitution.md`
