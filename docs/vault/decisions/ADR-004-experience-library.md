---
id: ADR-004
title: Immutable ExperienceLibrary over mutable HypothesisEvolutionEngine
status: accepted
date: 2026-02-16
tags: [hypothesis, experience-library, functional, FLEX]
---

# ADR-004: Immutable ExperienceLibrary Pattern (FLEX)

## Decision

Replace the mutable `HypothesisEvolutionEngine` class with a functional pipeline over an immutable `ExperienceLibrary = tuple[Experience, ...]`.

## Context

The original plan had a class with mutable internal state tracking hypothesis evolution. The FLEX paper showed a scaling law: performance improves log-linearly with library size, and reasoning patterns (not domain facts) are what's stored — making the library **inheritable across domains** (Epstein → lobbying patterns → intelligence networks).

A functional pipeline with immutable state is:
- Testable in isolation (pure functions, no side effects)
- Auditable (every experience is recorded, nothing overwritten)
- Domain-transferable (the library accumulates transferable reasoning patterns)

## Branching rule

- `confidence < 0.5` → breadth (Self-Consistency: generate competing hypotheses)
- `confidence >= 0.5` → depth (verification passes)

## Key functions

```python
evolve_hypothesis(h, new_evidence, library, provider) -> tuple[Hypothesis, Experience]
query_similar(library, hypothesis_text, evidence, top_k) -> ExperienceLibrary
```

## Files

- `src/detective/experience.py` — ExperienceLibrary, Experience, add_experience, query_similar (planned)
- `src/detective/evolution.py` — evolve_hypothesis, branching_rule (planned)
- `src/detective/hypothesis.py` — Hypothesis dataclass (existing)
