---
id: ADR-002
title: Moral compass as dual-function constitution
status: accepted
date: 2026-02-16
tags: [constitution, alignment, CAI, training]
---

# ADR-002: Moral Compass as Dual-Function Constitution

## Decision

`docs/constitution.md` serves two distinct functions simultaneously: (1) Constitutional AI training signal and (2) inline ID-Sampling reflection constraint.

## Context

Detective LLM's primary epistemic failure mode is **pattern completion over gap detection** — the model produces coherent narratives that suppress gaps rather than surface them. A constitution grounded in standpoint epistemology (Collins), structural silence (Wilkerson), and credibility resistance (Baldwin) was needed to correct this.

The constitution is:
- **Training signal**: Generate analysis → Critique against constitution (Claude as external critic) → Revise → Use as preference pair. Generates preference pairs at scale without human annotators.
- **Inline constraint**: The ID-Sampling "Wait — before I continue" trigger becomes constitution-grounded. When the constitution IS the instruction, the SynchToM failure mode (instruction-following over epistemic alignment) disappears.

## Why this matters

Using Claude (external, different model) as critic produces stronger critiques than same-model self-critique. The separation of generator (local vLLM) and critic (Azure Foundry Claude) is architecturally superior.

## Implementation

- `src/detective/constitution.py` provides `load_constitution()`, `critique_against_constitution()`, and `generate_preference_pair()`. Constitution path is configurable via `DETECTIVE_CONSTITUTION_PATH` env var (default: `docs/constitution.md`).
- `PreferencePair` is a frozen dataclass with `instruction`, `rejected` (original analysis), and `chosen` (revised analysis) fields.
- `src/security/prompt_guard.py` provides `build_mentor_critique_prompt()` (mentor-framed critique for Claude as trusted guide) and `build_revision_prompt()` (frames mentor guidance as internalization, not compliance).
- `src/training/constitutional_warmup.py` is a full implementation of the constitution-first training pipeline: document -> local model analysis -> Claude critique -> revised analysis -> DPO pair.

## Files

- `docs/constitution.md` -- the moral compass document
- `src/detective/constitution.py` -- load_constitution, critique_against_constitution, PreferencePair, generate_preference_pair
- `src/security/prompt_guard.py` -- build_analysis_prompt, build_critique_prompt, build_mentor_critique_prompt, build_revision_prompt
- `src/training/constitutional_warmup.py` -- constitutional warmup pipeline (ConstitutionalWarmupConfig, run_constitutional_warmup)
