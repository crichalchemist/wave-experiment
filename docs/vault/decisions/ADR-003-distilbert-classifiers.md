---
id: ADR-003
title: Regex-triggered LLM scoring for assumption detection modules
status: accepted
date: 2026-02-16
tags: [architecture, classification, modules, scoring]
---

# ADR-003: Regex-Triggered LLM Scoring for Assumption Detection Modules

## Decision

Modules A (cognitive bias), B (historical determinism), C (geopolitical presumption) use a two-stage architecture: regex trigger patterns detect candidate spans, then an LLM scores each candidate via `ModelProvider.complete()`. Scores are extracted with the shared `parse_score()` utility from `src/core/scoring.py`.

## Context

Assumption detection requires both pattern recognition (identifying language that signals potential assumptions) and contextual judgment (determining whether the pattern is a genuine assumption in context). A pure classifier approach cannot handle the contextual nuance; a pure LLM approach is too expensive to run on every sentence. The two-stage design uses cheap regex patterns to narrow candidates, then expensive LLM scoring only on triggered spans.

**Disambiguation:** DistilBERT is used elsewhere in the system as a welfare construct classifier (ADR-009), not for assumption detection. The A/B/C modules have never used DistilBERT.

## Architecture

### Module A — Cognitive Bias (`src/detective/module_a.py`)

- **Trigger patterns:** `_BIAS_PATTERNS` dict maps bias type names (confirmation, anchoring, survivorship, ingroup) to compiled regex patterns
- **LLM scoring:** When a `ModelProvider` is provided, each triggered pattern is scored via `_SCORING_PROMPT` asking the LLM to rate bias on 0.0-1.0
- **Keyword-only fallback:** When `provider=None`, matched patterns receive `score=1.0` (no LLM confirmation)
- **Threshold:** `_CONFIDENCE_THRESHOLD = 0.5`
- **Output:** `list[BiasDetection]` (frozen dataclass with `assumption_type`, `score`, `source_text`, `bias_type`)

### Module B — Historical Determinism (`src/detective/module_b.py`)

- **Trigger patterns:** `_DETERMINISM_TRIGGERS` tuple of 13 regex strings matching deterministic language (e.g., `\balways\b`, `\binvariably\b`, `\bhistorically\b`)
- **LLM scoring:** Every match is scored via `_SCORE_PROMPT` with span and sentence context
- **No keyword-only fallback:** `provider` parameter is required (not optional)
- **Threshold:** `_SCORE_THRESHOLD = 0.5`
- **Output:** `list[DeterminismDetection]` (frozen dataclass with `assumption_type`, `score`, `source_text`, `trigger_phrase`), sorted by score descending

### Module C — Geopolitical Presumption (`src/detective/module_c.py`)

- **Trigger patterns:** Two pattern sets must both match — `_ACTOR_PATTERNS` (9 institutional actor regexes: SEC, DOJ, FBI, regulators, courts, etc.) AND `_PRESUMPTION_VERBS` (14 normative verb phrases: "properly reviewed", "fully cooperated", "per standard protocol", etc.)
- **LLM scoring:** Each unique actor match in a sentence with a presumption verb is scored via `_SCORE_PROMPT`
- **No keyword-only fallback:** `provider` parameter is required (not optional)
- **Threshold:** `_SCORE_THRESHOLD = 0.5`
- **Output:** `list[GeopoliticalDetection]` (frozen dataclass with `assumption_type`, `score`, `source_text`, `presumed_actor`), sorted by score descending

### Shared infrastructure

- **`src/core/scoring.py`:** `parse_score(response, default=0.0)` extracts float from LLM responses matching `score:` or `confidence:` patterns (with `:` or `=` separator). `clamp_confidence(value)` bounds to [0.0, 1.0]. Both used by all three modules.
- **`src/core/types.py`:** `AssumptionType` enum with values `COGNITIVE_BIAS`, `HISTORICAL_DETERMINISM`, `GEOPOLITICAL_PRESUMPTION`. Each module's detection dataclass carries the appropriate enum value.

## Consequences

- Regex triggers are fast and deterministic; LLM scoring adds contextual judgment only when needed
- Module A degrades gracefully without a provider (keyword-only mode); B and C require a provider
- All three modules are callable as library functions and are also wired into the investigation agent's analyze phase via ADR-023
- The shared `parse_score()` eliminates the score parsing triplication that existed in the original per-module implementations

## Files

- `src/detective/module_a.py` — cognitive bias detection (regex + optional LLM)
- `src/detective/module_b.py` — historical determinism detection (regex + required LLM)
- `src/detective/module_c.py` — geopolitical presumption detection (actor + verb regex + required LLM)
- `src/core/scoring.py` — shared `parse_score()`, `clamp_confidence()`
- `src/core/types.py` — `AssumptionType` enum
