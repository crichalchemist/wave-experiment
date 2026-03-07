---
id: ADR-023
title: A+B+C assumption detection integration in investigation agent
status: accepted
date: 2026-03-07
tags: [investigation, assumptions, modules, counter-leads]
---

# ADR-023: A+B+C Assumption Detection Integration

## Decision

Integrate the three assumption detection modules (A: cognitive biases, B: historical determinism, C: geopolitical presumptions) into the investigation agent's analyze phase as a "Log + Investigate" sub-step. Each detection is logged as a `Finding(is_assumption_finding=True)` and generates a counter-lead that probes whether the assumption holds. Counter-leads enter the lead queue and are gathered in the next iteration.

## Context

Modules A, B, and C were fully implemented as standalone library functions (`src/detective/module_a.py`, `module_b.py`, `module_c.py`) but had zero wiring into any automated pipeline. The investigation agent's analyze phase ran the 4-layer analysis pipeline on documents but never scanned for assumptions. This meant the agent could reason about evidence without questioning whether that evidence carried inherent biases, deterministic framing, or institutional presumptions.

Three integration approaches were considered:

1. **Log only** — flag assumptions in findings, no action taken
2. **Log + Discount** — flag assumptions and inject warnings into hypothesis evolution prompts
3. **Log + Investigate** — flag assumptions and generate counter-leads to probe them

Approach 3 was chosen because it leverages the existing gather→analyze loop to self-correct rather than polluting evolution prompts with warnings (approach 2) or leaving assumptions unaddressed (approach 1).

## Architecture

```
ANALYZE phase
    ├─→ 4-layer pipeline (existing, per doc × hyp[:3])
    │     → list[AnalysisResult]
    │
    └─→ _assumption_scan(docs)  [NEW, per unique doc]
          Module A: detect_cognitive_biases(text, provider?, threshold)
          Module B: detect_historical_determinism(text, provider, threshold)
          Module C: detect_geopolitical_presumptions(text, provider, threshold)
            → AssumptionScanResult per doc
            → Finding(is_assumption_finding=True) per detection
            → _generate_counter_leads(detections)
                → Lead(query="counter-query", source_id=routed_source)
                → added to self._lead_queue
```

Counter-lead source routing:
- Module C (institutional presumptions) → `court_listener` > `sec_edgar` > `web_search`
- Modules A/B → `web_search` > `news_search`

Budget guardrails:
- Max 5 unique docs per scan (`_ASSUMPTION_SCAN_MAX_DOCS`)
- Max 10 findings per scan (`_ASSUMPTION_MAX_FINDINGS_PER_SCAN`)
- Max 3 counter-leads per scan (`_MAX_COUNTER_LEADS`)
- Module A falls back to keyword-only mode (no LLM) when remaining budget < 10
- Modules B/C skipped when remaining budget < 5
- Per-doc budget check stops scan early

## Consequences

**Positive:**
- Assumptions in evidence are now surfaced in the audit trail as typed findings
- Counter-leads create a self-correcting feedback loop: detected assumptions generate investigation queries that validate or refute them
- No changes to the evolve phase — keeps prompt clean, avoids warning injection pollution
- Budget-aware: degrades gracefully under constraint (keyword-only Module A, skip B/C)
- Backward compatible: `enable_assumption_scan=True` by default, old configs work unchanged

**Negative:**
- Adds ~30-55 LLM calls per iteration in worst case (most patterns won't match in practice)
- Counter-leads may produce low-value evidence if the assumption was already well-founded
- Module A keyword-only mode has higher false positive rate than LLM-scored mode

**Neutral:**
- New CLI flags: `--assumptions/--no-assumptions`, `--assumption-threshold`
- New API field: `assumptions_detected` in status response
- New types: `AssumptionDetection`, `AssumptionScanResult` (frozen dataclasses)

## Files

| File | Change |
|------|--------|
| `src/detective/investigation/types.py` | Added `AssumptionDetection`, `AssumptionScanResult`; extended `Finding`, `InvestigationConfig`, `InvestigationStep`, `InvestigationReport` |
| `src/detective/investigation/agent.py` | Added `_assumption_scan()`, `_generate_counter_leads()`, `_route_counter_lead()`; wired into `_analyze_phase()`, `run()`, `_record_step()`, `_build_report()`, `status` |
| `src/detective/investigation/__init__.py` | Re-exported `AssumptionDetection`, `AssumptionScanResult` |
| `src/cli/main.py` | Added `--assumptions`, `--assumption-threshold` flags; assumptions count in output |
| `src/api/routes.py` | Added `assumptions_detected` to `InvestigationStatusResponse` |
| `tests/detective/investigation/test_assumption_scan.py` | 42 tests: types, scan, counter-leads, integration, routing |
