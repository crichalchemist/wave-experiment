---
id: ADR-DRIFT-RESOLVE
title: ADR drift resolution audit (2026-03-07)
status: resolved
date: 2026-03-07
tags: [audit, documentation, drift, maintenance]
---

# ADR-DRIFT-RESOLVE: Full ADR Drift Resolution Audit

## Purpose

Comprehensive audit of all 27 ADRs against their implementations, cataloging every
discrepancy and recording the resolution direction chosen for each.

**Directions:**
- **D→C** (Docs→Code): Update the ADR to match the working implementation
- **C→D** (Code→Docs): Fix the implementation to match the ADR's design intent
- **DESIGN**: Requires new design work before resolution
- **DEFER**: Acknowledged but not resolved now
- **OK**: No action needed

---

## Critical Discrepancies

| # | ADR | Issue | Direction | Status |
|---|-----|-------|-----------|--------|
| 1 | ADR-003 | Claims DistilBERT classifiers for A/B/C; actual is regex+LLM scoring | D→C | done |
| 2 | ADR-007 | Claims LegalContext nodes; actual is edge-level legal_domain on KnowledgeEdge | DESIGN | done |
| 3 | ADR-008 | Wrong env vars, wrong Protocol methods, wrong Kuzu schema | D→C | done |
| 4 | ADR-010 | `curiosity_relevance` never populated in parallel_evolution | DESIGN | done |

## Medium Discrepancies

| # | ADR | Issue | Direction | Status |
|---|-----|-------|-----------|--------|
| 5 | ADR-001 | Only describes 2 providers; now has 3 + critic_provider + classify_prompt | D→C | done |
| 6 | ADR-005 | Missing CO_MENTIONED, ASSOCIATED relation types | D→C | done |
| 7 | ADR-012 | Overstates Scrapling features (claims StealthyFetcher/Spider; uses plain Fetcher) | D→C | done |
| 8 | ADR-013 | Missing 60s auto-recovery cooldown documentation | D→C | done |
| 9 | ADR-014 | Claims CORS GET-only; actual is GET+POST | D→C | done |
| 10 | ADR-015 | Layer 2 dedup now uses MinHash/LSH via dedup.py, not just difflib | D→C | done |
| 11 | ADR-016 | Same MinHash/LSH upgrade not reflected; timing claims may be stale | D→C | done |
| 12 | ADR-018 | `limit_results()` unused; `load_iicsa_reports()` violates DocumentLoader Protocol | C→D | done |
| 13 | ADR-020 | Claims spaCy removed; ADR-025 re-added it as optional dep | D→C | done |
| 14 | ADR-021 | Missing PersonAuditSummary, _audit_phase, dedup in gather; dataclass count outdated | D→C | done |

## Low Discrepancies

| # | ADR | Issue | Direction | Status |
|---|-----|-------|-----------|--------|
| 15 | ADR-002 | Files marked "planned" that are fully implemented | D→C | done |
| 16 | ADR-004 | Files marked "planned" that are fully implemented | D→C | done |
| 17 | ADR-006 | `list[str]` vs `tuple[str, ...]` for findings; missing mentor-framed critique | D→C | done |
| 18 | ADR-011 | Incorrectly claims compute_phi() doesn't accept derivatives | D→C | done |
| 19 | ADR-019 | File counts slightly off (more files added since ADR) | OK | — |
| 20 | ADR-021 | Test count "837" now 971 | D→C | done |
| 21 | ADR-ERR-LOG | BUG-010 incorrect KeyboardInterrupt claim; test counts stale; BUG-002 A/B/C no longer dead | D→C | done |
| 22 | ADR-024 | Missing chain init order, confidence averaging detail | D→C | done |
| 23 | ADR-025 | Missing secondary _audit_phase usage | D→C | done |
| 24 | ADR-026 | Missing is_duplicate() from API spec | D→C | done |
| 25 | ADR-027 | "action" claim type listed but not implemented; misleading "budget-aware" | D→C | done |

## Resolution Summary

**Resolved 2026-03-07** in four parallel work streams:

### Stream A: Documentation Fixes (17 items)
Items #1, 3, 5-11, 13-18, 20-21: ADR text updated to match actual implementations.
Key rewrites: ADR-003 (regex+LLM architecture, not DistilBERT), ADR-008 (correct env vars, Protocol, Kuzu schema).

### Stream B: ADR-018 Protocol Conformance (item #12)
- `load_iicsa_reports()` now accepts `max_documents` as keyword-only parameter
- All 6 loaders made keyword-only (`*` separator) and wired through `limit_results()`
- Protocol conformance test added

### Stream C: Curiosity Scoring (item #4)
- `score_hypothesis_curiosity()` wired into `evolve_parallel()` alongside welfare and trajectory
- `curiosity_relevance` now populated when `phi_metrics` is provided
- WEIGHTS_BRIDGE formula fully operational: `0.45*confidence + 0.25*welfare + 0.15*curiosity + 0.15*trajectory`

### Stream D: Legal Domain Graph Integration (item #2)
- `KnowledgeEdge.legal_domain: LegalDomain | None` field added (backward compatible)
- `GraphStore.add_edge()` Protocol updated with optional `legal_domain` parameter
- Both `InMemoryGraph` and `KuzuGraph` store and retrieve legal domain
- `detect_legal_domain_gaps()` function created in `src/detective/legal_gap_detector.py`
- `_enrich_phase()` wired to extract legal_domain from document metadata
- 9 new tests for legal gap detection, 4 new tests for edge legal_domain

### Pre-existing (items #22-25)
Fixed in prior session before this audit was conducted.

**Final test count: 989 passed, 17 skipped, 0 failures.**
