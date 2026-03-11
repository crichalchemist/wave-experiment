---
id: ADR-027
title: Person auditor for claim decomposition and verification
status: accepted
date: 2026-03-07
tags: [auditor, claims, verification, investigation, person]
---

# ADR-027: Person Auditor for Claim Decomposition and Verification

## Decision

Add a standalone person auditor module (`src/detective/person_auditor.py`) that decomposes compound claims about a person into atomic claims, verifies each against available evidence, and computes an overall severity score. Wired as optional post-processing in the investigation agent and exposed via CLI.

## Context

The investigation agent produces `Finding` objects but has no mechanism to:
1. Audit a specific person across all findings
2. Detect cross-finding contradictions (e.g., "person was in Location A" vs "person was in Location B at the same time")
3. Assess claim severity by type (temporal, financial, association)

The Epstein-Pipeline project solved this with a 5-phase "Person Integrity Auditor" that decomposes claims, classifies them, and cross-references evidence. We adapt this pattern to detective-llm's existing architecture.

Three approaches were considered:

1. **Full LLM pipeline** — use LLM for decomposition, classification, and verification. Accurate but expensive.
2. **Hybrid LLM + heuristic** — LLM for decomposition and verification, heuristics for classification and severity scoring. Balanced cost/accuracy.
3. **Pure heuristic** — sentence splitting, keyword matching. Fast but misses nuance.

Approach 2 was chosen because:
- LLM decomposition captures compound claims that sentence splitting misses
- Heuristic classification (temporal keywords, financial terms) is reliable for well-defined categories
- Severity scoring is a formula, not a judgment call — no LLM needed
- Graceful fallback: when no LLM provider is available, decomposition falls back to sentence splitting

## Architecture

```
AtomicClaim(frozen)
    text, source_text, claim_type, person
    claim_types: temporal, association, financial, location, role, other

VerificationResult(frozen)
    claim, status, confidence, supporting_evidence, contradicting_evidence
    statuses: supported, contradicted, unverified, partially_supported

PersonAudit(frozen)
    person, claims, verifications, overall_confidence,
    contradiction_count, severity_score
    properties: supported_count, contradicted_count, unverified_count

decompose_claims(person, text, provider=None) -> tuple[AtomicClaim, ...]
    LLM → parse "claim:" lines; fallback → sentence splitting

verify_claim(claim, evidence_texts, provider=None) -> VerificationResult
    LLM → parse status/confidence; fallback → keyword overlap

compute_severity(verification_count, contradiction_count,
                 threatened_constructs=()) -> float
    40% contradiction ratio + 35% welfare construct coverage + 25% volume

audit_person(person, finding_texts, evidence_texts,
             provider=None, threatened_constructs=()) -> PersonAudit
    Full pipeline: decompose → verify each → aggregate → score severity
```

## Consequences

- **Person-level analysis** — aggregates findings about a specific individual
- **Contradiction detection** — identifies conflicting claims across sources
- **Severity scoring** — quantifies how serious the claims are based on contradictions, welfare impact, and volume
- **LLM-optional** — fallback heuristics work without a provider; audit runs during report building (post-loop), not as a budget-gated investigation step
- **Composable** — each function is independently testable and usable

## Files

- `src/detective/person_auditor.py` — new module
- `tests/detective/test_person_auditor.py` — new tests
- `src/detective/investigation/types.py` — add `PersonAuditSummary`
- `src/detective/investigation/agent.py` — optional `_audit_phase()`
- `src/cli/main.py` — add `audit-person` command
