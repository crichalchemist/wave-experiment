---
id: ADR-021
title: Autonomous investigation agent
status: accepted
date: 2026-03-07
tags: [investigation, autonomy, agent-loop, security]
---

# ADR-021: Autonomous Investigation Agent

## Decision

Implement an autonomous investigation agent that connects all existing detective-llm subsystems (FOIA scraping, knowledge graph, 4-layer analysis pipeline, hypothesis evolution, welfare scoring, constitutional oversight) into a self-driving plan→gather→analyze→reflect→evolve→enrich loop.

## Context

The codebase had all the pieces for investigation but no autonomous loop connecting them. A human had to manually decide what to scrape, ingest, analyze, and evolve. Inspired by Robin (autonomous OSINT), OpenClaw (agent skill dispatch), and Scrapling (anti-bot web access), this feature creates an autonomous agent that runs until it exhausts leads or hits a budget/constitutional guardrail.

Key requirements:
- Three trigger modes: hypothesis (seed a specific claim), topic (LLM generates initial hypotheses), reactive (graph event triggers investigation)
- Budget enforcement: max steps, pages, LLM calls, wall-clock time
- Constitutional halt: investigation stops if analysis or reflection triggers halt signals
- Security-by-default: every document sanitized, injection attempts become findings
- Extensible source protocol: new sources (SEC EDGAR, CourtListener, .onion) implement `InvestigationSource`

## Architecture

### Module structure

```
src/detective/investigation/
    __init__.py              — re-exports InvestigationAgent, key types
    types.py                 — 10 frozen dataclasses (config, budget, lead, step, finding, report)
    source_protocol.py       — InvestigationSource Protocol + FOIA/Graph adapters
    planner.py               — LLM-assisted lead generation + hypothesis seeding
    agent.py                 — InvestigationAgent: core async loop + BudgetTracker
```

### Loop phases (each iteration)

1. **PLAN** — `planner.generate_leads()` from current hypotheses + graph context
2. **GATHER** — Execute top 5 leads via source adapters. Sanitize all documents. Injection → Finding.
3. **ANALYZE** — 4-layer pipeline (parse_intent → retrieve_evidence → fuse_reasoning → verify_inline)
4. **REFLECT** — Constitutional critique via `critique_against_constitution()`. Halt if triggered.
5. **EVOLVE** — `evolve_parallel()` with gathered evidence. Prune dead hypotheses (< 0.05). Extract findings (> 0.7). Spawn alternatives via `branching_rule()` for breadth.
6. **ENRICH** — Extract entities, filter through `entity_filter`, add CO_MENTIONED edges to graph.

### Security integration (4 points)

| Phase | Component | Integration |
|-------|-----------|-------------|
| GATHER | `sanitize_document()` | Every doc sanitized before LLM sees it |
| GATHER | Injection-as-finding | Injection attempts recorded as `Finding(is_injection_finding=True)` |
| ANALYZE | `build_analysis_prompt()` | Constitution-first framing, `<document>` isolation tags |
| REFLECT | `critique_against_constitution()` | Constitutional reflection + halt signal monitoring |

### Reuse of existing components

All 15 existing subsystems are reused without modification — the investigation module only adds new code:
- Hypothesis lifecycle (create, update_confidence, combined_score, evolve_parallel, branching_rule)
- Analysis pipeline (4-layer: parse_intent, retrieve_evidence, fuse_reasoning, verify_inline)
- Graph store (GraphStore Protocol, add_edge, successors, nodes)
- Providers (ModelProvider Protocol, provider_from_env)
- Security (sanitize_document, build_analysis_prompt, constitution)
- Welfare scoring (infer_threatened_constructs, score_hypothesis_welfare)
- Entity filtering (filter_entities)
- Experience library (add_experience, query_similar)
- Trace store (TraceStore for step tracing)

### Extensibility

New investigation sources implement the structural Protocol:
```python
class InvestigationSource(Protocol):
    @property
    def source_id(self) -> str: ...
    def search(self, query: str, max_pages: int = 10) -> SourceResult: ...
```

## Consequences

- **Positive**: Full automation of the investigation cycle; budget safety prevents runaway costs; constitutional oversight prevents epistemic overreach; injection-as-finding turns attacks into intelligence; Protocol-based sources are trivially extensible
- **Positive**: 74 new tests, full suite at 837 passing
- **Negative**: Agent loop has many moving parts — debugging requires understanding all 6 phases
- **Risk**: LLM-generated leads may produce low-quality queries — mitigated by fallback lead generation

## Files

- `src/detective/investigation/__init__.py` — re-exports
- `src/detective/investigation/types.py` — frozen data model (10 types)
- `src/detective/investigation/source_protocol.py` — Protocol + 2 adapters + builder
- `src/detective/investigation/planner.py` — 4 LLM-assisted planning functions
- `src/detective/investigation/agent.py` — InvestigationAgent + BudgetTracker
- `src/cli/main.py` — `detective investigate` command
- `src/api/routes.py` — POST /investigate, GET /investigation/{id}/status|report|stream
- `tests/detective/investigation/` — 74 tests across 4 files
