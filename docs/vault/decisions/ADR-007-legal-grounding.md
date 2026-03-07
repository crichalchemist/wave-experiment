---
id: ADR-007
title: American law as written vs. law as applied to marginalized people
status: accepted
date: 2026-02-17
tags: [epistemology, legal, jurisdiction, marginalized-communities, gap-detection]
---

# ADR-007: American Law as Written vs. Law as Applied

## Decision

The system explicitly distinguishes between (1) American law as codified in statutes, regulations, and court holdings, and (2) American law as experienced by marginalized people in practice. These are treated as two separate epistemic domains. The gap between them is a primary detection target, not a footnote.

## The epistemic distinction

**Law as written:** Statutes, federal and state regulations, constitutional text, court holdings, official enforcement guidance. These are formal legal instruments.

**Law as applied:** Police practices, prosecutorial discretion, sentencing disparities, civil forfeiture patterns, immigration enforcement targeting, housing discrimination enforcement rates, environmental permitting in low-income communities, access to public defenders, bail and pretrial detention practices, tribal sovereignty and treaty compliance, territorial rights (Puerto Rico, Guam, USVI, American Samoa, CNMI). These are lived outcomes.

The gap between them is not incidental — it is systemic and documented. Ignoring it produces complicit analysis (ADR-002: "analysis that accepts the powerful's account of what exists and does not exist is not neutral — it is complicit").

## Implementation in gap detection

`LegalDomain` is an enum in `src/core/types.py` with 7 values:

```python
class LegalDomain(Enum):
    STATUTE = "statute"                          # Black-letter law, codified text
    REGULATION = "regulation"                    # Federal/state agency rules
    CASE_LAW = "case_law"                        # Court holdings and precedent
    ENFORCEMENT_PRACTICE = "enforcement_practice"  # What regulators/police actually do
    COMMUNITY_EXPERIENCE = "community_experience"  # Documented lived reality of those affected
    TREATY = "treaty"                            # Federal Indian law, territorial agreements
    TERRITORIAL = "territorial"                  # Law as it applies in US territories
```

Legal domain is an **edge-level** property, not a node-level one. Entities don't inherently belong to a legal domain — they *appear in* legal domain contexts. `KnowledgeEdge` carries an optional `legal_domain: LegalDomain | None` field (defaults to `None` for backward compatibility).

The `GraphStore` Protocol's `add_edge()` method accepts `legal_domain` as an optional parameter. Both `InMemoryGraph` and `KuzuGraph` store and retrieve it.

**Gap detection** is implemented in `src/detective/legal_gap_detector.py`:
- `detect_legal_domain_gaps(graph, entity)` collects edges touching an entity, partitions them into "written" domains (`STATUTE`, `REGULATION`, `CASE_LAW`, `TREATY`, `TERRITORIAL`) and "applied" domains (`ENFORCEMENT_PRACTICE`, `COMMUNITY_EXPERIENCE`).
- Written edges without applied → `GapType.DOCTRINAL` finding: the law assumes its own enforcement, which may never have occurred for this population in this jurisdiction.
- Applied edges without written → `GapType.NORMATIVE` finding: enforcement without documented legal basis is investigatively significant.

## Geographic and jurisdictional scope

The system must recognize:
- **Federal vs. state law conflicts** (immigration, civil rights, tribal land)
- **US territories**: Puerto Rico, Guam, U.S. Virgin Islands, American Samoa, CNMI — different constitutional protections, different federal program access, non-voting representation
- **Tribal nations**: federal Indian law, treaty rights, sovereignty (distinct from state and federal courts)
- **Local enforcement variation**: same statute, vastly different application by city, county, or region

## Standpoint grounding (per ADR-002)

When analyzing legal claims, the system must ask: *from whose position is this analyzed?* A compliance record that reads as "no violations found" from a regulatory agency's perspective may read as "systematic non-enforcement" from the perspective of communities whose complaints were not investigated. Both readings of the same record are epistemically valid; only one is typically in the official record.

## Knowledge graph implications

- `LegalDomain` is defined in `src/core/types.py` as an enum with 7 values
- `KnowledgeEdge.legal_domain` is an optional field (`LegalDomain | None`, default `None`)
- `GraphStore.add_edge()` accepts `legal_domain` as an optional keyword argument
- Both `InMemoryGraph` and `KuzuGraph` (persistent, embedded) store and retrieve legal domain
- `KuzuGraph` stores legal domain as a `STRING` column (`DEFAULT ''`) — empty string maps to `None`
- The investigation agent's `_enrich_phase()` extracts `legal_domain` from `DocumentEvidence.metadata` and passes it to `graph.add_edge()`
- `detect_legal_domain_gaps()` is a query function that finds entities with asymmetric legal domain coverage

## Files

- `src/core/types.py` — `LegalDomain` enum, `KnowledgeEdge.legal_domain` field
- `src/data/graph_store.py` — `GraphStore` Protocol and `InMemoryGraph` with `legal_domain` support
- `src/data/kuzu_graph.py` — `KuzuGraph` with `legal_domain` column in Relationship table
- `src/detective/legal_gap_detector.py` — `LegalGap` frozen dataclass, `detect_legal_domain_gaps()` function
- `src/detective/investigation/agent.py` — `_enrich_phase()` wires legal domain from document metadata
- `docs/constitution.md` — legal grounding section (lex_glue scotus dataset used for warmup; 232 DPO pairs generated)
- `docs/vault/decisions/ADR-007-legal-grounding.md` — this document
