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

The knowledge graph schema includes `LegalContext` entity type with a mandatory `legal_domain` field:

```python
class LegalDomain(Enum):
    STATUTE = "statute"                  # Black-letter law, codified text
    REGULATION = "regulation"            # Federal/state agency rules
    CASE_LAW = "case_law"                # Court holdings and precedent
    ENFORCEMENT_PRACTICE = "enforcement_practice"  # What actually happens
    COMMUNITY_EXPERIENCE = "community_experience"  # Documented lived reality
    TREATY = "treaty"                    # Federal Indian law, territorial agreements
    TERRITORIAL = "territorial"          # Law in US territories (often different from states)
```

A gap between `STATUTE` and `ENFORCEMENT_PRACTICE` nodes for the same legal topic is a `GapType.DOCTRINAL` finding — the law assumes its own enforcement, which may never have occurred for this population in this jurisdiction.

A gap between `STATUTE` and `COMMUNITY_EXPERIENCE` is a `GapType.NORMATIVE` finding — the law's stated obligations are not being met.

## Geographic and jurisdictional scope

The system must recognize:
- **Federal vs. state law conflicts** (immigration, civil rights, tribal land)
- **US territories**: Puerto Rico, Guam, U.S. Virgin Islands, American Samoa, CNMI — different constitutional protections, different federal program access, non-voting representation
- **Tribal nations**: federal Indian law, treaty rights, sovereignty (distinct from state and federal courts)
- **Local enforcement variation**: same statute, vastly different application by city, county, or region

## Standpoint grounding (per ADR-002)

When analyzing legal claims, the system must ask: *from whose position is this analyzed?* A compliance record that reads as "no violations found" from a regulatory agency's perspective may read as "systematic non-enforcement" from the perspective of communities whose complaints were not investigated. Both readings of the same record are epistemically valid; only one is typically in the official record.

## Knowledge graph implications

- `LegalDomain` is added to `src/core/types.py` as an enum
- Edges between legal entities carry `LegalDomain` on both source and target
- A query for gaps between STATUTE and ENFORCEMENT_PRACTICE nodes triggers a normative gap check
- Community experience nodes (`COMMUNITY_EXPERIENCE`) are first-class entities, not annotations on other entities

## Files

- `src/core/types.py` — add `LegalDomain` enum
- `docs/constitution.md` — legal grounding section (lex_glue scotus dataset used for warmup; 232 DPO pairs generated)
- `docs/vault/decisions/ADR-007-legal-grounding.md` — this document
