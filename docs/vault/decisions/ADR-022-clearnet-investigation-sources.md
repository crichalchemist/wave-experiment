---
id: ADR-022
title: Clearnet investigation sources
status: accepted
date: 2026-03-07
tags: [investigation, scraping, clearnet, sources]
---

# ADR-022: Clearnet Investigation Sources

## Decision

Add six clearnet source adapters to the autonomous investigation agent, expanding its reach beyond FOIA portals and the local knowledge graph to the open web: DuckDuckGo web search, DuckDuckGo news search, CourtListener court records, SEC EDGAR filings, OCCRP investigative journalism, and IICSA government reports.

## Context

ADR-021 introduced the autonomous investigation agent with two source adapters: `FOIAInvestigationSource` (FBI Vault, NARA, State Dept via Scrapling) and `GraphNeighbourhoodSource` (local knowledge graph queries). This limited investigations to declassified government documents and previously-ingested data.

Real investigations require access to:
- **Court records** — public filings, opinions, and dockets (CourtListener)
- **Corporate filings** — SEC 10-K, 10-Q, 8-K, proxy statements (EDGAR)
- **Investigative journalism** — cross-border corruption and trafficking investigations (OCCRP)
- **Government inquiries** — statutory inquiry reports (IICSA, Crown Copyright OGL v3)
- **General web/news** — breaking stories, blog posts, forums (DuckDuckGo)

All sources are public and open-access. No authentication required for any endpoint.

## Architecture

### Module structure

All six sources live in a single module `src/detective/investigation/clearnet_sources.py` with shared helpers:

- **`_RateLimiter`** — thread-safe fixed-interval delay (configurable per source)
- **`_to_evidence()`** — sanitize text via `sanitize_document()`, truncate to 10K chars, convert to `DocumentEvidence`, capture injection findings
- **`_extract_text_from_page()`** — strip `<script>`, `<style>`, `<nav>`, collapse whitespace
- **`_httpx_get()`** — thin monkeypatchable wrapper (matches `doj_loader.py` pattern)
- Guarded Scrapling import (matches `foia_scraper.py` pattern)

### Source details

| Source ID | Transport | Endpoint | Rate Limit |
|-----------|-----------|----------|------------|
| `web_search` | Scrapling | DuckDuckGo HTML | 2s |
| `news_search` | Scrapling | DuckDuckGo HTML (news tab) | 2s |
| `court_listener` | httpx | CourtListener REST v4 `/search/` | 1s |
| `sec_edgar` | httpx | SEC EFTS `/LATEST/search-index` | 0.5s |
| `web_occrp` | Scrapling | OCCRP `/en/search` | 2s |
| `web_iicsa` | Scrapling | IICSA `/reports-recommendations` | 2s |

### Integration

`build_sources()` in `source_protocol.py` gains 6 `elif` branches with lazy imports — clearnet code only loads when a clearnet source is requested. `InvestigationConfig.source_ids` default updated to include all 8 sources.

### Security

Every document passes through `sanitize_document()` via `_to_evidence()`. Injection attempts are captured in `SourceResult.injection_findings` and surfaced as findings during the analyze phase. No change to the existing security model.

## Consequences

- Investigation agent can now access the open web, court records, SEC filings, and investigative journalism
- All 8 default sources are attempted unless the caller restricts `source_ids`
- Scrapling-based sources gracefully degrade if Scrapling is not installed (return empty results with a warning)
- SEC EDGAR requires a `User-Agent` header; configurable via `SEC_EDGAR_USER_AGENT` env var
- Rate limiting prevents abuse of public APIs
- No new dependencies — Scrapling and httpx were already project dependencies

## Files

- `src/detective/investigation/clearnet_sources.py` (new)
- `src/detective/investigation/source_protocol.py` (modified — 6 elif branches)
- `src/detective/investigation/types.py` (modified — default source_ids)
- `src/detective/investigation/__init__.py` (modified — re-exports)
- `tests/detective/investigation/test_clearnet_sources.py` (new)
