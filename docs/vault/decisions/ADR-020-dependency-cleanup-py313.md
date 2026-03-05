---
id: ADR-020
title: Dependency cleanup and Python 3.13 readiness
status: accepted
date: 2026-03-05
tags: [dependencies, python-3.13, maintenance]
---

# ADR-020: Dependency Cleanup and Python 3.13 Readiness

## Decision

Remove 4 unused dependencies, explicitly declare 5 transitive dependencies, and modernize deprecated Python patterns to prepare for Python 3.13.

## Context

A dependency audit (2026-03-05) compared `pyproject.toml` declarations against actual imports in `src/` and found:

**4 packages declared but never imported:**
- `pdfplumber` — PDF handling uses `pdf2image` + `pytesseract` (in `[scraping]` extras), not pdfplumber
- `llama-index` — retrieval pipeline was never built; knowledge graph uses custom `GraphStore`
- `spacy` — entity extraction uses regex patterns + LLM scoring, not spaCy NER
- `azure-ai-inference` — Azure integration uses `urllib.request` (stdlib) via `AzureFoundryProvider`

**5 packages imported but only present as transitive dependencies:**
- `numpy` — used in 5 forecasting/inference modules (arrays, RNG, linear algebra)
- `pandas` — used in 4 forecasting modules (DataFrames for trajectory data)
- `scikit-learn` — used in `forecasting/pipeline.py` (`RobustScaler`)
- `tqdm` — used in `forecasting/engine.py` (progress bars)
- `datasets` — used in `hf_loader.py`, `legal_sources.py`, training scripts

**Python 3.13 compatibility assessment:**
- Zero removed stdlib modules in use (no `aifc`, `audioop`, `cgi`, `cgitb`, `chunk`, `crypt`, `imghdr`, `mailcap`, `msilib`, `nis`, `nntplib`, `ossaudiodev`, `pipes`, `sndhdr`, `spwd`, `sunau`, `telnetlib`, `uu`, `xdrlib`)
- All dependencies have Python 3.13–compatible releases
- Modern union syntax (`X | Y`) already adopted throughout
- Two deprecated patterns found and fixed:
  1. `asyncio.get_event_loop()` → `asyncio.to_thread()` (DeprecationWarning in 3.12+)
  2. `typing.List`/`typing.Dict` → `list`/`dict` builtins (redundant since 3.9, PEP 585)

## Consequences

- ~700MB reduction in install footprint (spaCy alone is ~500MB with models)
- Transitive-to-direct promotion prevents silent breakage if upstream drops sub-dependencies
- Zero `DeprecationWarning` emissions on Python 3.12/3.13
- `requires-python = ">=3.12"` remains the floor; 3.13 is fully compatible

## Files

- `pyproject.toml` (modified — 4 deps removed, 5 added)
- `src/detective/parallel_evolution.py` (modified — asyncio modernization)
- `src/forecasting/synthetic.py` (modified — `typing.List` → `list`)
- `src/forecasting/gap_prediction.py` (modified — `typing.Dict`, `typing.List` → `dict`, `list`)
- `src/forecasting/urgency_forecast.py` (modified — `typing.Dict` → `dict`)
