# Codebase Audit — 2026-03-05

## Summary

Full audit of detective-llm codebase. 22 issues found: 3 critical, 7 high, 6 medium, 6 low.

**Remediation status: 22/22 fixed. Test suite: 971 passed, 0 failed.**

---

## CRITICAL

### BUG-001: Azure API keys exposed in tracked files
- **Location:** `.env.local:9`, `deployment/.env.service:9,27`
- **Impact:** Real Azure API keys in plaintext, `.env.local` has `M` (modified) git status
- **Fix:** Untrack `.env.local`, add `deployment/.env.service` to `.gitignore`, rotate keys in Azure portal
- **Status:** FIXED

### BUG-002: Modules A/B/C are dead code
- **Location:** `src/detective/module_a.py:66`, `module_b.py:80`, `module_c.py:99`
- **Impact:** `detect_cognitive_biases()`, `detect_historical_determinism()`, `detect_geopolitical_presumptions()` never called from pipeline or any entry point. CLAUDE.md lists as "Working" -- misleading
- **Fix:** Document accurately in CLAUDE.md (library functions, not pipeline-integrated). Keep for future wiring.
- **Status:** FIXED
- **Update (ADR-023):** A/B/C modules are now wired into the investigation agent's `_assumption_scan()` in the analyze phase. They are called on each unique document (max 5) during investigation runs, with counter-lead generation for detected assumptions. No longer dead code.

### BUG-003: Thread-unsafe global singleton
- **Location:** `src/inference/welfare_scoring.py:540-549`
- **Impact:** `_forecaster_cache` uses check-then-set without locking. Race condition under concurrent `asyncio.run_in_executor()` calls
- **Fix:** Double-checked locking with `threading.Lock`
- **Status:** FIXED

---

## HIGH

### BUG-004: Score parsing triplication
- **Location:** `module_a.py:59-63`, `module_b.py:69-77`, `module_c.py:88-96`
- **Impact:** Three independent `_parse_score()` implementations with divergent regex (module A accepts `confidence` + `=`; B/C only `score:`)
- **Fix:** Extract to `src/core/scoring.py` with unified `parse_score()` and `clamp_confidence()`
- **Status:** FIXED

### BUG-005: Dual graph abstraction
- **Location:** `src/data/knowledge_graph.py` vs `src/data/graph_store.py`
- **Impact:** Functional `make_graph()`, `add_edge()` in knowledge_graph.py are dead code. `InMemoryGraph` leaks abstraction by delegating to `n_hop_paths()`
- **Fix:** Remove dead functions (`make_graph`, `add_edge`), keep shared `n_hop_paths`, `PathResult`, `_EDGE_DATA_KEY`, `_HOP_DECAY`
- **Status:** FIXED

### BUG-006: Welfare construct scoring duplication
- **Location:** `welfare_classifier.py:64-127` vs `welfare_scoring.py:310-369`
- **Impact:** Both define `get_construct_scores()` and `infer_threatened_constructs()`. Callers must know which module to import
- **Fix:** Make `welfare_classifier.py` canonical. `welfare_scoring.py` keeps thin wrappers with keyword fallback + specific exception handling
- **Status:** FIXED

### BUG-007: Inconsistent env var loading
- **Location:** `src/core/providers.py:225-260`
- **Impact:** Three patterns: `.get()` + None check, `[]` (KeyError), `.get()` + default. Misconfiguring `VLLM_SCORING_URL` silently wrong-defaults; `AZURE_ENDPOINT` crashes
- **Fix:** Extract `_require_env()` helper, use consistently for all required vars
- **Status:** FIXED

### BUG-008: N+1 graph query in CLI
- **Location:** `src/cli/main.py:90-100`
- **Impact:** Nested loop: for each successor, for each next_hop, call `n_hop_paths()`. 100 successors x 10 next_hops = 1,000 sequential queries
- **Fix:** Prefetch all 2-hop targets into a set, then query paths once per unique target
- **Status:** FIXED

### BUG-009: No circuit breaker recovery
- **Location:** `src/core/providers.py:174-188`
- **Impact:** `HybridRoutingProvider` permanently falls back to Azure after one vLLM failure. No auto-recovery, no health check. Requires manual `reset_fallback()`
- **Fix:** Add `_circuit_opened_at` timestamp + configurable cooldown. Auto-retry after cooldown
- **Status:** FIXED

### BUG-010: Overly broad exception handling
- **Location:** `welfare_scoring.py:329,365,627`
- **Impact:** Bare `except Exception` swallows unexpected errors silently
- **Fix:** Replace with `except (FileNotFoundError, ValueError, OSError)` and `except (ImportError, ValueError, RuntimeError, OSError)`
- **Status:** FIXED
- **Correction:** The original description claimed `except Exception` catches `KeyboardInterrupt` in Python <3.11. This is incorrect -- `KeyboardInterrupt` inherits from `BaseException`, not `Exception`, in all Python versions. `except Exception` has never caught `KeyboardInterrupt`. The narrowing fix was still valid practice for avoiding accidental suppression of unexpected `Exception` subclasses.

---

## MEDIUM

### BUG-011: CLAUDE.md vs reality mismatches
- **Impact:** "Working modules: A/B/C" (dead code), "Stubs: constitutional_warmup" (307-line implementation), test count may be stale
- **Fix:** Update all sections to reflect reality
- **Status:** FIXED

### BUG-012: Conflicting scoring weights
- **Location:** `hypothesis.py:69` defaults `alpha=0.55` vs `parallel_evolution.py:169` bridge `alpha=0.45`
- **Impact:** Two weight schemes with no named constants, no documentation on when to use which
- **Fix:** Add `WEIGHTS_DEFAULT` and `WEIGHTS_BRIDGE` named dicts in hypothesis.py
- **Status:** FIXED

### BUG-013: Magic numbers in welfare formula
- **Location:** `welfare_scoring.py:139-141,158,447,574,641`
- **Impact:** Sigmoid parameters (10.0, -3.0), guard values (0.01), normalization offsets (k=1.0), noise scale (0.001) unnamed
- **Fix:** Extract all to named module-level constants with documentation
- **Status:** FIXED

### BUG-014: Test suite weaknesses
- **Impact:** 5 untested modules, weak `isinstance(result, list)` assertions, 6 duplicate frozen-dataclass tests, missing edge cases, 267 MagicMock usages
- **Fix:** Created `assert_frozen` conftest fixture (simplifies 7 frozen tests), wrote 24 new tests across 4 files (doj_loader, international_loader, ocr_provider, construct_forecast), strengthened 3 weak assertions
- **Status:** FIXED

### BUG-015: Loose dependency pinning
- **Location:** `pyproject.toml:6-30`
- **Impact:** `torch>=2.0.0`, `fastapi>=0.100.0`, `azure-ai-inference` (no version). Can break on major bumps
- **Fix:** Add upper-bound constraints (`<3.0`, `<1.0`, etc.)
- **Status:** FIXED

### BUG-016: Missing scraping dependency guards
- **Location:** `src/data/sourcing/foia_scraper.py:1-10`
- **Impact:** Unconditional import of `scrapling`, `pdf2image`, `pytesseract` — `ImportError` without `[scraping]` extra
- **Fix:** Wrap in try/except with helpful error message
- **Status:** FIXED

---

## LOW

### BUG-017: Hardcoded CORS origins
- **Location:** `src/api/routes.py:143-147`
- **Fix:** Read from `CORS_ORIGINS` env var with current values as default
- **Status:** FIXED

### BUG-018: Hardcoded paths
- **Location:** `welfare_classifier.py:21`, `constitution.py:8`
- **Fix:** Make configurable via `WELFARE_MODEL_PATH` and `DETECTIVE_CONSTITUTION_PATH` env vars
- **Status:** FIXED

### BUG-019: Silent provider fallback in API
- **Location:** `src/api/routes.py:125-131`
- **Impact:** `provider_from_env()` failure silently falls back to `MockProvider` — production could serve fake analysis
- **Fix:** Add `logging.warning()` before fallback
- **Status:** FIXED

### BUG-020: Inconsistent logging
- **Impact:** Some modules use `logger`, others `_logger`, some have none. No structured logging, no correlation IDs
- **Fix:** Standardized to `_logger = logging.getLogger(__name__)` across all modules, centralized config in `src/core/log.py`, replaced `except: pass` with logging, replaced `print(stderr)` with logging (ADR-019)
- **Status:** FIXED

### BUG-021: Data loader pattern fragmentation
- **Impact:** 4 loader files each independently implementing pagination, filtering, normalization. No shared protocol
- **Fix:** Created `SourceDocument` frozen dataclass + `DocumentLoader` Protocol in `src/data/sourcing/types.py`, unified `max_documents` parameter, all loaders return `list[SourceDocument]` (ADR-018)
- **Status:** FIXED

### BUG-022: Unbounded JSONL in trace store
- **Location:** `src/core/trace_store.py:79-96`
- **Impact:** `historical()` reads entire file into memory for pagination
- **Fix:** Stream with seek-based pagination
- **Status:** FIXED

---

## Deferred Items

All 22 bugs have been remediated. No items remain deferred.

## Remediation Plan

See `docs/plans/2026-03-05-full-audit-remediation.md` for full implementation plan.
