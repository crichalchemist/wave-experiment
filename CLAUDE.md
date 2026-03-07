# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Detective LLM** — an information gap analysis system that detects what's *absent* in investigative datasets (geopolitical influence networks, temporal gaps, cognitive biases, implied entity connections). Named `detective-llm` in `pyproject.toml`.

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/
pytest tests/test_hypothesis.py          # single file
pytest tests/test_hypothesis.py::test_hypothesis_update  # single test

# Lint & format
ruff check .
ruff format .

# Type check
mypy src/

# Run reference GPT (downloads names.txt if absent, trains 1000 steps)
python microgpt.py

# CLI (after install)
detective analyze "Entity A was active throughout 2010-2017"
detective network --entity "Entity A" --hops 3
detective investigate --mode hypothesis --seed "Entity X had undisclosed ties" --max-steps 20

# Bootstrap directory structure (idempotent, for new checkouts)
python bootstrap.py

# Note: async tests require pytest-asyncio (included in dev deps)
```

## Architecture

### Layered design

```
microgpt.py          ← Reference GPT (pure Python, custom autograd, no deps)
                       Foundation for understanding the core algorithm
    ↓ extends
src/core/model.py    ← DetectiveGPT (PyTorch nn.Module)
                       Adds entity embeddings, special tokens, multi-task heads
    ↓ uses
src/detective/       ← A+B+C assumption detection + hypothesis evolution + autonomous investigation
src/training/        ← Multi-task loss, baseline + gap-detection training loops
src/inference/       ← n-hop network reasoning, contradiction detection
src/forecasting/     ← Phi trajectory forecasting (PhiTrajectoryForecaster)
src/data/            ← Epstein dataset loaders, entity extraction, knowledge graph
src/data/sourcing/   ← FOIA scraping + legal source pipelines
src/security/        ← Constitution client (epistemic moral compass)
src/api/             ← FastAPI endpoints + D3.js visualization export
src/cli/main.py      ← Click CLI (entry: `detective` script)
```

### Key design decisions

**`microgpt.py` is read-only reference** — the atomic GPT (Karpathy-style) with custom `Value` autograd, character-level BOS tokenizer, and Adam from scratch. Do not modify; `src/core/model.py` is where extension work happens.

**Immutable hypothesis lineage** — `Hypothesis` is `@dataclass(frozen=True)`. Every update uses `dataclasses.replace(...)` and assigns a new UUID, carrying `parent_id` from the previous version. This creates an auditable evolution tree. Never mutate a hypothesis; always spawn a new one.

**Multi-task loss formula** — `L_total = L_language + α·L_gap + β·L_assumption` (α=β=0.3). Three output heads: language modeling, gap-type classification, assumption-type classification.

**A+B+C assumption taxonomy** (library functions — not wired into the default `analyze()` pipeline, callable independently or via future pipeline integration):
- Module A (`src/detective/module_a.py`): Cognitive biases (confirmation, anchoring, survivorship, ingroup) — regex triggers + LLM-scored confirmation
- Module B (`src/detective/module_b.py`): Historical determinism — regex triggers + LLM-scored spans
- Module C (`src/detective/module_c.py`): Geopolitical presumptions — actor + verb pattern matching + LLM scoring
- All three use shared `src/core/scoring.py` for response parsing (`parse_score`, `clamp_confidence`)

**Special tokens** extend vocabulary: `<GAP>`, `<ASSUME_A>`, `<ASSUME_B>`, `<ASSUME_C>`, `<LINK>`, `<IMPLIED>`, `<CONTRADICTION>`, `<CONFIDENCE:>`.

**n-hop confidence decay** — relationships degrade as `0.9 × 0.7^(hops-1)` per hop.

**Graph of Thought (GoT) parallel hypothesis evolution** — `src/detective/parallel_evolution.py` implements Generate(k) via `asyncio.gather()`. When `phi_metrics` is provided, branches ranked by `WEIGHTS_BRIDGE` scheme: `combined_score = 0.45·confidence + 0.25·welfare_relevance + 0.15·curiosity_relevance + 0.15·trajectory_urgency` (ADR-010). Without `phi_metrics`, falls back to `WEIGHTS_DEFAULT` (confidence-only: 0.55/0.30/0.15/0.0). Both schemes are named constants in `src/detective/hypothesis.py`. SC-first branching: breadth below 0.5 confidence, depth above.

**Phi(humanity) welfare function (v2.1)** — `src/inference/welfare_scoring.py` scores hypotheses by which welfare constructs they threaten. Formula at v2.1 with recovery-aware floors and derivative tracking. Gradients of Φ prioritize leads threatening scarce constructs. Curiosity coupling (love × truth) surfaces investigative hunches. Formula documented in `docs/humanity-phi-formalized.md`.

**Phi Forecaster Space (ZeroGPU)** — `spaces/maninagarden/` is a modular Gradio workbench (7 tabs) for forecasting welfare trajectories. `welfare.py` is the evolved formula source (v2.1-recovery-floors). Launches GPU training via HF Jobs. Runs on ZeroGPU H200 with `@spaces.GPU` decorators for GPU-accelerated inference. Tab 7 (Entity Network, admin-gated) provides interactive knowledge graph exploration via API bridge to the detective. Graph topology features (7 features, 43-dim input) can be enabled for graph-enhanced training and inference. Live at `crichalchemist/maninagarden`. See ADR-017.

**Detective-Forecaster bridge** — three-layer pipeline connecting the detective and forecaster (ADRs 009-011):
- Layer 1: Welfare classifier (DistilBERT, 8-construct regression, MAE 0.164) trained on HF Jobs, loaded from Hub with local fallback (ADR-009)
- Layer 2: Trajectory urgency — forecaster predicts whether welfare is declining, urgency feeds into 4-weight `combined_score` (ADR-010)
- Layer 3: Scenario extraction — real text → construct profiles → trajectory patterns → synthetic training scenarios for forecaster (ADR-011)
- Data flywheel: detective findings enrich forecaster training data, forecaster predictions inform detective prioritization

**Hybrid provider routing (ADR-013)** — `HybridRoutingProvider` in `src/core/providers.py` inspects prompt text via `classify_prompt()` to route scoring calls (modules A/B/C, evolution, graph — "Reply with ONLY: score:") to a local vLLM CPU instance (DeepSeek-R1-Distill-Qwen-1.5B via Docker, chain-of-thought reasoning before scoring) and reasoning calls to Azure Foundry. Circuit-breaker with auto-recovery: if vLLM fails, all calls fall back to Azure; after configurable cooldown (default 60s), scoring is automatically retried. `reset_fallback()` also available for manual recovery. Set `DETECTIVE_PROVIDER=hybrid` to activate; `VLLM_SCORING_URL`/`VLLM_SCORING_MODEL` override defaults. Required env vars validated at startup via `_require_env()`. No call-site changes required.

**Reasoning trace capture (ADR-014)** — `ReasoningTrace` frozen dataclass captures chain-of-thought from scoring calls. `TraceStore` persists to JSONL, keeps a bounded deque (500), and pushes to SSE subscribers. Wired into `HybridRoutingProvider._trace_store` as a zero-call-site-change side-effect. API endpoints: `GET /traces/recent`, `/traces/history`, `/traces/stream` (SSE). Frontend at `crichalchemist.com/reasoning`. Set `DETECTIVE_TRACE_PATH` to enable.

**Epstein-docs ingestion (ADR-015)** — `src/data/epstein_adapter.py` parses 29K page JSONs and 8K analyses with entity deduplication via `dedupe.json`. `src/data/ingest_epstein.py` populates the GraphStore with CO_MENTIONED (confidence 0.5, decay 0.6) and ASSOCIATED (confidence 0.8, decay 0.8) edges. CLI: `detective ingest-epstein [--root] [--max-pages] [--drop-log PATH]` drives ingestion; `detective network --entity NAME [--hops N] [--format text|json]` queries the graph.

**Autonomous investigation agent (ADR-021)** — `src/detective/investigation/` implements a plan→gather→analyze→reflect→evolve→enrich loop that connects all detective-llm subsystems into a self-driving agent. Three trigger modes: `hypothesis` (seed a claim), `topic` (LLM generates initial hypotheses), `reactive` (graph event triggers investigation). `InvestigationSource` Protocol enables pluggable sources — 8 adapters: FOIA (3 portals), graph neighbourhood, web search, news search, CourtListener, SEC EDGAR, OCCRP, IICSA. `BudgetTracker` enforces max steps/pages/LLM calls/time. Security: every document sanitized before LLM, injection attempts become `Finding(is_injection_finding=True)`, reflect phase runs `critique_against_constitution()` with halt monitoring. CLI: `detective investigate --mode MODE --seed TEXT [budget options] -o report.json`. API: `POST /investigate`, `GET /investigation/{id}/status|report|stream`.

**Clearnet investigation sources (ADR-022)** — `src/detective/investigation/clearnet_sources.py` adds 6 clearnet source adapters: `WebSearchSource` (DuckDuckGo HTML), `NewsSearchSource` (DuckDuckGo news tab), `CourtListenerSource` (REST v4 API), `SECEdgarSource` (EFTS API), `OCCRPSource` (investigative journalism), `IICSASource` (UK gov reports, OGL v3). Shared helpers: `_RateLimiter` (thread-safe), `_to_evidence()` (sanitize + truncate + metadata), `_extract_text_from_page()` (strip scripts/nav). Scrapling for HTML sources, httpx for REST APIs. Lazy imports in `build_sources()`. Graceful degradation when Scrapling is not installed.

**Entity filter (ADR-016)** — `src/data/entity_filter.py` implements a 3-layer noise filter for ingestion: Layer 1 (junk — FOIA codes, emails, short strings, numeric refs, bracket-redacted), Layer 2 (fuzzy dedup — `difflib.SequenceMatcher` ≥0.75 pre-pass augmenting `people_map`), Layer 3 (role descriptions — possessives, inmate patterns, anonymized officers, role prefixes, lowercase informal). Drops ~22% of raw entities. `DropLog` writes filtered entities to JSONL for investigative audit. Wired into `ingest_epstein()` with `--drop-log` CLI option.

### Implementation status

**Working modules:** `src/detective/` (hypothesis, evolution, parallel_evolution, modules A/B/C, experience library), `src/detective/investigation/` (autonomous investigation agent — plan→gather→analyze→reflect→evolve→enrich loop, InvestigationSource Protocol + FOIA/Graph/clearnet adapters (8 sources — ADR-021/022), LLM-assisted lead generation, budget enforcement, constitutional halt), `src/core/scoring.py` (shared `parse_score`/`clamp_confidence` used by all detective modules), `src/core/log.py` (centralized logging config — ADR-019), `src/data/sourcing/types.py` (`SourceDocument` frozen dataclass + `DocumentLoader` Protocol — ADR-018), `src/inference/welfare_scoring.py` (Phi v2.1 formula, recovery floors, derivatives, gradient prioritization, curiosity scoring, trajectory urgency — all formula parameters are named constants), `src/inference/welfare_classifier.py` (Hub-first DistilBERT classifier — canonical source for construct scoring, `welfare_scoring.py` wraps with keyword fallback), `src/inference/scenario_extraction.py` (corpus → construct profiles → scenario templates), `src/forecasting/` (PhiTrajectoryForecaster, pipeline, scenarios), `src/data/sourcing/` (foia_scraper.py with guarded scraping imports, dual_pipeline.py, legal_sources.py, hf_loader.py, doj_loader.py, international_loader.py, ocr_provider.py — all loaders return `SourceDocument` per ADR-018), `src/data/epstein_adapter.py` (epstein-docs parser + entity normalization — ADR-015), `src/data/ingest_epstein.py` (graph population pipeline — ADR-015), `src/data/entity_filter.py` (3-layer entity noise filter + drop logging — ADR-016), `src/security/constitution.py` (epistemic moral compass client), `src/core/graph.py` (HybridGraphLayer + GATv2Conv), `src/core/providers.py` (Azure Foundry, Ollama, hybrid routing providers with circuit breaker auto-recovery — ADR-013), `src/core/reasoning_trace.py` + `src/core/trace_store.py` (reasoning trace capture with streaming pagination — ADR-014), `src/api/routes.py` (FastAPI endpoints + trace streaming + graph export + investigation endpoints + configurable CORS), `src/cli/main.py` (includes `extract-scenarios`, `ingest-epstein`, `network`, `investigate` commands), `src/training/constitutional_warmup.py` (constitutional preference pair generation — full implementation), `spaces/maninagarden/` (ZeroGPU-accelerated Gradio workbench: graph_client, graph_analytics, graph_features — ADR-017). Full test suite: 899 tests passing.

**Stubs:** Multi-task loss integration.

**Design docs:** `docs/plans/` contains implementation plans and design docs. `docs/humanity-phi-formalized.md` is the welfare function paper. `docs/constitution.md` is the epistemic moral compass.

**Python 3.13 compatible** — no removed stdlib modules, no deprecated patterns, all deps verified compatible. `requires-python = ">=3.12"`. See ADR-020.

**ADRs:** `docs/vault/decisions/` contains Architecture Decision Records (ADR-001 through ADR-022). Consult before making changes to the systems they cover.

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DETECTIVE_PROVIDER` | Yes | — | Provider selection: `vllm`, `azure`, `hybrid` |
| `VLLM_BASE_URL` | vllm/hybrid | — | vLLM inference endpoint |
| `VLLM_MODEL` | vllm | — | vLLM model name |
| `VLLM_SCORING_URL` | — | `http://localhost:8100/v1` | Hybrid scoring endpoint |
| `VLLM_SCORING_MODEL` | — | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Hybrid scoring model |
| `AZURE_ENDPOINT` | azure/hybrid | — | Azure Foundry endpoint |
| `AZURE_API_KEY` | azure/hybrid | — | Azure API key |
| `AZURE_MODEL` | azure/hybrid | — | Azure deployment name |
| `AZURE_CRITIC_ENDPOINT` | CAI warmup | — | Separate critic endpoint |
| `AZURE_CRITIC_KEY` | CAI warmup | — | Separate critic key |
| `DETECTIVE_TRACE_PATH` | — | disabled | JSONL path for reasoning traces |
| `DETECTIVE_GRAPH_BACKEND` | — | `memory` | Graph backend: `memory`, `kuzu` |
| `CORS_ORIGINS` | — | 4 default origins | Comma-separated CORS origins |
| `WELFARE_MODEL_PATH` | — | `models/welfare-constructs-distilbert` | Local welfare classifier path |
| `DETECTIVE_CONSTITUTION_PATH` | — | `docs/constitution.md` | Constitution file path |
| `SEC_EDGAR_USER_AGENT` | — | `detective-llm research@example.com` | SEC EDGAR API User-Agent header |

### Data layout (planned)

```
data/epstein/raw/          ← Raw clones of three Epstein GitHub repos (gitignored)
data/epstein/processed/    ← Parsed Document objects (gitignored)
data/annotations/          ← Gap/assumption annotation JSON
checkpoints/               ← Model .pt files (gitignored)
```

## Testing conventions

Tests import directly from `src.*` (no package install needed for pytest due to editable install). Test files live in `tests/` and mirror the `src/` module being tested (e.g., `tests/test_hypothesis.py` → `src/detective/hypothesis.py`).

Target: 80%+ coverage, gap detection F1 >0.75, relationship precision >0.80.

## Completion requirements

When finishing a feature or architectural change, the following are required before the work is considered done:

**Architecture Decision Record (ADR):** If the change introduces a new architectural pattern, modifies a Protocol/interface, adds a new subsystem, changes scoring formulas, or alters deployment strategy, write an ADR in `docs/vault/decisions/ADR-NNN-slug.md`. Follow the established format: YAML frontmatter (id, title, status, date, tags) + sections (Decision, Context, Consequences, Files). Number sequentially from the last ADR. Consult existing ADRs to avoid contradicting accepted decisions.

**Test verification:** Run `pytest tests/ -q` and confirm 0 new failures before claiming completion.

**CLAUDE.md update:** If the change affects architecture descriptions, test counts, or implementation status listed in this file, update the relevant sections.