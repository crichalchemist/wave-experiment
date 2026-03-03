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
src/detective/       ← A+B+C assumption detection + hypothesis evolution
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

**A+B+C assumption taxonomy:**
- Module A (`src/detective/module_a.py`): Cognitive biases (confirmation, anchoring, survivorship, ingroup) — DistilBERT classifier
- Module B (`src/detective/module_b.py`): Historical determinism — regex triggers + LLM-scored spans
- Module C (`src/detective/module_c.py`): Geopolitical presumptions — actor + verb pattern matching + LLM scoring

**Special tokens** extend vocabulary: `<GAP>`, `<ASSUME_A>`, `<ASSUME_B>`, `<ASSUME_C>`, `<LINK>`, `<IMPLIED>`, `<CONTRADICTION>`, `<CONFIDENCE:>`.

**n-hop confidence decay** — relationships degrade as `0.9 × 0.7^(hops-1)` per hop.

**Graph of Thought (GoT) parallel hypothesis evolution** — `src/detective/parallel_evolution.py` implements Generate(k) via `asyncio.gather()`. When `phi_metrics` is provided, branches ranked by 4-weight `combined_score = 0.45·confidence + 0.25·welfare_relevance + 0.15·curiosity_relevance + 0.15·trajectory_urgency` (ADR-010). Without `phi_metrics`, falls back to confidence-only sorting. SC-first branching: breadth below 0.5 confidence, depth above.

**Phi(humanity) welfare function (v2.1)** — `src/inference/welfare_scoring.py` scores hypotheses by which welfare constructs they threaten. Formula at v2.1 with recovery-aware floors and derivative tracking. Gradients of Φ prioritize leads threatening scarce constructs. Curiosity coupling (love × truth) surfaces investigative hunches. Formula documented in `docs/humanity-phi-formalized.md`.

**Phi Forecaster Space** — `spaces/maninagarden/` is a modular Gradio workbench (6 tabs) for forecasting welfare trajectories. `welfare.py` is the evolved formula source (v2.1-recovery-floors). Launches GPU training via HF Jobs. Live at `crichalchemist/maninagarden`.

**Detective-Forecaster bridge** — three-layer pipeline connecting the detective and forecaster (ADRs 009-011):
- Layer 1: Welfare classifier (DistilBERT, 8-construct regression, MAE 0.164) trained on HF Jobs, loaded from Hub with local fallback (ADR-009)
- Layer 2: Trajectory urgency — forecaster predicts whether welfare is declining, urgency feeds into 4-weight `combined_score` (ADR-010)
- Layer 3: Scenario extraction — real text → construct profiles → trajectory patterns → synthetic training scenarios for forecaster (ADR-011)
- Data flywheel: detective findings enrich forecaster training data, forecaster predictions inform detective prioritization

**Hybrid provider routing (ADR-013)** — `HybridRoutingProvider` in `src/core/providers.py` inspects prompt text via `classify_prompt()` to route scoring calls (modules A/B/C, evolution, graph — "Reply with ONLY: score:") to a local vLLM CPU instance (DeepSeek-R1-Distill-Qwen-1.5B via Docker, chain-of-thought reasoning before scoring) and reasoning calls to Azure Foundry. Circuit-breaker: if vLLM fails, all calls fall back to Azure until `reset_fallback()`. Set `DETECTIVE_PROVIDER=hybrid` to activate; `VLLM_SCORING_URL`/`VLLM_SCORING_MODEL` override defaults. No call-site changes required.

**Reasoning trace capture (ADR-014)** — `ReasoningTrace` frozen dataclass captures chain-of-thought from scoring calls. `TraceStore` persists to JSONL, keeps a bounded deque (500), and pushes to SSE subscribers. Wired into `HybridRoutingProvider._trace_store` as a zero-call-site-change side-effect. API endpoints: `GET /traces/recent`, `/traces/history`, `/traces/stream` (SSE). Frontend at `crichalchemist.com/reasoning`. Set `DETECTIVE_TRACE_PATH` to enable.

**Epstein-docs ingestion (ADR-015)** — `src/data/epstein_adapter.py` parses 29K page JSONs and 8K analyses with entity deduplication via `dedupe.json`. `src/data/ingest_epstein.py` populates the GraphStore with CO_MENTIONED (confidence 0.5, decay 0.6) and ASSOCIATED (confidence 0.8, decay 0.8) edges. CLI: `detective ingest-epstein [--root] [--max-pages] [--drop-log PATH]` drives ingestion; `detective network --entity NAME [--hops N] [--format text|json]` queries the graph.

**Entity filter (ADR-016)** — `src/data/entity_filter.py` implements a 3-layer noise filter for ingestion: Layer 1 (junk — FOIA codes, emails, short strings, numeric refs, bracket-redacted), Layer 2 (fuzzy dedup — `difflib.SequenceMatcher` ≥0.75 pre-pass augmenting `people_map`), Layer 3 (role descriptions — possessives, inmate patterns, anonymized officers, role prefixes, lowercase informal). Drops ~22% of raw entities. `DropLog` writes filtered entities to JSONL for investigative audit. Wired into `ingest_epstein()` with `--drop-log` CLI option.

### Implementation status

**Working modules:** `src/detective/` (hypothesis, evolution, parallel_evolution, modules A/B/C, experience library), `src/inference/welfare_scoring.py` (Phi v2.1 formula, recovery floors, derivatives, gradient prioritization, curiosity scoring, trajectory urgency), `src/inference/welfare_classifier.py` (Hub-first DistilBERT classifier), `src/inference/scenario_extraction.py` (corpus → construct profiles → scenario templates), `src/forecasting/` (PhiTrajectoryForecaster, pipeline, scenarios), `src/data/sourcing/` (foia_scraper.py, dual_pipeline.py, legal_sources.py — FOIA + legal source ingestion), `src/data/epstein_adapter.py` (epstein-docs parser + entity normalization — ADR-015), `src/data/ingest_epstein.py` (graph population pipeline — ADR-015), `src/data/entity_filter.py` (3-layer entity noise filter + drop logging — ADR-016), `src/security/constitution.py` (epistemic moral compass client), `src/core/graph.py` (HybridGraphLayer + GATv2Conv), `src/core/providers.py` (Azure Foundry, Ollama, hybrid routing providers — ADR-013), `src/core/reasoning_trace.py` + `src/core/trace_store.py` (reasoning trace capture — ADR-014), `src/api/routes.py` (FastAPI endpoints + trace streaming), `src/cli/main.py` (includes `extract-scenarios`, `ingest-epstein`, `network` commands). Full test suite: 670+ tests passing.

**Stubs:** `src/training/constitutional_warmup.py`, multi-task loss integration.

**Design docs:** `docs/plans/` contains implementation plans and design docs. `docs/humanity-phi-formalized.md` is the welfare function paper. `docs/constitution.md` is the epistemic moral compass.

**ADRs:** `docs/vault/decisions/` contains Architecture Decision Records (ADR-001 through ADR-016). Consult before making changes to the systems they cover.

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