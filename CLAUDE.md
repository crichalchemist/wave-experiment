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
src/data/            ← Epstein dataset loaders, entity extraction, knowledge graph
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

**Graph of Thought (GoT) parallel hypothesis evolution** — `src/detective/parallel_evolution.py` implements Generate(k) via `asyncio.gather()`. Branches ranked by `combined_score = 0.55·confidence + 0.30·welfare_relevance + 0.15·curiosity_relevance`. SC-first branching: breadth below 0.5 confidence, depth above.

**Phi(humanity) welfare function** — `src/inference/welfare_scoring.py` scores hypotheses by which welfare constructs they threaten. Gradients of Φ prioritize leads threatening scarce constructs. Curiosity coupling (love × truth) surfaces investigative hunches. Formula documented in `docs/humanity-phi-formalized.md`.

**Phi Forecaster Space** — `spaces/maninagarden/` is a modular Gradio workbench (6 tabs) for forecasting welfare trajectories. `welfare.py` is the evolved formula source (v2.1-recovery-floors). Launches GPU training via HF Jobs. Live at `crichalchemist/maninagarden`.

### Implementation status

**Working modules:** `src/detective/` (hypothesis, evolution, parallel_evolution, modules A/B/C, experience library), `src/inference/welfare_scoring.py` (Phi formula, gradient prioritization, curiosity scoring), `src/core/graph.py` (HybridGraphLayer + GATv2Conv), `src/core/providers.py` (Azure Foundry provider), `src/api/routes.py` (FastAPI endpoints), `src/cli/main.py`. Full test suite: 291+ tests passing.

**Stubs:** `src/training/constitutional_warmup.py`, multi-task loss integration, document ingestion pipeline.

**Design docs:** `docs/plans/` contains implementation plans and design docs. `docs/humanity-phi-formalized.md` is the welfare function paper. `docs/constitution.md` is the epistemic moral compass.

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