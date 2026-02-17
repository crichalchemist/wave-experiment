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
- Module A (`src/detective/module_a.py`): Cognitive biases (confirmation, anchoring, survivorship, ingroup)
- Module B (planned `module_b.py`): Historical determinism — temporal language like "always", "continues to"
- Module C (planned `module_c.py`): Geopolitical presumptions — unstated actor interests

**Special tokens** extend vocabulary: `<GAP>`, `<ASSUME_A>`, `<ASSUME_B>`, `<ASSUME_C>`, `<LINK>`, `<IMPLIED>`, `<CONTRADICTION>`, `<CONFIDENCE:>`.

**n-hop confidence decay** — relationships degrade as `0.9 × 0.7^(hops-1)` per hop (defined in `IMPLEMENTATION_PLAN.md` Week 8).

### Implementation status

Most `src/` modules are stubs with `# TODO`. Only `src/detective/hypothesis.py` and the CLI/model skeletons have working code. The canonical reference for all planned implementations is `IMPLEMENTATION_PLAN.md` (16-week phased plan). The PRD in `DETECTIVE_LLM_PRD.md` is the source-of-truth for system design.

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