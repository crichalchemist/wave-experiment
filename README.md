# Detective LLM

**Detective LLM** is an information gap analysis system that detects what is *absent* from investigative datasets — temporal silences, evidential voids, implied entity connections, and cognitive biases — rather than what is present. It was built for geopolitical influence networks but the analytical framework generalises to any domain where the most significant findings are the things that were never documented.

> *"You cannot see what you cannot see. The most significant gaps may be invisible precisely because they were designed to be."*
> — from `docs/constitution.md`

---

## What it does

Most language models complete patterns. Detective LLM is trained to break that tendency. Given an investigative corpus, it asks:

- What time periods are missing from this record?
- What claims lack documentation that should exist?
- Where does the evidence contradict itself, and what does that reveal?
- Which institutional obligations are unmet — not because rules were broken, but because compliance was never expected of this actor toward this population?
- What assumptions is the analysis making without acknowledgement?

The system surfaces five gap types and three assumption types, runs them through a constitutional self-critique loop (Constitutional AI), and stores every finding in an auditable, immutable hypothesis lineage.

---

## Architecture

```
microgpt.py                     Reference GPT — pure Python, custom autograd, no deps
                                 (read-only; the complete algorithm in ~200 lines)
    ↓ extends
src/core/model.py               DetectiveGPT (PyTorch nn.Module)
                                 Multi-task: language model + gap detection heads
    ↓ uses
src/detective/                  Gap analysis core
  hypothesis.py                   Immutable Hypothesis with auditable lineage + welfare scoring
  evolution.py                    Hypothesis evolution via experience library (FLEX)
  parallel_evolution.py           Graph of Thought parallel hypothesis branching
  experience.py                   ExperienceLibrary — immutable tuple of past trajectories
  module_a.py                     Module A: cognitive bias detection (regex + LLM scoring)
  module_b.py                     Module B: historical determinism detection
  module_c.py                     Module C: geopolitical presumption detection
  constitution.py                 Constitutional AI self-critique loop
  person_auditor.py               Claim decomposition, verification, severity scoring
  legal_gap_detector.py           Written-vs-applied law gap detection
  investigation/                  Autonomous investigation agent
    agent.py                        plan→gather→analyze→reflect→evolve→enrich loop
    clearnet_sources.py             6 clearnet adapters (DDG, CourtListener, SEC, OCCRP, IICSA)
    source_protocol.py              InvestigationSource Protocol + source factory

src/inference/                  Analytical pipeline
  pipeline.py                     4-layer pipeline + inline gap detection + token budgeting
  welfare_scoring.py              Φ(humanity) welfare function (v2.1, 8 constructs)
  welfare_classifier.py           DistilBERT construct classifier (Hub-first loading)
  scenario_extraction.py          Corpus → construct profiles → training scenarios
  reflection.py                   Constitutional reflection trigger injection

src/forecasting/                Phi trajectory forecasting
  model.py                        PhiForecaster (ForecastBackbone: CNN1D→LSTM→Attention)
  pipeline.py                     PhiPipeline — feature engineering + inference
  signals.py                      PhiSignalProcessor — signal preprocessing

src/data/                       Data & knowledge graph
  graph_store.py                  GraphStore Protocol (InMemoryGraph / KuzuGraph)
  kuzu_graph.py                   Persistent Kuzu backend (production)
  epstein_adapter.py              Epstein-docs parser + entity normalisation
  ingest_epstein.py               Graph population pipeline
  entity_filter.py                3-layer entity noise filter + drop logging
  ner.py                          spaCy NER + heuristic fallback
  dedup.py                        MinHash/LSH near-duplicate detection
  sourcing/                       FOIA scraping, legal sources, HF/DOJ loaders, OCR

src/security/                   Injection defence
  sanitizer.py                    Layer 1: pattern-based injection detection at ingestion
  prompt_guard.py                 Layer 2: constitutional prompt isolation at inference

src/training/                   Alignment training
  constitutional_warmup.py        Constitutional preference pair generation (CAI loop)
  train_multitask.py              Multi-task PyTorch trainer (3-head loss)
  train_dpo.py                    DPO preference learning from critique pairs

src/core/                       Shared infrastructure
  providers.py                    VLLMProvider, AzureFoundryProvider, HybridRoutingProvider
  reasoning_trace.py              Chain-of-thought capture from scoring calls
  trace_store.py                  JSONL persistence + SSE streaming for traces

src/api/                        REST API
  routes.py                       FastAPI endpoints + trace streaming + investigation API

src/cli/                        Command-line interface
  main.py                         Click CLI — `detective` entry point

spaces/maninagarden/            Phi Forecaster Gradio workbench (ZeroGPU, 7 tabs)
```

### Multi-task model

```
token_emb + pos_emb
        ↓
  TransformerEncoder backbone
        ↓                    ↓
   lm_head               temporal_emb
 (language model)              ↓
                       temporal_encoder
                               ↓
                           gap_head
                     (gap type classification)
```

Multi-task loss: `L_total = L_language + α·L_gap + β·L_assumption` (α=β=0.3)

### Gap type taxonomy

| Gap type | Description |
|---|---|
| `TEMPORAL` | Missing time periods, unexplained silences in the record |
| `EVIDENTIAL` | Claims that lack documentation which should exist |
| `CONTRADICTION` | Conflicting accounts that cannot both be accurate |
| `NORMATIVE` | What should be documented given stated obligations but is not |
| `DOCTRINAL` | Unstated institutional rules assumed to apply that may not have |

### Assumption taxonomy (Modules A / B / C)

| Module | Assumption type | Description |
|---|---|---|
| A | `COGNITIVE_BIAS` | Systematic reasoning errors — confirmation, anchoring, survivorship |
| B | `HISTORICAL_DETERMINISM` | Assuming documents record events neutrally and in order |
| C | `GEOPOLITICAL_PRESUMPTION` | Assuming institutions behaved as their stated norms describe |

All three modules are implemented: Module A uses regex triggers + LLM-scored confirmation for cognitive biases, Module B detects historical determinism via regex + LLM spans, Module C matches geopolitical actor + verb patterns with LLM scoring. All share `src/core/scoring.py` for response parsing.

### Knowledge graph

Edges carry typed semantic relations with per-hop confidence decay:

| Relation | Decay per hop | Rationale |
|---|---|---|
| `CAUSAL` | 0.70 | Causation chains degrade fast — correlation ≠ causation |
| `CONDITIONAL` | 0.75 | Depends on an unstated premise being true |
| `INSTANTIATIVE` | 0.85 | Instance-of is more stable than causal attribution |
| `SEQUENTIAL` | 0.90 | Temporal precedence is more recoverable than causation |

### Dual LLM roles

| Role | Model | Purpose |
|---|---|---|
| Generator | Local vLLM (DeepSeek-R1-Distill-Qwen-7B) | Gap analysis, hypothesis evolution |
| Critic | Azure AI Foundry (Claude) | CAI self-critique, preference pair generation |

Using a different model as critic produces stronger training signal than same-model self-critique.

### 4-layer analytical pipeline

1. **`parse_intent`** — keyword extraction, no LLM call, selects gap hint
2. **`retrieve_evidence`** — graph neighbourhood query (direct + 1-hop expansion)
3. **`fuse_reasoning`** — single LLM call: evidence → numbered reasoning chain
4. **`verify_inline`** — constitutional reflection trigger + confidence scoring

Evidence is truncated to fit within the vLLM context window (token budget: 6000 tokens, reserving 2000 for verification). After verification, if confidence exceeds 0.3, an inline gap scan detects up to 3 information gaps with welfare scoring when Φ metrics are available.

### Immutable hypothesis lineage

`Hypothesis` is a `frozen=True` dataclass. Every update spawns a new instance with a new UUID and a `parent_id` pointing back to its predecessor. Nothing is ever overwritten. The full evolution tree is auditable.

```python
h1 = Hypothesis.create("Entity A controlled the network", confidence=0.6)
h2 = h1.update_confidence(0.8)   # new UUID, parent_id = h1.id
```

### Security model

Two-layer prompt injection defence:

1. **Sanitizer** (`src/security/sanitizer.py`) — detects instruction overrides, role switches, fake conversation turns, and Unicode control characters at ingestion time
2. **Prompt guard** (`src/security/prompt_guard.py`) — wraps document content in `<document>…</document>` delimiters with explicit untrusted-data framing; constitution always precedes document at the system level

Detected injection attempts are recorded as `GapType.NORMATIVE` findings — a document designed to suppress gap detection is itself evidence of deliberate concealment.

### Legal domain grounding

The system explicitly distinguishes *law as written* (statutes, regulations, court holdings) from *law as applied* (enforcement practices, prosecutorial discretion, territorial and tribal law). Seven `LegalDomain` values are tracked; gaps between `STATUTE` and `ENFORCEMENT_PRACTICE` nodes produce `DOCTRINAL` findings. US territories and tribal nations are first-class jurisdictions, not footnotes.

Legal knowledge is currently drawn from what the base model absorbed during pretraining. The extent to which statutory and regulatory text should be part of a dedicated fine-tuning pass — versus treated as background knowledge — is an open design question. A targeted SFT stage on public legal corpora (US Code, CFR, tribal treaty text) would sharpen the system's ability to identify normative and doctrinal gaps that require knowing the text of the law, not just its general contours.

---

## Installation

Python 3.12 or higher is required.

```bash
# Clone and install in editable mode with dev dependencies
pip install -e ".[dev]"
```

Core dependencies: `torch`, `transformers`, `networkx`, `pydantic`, `fastapi`, `click`, `trl`, `peft`, `torch-geometric`, `llama-index`, `spacy`, `openai`, `azure-ai-inference`, `httpx`.

---

## Configuration

Copy `deployment/.env.example` to `.env.local` (gitignored) and fill in real values. **Never commit credentials.**

```bash
# Select provider: vllm (local), azure (Claude via Azure AI Foundry), or hybrid
DETECTIVE_PROVIDER=vllm

# Local vLLM endpoint (OpenAI-compatible)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=detective                  # LoRA module name (or base model ID)

# Azure AI Foundry — Claude critic
AZURE_ENDPOINT=https://<your-resource>.services.ai.azure.com/
AZURE_API_KEY=<your-key>
AZURE_MODEL=claude-sonnet-4-5-2

# Persistent graph store
DETECTIVE_GRAPH_BACKEND=kuzu
DETECTIVE_KUZU_PATH=data/kuzu_db
```

`DETECTIVE_PROVIDER` is required at startup — the system fails fast with a clear error if it is not set.

---

## CLI

After installation the `detective` command is available.

### Analyze a document for information gaps

```bash
detective analyze path/to/document.txt
detective analyze path/to/document.txt --query "What financial records are absent?"
detective analyze path/to/document.txt --constitution docs/constitution.md
```

### Build and inspect the entity network

```bash
detective network path/to/document.txt
```

### Run constitutional self-critique on an analysis

```bash
detective critique "Entity A was active throughout 2010-2017."
detective critique "Entity A controlled the network." --constitution docs/constitution.md
```

---

## Reference GPT (`microgpt.py`)

`microgpt.py` is a self-contained, dependency-free GPT implementation in ~200 lines of pure Python. It is the algorithmic foundation of the project — read it first to understand the core mechanism before reading `src/core/model.py`.

```bash
# Downloads input.txt if absent, trains for 1000 steps, then generates samples
python microgpt.py
```

It implements custom scalar autograd (`Value`), BOS tokenisation, multi-head attention, RMSNorm, Adam from scratch, and temperature-sampled inference. **Do not modify this file.** Extension work belongs in `src/core/model.py`.

---

## Training

The alignment pipeline has three stages:

| Stage | Script | Description |
|---|---|---|
| SFT | `src/training/train_sft.py` | Supervised fine-tuning on gap-annotation JSONL data |
| DPO | `src/training/train_dpo.py` | Direct preference optimisation on constitution-generated pairs |
| GRPO | `src/training/train_grpo.py` | Online RL with constitution-grounded reward |

Preference pairs are generated automatically in `src/training/generate_preferences.py` using the generator+critic loop — no human annotators required.

Gap annotation data goes in `data/annotations/` (gitignored). Model checkpoints go in `checkpoints/` (gitignored).

---

## Evaluation

The independent-discovery harness in `evaluation/independent_discovery.py` compares system-detected gaps against expert-independently-discovered gaps:

```python
from evaluation.independent_discovery import summarise, DiscoveryEvaluation

items = [
    DiscoveryEvaluation(gap_id="g1", independently_discovered=True,  path_taken="...", matches_system=True),
    DiscoveryEvaluation(gap_id="g2", independently_discovered=True,  path_taken="...", matches_system=False),
    DiscoveryEvaluation(gap_id="g3", independently_discovered=False, path_taken="...", matches_system=True),
]
summary = summarise(items)
# EvaluationSummary(total=3, discovery_rate=0.67, precision=0.50, recall=0.50, f1=0.50)
```

**Quality targets:** gap detection F1 > 0.75, relationship precision > 0.80.

---

## Testing

```bash
pytest tests/                                              # full suite
pytest tests/test_hypothesis.py                           # single file
pytest tests/test_hypothesis.py::test_hypothesis_update   # single test
```

**Coverage target:** 80%+. Tests use `MockProvider` as a standard test double throughout — no network calls in the test suite.

---

## Development

```bash
ruff check .          # lint
ruff format .         # format
mypy src/             # type check
python bootstrap.py   # bootstrap directory structure (idempotent)
```

### Architectural decisions

Key design decisions are recorded as ADRs in `docs/vault/decisions/`:

| ADR | Decision |
|---|---|
| ADR-001 | `ModelProvider` Protocol abstraction (vLLM / Azure Foundry) |
| ADR-002 | Moral compass as dual-function constitution |
| ADR-003 | Independent DistilBERT classifiers for Modules A/B/C |
| ADR-004 | Immutable ExperienceLibrary (FLEX) over mutable engine |
| ADR-005 | Typed semantic relations on knowledge graph edges |
| ADR-006 | Prompt injection defence and constitutional resilience |
| ADR-007 | American law as written vs. law as applied |
| ADR-008 | Persistent graph store (Kuzu) replacing NetworkX in-memory |
| ADR-009 | Hub-first welfare classifier loading |
| ADR-010 | Trajectory urgency scoring (4-weight combined score) |
| ADR-011 | Scenario extraction pipeline (detective → forecaster) |
| ADR-013 | Hybrid provider routing (local scoring + cloud reasoning) |
| ADR-014 | Reasoning trace capture with SSE streaming |
| ADR-015 | Epstein-docs ingestion pipeline |
| ADR-016 | Entity filter (3-layer noise reduction) |
| ADR-017 | ZeroGPU Gradio workbench integration |
| ADR-018 | SourceDocument Protocol for sourcing pipeline |
| ADR-019 | Centralized logging convention |
| ADR-020 | Python 3.13 compatibility + dependency cleanup |
| ADR-021 | Autonomous investigation agent |
| ADR-022 | Clearnet investigation sources |
| ADR-023 | Assumption detection integration (A+B+C in agent) |
| ADR-024 | OCR fallback chain with confidence scoring |
| ADR-025 | spaCy NER pipeline |
| ADR-026 | MinHash/LSH near-duplicate detection |
| ADR-027 | Person auditor (claim decomposition + verification) |
| ADR-028 | Multi-task loss integration (3-head training) |

### Vault memory

Hypothesis traces and gap findings are persisted to an Obsidian-compatible Markdown vault (`docs/vault/`). Two implementations are available:

- **`FileVaultClient`** (`src/memory/vault.py`) — writes plain `.md` files to a directory on disk. No Obsidian required. This is the recommended mode for server and CI deployments; set `DETECTIVE_VAULT_PATH` and it works standalone.
- **`MCPVaultClient`** (`src/memory/mcp_vault.py`) — connects to the [Obsidian Local REST API](https://github.com/coddingtonbear/obsidian-local-rest-api). Enables the full Obsidian graph view, backlink traversal, and search from the desktop app. Obsidian does not need to be installed on the server; this mode is useful when a researcher wants to browse findings interactively on a local machine pointed at the same vault directory.

Both implement the `VaultClient` Protocol and are interchangeable at runtime.

---

## Data layout

```
data/epstein-docs/         ← Epstein document corpus (gitignored)
data/foia/                 ← FOIA document archives (gitignored)
data/kuzu_db/              ← Kuzu graph database directory (gitignored)
data/training/             ← Training data: constitutional pairs, welfare data (gitignored)
data/annotations/          ← Gap/assumption annotation JSONL (gitignored)
checkpoints/               ← Model .pt files and DPO adapters (gitignored)
docs/vault/                ← ADRs, hypothesis traces, gap findings
```

---

## Project status

The analytical framework is implemented and tested (1097 tests passing): gap taxonomy (5 types), assumption taxonomy (3 types, Modules A/B/C all implemented), immutable hypothesis lineage, parallel hypothesis evolution (Graph of Thought), 4-layer analytical pipeline with inline gap detection, constitutional self-critique, injection defence, legal domain grounding, Φ(humanity) welfare scoring (v2.1, 8 constructs), autonomous investigation agent (plan→gather→analyze→reflect→evolve→enrich), 8 pluggable investigation sources (FOIA, graph, web, news, CourtListener, SEC EDGAR, OCCRP, IICSA), person auditing, entity NER, MinHash dedup, and a Phi trajectory forecasting bridge.

Deployment uses vLLM with optional LoRA adapter support for the DPO-trained detective model. See `deployment/vllm_start.sh --lora` for setup.

The canonical implementation plan and system design documents are in `docs/plans/`. The living intellectual foundation is `docs/constitution.md`.

---

## Intellectual foundation

This system is grounded in a tradition of thinkers who understood, from lived experience, that the record is never neutral:

Patricia Hill Collins (standpoint knowledge; who gets to decide what counts as evidence), Paul Ekman (deception through omission; the hundred-thousand shapes of falsehood — including through micro-expression, a direction that opens toward vision-service integration for video and recorded testimony), Isabel Wilkerson (structural silence as finding; the evil of silence), James Baldwin (credibility resistance — being disbelieved precisely because what you said was true), bell hooks (honest analysis as an act of care, not aggression), Kelly Hayes & Mariame Kaba (the gap between official narrative and the knowledge held by those most affected), Richard Delgado & Jean Stefancic (the curated record; counter-narrative as valid evidence), Assata Shakur (the record used as a weapon against the documented), Chancellor Williams (documentary suppression across a civilisational horizon).

These are not decorative citations. They are the load-bearing structure of the detection logic. See `docs/constitution.md` for the full moral compass.
