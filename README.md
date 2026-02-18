# Detective LLM

**Detective LLM** is an information gap analysis system that detects what is *absent* from investigative datasets — temporal silences, evidential voids, implied entity connections, and cognitive biases — rather than what is present. It was built for geopolitical influence networks but the epistemic framework generalises to any domain where the most significant findings are the things that were never documented.

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
  hypothesis.py                   Immutable Hypothesis with auditable lineage
  evolution.py                    Hypothesis evolution via experience library (FLEX)
  experience.py                   ExperienceLibrary — immutable tuple of past trajectories
  module_a.py                     Module A: cognitive bias classifier (DistilBERT)
  constitution.py                 Constitutional AI self-critique loop

src/inference/                  Analytical pipeline
  pipeline.py                     4-layer pipeline: intent → evidence → reasoning → verify
  reflection.py                   Constitutional reflection trigger injection

src/data/                       Knowledge graph
  knowledge_graph.py              NetworkX-backed graph with typed edges + n-hop decay
  graph_store.py                  GraphStore Protocol (InMemoryGraph / KuzuGraph)
  kuzu_graph.py                   Persistent Kuzu backend (production)

src/security/                   Injection defence
  sanitizer.py                    Layer 1: pattern-based injection detection at ingestion
  prompt_guard.py                 Layer 2: constitutional prompt isolation at inference

src/training/                   Alignment training
  train_sft.py                    SFT warm-up on gap-annotation data
  train_dpo.py                    DPO preference learning from critique pairs
  train_grpo.py                   GRPO online RL from constitution-grounded rewards
  generate_preferences.py         Automated preference pair generation (no annotators)

src/memory/                     Persistent memory
  vault.py                        FileVaultClient — Obsidian-compatible filesystem vault
  mcp_vault.py                    MCP integration for Obsidian Local REST API
  adr.py                          Architectural Decision Record (ADR) helpers

src/api/                        REST API
  routes.py                       FastAPI endpoints for gap analysis and visualisation

src/cli/                        Command-line interface
  main.py                         Click CLI — `detective` entry point

evaluation/
  independent_discovery.py        Precision / recall / F1 evaluation harness
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

Multi-task loss: `L_total = L_language + α·L_gap` (α = 0.3)

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

Modules B and C are planned; Module A (`src/detective/module_a.py`) is implemented using a DistilBERT classifier (`distilbert-base-uncased` placeholder — replace with a fine-tuned checkpoint).

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
# Select provider: vllm (local) or azure (Claude via Azure AI Foundry)
DETECTIVE_PROVIDER=vllm

# Local vLLM endpoint (OpenAI-compatible)
VLLM_BASE_URL=http://localhost:11434/v1
VLLM_MODEL=deepseek-r1:7b

# Azure AI Foundry — Claude critic
AZURE_FOUNDRY_ENDPOINT=https://<your-resource>.services.ai.azure.com/models
AZURE_FOUNDRY_KEY=<your-key>
AZURE_FOUNDRY_MODEL=claude-sonnet-4-5

# Persistent graph store
DETECTIVE_GRAPH_BACKEND=kuzu
DETECTIVE_KUZU_PATH=/data/detective/graph.kuzu

# Obsidian vault path (filesystem mode)
DETECTIVE_VAULT_PATH=/data/detective/vault
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
# Downloads names.txt if absent, trains for 1000 steps, then generates samples
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
| ADR-002 | Moral compass as dual-function constitution (training signal + inline constraint) |
| ADR-003 | Independent DistilBERT classifiers for Modules A/B/C |
| ADR-004 | Immutable ExperienceLibrary (FLEX) over mutable engine |
| ADR-005 | Typed semantic relations on knowledge graph edges |
| ADR-006 | Prompt injection defence and constitutional resilience |
| ADR-007 | American law as written vs. law as applied |
| ADR-008 | Persistent graph store (Kuzu) replacing NetworkX in-memory |

### Vault memory

Hypothesis traces and gap findings are persisted to an Obsidian-compatible Markdown vault (`docs/vault/`). Use `FileVaultClient` (filesystem) or `MCPVaultClient` (Obsidian Local REST API) — both implement the `VaultClient` Protocol.

---

## Data layout

```
data/epstein/raw/          ← Raw clones of source document repositories (gitignored)
data/epstein/processed/    ← Parsed Document objects (gitignored)
data/annotations/          ← Gap/assumption annotation JSONL (gitignored)
checkpoints/               ← Model .pt files (gitignored)
docs/vault/                ← Obsidian vault: ADRs, hypothesis traces, gap findings
```

---

## Project status

The core epistemic framework — gap taxonomy, assumption taxonomy, immutable hypothesis lineage, experience library, 4-layer pipeline, constitutional self-critique, injection defence, and legal domain grounding — is implemented and tested. Some capabilities (Modules B and C, full Kuzu persistence, LlamaIndex retrieval, complete training loops) are scaffolded and documented but not yet fully wired.

The canonical implementation plan is `IMPLEMENTATION_PLAN.md`. The system design source of truth is `DETECTIVE_LLM_PRD.md`. The epistemic foundation is `docs/constitution.md`.

---

## Epistemic foundation

This system is grounded in a tradition of thinkers who understood, from lived experience, that the record is never neutral:

Patricia Hill Collins (standpoint epistemology), Paul Ekman (deception through omission), Isabel Wilkerson (structural silence as finding), James Baldwin (credibility resistance), bell hooks (honest analysis as care), Kelly Hayes & Mariame Kaba (gap between official narrative and community knowledge), Richard Delgado & Jean Stefancic (curated record and counter-narrative), Assata Shakur (the record as weapon), Chancellor Williams (documentary suppression on a civilisational scale).

These are not decorative citations. They are the epistemic architecture of the detection system. See `docs/constitution.md` for the full moral compass.
