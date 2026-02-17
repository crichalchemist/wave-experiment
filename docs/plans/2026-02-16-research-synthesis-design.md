# Detective LLM: Research Synthesis Design Update

**Date:** 2026-02-16
**Based on:** 17 arXiv papers (2024-2026) + LLM Engineers Handbook + Agentic Patterns
**Status:** Approved — ready for implementation plan revision

---

## Summary of Changes

This document supersedes and updates IMPLEMENTATION_PLAN.md based on synthesis of current
research. Approximately 60-70% of the originally planned custom infrastructure can be replaced
with open-source components. The genuinely novel work is the moral compass, gap detection logic,
domain knowledge graph schema, and evaluation protocol.

---

## Infrastructure

**Local inference (primary):**
- Azure GPU VM running vLLM (OpenAI-compatible API)
- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (fits A10 24GB / V100 16GB)
- vLLM handles batching, quantization (AWQ/GPTQ), and request routing

**Fallback (critique + overflow):**
- Azure AI Foundry Claude API
- Used as the *critic* model in constitutional self-critique loops
- Separation of generator (local) and critic (Claude) is architecturally superior to same-model CAI

**Provider abstraction** (`src/core/providers.py`):
```python
from typing import Protocol

class ModelProvider(Protocol):
    def complete(self, prompt: str, **kwargs) -> str: ...
    def embed(self, text: str) -> list[float]: ...

# Two concrete implementations:
# - VLLMProvider (local Azure VM)
# - AzureFoundryProvider (Claude fallback)
```

The rest of the codebase calls `ModelProvider`, never a concrete class.

---

## Section 1: Training Strategy

**Sources:** Learning-at-Criticality (LaC), Beyond Scaling Law (DED), Iterative Deepening
Sampling (ID-Sampling), When To Solve/When To Verify

### Changes from original plan

| Original | Updated |
|---|---|
| 1,000+ annotated examples before training | 200 high-quality, high-diversity gap examples |
| Week 5: Pure SFT on Epstein corpus | Short SFT warm-up (200 examples) → GRPO immediately |
| Week 7: DPO with human expert preference pairs | Constitutional AI self-critique (no human annotators) |
| Separate reflection module (Week 10) | Inline ID-Sampling reflection trigger during generation |

### Rationale

**LaC** shows that with GRPO, a model generalizes from a single exemplar at the "critical
learning transition" — a phase where reasoning path length variance diverges, forcing rule
extraction over pattern matching. The 1,000-example annotation target was based on SFT
assumptions; RL changes the data efficiency equation.

**DED** replaces the DPO pipeline with a 3-stage distillation framework:
1. Teacher selection via smoke tests (best teacher ≠ largest model)
2. Corpus filtering for diversity, not just correctness (remove redundant reasoning paths)
3. Diverse trajectory sampling per question

**ID-Sampling** — inject `"Wait — let me reconsider this gap."` as a trigger mid-generation.
Improves output quality without retraining. Maps onto the reflection module without requiring
a separate critique pass.

**SC-first compute allocation** (When To Solve/Verify):
- Confidence < 0.5: generate more competing hypotheses (breadth, Self-Consistency)
- Confidence ≥ 0.5: invest in verification passes (GenRM-style depth)

### Open-source components replacing custom code

- `huggingface/trl` — `GRPOTrainer` (replaces `src/training/train_multitask.py`)
- `huggingface/peft` — LoRA adapters (replaces full fine-tune loops)
- `huggingface/trl` — `DPOTrainer` (available as fallback if DED is insufficient)

---

## Section 2: Core Architecture

**Sources:** Analytical Search, LLM-Enhanced Rumor Detection (LLM+GNN), Thinking Machines survey

### Changes from original plan

| Original | Updated |
|---|---|
| Verification as post-hoc step (Week 10) | Inline verification at each reasoning step |
| Pure `GraphAttentionLayer` | LLM-scores-edges + Bi-GAT aggregation (two-track) |
| Week 5 SFT baseline first | Short SFT warm-up → RL (System 2 capabilities require RL, not SFT scale) |

### Analytical Search — 4-layer inference pipeline

Replaces the current monolithic `detective_llm.analyze()` design:

```
Layer 1: Query Understanding
  → parse analytical intent: descriptive | predictive | prescriptive
  → identify entity scope and temporal bounds

Layer 2: Recall-Oriented Retrieval
  → multi-path evidence gathering from knowledge graph
  → retrieve across document, entity, and temporal dimensions simultaneously

Layer 3: Reasoning-Aware Fusion
  → synthesize heterogeneous evidence
  → maintain provenance chain for each claim

Layer 4: Adaptive Verification (NEW — not in original plan)
  → validate each intermediate conclusion before proceeding
  → flag low-confidence steps for alternative path exploration
```

### LLM+GNN hybrid graph architecture

Replace `GraphAttentionLayer` (pure attention) with two-track hybrid:

```python
# Track 1: LLM scores edge plausibility (semantic)
edge_scores = llm_provider.complete(
    f"How likely is {entity_a} → {entity_b} given: {context}?"
)

# Track 2: Bi-GAT aggregates structurally (topology)
# Uses pytorch_geometric GATv2Conv
node_embeddings = bi_gat(entity_embeddings, adjacency, edge_weights=edge_scores)
```

Finding from LLM+GNN paper: standalone LLMs exhibit systematic bias in graph reasoning
(different models reach different conclusions on identical graphs). The GNN provides
structural grounding that corrects LLM bias.

### Open-source components

- `pyg-team/pytorch_geometric` — `GATv2Conv` (replaces custom `GraphAttentionLayer`)
- `llamaindex` or `langchain` — retrieval pipeline for Layer 2 (configure, don't build)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` — teacher model for DED distillation

---

## Section 3: Hypothesis Evolution Engine

**Sources:** FLEX (Forward Learning from Experience), SynchToM (Theory of Mind), When To Solve/Verify

### Changes from original plan

| Original | Updated |
|---|---|
| Mutable `HypothesisEvolutionEngine` class | Functional pipeline over immutable `ExperienceLibrary` |
| 0.3 confidence threshold for spawning alternatives | 0.5 branching rule (SC below, verify above) |
| Open-ended self-critique prompts | Constitution-grounded epistemic alignment check |

### ExperienceLibrary pattern (from FLEX)

```python
from typing import NamedTuple, Literal

class Experience(NamedTuple):
    hypothesis_id: str
    evidence: str
    action: Literal["confirmed", "refuted", "spawned_alternative"]
    confidence_delta: float
    outcome_quality: float  # scored post-hoc against ground truth

# Functional pipeline — no mutable state
def evolve_hypothesis(
    h: Hypothesis,
    new_evidence: str,
    library: tuple[Experience, ...],
    provider: ModelProvider,
) -> tuple[Hypothesis, Experience]:
    """Consult library for similar past trajectories before evolving."""
    similar = query_library(library, h, new_evidence)
    evolved = apply_evolution(h, new_evidence, similar, provider)
    record = Experience(h.id, new_evidence, classify_action(h, evolved), ...)
    return evolved, record
```

FLEX shows a scaling law of experience: performance improves log-linearly with library size.
The library is also **inheritable** across domains — Epstein → lobbying patterns → intelligence
networks — because reasoning patterns (not domain facts) are what's stored.

### SynchToM epistemic gap framing

Reflection prompts must ask: *"What would the true state be, independent of what the
documents say?"* not just *"Is this hypothesis internally consistent?"*

The distinction maps to:
- `belief b` = what evidence in the corpus supports
- `true state s` = what actually occurred
- `gap Δ(b, s)` = the information gap to surface

Constitution-grounded reflection (see Section 5) ensures the model prioritizes epistemic
alignment over instruction-following — the failure mode SynchToM identified in all 11
models they tested.

### Open-source components

- FLEX codebase (open-source) — wire `ExperienceLibrary` to `Hypothesis` dataclass
- Constitutional AI self-critique — prompting pattern, no library needed

---

## Section 4: Evaluation & Governance

**Sources:** Turing Tests for AI Scientist, Legal Reasoning Challenges, AgentLeak, VaryBalance

### Changes from original plan

| Original | Updated |
|---|---|
| F1 vs annotation checklist | Independent-discovery evaluation protocol |
| temporal / evidential / contradiction | + normative + doctrinal gap types |
| Implicit API privacy | Explicit data minimization per inter-module channel |
| Manual inter-annotator agreement (IAA) | VaryBalance-style automated consistency scoring |

### Independent-discovery evaluation protocol

A gap detection result is valid if a skilled analyst would independently reach the same gap
through a different path — not if it matches an annotation. Expert evaluators in Week 16
should be asked: *"Would you have identified this gap without being told where to look?"*

This is a stronger and more honest test than F1, and avoids rewarding systems that memorize
annotation patterns.

### Extended gap taxonomy

```python
class GapType(Enum):
    TEMPORAL = "temporal"           # Missing time periods
    EVIDENTIAL = "evidential"       # Claims without documentation
    CONTRADICTION = "contradiction" # Conflicting information
    NORMATIVE = "normative"         # What *should* be documented but isn't
    DOCTRINAL = "doctrinal"         # Unstated institutional rules assumed to apply
```

Normative and doctrinal gaps come from Legal Reasoning Challenges — they represent the
class of gaps that traditional IR systems cannot detect (because they require knowing what
*should* exist, not just what does exist).

### API data minimization (from AgentLeak)

AgentLeak finding: multi-agent configurations *reduce* per-channel output leakage (27.2% →
43.2% in single-agent) but *increase* total exposure 1.6× through internal channels.

Hard constraint on `src/api/` design: each inter-module call carries only the minimum data
required to complete its task. The hypothesis engine does not receive raw document text —
only extracted claims. The knowledge graph does not receive hypothesis confidence — only
entity queries.

### Automated annotation quality (VaryBalance)

VaryBalance finding: LLM-generated text rewrites more consistently than human text
(lower mean standard deviation of log perplexity across rewrites).

Use as annotation filter: generate two independent gap analyses of the same claim,
measure rewrite-similarity. High similarity = reliable annotation. Replaces manual IAA
calculation in Week 2 with a ~50-line automated check.

---

## Section 5: Temporal Reasoning, Classification & Semantic Relations

**Sources:** TSRBench, System 1 vs 2 for TSF, Cyberbullying Detection, SRA FlowchartQA

### Changes from original plan

| Original | Updated |
|---|---|
| `gap_head` over shared hidden state | Dedicated temporal embedding track (separate from LM head) |
| Single deep reasoning pass for temporal gaps | SC majority-vote across multiple timeline analyses |
| DetectiveGPT forward pass for Modules A/B/C | DistilBERT classifiers, independent of generative model |
| Untyped knowledge graph edges | Typed semantic relations (4 types from SRA) |

### Temporal embedding track (TSRBench finding)

Semantic understanding and numerical/temporal prediction are *decoupled* capabilities.
A model that understands *what* an entity was doing cannot automatically infer *when* it
should have appeared in records.

The `gap_head` needs its own dedicated track:

```python
class DetectiveGPT(nn.Module):
    def __init__(self, ...):
        # Existing: shared hidden state → lm_head
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # New: dedicated temporal track (does NOT share weights with lm_head)
        self.temporal_emb = nn.Embedding(vocab_size, temporal_dim)
        self.temporal_encoder = nn.TransformerEncoder(...)
        self.gap_head = nn.Linear(temporal_dim, len(GapType))
```

### SC for temporal analysis (REC4TS finding)

Self-Consistency (majority vote) is the most effective test-time strategy for time series
reasoning — more reliable than single deep reasoning passes. Apply to temporal gap detection:
generate N=5 independent timeline analyses, majority-vote on which time periods are flagged
as gaps.

### DistilBERT for Modules A/B/C (Cyberbullying Detection finding)

BERT family achieves 95% accuracy at 0.053s inference / 35MB RAM for text classification.
DistilBERT matches BERT recall at lower compute.

Modules A, B, C become **independent DistilBERT classifiers**, not heads on the generative
model. Benefits: (1) separable training, (2) faster inference, (3) can update classifiers
without retraining the generative backbone.

```python
# src/detective/module_a.py
from transformers import pipeline

_bias_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased",  # fine-tuned on bias detection data
)

def detect_cognitive_biases(text: str) -> list[BiasDetection]:
    results = _bias_classifier(text, top_k=None)
    return [BiasDetection(...) for r in results if r["score"] > 0.7]
```

### Typed semantic relations (SRA FlowchartQA)

Replace untyped knowledge graph edges with four semantic relation types:

```python
class RelationType(Enum):
    CONDITIONAL = "conditional"   # A describes outcome contingent on B
    CAUSAL = "causal"             # A directly causes B
    INSTANTIATIVE = "instantiative"  # B is a specific instance of A
    SEQUENTIAL = "sequential"    # B occurs chronologically after A
```

A 2-hop **causal** chain (A caused B caused C) is investigatively different from a 2-hop
**sequential** chain (A preceded B preceded C). Typed relations make n-hop confidence
decay meaningful rather than generic.

---

## Section 6: The Moral Compass

**Sources:** User-provided epistemic alignment training material

**Location:** `docs/constitution.md` (to be written from user's material)

The moral compass serves two distinct functions:

### 1. Training signal (Constitutional AI)

```
Generate gap analysis →
  Critique against constitution (Claude via Azure Foundry) →
    Revise →
      Treat [original, revised] as preference pair
```

This generates preference pairs at scale without human expert annotators. The constitution
defines what epistemically *honest* gap detection looks like vs. pattern-matching or
motivated reasoning. Claude-as-critic (external, different model) produces stronger
critiques than self-critique with the same model.

### 2. Inline constraint (reflection grounding)

The ID-Sampling trigger becomes constitution-grounded:

> *"Wait — before continuing, does this analysis respect [principle X]? Let me reconsider."*

This resolves the SynchToM failure mode: models that prioritize instruction-following over
epistemic alignment. When the constitution *is* the instruction, that tension disappears.

---

## What Must Be Built (Novel Work)

Everything else can be assembled from open-source. The irreducible custom work:

1. **`docs/constitution.md`** — the moral compass document
2. **Gap detection training data** — 200 annotated examples across 5 gap types
3. **Module B (historical determinism)** — no open dataset for this bias type
4. **Module C (geopolitical presumptions)** — no open dataset; domain-specific
5. **Knowledge graph schema** — entity types, relation constraints for investigative domain
6. **Independent-discovery evaluation protocol** — the evaluation instrument itself
7. **Data minimization API design** — architectural pattern applied to `src/api/`

---

## Revised Phase Timeline

| Phase | Weeks | Key change |
|---|---|---|
| Foundation | 1-4 | Annotation target: 200 (not 1,000). DistilBERT classifiers from Week 4. |
| Training | 5-8 | SFT warm-up → GRPO (not SFT → DPO). CAI replaces human preference pairs. |
| Hypothesis | 9-12 | ExperienceLibrary pattern. Constitution-grounded reflection. |
| Interface | 13-16 | Provider abstraction (vLLM + Azure Foundry). Data minimization in API. |

---

## Open-Source Stack Summary

| Layer | Tool |
|---|---|
| Local inference | vLLM on Azure GPU VM |
| Base model | DeepSeek-R1-Distill-Qwen-7B |
| Critique model | Claude via Azure AI Foundry |
| RL training | `huggingface/trl` GRPOTrainer |
| Efficient fine-tuning | `huggingface/peft` LoRA |
| Classification modules A/B/C | `huggingface/transformers` DistilBERT |
| Graph layer | `pytorch_geometric` GATv2Conv |
| Knowledge graph | `networkx` (already in deps) |
| Retrieval pipeline | `llamaindex` |
| Experience library | FLEX (open-source, adapt to Hypothesis dataclass) |
| NER / entity extraction | `spacy` en_core_web_trf |
| PDF ingestion | `pdfplumber` (already in deps) |
| API serving | `fastapi` + `uvicorn` (already in deps) |
| CLI | `click` (already in deps) |
