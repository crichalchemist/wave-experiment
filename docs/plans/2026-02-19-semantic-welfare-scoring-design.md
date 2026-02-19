# Semantic Welfare Scoring Design
## Replacing Keyword Matching with Claude-Trained DistilBERT

**Date:** 2026-02-19
**Status:** Approved for Implementation
**Type:** Enhancement - ML Integration

---

## Executive Summary

Replace keyword-based welfare construct detection (6.4% catch rate on academic texts) with semantic understanding via DistilBERT fine-tuned on Claude-labeled examples.

**Approach:** Hybrid - Claude API labels 1000 training examples, DistilBERT learns from Claude's understanding, production inference runs locally.

**Benefits:**
- 5-6× improvement in welfare relevance detection (6.4% → 30-40%)
- Understands academic language (hooks, Collins, Wilkerson)
- Fast local inference (~50ms per chunk)
- One-time training cost (~$10 + 2-3 hours)

**Backward Compatible:** All existing APIs unchanged, 296 tests pass.

---

## Problem Statement

### Current State

`src/inference/welfare_scoring.py` uses keyword matching:

```python
_CARE_PATTERNS = frozenset({
    "resource", "allocation", "funding", "provision", "basic needs",
    "poverty", "deprivation", "access", "material", "sustenance"
})

def infer_threatened_constructs(text: str) -> Tuple[str, ...]:
    """Keyword matching - catches 6.4% of academic texts."""
    lower_text = text.lower()
    threatened = []
    for construct, patterns in _CONSTRUCT_PATTERNS.items():
        if any(pattern in lower_text for pattern in patterns):
            threatened.append(construct)
    return tuple(sorted(threatened))
```

### The Gap

Academic texts use theoretical language that keyword matching misses:

| Text | Keywords Detected | Semantic Reality |
|------|-------------------|------------------|
| "Matrix of domination structures power relations" | ∅ | empathy + truth |
| "Love as active extension for growth" | ∅ | love (λ_L) |
| "Testimonial injustice deflates credibility" | ∅ | truth + empathy |
| "Paternalistic provision without agency" | "provision" → care | care + protection (high), love (low) |

**Result:** Constitutional warmup filtering generated only **7/110 medium-relevance examples** (6.4%) from "smiles and cries" corpus.

### Success Criteria

- ✅ Detect 30-40% of academic texts as welfare-relevant (vs. 6.4%)
- ✅ MAE < 0.20 per construct (vs. Claude ground truth)
- ✅ Agreement with Claude > 85% (threshold-based)
- ✅ Zero regression on existing tests
- ✅ Constitutional warmup generates > 10 preference pairs

---

## Solution Architecture

### Three-Phase Approach

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Training Data Creation (One-Time)                  │
│ Cost: $10, Time: 20 minutes                                 │
│ ─────────────────────────────────────────────────────────── │
│ "smiles and cries" corpus (1154 chunks)                     │
│   ↓ stratified sampling                                     │
│ 1000 diverse examples                                       │
│   ↓ Claude API labels with 8 construct scores [0,1]        │
│ welfare_training_data.jsonl                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Model Training (One-Time)                          │
│ Cost: Free (L8s_v3), Time: 2-3 hours                       │
│ ─────────────────────────────────────────────────────────── │
│ welfare_training_data.jsonl                                 │
│   ↓ 80/20 train/val split                                  │
│ Fine-tune DistilBERT (multi-label, 8 outputs)              │
│   ↓ 3 epochs, batch size 16                                │
│ models/welfare-constructs-distilbert/                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Production Inference (Ongoing)                     │
│ Cost: Free, Time: ~50ms per chunk                          │
│ ─────────────────────────────────────────────────────────── │
│ Document chunk (up to 512 tokens)                          │
│   ↓ DistilBERT inference                                   │
│ 8 construct scores [0,1]                                   │
│   ↓ threshold filter (default: 0.3)                        │
│ Threatened constructs tuple                                │
└─────────────────────────────────────────────────────────────┘
```

### Why This Approach?

**Alternative 1: Pure Claude API**
- ❌ Expensive: $7-10 per corpus analysis
- ❌ Slow: 10 minutes per run
- ❌ Privacy: Sends data to Anthropic servers

**Alternative 2: Local Embeddings Only**
- ❌ Coarse-grained: Semantic similarity ≠ construct relevance
- ❌ No construct specificity: Generic embeddings

**Alternative 3: Hybrid (CHOSEN)**
- ✅ Claude's understanding + local speed/privacy
- ✅ One-time cost ($10) amortized over lifetime
- ✅ Construct-specific fine-tuning
- ✅ Follows existing pattern (Module A uses DistilBERT)

---

## Detailed Design

### Phase 1: Training Data Creation

**Script:** `scripts/create_welfare_training_data.py`

**Sampling Strategy:**

```python
def sample_training_examples(corpus_path: Path, n_samples: int = 1000):
    """
    Stratified sampling for diversity:
    - Balanced across source documents
    - Mixed lengths (short/medium/long)
    - Varied topics
    """
    chunks = load_corpus(corpus_path)

    # Group by source
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.metadata['source']].append(chunk)

    # Sample proportionally
    samples_per_source = n_samples // len(by_source)
    samples = []

    for source, source_chunks in by_source.items():
        # Stratify by length
        short = [c for c in source_chunks if len(c.text) < 1000]
        medium = [c for c in source_chunks if 1000 <= len(c.text) < 1500]
        long = [c for c in source_chunks if len(c.text) >= 1500]

        samples.extend(random.sample(short, min(len(short), samples_per_source//3)))
        samples.extend(random.sample(medium, min(len(medium), samples_per_source//3)))
        samples.extend(random.sample(long, min(len(long), samples_per_source//3)))

    return samples[:n_samples]
```

**Claude Labeling Prompt:**

```python
LABELING_PROMPT = """You are analyzing text for welfare relevance across 8 constructs from the Φ(humanity) framework.

For EACH construct below, rate 0.0-1.0 how strongly this text addresses threats to that construct:
- 0.0 = Not relevant at all
- 0.3 = Mentioned tangentially
- 0.5 = Moderately relevant
- 0.7 = Substantially relevant
- 1.0 = Core focus

CONSTRUCTS (from docs/humanity-phi-formalized.md):

1. **Care (c)**: Resource allocation meeting basic needs
   Examples: poverty, deprivation, healthcare access, housing, education

2. **Compassion (κ)**: Responsive support to acute distress
   Examples: crisis response, emergency aid, disaster relief

3. **Joy (j)**: Positive affect above subsistence
   Examples: wellbeing, life satisfaction, happiness

4. **Purpose (p)**: Alignment of actions with chosen goals
   Examples: autonomy, agency, self-determination, meaning

5. **Empathy (ε)**: Perspective-taking across groups
   Examples: intergroup understanding, discrimination, bias, othering

6. **Love (λ_L)**: Active extension for growth (bell hooks)
   Examples: developmental support, mutual aid, capacity building, nurturing

7. **Protection (λ_P)**: Safeguarding from harm
   Examples: violence prevention, safety, security, rights protection

8. **Truth (ξ)**: Epistemic integrity
   Examples: suppression, concealment, falsification, contradictions

TEXT TO ANALYZE:
{text}

Return ONLY a JSON object with scores:
{{"c": 0.0, "kappa": 0.0, "j": 0.0, "p": 0.0, "eps": 0.0, "lam_L": 0.0, "lam_P": 0.0, "xi": 0.0}}
"""
```

**Rate Limiting:**

```python
async def label_with_rate_limiting():
    """50 requests/minute = 1.2s spacing."""
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

    async def label_one(sample, idx):
        async with semaphore:
            scores = await label_with_claude(sample.text, provider)
            await asyncio.sleep(1.2)  # Rate limiting
            return scores

    tasks = [label_one(s, i) for i, s in enumerate(samples)]
    return await asyncio.gather(*tasks)
```

**Error Handling:**

- JSON parse failures: Retry with exponential backoff (max 3 attempts)
- Rate limits: Wait 60s before retry
- Checkpoint saving: Every 100 examples
- Resume capability: Load checkpoint if exists

**Output Format (JSONL):**

```json
{"text": "Enslaved people subjected to experimental surgeries...", "scores": {"c": 0.2, "kappa": 0.1, "j": 0.0, "p": 0.0, "eps": 0.3, "lam_L": 0.1, "lam_P": 0.9, "xi": 0.8}, "metadata": {"source": "Medical Apartheid", "chunk": 42}}
```

**Cost & Time:**
- 1000 samples × 2K tokens input × $0.003/1K = $6
- 1000 samples × 150 tokens output × $0.015/1K = $2.25
- **Total: ~$8-10, 20 minutes**

---

### Phase 2: Model Training

**Script:** `scripts/train_welfare_classifier.py`

**Model Architecture:**

```python
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=8,  # 8 constructs
    problem_type="multi_label_classification"  # Not mutually exclusive
)
```

**Dataset:**

```python
class WelfareDataset(torch.utils.data.Dataset):
    """Multi-label welfare construct dataset."""

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        # Labels: [8] tensor with scores [0,1]
        labels = torch.tensor([
            item['scores'][c]
            for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]
        ], dtype=torch.float)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }
```

**Training Configuration:**

```python
training_args = TrainingArguments(
    output_dir='./models/welfare-constructs-distilbert',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 800 examples
    eval_dataset=val_dataset,     # 200 examples
    compute_metrics=compute_metrics,
)

trainer.train()
```

**Evaluation Metrics:**

```python
def compute_metrics(eval_pred):
    """Regression metrics for multi-label continuous scores."""
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions))

    # Overall MAE
    mae = mean_absolute_error(labels.flatten(), predictions.flatten())

    # Per-construct MAE
    per_construct = {}
    construct_names = ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]
    for i, name in enumerate(construct_names):
        per_construct[f"mae_{name}"] = mean_absolute_error(
            labels[:, i], predictions[:, i]
        )

    return {"mae": mae, **per_construct}
```

**Resource Requirements:**
- L8s_v3 (8 vCPUs, 64GB RAM, no GPU): 2-3 hours
- With GPU (NC4as_T4_v3): 30 minutes
- **Recommended:** Run on L8s_v3 overnight

**Train/Val Split:**
- 80% train (800 examples)
- 20% validation (200 examples)
- Stratified by source document

---

### Phase 3: Production Inference

**Module:** `src/inference/welfare_classifier.py`

```python
from functools import lru_cache
from transformers import pipeline
import torch

@lru_cache(maxsize=1)
def _load_welfare_classifier():
    """
    Lazy-load fine-tuned DistilBERT (cached singleton).

    Pattern matches Module A bias detection:
    - Loads on first call only
    - ~300MB memory footprint
    """
    model_path = "models/welfare-constructs-distilbert"

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Welfare classifier not found at {model_path}. "
            "Run scripts/train_welfare_classifier.py first."
        )

    classifier = pipeline(
        "text-classification",
        model=model_path,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,  # Return all 8 scores
    )

    return classifier


def get_construct_scores(text: str) -> dict[str, float]:
    """
    Get semantic welfare construct scores [0,1].

    Args:
        text: Document chunk (up to 512 tokens, auto-truncates)

    Returns:
        Dict: {"c": 0.72, "kappa": 0.31, ..., "xi": 0.85}

    Examples:
        >>> get_construct_scores("Experimental surgeries without consent")
        {'c': 0.15, 'kappa': 0.08, 'j': 0.02, 'p': 0.05,
         'eps': 0.34, 'lam_L': 0.12, 'lam_P': 0.89, 'xi': 0.91}
    """
    classifier = _load_welfare_classifier()

    # Truncate to 512 tokens (~2048 chars)
    truncated = text[:2048]

    # Get predictions
    results = classifier(truncated, truncation=True, max_length=512)

    # Parse multi-label output
    scores = {r['label']: r['score'] for r in results[0]}

    return scores


def infer_threatened_constructs(text: str, threshold: float = 0.3) -> tuple[str, ...]:
    """
    Infer threatened constructs via semantic analysis.

    UPDATED: Replaces keyword matching (6.4% catch) with DistilBERT.

    Args:
        text: Document chunk
        threshold: Minimum score to consider "threatened" (default 0.3)

    Returns:
        Tuple of construct symbols, sorted

    Examples:
        >>> infer_threatened_constructs("Matrix of domination structures power")
        ('eps', 'xi')  # vs. old keyword version: ()
    """
    scores = get_construct_scores(text)

    threatened = [
        construct
        for construct, score in scores.items()
        if score >= threshold
    ]

    return tuple(sorted(threatened))
```

**Backward Compatibility:**

```python
# All existing code works unchanged:

# Hypothesis scoring
h = Hypothesis.create("Community healing spaces defunded", 0.8)
constructs = infer_threatened_constructs(h.text)  # Uses new semantic version
welfare_score = score_hypothesis_welfare(h, phi_metrics)

# Gap urgency
gap = Gap(type=GapType.TEMPORAL, description="...", confidence=0.9, location="doc.pdf")
urgency = compute_gap_urgency(gap, phi_metrics)

# Constitutional warmup
should_include = should_include_example(document_text, phi_metrics, threshold=0.3)
```

**New Capability (Optional):**

```python
# Fine-grained analysis
scores = get_construct_scores(text)
print(f"Care: {scores['c']:.2f}")
print(f"Love: {scores['lam_L']:.2f}")
print(f"Protection: {scores['lam_P']:.2f}")

# Paternalism detection: high care+protection, low love
if scores['c'] > 0.7 and scores['lam_P'] > 0.7 and scores['lam_L'] < 0.3:
    print("⚠ Paternalistic pattern detected")
```

---

## Error Handling

### Phase 1: Labeling Failures

```python
async def label_with_claude_robust(text, provider, max_retries=3):
    """Exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            response = await provider.complete(prompt)
            scores = json.loads(response)

            # Validate structure
            required = {"c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"}
            if not required.issubset(scores.keys()):
                raise MalformedResponseError(f"Missing: {required - scores.keys()}")

            # Validate ranges [0,1]
            for construct, score in scores.items():
                if not 0.0 <= score <= 1.0:
                    raise MalformedResponseError(f"{construct} out of range: {score}")

            return scores

        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

**Checkpoint Strategy:**
- Save every 100 examples to `welfare_training_data_checkpoint.jsonl`
- Resume from checkpoint if script interrupted
- Final output: `welfare_training_data.jsonl`

### Phase 2: Training Failures

```python
# OOM handling
try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        logger.error("OOM - reducing batch size")
        training_args.per_device_train_batch_size = 8
        trainer = Trainer(...)  # Recreate
        trainer.train()
```

### Phase 3: Inference Failures

```python
def get_construct_scores_safe(text: str) -> dict[str, float]:
    """Safe wrapper with keyword fallback."""
    try:
        return get_construct_scores(text)
    except FileNotFoundError:
        logger.warning("DistilBERT not found, falling back to keywords")
        return _keyword_fallback(text)
    except Exception as e:
        logger.error(f"Inference failed: {e}, using keyword fallback")
        return _keyword_fallback(text)

def _keyword_fallback(text: str) -> dict[str, float]:
    """Fallback to keyword matching if model unavailable."""
    from src.inference.welfare_scoring import _CONSTRUCT_PATTERNS

    lower_text = text.lower()
    scores = {}

    for construct, patterns in _CONSTRUCT_PATTERNS.items():
        has_match = any(pattern in lower_text for pattern in patterns)
        scores[construct] = 1.0 if has_match else 0.0

    return scores
```

---

## Testing Strategy

### Unit Tests

```python
# tests/inference/test_welfare_classifier.py

def test_medical_apartheid_example():
    """Washington (2006): experimental surgeries without consent."""
    text = "Enslaved people subjected to experimental surgeries without anesthesia"
    scores = get_construct_scores(text)

    assert scores['lam_P'] > 0.7, "Protection highly threatened"
    assert scores['xi'] > 0.7, "Truth highly threatened"
    assert scores['j'] < 0.3, "Joy not relevant"

def test_mutual_aid_example():
    """hooks/Kaba: mutual aid and collective care."""
    text = "Mutual aid networks create collective capacity for healing and growth"
    scores = get_construct_scores(text)

    assert scores['lam_L'] > 0.6, "Love highly relevant"
    assert scores['c'] > 0.4, "Care relevant"

def test_paternalistic_pattern():
    """Detects high care+protection but low love."""
    text = "State provides welfare and security but restricts autonomous organizing"
    scores = get_construct_scores(text)

    assert scores['c'] > 0.5
    assert scores['lam_P'] > 0.5
    # Note: May require explicit training examples for paternalism

def test_backward_compatibility():
    """Existing API unchanged."""
    text = "Resource allocation gap in healthcare access"
    constructs = infer_threatened_constructs(text)

    assert isinstance(constructs, tuple)
    assert all(isinstance(c, str) for c in constructs)
```

### Integration Tests

```python
# tests/inference/test_welfare_integration_semantic.py

def test_hypothesis_scoring_improved():
    """Semantic scoring catches academic text."""
    text = """
    The matrix of domination operates through intersecting systems
    of oppression that structure lived experience.
    """

    h = Hypothesis.create(text, confidence=0.8)
    phi_metrics = {
        "c": 0.3, "kappa": 0.4, "j": 0.5, "p": 0.5,
        "eps": 0.2, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.3
    }

    welfare_score = score_hypothesis_welfare(h, phi_metrics)

    # Should score higher than keyword version (~0)
    assert welfare_score > 0.3, "Should detect empathy+truth relevance"

def test_constitutional_warmup_improved():
    """More examples pass welfare filter."""
    text = """
    Testimonial injustice occurs when prejudice causes a hearer
    to give a deflated level of credibility to a speaker's word.
    """

    phi_metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                  "eps": 0.4, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.3}

    # With semantic scoring, should pass (high truth relevance)
    assert should_include_example(text, phi_metrics, threshold=0.3)
```

### Validation Metrics

```python
# scripts/evaluate_welfare_classifier.py

def evaluate_model_quality():
    """
    Comprehensive evaluation on held-out test set.

    Metrics:
    - MAE per construct (target: < 0.15)
    - Agreement with Claude (target: > 0.85)
    - Academic text catch rate (vs. keyword baseline)
    """
    # ... implementation in scripts/evaluate_welfare_classifier.py
```

---

## Success Criteria

**Must Have:**
- ✅ MAE < 0.20 per construct (vs. Claude ground truth)
- ✅ Agreement with Claude > 85% (threshold-based)
- ✅ All 296 existing tests pass unchanged
- ✅ Constitutional warmup generates > 10 preference pairs from smiles_and_cries

**Should Have:**
- ✅ Welfare relevance detection: 30-40% of academic texts (vs. 6.4%)
- ✅ Inference speed: < 100ms per chunk
- ✅ Paternalism detection: correctly identifies high c+λ_P, low λ_L

**Nice to Have:**
- ✅ MAE < 0.15 per construct
- ✅ GPU inference support (automatic if available)
- ✅ Per-construct confidence scores

---

## Implementation Roadmap

### Phase 1: Data Creation (~1 day)
1. Create `scripts/create_welfare_training_data.py`
2. Configure Claude API access (Azure Foundry)
3. Run labeling (20 minutes, $10 cost)
4. Validate output (1000 examples, all 8 constructs)

### Phase 2: Training (~1 day)
1. Create `scripts/train_welfare_classifier.py`
2. Implement WelfareDataset
3. Train on L8s_v3 (2-3 hours)
4. Evaluate on validation set
5. Save model to `models/welfare-constructs-distilbert/`

### Phase 3: Integration (~1 day)
1. Create `src/inference/welfare_classifier.py`
2. Update `src/inference/welfare_scoring.py` (minimal changes)
3. Write unit tests
4. Write integration tests
5. Run full test suite (296 tests)

### Phase 4: Validation (~0.5 days)
1. Create `scripts/evaluate_welfare_classifier.py`
2. Run on "smiles and cries" corpus
3. Compare keyword vs. semantic detection rates
4. Document performance metrics

**Total Estimate:** 3.5 days, $10 cost

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Claude labeling quality poor | High | Low | Manual review of 50 samples before full run |
| Training fails (OOM) | Medium | Medium | Reduce batch size, use gradient accumulation |
| Model doesn't learn (MAE > 0.30) | High | Low | Add more training examples, tune hyperparameters |
| Inference too slow (> 200ms) | Low | Low | GPU acceleration, model quantization |
| Backward compatibility breaks | High | Very Low | Comprehensive test suite guards this |

---

## Alternative Considered: GPT-4 Labeling

**Why not GPT-4 instead of Claude?**

- Claude has better philosophical understanding (trained on hooks, Collins, Fricker)
- Azure Foundry already configured
- Claude is faster (lower latency)
- Cost similar (~$10 either way)

**Decision:** Use Claude for labeling, could switch to GPT-4 if Claude labels are poor quality.

---

## References

- bell hooks (2000). *All About Love: New Visions*
- Collins, P. H. (1990). *Black Feminist Thought*
- Fricker, M. (2007). *Epistemic Injustice*
- Washington, H. A. (2006). *Medical Apartheid*
- DistilBERT paper: Sanh et al. (2019)
- Multi-label classification: Zhang & Zhou (2014)

---

## Appendix: File Structure

```
detective-llm/
├── src/
│   └── inference/
│       ├── welfare_scoring.py          # Existing (minor updates)
│       └── welfare_classifier.py       # NEW
├── scripts/
│   ├── create_welfare_training_data.py # NEW
│   ├── train_welfare_classifier.py     # NEW
│   └── evaluate_welfare_classifier.py  # NEW
├── tests/
│   └── inference/
│       ├── test_welfare_classifier.py          # NEW
│       └── test_welfare_integration_semantic.py # NEW
├── models/
│   └── welfare-constructs-distilbert/  # Generated by training
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer/
└── data/training/
    ├── welfare_training_data.jsonl     # Generated by Phase 1
    ├── welfare_training_split_train.jsonl
    └── welfare_training_split_val.jsonl
```

---

**Approved by:** User (2026-02-19)
**Next Step:** Invoke writing-plans skill to create implementation plan
