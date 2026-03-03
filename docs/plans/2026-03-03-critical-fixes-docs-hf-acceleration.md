# Critical Fixes, Documentation, and HF Training Acceleration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 5 critical/high issues blocking production, update all stale documentation, and port DPO training to HF Jobs for GPU-accelerated training.

**Architecture:** 3 execution batches. Batch A (Tasks 1-5) fixes code bugs and formula divergence. Batch B (Tasks 6-8) updates documentation. Batch C (Tasks 9-11) builds HF training infrastructure and publishes the paper.

**Tech Stack:** Python 3.12, PyTorch, transformers, trl, huggingface_hub, FastAPI, pytest.

---

## Batch A: Critical Code Fixes (Tasks 1-5)

### Task 1: Sync Phi Formula to v2.1 in welfare_scoring.py

The detective's `compute_phi()` is at v2.0 (no recovery floors, no derivatives). The Space's `welfare.py` is at v2.1. This is the #1 issue.

**Files:**
- Modify: `src/inference/welfare_scoring.py:246-277`
- Test: `tests/inference/test_welfare_scoring.py`

**Step 1: Write the failing test**

```python
# Add to tests/inference/test_welfare_scoring.py

def test_compute_phi_accepts_derivatives():
    """v2.1: compute_phi should accept optional derivatives parameter."""
    from src.inference.welfare_scoring import compute_phi
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}
    derivs = {"c": 0.01, "lam_L": -0.02}
    # Should not raise
    result = compute_phi(metrics, derivatives=derivs)
    assert 0.0 <= result <= 1.0


def test_compute_phi_recovery_floor_activates():
    """v2.1: below-floor constructs should be lifted by recovery potential."""
    from src.inference.welfare_scoring import compute_phi, CONSTRUCT_FLOORS
    # Care below its floor (0.20) with positive trajectory
    metrics = {"c": 0.10, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}
    derivs = {"c": 0.05}  # recovering
    phi_recovering = compute_phi(metrics, derivatives=derivs)

    # Same metrics, no recovery (stagnant)
    phi_stagnant = compute_phi(metrics)

    # Recovery should produce higher Phi (community + trajectory lift the floor)
    assert phi_recovering > phi_stagnant


def test_compute_phi_backward_compatible():
    """v2.1: no derivatives = same as v2.0 when all above floor."""
    from src.inference.welfare_scoring import compute_phi
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}
    # All above floor, no derivatives → recovery_aware_input is pass-through
    result = compute_phi(metrics)
    assert 0.0 < result < 1.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/inference/test_welfare_scoring.py::test_compute_phi_accepts_derivatives -v`
Expected: FAIL (TypeError: compute_phi() got unexpected keyword argument 'derivatives')

**Step 3: Update compute_phi to v2.1**

Replace `welfare_scoring.py:246-277` with the v2.1 implementation from the Space's `welfare.py:168-221`:

```python
def compute_phi(
    metrics: Dict[str, float],
    derivatives: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute Phi(humanity) — the full welfare function (v2.1).

    Phi = f(lam_L) * product(x_tilde_i ^ w_i) * Psi_ubuntu * (1 - Psi_penalty)

    v2.1: recovery_aware_input() is called for each construct before the
    weighted geometric mean. Synergy and penalty still operate on raw
    metrics (they detect actual state).

    Args:
        metrics: Dict mapping each construct symbol to a value in [0, 1].
        derivatives: Optional dict of dx/dt per construct. Defaults to 0.0.
    """
    if derivatives is None:
        derivatives = {}

    lam_L_raw = max(0.01, metrics.get("lam_L", 0.5))
    f_lam = community_multiplier(lam_L_raw)

    # Recovery-aware effective values
    effective: Dict[str, float] = {}
    for c in ALL_CONSTRUCTS:
        x_raw = max(0.01, metrics.get(c, 0.5))
        floor_c = CONSTRUCT_FLOORS[c]
        dx_dt_c = derivatives.get(c, 0.0)
        effective[c] = recovery_aware_input(x_raw, floor_c, dx_dt_c, lam_L_raw)

    # Equity weights on effective values
    weights = equity_weights(effective)

    # Weighted geometric mean of effective values
    product = 1.0
    for c in ALL_CONSTRUCTS:
        x_eff = max(0.01, effective[c])
        product *= x_eff ** weights[c]

    # Synergy and penalty on RAW metrics
    synergy = ubuntu_synergy(metrics)
    penalty = divergence_penalty(metrics)

    phi = f_lam * product * synergy * (1.0 - penalty)
    return max(0.0, phi)
```

Add `from typing import Optional` to imports if not already present.

**Step 4: Run tests**

Run: `pytest tests/inference/test_welfare_scoring.py -v`
Expected: All pass including 3 new tests

**Step 5: Run full suite**

Run: `pytest tests/ -q --deselect tests/detective/test_parallel_evolution.py::test_welfare_scoring_applied_to_evolved_hypotheses --deselect tests/detective/test_parallel_evolution.py::test_backward_compatible_without_phi_metrics --deselect tests/detective/test_parallel_evolution.py::test_high_welfare_relevance_for_urgent_findings --deselect tests/detective/test_parallel_evolution.py::test_welfare_aware_sorting`
Expected: 0 new failures

**Step 6: Commit**

```bash
git add src/inference/welfare_scoring.py tests/inference/test_welfare_scoring.py
git commit -m "fix(welfare): upgrade compute_phi to v2.1 — recovery floors + derivatives

Syncs the inference layer with the Space's welfare.py. Below-floor
constructs now receive community-mediated recovery potential.
Equity weights computed on effective (recovery-adjusted) values.
Backward compatible: no derivatives = pass-through when above floor."
```

---

### Task 2: Fix Module A Classifier (Provider-Based Scoring)

Module A uses `distilbert-base-uncased` which outputs `POSITIVE/NEGATIVE`, not `cognitive_bias`. Rewrite to use the `ModelProvider` pattern like Modules B and C.

**Files:**
- Modify: `src/detective/module_a.py`
- Modify: `tests/detective/test_module_a.py`

**Step 1: Write the failing test**

```python
# Add to tests/detective/test_module_a.py

def test_detect_cognitive_biases_accepts_provider():
    """Module A should accept a ModelProvider like B and C."""
    from src.detective.module_a import detect_cognitive_biases
    from src.core.providers import MockProvider
    provider = MockProvider(response="cognitive_bias, score: 0.85")
    results = detect_cognitive_biases(
        "Survivors recall only the successful outcomes, ignoring failures.",
        provider=provider,
    )
    assert isinstance(results, list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/detective/test_module_a.py::test_detect_cognitive_biases_accepts_provider -v`
Expected: FAIL (TypeError: unexpected keyword argument 'provider')

**Step 3: Rewrite module_a.py**

Replace the DistilBERT classifier approach with a provider-based scorer matching Modules B and C:

```python
"""Cognitive bias detection — Module A.

Uses regex triggers for known bias patterns + LLM-scored confirmation,
matching the architecture of Modules B and C.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.types import AssumptionType
from src.core.providers import ModelProvider

_CONFIDENCE_THRESHOLD: float = 0.5

# Bias pattern triggers
_BIAS_PATTERNS: dict[str, re.Pattern] = {
    "confirmation": re.compile(
        r"confirm|consistent with|supports? (the|our|my) (view|hypothesis|belief)",
        re.IGNORECASE,
    ),
    "anchoring": re.compile(
        r"initial|first (report|estimate)|anchor|starting point|baseline assumption",
        re.IGNORECASE,
    ),
    "survivorship": re.compile(
        r"surviv|success stor|only (the|those) (who|that) (made|succeeded)|ignoring fail",
        re.IGNORECASE,
    ),
    "ingroup": re.compile(
        r"our (group|side|team)|they (always|never)|us vs\.? them|in-?group|out-?group",
        re.IGNORECASE,
    ),
}

_SCORE_RE = re.compile(r"(?:score|confidence)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

_SCORING_PROMPT = (
    "Rate the cognitive bias in the following text on a scale of 0.0 to 1.0.\n"
    "Bias type: {bias_type}\n"
    "Text: {text}\n\n"
    "Reply with: score: <float between 0 and 1>"
)


@dataclass(frozen=True)
class BiasDetection:
    """A detected cognitive bias."""
    assumption_type: AssumptionType
    score: float
    source_text: str
    bias_type: str  # which bias pattern triggered

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"BiasDetection.score must be in [0.0, 1.0], got {self.score!r}")


def _parse_score(response: str, default: float = 0.0) -> float:
    match = _SCORE_RE.search(response)
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    return default


def detect_cognitive_biases(
    text: str,
    provider: ModelProvider | None = None,
    threshold: float = _CONFIDENCE_THRESHOLD,
) -> list[BiasDetection]:
    """Detect cognitive biases via regex triggers + optional LLM scoring.

    When provider is None, returns detections with score=1.0 for any
    matched pattern (keyword-only mode). When provider is given, each
    trigger is scored by the LLM.
    """
    detections: list[BiasDetection] = []

    for bias_type, pattern in _BIAS_PATTERNS.items():
        if not pattern.search(text):
            continue

        if provider is not None:
            prompt = _SCORING_PROMPT.format(bias_type=bias_type, text=text[:500])
            response = provider.complete(prompt)
            score = _parse_score(response)
        else:
            score = 1.0  # keyword match without LLM confirmation

        if score >= threshold:
            detections.append(BiasDetection(
                assumption_type=AssumptionType.COGNITIVE_BIAS,
                score=score,
                source_text=text,
                bias_type=bias_type,
            ))

    return detections
```

**Step 4: Run tests**

Run: `pytest tests/detective/test_module_a.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/detective/module_a.py tests/detective/test_module_a.py
git commit -m "fix(module_a): rewrite as provider-based scorer matching B and C

Replaces broken DistilBERT classifier (wrong labels) with regex
trigger + LLM scoring pattern. Accepts optional ModelProvider.
Keyword-only mode when provider=None."
```

---

### Task 3: Fix AsyncIO Anti-Pattern in /evolve Endpoint

**Files:**
- Modify: `src/api/routes.py:196-207`
- Test: `tests/api/test_routes.py`

**Step 1: Fix the endpoint**

Change the `/evolve` handler from sync to async:

```python
# In routes.py, change:
#   @app.post("/evolve", response_model=EvolveResponse)
#   def evolve(request: EvolveRequest) -> EvolveResponse:
#       ...
#       results = asyncio.run(evolve_parallel(...))

# To:
    @app.post("/evolve", response_model=EvolveResponse)
    async def evolve(request: EvolveRequest) -> EvolveResponse:
        ...
        from src.detective.parallel_evolution import evolve_parallel
        results = await evolve_parallel(
            hypothesis=base,
            evidence_list=[request.evidence_path],
            provider=_provider,
            k=1,
            library=EMPTY_LIBRARY,
            phi_metrics=request.phi_metrics,
        )
```

Remove the `import asyncio` line from inside the function.

**Step 2: Run tests**

Run: `pytest tests/api/test_routes.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add src/api/routes.py
git commit -m "fix(api): make /evolve endpoint async to avoid nested asyncio.run"
```

---

### Task 4: Remove Dead alpha/beta Parameters from evolve_parallel

**Files:**
- Modify: `src/detective/parallel_evolution.py:99-108`

**Step 1: Remove dead parameters**

In `evolve_parallel()`, remove the `alpha` and `beta` parameters from the function signature. They are never used — the function hardcodes the 4-weight sort at line 172.

Change:
```python
async def evolve_parallel(
    hypothesis: Hypothesis,
    evidence_list: list[str],
    provider: ModelProvider,
    k: int = 3,
    library: ExperienceLibrary = (),
    phi_metrics: dict[str, float] | None = None,
    alpha: float = 0.7,  # REMOVE
    beta: float = 0.3,   # REMOVE
) -> list[ParallelEvolutionResult]:
```

To:
```python
async def evolve_parallel(
    hypothesis: Hypothesis,
    evidence_list: list[str],
    provider: ModelProvider,
    k: int = 3,
    library: ExperienceLibrary = (),
    phi_metrics: dict[str, float] | None = None,
) -> list[ParallelEvolutionResult]:
```

Update the docstring to remove alpha/beta Args entries.

**Step 2: Search for callers and fix**

Run: `grep -rn "evolve_parallel.*alpha\|evolve_parallel.*beta" src/ tests/`
Fix any callers that pass alpha/beta.

**Step 3: Run tests**

Run: `pytest tests/ -q` (with deselects)
Expected: All pass

**Step 4: Commit**

```bash
git add src/detective/parallel_evolution.py
git commit -m "refactor(evolution): remove dead alpha/beta params from evolve_parallel

These parameters were never used — the function hardcodes 4-weight
combined_score(0.45, 0.25, 0.15, 0.15) when phi_metrics is provided."
```

---

### Task 5: Wire Welfare Scoring into analyze() Pipeline

**Files:**
- Modify: `src/inference/pipeline.py`
- Test: `tests/inference/test_pipeline.py`

**Step 1: Write the failing test**

```python
def test_analyze_includes_welfare_scoring():
    """analyze() should return welfare-scored gaps when phi_metrics provided."""
    from src.inference.pipeline import analyze
    # ... test that gaps in result carry welfare_impact > 0
```

**Step 2: Wire score_gaps_welfare into analyze()**

In `pipeline.py:analyze()`, after gap detection, call `score_gaps_welfare()` on the detected gaps when phi_metrics is available. This connects the welfare scoring infrastructure to the main pipeline.

**Step 3: Run tests and commit**

```bash
git commit -m "feat(pipeline): wire welfare scoring into analyze() for gap prioritization"
```

---

## Batch B: Documentation Updates (Tasks 6-8)

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Changes:**
1. Update test count: "531+" → "561+"
2. Update ADR range: "ADR-011" → "ADR-012"
3. Add `src/security/`, `src/forecasting/`, `src/data/sourcing/` to architecture diagram
4. Add `constitution.py` and `kuzu_graph.py` to implementation status
5. Note the v2.1 formula version explicitly

**Commit:** `docs: update CLAUDE.md — architecture diagram, test count, formula version`

---

### Task 7: Fix INTEGRATION_SUMMARY.md and Stale ADRs

**Files:**
- Modify: `INTEGRATION_SUMMARY.md` (add SUPERSEDED header)
- Modify: `docs/vault/decisions/ADR-002-moral-compass-constitution.md` (planned → implemented)
- Modify: `docs/vault/decisions/ADR-003-distilbert-classifiers.md` (planned → implemented)
- Modify: `docs/vault/decisions/ADR-004-experience-library.md` (planned → implemented)
- Modify: `docs/vault/decisions/ADR-007-legal-grounding.md` (remove "to be added")
- Modify: `docs/vault/decisions/ADR-008-persistent-graph-store.md` (proposed → accepted)

**Changes to INTEGRATION_SUMMARY.md:**
Add at top:
```markdown
> **SUPERSEDED**: This document describes the v1.0 Phi integration state from 2026-02-19.
> The current state uses v2.1 with 8 constructs, 4-weight scoring, and recovery floors.
> See CLAUDE.md and ADR-009 through ADR-012 for current architecture.
```

**Commit:** `docs: fix stale ADRs and deprecate INTEGRATION_SUMMARY.md`

---

### Task 8: Update Phi Paper Reference Implementation

**Files:**
- Modify: `docs/humanity-phi-formalized.md` (Section 5.1, lines ~474-549)

**Changes:**
1. Replace v1.0 reference implementation with v2.1 code from `welfare_scoring.py`
2. Fix suggested citation version: "v2.0" → "v2.1"
3. Fix repository placeholder: `[repository]` → `https://github.com/crichalchemist/wave-experiment`
4. Fix "Bostom" → "Bostrom" typo in bibliography

**Commit:** `docs(phi): update reference implementation to v2.1 — recovery floors, equity weights`

---

## Batch C: HF Training Acceleration (Tasks 9-11)

### Task 9: Upload Preference Pairs to Hub

**Files:**
- No code changes — operational

**Step 1: Upload constitutional pairs**

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("crichalchemist/detective-llm-dpo-data", repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj="data/training/constitutional_pairs.jsonl",
    path_in_repo="constitutional_pairs.jsonl",
    repo_id="crichalchemist/detective-llm-dpo-data", repo_type="dataset",
)
api.upload_file(
    path_or_fileobj="data/training/legal_pairs.jsonl",
    path_in_repo="legal_pairs.jsonl",
    repo_id="crichalchemist/detective-llm-dpo-data", repo_type="dataset",
)
```

**Step 2: Verify**

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().list_repo_files('crichalchemist/detective-llm-dpo-data', repo_type='dataset'))"
```

---

### Task 10: Write DPO HF Jobs Training Script

**Files:**
- Create: `scripts/train_dpo_hf_job.py`
- Create: `tests/scripts/test_train_dpo_hf_job.py`

Following the exact pattern of `scripts/train_welfare_classifier_hf_job.py`:
- `get_training_script(epochs, lr, batch_size, beta, lora_rank, data_source)` → PEP 723 self-contained script
- `launch_dpo_training(...)` → `run_uv_job()` with flavor + secrets
- Script downloads data from Hub, configures LoRA (r=16, α=32), trains with bf16 on GPU
- Pushes adapter to `crichalchemist/detective-llm-dpo-adapter`
- Logs via Trackio

**Tests:**
```python
def test_dpo_script_valid_python():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4)
    compile(script, "<dpo_job>", "exec")

def test_dpo_script_has_pep723():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4)
    assert "# /// script" in script

def test_launch_function_exists():
    from scripts.train_dpo_hf_job import launch_dpo_training
    assert callable(launch_dpo_training)
```

**Commit:** `feat(scripts): add HF Jobs DPO training script for GPU-accelerated training`

---

### Task 11: Publish Phi Paper on HF Hub

**Files:**
- No code changes — operational

**Step 1: Publish paper**

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("crichalchemist/phi-humanity-welfare-function", exist_ok=True)
api.upload_file(
    path_or_fileobj="docs/humanity-phi-formalized.md",
    path_in_repo="README.md",
    repo_id="crichalchemist/phi-humanity-welfare-function",
)
```

**Step 2: Link to related repos**

Add model card references linking the paper to:
- `crichalchemist/welfare-constructs-distilbert` (classifier)
- `crichalchemist/maninagarden` (Space)
- `crichalchemist/phi-forecaster` (forecaster model)

---

## Execution Batches

| Batch | Tasks | Scope | Parallel? |
|-------|-------|-------|-----------|
| **A** | 1-5 | Critical code fixes | Tasks 1-4 parallelizable; Task 5 depends on 1 |
| **B** | 6-8 | Documentation | All parallelizable |
| **C** | 9-11 | HF infrastructure | Task 10 is code; 9 and 11 are operational |

**Estimated time:** Batch A: 2-3 hours. Batch B: 1 hour. Batch C: 2-3 hours.
**Estimated HF cost:** ~$0.35 for smoke test + production DPO run.
