# Detective-Forecaster Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect Detective LLM and Phi Forecaster through a three-layer pipeline — welfare classifier training (Layer 1), forecast-informed hypothesis scoring (Layer 2), and detective-generated scenario extraction (Layer 3) — creating a data flywheel between investigation and prediction.

**Architecture:** Three sequential layers, each independently useful. Layer 1 trains a DistilBERT welfare classifier on HF Jobs, replacing keyword fallback with semantic understanding. Layer 2 adds trajectory urgency to hypothesis scoring by running the forecaster on detected construct levels. Layer 3 extracts new training scenarios from real text via A/B/C modules, enriching the forecaster.

**Tech Stack:** PyTorch, transformers (DistilBERT), HF Jobs (t4-small), Gradio, src/forecasting (PhiTrajectoryForecaster), src/detective modules A/B/C, pandas/numpy.

**Design doc:** `docs/plans/2026-02-26-bridge-detective-forecaster-design.md`

---

## Layer 1: Welfare Classifier (HF Jobs Training + Hub Loading)

### Task 1: HF Jobs Training Script for Welfare Classifier

The existing `scripts/train_welfare_classifier.py` runs locally. We need a self-contained HF Jobs version that pushes to the Hub.

**Files:**
- Create: `scripts/train_welfare_classifier_hf_job.py`
- Read: `scripts/train_welfare_classifier.py` (reference for architecture)
- Read: `spaces/maninagarden/training.py` (reference for HF Jobs pattern)
- Read: `data/training/welfare_training_split_train.jsonl` (data format)

**Step 1: Write the failing test**

```python
# tests/scripts/test_train_welfare_classifier_hf_job.py
"""Test welfare classifier HF Jobs script generation."""
import pytest


def test_script_generation_returns_valid_python():
    from scripts.train_welfare_classifier_hf_job import get_training_script
    script = get_training_script(epochs=1, lr=2e-5, batch_size=16)
    assert "distilbert-base-uncased" in script
    assert "welfare-constructs-distilbert" in script
    compile(script, "<hf_job>", "exec")  # valid Python


def test_script_includes_pep723_dependencies():
    from scripts.train_welfare_classifier_hf_job import get_training_script
    script = get_training_script(epochs=1, lr=2e-5, batch_size=16)
    assert "# /// script" in script
    assert "torch" in script
    assert "transformers" in script
    assert "datasets" in script


def test_script_includes_data_download():
    """The script must download training data from the Hub, not assume local files."""
    from scripts.train_welfare_classifier_hf_job import get_training_script
    script = get_training_script(epochs=1, lr=2e-5, batch_size=16)
    assert "huggingface_hub" in script or "hf_hub_download" in script


def test_script_pushes_to_hub():
    from scripts.train_welfare_classifier_hf_job import get_training_script
    script = get_training_script(epochs=1, lr=2e-5, batch_size=16)
    assert "push_to_hub" in script or "upload" in script


def test_launch_function_exists():
    from scripts.train_welfare_classifier_hf_job import launch_classifier_training
    assert callable(launch_classifier_training)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_train_welfare_classifier_hf_job.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the HF Jobs training script**

Create `scripts/train_welfare_classifier_hf_job.py`. This follows the same pattern as `spaces/maninagarden/training.py` — a `get_training_script()` function that returns a self-contained Python string with PEP 723 dependencies, and a `launch_classifier_training()` function that calls `huggingface_hub.run_uv_job()`.

The generated script must:
1. Download `welfare_training_split_train.jsonl` and `welfare_training_split_val.jsonl` from the Hub (upload these to a dataset repo first, or embed them)
2. Fine-tune `distilbert-base-uncased` with `num_labels=8, problem_type="multi_label_classification"`
3. Use `Trainer` with: 3 epochs, lr=2e-5, batch=16, FP16, eval each epoch, load best model
4. Compute per-construct MAE via `compute_metrics`
5. Push final model to `crichalchemist/welfare-constructs-distilbert` on the Hub
6. Log metrics via Trackio

Key details from existing `scripts/train_welfare_classifier.py`:
- `WelfareDataset.__getitem__` returns `{'input_ids', 'attention_mask', 'labels'}` where labels is `torch.tensor([scores[c] for c in CONSTRUCT_NAMES], dtype=torch.float)`
- `compute_metrics` applies `torch.sigmoid` to logits, then computes `mean_absolute_error`
- CONSTRUCT_NAMES = `["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]`

**Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_train_welfare_classifier_hf_job.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/train_welfare_classifier_hf_job.py tests/scripts/test_train_welfare_classifier_hf_job.py
git commit -m "feat(bridge): add HF Jobs training script for welfare classifier"
```

---

### Task 2: Update Welfare Classifier to Load from Hub

Currently `src/inference/welfare_classifier.py` loads from `models/welfare-constructs-distilbert/` (local path). Update it to load from the Hub with local fallback.

**Files:**
- Modify: `src/inference/welfare_classifier.py`
- Modify: `tests/inference/test_welfare_classifier.py` (or create if missing)

**Step 1: Write the failing test**

```python
# tests/inference/test_welfare_classifier.py
"""Test welfare classifier Hub loading and fallback."""
import pytest
from unittest.mock import patch, MagicMock


def test_hub_model_id_is_configured():
    """Classifier should have a Hub model ID configured."""
    from src.inference.welfare_classifier import HUB_MODEL_ID
    assert HUB_MODEL_ID == "crichalchemist/welfare-constructs-distilbert"


def test_load_tries_hub_first(monkeypatch):
    """Loading should try Hub before local path."""
    import src.inference.welfare_classifier as wc
    wc._load_welfare_classifier.cache_clear()

    mock_pipeline = MagicMock(return_value=[
        [{"label": f"LABEL_{i}", "score": 0.5} for i in range(8)]
    ])

    with patch("src.inference.welfare_classifier.pipeline", mock_pipeline):
        wc._load_welfare_classifier()
        # Should have been called with Hub model ID
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args
        assert "crichalchemist/welfare-constructs-distilbert" in str(call_kwargs)

    wc._load_welfare_classifier.cache_clear()


def test_fallback_to_local_when_hub_unavailable(monkeypatch):
    """If Hub is unreachable, try local path."""
    import src.inference.welfare_classifier as wc
    wc._load_welfare_classifier.cache_clear()

    call_count = 0

    def mock_pipeline_fn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("Hub unreachable")
        return MagicMock()

    with patch("src.inference.welfare_classifier.pipeline", mock_pipeline_fn):
        with patch.object(wc.Path, "exists", return_value=True):
            try:
                wc._load_welfare_classifier()
            except Exception:
                pass  # May fail if local also missing; we just test the fallback attempt
            assert call_count >= 2  # Tried Hub then local

    wc._load_welfare_classifier.cache_clear()


def test_get_construct_scores_returns_zeros_when_no_model():
    """Without model, should return zeros for all 8 constructs."""
    import src.inference.welfare_classifier as wc
    wc._load_welfare_classifier.cache_clear()

    scores = wc.get_construct_scores("test text")
    assert len(scores) == 8
    assert all(v == 0.0 for v in scores.values())
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/inference/test_welfare_classifier.py -v`
Expected: FAIL on `HUB_MODEL_ID` import

**Step 3: Update welfare_classifier.py**

In `src/inference/welfare_classifier.py`:
- Add `HUB_MODEL_ID = "crichalchemist/welfare-constructs-distilbert"`
- Update `_load_welfare_classifier()` to try `pipeline("text-classification", model=HUB_MODEL_ID, ...)` first
- On `OSError` / `HTTPError`, fall back to local `MODEL_PATH`
- Keep existing graceful fallback to zero scores if both fail

```python
HUB_MODEL_ID = "crichalchemist/welfare-constructs-distilbert"

@lru_cache(maxsize=1)
def _load_welfare_classifier():
    # Try Hub first
    try:
        logger.info(f"Loading welfare classifier from Hub: {HUB_MODEL_ID}...")
        return pipeline(
            "text-classification", model=HUB_MODEL_ID,
            device=0 if torch.cuda.is_available() else -1, top_k=None,
        )
    except (OSError, Exception) as e:
        logger.debug(f"Hub loading failed: {e}")

    # Fallback to local
    config_file = MODEL_PATH / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Welfare classifier not found at Hub ({HUB_MODEL_ID}) "
            f"or local ({MODEL_PATH}). Train with scripts/train_welfare_classifier_hf_job.py."
        )
    logger.info(f"Loading welfare classifier from local: {MODEL_PATH}...")
    return pipeline(
        "text-classification", model=str(MODEL_PATH),
        device=0 if torch.cuda.is_available() else -1, top_k=None,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/inference/test_welfare_classifier.py -v`
Expected: PASS

**Step 5: Run full test suite for regressions**

Run: `pytest tests/ -x -q`
Expected: 291+ tests pass, 0 failures

**Step 6: Commit**

```bash
git add src/inference/welfare_classifier.py tests/inference/test_welfare_classifier.py
git commit -m "feat(bridge): welfare classifier loads from Hub with local fallback"
```

---

### Task 3: Upload Training Data to Hub Dataset Repo

The HF Jobs script needs data on the Hub. Upload the welfare training splits.

**Files:**
- Read: `data/training/welfare_training_split_train.jsonl`
- Read: `data/training/welfare_training_split_val.jsonl`

**Step 1: Create Hub dataset repo and upload**

```bash
# Create dataset repo
huggingface-cli repo create crichalchemist/welfare-training-data --type dataset 2>/dev/null || true

# Upload training splits
huggingface-cli upload crichalchemist/welfare-training-data \
    data/training/welfare_training_split_train.jsonl \
    welfare_training_split_train.jsonl --repo-type dataset

huggingface-cli upload crichalchemist/welfare-training-data \
    data/training/welfare_training_split_val.jsonl \
    welfare_training_split_val.jsonl --repo-type dataset
```

**Step 2: Verify upload**

```bash
huggingface-cli download crichalchemist/welfare-training-data welfare_training_split_train.jsonl --repo-type dataset --local-dir /tmp/verify
wc -l /tmp/verify/welfare_training_split_train.jsonl
# Expected: 791
```

**Step 3: Commit** (no code changes, just verify)

---

### Task 4: Launch Welfare Classifier Training on HF Jobs

**Step 1: Launch training**

```python
from scripts.train_welfare_classifier_hf_job import launch_classifier_training
job_id, msg = launch_classifier_training(epochs=3, lr=2e-5, batch_size=16, hardware="t4-small")
print(f"Job: {job_id}\n{msg}")
```

Or via CLI:
```bash
python -c "
from scripts.train_welfare_classifier_hf_job import launch_classifier_training
job_id, msg = launch_classifier_training(epochs=3, lr=2e-5, batch_size=16)
print(f'Job: {job_id}')
print(msg)
"
```

**Step 2: Monitor training**

```bash
# Check job status
python -c "from huggingface_hub import get_job; print(get_job('<job_id>').status)"
```

**Step 3: Verify model on Hub**

```bash
# After job completes:
python -c "
from transformers import pipeline
clf = pipeline('text-classification', model='crichalchemist/welfare-constructs-distilbert', top_k=None)
result = clf('Resource allocation gap in funding for basic needs')
print(result)
"
```

**Success criterion:** Per-construct MAE < 0.20 on val set. If not met, stop — more training data needed before Layer 2.

---

## Layer 2: Forecast-Informed Hypothesis Scoring

### Task 5: Add `trajectory_urgency` Field to Hypothesis

**Files:**
- Modify: `src/detective/hypothesis.py`
- Modify: `tests/test_hypothesis.py`

**Step 1: Write the failing test**

```python
# In tests/test_hypothesis.py — add these tests

def test_hypothesis_has_trajectory_urgency():
    from src.detective.hypothesis import Hypothesis
    h = Hypothesis.create("Test", 0.8)
    assert h.trajectory_urgency == 0.0


def test_trajectory_urgency_validation():
    from src.detective.hypothesis import Hypothesis
    with pytest.raises(ValueError, match="trajectory_urgency"):
        Hypothesis(
            id="x", text="t", confidence=0.5,
            timestamp=datetime.now(), trajectory_urgency=-0.1,
        )


def test_combined_score_with_trajectory_urgency():
    """combined_score with delta parameter for trajectory urgency."""
    from src.detective.hypothesis import Hypothesis
    from dataclasses import replace
    h = Hypothesis.create("Test", 0.8)
    h = replace(h, welfare_relevance=0.6, curiosity_relevance=0.4, trajectory_urgency=1.0)

    # New weights: alpha=0.45, beta=0.25, gamma=0.15, delta=0.15
    score = h.combined_score(alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)
    expected = 0.45 * 0.8 + 0.25 * 0.6 + 0.15 * 0.4 + 0.15 * 1.0
    assert abs(score - expected) < 1e-9


def test_combined_score_backward_compatible():
    """Default delta=0.0 means old 3-parameter behavior is preserved."""
    from src.detective.hypothesis import Hypothesis
    h = Hypothesis.create("Test", 0.8)
    # Default delta=0.0 should give same result as before
    score_new = h.combined_score(alpha=0.55, beta=0.30, gamma=0.15, delta=0.0)
    score_old = 0.55 * 0.8 + 0.30 * 0.0 + 0.15 * 0.0
    assert abs(score_new - score_old) < 1e-9
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis.py::test_hypothesis_has_trajectory_urgency -v`
Expected: FAIL with `AttributeError`

**Step 3: Update hypothesis.py**

In `src/detective/hypothesis.py`:
- Add field: `trajectory_urgency: float = 0.0` after `curiosity_relevance`
- Add validation in `__post_init__`: `0.0 <= self.trajectory_urgency <= 1.0`
- Update `combined_score()` signature: add `delta: float = 0.0`
- Update formula: `alpha * confidence + beta * welfare + gamma * curiosity + delta * trajectory_urgency`

```python
# New field (line ~24, after curiosity_relevance):
trajectory_urgency: float = 0.0  # [0, 1] — forecast-informed urgency

# New validation (in __post_init__):
if not (0.0 <= self.trajectory_urgency <= 1.0):
    raise ValueError(
        f"Hypothesis.trajectory_urgency must be in [0.0, 1.0], got {self.trajectory_urgency!r}"
    )

# Updated combined_score:
def combined_score(
    self,
    alpha: float = 0.55,
    beta: float = 0.30,
    gamma: float = 0.15,
    delta: float = 0.0,
) -> float:
    return (
        alpha * self.confidence
        + beta * self.welfare_relevance
        + gamma * self.curiosity_relevance
        + delta * self.trajectory_urgency
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hypothesis.py -v`
Expected: PASS (all existing + new tests)

**Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass. The default `delta=0.0` ensures backward compatibility.

**Step 6: Commit**

```bash
git add src/detective/hypothesis.py tests/test_hypothesis.py
git commit -m "feat(bridge): add trajectory_urgency to Hypothesis with delta parameter"
```

---

### Task 6: Implement `score_hypothesis_trajectory()` in welfare_scoring.py

This is the core bridge function: given a hypothesis and Phi metrics, use the forecaster to predict trajectory slope and normalize to [0,1] urgency.

**Files:**
- Modify: `src/inference/welfare_scoring.py`
- Create: `tests/inference/test_trajectory_scoring.py`

**Step 1: Write the failing test**

```python
# tests/inference/test_trajectory_scoring.py
"""Test forecast-informed trajectory urgency scoring."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_score_hypothesis_trajectory_returns_float():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Resource allocation gap 2013-2017", 0.8)
    metrics = {"c": 0.3, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    # Mock the forecaster to avoid loading real model
    mock_predictions = np.array([0.5, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_predictions):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert isinstance(urgency, float)
    assert 0.0 <= urgency <= 1.0


def test_declining_trajectory_gives_high_urgency():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Truth suppression in oversight records", 0.8)
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.2}

    # Steeply declining trajectory
    mock_pred = np.array([0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_pred):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert urgency > 0.5  # Declining = urgent


def test_stable_trajectory_gives_low_urgency():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Minor record discrepancy", 0.8)
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    # Flat trajectory
    mock_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_pred):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert urgency < 0.2  # Stable = not urgent


def test_rising_trajectory_gives_zero_urgency():
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import score_hypothesis_trajectory

    h = Hypothesis.create("Recovery underway", 0.8)
    metrics = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    # Rising trajectory
    mock_pred = np.array([0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    with patch("src.inference.welfare_scoring._get_trajectory_prediction", return_value=mock_pred):
        urgency = score_hypothesis_trajectory(h, metrics)

    assert urgency == 0.0  # Rising = no urgency


def test_trajectory_prediction_builds_constant_scenario():
    """_get_trajectory_prediction should build a 200-step constant-level scenario."""
    from src.inference.welfare_scoring import _get_trajectory_prediction

    metrics = {"c": 0.3, "kappa": 0.5, "j": 0.5, "p": 0.5,
               "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    # Mock PhiTrajectoryForecaster
    mock_forecaster = MagicMock()
    mock_forecaster.pred_len = 10
    mock_model = MagicMock()
    mock_forecaster.model = mock_model

    with patch("src.inference.welfare_scoring._get_forecaster", return_value=mock_forecaster):
        with patch("src.inference.welfare_scoring._forecast_from_metrics", return_value=np.ones(10) * 0.5):
            pred = _get_trajectory_prediction(metrics)

    assert len(pred) == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/inference/test_trajectory_scoring.py -v`
Expected: FAIL with `ImportError`

**Step 3: Implement in welfare_scoring.py**

Add to the bottom of `src/inference/welfare_scoring.py`:

```python
# ---------------------------------------------------------------------------
# Layer 2: Forecast-informed trajectory urgency
# ---------------------------------------------------------------------------

_forecaster_cache = None

def _get_forecaster():
    """Lazy-load PhiTrajectoryForecaster (cached singleton)."""
    global _forecaster_cache
    if _forecaster_cache is None:
        from src.forecasting.phi_trajectory import PhiTrajectoryForecaster
        _forecaster_cache = PhiTrajectoryForecaster()
    return _forecaster_cache


def _forecast_from_metrics(
    metrics: Dict[str, float],
    history_len: int = 200,
) -> "np.ndarray":
    """
    Build a constant-level scenario from current metrics and forecast Phi trajectory.

    Creates a 200-step DataFrame where all construct values are held constant
    at their current levels (with tiny noise for signal processing stability),
    then runs the forecaster to predict 10 future Phi values.
    """
    import numpy as np
    import pandas as pd
    import torch

    forecaster = _get_forecaster()
    rng = np.random.default_rng(42)

    # Build constant scenario: each construct stays at current level
    data = {}
    for c in ALL_CONSTRUCTS:
        level = max(0.01, min(1.0, metrics.get(c, 0.5)))
        data[c] = np.full(history_len, level) + rng.normal(0, 0.001, history_len)
        data[c] = np.clip(data[c], 0.0, 1.0)

    df = pd.DataFrame(data)

    # Compute Phi column
    phi_vals = np.array([
        compute_phi({c: df.at[i, c] for c in ALL_CONSTRUCTS})
        for i in range(len(df))
    ])
    df["phi"] = phi_vals

    # Feature engineering via pipeline
    X = forecaster.pipeline.fit_transform(df)
    X_seq = X[np.newaxis, -forecaster.pipeline.seq_len:]  # [1, seq_len, 36]
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        phi_pred = forecaster.model.predict_phi(X_tensor)

    return phi_pred[0, :, 0].numpy()


def _get_trajectory_prediction(metrics: Dict[str, float]) -> "np.ndarray":
    """Get 10-step Phi trajectory prediction from current metrics."""
    return _forecast_from_metrics(metrics)


def score_hypothesis_trajectory(
    hypothesis: "Hypothesis",
    phi_metrics: Dict[str, float],
) -> float:
    """
    Compute trajectory urgency for a hypothesis.

    Runs the Phi forecaster on current construct levels to predict
    whether welfare is declining. Declining trajectories increase urgency.

    urgency = max(0, -slope) / (max(0, -slope) + k)

    Where slope = (phi[-1] - phi[0]) / len(phi), normalized to [0,1].
    Rising or stable trajectories → 0.0 urgency.

    Args:
        hypothesis: Hypothesis to score (not used directly — urgency
                    depends on the overall welfare state, not hypothesis text)
        phi_metrics: Current Phi construct levels

    Returns:
        Trajectory urgency in [0, 1], where 1.0 = steepest decline
    """
    try:
        predictions = _get_trajectory_prediction(phi_metrics)
    except Exception as e:
        logger.debug(f"Trajectory prediction failed: {e}")
        return 0.0

    if len(predictions) < 2:
        return 0.0

    slope = (float(predictions[-1]) - float(predictions[0])) / len(predictions)

    # Only declining trajectories create urgency
    if slope >= 0:
        return 0.0

    decline = -slope  # positive value
    k = 0.02  # normalize: decline of 0.02/step → urgency ~0.5
    urgency = decline / (decline + k)

    return min(1.0, max(0.0, urgency))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/inference/test_trajectory_scoring.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/inference/welfare_scoring.py tests/inference/test_trajectory_scoring.py
git commit -m "feat(bridge): forecast-informed trajectory urgency scoring"
```

---

### Task 7: Wire Trajectory Urgency into parallel_evolution.py

When `phi_metrics` is provided, also compute trajectory urgency and use the 4-weight combined_score.

**Files:**
- Modify: `src/detective/parallel_evolution.py`
- Modify: `tests/detective/test_parallel_evolution.py`

**Step 1: Write the failing test**

```python
# Add to tests/detective/test_parallel_evolution.py

@pytest.mark.asyncio
async def test_evolve_parallel_sets_trajectory_urgency():
    """When phi_metrics provided, trajectory_urgency should be set."""
    from src.detective.hypothesis import Hypothesis
    from src.detective.parallel_evolution import evolve_parallel
    from unittest.mock import patch

    root = Hypothesis.create("Test gap hypothesis", 0.7)
    mock_provider = MockProvider()  # Use existing test mock
    phi_metrics = {"c": 0.3, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    with patch("src.detective.parallel_evolution.score_hypothesis_trajectory", return_value=0.7):
        results = await evolve_parallel(
            hypothesis=root,
            evidence_list=["evidence A"],
            provider=mock_provider,
            k=1,
            phi_metrics=phi_metrics,
        )

    assert len(results) == 1
    assert results[0].hypothesis.trajectory_urgency == 0.7


@pytest.mark.asyncio
async def test_evolve_parallel_uses_4_weight_scoring():
    """With phi_metrics, sort by 4-weight combined_score including trajectory_urgency."""
    from src.detective.hypothesis import Hypothesis
    from src.detective.parallel_evolution import evolve_parallel
    from unittest.mock import patch

    root = Hypothesis.create("Test", 0.5)
    mock_provider = MockProvider()
    phi_metrics = {"c": 0.3, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    with patch("src.detective.parallel_evolution.score_hypothesis_trajectory", return_value=0.9):
        results = await evolve_parallel(
            hypothesis=root,
            evidence_list=["ev A", "ev B"],
            provider=mock_provider,
            k=2,
            phi_metrics=phi_metrics,
        )

    # All results should have trajectory_urgency set
    for r in results:
        assert r.hypothesis.trajectory_urgency == 0.9
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/detective/test_parallel_evolution.py::test_evolve_parallel_sets_trajectory_urgency -v`
Expected: FAIL

**Step 3: Update parallel_evolution.py**

In `src/detective/parallel_evolution.py`, inside `evolve_parallel()`:

1. Import `score_hypothesis_trajectory` alongside existing welfare imports
2. After computing `welfare_score`, also compute `trajectory_urgency`
3. Set `trajectory_urgency` on the hypothesis via `replace()`
4. Update sort key to use 4-weight `combined_score(alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)`

```python
# In the phi_metrics block (line ~144):
if phi_metrics is not None:
    from src.inference.welfare_scoring import (
        score_hypothesis_welfare,
        infer_threatened_constructs,
        score_hypothesis_trajectory,  # NEW
    )

    for i, result in enumerate(results):
        h = result.hypothesis
        constructs = infer_threatened_constructs(h.text)
        welfare_score = score_hypothesis_welfare(h, phi_metrics)
        trajectory_urgency = score_hypothesis_trajectory(h, phi_metrics)  # NEW

        updated_h = replace(
            h,
            welfare_relevance=welfare_score,
            threatened_constructs=constructs,
            trajectory_urgency=trajectory_urgency,  # NEW
        )
        results[i] = replace(result, hypothesis=updated_h)

# Update sort (line ~166):
if phi_metrics is not None:
    return sorted(
        results,
        key=lambda r: r.hypothesis.combined_score(
            alpha=0.45, beta=0.25, gamma=0.15, delta=0.15  # NEW weights
        ),
        reverse=True
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/detective/test_parallel_evolution.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/detective/parallel_evolution.py tests/detective/test_parallel_evolution.py
git commit -m "feat(bridge): wire trajectory urgency into parallel evolution scoring"
```

---

## Layer 3: Detective-Generated Scenarios

### Task 8: Implement scenario_extraction.py — Extract Construct Profiles

**Files:**
- Create: `src/inference/scenario_extraction.py`
- Create: `tests/inference/test_scenario_extraction.py`

**Step 1: Write the failing test**

```python
# tests/inference/test_scenario_extraction.py
"""Test detective-generated scenario extraction pipeline."""
import pytest
from unittest.mock import patch, MagicMock


def test_extract_construct_profiles_returns_list():
    from src.inference.scenario_extraction import extract_construct_profiles

    mock_classifier_scores = {
        "c": 0.6, "kappa": 0.3, "j": 0.4, "p": 0.5,
        "eps": 0.2, "lam_L": 0.7, "lam_P": 0.3, "xi": 0.5,
    }

    corpus_text = "Sample text about resource allocation. " * 100  # ~500 words

    with patch("src.inference.scenario_extraction.get_construct_scores", return_value=mock_classifier_scores):
        profiles = extract_construct_profiles(corpus_text)

    assert isinstance(profiles, list)
    assert len(profiles) >= 1
    assert all("scores" in p for p in profiles)
    assert all("chunk_index" in p for p in profiles)
    assert all(len(p["scores"]) == 8 for p in profiles)


def test_extract_profiles_chunks_corpus():
    """Should split corpus into ~500-word chunks."""
    from src.inference.scenario_extraction import extract_construct_profiles

    # 2000 words → ~4 chunks
    corpus_text = ("word " * 500) * 4

    call_count = 0
    def counting_scorer(text):
        nonlocal call_count
        call_count += 1
        return {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    with patch("src.inference.scenario_extraction.get_construct_scores", side_effect=counting_scorer):
        profiles = extract_construct_profiles(corpus_text)

    assert call_count >= 3  # Should have chunked


def test_identify_trajectory_patterns():
    from src.inference.scenario_extraction import identify_trajectory_patterns

    # Simulate declining lam_L pattern over 5 consecutive chunks
    profiles = [
        {"chunk_index": i, "scores": {
            "c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
            "eps": 0.5, "lam_L": 0.7 - i * 0.1, "lam_P": 0.5, "xi": 0.5,
        }}
        for i in range(5)
    ]

    patterns = identify_trajectory_patterns(profiles)
    assert isinstance(patterns, list)
    assert len(patterns) >= 1
    assert all("label" in p for p in patterns)
    assert all("start_levels" in p for p in patterns)
    assert all("end_levels" in p for p in patterns)


def test_generate_from_template():
    import pandas as pd
    from src.inference.scenario_extraction import generate_from_template

    template = {
        "label": "declining_love",
        "start_levels": {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                         "eps": 0.5, "lam_L": 0.7, "lam_P": 0.5, "xi": 0.5},
        "end_levels": {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                       "eps": 0.5, "lam_L": 0.2, "lam_P": 0.5, "xi": 0.5},
    }

    df = generate_from_template(template, length=200)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "phi" in df.columns
    assert all(c in df.columns for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"])
    # lam_L should decline
    assert df["lam_L"].iloc[0] > df["lam_L"].iloc[-1]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/inference/test_scenario_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement scenario_extraction.py**

Create `src/inference/scenario_extraction.py`:

```python
"""
Detective-generated scenario extraction.

Extracts welfare construct patterns from real text via the classifier,
identifies trajectory patterns (declining, rising, diverging constructs),
and generates synthetic training scenarios from those patterns.

This is Layer 3 of the Detective-Forecaster bridge: the detective's
findings enrich the forecaster's training data.
"""
from typing import Dict, List, Optional
import logging
import math

import numpy as np
import pandas as pd

from src.inference.welfare_scoring import (
    ALL_CONSTRUCTS, compute_phi, get_construct_scores,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE_WORDS = 500


def extract_construct_profiles(
    corpus_text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
) -> List[Dict]:
    """
    Split corpus into chunks and score each for welfare constructs.

    Args:
        corpus_text: Full text corpus.
        chunk_size: Target words per chunk.

    Returns:
        List of dicts with keys: chunk_index, scores, text_preview.
    """
    words = corpus_text.split()
    profiles = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < chunk_size // 4:
            break  # skip tiny trailing chunks

        chunk_text = " ".join(chunk_words)
        scores = get_construct_scores(chunk_text)

        profiles.append({
            "chunk_index": len(profiles),
            "scores": scores,
            "text_preview": chunk_text[:100],
        })

    return profiles


def identify_trajectory_patterns(
    profiles: List[Dict],
    min_run_length: int = 3,
    change_threshold: float = 0.15,
) -> List[Dict]:
    """
    Find trajectory patterns in consecutive construct profiles.

    Looks for runs of min_run_length chunks where a construct changes
    by at least change_threshold total. Each pattern becomes a scenario template.

    Args:
        profiles: Output from extract_construct_profiles.
        min_run_length: Minimum consecutive chunks to form a pattern.
        change_threshold: Minimum total change to qualify.

    Returns:
        List of scenario templates with label, start_levels, end_levels,
        dominant_construct, direction.
    """
    if len(profiles) < min_run_length:
        return []

    patterns = []

    for construct in ALL_CONSTRUCTS:
        values = [p["scores"][construct] for p in profiles]

        # Sliding window: find runs where construct changes significantly
        for start in range(len(values) - min_run_length + 1):
            end = start + min_run_length
            # Extend run while trend continues
            while end < len(values):
                delta = values[end] - values[start]
                prev_delta = values[end - 1] - values[start]
                if abs(delta) < abs(prev_delta):
                    break  # trend reversed
                end += 1

            total_change = values[min(end, len(values)) - 1] - values[start]

            if abs(total_change) >= change_threshold:
                direction = "declining" if total_change < 0 else "rising"
                start_levels = profiles[start]["scores"].copy()
                end_levels = profiles[min(end, len(profiles)) - 1]["scores"].copy()

                label = f"{direction}_{construct}"
                # Deduplicate: don't add if we already have this label
                if not any(p["label"] == label for p in patterns):
                    patterns.append({
                        "label": label,
                        "start_levels": start_levels,
                        "end_levels": end_levels,
                        "dominant_construct": construct,
                        "direction": direction,
                        "run_length": end - start,
                    })

    return patterns


def generate_from_template(
    template: Dict,
    length: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic trajectory from a scenario template.

    Interpolates between start and end levels, adds noise calibrated
    to the observed change magnitude, and computes Phi with derivatives.

    Args:
        template: Dict with start_levels, end_levels keys.
        length: Number of time steps.
        rng: Random generator for noise.

    Returns:
        DataFrame with 8 construct columns + phi column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data = {}
    for c in ALL_CONSTRUCTS:
        start = template["start_levels"].get(c, 0.5)
        end = template["end_levels"].get(c, 0.5)
        # Linear interpolation + noise proportional to change magnitude
        noise_scale = max(0.005, abs(end - start) * 0.05)
        data[c] = np.linspace(start, end, length) + rng.normal(0, noise_scale, length)
        data[c] = np.clip(data[c], 0.0, 1.0)

    df = pd.DataFrame(data)

    # Compute Phi with finite-difference derivatives
    phi_vals = np.empty(length, dtype=np.float64)
    prev_metrics = None
    for idx in range(length):
        metrics = {c: df.at[idx, c] for c in ALL_CONSTRUCTS}
        if idx == 0 or prev_metrics is None:
            derivs = {}
        else:
            derivs = {c: metrics[c] - prev_metrics[c] for c in ALL_CONSTRUCTS}
        phi_vals[idx] = compute_phi(metrics, derivatives=derivs)
        prev_metrics = metrics

    df["phi"] = phi_vals
    return df
```

Note: `compute_phi` in `welfare_scoring.py` currently doesn't accept `derivatives` — it uses the simpler formula (no recovery-aware floors). The space version in `spaces/maninagarden/welfare.py` does accept derivatives. For this implementation, pass `derivatives` only if the function accepts it; otherwise omit. Check the function signature at implementation time and adapt.

Actually — looking at `welfare_scoring.py:compute_phi` (line 246), it does NOT accept `derivatives`. The space's `welfare.py` does. For Layer 3, use the simpler call `compute_phi(metrics)` without derivatives. This is fine — the data will still capture construct-level patterns. Recovery-aware floors apply during forecaster training (which uses the space's formula).

So the implementation above should call `compute_phi(metrics)` instead of `compute_phi(metrics, derivatives=derivs)`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/inference/test_scenario_extraction.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/inference/scenario_extraction.py tests/inference/test_scenario_extraction.py
git commit -m "feat(bridge): scenario extraction pipeline — profiles, patterns, generation"
```

---

### Task 9: Full Extraction Pipeline — Corpus to Scenarios

End-to-end: load `smiles_and_cries_extracted.txt`, run extraction, produce scenario DataFrames.

**Files:**
- Modify: `src/inference/scenario_extraction.py` (add `run_extraction_pipeline()`)
- Add tests to: `tests/inference/test_scenario_extraction.py`

**Step 1: Write the failing test**

```python
# Add to tests/inference/test_scenario_extraction.py

def test_run_extraction_pipeline():
    from src.inference.scenario_extraction import run_extraction_pipeline
    from unittest.mock import patch
    import tempfile, os

    # Create small test corpus
    corpus = "Resource allocation suffered as community bonds weakened. " * 200
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(corpus)
        corpus_path = f.name

    try:
        mock_scores = {"c": 0.7, "kappa": 0.3, "j": 0.4, "p": 0.5,
                       "eps": 0.2, "lam_L": 0.3, "lam_P": 0.3, "xi": 0.5}
        with patch("src.inference.scenario_extraction.get_construct_scores", return_value=mock_scores):
            result = run_extraction_pipeline(corpus_path)

        assert "profiles" in result
        assert "patterns" in result
        assert "scenarios" in result  # list of (label, DataFrame) tuples
    finally:
        os.unlink(corpus_path)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/inference/test_scenario_extraction.py::test_run_extraction_pipeline -v`
Expected: FAIL with `ImportError`

**Step 3: Add `run_extraction_pipeline()` to scenario_extraction.py**

```python
def run_extraction_pipeline(
    corpus_path: str,
    scenario_length: int = 200,
    seed: int = 42,
) -> Dict:
    """
    End-to-end: corpus file → construct profiles → trajectory patterns → synthetic scenarios.

    Args:
        corpus_path: Path to text corpus file.
        scenario_length: Length of generated scenarios.
        seed: Random seed.

    Returns:
        Dict with keys: profiles, patterns, scenarios (list of (label, DataFrame) tuples).
    """
    from pathlib import Path

    text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    logger.info(f"Loaded corpus: {len(text)} chars, ~{len(text.split())} words")

    profiles = extract_construct_profiles(text)
    logger.info(f"Extracted {len(profiles)} construct profiles")

    patterns = identify_trajectory_patterns(profiles)
    logger.info(f"Identified {len(patterns)} trajectory patterns")

    rng = np.random.default_rng(seed)
    scenarios = []
    for pattern in patterns:
        df = generate_from_template(pattern, length=scenario_length, rng=rng)
        scenarios.append((pattern["label"], df))
    logger.info(f"Generated {len(scenarios)} synthetic scenarios")

    return {
        "profiles": profiles,
        "patterns": patterns,
        "scenarios": scenarios,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/inference/test_scenario_extraction.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/inference/scenario_extraction.py tests/inference/test_scenario_extraction.py
git commit -m "feat(bridge): end-to-end corpus-to-scenarios extraction pipeline"
```

---

### Task 10: Update Space scenarios.py to Import Extracted Scenarios

**Files:**
- Modify: `spaces/maninagarden/scenarios.py`
- Modify: `spaces/maninagarden/app.py` (Scenario Explorer dropdown)

**Step 1: Write the failing test**

```python
# tests/spaces/test_extracted_scenarios.py
"""Test Space integration with extracted scenarios."""
import pytest
import sys
from pathlib import Path


def test_scenarios_module_has_extracted_loader():
    """scenarios.py should have a function to load extracted scenarios."""
    # Add space to path for import
    space_dir = str(Path(__file__).parent.parent.parent / "spaces" / "maninagarden")
    if space_dir not in sys.path:
        sys.path.insert(0, space_dir)
    try:
        from scenarios import load_extracted_scenarios
        assert callable(load_extracted_scenarios)
    finally:
        sys.path.remove(space_dir)


def test_extracted_scenarios_returns_dict():
    space_dir = str(Path(__file__).parent.parent.parent / "spaces" / "maninagarden")
    if space_dir not in sys.path:
        sys.path.insert(0, space_dir)
    try:
        from scenarios import load_extracted_scenarios
        result = load_extracted_scenarios()
        assert isinstance(result, dict)
        # Keys are scenario names, values are descriptions
    finally:
        sys.path.remove(space_dir)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/spaces/test_extracted_scenarios.py -v`
Expected: FAIL

**Step 3: Update scenarios.py**

Add to the bottom of `spaces/maninagarden/scenarios.py`:

```python
# ============================================================================
# Extracted scenario support (Layer 3 bridge)
# ============================================================================

EXTRACTED_SCENARIOS_PATH = Path(__file__).parent / "extracted_scenarios.json"


def load_extracted_scenarios() -> dict:
    """Load extracted scenario templates from JSON file.

    Returns dict mapping scenario_name → description string.
    Returns empty dict if no extracted scenarios exist yet.
    """
    import json
    if not EXTRACTED_SCENARIOS_PATH.exists():
        return {}
    try:
        with open(EXTRACTED_SCENARIOS_PATH) as f:
            templates = json.load(f)
        return {t["label"]: t.get("description", f"Extracted: {t['label']}") for t in templates}
    except (json.JSONDecodeError, KeyError):
        return {}


def generate_extracted_scenario(label, length=200, rng=None):
    """Generate trajectory from an extracted scenario template."""
    import json
    if not EXTRACTED_SCENARIOS_PATH.exists():
        raise ValueError(f"No extracted scenarios available")

    with open(EXTRACTED_SCENARIOS_PATH) as f:
        templates = json.load(f)

    template = next((t for t in templates if t["label"] == label), None)
    if template is None:
        raise ValueError(f"Unknown extracted scenario: {label}")

    if rng is None:
        rng = np.random.default_rng(42)

    data = {}
    for c in ALL_CONSTRUCTS:
        start = template["start_levels"].get(c, 0.5)
        end = template["end_levels"].get(c, 0.5)
        noise_scale = max(0.005, abs(end - start) * 0.05)
        data[c] = np.linspace(start, end, length) + rng.normal(0, noise_scale, length)
        data[c] = np.clip(data[c], 0.0, 1.0)

    df = pd.DataFrame(data)

    # Compute phi
    n = len(df)
    phi_vals = np.empty(n, dtype=np.float64)
    prev_metrics = None
    for idx in range(n):
        metrics = {c: df.at[idx, c] for c in ALL_CONSTRUCTS}
        if idx == 0 or prev_metrics is None:
            derivs = {}
        else:
            derivs = {c: metrics[c] - prev_metrics[c] for c in ALL_CONSTRUCTS}
        phi_vals[idx] = compute_phi(metrics, derivatives=derivs)
        prev_metrics = metrics

    df["phi"] = phi_vals
    return df
```

Add `from pathlib import Path` to imports at the top of the file.

**Step 4: Run test to verify it passes**

Run: `pytest tests/spaces/test_extracted_scenarios.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add spaces/maninagarden/scenarios.py tests/spaces/test_extracted_scenarios.py
git commit -m "feat(bridge): extracted scenario support in Gradio Space"
```

---

### Task 11: Extraction CLI Command

Add a CLI command to run the extraction pipeline and save results.

**Files:**
- Modify: `src/cli/main.py` (add `detective extract-scenarios` command)
- Create: `tests/cli/test_extract_command.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_extract_command.py
"""Test extract-scenarios CLI command."""
import pytest
from click.testing import CliRunner


def test_extract_scenarios_command_exists():
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["extract-scenarios", "--help"])
    assert result.exit_code == 0
    assert "corpus" in result.output.lower() or "extract" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cli/test_extract_command.py -v`
Expected: FAIL

**Step 3: Add command to src/cli/main.py**

```python
@cli.command("extract-scenarios")
@click.argument("corpus_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="spaces/maninagarden/extracted_scenarios.json",
              help="Output JSON path for extracted scenario templates")
@click.option("--length", type=int, default=200, help="Scenario trajectory length")
def extract_scenarios(corpus_path, output, length):
    """Extract welfare trajectory patterns from a text corpus.

    Runs the A/B/C detection pipeline on CORPUS_PATH, identifies trajectory
    patterns in construct scores, and saves scenario templates for the forecaster.
    """
    import json
    from src.inference.scenario_extraction import run_extraction_pipeline

    click.echo(f"Extracting scenarios from {corpus_path}...")
    result = run_extraction_pipeline(corpus_path, scenario_length=length)

    click.echo(f"  Profiles: {len(result['profiles'])}")
    click.echo(f"  Patterns: {len(result['patterns'])}")
    click.echo(f"  Scenarios: {len(result['scenarios'])}")

    # Save templates (not DataFrames — those are generated at runtime)
    templates = result["patterns"]
    with open(output, "w") as f:
        json.dump(templates, f, indent=2)

    click.echo(f"Saved {len(templates)} scenario templates to {output}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/cli/test_extract_command.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cli/main.py tests/cli/test_extract_command.py
git commit -m "feat(bridge): CLI command for scenario extraction from corpus"
```

---

### Task 12: Update Space Training Tab for Expanded Scenarios

Update the training script generator to include extracted scenarios alongside the 8 hand-designed ones.

**Files:**
- Modify: `spaces/maninagarden/training.py`

**Step 1: Write the failing test**

```python
# tests/spaces/test_training_with_extracted.py
"""Test training script includes extracted scenarios."""
import sys
from pathlib import Path
import pytest


def test_training_script_references_extracted_scenarios():
    space_dir = str(Path(__file__).parent.parent.parent / "spaces" / "maninagarden")
    if space_dir not in sys.path:
        sys.path.insert(0, space_dir)
    try:
        from training import get_training_script
        script = get_training_script(
            epochs=1, lr=1e-3, hidden_size=128,
            batch_size=32, scenarios_per_type=10,
        )
        assert "extracted_scenarios" in script or "load_extracted" in script
    finally:
        sys.path.remove(space_dir)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/spaces/test_training_with_extracted.py -v`
Expected: FAIL

**Step 3: Update training.py**

In the generated training script string within `get_training_script()`, after the section that generates scenarios from the 8 hard-coded types, add a section that:
1. Checks if `extracted_scenarios.json` exists (uploaded as a repo file)
2. If yes, loads templates and generates additional scenarios using the same `generate_from_template` logic
3. Concatenates these with the original scenarios for training

This is a targeted edit to the existing f-string in `training.py`. The exact insertion point is after the loop over `SCENARIOS` in the generated script.

**Step 4: Run test to verify it passes**

Run: `pytest tests/spaces/test_training_with_extracted.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add spaces/maninagarden/training.py tests/spaces/test_training_with_extracted.py
git commit -m "feat(bridge): training script includes extracted scenarios from corpus"
```

---

### Task 13: Integration Test — Full Bridge Pipeline

**Files:**
- Create: `tests/integration/test_bridge_pipeline.py`

**Step 1: Write integration test**

```python
# tests/integration/test_bridge_pipeline.py
"""Integration test: full bridge pipeline with mocks at boundaries."""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import replace
import numpy as np


def test_layer1_classifier_scores_flow_to_welfare_scoring():
    """Classifier scores → infer_threatened_constructs → welfare_relevance."""
    from src.detective.hypothesis import Hypothesis
    from src.inference.welfare_scoring import (
        score_hypothesis_welfare, infer_threatened_constructs,
    )

    # Mock classifier returning real scores
    mock_scores = {
        "c": 0.8, "kappa": 0.1, "j": 0.1, "p": 0.1,
        "eps": 0.1, "lam_L": 0.1, "lam_P": 0.1, "xi": 0.7,
    }

    with patch("src.inference.welfare_scoring.get_construct_scores", return_value=mock_scores):
        constructs = infer_threatened_constructs("Resource gap in records")
        assert "c" in constructs
        assert "xi" in constructs

    h = Hypothesis.create("Resource gap in records", 0.8)
    h = replace(h, threatened_constructs=constructs)
    phi_metrics = {"c": 0.2, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.3}
    welfare = score_hypothesis_welfare(h, phi_metrics)
    assert welfare > 0.3  # Both c and xi are scarce → high relevance


def test_layer2_trajectory_urgency_into_hypothesis():
    """Trajectory urgency flows through combined_score."""
    from src.detective.hypothesis import Hypothesis

    h = Hypothesis.create("Declining welfare detected", 0.7)
    h = replace(h, welfare_relevance=0.5, curiosity_relevance=0.3, trajectory_urgency=0.8)

    score = h.combined_score(alpha=0.45, beta=0.25, gamma=0.15, delta=0.15)
    assert score > 0.5
    # Verify trajectory_urgency contributes
    score_no_traj = h.combined_score(alpha=0.45, beta=0.25, gamma=0.15, delta=0.0)
    assert score > score_no_traj


def test_layer3_extraction_produces_valid_scenarios():
    """Extraction pipeline → templates → generateable scenarios."""
    from src.inference.scenario_extraction import (
        identify_trajectory_patterns, generate_from_template,
    )

    profiles = [
        {"chunk_index": i, "scores": {
            "c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
            "eps": 0.5, "lam_L": max(0.1, 0.7 - i * 0.15), "lam_P": 0.5, "xi": 0.5,
        }}
        for i in range(6)
    ]

    patterns = identify_trajectory_patterns(profiles)
    assert len(patterns) >= 1

    for pattern in patterns:
        df = generate_from_template(pattern, length=200)
        assert len(df) == 200
        assert "phi" in df.columns
        assert all(df["phi"] > 0)  # Phi should be positive
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_bridge_pipeline.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/integration/test_bridge_pipeline.py
git commit -m "test(bridge): integration tests for full 3-layer bridge pipeline"
```

---

## Summary

| Task | Layer | What | Key Files |
|------|-------|------|-----------|
| 1 | L1 | HF Jobs training script | `scripts/train_welfare_classifier_hf_job.py` |
| 2 | L1 | Hub model loading | `src/inference/welfare_classifier.py` |
| 3 | L1 | Upload training data | Hub: `crichalchemist/welfare-training-data` |
| 4 | L1 | Launch training | HF Jobs |
| 5 | L2 | `trajectory_urgency` field | `src/detective/hypothesis.py` |
| 6 | L2 | `score_hypothesis_trajectory()` | `src/inference/welfare_scoring.py` |
| 7 | L2 | Wire into parallel_evolution | `src/detective/parallel_evolution.py` |
| 8 | L3 | Scenario extraction core | `src/inference/scenario_extraction.py` |
| 9 | L3 | Full extraction pipeline | `src/inference/scenario_extraction.py` |
| 10 | L3 | Space scenario integration | `spaces/maninagarden/scenarios.py` |
| 11 | L3 | CLI extract command | `src/cli/main.py` |
| 12 | L3 | Training with extracted scenarios | `spaces/maninagarden/training.py` |
| 13 | All | Integration tests | `tests/integration/test_bridge_pipeline.py` |

**Dependencies:** Tasks 1-4 (Layer 1) are independent. Tasks 5-7 (Layer 2) depend on Layer 1 being deployed. Tasks 8-12 (Layer 3) depend on Layer 1 (classifier). Task 13 validates everything.
