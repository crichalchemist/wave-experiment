# Full Audit Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remediate all 22 issues found in the codebase audit — security, dead code, thread safety, duplication, consistency, and packaging.

**Architecture:** Changes organized into 13 independent task groups. Each group touches a distinct subsystem, enabling parallel execution. Groups: security, shared scoring utility, welfare consolidation, thread safety, provider hardening, graph consolidation, API/config, trace store, dependencies, exception handling, named constants, CLAUDE.md, CLI N+1 fix.

**Tech Stack:** Python 3.12+, pytest, networkx, FastAPI, threading

---

### Task 1: Security — Remove tracked secrets

**Files:**
- Modify: `.gitignore`
- Modify: `deployment/.env.service` (ensure untracked)

**Step 1: Verify .env.local is gitignored**

`.gitignore:65` already has `.env.local` — confirm git is not tracking it:

```bash
git ls-files --error-unmatch .env.local 2>&1 || echo "Not tracked"
```

If tracked, untrack without deleting:
```bash
git rm --cached .env.local
```

**Step 2: Add deployment/.env.service to .gitignore**

Add below line 66 of `.gitignore`:
```
deployment/.env.service
```

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "fix(security): ensure .env.local and .env.service stay untracked"
```

> **Note:** Key rotation is an out-of-band operation (Azure portal). Flag to user.

---

### Task 2: Extract shared scoring utility — `src/core/scoring.py`

**Files:**
- Create: `src/core/scoring.py`
- Create: `tests/core/test_scoring.py`
- Modify: `src/detective/module_a.py:36,59-63`
- Modify: `src/detective/module_b.py:69-77`
- Modify: `src/detective/module_c.py:88-96`
- Modify: `src/detective/parallel_evolution.py:43,60-68`

**Step 1: Write failing tests for shared scoring**

```python
# tests/core/test_scoring.py
import pytest
from src.core.scoring import parse_score, clamp_confidence, SCORE_RE

class TestParseScore:
    def test_standard_format(self):
        assert parse_score("score: 0.85") == 0.85

    def test_confidence_format(self):
        assert parse_score("confidence: 0.7") == 0.7

    def test_equals_separator(self):
        assert parse_score("score = 0.6") == 0.6

    def test_clamps_above_one(self):
        assert parse_score("score: 1.5") == 1.0

    def test_clamps_below_zero(self):
        assert parse_score("score: -0.3") == 0.0

    def test_no_match_returns_default(self):
        assert parse_score("no score here") == 0.0

    def test_custom_default(self):
        assert parse_score("no score here", default=0.5) == 0.5

    def test_invalid_number_returns_default(self):
        assert parse_score("score: abc") == 0.0


class TestClampConfidence:
    def test_within_range(self):
        assert clamp_confidence(0.5) == 0.5

    def test_above_one(self):
        assert clamp_confidence(1.5) == 1.0

    def test_below_zero(self):
        assert clamp_confidence(-0.5) == 0.0

    def test_boundary_zero(self):
        assert clamp_confidence(0.0) == 0.0

    def test_boundary_one(self):
        assert clamp_confidence(1.0) == 1.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/core/test_scoring.py -v
```

**Step 3: Implement `src/core/scoring.py`**

```python
"""Shared scoring utilities for LLM response parsing.

Consolidates the _parse_score() pattern used across detective modules A, B, C
and parallel_evolution into a single source of truth.
"""
from __future__ import annotations

import re

# Accepts "score:", "confidence:", with ":" or "=" separator.
# Module A was more permissive than B/C — this unifies to the broadest pattern.
SCORE_RE = re.compile(
    r"(?:score|confidence)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)


def clamp_confidence(value: float) -> float:
    """Clamp a confidence/score value to [0.0, 1.0]."""
    return min(1.0, max(0.0, value))


def parse_score(response: str, default: float = 0.0) -> float:
    """Extract and clamp a float score from an LLM response.

    Handles formats: "score: 0.85", "confidence = 0.7", "score:0.9"
    Returns default if no match found. Always returns a value in [0.0, 1.0].
    """
    match = SCORE_RE.search(response)
    if not match:
        return default
    try:
        return clamp_confidence(float(match.group(1)))
    except ValueError:
        return default
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/core/test_scoring.py -v
```

**Step 5: Refactor module_a.py** — remove `_SCORE_RE` (line 36) and `_parse_score` (lines 59-63), import from `src.core.scoring`:

```python
# Replace line 36 and lines 59-63 with:
from src.core.scoring import parse_score as _parse_score
```

Remove the old `_SCORE_RE` definition at line 36 entirely.

**Step 6: Refactor module_b.py** — remove `_parse_score` (lines 69-77), import:

```python
from src.core.scoring import parse_score as _parse_score
```

**Step 7: Refactor module_c.py** — remove `_parse_score` (lines 88-96), import:

```python
from src.core.scoring import parse_score as _parse_score
```

**Step 8: Refactor parallel_evolution.py** — replace `_parse_confidence` (lines 60-68) to use shared utility:

```python
from src.core.scoring import clamp_confidence, SCORE_RE as _CONFIDENCE_RE

def _parse_confidence(response: str, current: float) -> float:
    """Extract confidence from response. Falls back to slight decay on parse failure."""
    match = _CONFIDENCE_RE.search(response)
    if not match:
        return max(0.0, current - 0.05)
    try:
        return clamp_confidence(float(match.group(1)))
    except ValueError:
        return max(0.0, current - 0.05)
```

Remove the old `_CONFIDENCE_RE` at line 43.

**Step 9: Run full test suite to verify no regressions**

```bash
pytest tests/detective/ tests/core/test_scoring.py -v
```

**Step 10: Commit**

```bash
git add src/core/scoring.py tests/core/test_scoring.py src/detective/module_a.py src/detective/module_b.py src/detective/module_c.py src/detective/parallel_evolution.py
git commit -m "refactor: extract shared scoring utility, deduplicate _parse_score across modules A/B/C"
```

---

### Task 3: Consolidate welfare construct scoring

**Files:**
- Modify: `src/inference/welfare_scoring.py:310-370`
- Modify: `src/inference/welfare_classifier.py` (canonical source — no changes needed)

**Step 1: Remove duplicate `get_construct_scores` and `infer_threatened_constructs` from welfare_scoring.py**

Replace lines 310-369 of `welfare_scoring.py` with thin imports:

```python
def get_construct_scores(text: str) -> Dict[str, float]:
    """Get welfare construct scores — delegates to semantic classifier with keyword fallback."""
    try:
        from src.inference.welfare_classifier import get_construct_scores as _semantic
        scores = _semantic(text)
        if any(score > 0.0 for score in scores.values()):
            return scores
    except (FileNotFoundError, ValueError, OSError):
        logger.debug("Semantic classifier unavailable, using keyword fallback")

    keyword_constructs = _keyword_fallback(text)
    return {
        construct: (1.0 if construct in keyword_constructs else 0.0)
        for construct in _CONSTRUCT_PATTERNS
    }


def infer_threatened_constructs(text: str) -> Tuple[str, ...]:
    """Infer which Phi constructs a hypothesis/gap threatens."""
    try:
        from src.inference.welfare_classifier import get_construct_scores as _semantic
        scores = _semantic(text)
        if any(score > 0.0 for score in scores.values()):
            return tuple(sorted(c for c, s in scores.items() if s >= 0.3))
    except (FileNotFoundError, ValueError, OSError):
        logger.debug("Semantic classifier unavailable, using keyword fallback")

    return _keyword_fallback(text)
```

Key change: `except Exception` → `except (FileNotFoundError, ValueError, OSError)` (fixes issue #10 at the same time).

**Step 2: Run tests**

```bash
pytest tests/inference/test_welfare_scoring.py tests/inference/test_welfare_classifier.py -v
```

**Step 3: Commit**

```bash
git add src/inference/welfare_scoring.py
git commit -m "refactor: consolidate welfare scoring — single canonical path through welfare_classifier"
```

---

### Task 4: Fix thread-unsafe forecaster singleton

**Files:**
- Modify: `src/inference/welfare_scoring.py:540-549`

**Step 1: Replace global singleton with thread-safe pattern**

Replace lines 540-549:

```python
import threading

_forecaster_lock = threading.Lock()
_forecaster_cache = None


def _get_forecaster():
    """Lazy-load PhiTrajectoryForecaster (thread-safe cached singleton)."""
    global _forecaster_cache
    if _forecaster_cache is not None:
        return _forecaster_cache
    with _forecaster_lock:
        if _forecaster_cache is None:
            from src.forecasting.phi_trajectory import PhiTrajectoryForecaster
            _forecaster_cache = PhiTrajectoryForecaster()
    return _forecaster_cache
```

**Step 2: Run tests**

```bash
pytest tests/inference/test_welfare_scoring.py -v
```

**Step 3: Commit**

```bash
git add src/inference/welfare_scoring.py
git commit -m "fix: add thread-safe locking to forecaster singleton"
```

---

### Task 5: Standardize provider env var loading

**Files:**
- Modify: `src/core/providers.py:236-273`

**Step 1: Add helper for required env vars**

Add after line 41:

```python
def _require_env(var: str) -> str:
    """Get a required environment variable, raising a clear error if missing."""
    value = os.environ.get(var)
    if not value:
        raise ValueError(f"Required environment variable {var!r} is not set.")
    return value
```

**Step 2: Refactor provider_from_env to use _require_env consistently**

Replace lines 244-262:

```python
    if provider_type == _PROVIDER_VLLM:
        return VLLMProvider(
            base_url=_require_env(_ENV_VLLM_URL),
            model=_require_env(_ENV_VLLM_MODEL),
        )
    elif provider_type == _PROVIDER_AZURE:
        return AzureFoundryProvider(
            endpoint=_require_env(_ENV_AZURE_ENDPOINT),
            api_key=_require_env(_ENV_AZURE_KEY),
            model=_require_env(_ENV_AZURE_MODEL),
        )
    elif provider_type == _PROVIDER_HYBRID:
        scoring_url = os.environ.get(_ENV_VLLM_SCORING_URL, "http://localhost:8100/v1")
        scoring_model = os.environ.get(
            _ENV_VLLM_SCORING_MODEL, "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        )
        scoring = VLLMProvider(base_url=scoring_url, model=scoring_model)
        reasoning = AzureFoundryProvider(
            endpoint=_require_env(_ENV_AZURE_ENDPOINT),
            api_key=_require_env(_ENV_AZURE_KEY),
            model=_require_env(_ENV_AZURE_MODEL),
        )
```

**Step 3: Run tests**

```bash
pytest tests/core/test_providers.py -v
```

**Step 4: Commit**

```bash
git add src/core/providers.py
git commit -m "refactor: standardize env var loading with _require_env helper"
```

---

### Task 6: Add circuit breaker auto-recovery

**Files:**
- Modify: `src/core/providers.py` (HybridRoutingProvider)
- Modify: `tests/core/test_providers.py`

**Step 1: Write failing test**

```python
def test_hybrid_circuit_breaker_auto_recovers():
    """After cooldown, scoring provider is retried."""
    import time
    from unittest.mock import MagicMock
    from src.core.providers import HybridRoutingProvider, VLLMProvider, AzureFoundryProvider

    scoring = MagicMock(spec=VLLMProvider)
    scoring.complete = MagicMock(side_effect=[ConnectionError("down"), "score: 0.8"])
    reasoning = MagicMock(spec=AzureFoundryProvider)
    reasoning.complete = MagicMock(return_value="reasoning result")

    hybrid = HybridRoutingProvider(scoring_provider=scoring, reasoning_provider=reasoning)
    hybrid._circuit_breaker_cooldown = 0.1  # 100ms for test

    # First scoring call fails → circuit opens
    result1 = hybrid.complete("Reply with ONLY: score: <float>")
    assert not hybrid._scoring_available

    # Wait for cooldown
    time.sleep(0.15)

    # Next scoring call retries scoring provider
    result2 = hybrid.complete("Reply with ONLY: score: <float>")
    assert result2 == "score: 0.8"
```

**Step 2: Implement auto-recovery in HybridRoutingProvider**

Add `_circuit_opened_at` field and `_circuit_breaker_cooldown` field. Modify `complete()`:

```python
@dataclass
class HybridRoutingProvider:
    scoring_provider: VLLMProvider
    reasoning_provider: AzureFoundryProvider
    _scoring_available: bool = field(default=True, init=False, repr=False)
    _circuit_opened_at: float = field(default=0.0, init=False, repr=False)
    _circuit_breaker_cooldown: float = field(default=60.0, init=False, repr=False)  # 60s
    _trace_store: Any = field(default=None, init=False, repr=False)

    def complete(self, prompt: str, **kwargs) -> str:
        route = classify_prompt(prompt)
        t0 = time.monotonic()

        # Auto-recovery: retry scoring after cooldown
        if not self._scoring_available and self._circuit_opened_at > 0:
            if (t0 - self._circuit_opened_at) >= self._circuit_breaker_cooldown:
                _logger.info("Circuit breaker cooldown elapsed, retrying scoring provider")
                self._scoring_available = True

        if route == "scoring" and self._scoring_available:
            try:
                response = self.scoring_provider.complete(prompt, **kwargs)
                provider_used = self.scoring_provider
            except Exception as exc:
                _logger.warning("Scoring provider failed (%s), falling back to Azure", exc)
                self._scoring_available = False
                self._circuit_opened_at = t0
                response = self.reasoning_provider.complete(prompt, **kwargs)
                provider_used = self.reasoning_provider
        else:
            response = self.reasoning_provider.complete(prompt, **kwargs)
            provider_used = self.reasoning_provider
        # ... rest unchanged (trace recording)
```

**Step 3: Run tests**

```bash
pytest tests/core/test_providers.py -v
```

**Step 4: Commit**

```bash
git add src/core/providers.py tests/core/test_providers.py
git commit -m "feat: add circuit breaker auto-recovery with configurable cooldown"
```

---

### Task 7: Remove dead functional graph API

**Files:**
- Modify: `src/data/knowledge_graph.py:33-57` (remove `make_graph`, `add_edge`)
- Modify: `src/data/graph_store.py:10` (update import)

**Step 1: Remove dead functions from knowledge_graph.py**

Remove `make_graph()` (lines 33-35) and `add_edge()` (lines 38-57). Keep `get_edge()`, `n_hop_paths()`, `PathResult`, `_EDGE_DATA_KEY`, `_HOP_DECAY`, `KnowledgeGraph` — all used by `graph_store.py`.

**Step 2: Verify no callers**

```bash
grep -r "make_graph\|knowledge_graph\.add_edge" src/ --include="*.py"
```

Expected: no matches outside the removed functions themselves.

**Step 3: Run tests**

```bash
pytest tests/data/ tests/core/ -v
```

**Step 4: Commit**

```bash
git add src/data/knowledge_graph.py
git commit -m "refactor: remove dead copy-on-write graph functions (make_graph, add_edge)"
```

---

### Task 8: CORS from env var

**Files:**
- Modify: `src/api/routes.py:141-151`

**Step 1: Replace hardcoded origins with env var**

```python
    _DEFAULT_CORS_ORIGINS = [
        "https://www.crichalchemist.com",
        "https://crichalchemist.com",
        "https://crichalchemist-maninagarden.hf.space",
        "http://localhost:7860",
    ]

    cors_origins_raw = os.environ.get("CORS_ORIGINS")
    cors_origins = (
        [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
        if cors_origins_raw
        else _DEFAULT_CORS_ORIGINS
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
```

Add `import os` at top of file if not already present.

**Step 2: Run tests**

```bash
pytest tests/api/ -v
```

**Step 3: Commit**

```bash
git add src/api/routes.py
git commit -m "refactor: make CORS origins configurable via CORS_ORIGINS env var"
```

---

### Task 9: Add logging on silent provider fallback

**Files:**
- Modify: `src/api/routes.py:125-131`

**Step 1: Add warning log before MockProvider fallback**

```python
    if provider is not None:
        _provider: ModelProvider = provider
    else:
        try:
            _provider = provider_from_env()
        except (ValueError, KeyError, ImportError) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Provider configuration failed (%s), using MockProvider. "
                "API will return synthetic responses.",
                exc,
            )
            _provider = MockProvider(response=_API_MOCK_RESPONSE)
```

**Step 2: Run tests**

```bash
pytest tests/api/ -v
```

**Step 3: Commit**

```bash
git add src/api/routes.py
git commit -m "fix: log warning when API falls back to MockProvider"
```

---

### Task 10: Stream JSONL pagination in trace store

**Files:**
- Modify: `src/core/trace_store.py:79-96`

**Step 1: Replace readlines with streaming**

```python
    def historical(self, offset: int = 0, limit: int = 50) -> list[ReasoningTrace]:
        """Read traces from JSONL with pagination (newest first)."""
        if not self.path.exists():
            return []
        # Count lines and seek to the right offset from end
        with self._lock:
            lines: list[str] = []
            with open(self.path, encoding="utf-8") as f:
                # Read in reverse by collecting all line positions first
                all_positions: list[int] = []
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        all_positions.append(pos)

            # Reverse for newest-first, then slice
            all_positions.reverse()
            page_positions = all_positions[offset : offset + limit]

            with open(self.path, encoding="utf-8") as f:
                for pos in page_positions:
                    f.seek(pos)
                    lines.append(f.readline().strip())

        traces: list[ReasoningTrace] = []
        for line in lines:
            if not line:
                continue
            data = json.loads(line)
            traces.append(ReasoningTrace(**data))
        return traces
```

**Step 2: Run tests**

```bash
pytest tests/core/test_trace_store.py -v
```

**Step 3: Commit**

```bash
git add src/core/trace_store.py
git commit -m "perf: stream JSONL pagination instead of loading entire file into memory"
```

---

### Task 11: Tighten dependency versions

**Files:**
- Modify: `pyproject.toml:6-30`

**Step 1: Update dependency constraints**

```toml
dependencies = [
    "torch>=2.2.0,<3.0",
    "transformers>=4.40.0,<5.0",
    "networkx>=3.0,<4.0",
    "pydantic>=2.0,<3.0",
    "fastapi>=0.100.0,<1.0",
    "uvicorn>=0.23.0,<1.0",
    "click>=8.1.0,<9.0",
    "pdfplumber>=0.10.0,<1.0",
    "trl>=0.8.0,<1.0",
    "peft>=0.10.0,<1.0",
    "torch-geometric>=2.5.0,<3.0",
    "llama-index>=0.10.0,<1.0",
    "spacy>=3.7.0,<4.0",
    "openai>=1.0.0,<2.0",
    "azure-ai-inference>=1.0.0b1",
    "httpx>=0.27.0,<1.0",
]
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "fix: add upper-bound version constraints to prevent breaking upgrades"
```

---

### Task 12: Guard scraping imports

**Files:**
- Modify: `src/data/sourcing/foia_scraper.py` (top-level imports)

**Step 1: Wrap conditional imports**

Replace unconditional imports at top of foia_scraper.py with:

```python
try:
    from scrapling import Fetcher
except ImportError:
    Fetcher = None  # type: ignore[assignment,misc]

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None  # type: ignore[assignment]

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore[assignment]
```

Add a guard in `__init__` or at first use:

```python
if Fetcher is None:
    raise ImportError(
        "Scraping dependencies not installed. Run: pip install detective-llm[scraping]"
    )
```

**Step 2: Run tests**

```bash
pytest tests/data/sourcing/ -v
```

**Step 3: Commit**

```bash
git add src/data/sourcing/foia_scraper.py
git commit -m "fix: guard scraping imports to avoid ImportError without [scraping] extra"
```

---

### Task 13: Named constants for magic numbers and scoring weights

**Files:**
- Modify: `src/inference/welfare_scoring.py` (magic numbers)
- Modify: `src/detective/hypothesis.py` (weight schemes)
- Modify: `src/detective/parallel_evolution.py:168-169`

**Step 1: Add named constants to welfare_scoring.py**

```python
# Recovery-aware sigmoid parameters (recovery_aware_input)
_SIGMOID_STEEPNESS: float = 10.0  # How sharply sigmoid responds to trajectory change
_SIGMOID_BIAS: float = -3.0       # Shift: dx_dt=0 → ~0.047 (not 0.5)
_COMMUNITY_GUARD: float = 0.01    # Guard against lam_L=0 in community_capacity

# Equity weight defaults
_EQUITY_GUARD: float = 0.01       # Guard against division by zero in equity weights
_EQUITY_DEFAULT: float = 0.5      # Default construct level when not in metrics

# Normalization offsets
_WELFARE_NORMALIZATION_K: float = 1.0   # score → 0.5 when gradient_sum = 1.0
_CURIOSITY_NORMALIZATION_K: float = 1.0
_TRAJECTORY_URGENCY_K: float = 0.02     # decline of 0.02/step → urgency ~0.5

# Forecast noise
_FORECAST_NOISE_SCALE: float = 0.001
```

Then replace the inline magic numbers with these constants.

**Step 2: Add named weight schemes to hypothesis.py**

Add after the class:

```python
# Named scoring weight schemes (ADR-010)
WEIGHTS_DEFAULT = {"alpha": 0.55, "beta": 0.30, "gamma": 0.15, "delta": 0.0}
WEIGHTS_BRIDGE = {"alpha": 0.45, "beta": 0.25, "gamma": 0.15, "delta": 0.15}
```

**Step 3: Use named weights in parallel_evolution.py**

Replace lines 168-169:

```python
from src.detective.hypothesis import WEIGHTS_BRIDGE

# ...
key=lambda r: r.hypothesis.combined_score(**WEIGHTS_BRIDGE),
```

**Step 4: Run tests**

```bash
pytest tests/inference/ tests/detective/ tests/test_hypothesis.py -v
```

**Step 5: Commit**

```bash
git add src/inference/welfare_scoring.py src/detective/hypothesis.py src/detective/parallel_evolution.py
git commit -m "refactor: extract magic numbers to named constants, add WEIGHTS_BRIDGE/WEIGHTS_DEFAULT"
```

---

### Task 14: Fix CLI N+1 graph query pattern

**Files:**
- Modify: `src/cli/main.py:90-100`

**Step 1: Prefetch all 2-hop targets in one pass**

Replace lines 90-100:

```python
    # n-hop exploration: prefetch 2-hop targets, then query paths in batch
    click.echo(f"\nPaths (up to {hops} hops):")
    two_hop_targets: set[str] = set()
    for successor in successors:
        for next_hop in graph.successors(successor):
            if next_hop != entity:
                two_hop_targets.add(next_hop)

    for target in sorted(two_hop_targets):
        paths = graph.n_hop_paths(entity, target, max_hops=hops)
        for p in paths[:3]:
            path_str = " → ".join(p.path)
            click.echo(f"  {path_str} (confidence: {p.confidence:.4f}, hops: {p.hops})")
```

This eliminates the per-successor inner loop calling n_hop_paths redundantly.

**Step 2: Run tests**

```bash
pytest tests/cli/ -v
```

**Step 3: Commit**

```bash
git add src/cli/main.py
git commit -m "perf: fix N+1 graph query in CLI network command"
```

---

### Task 15: Fix hardcoded paths — configurable via env vars

**Files:**
- Modify: `src/inference/welfare_classifier.py:21`
- Modify: `src/detective/constitution.py:8`

**Step 1: Make welfare classifier path configurable**

```python
MODEL_PATH = Path(os.environ.get(
    "WELFARE_MODEL_PATH", "models/welfare-constructs-distilbert"
))
```

Add `import os` if not present.

**Step 2: Make constitution path configurable**

```python
_DEFAULT_CONSTITUTION_PATH: Path = Path(
    os.environ.get("DETECTIVE_CONSTITUTION_PATH", "docs/constitution.md")
)
```

**Step 3: Run tests**

```bash
pytest tests/ -q
```

**Step 4: Commit**

```bash
git add src/inference/welfare_classifier.py src/detective/constitution.py
git commit -m "refactor: make model and constitution paths configurable via env vars"
```

---

### Task 16: Fix broad exception handling in welfare_scoring.py

Already handled in Task 3 — the welfare scoring consolidation replaces `except Exception` with specific exception types.

Additionally at line 627 (score_hypothesis_trajectory):

```python
    except Exception as e:
```

Replace with:

```python
    except (ImportError, ValueError, RuntimeError, OSError) as e:
```

---

### Task 17: Update CLAUDE.md to reflect reality

**Files:**
- Modify: `CLAUDE.md`

**Changes:**
1. Modules A/B/C description: add note that they are library functions callable via tests and CLI extension, not wired into the default `analyze()` pipeline
2. `constitutional_warmup.py`: move from "Stubs" to "Working modules"
3. Update test count after all changes
4. Add `src/core/scoring.py` to architecture description
5. Document `WEIGHTS_DEFAULT` and `WEIGHTS_BRIDGE` under scoring weights
6. Document new env vars: `CORS_ORIGINS`, `WELFARE_MODEL_PATH`, `DETECTIVE_CONSTITUTION_PATH`

**Commit:**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to match codebase reality"
```

---

## Execution Order

Tasks can be executed in parallel within these dependency tiers:

**Tier 1 (no deps):** Tasks 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15
**Tier 2 (after Task 2):** Task 3 (uses new scoring util indirectly), Task 13
**Tier 3 (after all):** Task 16, Task 17 (CLAUDE.md reflects all changes)
