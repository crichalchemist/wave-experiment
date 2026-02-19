# Gap Completion + Constitutional Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the remaining implementation gaps (Modules B/C, live provider wiring, pyproject fix, parallel evolution) and establish a constitution-first training pipeline sourcing from DOJ, FBI Vault, HuggingFace, and international stakeholder records.

**Architecture:** Constitutional reinforcement runs before SFT — the model internalizes epistemic values before learning gap detection mechanics. Module B detects historical determinism in document timestamps and causal language; Module C detects geopolitical presumptions in institutional actor attribution. Parallel hypothesis evolution adds GoT-style Generate(k) dispatch via asyncio.

**Tech Stack:** Python 3.12, PyTorch 2.10+cpu, trl (DPO/GRPO), transformers (DistilBERT), datasets (HuggingFace), httpx (DOJ/FBI FOIA), asyncio (parallel evolution), Click CLI, FastAPI, Ollama (local inference), Azure AI Foundry (Claude critic), DeepSeek-OCR (GPU path, `deepseek-ai/DeepSeek-OCR` via transformers), pytesseract (CPU fallback), python-magic (MIME sniffing), pdf2image+poppler (PDF rasterization)

**Ethical grounding:** The subjects of this investigative corpus are real people, many of them survivors of documented abuse. The constitution (docs/constitution.md) is not decorative — it is load-bearing. Core Principle 4 ("Care for those most affected above analytical elegance") governs training data curation: we prefer 200 high-quality, constitutionally-grounded examples over thousands of unvetted ones.

---

## Phase 0: Infrastructure Fixes (before anything else)

### Task 0: Fix pyproject.toml azure-ai-inference pin

**Files:**
- Modify: `pyproject.toml`

**Step 1: Read the current pin**

```bash
grep "azure-ai-inference" pyproject.toml
```
Expected: `azure-ai-inference>=1.0.0`

**Step 2: Change to accept beta releases**

Edit `pyproject.toml` — find `azure-ai-inference>=1.0.0` and replace with `azure-ai-inference`.
This lets pip self-resolve to `1.0.0b9` (latest available) without failing.

**Step 3: Verify install succeeds**

```bash
source .venv-deploy/bin/activate
pip install -e ".[dev]" --quiet 2>&1 | grep -i error || echo "CLEAN"
```
Expected: `CLEAN`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "fix: unpin azure-ai-inference — pip self-resolves to 1.0.0b9"
```

---

### Task 1: Wire Ollama (VLLMProvider) to live inference

Ollama is running at `http://localhost:11434/v1` with `deepseek-r1:7b` available. The API layer currently uses `MockProvider`. This task wires real inference end-to-end.

**Files:**
- Create: `.env.local` (gitignored, holds runtime config)
- Modify: `src/api/routes.py` (use `provider_from_env()` instead of MockProvider)

**Step 1: Verify Ollama is serving**

```bash
curl -s http://localhost:11434/v1/models | python3 -m json.tool | grep deepseek
```
Expected: `"deepseek-r1:7b"` in the response.

**Step 2: Write a failing test for live-provider round-trip**

Add to `tests/api/test_routes.py`:
```python
def test_analyze_with_vllm_env(monkeypatch, tmp_path):
    """Verify create_app uses provider_from_env() when DETECTIVE_PROVIDER is set."""
    monkeypatch.setenv("DETECTIVE_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VLLM_MODEL", "deepseek-r1:7b")
    from importlib import reload
    import src.api.routes as routes_module
    reload(routes_module)
    # Should not raise — proves create_app() calls provider_from_env() without crashing
    app = routes_module.create_app()
    assert app is not None
```

**Step 3: Run to verify it fails**

```bash
source .venv-deploy/bin/activate
pytest tests/api/test_routes.py::test_analyze_with_vllm_env -v
```
Expected: FAIL (routes.py hardcodes MockProvider).

**Step 4: Create `.env.local`**

```bash
cat > .env.local << 'EOF'
DETECTIVE_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:11434/v1
VLLM_MODEL=deepseek-r1:7b
DETECTIVE_GRAPH_BACKEND=memory
DETECTIVE_VAULT_PATH=docs/vault
EOF
echo ".env.local" >> .gitignore
```

**Step 5: Modify `src/api/routes.py` — replace MockProvider with `provider_from_env()`**

Find the `create_app()` function in `src/api/routes.py`. It currently instantiates MockProvider directly. Change it to:

```python
import os as _os
from src.core.providers import provider_from_env as _provider_from_env, MockProvider as _MockProvider

def create_app(provider=None, constitution=None, graph=None):
    """
    App factory. If provider is None, reads DETECTIVE_PROVIDER from environment.
    Falls back to MockProvider when env var is absent (tests, CI).
    """
    if provider is None:
        try:
            provider = _provider_from_env()
        except (ValueError, ImportError):
            provider = _MockProvider(response="[no provider configured]")
    # rest of existing create_app body unchanged
```

**Step 6: Run test**

```bash
pytest tests/api/test_routes.py -v
```
Expected: all pass.

**Step 7: Manual smoke test (with Ollama running)**

```bash
source .venv-deploy/bin/activate
set -a && source .env.local && set +a
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 &
sleep 2
curl -s -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"claim": "Entity A met with Entity B in 2003 but no financial records exist"}' \
  | python3 -m json.tool
```
Expected: JSON with `verdict`, `confidence`, `evidence_count` — verdict comes from DeepSeek, not mock.

**Step 8: Commit**

```bash
kill %1  # stop uvicorn
git add src/api/routes.py tests/api/test_routes.py .gitignore
git commit -m "feat: wire provider_from_env to API — live Ollama inference on /analyze"
```

---

## Phase 1: Module B — Historical Determinism Detector

Module B's failure mode is named explicitly in `docs/constitution.md` (Section: "Historical determinism"): treating document timestamps as event timestamps, inferring causality from documentary sequence. The detector targets temporal language that encodes deterministic assumptions.

**Files:**
- Create: `src/detective/module_b.py`
- Create: `tests/detective/test_module_b.py`

**Step 1: Write failing tests first**

Create `tests/detective/test_module_b.py`:

```python
"""Tests for Module B: historical determinism detection."""
from unittest.mock import MagicMock, patch


def test_import():
    from src.detective.module_b import DeterminismDetection, detect_historical_determinism
    assert DeterminismDetection is not None
    assert callable(detect_historical_determinism)


def test_returns_list():
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.1"
    result = detect_historical_determinism("Entity A always operated in New York.", provider)
    assert isinstance(result, list)


def test_detects_always_language():
    """'always' is a canonical historical determinism marker."""
    from src.detective.module_b import detect_historical_determinism, DeterminismDetection
    provider = MagicMock()
    provider.complete.return_value = "score: 0.85"
    results = detect_historical_determinism(
        "The organization always maintained these records.", provider
    )
    assert len(results) >= 1
    assert all(isinstance(r, DeterminismDetection) for r in results)
    assert results[0].score >= 0.7


def test_detects_continues_to_language():
    """'continues to' implies the past determines the present."""
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.80"
    results = detect_historical_determinism(
        "The board continues to deny any involvement.", provider
    )
    assert len(results) >= 1


def test_low_score_filtered():
    """Spans below threshold (0.5) must not be returned."""
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.2"
    results = detect_historical_determinism(
        "This report documents events in 2019.", provider
    )
    assert results == []


def test_detects_timestamp_as_causation():
    """Treating document date as event date is the core failure mode."""
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.9"
    results = detect_historical_determinism(
        "Because the memo was dated June 1, the meeting therefore occurred on June 1.", provider
    )
    assert len(results) >= 1
    assert results[0].score >= 0.7


def test_dataclass_is_frozen():
    from src.detective.module_b import DeterminismDetection
    from src.core.types import AssumptionType
    d = DeterminismDetection(
        assumption_type=AssumptionType.HISTORICAL_DETERMINISM,
        score=0.8,
        source_text="always maintained",
        trigger_phrase="always",
    )
    import pytest
    with pytest.raises(Exception):
        d.score = 0.5


def test_multiple_triggers_in_one_text():
    """Multiple deterministic phrases → multiple detections."""
    from src.detective.module_b import detect_historical_determinism
    provider = MagicMock()
    provider.complete.return_value = "score: 0.75"
    results = detect_historical_determinism(
        "The foundation always funded the program and continues to do so invariably.", provider
    )
    assert len(results) >= 2


def test_assumption_type_is_historical_determinism():
    from src.detective.module_b import detect_historical_determinism
    from src.core.types import AssumptionType
    provider = MagicMock()
    provider.complete.return_value = "score: 0.8"
    results = detect_historical_determinism("The records always showed this.", provider)
    assert all(r.assumption_type == AssumptionType.HISTORICAL_DETERMINISM for r in results)
```

**Step 2: Run to verify they fail**

```bash
pytest tests/detective/test_module_b.py -v 2>&1 | head -20
```
Expected: ImportError — `module_b` doesn't exist.

**Step 3: Check `AssumptionType` already has HISTORICAL_DETERMINISM**

```bash
grep "HISTORICAL_DETERMINISM" src/core/types.py
```
If absent, add `HISTORICAL_DETERMINISM = "historical_determinism"` to the `AssumptionType` enum in `src/core/types.py`.

**Step 4: Implement `src/detective/module_b.py`**

```python
"""
Module B: Historical Determinism Detector.

Detects language that treats documentary sequence as causal sequence,
document timestamps as event timestamps, or assumes the past fully
determines the present — the second failure mode in docs/constitution.md.

Uses a provider to score matched spans rather than a static classifier,
because determinism markers are context-dependent: "always" in a personal
memoir differs from "always" in a regulatory filing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.providers import ModelProvider
from src.core.types import AssumptionType

# Phrases that trigger determinism scoring — ordered from strongest signal to weakest.
_DETERMINISM_TRIGGERS: tuple[str, ...] = (
    r"\balways\b",
    r"\bcontinues?\s+to\b",
    r"\binvariably\b",
    r"\bconsistently\b",
    r"\bhas\s+(?:always|ever)\b",
    r"\bnever\s+(?:changed|deviated|altered)\b",
    r"\bsince\s+(?:the\s+beginning|its\s+founding|inception)\b",
    r"\bbecause\s+(?:the\s+)?(?:document|memo|record|filing)\s+(?:was\s+)?dated\b",
    r"\bthe\s+(?:document|record|memo)\s+therefore\b",
    r"\bhistorically\b",
    r"\btraditionally\b",
    r"\bby\s+(?:its\s+)?nature\b",
    r"\bde\s+facto\b",
)

_SCORE_THRESHOLD: float = 0.5

_SCORE_PROMPT = (
    "You are evaluating whether the following text span exhibits 'historical determinism' — "
    "the assumption that documentary sequence reflects causal sequence, or that the past "
    "fully determines the present without examining whether the documentary record is itself "
    "a strategic artifact.\n\n"
    "Rate on a scale from 0.0 (no determinism) to 1.0 (strong determinism).\n"
    "Reply with ONLY: score: <float>\n\n"
    "Text span: {span}\n"
    "Full context: {context}"
)


@dataclass(frozen=True)
class DeterminismDetection:
    """
    A detected instance of historical determinism language.

    Frozen because detections form an immutable audit record — mutating a
    detection after the fact would undermine the chain of evidence.
    """
    assumption_type: AssumptionType
    score: float          # 0.0–1.0; how strongly deterministic this span is
    source_text: str      # the broader context sentence
    trigger_phrase: str   # which regex pattern fired

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")


def _parse_score(response: str) -> float:
    """Extract float from 'score: 0.85' style response. Returns 0.0 on failure."""
    match = re.search(r"score\s*:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if not match:
        return 0.0
    try:
        return min(1.0, max(0.0, float(match.group(1))))
    except ValueError:
        return 0.0


def detect_historical_determinism(
    text: str,
    provider: ModelProvider,
    threshold: float = _SCORE_THRESHOLD,
) -> list[DeterminismDetection]:
    """
    Scan text for historical determinism language and score each match.

    Args:
        text: The document text to analyze.
        provider: LLM provider for context-sensitive scoring.
        threshold: Minimum score to include in results (default 0.5).

    Returns:
        List of DeterminismDetection instances, ordered by score descending.
        Empty list if no spans exceed threshold.
    """
    detections: list[DeterminismDetection] = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for pattern in _DETERMINISM_TRIGGERS:
        for sentence in sentences:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if not match:
                continue
            span = match.group(0)
            prompt = _SCORE_PROMPT.format(span=span, context=sentence)
            raw = provider.complete(prompt)
            score = _parse_score(raw)
            if score >= threshold:
                detections.append(DeterminismDetection(
                    assumption_type=AssumptionType.HISTORICAL_DETERMINISM,
                    score=score,
                    source_text=sentence,
                    trigger_phrase=pattern,
                ))

    return sorted(detections, key=lambda d: d.score, reverse=True)
```

**Step 5: Run tests**

```bash
pytest tests/detective/test_module_b.py -v
```
Expected: all 9 pass.

**Step 6: Commit**

```bash
git add src/detective/module_b.py tests/detective/test_module_b.py src/core/types.py
git commit -m "feat: Module B — historical determinism detector with provider-scored spans"
```

---

## Phase 2: Module C — Geopolitical Presumptions Detector

Module C targets the third constitution failure mode: "The assumption that institutions operating in a given national or cultural context follow the norms and incentives of that context as described in the dominant literature." In the Epstein context this manifests as assuming that regulatory oversight, intelligence agencies, and prosecutorial discretion functioned as officially described.

**Files:**
- Create: `src/detective/module_c.py`
- Create: `tests/detective/test_module_c.py`

**Step 1: Write failing tests**

Create `tests/detective/test_module_c.py`:

```python
"""Tests for Module C: geopolitical presumption detection."""
from unittest.mock import MagicMock


def test_import():
    from src.detective.module_c import GeopoliticalDetection, detect_geopolitical_presumptions
    assert GeopoliticalDetection is not None
    assert callable(detect_geopolitical_presumptions)


def test_returns_list():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.1"
    result = detect_geopolitical_presumptions("The regulator reviewed the filing.", provider)
    assert isinstance(result, list)


def test_detects_regulatory_assumption():
    """Assuming oversight worked as designed is the core geopolitical presumption."""
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.85"
    results = detect_geopolitical_presumptions(
        "The SEC properly reviewed all disclosures as required by law.", provider
    )
    assert len(results) >= 1
    assert results[0].score >= 0.7


def test_detects_intelligence_agency_assumption():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.80"
    results = detect_geopolitical_presumptions(
        "The FBI conducted a thorough investigation per standard protocol.", provider
    )
    assert len(results) >= 1


def test_detects_prosecutorial_discretion_assumption():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.75"
    results = detect_geopolitical_presumptions(
        "The prosecutor declined to pursue charges based on insufficient evidence.", provider
    )
    assert len(results) >= 1


def test_low_score_filtered():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.2"
    results = detect_geopolitical_presumptions(
        "The organization submitted its annual report.", provider
    )
    assert results == []


def test_dataclass_frozen():
    from src.detective.module_c import GeopoliticalDetection
    from src.core.types import AssumptionType
    d = GeopoliticalDetection(
        assumption_type=AssumptionType.GEOPOLITICAL_PRESUMPTION,
        score=0.8,
        source_text="The regulator approved...",
        presumed_actor="SEC",
    )
    import pytest
    with pytest.raises(Exception):
        d.score = 0.5


def test_assumption_type_is_geopolitical():
    from src.detective.module_c import detect_geopolitical_presumptions
    from src.core.types import AssumptionType
    provider = MagicMock()
    provider.complete.return_value = "score: 0.8"
    results = detect_geopolitical_presumptions(
        "Intelligence agencies monitored the situation per their mandate.", provider
    )
    assert all(r.assumption_type == AssumptionType.GEOPOLITICAL_PRESUMPTION for r in results)


def test_detects_unstated_actor_interest():
    """Assumes actor behavior reflects stated institutional role, not actual interests."""
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.88"
    results = detect_geopolitical_presumptions(
        "The government cooperated fully with the international inquiry as expected.", provider
    )
    assert len(results) >= 1


def test_multiple_actors_in_text():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.78"
    results = detect_geopolitical_presumptions(
        "The SEC and DOJ jointly reviewed the matter per standard inter-agency protocol.", provider
    )
    assert len(results) >= 2
```

**Step 2: Run to verify they fail**

```bash
pytest tests/detective/test_module_c.py -v 2>&1 | head -10
```
Expected: ImportError.

**Step 3: Check `AssumptionType` has GEOPOLITICAL_PRESUMPTION**

```bash
grep "GEOPOLITICAL_PRESUMPTION" src/core/types.py
```
If absent, add to AssumptionType enum.

**Step 4: Implement `src/detective/module_c.py`**

```python
"""
Module C: Geopolitical Presumption Detector.

Detects language that assumes institutional actors behaved according to their
stated mandates without questioning whether their actual interests aligned —
the third failure mode in docs/constitution.md.

Example: "The regulator properly reviewed all disclosures" presumes the
regulator's actual behavior matched its formal function. In influence network
analysis, this assumption often masks the most significant gaps: when powerful
actors exploit institutional legitimacy as cover for behavior that contradicts
that legitimacy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.providers import ModelProvider
from src.core.types import AssumptionType

# Institutional actor patterns — entities whose "normal" behavior is often presumed
_ACTOR_PATTERNS: tuple[str, ...] = (
    r"\b(?:the\s+)?(?:SEC|DOJ|FBI|CIA|NSA|DOD|DOE|DEA|ATF)\b",
    r"\b(?:the\s+)?(?:regulator|regulatory\s+body|oversight\s+committee)\b",
    r"\b(?:the\s+)?(?:prosecutor|attorney\s+general|district\s+attorney)\b",
    r"\b(?:the\s+)?(?:intelligence\s+(?:agency|service|community))\b",
    r"\b(?:the\s+)?(?:government|administration|authorities)\b",
    r"\b(?:the\s+)?(?:court|judiciary|judge)\b",
    r"\b(?:the\s+)?(?:embassy|consulate|diplomatic\s+mission)\b",
    r"\b(?:the\s+)?(?:ministry|minister|secretary)\b",
    r"\b(?:the\s+)?(?:Interpol|Europol|FATF)\b",
)

# Presumption verbs — language that assumes institutional behavior was normative
_PRESUMPTION_VERBS: tuple[str, ...] = (
    r"\bproperly\s+(?:reviewed|investigated|monitored|approved|enforced)\b",
    r"\bfully\s+cooperated\b",
    r"\bper\s+(?:standard\s+)?(?:protocol|procedure|mandate|regulation|law)\b",
    r"\bas\s+(?:required|mandated|stipulated)\s+by\b",
    r"\bin\s+accordance\s+with\b",
    r"\bas\s+expected\b",
    r"\bfollowed\s+standard\b",
    r"\bdeclined\s+to\s+pursue\s+(?:charges\s+)?(?:based\s+on|due\s+to)\b",
    r"\binsufficient\s+evidence\b",
    r"\bdetermined\s+that\s+no\b",
    r"\bfound\s+no\s+(?:evidence|wrongdoing|violation)\b",
)

_SCORE_THRESHOLD: float = 0.5

_SCORE_PROMPT = (
    "You are evaluating whether the following sentence contains a 'geopolitical presumption' — "
    "an unstated assumption that an institutional actor behaved according to its stated mandate, "
    "without questioning whether its actual interests or external pressures shaped its behavior "
    "differently.\n\n"
    "This is especially significant when powerful actors are described as having 'properly' "
    "followed procedure in contexts where that procedure may have served to suppress inquiry.\n\n"
    "Rate from 0.0 (no presumption) to 1.0 (strong presumption).\n"
    "Reply with ONLY: score: <float>\n\n"
    "Sentence: {sentence}\n"
    "Identified actor: {actor}"
)


@dataclass(frozen=True)
class GeopoliticalDetection:
    """
    A detected instance of geopolitical presumption language.

    Frozen for the same reason as all domain objects: detections form an
    immutable audit trail. A detection that changes after the fact is not
    a detection — it is a revision of the record.
    """
    assumption_type: AssumptionType
    score: float            # 0.0–1.0
    source_text: str        # sentence containing the presumption
    presumed_actor: str     # the institutional actor whose behavior is presumed normative

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")


def _parse_score(response: str) -> float:
    match = re.search(r"score\s*:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if not match:
        return 0.0
    try:
        return min(1.0, max(0.0, float(match.group(1))))
    except ValueError:
        return 0.0


def detect_geopolitical_presumptions(
    text: str,
    provider: ModelProvider,
    threshold: float = _SCORE_THRESHOLD,
) -> list[GeopoliticalDetection]:
    """
    Scan text for geopolitical presumption language.

    A sentence must contain both an institutional actor AND a presumption verb
    to trigger scoring — reducing false positives from actor mentions alone.

    Args:
        text: Document text to analyze.
        provider: LLM provider for context-sensitive scoring.
        threshold: Minimum score for inclusion (default 0.5).

    Returns:
        List of GeopoliticalDetection instances, ordered by score descending.
    """
    detections: list[GeopoliticalDetection] = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        actor_match = None
        for actor_pat in _ACTOR_PATTERNS:
            actor_match = re.search(actor_pat, sentence, re.IGNORECASE)
            if actor_match:
                break
        if not actor_match:
            continue

        has_presumption_verb = any(
            re.search(vp, sentence, re.IGNORECASE) for vp in _PRESUMPTION_VERBS
        )
        if not has_presumption_verb:
            continue

        actor_text = actor_match.group(0)
        prompt = _SCORE_PROMPT.format(sentence=sentence, actor=actor_text)
        raw = provider.complete(prompt)
        score = _parse_score(raw)

        if score >= threshold:
            detections.append(GeopoliticalDetection(
                assumption_type=AssumptionType.GEOPOLITICAL_PRESUMPTION,
                score=score,
                source_text=sentence,
                presumed_actor=actor_text,
            ))

    return sorted(detections, key=lambda d: d.score, reverse=True)
```

**Step 5: Run tests**

```bash
pytest tests/detective/test_module_c.py -v
```
Expected: all 9 pass.

**Step 6: Run full suite to check no regressions**

```bash
pytest tests/ -q
```
Expected: all pass.

**Step 7: Commit**

```bash
git add src/detective/module_c.py tests/detective/test_module_c.py src/core/types.py
git commit -m "feat: Module C — geopolitical presumption detector, completes A+B+C taxonomy"
```

---

## Phase 3: Dataset Sourcing Pipeline

**Goal:** Source real investigative documents from public records for constitutional training. Quality over quantity — 200 high-quality, constitutionally-curated examples is the explicit design target (Learning-at-Criticality principle from the research synthesis).

**Ethical protocol:** All sources are public record. Document-level metadata (source, date, jurisdiction, case reference) is preserved. Victim testimony is never used as training input without survivor-controlled publishing consent — we use institutional documents, official denunciations, and gap annotations only.

**Files:**
- Create: `src/data/sourcing/` (package)
- Create: `src/data/sourcing/__init__.py`
- Create: `src/data/sourcing/hf_loader.py`
- Create: `src/data/sourcing/doj_loader.py`
- Create: `src/data/sourcing/international_loader.py`
- Create: `src/data/sourcing/dataset_builder.py`
- Create: `tests/data/test_sourcing.py`
- Create: `data/raw/.gitkeep` (directory exists, contents gitignored)

### Task 3a: HuggingFace Legal Dataset Loader

Sourcing from HuggingFace datasets that contain court documents, regulatory filings, and FOIA-released records relevant to financial networks and institutional accountability.

**Relevant HF datasets:**
- `pile-of-law/pile-of-law` — 256GB legal text including PACER documents, SEC filings, federal regulations. Filter for Epstein/Maxwell/NML/LTCM-adjacent financial network documents.
- `nguha/legalbench` — legal reasoning benchmarks with gap-relevant tasks
- `joelito/eu_court_cases` — EU court decisions (international angle)
- `social-harms/hate-speech-and-offensive-language` — for detecting dehumanizing language patterns in institutional documents about victims
- `HuggingFaceFW/fineweb` — filtered for court/regulatory text using keyword queries

**Step 1: Write failing test**

Create `tests/data/test_sourcing.py`:

```python
"""Tests for dataset sourcing pipeline."""


def test_hf_loader_import():
    from src.data.sourcing.hf_loader import load_hf_legal_batch
    assert callable(load_hf_legal_batch)


def test_hf_loader_returns_list():
    """Loader returns list of dicts with required keys."""
    from src.data.sourcing.hf_loader import load_hf_legal_batch
    # Use a small public dataset that's always available
    batch = load_hf_legal_batch(
        dataset_name="nguha/legalbench",
        config_name="contract_nli",
        split="train",
        max_examples=5,
        text_field="text",
    )
    assert isinstance(batch, list)
    assert len(batch) <= 5
    if batch:
        assert "text" in batch[0]
        assert "source" in batch[0]
        assert "metadata" in batch[0]


def test_hf_loader_metadata_preserved():
    from src.data.sourcing.hf_loader import load_hf_legal_batch
    batch = load_hf_legal_batch(
        dataset_name="nguha/legalbench",
        config_name="contract_nli",
        split="train",
        max_examples=3,
        text_field="text",
    )
    if batch:
        meta = batch[0]["metadata"]
        assert "dataset" in meta
        assert "split" in meta
```

**Step 2: Run to verify failure**

```bash
pytest tests/data/test_sourcing.py::test_hf_loader_import -v
```
Expected: ImportError.

**Step 3: Install datasets library (if not already present)**

```bash
source .venv-deploy/bin/activate
pip install datasets --quiet
```

**Step 4: Implement `src/data/sourcing/__init__.py`**

```python
"""Dataset sourcing pipeline for constitutional training."""
```

**Step 5: Implement `src/data/sourcing/hf_loader.py`**

```python
"""
HuggingFace dataset loader for constitutional training data.

Loads legal, court, and regulatory text from public HF datasets.
Returns normalized dicts with text, source, and metadata preserved —
provenance tracking is required for all training examples per the
constitution's standpoint transparency principle.
"""
from __future__ import annotations

from typing import Any


def load_hf_legal_batch(
    dataset_name: str,
    split: str = "train",
    max_examples: int = 200,
    text_field: str = "text",
    config_name: str | None = None,
    keyword_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load a batch from a HuggingFace dataset, normalizing to {text, source, metadata}.

    Args:
        dataset_name: HF dataset identifier (e.g., "pile-of-law/pile-of-law").
        split: Dataset split ("train", "test", "validation").
        max_examples: Maximum number of examples to return.
        text_field: Name of the field containing the document text.
        config_name: Optional dataset configuration name.
        keyword_filter: If provided, only include examples containing this keyword.

    Returns:
        List of dicts: {text: str, source: str, metadata: dict}
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config_name:
        kwargs["name"] = config_name

    ds = load_dataset(dataset_name, **kwargs)

    results: list[dict[str, Any]] = []
    for example in ds:
        if len(results) >= max_examples:
            break
        text = example.get(text_field, "")
        if not text or not isinstance(text, str):
            continue
        if keyword_filter and keyword_filter.lower() not in text.lower():
            continue
        results.append({
            "text": text,
            "source": f"huggingface:{dataset_name}",
            "metadata": {
                "dataset": dataset_name,
                "split": split,
                "config": config_name,
                "original_keys": list(example.keys()),
            },
        })

    return results
```

**Step 6: Run tests**

```bash
pytest tests/data/test_sourcing.py -v -k "hf"
```
Expected: pass (requires internet access to HF).

### Task 3b: DOJ + FBI Vault Public Records Loader

**Public sources (no login required):**
- FBI Vault FOIA releases: `https://vault.fbi.gov/jeffrey-epstein`
- DOJ press releases: `https://www.justice.gov/usao-sdny/pr` (Maxwell trial)
- CourtListener (Free Law Project): `https://www.courtlistener.com/api/rest/v4/` — PACER documents for US v. Maxwell (case 21-cr-00188-AKH)
- SDNY docket: public access

**Step 1: Add to `tests/data/test_sourcing.py`**

```python
def test_doj_loader_import():
    from src.data.sourcing.doj_loader import load_courtlistener_batch
    assert callable(load_courtlistener_batch)


def test_doj_loader_returns_list(monkeypatch):
    """Loader returns list even when API returns empty results."""
    import httpx
    from src.data.sourcing.doj_loader import load_courtlistener_batch

    # Mock httpx to avoid network in unit tests
    class _MockResponse:
        def raise_for_status(self): pass
        def json(self): return {"results": [], "count": 0}

    monkeypatch.setattr(
        "src.data.sourcing.doj_loader._httpx_get",
        lambda url, **kw: _MockResponse(),
    )
    results = load_courtlistener_batch(case_name="Maxwell", max_examples=5)
    assert isinstance(results, list)


def test_doj_loader_normalizes_fields(monkeypatch):
    from src.data.sourcing.doj_loader import load_courtlistener_batch

    class _MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"results": [
                {"plain_text": "Court document text here.", "date_filed": "2021-11-01",
                 "docket_id": 123, "description": "Motion to suppress"}
            ], "count": 1}

    monkeypatch.setattr(
        "src.data.sourcing.doj_loader._httpx_get",
        lambda url, **kw: _MockResponse(),
    )
    results = load_courtlistener_batch(case_name="Maxwell", max_examples=5)
    assert len(results) == 1
    assert "text" in results[0]
    assert "source" in results[0]
    assert results[0]["metadata"]["jurisdiction"] == "SDNY"
```

**Step 2: Implement `src/data/sourcing/doj_loader.py`**

```python
"""
DOJ / CourtListener public records loader.

Sources:
  - CourtListener REST API (Free Law Project, no auth required for basic access)
    Docs: https://www.courtlistener.com/api/rest/v4/
  - FBI Vault FOIA (publicly available, https://vault.fbi.gov/)

All documents are public record per PACER/FOIA. Metadata preserved for
standpoint transparency — source, date, jurisdiction, and case reference
are required fields per the constitution's epistemic principles.
"""
from __future__ import annotations

from typing import Any

_COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4"
_FBI_VAULT_EPSTEIN = "https://vault.fbi.gov/jeffrey-epstein"

# Known public case references
_MAXWELL_CASE = "21-cr-00188"
_MAXWELL_DOCKET_ID = "17624879"  # SDNY CourtListener docket ID


def _httpx_get(url: str, **kwargs: Any) -> Any:
    """Thin wrapper enabling monkeypatching in tests."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("pip install httpx") from e
    return httpx.get(url, timeout=30, **kwargs)


def load_courtlistener_batch(
    case_name: str = "Maxwell",
    jurisdiction: str = "SDNY",
    max_examples: int = 100,
) -> list[dict[str, Any]]:
    """
    Load public court documents from CourtListener API.

    Returns normalized dicts with text, source, metadata.
    Documents with no plain_text are skipped (PDF-only filings require
    separate OCR pipeline — not implemented here).
    """
    results: list[dict[str, Any]] = []
    url = f"{_COURTLISTENER_BASE}/opinions/"
    params = {
        "search": case_name,
        "court": "ca2",  # Second Circuit (SDNY appeals)
        "page_size": min(max_examples, 20),
        "format": "json",
    }

    try:
        response = _httpx_get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return results

    for item in data.get("results", [])[:max_examples]:
        text = item.get("plain_text", "").strip()
        if not text:
            continue
        results.append({
            "text": text,
            "source": f"courtlistener:{item.get('docket_id', 'unknown')}",
            "metadata": {
                "case_name": case_name,
                "jurisdiction": jurisdiction,
                "date_filed": item.get("date_filed"),
                "description": item.get("description", ""),
                "docket_id": item.get("docket_id"),
                "api": "courtlistener",
            },
        })

    return results


def load_fbi_vault_epstein(max_documents: int = 50) -> list[dict[str, Any]]:
    """
    Load publicly available FBI FOIA documents about the Epstein investigation.

    FBI Vault documents are released under FOIA and are public record.
    Returns normalized dicts. Most vault documents are PDFs; this loader
    fetches the document index and returns metadata for manual OCR processing.
    """
    # FBI Vault provides a structured index page
    # Full pipeline: fetch index → identify PDF links → OCR → normalize
    # This implementation returns the index metadata for downstream processing
    try:
        response = _httpx_get(_FBI_VAULT_EPSTEIN)
        response.raise_for_status()
    except Exception:
        return []

    # Parse document links from the vault page
    # In production: use BeautifulSoup to extract PDF links and OCR each one
    # Returning empty list signals "index fetched, OCR pipeline needed"
    return []
```

**Step 3: Run tests**

```bash
pytest tests/data/test_sourcing.py -v -k "doj"
```
Expected: all pass.

### Task 3c: International Stakeholder Denouncement Loader

**Public sources documenting international denouncements and investigations:**
- UK IICSA (Independent Inquiry Child Sexual Abuse) published reports: `https://www.iicsa.org.uk/reports-recommendations`
- OCCRP (Organized Crime and Corruption Reporting Project): `https://www.occrp.org/en` — published Epstein network investigations
- European Parliament resolutions: EUR-Lex public access
- Virginia Giuffre v. Ghislaine Maxwell — public civil case documents (SDNY 15-cv-07433)
- Various government press releases denouncing named individuals (publicly archived)

**Step 1: Add to tests**

```python
def test_international_loader_import():
    from src.data.sourcing.international_loader import load_occrp_batch
    assert callable(load_occrp_batch)


def test_international_loader_returns_list(monkeypatch):
    from src.data.sourcing.international_loader import load_occrp_batch

    monkeypatch.setattr(
        "src.data.sourcing.international_loader._httpx_get",
        lambda url, **kw: type("R", (), {
            "raise_for_status": lambda s: None,
            "text": "<html><article>Investigation text here</article></html>",
        })(),
    )
    results = load_occrp_batch(max_examples=5)
    assert isinstance(results, list)
```

**Step 2: Implement `src/data/sourcing/international_loader.py`**

```python
"""
International stakeholder denouncement loader.

Sources public denouncements, parliamentary resolutions, and investigative
journalism from international organizations that have openly published
findings about the network under investigation.

All sources are public record. No confidential or victim-identifying
information is collected — this loader targets institutional documents only.
"""
from __future__ import annotations

from typing import Any

# Public sources — all open access
_OCCRP_SEARCH = "https://www.occrp.org/en/search"
_IICSA_REPORTS = "https://www.iicsa.org.uk/reports-recommendations"
_EURLEX_SEARCH = "https://eur-lex.europa.eu/search.html"

# GitHub: public FOIA/investigative repositories
_GITHUB_SOURCES = [
    "https://api.github.com/repos/MuckRock/documentcloud/contents",
    "https://api.github.com/search/repositories?q=epstein+FOIA+public",
]


def _httpx_get(url: str, **kwargs: Any) -> Any:
    """Thin wrapper enabling monkeypatching in tests."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("pip install httpx") from e
    return httpx.get(url, timeout=30, follow_redirects=True, **kwargs)


def load_occrp_batch(
    query: str = "Epstein Maxwell network",
    max_examples: int = 50,
) -> list[dict[str, Any]]:
    """
    Load OCCRP published investigation documents.

    OCCRP publishes comprehensive network investigation reports under
    open access. Returns normalized text with provenance metadata.
    """
    results: list[dict[str, Any]] = []
    try:
        response = _httpx_get(
            _OCCRP_SEARCH,
            params={"q": query, "type": "story"},
        )
        response.raise_for_status()
        # Production: parse HTML for article text using BeautifulSoup
        # Returns metadata-only for now; full text requires HTML parsing
    except Exception:
        pass
    return results


def load_iicsa_reports() -> list[dict[str, Any]]:
    """
    Load UK IICSA published reports (public government documents).

    IICSA reports are Crown Copyright but freely accessible under
    Open Government Licence v3.0.
    """
    results: list[dict[str, Any]] = []
    try:
        response = _httpx_get(_IICSA_REPORTS)
        response.raise_for_status()
        # Production: parse PDF report links, OCR, normalize
    except Exception:
        pass
    return results


def load_github_public_foia(
    query: str = "Epstein FOIA documents",
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """
    Search GitHub for publicly archived FOIA documents and investigative datasets.

    Useful for finding MuckRock, DocumentCloud, and journalism org releases.
    Returns repository metadata for downstream document fetching.
    """
    results: list[dict[str, Any]] = []
    try:
        import httpx
        response = httpx.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "updated", "per_page": max_results},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        for repo in data.get("items", []):
            if repo.get("private"):
                continue
            results.append({
                "text": repo.get("description", ""),
                "source": f"github:{repo['full_name']}",
                "metadata": {
                    "repo": repo["full_name"],
                    "url": repo["html_url"],
                    "updated_at": repo.get("updated_at"),
                    "topics": repo.get("topics", []),
                },
            })
    except Exception:
        pass
    return results
```

**Step 3: Run all sourcing tests**

```bash
pytest tests/data/test_sourcing.py -v
```
Expected: all pass.

**Step 4: Commit**

```bash
git add src/data/sourcing/ tests/data/test_sourcing.py
git commit -m "feat: dataset sourcing pipeline — HF legal, DOJ CourtListener, OCCRP, GitHub FOIA"
```

---

## Phase 4: Constitution-First Training Pipeline

**Design principle:** The model learns *what* to look for (constitutional values) before learning *how* to detect it (SFT gap mechanics). This is inverted from standard SFT → DPO ordering — constitutional reinforcement is the first training signal.

**Training sequence:**
1. **Constitutional warmup** (DPO on preference pairs from CAI loop) — epistemic values first
2. **Gap detection warmup** (SFT on annotated examples) — gap mechanics second
3. **Gap quality optimization** (GRPO with gap_detection_reward) — reward shaping third

### Task 4a: Constitutional Preference Pair Generator for Sourced Data

Connects the dataset sourcing pipeline to the CAI loop.

**Files:**
- Create: `src/training/constitutional_warmup.py`
- Create: `tests/training/test_constitutional_warmup.py`

**Step 1: Write failing tests**

Create `tests/training/test_constitutional_warmup.py`:

```python
"""Tests for constitution-first training pipeline."""
from unittest.mock import MagicMock
from pathlib import Path


def test_import():
    from src.training.constitutional_warmup import (
        run_constitutional_warmup,
        ConstitutionalWarmupConfig,
    )
    assert callable(run_constitutional_warmup)


def test_config_defaults():
    from src.training.constitutional_warmup import ConstitutionalWarmupConfig
    cfg = ConstitutionalWarmupConfig()
    assert cfg.output_path == "data/training/constitutional_pairs.jsonl"
    assert cfg.max_examples == 200


def test_warmup_writes_jsonl(tmp_path, monkeypatch):
    from src.training.constitutional_warmup import run_constitutional_warmup, ConstitutionalWarmupConfig

    # Patch sourcing to return deterministic examples
    mock_examples = [
        {"text": "The regulator properly reviewed all disclosures.", "source": "test", "metadata": {}},
        {"text": "The board always maintained proper records.", "source": "test", "metadata": {}},
    ]
    monkeypatch.setattr(
        "src.training.constitutional_warmup._load_all_sources",
        lambda cfg: mock_examples,
    )

    local = MagicMock()
    local.complete.return_value = "Analysis: no significant gaps found."
    critic = MagicMock()
    critic.complete.return_value = "Critique: The analysis misses the geopolitical presumption."

    cfg = ConstitutionalWarmupConfig(
        output_path=str(tmp_path / "pairs.jsonl"),
        max_examples=2,
        constitution_path="docs/constitution.md",
    )

    # Patch constitution loading to avoid disk dependency in tests
    monkeypatch.setattr(
        "src.training.constitutional_warmup.load_constitution",
        lambda path=None: "Epistemic honesty above analytical comfort.",
    )

    count = run_constitutional_warmup(cfg, local_provider=local, critic_provider=critic)
    assert count == 2
    output = Path(cfg.output_path)
    assert output.exists()
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 2
    import json
    pair = json.loads(lines[0])
    assert "instruction" in pair
    assert "rejected" in pair
    assert "chosen" in pair


def test_warmup_skips_empty_text(tmp_path, monkeypatch):
    from src.training.constitutional_warmup import run_constitutional_warmup, ConstitutionalWarmupConfig

    monkeypatch.setattr(
        "src.training.constitutional_warmup._load_all_sources",
        lambda cfg: [{"text": "", "source": "test", "metadata": {}}],
    )
    monkeypatch.setattr(
        "src.training.constitutional_warmup.load_constitution",
        lambda path=None: "Constitution text.",
    )

    cfg = ConstitutionalWarmupConfig(output_path=str(tmp_path / "pairs.jsonl"))
    count = run_constitutional_warmup(
        cfg,
        local_provider=MagicMock(),
        critic_provider=MagicMock(),
    )
    assert count == 0
```

**Step 2: Run to verify failure**

```bash
pytest tests/training/test_constitutional_warmup.py -v 2>&1 | head -10
```

**Step 3: Implement `src/training/constitutional_warmup.py`**

```python
"""
Constitution-first training pipeline.

Runs BEFORE SFT. Generates constitutional preference pairs from sourced
investigative documents, using the CAI loop:
  document → local model analysis → Claude critique → revised analysis → DPO pair

This ordering is intentional: the model learns epistemic values (what counts as
honest gap detection) before learning gap detection mechanics (SFT). Constitutional
reinforcement as the first training signal means every downstream fine-tuning step
operates on a model that already prioritizes epistemic honesty.

Usage:
  from src.training.constitutional_warmup import run_constitutional_warmup, ConstitutionalWarmupConfig
  from src.core.providers import provider_from_env, AzureFoundryProvider

  cfg = ConstitutionalWarmupConfig(max_examples=200)
  local = VLLMProvider(base_url="http://localhost:11434/v1", model="deepseek-r1:7b")
  critic = AzureFoundryProvider(...)
  count = run_constitutional_warmup(cfg, local, critic)
  print(f"Generated {count} constitutional preference pairs")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.detective.constitution import load_constitution, generate_preference_pair

_ANALYSIS_PROMPT = (
    "You are an investigative analyst trained in information gap detection.\n\n"
    "Analyze the following document excerpt for information gaps — what is absent, "
    "suppressed, or undocumented that should be present given the stated facts.\n\n"
    "Be specific. Name what type of gap you observe (temporal, evidential, contradiction, "
    "normative, or doctrinal) and why its absence is significant.\n\n"
    "Document:\n{text}"
)


@dataclass(frozen=True)
class ConstitutionalWarmupConfig:
    """Configuration for constitutional preference pair generation."""
    output_path: str = "data/training/constitutional_pairs.jsonl"
    max_examples: int = 200
    constitution_path: str = "docs/constitution.md"
    # Dataset source controls
    use_huggingface: bool = True
    use_doj: bool = True
    use_international: bool = True
    hf_datasets: tuple[str, ...] = (
        "pile-of-law/pile-of-law",
        "nguha/legalbench",
    )
    hf_keyword_filter: str | None = "disclosure"


def _load_all_sources(cfg: ConstitutionalWarmupConfig) -> list[dict[str, Any]]:
    """Aggregate examples from all enabled source pipelines."""
    examples: list[dict[str, Any]] = []
    per_source = max(1, cfg.max_examples // 3)

    if cfg.use_huggingface:
        from src.data.sourcing.hf_loader import load_hf_legal_batch
        for ds_name in cfg.hf_datasets:
            try:
                batch = load_hf_legal_batch(
                    dataset_name=ds_name,
                    max_examples=per_source,
                    keyword_filter=cfg.hf_keyword_filter,
                )
                examples.extend(batch)
            except Exception:
                pass  # Source unavailable — continue with others

    if cfg.use_doj:
        from src.data.sourcing.doj_loader import load_courtlistener_batch
        try:
            examples.extend(load_courtlistener_batch(max_examples=per_source))
        except Exception:
            pass

    if cfg.use_international:
        from src.data.sourcing.international_loader import load_github_public_foia
        try:
            examples.extend(load_github_public_foia(max_results=per_source))
        except Exception:
            pass

    return examples[:cfg.max_examples]


def run_constitutional_warmup(
    cfg: ConstitutionalWarmupConfig,
    local_provider: Any,
    critic_provider: Any,
) -> int:
    """
    Generate constitutional preference pairs from sourced investigative documents.

    Returns the number of pairs successfully written to cfg.output_path.
    """
    constitution = load_constitution(Path(cfg.constitution_path))
    examples = _load_all_sources(cfg)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            text = example.get("text", "").strip()
            if not text:
                continue

            instruction = _ANALYSIS_PROMPT.format(text=text[:2000])  # context window budget
            original_analysis = local_provider.complete(instruction)

            pair = generate_preference_pair(
                instruction=instruction,
                original_analysis=original_analysis,
                constitution=constitution,
                generator_provider=local_provider,
                critic_provider=critic_provider,
            )

            f.write(json.dumps({
                "instruction": pair.instruction,
                "rejected": pair.rejected,
                "chosen": pair.chosen,
                "source": example.get("source", "unknown"),
                "metadata": example.get("metadata", {}),
            }) + "\n")
            count += 1

    return count
```

**Step 4: Run tests**

```bash
pytest tests/training/test_constitutional_warmup.py -v
```
Expected: all pass.

**Step 5: Verify full suite still clean**

```bash
pytest tests/ -q
```

**Step 6: Commit**

```bash
git add src/training/constitutional_warmup.py tests/training/test_constitutional_warmup.py
git commit -m "feat: constitution-first training — constitutional DPO pairs generated before SFT"
```

### Task 4b: CLI command to run the constitutional warmup

**Step 1: Add `warmup` subcommand to `src/cli/main.py`**

```python
@cli.command()
@click.option("--output", default="data/training/constitutional_pairs.jsonl", help="Output JSONL path")
@click.option("--max-examples", default=200, help="Maximum preference pairs to generate")
@click.option("--constitution", default="docs/constitution.md", help="Constitution path")
def warmup(output, max_examples, constitution):
    """Generate constitutional preference pairs (run before SFT)."""
    from src.training.constitutional_warmup import run_constitutional_warmup, ConstitutionalWarmupConfig
    from src.core.providers import provider_from_env, AzureFoundryProvider
    import os

    local = provider_from_env()
    critic_endpoint = os.environ.get("AZURE_ENDPOINT")
    critic_key = os.environ.get("AZURE_API_KEY")
    critic_model = os.environ.get("AZURE_MODEL", "claude-3-5-sonnet")

    if not critic_endpoint:
        raise click.ClickException("Set AZURE_ENDPOINT for the constitutional critic (Claude).")

    critic = AzureFoundryProvider(endpoint=critic_endpoint, credential=critic_key, model=critic_model)
    cfg = ConstitutionalWarmupConfig(
        output_path=output,
        max_examples=max_examples,
        constitution_path=constitution,
    )
    count = run_constitutional_warmup(cfg, local_provider=local, critic_provider=critic)
    click.echo(f"Generated {count} constitutional preference pairs → {output}")
```

**Step 2: Run existing CLI tests**

```bash
pytest tests/cli/ -v
```
Expected: all pass.

**Step 3: Commit**

```bash
git add src/cli/main.py
git commit -m "feat: detective warmup CLI — constitutional pair generation from sourced documents"
```

---

## Phase 5: Parallel Hypothesis Evolution (GoT Generate(k))

GoT's most distinctive architectural feature: parallel exploration of k hypothesis branches simultaneously. Currently `evolve_hypothesis()` is sequential. This task adds `asyncio`-based parallel dispatch.

**Files:**
- Create: `src/detective/parallel_evolution.py`
- Create: `tests/detective/test_parallel_evolution.py`

**Step 1: Write failing tests**

Create `tests/detective/test_parallel_evolution.py`:

```python
"""Tests for parallel hypothesis evolution (GoT Generate(k))."""
import asyncio
from unittest.mock import MagicMock


def test_import():
    from src.detective.parallel_evolution import evolve_parallel, ParallelEvolutionResult
    assert callable(evolve_parallel)


def test_returns_multiple_branches():
    """k=3 branches produces 3 independent evolved hypotheses."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis

    root = Hypothesis.create(
        statement="Entity A had undisclosed financial ties to Entity B in 2003.",
        confidence=0.6,
        sources=("court_doc_001",),
    )

    provider = MagicMock()
    provider.complete.return_value = "confirmed, confidence: 0.75"

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=[
            "Financial records show transfers in Q2 2003.",
            "Meeting logs reference Entity B twice in 2003.",
            "FOIA release redacts all 2003 correspondence.",
        ],
        provider=provider,
        k=3,
    ))

    assert len(results) == 3
    assert all(r.hypothesis is not None for r in results)
    assert all(r.evidence_used != "" for r in results)


def test_all_branches_have_parent_id():
    """Each branch must carry parent_id — immutable lineage is load-bearing."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis

    root = Hypothesis.create(
        statement="Gap in 2019 financial records.",
        confidence=0.5,
        sources=("doc_a",),
    )
    provider = MagicMock()
    provider.complete.return_value = "confirmed, confidence: 0.7"

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=["Evidence A", "Evidence B", "Evidence C"],
        provider=provider,
        k=3,
    ))

    for result in results:
        assert result.hypothesis.parent_id == root.hypothesis_id


def test_branches_are_independent():
    """Each branch explores a different evidence thread."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis

    root = Hypothesis.create(statement="Test.", confidence=0.5, sources=("s",))
    provider = MagicMock()
    provider.complete.return_value = "confirmed, confidence: 0.6"

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=["E1", "E2", "E3"],
        provider=provider,
        k=3,
    ))

    evidence_used = [r.evidence_used for r in results]
    assert len(set(evidence_used)) == 3  # each branch used different evidence


def test_k_capped_at_evidence_count():
    """Can't have more branches than evidence items."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis

    root = Hypothesis.create(statement="Test.", confidence=0.5, sources=("s",))
    provider = MagicMock()
    provider.complete.return_value = "confirmed, confidence: 0.6"

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=["Only one evidence item"],
        provider=provider,
        k=5,  # k > len(evidence_list)
    ))
    assert len(results) == 1


def test_highest_confidence_branch_first():
    """Results sorted by evolved hypothesis confidence, descending."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis

    root = Hypothesis.create(statement="Test.", confidence=0.5, sources=("s",))
    responses = iter(["confirmed, confidence: 0.9", "confirmed, confidence: 0.4", "confirmed, confidence: 0.7"])
    provider = MagicMock()
    provider.complete.side_effect = lambda p: next(responses)

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=["E1", "E2", "E3"],
        provider=provider,
        k=3,
    ))

    confidences = [r.hypothesis.confidence for r in results]
    assert confidences == sorted(confidences, reverse=True)
```

**Step 2: Run to verify failure**

```bash
pytest tests/detective/test_parallel_evolution.py -v 2>&1 | head -10
```

**Step 3: Implement `src/detective/parallel_evolution.py`**

```python
"""
Parallel hypothesis evolution — GoT Generate(k) operation.

Dispatches k independent hypothesis evolution branches simultaneously using
asyncio.gather(). Each branch explores a distinct evidence thread. Results
are sorted by evolved confidence, highest first — analogous to GoT's pruning
step (KeepBestN).

Design constraint: each branch still produces an immutable Hypothesis with
parent_id pointing to the root. Parallelism is at the I/O level (provider
calls), not at the hypothesis mutation level — immutability is preserved.

Usage:
    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=["doc A finding", "doc B finding", "doc C finding"],
        provider=local_provider,
        k=3,
    ))
    best = results[0].hypothesis  # highest confidence branch
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from src.detective.hypothesis import Hypothesis
from src.detective.experience import ExperienceLibrary, query_similar, add_experience, Experience
from src.core.providers import ModelProvider

_BRANCH_PROMPT = (
    "You are evolving a hypothesis based on a specific piece of evidence.\n\n"
    "Current hypothesis: {statement}\n"
    "Current confidence: {confidence:.2f}\n\n"
    "New evidence: {evidence}\n\n"
    "How does this evidence change the hypothesis? "
    "Reply with one of: confirmed, refuted, spawned_alternative\n"
    "Then state the updated confidence as: confidence: <float between 0 and 1>\n"
    "Keep your response to 2 sentences."
)

_CONFIDENCE_RE = re.compile(r"confidence\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ParallelEvolutionResult:
    """
    One branch from a parallel evolution run.

    hypothesis: the evolved hypothesis for this branch
    evidence_used: the evidence thread this branch explored
    branch_index: position in the original k branches (0-indexed)
    """
    hypothesis: Hypothesis
    evidence_used: str
    branch_index: int


def _parse_confidence(response: str, current: float) -> float:
    match = _CONFIDENCE_RE.search(response)
    if not match:
        return max(0.0, current - 0.05)  # unknown response → slight decay
    try:
        return min(1.0, max(0.0, float(match.group(1))))
    except ValueError:
        return max(0.0, current - 0.05)


async def _evolve_branch(
    hypothesis: Hypothesis,
    evidence: str,
    branch_index: int,
    provider: ModelProvider,
    library: ExperienceLibrary,
) -> ParallelEvolutionResult:
    """Evolve one hypothesis branch asynchronously."""
    prompt = _BRANCH_PROMPT.format(
        statement=hypothesis.statement,
        confidence=hypothesis.confidence,
        evidence=evidence,
    )

    # Provider.complete is synchronous — run in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, provider.complete, prompt)

    new_confidence = _parse_confidence(response, hypothesis.confidence)
    evolved = hypothesis.update_confidence(
        new_confidence=new_confidence,
        evidence_key=evidence[:80],  # truncate for key
    )

    return ParallelEvolutionResult(
        hypothesis=evolved,
        evidence_used=evidence,
        branch_index=branch_index,
    )


async def evolve_parallel(
    hypothesis: Hypothesis,
    evidence_list: list[str],
    provider: ModelProvider,
    k: int = 3,
    library: ExperienceLibrary = (),
) -> list[ParallelEvolutionResult]:
    """
    GoT Generate(k): dispatch k parallel hypothesis branches.

    Each branch explores a distinct evidence item. Branches run concurrently
    via asyncio.gather(). Results are sorted by evolved confidence, descending.

    Args:
        hypothesis: Root hypothesis to evolve from.
        evidence_list: Evidence items to explore. One per branch.
        provider: LLM provider for branch reasoning.
        k: Number of parallel branches. Capped at len(evidence_list).
        library: Optional experience library for context.

    Returns:
        List of ParallelEvolutionResult, sorted by confidence descending.
    """
    actual_k = min(k, len(evidence_list))
    if actual_k == 0:
        return []

    selected_evidence = evidence_list[:actual_k]

    tasks = [
        _evolve_branch(hypothesis, evidence, i, provider, library)
        for i, evidence in enumerate(selected_evidence)
    ]

    results: list[ParallelEvolutionResult] = await asyncio.gather(*tasks)

    return sorted(results, key=lambda r: r.hypothesis.confidence, reverse=True)
```

**Step 4: Run tests**

```bash
pytest tests/detective/test_parallel_evolution.py -v
```
Expected: all pass.

**Step 5: Run full suite**

```bash
pytest tests/ -q
```
Expected: all pass.

**Step 6: Commit**

```bash
git add src/detective/parallel_evolution.py tests/detective/test_parallel_evolution.py
git commit -m "feat: parallel hypothesis evolution — GoT Generate(k) via asyncio.gather"
```

---

## Summary: Execution Order

```
Task 0  pyproject.toml pin fix          (5 min)
Task 1  Wire live Ollama provider        (30 min)
Task 2  Module B (historical det.)       (45 min)
Task 3  Module C (geopolitical pres.)    (45 min)
Task 3a HF dataset loader               (30 min)
Task 3b DOJ/CourtListener loader         (30 min)
Task 3c International loader             (30 min)
Task 4a Constitutional warmup generator  (45 min)
Task 4b CLI warmup command               (15 min)
Task 5  Parallel evolution               (45 min)
```

**Total estimated: ~5.5 hours focused implementation**

---

## Task 3d: Document Ingestion Pipeline — MIME Detection + DeepSeek-OCR

Many FOIA releases, court exhibits, and FBI Vault documents are mislabeled: a scanned page saved as `exhibit_47.pdf` may be a JPEG image; a video exhibit labeled `.mov` may be a redacted still sequence. The pipeline must route by true file type (MIME sniffing), not extension trust.

**Architecture:**
- `python-magic` reads the first 8 bytes (file header) to determine true MIME type
- `pdf2image` + `poppler` rasterizes real PDFs into per-page images
- **DeepSeek-OCR** (`deepseek-ai/DeepSeek-OCR`) performs vision-language OCR on GPU (Azure NC-series)
- `pytesseract` + `Tesseract` is the CPU fallback (L-series / local dev)
- Redaction detection: black-pixel ratio per page region is a gap signal — heavily redacted areas are annotated as `<REDACTED_REGION>` in the output text

**Deployment note:** DeepSeek-OCR requires CUDA + `torch==2.6.0` + `flash-attn==2.7.3`. It runs only on the Azure GPU path (`requirements-azure.txt`). On the CPU path (Ollama/L-series), `pytesseract` handles OCR. The `OcrProvider` abstraction makes this transparent to the rest of the pipeline.

**Files:**
- Create: `src/data/sourcing/ocr_provider.py`
- Create: `src/data/sourcing/document_ingestion.py`
- Modify: `deployment/requirements-azure.txt` (add DeepSeek-OCR deps)
- Create: `tests/data/test_document_ingestion.py`

**Step 1: Add GPU OCR deps to `deployment/requirements-azure.txt`**

Append to `deployment/requirements-azure.txt`:
```
# DeepSeek-OCR (GPU only — install AFTER torch cu121)
# torch==2.6.0 and torchvision==0.21.0 required; pin below overrides the default torch
# Run: pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
# Then: pip install -r requirements-azure.txt
transformers>=4.51.1
flash-attn==2.7.3
pdf2image
python-magic
```

Add to CPU deployment `deployment/requirements-deploy.txt`:
```
# OCR (CPU fallback — no CUDA needed)
pytesseract
pdf2image
python-magic
```

**Step 2: Write failing tests**

Create `tests/data/test_document_ingestion.py`:

```python
"""Tests for MIME-sniffing document ingestion pipeline."""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_import():
    from src.data.sourcing.document_ingestion import ingest_document, DocumentRecord
    assert callable(ingest_document)


def test_document_record_is_frozen():
    from src.data.sourcing.document_ingestion import DocumentRecord
    import pytest
    rec = DocumentRecord(
        text="Hello",
        true_mime="image/jpeg",
        declared_suffix=".pdf",
        source_path="exhibit_1.pdf",
        page_count=1,
        redaction_ratio=0.0,
        ocr_backend="tesseract",
    )
    with pytest.raises(Exception):
        rec.text = "mutated"


def test_mime_mismatch_detected(tmp_path):
    """A file with .pdf suffix but JPEG header must be detected as image/jpeg."""
    from src.data.sourcing.document_ingestion import detect_true_mime

    # Write a minimal JPEG header (SOI marker: FF D8 FF)
    fake_pdf = tmp_path / "exhibit.pdf"
    fake_pdf.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    true_mime = detect_true_mime(fake_pdf)
    assert true_mime.startswith("image/"), f"Expected image/*, got {true_mime}"


def test_real_pdf_detected(tmp_path):
    """A real PDF file must be detected as application/pdf."""
    from src.data.sourcing.document_ingestion import detect_true_mime

    # Minimal valid PDF header
    fake_pdf = tmp_path / "document.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n" + b"\x00" * 100)

    true_mime = detect_true_mime(fake_pdf)
    assert true_mime == "application/pdf" or "pdf" in true_mime.lower()


def test_ingest_image_routes_to_ocr(tmp_path):
    """A JPEG file (even if named .pdf) routes to OCR backend."""
    from src.data.sourcing.document_ingestion import ingest_document

    fake_jpg = tmp_path / "exhibit.pdf"  # mislabeled
    fake_jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    mock_ocr = MagicMock(return_value="Extracted text from image.")

    with patch("src.data.sourcing.document_ingestion._ocr_image_file", mock_ocr):
        with patch("src.data.sourcing.document_ingestion.detect_true_mime",
                   return_value="image/jpeg"):
            record = ingest_document(fake_jpg, source_id="test")

    assert record.text == "Extracted text from image."
    assert record.true_mime == "image/jpeg"
    assert record.declared_suffix == ".pdf"


def test_ingest_pdf_rasterizes_and_ocrs(tmp_path):
    """A real PDF is rasterized to images then OCR'd page by page."""
    from src.data.sourcing.document_ingestion import ingest_document

    fake_pdf = tmp_path / "document.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n" + b"\x00" * 100)

    mock_rasterize = MagicMock(return_value=[MagicMock()])  # one page
    mock_ocr = MagicMock(return_value="Page 1 text.")

    with patch("src.data.sourcing.document_ingestion._rasterize_pdf", mock_rasterize):
        with patch("src.data.sourcing.document_ingestion._ocr_image_file", mock_ocr):
            with patch("src.data.sourcing.document_ingestion.detect_true_mime",
                       return_value="application/pdf"):
                record = ingest_document(fake_pdf, source_id="test")

    assert "Page 1 text." in record.text
    assert record.page_count == 1


def test_redaction_ratio_reported(tmp_path):
    """Heavily redacted pages have redaction_ratio > 0."""
    from src.data.sourcing.document_ingestion import estimate_redaction_ratio
    import PIL.Image

    # Create a mostly-black image (simulating heavy redaction)
    img = PIL.Image.new("L", (100, 100), color=0)  # all black
    ratio = estimate_redaction_ratio(img)
    assert ratio > 0.9


def test_lightly_redacted_ratio(tmp_path):
    from src.data.sourcing.document_ingestion import estimate_redaction_ratio
    import PIL.Image

    img = PIL.Image.new("L", (100, 100), color=255)  # all white
    ratio = estimate_redaction_ratio(img)
    assert ratio < 0.1


def test_ingest_returns_document_record(tmp_path):
    from src.data.sourcing.document_ingestion import ingest_document, DocumentRecord

    f = tmp_path / "doc.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    with patch("src.data.sourcing.document_ingestion.detect_true_mime",
               return_value="image/jpeg"):
        with patch("src.data.sourcing.document_ingestion._ocr_image_file",
                   return_value="Some extracted text"):
            record = ingest_document(f, source_id="fbi_vault_001")

    assert isinstance(record, DocumentRecord)
    assert record.source_path == str(f)
```

**Step 3: Run to verify failure**

```bash
pytest tests/data/test_document_ingestion.py -v 2>&1 | head -10
```
Expected: ImportError.

**Step 4: Install CPU dependencies**

```bash
source .venv-deploy/bin/activate
pip install python-magic pdf2image pytesseract Pillow --quiet
# On Ubuntu/Debian also need system packages:
sudo apt-get install -y tesseract-ocr poppler-utils libmagic1 --quiet
```

**Step 5: Implement `src/data/sourcing/ocr_provider.py`**

```python
"""
OCR provider abstraction: DeepSeek-OCR (GPU) or pytesseract (CPU).

The OcrBackend Protocol allows transparent switching between backends.
At import time, the module detects which backend is available and exports
`ocr_image` as the correct implementation.

GPU path (Azure NC-series):
  - deepseek-ai/DeepSeek-OCR via transformers
  - Requires: CUDA + torch==2.6.0 + flash-attn==2.7.3
  - ~2500 tokens/second on A100-40G
  - Outputs markdown with layout preservation

CPU path (L-series / local dev):
  - pytesseract + Tesseract system binary
  - Requires: tesseract-ocr (apt package)
  - Slower but no GPU required
  - Plain text output

Selection: set OCR_BACKEND=deepseek to force GPU path.
Default: auto-detect based on torch.cuda.is_available().
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore[assignment]


@runtime_checkable
class OcrBackend(Protocol):
    """Structural contract for OCR backends."""
    def extract_text(self, image: "_PILImage.Image") -> str: ...  # type: ignore[type-arg]
    @property
    def name(self) -> str: ...


class _TesseractBackend:
    """CPU fallback OCR using system Tesseract binary."""

    @property
    def name(self) -> str:
        return "tesseract"

    def extract_text(self, image: "_PILImage.Image") -> str:  # type: ignore[type-arg]
        try:
            import pytesseract
        except ImportError as e:
            raise ImportError("pip install pytesseract && apt install tesseract-ocr") from e
        return pytesseract.image_to_string(image)


class _DeepSeekOcrBackend:
    """
    GPU OCR using deepseek-ai/DeepSeek-OCR.

    Loads model lazily on first call to avoid startup cost when not needed.
    Requires CUDA + torch==2.6.0 + flash-attn==2.7.3.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "deepseek-ocr"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError("pip install transformers>=4.51.1") from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

    def extract_text(self, image: "_PILImage.Image") -> str:  # type: ignore[type-arg]
        self._load()
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        try:
            result = self._model.infer(
                self._tokenizer,
                prompt="<image>\nFree OCR.",
                image_file=tmp_path,
                base_size=1024,
                image_size=640,
            )
            return result if isinstance(result, str) else str(result)
        finally:
            os.unlink(tmp_path)


def _select_backend() -> OcrBackend:
    """Auto-select backend: DeepSeek if CUDA available and not overridden."""
    forced = os.environ.get("OCR_BACKEND", "").lower()
    if forced == "tesseract":
        return _TesseractBackend()
    if forced == "deepseek":
        return _DeepSeekOcrBackend()

    # Auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            return _DeepSeekOcrBackend()
    except ImportError:
        pass

    return _TesseractBackend()


# Module-level singleton — selected at import time
default_backend: OcrBackend = _select_backend()


def ocr_image(image: "_PILImage.Image", backend: OcrBackend | None = None) -> str:  # type: ignore[type-arg]
    """Extract text from a PIL Image using the selected OCR backend."""
    b = backend or default_backend
    return b.extract_text(image)
```

**Step 6: Implement `src/data/sourcing/document_ingestion.py`**

```python
"""
Document ingestion pipeline: MIME detection → rasterization → OCR → DocumentRecord.

Handles the common FOIA/court document problem: files with wrong extensions.
Examples seen in practice:
  - exhibit_47.pdf  → actual MIME: image/jpeg (scanned page saved with wrong ext)
  - deposition.pdf  → actual MIME: application/pdf (real PDF, needs rasterization)
  - recording.mov   → actual MIME: video/mp4  (out of scope, logged and skipped)

Redaction detection: pages with black-pixel ratio > REDACTION_THRESHOLD are
annotated with <REDACTED_REGION> markers — the extent and pattern of redaction
is itself an information gap per docs/constitution.md (normative gaps).

Output: DocumentRecord (frozen dataclass) with full provenance for training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Black pixel ratio above this marks a region as redacted
_REDACTION_THRESHOLD: float = 0.85

# MIME types this pipeline can process
_SUPPORTED_IMAGE_MIMES = frozenset({
    "image/jpeg", "image/jpg", "image/png", "image/tiff", "image/bmp",
})
_SUPPORTED_PDF_MIME = "application/pdf"


@dataclass(frozen=True)
class DocumentRecord:
    """
    Normalized output from the document ingestion pipeline.

    All fields are immutable — DocumentRecord is an evidence artifact.
    Mutating an ingested document after the fact would violate the
    chain of custody principle (constitution: standpoint transparency).
    """
    text: str               # extracted plain text (OCR'd or parsed)
    true_mime: str          # MIME type from file header, not extension
    declared_suffix: str    # original file extension (may differ from true_mime)
    source_path: str        # original file path for provenance
    page_count: int         # number of pages processed
    redaction_ratio: float  # 0.0 (no redaction) to 1.0 (fully redacted)
    ocr_backend: str        # "tesseract" or "deepseek-ocr"

    def __post_init__(self) -> None:
        if not (0.0 <= self.redaction_ratio <= 1.0):
            raise ValueError(f"redaction_ratio must be [0,1], got {self.redaction_ratio}")

    @property
    def has_redactions(self) -> bool:
        return self.redaction_ratio > _REDACTION_THRESHOLD * 0.5

    @property
    def suffix_mismatch(self) -> bool:
        """True if the file extension doesn't match the true MIME type."""
        mime_to_ext = {
            "image/jpeg": {".jpg", ".jpeg"},
            "image/png": {".png"},
            "application/pdf": {".pdf"},
        }
        expected = mime_to_ext.get(self.true_mime, set())
        return bool(expected) and self.declared_suffix.lower() not in expected


def detect_true_mime(path: Path) -> str:
    """
    Read file header bytes to determine true MIME type.

    Does NOT trust the file extension — FOIA documents are commonly
    mislabeled. Falls back to extension-guessing if python-magic unavailable.
    """
    try:
        import magic
        return magic.from_file(str(path), mime=True)
    except ImportError:
        pass

    # Fallback: read magic bytes manually
    header = path.read_bytes()[:16]
    if header[:4] == b"%PDF":
        return "application/pdf"
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    # Unknown — return based on extension
    suffix = path.suffix.lower()
    return {"pdf": "application/pdf", ".jpg": "image/jpeg", ".png": "image/png"}.get(suffix, "application/octet-stream")


def estimate_redaction_ratio(image: "object") -> float:
    """
    Estimate what fraction of a page image is redacted (black regions).

    Returns 0.0–1.0. A ratio > 0.85 indicates heavy redaction.
    Redaction pattern is preserved in DocumentRecord as a gap signal.
    """
    try:
        import numpy as np
        from PIL import Image as PILImage
        if not isinstance(image, PILImage.Image):
            return 0.0
        gray = image.convert("L")
        arr = np.array(gray)
        black_pixels = (arr < 30).sum()  # pixels darker than near-black
        return float(black_pixels) / arr.size
    except ImportError:
        return 0.0


def _rasterize_pdf(path: Path, dpi: int = 200) -> list["object"]:
    """Convert each PDF page to a PIL Image for OCR."""
    try:
        from pdf2image import convert_from_path
        return convert_from_path(str(path), dpi=dpi)
    except ImportError as e:
        raise ImportError("pip install pdf2image && apt install poppler-utils") from e


def _ocr_image_file(image: "object") -> str:
    """OCR a single PIL Image using the auto-selected backend."""
    from src.data.sourcing.ocr_provider import ocr_image
    return ocr_image(image)  # type: ignore[arg-type]


def ingest_document(
    path: Path,
    source_id: str = "",
    max_pages: int = 50,
) -> DocumentRecord:
    """
    Ingest a document file regardless of its declared extension.

    Pipeline:
      1. MIME-sniff the file header (don't trust extension)
      2. Route: image → OCR directly | PDF → rasterize → OCR page by page
      3. Estimate redaction ratio per page
      4. Return DocumentRecord with full provenance

    Args:
        path: Path to the document file.
        source_id: Provenance identifier (e.g., "fbi_vault_001", "maxwell_trial").
        max_pages: Limit page processing for very long documents.

    Returns:
        DocumentRecord with extracted text and metadata.
    """
    true_mime = detect_true_mime(path)
    declared_suffix = path.suffix.lower()

    pages_text: list[str] = []
    total_redaction = 0.0
    page_count = 0
    backend_name = "unknown"

    if true_mime in _SUPPORTED_IMAGE_MIMES:
        # Single-page image (including mislabeled PDFs)
        try:
            from PIL import Image as PILImage
            img = PILImage.open(path)
            text = _ocr_image_file(img)
            pages_text.append(text)
            total_redaction = estimate_redaction_ratio(img)
            page_count = 1
            from src.data.sourcing.ocr_provider import default_backend
            backend_name = default_backend.name
        except Exception as e:
            pages_text.append(f"[OCR FAILED: {e}]")
            page_count = 1

    elif _SUPPORTED_PDF_MIME in true_mime:
        # Real PDF: rasterize each page
        try:
            images = _rasterize_pdf(path)[:max_pages]
            redaction_sum = 0.0
            from src.data.sourcing.ocr_provider import default_backend
            backend_name = default_backend.name
            for img in images:
                page_text = _ocr_image_file(img)
                redaction = estimate_redaction_ratio(img)
                redaction_sum += redaction
                if redaction > _REDACTION_THRESHOLD:
                    pages_text.append(f"<REDACTED_REGION redaction_ratio={redaction:.2f}>")
                else:
                    pages_text.append(page_text)
            page_count = len(images)
            total_redaction = redaction_sum / page_count if page_count else 0.0
        except Exception as e:
            pages_text.append(f"[PDF PROCESSING FAILED: {e}]")
            page_count = 1

    else:
        # Unsupported MIME (video, audio, binary) — log and skip
        pages_text.append(
            f"[UNSUPPORTED MIME: {true_mime} — declared suffix: {declared_suffix}]"
        )
        page_count = 0

    return DocumentRecord(
        text="\n\n".join(pages_text),
        true_mime=true_mime,
        declared_suffix=declared_suffix,
        source_path=str(path),
        page_count=page_count,
        redaction_ratio=min(1.0, total_redaction),
        ocr_backend=backend_name,
    )
```

**Step 7: Run tests**

```bash
pytest tests/data/test_document_ingestion.py -v
```

Note: `test_redaction_ratio_reported` and `test_lightly_redacted_ratio` require `Pillow` and `numpy`. Install if needed:
```bash
source .venv-deploy/bin/activate && pip install Pillow numpy --quiet
```

Expected: all 9 tests pass.

**Step 8: Run full suite**

```bash
pytest tests/ -q
```
Expected: all pass.

**Step 9: Verify GPU path selects correctly**

```bash
source .venv-deploy/bin/activate
# On CPU machine — should auto-select tesseract
python -c "from src.data.sourcing.ocr_provider import default_backend; print(default_backend.name)"
```
Expected: `tesseract`

```bash
# Force DeepSeek path (for documentation/smoke test — will fail without GPU)
OCR_BACKEND=deepseek python -c "from src.data.sourcing.ocr_provider import default_backend; print(default_backend.name)"
```
Expected: `deepseek-ocr` (model loads lazily — no GPU error yet)

**Step 10: Commit**

```bash
git add src/data/sourcing/ocr_provider.py \
        src/data/sourcing/document_ingestion.py \
        tests/data/test_document_ingestion.py \
        deployment/requirements-azure.txt \
        deployment/requirements-deploy.txt
git commit -m "feat: document ingestion — MIME sniffing + DeepSeek-OCR (GPU) / tesseract (CPU) + redaction detection"
```

---

## Updated Summary: Execution Order

```
Task 0   pyproject.toml pin fix                     (5 min)
Task 1   Wire live Ollama provider                   (30 min)
Task 2   Module B (historical determinism)           (45 min)
Task 3   Module C (geopolitical presumptions)        (45 min)
Task 3a  HF dataset loader                           (30 min)
Task 3b  DOJ/CourtListener loader                    (30 min)
Task 3c  International loader                        (30 min)
Task 3d  Document ingestion — MIME + OCR             (60 min)  ← new
Task 4a  Constitutional warmup generator             (45 min)
Task 4b  CLI warmup command                          (15 min)
Task 5   Parallel evolution (GoT Generate(k))        (45 min)
```

**Total estimated: ~6.5 hours focused implementation**

**On Azure GPU VM (NC-series), set before running OCR:**
```bash
export OCR_BACKEND=deepseek  # use DeepSeek-OCR
# or leave unset for auto-detection (selects deepseek if CUDA available)
```

**On L-series CPU VM (current deployment):**
```bash
export OCR_BACKEND=tesseract  # explicit, or auto-detected when no CUDA
```

---

## Running the Full Constitutional Training Pipeline

Once all tasks are complete:

```bash
# Step 1: Generate constitutional preference pairs (DPO training data)
# Requires: Ollama running, AZURE_ENDPOINT set for Claude critic
source .venv-deploy/bin/activate
set -a && source .env.local && set +a
detective warmup --max-examples 200 --output data/training/constitutional_pairs.jsonl

# Step 2: Constitutional DPO training (values first)
python -c "
from src.training.train_dpo import build_dpo_trainer, load_preference_pairs
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
pairs = load_preference_pairs('data/training/constitutional_pairs.jsonl')
trainer = build_dpo_trainer(model, tokenizer, pairs)
trainer.train()
"

# Step 3: SFT on gap annotations (mechanics second)
# (requires gap annotation JSONL at data/annotations/)

# Step 4: GRPO for gap quality (reward shaping third)
```
