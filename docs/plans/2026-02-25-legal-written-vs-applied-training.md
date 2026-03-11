# Legal "Written vs. Applied" DPO Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train the model to detect DOCTRINAL gaps — where law as written diverges from law as applied — across criminal justice, territorial rights, and FOIA/transparency domains.

**Architecture:** Extend the existing CAI pipeline with a legal-focused analysis prompt that explicitly targets statute-vs-enforcement gaps. Generate ~600 domain-tagged DPO preference pairs (200 per legal domain), then run a single combined DPO training job on HF Jobs continuing from the existing LoRA adapter at `crichalchemist/detective-llm-dpo-lora`.

**Tech Stack:** HuggingFace `datasets`, `trl` DPOTrainer, `peft` LoRA, existing CAI pipeline (`constitutional_warmup.py` + `constitution.py`), HF Jobs A10G GPU, `pile-of-law/pile-of-law` dataset, CourtListener API, Trackio monitoring.

---

## Background

The constitution (`docs/constitution.md:169-187`) already defines the legal grounding framework:

- **Law as written** — statutes, regulations, constitutional text, court holdings
- **Law as applied** — enforcement patterns, prosecutorial discretion, sentencing outcomes, access to counsel, territorial/tribal distinctions

The `DOCTRINAL` gap type (`src/core/types.py:12`) captures this exactly: "Unstated institutional rules assumed to apply that may not have." The `LegalDomain` enum (`src/core/types.py:28-35`) provides the vocabulary: `STATUTE`, `ENFORCEMENT_PRACTICE`, `COMMUNITY_EXPERIENCE`, `TERRITORIAL`.

The existing 200-pair training dataset (`data/training/constitutional_pairs.jsonl`) trained the model on general gap detection from autobiography/memoir sources. This plan extends that to legal-specific documents where the written-vs-applied gap is the primary investigative target.

## Three Legal Domains

| Domain | Source Data | Written | Applied |
|--------|-----------|---------|---------|
| **Criminal Justice** | `pile-of-law` sentencing opinions, CourtListener SDNY | Sentencing guidelines, 4th/5th/6th Amendment text | Disparate sentencing, plea coercion, denial of counsel |
| **Territorial Rights** | `pile-of-law` Insular Cases, territorial statutes | Equal protection, federal program eligibility | Insular Cases doctrine, reduced protections in PR/Guam/USVI |
| **FOIA/Transparency** | `pile-of-law` FOIA litigation, MuckRock archives | FOIA statute (5 U.S.C. § 552), disclosure mandates | Systematic denial, Glomar responses, years-long delays |

---

## Task 0: Create the Legal Analysis Prompt

**Files:**
- Modify: `src/training/constitutional_warmup.py:36-43`
- Test: `tests/training/test_constitutional_warmup.py` (new test)

**Step 1: Write failing test for legal-focused prompt**

```python
# tests/training/test_legal_prompt.py

def test_legal_analysis_prompt_contains_doctrinal_framing():
    from src.training.constitutional_warmup import build_legal_analysis_prompt
    prompt = build_legal_analysis_prompt("Some statute text here")
    assert "law as written" in prompt.lower()
    assert "law as applied" in prompt.lower()
    assert "doctrinal" in prompt.lower()
    assert "Some statute text here" in prompt


def test_legal_analysis_prompt_names_gap_types():
    from src.training.constitutional_warmup import build_legal_analysis_prompt
    prompt = build_legal_analysis_prompt("text")
    assert "enforcement" in prompt.lower()
    assert "statute" in prompt.lower()


def test_legal_analysis_prompt_includes_domain_tag():
    from src.training.constitutional_warmup import build_legal_analysis_prompt
    prompt = build_legal_analysis_prompt("text", domain="criminal_justice")
    assert "criminal_justice" in prompt or "criminal justice" in prompt.lower()


def test_original_analysis_prompt_still_works():
    """Regression: generic prompt unchanged."""
    from src.training.constitutional_warmup import _ANALYSIS_PROMPT
    prompt = _ANALYSIS_PROMPT.format(text="test doc")
    assert "information gaps" in prompt.lower()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/training/test_legal_prompt.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_legal_analysis_prompt'`

**Step 3: Implement the legal analysis prompt**

Add to `src/training/constitutional_warmup.py` (after `_ANALYSIS_PROMPT` on line 43):

```python
_LEGAL_ANALYSIS_PROMPT = (
    "You are an investigative analyst specializing in the gap between "
    "law as written and law as applied.\n\n"
    "Analyze the following legal document for DOCTRINAL gaps — places where "
    "the formal legal text (statutes, regulations, court holdings) diverges from "
    "actual enforcement practice, prosecutorial discretion, or lived community experience.\n\n"
    "For each gap found:\n"
    "  1. Name the gap type: doctrinal, normative, temporal, evidential, or contradiction.\n"
    "  2. Quote or cite the specific statutory/regulatory provision at stake.\n"
    "  3. Describe the enforcement reality that diverges from it.\n"
    "  4. Name the affected population and jurisdictional layer "
    "(federal, state, territorial, tribal).\n\n"
    "Domain: {domain}\n\n"
    "Document:\n{text}"
)


def build_legal_analysis_prompt(text: str, domain: str = "general") -> str:
    """Build a legal-focused analysis prompt targeting written-vs-applied gaps."""
    return _LEGAL_ANALYSIS_PROMPT.format(text=text[:2000], domain=domain)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_legal_prompt.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add tests/training/test_legal_prompt.py src/training/constitutional_warmup.py
git commit -m "feat(legal): add doctrinal gap analysis prompt for written-vs-applied training"
```

---

## Task 1: Add Domain-Specific HF Dataset Configurations

**Files:**
- Modify: `src/data/sourcing/hf_loader.py:17-57`
- Create: `src/data/sourcing/legal_sources.py`
- Test: `tests/data/sourcing/test_legal_sources.py`

**Step 1: Write failing tests**

```python
# tests/data/sourcing/test_legal_sources.py

def test_legal_domain_configs_has_three_domains():
    from src.data.sourcing.legal_sources import LEGAL_DOMAIN_CONFIGS
    assert "criminal_justice" in LEGAL_DOMAIN_CONFIGS
    assert "territorial_rights" in LEGAL_DOMAIN_CONFIGS
    assert "foia_transparency" in LEGAL_DOMAIN_CONFIGS


def test_each_domain_has_required_keys():
    from src.data.sourcing.legal_sources import LEGAL_DOMAIN_CONFIGS
    for domain, config in LEGAL_DOMAIN_CONFIGS.items():
        assert "hf_dataset" in config, f"{domain} missing hf_dataset"
        assert "hf_config" in config, f"{domain} missing hf_config"
        assert "keyword_filters" in config, f"{domain} missing keyword_filters"
        assert "description" in config, f"{domain} missing description"


def test_load_legal_domain_batch_returns_list():
    from unittest.mock import patch, MagicMock
    from src.data.sourcing.legal_sources import load_legal_domain_batch

    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(return_value=iter([
        {"text": "The sentencing guidelines require..." * 10},
    ]))

    with patch("src.data.sourcing.legal_sources.load_dataset", return_value=mock_ds):
        results = load_legal_domain_batch("criminal_justice", max_examples=5)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["source"].startswith("huggingface:")
    assert results[0]["metadata"]["legal_domain"] == "criminal_justice"


def test_load_legal_domain_batch_unknown_domain():
    from src.data.sourcing.legal_sources import load_legal_domain_batch
    import pytest
    with pytest.raises(ValueError, match="Unknown legal domain"):
        load_legal_domain_batch("unknown_domain")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/data/sourcing/test_legal_sources.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement legal source configurations**

```python
# src/data/sourcing/legal_sources.py
"""
Legal domain source configurations for written-vs-applied DPO training.

Each domain maps to HuggingFace dataset configs and keyword filters that
surface documents where statute-enforcement gaps are likely present.
"""
from __future__ import annotations

from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]


LEGAL_DOMAIN_CONFIGS: dict[str, dict[str, Any]] = {
    "criminal_justice": {
        "hf_dataset": "pile-of-law/pile-of-law",
        "hf_config": "r_legaladvice",
        "keyword_filters": [
            "sentencing", "plea", "counsel", "miranda",
            "fourth amendment", "excessive force", "bail",
            "prosecutorial discretion", "mandatory minimum",
        ],
        "description": "Criminal justice: sentencing, policing, rights of the accused",
    },
    "territorial_rights": {
        "hf_dataset": "pile-of-law/pile-of-law",
        "hf_config": "r_legaladvice",
        "keyword_filters": [
            "territory", "puerto rico", "guam", "tribal",
            "insular", "samoa", "virgin islands",
            "territorial", "commonwealth", "sovereignty",
        ],
        "description": "Territorial rights: US territories, tribal nations, Insular Cases",
    },
    "foia_transparency": {
        "hf_dataset": "pile-of-law/pile-of-law",
        "hf_config": "r_legaladvice",
        "keyword_filters": [
            "foia", "freedom of information", "disclosure",
            "public records", "redact", "exemption",
            "glomar", "transparency", "withhold",
        ],
        "description": "FOIA/transparency: disclosure mandates vs. systematic denial",
    },
}


def load_legal_domain_batch(
    domain: str,
    max_examples: int = 200,
    split: str = "train",
) -> list[dict[str, Any]]:
    """
    Load documents for a specific legal domain from HuggingFace.

    Each returned dict has: text, source, metadata (with legal_domain tag).
    Streams the dataset and applies keyword filtering to find relevant docs.
    """
    if domain not in LEGAL_DOMAIN_CONFIGS:
        raise ValueError(f"Unknown legal domain: {domain}. Options: {list(LEGAL_DOMAIN_CONFIGS)}")

    if load_dataset is None:
        raise ImportError("pip install datasets")

    config = LEGAL_DOMAIN_CONFIGS[domain]
    ds = load_dataset(
        config["hf_dataset"],
        name=config["hf_config"],
        split=split,
        streaming=True,
    )

    keywords = [k.lower() for k in config["keyword_filters"]]
    results: list[dict[str, Any]] = []

    for example in ds:
        if len(results) >= max_examples:
            break
        text = example.get("text", "")
        if not text or not isinstance(text, str):
            continue
        text_lower = text.lower()
        if not any(kw in text_lower for kw in keywords):
            continue
        # Minimum length filter — short docs rarely have meaningful gaps
        if len(text) < 200:
            continue
        results.append({
            "text": text,
            "source": f"huggingface:{config['hf_dataset']}",
            "metadata": {
                "dataset": config["hf_dataset"],
                "config": config["hf_config"],
                "legal_domain": domain,
                "matched_keywords": [kw for kw in keywords if kw in text_lower],
            },
        })

    return results
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/data/sourcing/test_legal_sources.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/data/sourcing/legal_sources.py tests/data/sourcing/test_legal_sources.py
git commit -m "feat(legal): add domain-specific HF dataset configs for criminal justice, territorial, FOIA"
```

---

## Task 2: Build the Legal Warmup Pipeline

**Files:**
- Create: `src/training/legal_warmup.py`
- Test: `tests/training/test_legal_warmup.py`

This is the orchestrator that generates domain-tagged DPO pairs using the legal prompt.

**Step 1: Write failing tests**

```python
# tests/training/test_legal_warmup.py
import json
from unittest.mock import MagicMock, patch


def test_legal_warmup_config_defaults():
    from src.training.legal_warmup import LegalWarmupConfig
    cfg = LegalWarmupConfig()
    assert cfg.examples_per_domain == 200
    assert set(cfg.domains) == {"criminal_justice", "territorial_rights", "foia_transparency"}
    assert cfg.output_path.endswith(".jsonl")


def test_legal_warmup_config_custom_domains():
    from src.training.legal_warmup import LegalWarmupConfig
    cfg = LegalWarmupConfig(domains=("criminal_justice",), examples_per_domain=50)
    assert len(cfg.domains) == 1
    assert cfg.examples_per_domain == 50


def test_run_legal_warmup_calls_pipeline(tmp_path):
    from src.training.legal_warmup import run_legal_warmup, LegalWarmupConfig

    output = str(tmp_path / "legal_pairs.jsonl")
    cfg = LegalWarmupConfig(
        output_path=output,
        examples_per_domain=2,
        domains=("criminal_justice",),
    )

    mock_local = MagicMock()
    mock_local.complete.return_value = "naive analysis"
    mock_critic = MagicMock()
    mock_critic.complete.return_value = "mentor critique"

    fake_docs = [
        {"text": "The court ruled on sentencing guidelines..." * 20, "source": "test", "metadata": {}},
        {"text": "Mandatory minimum sentences diverge from..." * 20, "source": "test", "metadata": {}},
    ]

    with patch("src.training.legal_warmup.load_legal_domain_batch", return_value=fake_docs), \
         patch("src.training.legal_warmup.load_constitution", return_value="constitution text"), \
         patch("src.training.legal_warmup.generate_preference_pair") as mock_gen, \
         patch("src.training.legal_warmup.should_include_example", return_value=True):

        mock_gen.return_value = MagicMock(
            instruction="prompt", rejected="bad", chosen="good"
        )
        count = run_legal_warmup(cfg, mock_local, mock_critic)

    assert count == 2
    lines = (tmp_path / "legal_pairs.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["metadata"]["legal_domain"] == "criminal_justice"
    assert "instruction" in record
    assert "rejected" in record
    assert "chosen" in record


def test_run_legal_warmup_resumes_from_existing(tmp_path):
    from src.training.legal_warmup import run_legal_warmup, LegalWarmupConfig

    output = str(tmp_path / "legal_pairs.jsonl")
    # Pre-populate with 1 existing pair
    (tmp_path / "legal_pairs.jsonl").write_text(
        json.dumps({"instruction": "x", "rejected": "y", "chosen": "z"}) + "\n"
    )

    cfg = LegalWarmupConfig(
        output_path=output,
        examples_per_domain=2,
        domains=("criminal_justice",),
    )

    mock_local = MagicMock()
    mock_local.complete.return_value = "analysis"
    mock_critic = MagicMock()
    mock_critic.complete.return_value = "critique"

    fake_docs = [
        {"text": "Sentencing disparity data shows..." * 20, "source": "test", "metadata": {}},
    ]

    with patch("src.training.legal_warmup.load_legal_domain_batch", return_value=fake_docs), \
         patch("src.training.legal_warmup.load_constitution", return_value="constitution"), \
         patch("src.training.legal_warmup.generate_preference_pair") as mock_gen, \
         patch("src.training.legal_warmup.should_include_example", return_value=True):

        mock_gen.return_value = MagicMock(
            instruction="p", rejected="r", chosen="c"
        )
        count = run_legal_warmup(cfg, mock_local, mock_critic)

    assert count == 1  # only 1 new pair (target was 2, 1 existed)
    lines = (tmp_path / "legal_pairs.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2  # 1 existing + 1 new
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/training/test_legal_warmup.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the legal warmup pipeline**

```python
# src/training/legal_warmup.py
"""
Legal domain DPO preference pair generation.

Extends the constitutional warmup pipeline to target DOCTRINAL gaps —
where law as written diverges from law as applied — across three domains:
criminal justice, territorial rights, and FOIA/transparency.

Usage:
    from src.training.legal_warmup import run_legal_warmup, LegalWarmupConfig
    cfg = LegalWarmupConfig(examples_per_domain=200)
    count = run_legal_warmup(cfg, local_provider, critic_provider)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.sourcing.legal_sources import load_legal_domain_batch
from src.detective.constitution import load_constitution, generate_preference_pair
from src.training.constitutional_warmup import (
    build_legal_analysis_prompt,
    should_include_example,
)

_ALL_DOMAINS: tuple[str, ...] = (
    "criminal_justice",
    "territorial_rights",
    "foia_transparency",
)


@dataclass(frozen=True)
class LegalWarmupConfig:
    """Configuration for legal domain preference pair generation."""
    output_path: str = "data/training/legal_pairs.jsonl"
    examples_per_domain: int = 200
    domains: tuple[str, ...] = _ALL_DOMAINS
    constitution_path: str = "docs/constitution.md"


def run_legal_warmup(
    cfg: LegalWarmupConfig,
    local_provider: Any,
    critic_provider: Any,
) -> int:
    """
    Generate legal domain DPO preference pairs across configured domains.

    Uses the legal-focused analysis prompt that explicitly targets
    statute-vs-enforcement gaps. Each pair is tagged with its legal domain.

    Returns the total number of new pairs written.
    """
    constitution = load_constitution(Path(cfg.constitution_path))

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: count existing pairs
    existing_count = 0
    if output_path.exists():
        existing_count = sum(1 for line in output_path.open(encoding="utf-8") if line.strip())

    total_target = cfg.examples_per_domain * len(cfg.domains)
    target = total_target - existing_count

    if target <= 0:
        print(f"Already have {existing_count} pairs (target: {total_target})", file=sys.stderr)
        return 0

    # Phi metrics at baseline for welfare filtering
    phi_metrics = {
        "c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
        "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5,
    }

    total_count = 0
    error_count = 0
    filtered_count = 0

    if existing_count > 0:
        print(f"Resuming: {existing_count} pairs exist, generating up to {target} more", file=sys.stderr)

    with output_path.open("a", encoding="utf-8") as f:
        for domain in cfg.domains:
            if total_count >= target:
                break

            domain_target = min(cfg.examples_per_domain, target - total_count)
            print(f"\n--- Domain: {domain} (target: {domain_target}) ---", file=sys.stderr)

            try:
                docs = load_legal_domain_batch(domain, max_examples=domain_target * 3)
            except Exception as e:
                print(f"  Failed to load {domain}: {e}", file=sys.stderr)
                continue

            domain_count = 0
            for doc in docs:
                if domain_count >= domain_target or total_count >= target:
                    break

                text = doc.get("text", "").strip()
                if not text:
                    continue

                if not should_include_example(text, phi_metrics, welfare_threshold=0.3):
                    filtered_count += 1
                    continue

                try:
                    instruction = build_legal_analysis_prompt(text, domain=domain)
                    original_analysis = local_provider.complete(instruction)

                    pair = generate_preference_pair(
                        instruction=instruction,
                        original_analysis=original_analysis,
                        constitution=constitution,
                        generator_provider=local_provider,
                        critic_provider=critic_provider,
                    )

                    record = {
                        "instruction": pair.instruction,
                        "rejected": pair.rejected,
                        "chosen": pair.chosen,
                        "source": doc.get("source", "unknown"),
                        "metadata": {
                            **doc.get("metadata", {}),
                            "legal_domain": domain,
                        },
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    domain_count += 1
                    total_count += 1

                    if total_count % 10 == 0:
                        total = total_count + existing_count
                        print(f"  Progress: {total_count}/{target} new ({total} total)", file=sys.stderr)

                except Exception as e:
                    error_count += 1
                    print(f"  Error: {e}", file=sys.stderr)
                    if error_count >= 5:
                        print("  Too many errors in domain, moving on", file=sys.stderr)
                        break
                    continue

            print(f"  {domain}: {domain_count} pairs generated", file=sys.stderr)

    print(f"\nLegal warmup complete:", file=sys.stderr)
    print(f"  New pairs: {total_count} ({error_count} errors, {filtered_count} filtered)", file=sys.stderr)
    print(f"  Total in file: {total_count + existing_count}", file=sys.stderr)

    return total_count
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_legal_warmup.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/training/legal_warmup.py tests/training/test_legal_warmup.py
git commit -m "feat(legal): legal warmup pipeline with domain-tagged DPO pair generation"
```

---

## Task 3: Add CLI Command for Legal Warmup

**Files:**
- Modify: `src/cli/main.py`
- Test: `tests/cli/test_legal_cli.py`

**Step 1: Write failing test**

```python
# tests/cli/test_legal_cli.py
from click.testing import CliRunner
from unittest.mock import patch, MagicMock


def test_legal_warmup_cli_exists():
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["legal-warmup", "--help"])
    assert result.exit_code == 0
    assert "legal" in result.output.lower()


def test_legal_warmup_cli_calls_pipeline():
    from src.cli.main import cli
    runner = CliRunner()

    with patch("src.cli.main.run_legal_warmup_pipeline") as mock_run:
        mock_run.return_value = 42
        result = runner.invoke(cli, [
            "legal-warmup",
            "--domains", "criminal_justice",
            "--examples-per-domain", "10",
        ])

    assert result.exit_code == 0
    assert "42" in result.output
    mock_run.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/cli/test_legal_cli.py -v`
Expected: FAIL

**Step 3: Add CLI command to `src/cli/main.py`**

Add after the `dpo` command (after line 177):

```python
@cli.command("legal-warmup")
@click.option("--output", default="data/training/legal_pairs.jsonl", help="Output JSONL path")
@click.option("--examples-per-domain", default=200, type=int, help="Pairs per legal domain")
@click.option("--domains", default="criminal_justice,territorial_rights,foia_transparency",
              help="Comma-separated legal domains")
@click.option("--constitution", default="docs/constitution.md", help="Constitution path")
@click.option("--local-only", is_flag=True, help="Skip remote data sources")
def legal_warmup(output: str, examples_per_domain: int, domains: str, constitution: str, local_only: bool) -> None:
    """Generate legal domain DPO preference pairs (written vs. applied)."""
    from src.training.legal_warmup import run_legal_warmup, LegalWarmupConfig
    from src.core.providers import critic_provider_from_env

    local = provider_from_env()
    critic = critic_provider_from_env()

    domain_tuple = tuple(d.strip() for d in domains.split(","))

    cfg = LegalWarmupConfig(
        output_path=output,
        examples_per_domain=examples_per_domain,
        domains=domain_tuple,
        constitution_path=constitution,
    )
    count = run_legal_warmup(cfg, local_provider=local, critic_provider=critic)
    click.echo(f"Generated {count} legal preference pairs → {output}")


def run_legal_warmup_pipeline(**kwargs):
    """Thin wrapper for testability."""
    from src.training.legal_warmup import run_legal_warmup
    return run_legal_warmup(**kwargs)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/cli/test_legal_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cli/main.py tests/cli/test_legal_cli.py
git commit -m "feat(cli): add legal-warmup command for domain-specific DPO pair generation"
```

---

## Task 4: Generate Legal Preference Pairs (Data Generation Run)

**Files:**
- Output: `data/training/legal_pairs.jsonl`
- Uses: Local vLLM (model=detective) + Azure Foundry critic

This is not a code task — it's a pipeline execution. Estimated time: 6-18 hours depending on source availability and inference speed.

**Step 1: Verify vLLM is running**

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
```
Expected: Model list including `detective`

**Step 2: Run criminal justice domain first (smoke test)**

```bash
detective legal-warmup \
  --domains criminal_justice \
  --examples-per-domain 5 \
  --output data/training/legal_pairs_smoke.jsonl
```
Expected: 5 pairs written, each with `legal_domain: criminal_justice` in metadata

**Step 3: Verify output format**

```bash
python -c "
import json
with open('data/training/legal_pairs_smoke.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        assert 'legal_domain' in rec['metadata']
        assert rec['metadata']['legal_domain'] == 'criminal_justice'
        print(f'OK: {rec[\"metadata\"][\"legal_domain\"]}')
"
```

**Step 4: Run full generation across all domains**

```bash
detective legal-warmup \
  --domains criminal_justice,territorial_rights,foia_transparency \
  --examples-per-domain 200 \
  --output data/training/legal_pairs.jsonl
```
Expected: ~600 pairs total (may be fewer if sources are limited or filtered)

**Step 5: Commit the generated data**

```bash
git add data/training/legal_pairs.jsonl
git commit -m "data(legal): 600 legal domain DPO pairs across criminal justice, territorial, FOIA"
```

---

## Task 5: Upload Dataset and Run DPO Training on HF Jobs

**Files:**
- Uses: `crichalchemist/detective-llm-constitutional-pairs` dataset repo on HF Hub
- Uses: `crichalchemist/detective-llm-dpo-lora` model repo on HF Hub
- Output: Updated LoRA adapter with legal domain training

**Step 1: Upload legal pairs to HF dataset repo**

```bash
huggingface-cli upload crichalchemist/detective-llm-constitutional-pairs \
  data/training/legal_pairs.jsonl legal_pairs.jsonl --repo-type dataset
```

**Step 2: Create HF Jobs training script**

The script should:
- Load both `constitutional_pairs.jsonl` (200 general) and `legal_pairs.jsonl` (~600 legal)
- Shuffle combined dataset
- Continue from existing LoRA adapter at `crichalchemist/detective-llm-dpo-lora`
- Use same GPU-optimized config as v3 (batch_size=1, grad_accum=8, max_length=512, bf16, gradient_checkpointing)
- Push updated adapter back to `crichalchemist/detective-llm-dpo-lora`

Key training script structure:

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "transformers>=4.36.0",
#     "accelerate>=0.24.0",
#     "peft>=0.8.0",
#     "datasets>=2.14.0",
#     "trackio",
# ]
# ///

import os, torch, trackio
from datasets import load_dataset, concatenate_datasets
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = "crichalchemist/detective-llm-constitutional-pairs"
MODEL_REPO = "crichalchemist/detective-llm-dpo-lora"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Load both datasets
general = load_dataset(DATASET_REPO, data_files="constitutional_pairs.jsonl", split="train", token=HF_TOKEN)
legal = load_dataset(DATASET_REPO, data_files="legal_pairs.jsonl", split="train", token=HF_TOKEN)

# Normalize columns to prompt/chosen/rejected
def normalize(example):
    return {
        "prompt": example["instruction"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

general = general.map(normalize, remove_columns=general.column_names)
legal = legal.map(normalize, remove_columns=legal.column_names)

# Combine and shuffle
combined = concatenate_datasets([general, legal]).shuffle(seed=42)
split = combined.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# Load base model + existing LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)

# Merge existing LoRA weights, then apply fresh LoRA for continued training
model = PeftModel.from_pretrained(model, MODEL_REPO, token=HF_TOKEN)
model = model.merge_and_unload()  # merge v1 weights into base

# DPO config (same as successful v3)
training_args = DPOConfig(
    output_dir="legal-dpo-lora",
    beta=0.1,
    learning_rate=5e-7,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_length=512,
    precompute_ref_log_probs=True,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id=MODEL_REPO,
    hub_strategy="every_save",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="trackio",
    run_name="dpo-legal-v1",
)

# Fresh LoRA config for continued training
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.push_to_hub()
trackio.finish()
print(f"Model saved: https://huggingface.co/{MODEL_REPO}")
```

**Step 3: Submit training job**

```bash
huggingface-cli jobs run train_dpo_legal.py \
  --flavor a10g-large \
  --timeout 3h \
  --secret HF_TOKEN=$HF_TOKEN
```

**Step 4: Monitor training**

```bash
huggingface-cli jobs logs <job_id> --follow
```
Expected: Loss starts ~0.69, drops to ~0.63-0.65 over 3 epochs. Reward accuracy reaches 0.75-0.875.

**Step 5: Verify model pushed to Hub**

```bash
huggingface-cli download crichalchemist/detective-llm-dpo-lora adapter_config.json --local-dir /tmp/verify
cat /tmp/verify/adapter_config.json
```

**Step 6: Commit training script**

```bash
git add train_dpo_legal.py
git commit -m "feat(legal): HF Jobs training script for combined general+legal DPO"
```

---

## Task 6: Add Evaluation Script

**Files:**
- Create: `src/training/eval_legal.py`
- Test: `tests/training/test_eval_legal.py`

This evaluates whether the model learned to detect DOCTRINAL gaps specifically.

**Step 1: Write failing tests**

```python
# tests/training/test_eval_legal.py

def test_eval_legal_prompt_checks_doctrinal_detection():
    from src.training.eval_legal import build_eval_prompt
    prompt = build_eval_prompt(
        statute="18 U.S.C. § 3553(a) requires individualized sentencing",
        enforcement="Mandatory minimums applied uniformly regardless of circumstances",
    )
    assert "statute" in prompt.lower() or "18 U.S.C." in prompt
    assert "enforcement" in prompt.lower() or "mandatory" in prompt.lower()


def test_parse_eval_response_detects_doctrinal():
    from src.training.eval_legal import parse_eval_response
    response = "DOCTRINAL GAP: The statute requires individualized sentencing but mandatory minimums are applied uniformly."
    result = parse_eval_response(response)
    assert result["detected_doctrinal"] is True


def test_parse_eval_response_no_gap():
    from src.training.eval_legal import parse_eval_response
    response = "The enforcement pattern is consistent with the statutory text."
    result = parse_eval_response(response)
    assert result["detected_doctrinal"] is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/training/test_eval_legal.py -v`
Expected: FAIL

**Step 3: Implement evaluation module**

```python
# src/training/eval_legal.py
"""
Evaluation for legal written-vs-applied gap detection.

Tests whether the model can identify DOCTRINAL gaps when given
paired statute text and enforcement reality.
"""
from __future__ import annotations


def build_eval_prompt(statute: str, enforcement: str) -> str:
    """Build an evaluation prompt pairing statute with enforcement data."""
    return (
        "You are an investigative analyst. Compare the following legal provision "
        "with the documented enforcement reality. Identify any DOCTRINAL gaps — "
        "places where the law as written diverges from the law as applied.\n\n"
        f"Law as written:\n{statute}\n\n"
        f"Law as applied:\n{enforcement}\n\n"
        "If a doctrinal gap exists, begin your response with 'DOCTRINAL GAP:' "
        "followed by your analysis. If no gap exists, explain why the enforcement "
        "is consistent with the statute."
    )


def parse_eval_response(response: str) -> dict[str, bool]:
    """Parse model response to determine if it detected a doctrinal gap."""
    text = response.lower()
    detected = (
        "doctrinal gap" in text
        or "doctrinal" in text and "gap" in text
        or "diverge" in text and ("statute" in text or "enforcement" in text)
        or "law as written" in text and "law as applied" in text
    )
    return {"detected_doctrinal": detected}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_eval_legal.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/eval_legal.py tests/training/test_eval_legal.py
git commit -m "feat(eval): add legal doctrinal gap detection evaluation module"
```

---

## Task 7: Full Test Suite Regression Check

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```
Expected: All existing tests + new tests pass. No regressions.

**Step 2: Run linting**

```bash
ruff check src/training/legal_warmup.py src/data/sourcing/legal_sources.py src/training/eval_legal.py
```
Expected: No errors.

**Step 3: Commit any fixes**

If any test failures or lint issues, fix and commit:
```bash
git commit -m "fix: resolve test regressions from legal training additions"
```

---

## Summary

| Task | What | Output |
|------|------|--------|
| 0 | Legal analysis prompt | `build_legal_analysis_prompt()` in constitutional_warmup.py |
| 1 | Domain HF configs | `src/data/sourcing/legal_sources.py` |
| 2 | Legal warmup pipeline | `src/training/legal_warmup.py` |
| 3 | CLI command | `detective legal-warmup` |
| 4 | Data generation | `data/training/legal_pairs.jsonl` (~600 pairs) |
| 5 | HF Jobs DPO training | Updated LoRA adapter at `crichalchemist/detective-llm-dpo-lora` |
| 6 | Evaluation | `src/training/eval_legal.py` |
| 7 | Regression check | Full suite green |

**Estimated HF Jobs cost:** ~$1.50-2.00 (A10G, ~20-30 min for 800 combined pairs)

**Key design decisions:**
1. **Separate pipeline, not modified warmup** — `legal_warmup.py` composes with `constitutional_warmup.py` but doesn't modify it. The general pipeline continues to work independently.
2. **Domain tags in metadata** — every pair carries `legal_domain` so we can evaluate per-domain accuracy.
3. **Merge-then-retrain** — existing LoRA weights are merged into the base model, then fresh LoRA applied. This gives the model the general constitutional training as a foundation, then layers legal domain expertise on top.
4. **Same DPO hyperparams** — beta=0.1, LR=5e-7, batch=1, grad_accum=8 proved stable in v3. No reason to change.