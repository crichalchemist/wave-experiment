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
