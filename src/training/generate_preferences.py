"""
Constitutional preference pair generator.

For each annotated gap example:
1. Generate analysis with the local model (VLLMProvider / any ModelProvider)
2. Critique with Claude (AzureFoundryProvider / any ModelProvider), then revise
   using the local model guided by that critique — full CAI loop via constitution.py
3. Write {"instruction", "rejected", "chosen"} to JSONL

The preference pair format is what DPO training (Task 14) will consume.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.core.providers import ModelProvider
from src.detective.constitution import generate_preference_pair, load_constitution

PREFERENCE_JSONL_PATH: str = "data/training/preferences.jsonl"


@dataclass(frozen=True)
class AnnotationExample:
    """One annotated gap example. instruction drives the analysis prompt."""

    instruction: str
    text: str       # document/claim text to analyze
    gap_type: str   # expected gap type label (metadata, not used in prompt)


def generate_analysis(instruction: str, text: str, provider: ModelProvider) -> str:
    """
    Ask the local model to analyze text for gaps per the instruction.
    Prompt structure keeps instruction as context anchor preceding the document —
    matching the layering convention in prompt_guard.build_analysis_prompt.
    """
    prompt = f"{instruction}\n\nText to analyze:\n{text}"
    return provider.complete(prompt)


def generate_preferences_to_jsonl(
    examples: Iterable[AnnotationExample],
    local_provider: ModelProvider,
    critic_provider: ModelProvider,
    output_path: str = PREFERENCE_JSONL_PATH,
    constitution: str | None = None,
) -> int:
    """
    Generate preference pairs for all examples and write to JSONL.
    Returns the count of pairs written.

    Each output line: {"instruction": str, "rejected": str, "chosen": str}
    - rejected: original analysis from local_provider (before critique)
    - chosen: constitutionally revised analysis — local_provider rewriting
              its own output guided by critic_provider mentor feedback

    Passes local_provider as both generator AND revision writer so the CAI
    loop teaches the local model to internalize the critic's values rather
    than producing a critic-written response that the local model never authored.
    """
    if constitution is None:
        constitution = load_constitution()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as f:
        for example in examples:
            analysis = generate_analysis(example.instruction, example.text, local_provider)
            pair = generate_preference_pair(
                instruction=example.instruction,
                original_analysis=analysis,
                constitution=constitution,
                generator_provider=local_provider,
                critic_provider=critic_provider,
            )
            f.write(json.dumps(dataclasses.asdict(pair)) + "\n")
            count += 1

    return count
