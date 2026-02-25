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
from dataclasses import dataclass
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
