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
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.sourcing.legal_sources import load_legal_domain_batch
from src.detective.constitution import load_constitution, generate_preference_pair
from src.training.constitutional_warmup import (
    build_legal_analysis_prompt,
    should_include_example,
)

_logger = logging.getLogger(__name__)

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
        _logger.info("Already have %d pairs (target: %d)", existing_count, total_target)
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
        _logger.info("Resuming: %d pairs exist, generating up to %d more", existing_count, target)

    with output_path.open("a", encoding="utf-8") as f:
        for domain in cfg.domains:
            if total_count >= target:
                break

            domain_target = min(cfg.examples_per_domain, target - total_count)
            _logger.info("--- Domain: %s (target: %d) ---", domain, domain_target)

            try:
                docs = load_legal_domain_batch(domain=domain, max_documents=domain_target * 3)
            except Exception as e:
                _logger.warning("Failed to load %s: %s", domain, e)
                continue

            domain_count = 0
            for doc in docs:
                if domain_count >= domain_target or total_count >= target:
                    break

                text = doc.text.strip()
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
                        "source": doc.source,
                        "metadata": {
                            **doc.metadata,
                            "legal_domain": domain,
                        },
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    domain_count += 1
                    total_count += 1

                    if total_count % 10 == 0:
                        total = total_count + existing_count
                        _logger.info("Progress: %d/%d new (%d total)", total_count, target, total)

                except Exception as e:
                    error_count += 1
                    _logger.warning("Error: %s", e)
                    if error_count >= 5:
                        _logger.warning("Too many errors in domain, moving on")
                        break
                    continue

            _logger.info("%s: %d pairs generated", domain, domain_count)

    _logger.info("Legal warmup complete:")
    _logger.info("New pairs: %d (%d errors, %d filtered)", total_count, error_count, filtered_count)
    _logger.info("Total in file: %d", total_count + existing_count)

    return total_count
