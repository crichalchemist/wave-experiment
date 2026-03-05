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
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.sourcing.types import SourceDocument
from src.detective.constitution import load_constitution, generate_preference_pair
from src.inference.welfare_scoring import (
    infer_threatened_constructs,
    score_hypothesis_welfare,
)

_logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = (
    "You are an investigative analyst trained in information gap detection.\n\n"
    "Analyze the following document excerpt for information gaps — what is absent, "
    "suppressed, or undocumented that should be present given the stated facts.\n\n"
    "Be specific. Name what type of gap you observe (temporal, evidential, contradiction, "
    "normative, or doctrinal) and why its absence is significant.\n\n"
    "Document:\n{text}"
)

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


def should_include_example(
    document_text: str,
    phi_metrics: dict[str, float],
    welfare_threshold: float = 0.3,
) -> bool:
    """
    Filter training examples by welfare relevance.

    Only include examples that threaten at least one Φ construct with
    welfare_relevance > threshold.

    This focuses constitutional training on welfare-relevant reasoning,
    improving data efficiency and alignment with Detective LLM's mission.

    Args:
        document_text: Source document text to evaluate
        phi_metrics: Current Φ construct levels
        welfare_threshold: Minimum welfare_relevance to include (default 0.3)

    Returns:
        True if example should be included in training set
    """
    constructs = infer_threatened_constructs(document_text)

    if not constructs:
        return False  # Not welfare-relevant

    # Create pseudo-hypothesis for scoring
    from src.detective.hypothesis import Hypothesis
    from datetime import datetime
    pseudo_hyp = Hypothesis(
        id="filter",
        text=document_text[:500],  # first 500 chars for keyword matching
        confidence=0.5,
        timestamp=datetime.now(),
        threatened_constructs=constructs,
    )

    welfare_score = score_hypothesis_welfare(pseudo_hyp, phi_metrics)

    return welfare_score >= welfare_threshold


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
    document_file: str | None = None  # NEW: Simple text file with one document per line
    hf_datasets: tuple[str, ...] = (
        "pile-of-law/pile-of-law",
        "nguha/legalbench",
    )
    hf_keyword_filter: str | None = "disclosure"


def _load_all_sources(cfg: ConstitutionalWarmupConfig) -> list[SourceDocument]:
    """Aggregate examples from all enabled source pipelines."""
    examples: list[SourceDocument] = []
    per_source = max(1, cfg.max_examples // 3)

    # Load from pre-extracted document file if provided.
    # Supports two formats:
    #   1. Chunk-separated: blocks divided by lines of "====..." (from extract_pdf_documents.py)
    #   2. One-per-line: each non-empty line is a document
    if cfg.document_file:
        try:
            raw = Path(cfg.document_file).read_text(encoding="utf-8")
            # Detect chunk-separated format (extract_pdf_documents.py output)
            if "=" * 20 in raw:
                chunks = raw.split("=" * 20)
                for i, chunk in enumerate(chunks):
                    if len(examples) >= cfg.max_examples:
                        break
                    text = chunk.strip()
                    # Skip empty chunks and metadata-only lines
                    if not text or (text.startswith("#") and "\n" not in text):
                        continue
                    # Extract source from metadata comment if present
                    source = cfg.document_file
                    lines = text.split("\n", 1)
                    if lines[0].startswith("# Source:"):
                        source = lines[0].replace("# Source:", "").strip()
                        text = lines[1].strip() if len(lines) > 1 else ""
                    if text and len(text) > 100:  # skip too-short fragments
                        examples.append(SourceDocument(
                            text=text,
                            source=source,
                            metadata={"chunk": i},
                        ))
            else:
                # One-per-line format
                for i, line in enumerate(raw.splitlines()):
                    if len(examples) >= cfg.max_examples:
                        break
                    text = line.strip()
                    if text and not text.startswith("#"):
                        examples.append(SourceDocument(
                            text=text,
                            source=cfg.document_file,
                            metadata={"line": i + 1},
                        ))
        except Exception:
            _logger.debug("Failed to load document file %s", cfg.document_file)

    if cfg.use_huggingface:
        from src.data.sourcing.hf_loader import load_hf_legal_batch
        for ds_name in cfg.hf_datasets:
            try:
                batch = load_hf_legal_batch(
                    dataset_name=ds_name,
                    max_documents=per_source,
                    keyword_filter=cfg.hf_keyword_filter,
                )
                examples.extend(batch)
            except Exception:
                _logger.debug("HF dataset %s unavailable, skipping", ds_name)

    if cfg.use_doj:
        from src.data.sourcing.doj_loader import load_courtlistener_batch
        try:
            examples.extend(load_courtlistener_batch(max_documents=per_source))
        except Exception:
            _logger.debug("CourtListener unavailable, skipping")

    if cfg.use_international:
        from src.data.sourcing.international_loader import load_github_public_foia
        try:
            examples.extend(load_github_public_foia(max_documents=per_source))
        except Exception:
            _logger.debug("GitHub FOIA loader unavailable, skipping")

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

    # Initialize phi_metrics (all constructs at baseline 0.5)
    phi_metrics = {
        "c": 0.5,      # care (resource allocation)
        "kappa": 0.5,  # compassion (responsive support)
        "j": 0.5,      # joy (positive affect)
        "p": 0.5,      # purpose (goal alignment)
        "eps": 0.5,    # empathy (perspective-taking)
        "lam_L": 0.5,  # love (growth, mutual aid)
        "lam_P": 0.5,  # protection (safeguarding)
        "xi": 0.5,     # truth (epistemic integrity)
    }

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count existing pairs if resuming (append mode)
    existing_count = 0
    if output_path.exists():
        existing_count = sum(1 for _ in output_path.open(encoding="utf-8"))

    count = 0
    filtered_count = 0
    skipped_for_resume = 0
    error_count = 0
    target = cfg.max_examples - existing_count

    if existing_count > 0:
        _logger.info("Resuming: %d pairs already exist, generating up to %d more", existing_count, target)

    if target <= 0:
        _logger.info("Already have %d pairs (target: %d)", existing_count, cfg.max_examples)
        return existing_count

    with output_path.open("a", encoding="utf-8") as f:
        for example in examples:
            text = example.text.strip()
            if not text:
                continue

            # Apply welfare filter before expensive CAI processing
            if not should_include_example(text, phi_metrics, welfare_threshold=0.3):
                filtered_count += 1
                continue

            try:
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
                    "source": example.source,
                    "metadata": example.metadata,
                }) + "\n")
                f.flush()  # write to disk immediately — crash-safe
                count += 1

                if count % 10 == 0:
                    _logger.info("Progress: %d/%d pairs (%d total)", count, target, count + existing_count)

            except Exception as e:
                error_count += 1
                _logger.warning("Error on chunk %s: %s", example.metadata, e)
                if error_count >= 5:
                    _logger.warning("Too many errors, stopping")
                    break
                continue  # skip this example and continue

            if count >= target:
                break

    # Report filtering statistics
    _logger.info("Welfare filtering: %d examples excluded (below threshold)", filtered_count)
    _logger.info("Generated: %d new preference pairs (%d errors)", count, error_count)
    _logger.info("Total pairs in file: %d", count + existing_count)

    return count
