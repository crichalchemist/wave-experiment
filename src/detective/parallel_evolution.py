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
from src.detective.experience import ExperienceLibrary
from src.core.providers import ModelProvider

_BRANCH_PROMPT = (
    "You are evolving a hypothesis based on a specific piece of evidence.\n\n"
    "Current hypothesis: {text}\n"
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
    """Extract confidence from response. Falls back to slight decay on parse failure."""
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
        text=hypothesis.text,
        confidence=hypothesis.confidence,
        evidence=evidence,
    )

    # Provider.complete is synchronous — run in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, provider.complete, prompt)

    new_confidence = _parse_confidence(response, hypothesis.confidence)
    evolved = hypothesis.update_confidence(new_confidence=new_confidence)

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
