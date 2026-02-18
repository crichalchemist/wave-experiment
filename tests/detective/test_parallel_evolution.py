"""Tests for parallel hypothesis evolution (GoT Generate(k))."""
import asyncio
import pytest
from unittest.mock import MagicMock


def test_import():
    from src.detective.parallel_evolution import evolve_parallel, ParallelEvolutionResult
    assert callable(evolve_parallel)
    assert ParallelEvolutionResult is not None


def test_returns_multiple_branches():
    """k=3 branches produces 3 independent evolved hypotheses."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis

    root = Hypothesis.create(
        text="Entity A had undisclosed financial ties to Entity B in 2003.",
        confidence=0.6,
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
        text="Gap in 2019 financial records.",
        confidence=0.5,
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
        assert result.hypothesis.parent_id == root.id


def test_branches_are_independent():
    """Each branch explores a different evidence thread."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis

    root = Hypothesis.create(text="Test.", confidence=0.5)
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

    root = Hypothesis.create(text="Test.", confidence=0.5)
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

    root = Hypothesis.create(text="Test.", confidence=0.5)
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
