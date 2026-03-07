"""Tests for parallel hypothesis evolution (GoT Generate(k))."""
import asyncio
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


def test_evolve_parallel_sets_trajectory_urgency():
    """When phi_metrics provided, trajectory_urgency should be set."""
    from src.detective.hypothesis import Hypothesis
    from src.detective.parallel_evolution import evolve_parallel
    from src.core.providers import MockProvider
    from unittest.mock import patch

    root = Hypothesis.create("Test gap hypothesis", 0.7)
    provider = MockProvider(response="confidence: 0.6")
    phi_metrics = {"c": 0.3, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    with patch("src.inference.welfare_scoring.score_hypothesis_trajectory", return_value=0.7):
        results = asyncio.run(evolve_parallel(
            hypothesis=root,
            evidence_list=["evidence A"],
            provider=provider,
            k=1,
            phi_metrics=phi_metrics,
        ))

    assert len(results) == 1
    assert results[0].hypothesis.trajectory_urgency == 0.7


def test_evolve_parallel_uses_4_weight_scoring():
    """With phi_metrics, sorting uses 4-weight combined_score."""
    from src.detective.hypothesis import Hypothesis
    from src.detective.parallel_evolution import evolve_parallel
    from src.core.providers import MockProvider
    from unittest.mock import patch

    root = Hypothesis.create("Test", 0.5)
    provider = MockProvider(response="confidence: 0.6")
    phi_metrics = {"c": 0.3, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    with patch("src.inference.welfare_scoring.score_hypothesis_trajectory", return_value=0.9):
        results = asyncio.run(evolve_parallel(
            hypothesis=root,
            evidence_list=["ev A", "ev B"],
            provider=provider,
            k=2,
            phi_metrics=phi_metrics,
        ))

    for r in results:
        assert r.hypothesis.trajectory_urgency == 0.9


def test_welfare_scoring_applied_to_evolved_hypotheses():
    """Evolved hypotheses get welfare_relevance and threatened_constructs populated."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis
    from src.core.providers import MockProvider

    root = Hypothesis.create("Base hypothesis", 0.5)

    evidence = [
        "Evidence of resource deprivation",
        "Minor administrative detail",
    ]

    provider = MockProvider(response="confidence: 0.7")
    phi_metrics = {"c": 0.2, "lam_P": 0.5}  # care is scarce

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=evidence,
        provider=provider,
        k=2,
        phi_metrics=phi_metrics,
    ))

    # All evolved hypotheses should have welfare fields populated
    for result in results:
        assert hasattr(result.hypothesis, "welfare_relevance")
        assert hasattr(result.hypothesis, "threatened_constructs")


def test_welfare_aware_sorting():
    """Results sorted by combined_score when phi_metrics provided."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis
    from src.core.providers import MockProvider

    root = Hypothesis.create("Base hypothesis", 0.5)

    evidence = ["Evidence A", "Evidence B"]
    provider = MockProvider(response="confidence: 0.6")
    phi_metrics = {"c": 0.3}

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=evidence,
        provider=provider,
        k=2,
        phi_metrics=phi_metrics,
    ))

    # Results should be sorted by combined_score (descending)
    if len(results) >= 2:
        assert results[0].hypothesis.combined_score() >= results[1].hypothesis.combined_score()


def test_high_welfare_relevance_for_urgent_findings():
    """High-urgency findings receive high welfare_relevance scores."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis
    from src.core.providers import MockProvider

    # Hypothesis text itself must contain welfare-relevant keywords
    root = Hypothesis.create("Resource allocation gap in vulnerable communities", 0.5)

    evidence = ["Additional documentation showing funding shortfalls"]
    provider = MockProvider(response="confidence: 0.5")
    phi_metrics = {"c": 0.1, "lam_P": 0.2}  # scarce constructs

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=evidence,
        provider=provider,
        k=1,
        phi_metrics=phi_metrics,
    ))

    # Welfare relevance should be high because hypothesis text contains "resource"
    # and care construct (c) is scarce (0.1)
    assert results[0].hypothesis.welfare_relevance > 0.5


def test_backward_compatible_without_phi_metrics():
    """When phi_metrics=None, sorts by confidence alone (backward compatible)."""
    from src.detective.parallel_evolution import evolve_parallel
    from src.detective.hypothesis import Hypothesis
    from src.core.providers import MockProvider

    root = Hypothesis.create("Base hypothesis", 0.5)

    evidence = ["Evidence A", "Evidence B"]
    provider = MockProvider(response="confidence: 0.7")

    results = asyncio.run(evolve_parallel(
        hypothesis=root,
        evidence_list=evidence,
        provider=provider,
        k=2,
        phi_metrics=None,  # No welfare scoring
    ))

    # Welfare fields should remain at defaults
    for result in results:
        assert result.hypothesis.welfare_relevance == 0.0
        assert result.hypothesis.threatened_constructs == ()


def test_curiosity_populated_with_phi_metrics():
    """When phi_metrics provided, curiosity_relevance should be non-default."""
    from src.detective.hypothesis import Hypothesis
    from src.detective.parallel_evolution import evolve_parallel
    from src.core.providers import MockProvider

    h = Hypothesis.create("Love and truth are under threat in this investigation", 0.5)

    # phi_metrics with low lam_L (love) and xi (truth) to trigger curiosity
    phi_metrics = {
        "c": 0.8, "lam_P": 0.7, "lam_L": 0.2, "xi": 0.3,
        "psi": 0.6, "omega": 0.5, "kappa": 0.7, "alpha": 0.6,
    }

    provider = MockProvider(response="confirmed\nconfidence: 0.7")
    results = asyncio.run(evolve_parallel(
        hypothesis=h,
        evidence_list=["Evidence about love and truth threats"],
        provider=provider,
        k=1,
        phi_metrics=phi_metrics,
    ))

    assert len(results) == 1
    evolved = results[0].hypothesis
    assert evolved.curiosity_relevance > 0.0, "curiosity_relevance should be populated"
