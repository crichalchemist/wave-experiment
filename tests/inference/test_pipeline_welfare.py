"""Tests for welfare-aware gap prioritization in analysis pipeline."""
import pytest
from unittest.mock import patch
from src.inference.pipeline import analyze, score_gaps_welfare
from src.core.providers import MockProvider
from src.data.graph_store import InMemoryGraph
from src.detective.experience import EMPTY_LIBRARY
from src.core.types import Gap, GapType

# Force keyword fallback — semantic classifier tested in test_welfare_integration_semantic.py
_force_keyword = patch(
    'src.inference.welfare_classifier.get_construct_scores',
    side_effect=FileNotFoundError("mocked out"),
)


def test_analyze_returns_gaps_with_welfare_impact():
    """analyze() should populate welfare_impact for detected gaps."""
    # This test requires gaps to be detected and returned by analyze()
    # Current analyze() returns AnalysisResult without gaps field
    # This is a placeholder for future gap detection integration
    pytest.skip("Gap detection not yet integrated into analyze() pipeline")


def test_gaps_sorted_by_welfare_urgency():
    """Gaps returned by analyze() should be sorted by welfare urgency."""
    pytest.skip("Gap detection not yet integrated into analyze() pipeline")


@_force_keyword
def test_score_gaps_welfare_populates_welfare_impact(_mock):
    """score_gaps_welfare() computes and populates welfare_impact."""
    gaps = [
        Gap(
            type=GapType.TEMPORAL,
            description="Resource allocation gap 2013-2017",
            confidence=0.9,
            location="doc.pdf",
        ),
        Gap(
            type=GapType.EVIDENTIAL,
            description="Typo in document",
            confidence=0.95,
            location="memo.txt",
        ),
    ]

    phi_metrics = {"c": 0.2}  # care is scarce
    scored = score_gaps_welfare(gaps, phi_metrics)

    # First gap (resource) should have higher welfare impact than second (typo)
    assert scored[0].welfare_impact > scored[1].welfare_impact
    assert scored[0].description == "Resource allocation gap 2013-2017"


@_force_keyword
def test_score_gaps_welfare_sorts_by_urgency(_mock):
    """score_gaps_welfare() sorts gaps by welfare urgency (descending)."""
    gaps = [
        Gap(
            type=GapType.EVIDENTIAL,
            description="Administrative note",
            confidence=0.8,
            location="a.pdf",
        ),
        Gap(
            type=GapType.TEMPORAL,
            description="Evidence of resource deprivation",
            confidence=0.7,
            location="b.pdf",
        ),
        Gap(
            type=GapType.CONTRADICTION,
            description="Suppressed testimony about violence",
            confidence=0.9,
            location="c.pdf",
        ),
    ]

    phi_metrics = {"c": 0.1, "lam_P": 0.2, "xi": 0.3}  # all scarce
    scored = score_gaps_welfare(gaps, phi_metrics)

    # Should be sorted by welfare urgency
    assert scored[0].welfare_impact >= scored[1].welfare_impact >= scored[2].welfare_impact

    # Violence/protection gap should be first (most urgent)
    assert "violence" in scored[0].description.lower()


@_force_keyword
def test_score_gaps_welfare_infers_constructs(_mock):
    """score_gaps_welfare() infers threatened_constructs if not set."""
    gap = Gap(
        type=GapType.TEMPORAL,
        description="Resource allocation gap",
        confidence=0.8,
        location="doc.pdf",
        # threatened_constructs defaults to ()
    )

    phi_metrics = {"c": 0.3}
    scored = score_gaps_welfare([gap], phi_metrics)

    # Should infer "c" from "resource"
    assert "c" in scored[0].threatened_constructs
