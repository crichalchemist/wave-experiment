"""Tests for legal domain gap detection (ADR-007)."""
from __future__ import annotations

import pytest

from src.core.types import GapType, LegalDomain, RelationType
from src.data.graph_store import InMemoryGraph
from src.detective.legal_gap_detector import (
    LegalGap,
    detect_legal_domain_gaps,
)


@pytest.fixture
def graph() -> InMemoryGraph:
    return InMemoryGraph()


def test_no_legal_edges_returns_empty(graph: InMemoryGraph) -> None:
    """No gaps when entity has no legal-domain-annotated edges."""
    graph.add_edge("A", "B", RelationType.CAUSAL, 0.8)
    assert detect_legal_domain_gaps(graph, "A") == []


def test_nonexistent_entity_returns_empty(graph: InMemoryGraph) -> None:
    """Absent entity should not raise — absence is investigatively meaningful."""
    assert detect_legal_domain_gaps(graph, "ghost") == []


def test_written_without_applied_is_doctrinal_gap(graph: InMemoryGraph) -> None:
    """STATUTE edge without ENFORCEMENT_PRACTICE → DOCTRINAL gap."""
    graph.add_edge(
        "Corp X", "Regulation Y", RelationType.CAUSAL, 0.9,
        legal_domain=LegalDomain.STATUTE,
    )
    gaps = detect_legal_domain_gaps(graph, "Corp X")
    assert len(gaps) == 1
    assert gaps[0].gap_type == GapType.DOCTRINAL
    assert gaps[0].topic_entity == "Corp X"
    assert len(gaps[0].written_edges) == 1
    assert gaps[0].applied_edges == ()
    assert "statute" in gaps[0].description.lower()


def test_applied_without_written_is_normative_gap(graph: InMemoryGraph) -> None:
    """ENFORCEMENT_PRACTICE edge without STATUTE → NORMATIVE gap."""
    graph.add_edge(
        "Agency Z", "Community Q", RelationType.CAUSAL, 0.7,
        legal_domain=LegalDomain.ENFORCEMENT_PRACTICE,
    )
    gaps = detect_legal_domain_gaps(graph, "Agency Z")
    assert len(gaps) == 1
    assert gaps[0].gap_type == GapType.NORMATIVE
    assert "enforcement" in gaps[0].description.lower()


def test_both_written_and_applied_no_gap(graph: InMemoryGraph) -> None:
    """When both domains are present, no gap is reported."""
    graph.add_edge(
        "Corp X", "Statute A", RelationType.CAUSAL, 0.9,
        legal_domain=LegalDomain.STATUTE,
    )
    graph.add_edge(
        "Corp X", "Practice B", RelationType.CAUSAL, 0.7,
        legal_domain=LegalDomain.ENFORCEMENT_PRACTICE,
    )
    assert detect_legal_domain_gaps(graph, "Corp X") == []


def test_multiple_written_domains_in_gap(graph: InMemoryGraph) -> None:
    """Multiple written-domain edges all appear in the gap's written_edges."""
    graph.add_edge(
        "Entity", "StatuteEdge", RelationType.CAUSAL, 0.8,
        legal_domain=LegalDomain.STATUTE,
    )
    graph.add_edge(
        "Entity", "RegulationEdge", RelationType.CAUSAL, 0.7,
        legal_domain=LegalDomain.REGULATION,
    )
    gaps = detect_legal_domain_gaps(graph, "Entity")
    assert len(gaps) == 1
    assert len(gaps[0].written_edges) == 2


def test_legal_gap_is_frozen() -> None:
    """LegalGap is a frozen dataclass."""
    gap = LegalGap(
        topic_entity="A",
        written_edges=(),
        applied_edges=(),
        gap_type=GapType.DOCTRINAL,
        description="test",
    )
    with pytest.raises(AttributeError):
        gap.topic_entity = "B"  # type: ignore[misc]


def test_community_experience_is_applied_domain(graph: InMemoryGraph) -> None:
    """COMMUNITY_EXPERIENCE is in the applied domain set."""
    graph.add_edge(
        "Entity", "CommunityDoc", RelationType.CAUSAL, 0.6,
        legal_domain=LegalDomain.COMMUNITY_EXPERIENCE,
    )
    gaps = detect_legal_domain_gaps(graph, "Entity")
    assert len(gaps) == 1
    assert gaps[0].gap_type == GapType.NORMATIVE


def test_treaty_is_written_domain(graph: InMemoryGraph) -> None:
    """TREATY is in the written domain set."""
    graph.add_edge(
        "Nation", "TreatyDoc", RelationType.CAUSAL, 0.8,
        legal_domain=LegalDomain.TREATY,
    )
    gaps = detect_legal_domain_gaps(graph, "Nation")
    assert len(gaps) == 1
    assert gaps[0].gap_type == GapType.DOCTRINAL
