import pytest

from src.core.types import (
    AssumptionType,
    Gap,
    GapType,
    KnowledgeEdge,
    RelationType,
)


def test_gap_type_has_five_members() -> None:
    assert len(GapType) == 5


def test_relation_type_has_four_members() -> None:
    assert len(RelationType) == 4


def test_assumption_type_has_three_members() -> None:
    assert len(AssumptionType) == 3


def test_gap_is_immutable() -> None:
    from dataclasses import FrozenInstanceError
    gap = Gap(type=GapType.TEMPORAL, description="missing Q3", confidence=0.8, location="doc-1")
    with pytest.raises(FrozenInstanceError):
        gap.confidence = 0.5  # type: ignore[misc]


def test_knowledge_edge_default_hop_count() -> None:
    edge = KnowledgeEdge(source="A", target="B", relation=RelationType.CAUSAL, confidence=0.9)
    assert edge.hop_count == 1


def test_gap_normative_and_doctrinal_are_distinct() -> None:
    assert GapType.NORMATIVE != GapType.DOCTRINAL
    assert GapType.NORMATIVE.value == "normative"
    assert GapType.DOCTRINAL.value == "doctrinal"
