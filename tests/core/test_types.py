import pytest

from src.core.types import (
    AssumptionType,
    Gap,
    GapType,
    KnowledgeEdge,
    RelationType,
)


def test_gap_type_contains_required_members() -> None:
    required = {GapType.TEMPORAL, GapType.EVIDENTIAL, GapType.CONTRADICTION,
                GapType.NORMATIVE, GapType.DOCTRINAL}
    assert required.issubset(set(GapType))


def test_relation_type_contains_required_members() -> None:
    required = {RelationType.CONDITIONAL, RelationType.CAUSAL,
                RelationType.INSTANTIATIVE, RelationType.SEQUENTIAL}
    assert required.issubset(set(RelationType))


def test_assumption_type_contains_required_members() -> None:
    required = {AssumptionType.COGNITIVE_BIAS, AssumptionType.HISTORICAL_DETERMINISM,
                AssumptionType.GEOPOLITICAL_PRESUMPTION}
    assert required.issubset(set(AssumptionType))


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


def test_gap_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        Gap(type=GapType.TEMPORAL, description="x", confidence=1.5, location="doc-1")


def test_gap_confidence_boundary_values_valid() -> None:
    # 0.0 and 1.0 are valid boundaries
    gap_zero = Gap(type=GapType.TEMPORAL, description="x", confidence=0.0, location="doc-1")
    gap_one = Gap(type=GapType.TEMPORAL, description="x", confidence=1.0, location="doc-1")
    assert gap_zero.confidence == 0.0
    assert gap_one.confidence == 1.0


def test_knowledge_edge_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        KnowledgeEdge(source="A", target="B", relation=RelationType.CAUSAL, confidence=-0.1)


def test_knowledge_edge_hop_count_zero_raises() -> None:
    with pytest.raises(ValueError, match="hop_count must be"):
        KnowledgeEdge(source="A", target="B", relation=RelationType.CAUSAL, confidence=0.5, hop_count=0)


def test_knowledge_edge_empty_source_raises() -> None:
    with pytest.raises(ValueError, match="source must not be empty"):
        KnowledgeEdge(source="", target="B", relation=RelationType.CAUSAL, confidence=0.5)


def test_gap_with_welfare_fields():
    """Gap accepts welfare_impact and threatened_constructs fields."""
    gap = Gap(
        type=GapType.TEMPORAL,
        description="Resource gap",
        confidence=0.8,
        location="doc.pdf",
        welfare_impact=5.2,
        threatened_constructs=("c", "lam"),
    )
    assert gap.welfare_impact == 5.2
    assert gap.threatened_constructs == ("c", "lam")


def test_gap_welfare_fields_default_to_zero_and_empty():
    """Welfare fields default to 0.0 and () for backward compatibility."""
    gap = Gap(
        type=GapType.TEMPORAL,
        description="Gap",
        confidence=0.7,
        location="doc.pdf",
    )
    assert gap.welfare_impact == 0.0
    assert gap.threatened_constructs == ()


def test_gap_welfare_impact_validation():
    """Gap.welfare_impact must be >= 0."""
    with pytest.raises(ValueError, match="welfare_impact must be >= 0"):
        Gap(
            type=GapType.TEMPORAL,
            description="Gap",
            confidence=0.7,
            location="doc.pdf",
            welfare_impact=-1.0,
        )
