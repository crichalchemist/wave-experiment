from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GapType(Enum):
    TEMPORAL = "temporal"           # Missing time periods, unexplained silences
    EVIDENTIAL = "evidential"       # Claims without documentation
    CONTRADICTION = "contradiction" # Conflicting information that cannot both be accurate
    NORMATIVE = "normative"         # What should be documented given stated obligations but is not
    DOCTRINAL = "doctrinal"         # Unstated institutional rules assumed to apply that may not have


class RelationType(Enum):
    CONDITIONAL = "conditional"     # A describes outcome contingent on B
    CAUSAL = "causal"               # A directly causes B
    INSTANTIATIVE = "instantiative" # B is a specific instance of A
    SEQUENTIAL = "sequential"       # B occurs chronologically after A


class AssumptionType(Enum):
    COGNITIVE_BIAS = "cognitive_bias"                      # Systematic reasoning errors regardless of domain
    HISTORICAL_DETERMINISM = "historical_determinism"      # Assuming documents record events neutrally and in order
    GEOPOLITICAL_PRESUMPTION = "geopolitical_presumption"  # Assuming institutions behaved as their stated norms describe


class LegalDomain(Enum):
    STATUTE = "statute"                          # Black-letter law, codified text
    REGULATION = "regulation"                    # Federal/state agency rules
    CASE_LAW = "case_law"                        # Court holdings and precedent
    ENFORCEMENT_PRACTICE = "enforcement_practice"  # What regulators/police actually do
    COMMUNITY_EXPERIENCE = "community_experience"  # Documented lived reality of those affected
    TREATY = "treaty"                            # Federal Indian law, territorial agreements
    TERRITORIAL = "territorial"                  # Law as it applies in US territories

@dataclass(frozen=True)
class Gap:
    """A detected information gap with its type, confidence, and location in the corpus."""
    type: GapType
    description: str
    confidence: float
    location: str  # document identifier or section reference

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Gap.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )


@dataclass(frozen=True)
class KnowledgeEdge:
    """A typed, weighted edge in the knowledge graph with hop-count for decay."""
    source: str
    target: str
    relation: RelationType
    confidence: float
    hop_count: int = 1

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"KnowledgeEdge.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if self.hop_count < 1:
            raise ValueError(
                f"KnowledgeEdge.hop_count must be >= 1, got {self.hop_count!r}"
            )
        if not self.source:
            raise ValueError("KnowledgeEdge.source must not be empty")
        if not self.target:
            raise ValueError("KnowledgeEdge.target must not be empty")
