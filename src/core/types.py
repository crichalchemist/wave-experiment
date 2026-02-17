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
    COGNITIVE_BIAS = "cognitive_bias"                      # Module A
    HISTORICAL_DETERMINISM = "historical_determinism"      # Module B
    GEOPOLITICAL_PRESUMPTION = "geopolitical_presumption"  # Module C


@dataclass(frozen=True)
class Gap:
    """A detected information gap with its type, confidence, and location in the corpus."""
    type: GapType
    description: str
    confidence: float
    location: str  # document identifier or section reference


@dataclass(frozen=True)
class KnowledgeEdge:
    """A typed, weighted edge in the knowledge graph with hop-count for decay."""
    source: str
    target: str
    relation: RelationType
    confidence: float
    hop_count: int = 1
