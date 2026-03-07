"""Autonomous investigation agent — plan → gather → analyze → reflect → evolve → enrich."""

from src.detective.investigation.agent import InvestigationAgent
from src.detective.investigation.source_protocol import (
    FOIAInvestigationSource,
    GraphNeighbourhoodSource,
    InvestigationSource,
)
from src.detective.investigation.types import (
    DocumentEvidence,
    Finding,
    HypothesisSnapshot,
    InvestigationBudget,
    InvestigationConfig,
    InvestigationReport,
    InvestigationStep,
    Lead,
    SourceResult,
)

__all__ = [
    "DocumentEvidence",
    "FOIAInvestigationSource",
    "Finding",
    "GraphNeighbourhoodSource",
    "HypothesisSnapshot",
    "InvestigationAgent",
    "InvestigationBudget",
    "InvestigationConfig",
    "InvestigationReport",
    "InvestigationSource",
    "InvestigationStep",
    "Lead",
    "SourceResult",
]
