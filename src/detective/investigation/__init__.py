"""Autonomous investigation agent — plan → gather → analyze → reflect → evolve → enrich."""

from src.detective.investigation.agent import InvestigationAgent
from src.detective.investigation.clearnet_sources import (
    CourtListenerSource,
    IICSASource,
    NewsSearchSource,
    OCCRPSource,
    SECEdgarSource,
    WebSearchSource,
)
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
    "CourtListenerSource",
    "DocumentEvidence",
    "FOIAInvestigationSource",
    "Finding",
    "GraphNeighbourhoodSource",
    "HypothesisSnapshot",
    "IICSASource",
    "InvestigationAgent",
    "InvestigationBudget",
    "InvestigationConfig",
    "InvestigationReport",
    "InvestigationSource",
    "InvestigationStep",
    "Lead",
    "NewsSearchSource",
    "OCCRPSource",
    "SECEdgarSource",
    "SourceResult",
    "WebSearchSource",
]
