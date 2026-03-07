"""Frozen data model for autonomous investigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import uuid

from src.core.types import AssumptionType


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InvestigationBudget:
    """Resource limits for an investigation run."""

    max_steps: int = 50
    max_pages: int = 200
    max_llm_calls: int = 300
    max_time_seconds: int = 3600

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_pages", "max_llm_calls", "max_time_seconds"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TriggerMode = Literal["hypothesis", "topic", "reactive"]


@dataclass(frozen=True)
class InvestigationConfig:
    """Full configuration for an investigation run."""

    trigger_mode: TriggerMode
    seed: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    budget: InvestigationBudget = field(default_factory=InvestigationBudget)
    source_ids: tuple[str, ...] = (
        "foia_fbi_vault",
        "graph_neighbourhood",
        "web_search",
        "news_search",
        "court_listener",
        "sec_edgar",
        "web_occrp",
        "web_iicsa",
    )
    constitution_path: str | None = None
    phi_metrics: dict[str, float] | None = None
    enable_assumption_scan: bool = True
    assumption_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.trigger_mode not in ("hypothesis", "topic", "reactive"):
            raise ValueError(f"Invalid trigger_mode: {self.trigger_mode}")
        if not self.seed:
            raise ValueError("seed must be non-empty")


# ---------------------------------------------------------------------------
# Leads & evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lead:
    """A query to execute against a source."""

    id: str
    query: str
    source_id: str
    priority: float
    parent_hypothesis_id: str | None = None
    generation_step: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be in [0, 1], got {self.priority}")

    @classmethod
    def create(
        cls,
        query: str,
        source_id: str,
        priority: float,
        parent_hypothesis_id: str | None = None,
        generation_step: int = 0,
    ) -> Lead:
        return cls(
            id=uuid.uuid4().hex[:12],
            query=query,
            source_id=source_id,
            priority=priority,
            parent_hypothesis_id=parent_hypothesis_id,
            generation_step=generation_step,
        )


@dataclass(frozen=True)
class DocumentEvidence:
    """A sanitized document retrieved from a source."""

    text: str
    source_url: str
    source_portal: str
    title: str
    risk_level: Literal["low", "medium", "high", "critical"] = "low"
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SourceResult:
    """Result of executing a lead against a source."""

    lead_id: str
    documents: tuple[DocumentEvidence, ...]
    pages_consumed: int
    injection_findings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Assumption detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssumptionDetection:
    """A single assumption detected in a document."""

    module: Literal["A", "B", "C"]
    assumption_type: AssumptionType
    score: float
    source_text: str
    detail: str  # bias_type (A), trigger_phrase (B), presumed_actor (C)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass(frozen=True)
class AssumptionScanResult:
    """Assumption scan results for a single document."""

    document_url: str
    detections: tuple[AssumptionDetection, ...]
    llm_calls: int

    @property
    def has_assumptions(self) -> bool:
        return len(self.detections) > 0

    @property
    def max_score(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.score for d in self.detections)

    @property
    def assumption_summary(self) -> str:
        if not self.detections:
            return "No assumptions detected"
        parts = []
        for d in self.detections:
            parts.append(f"[{d.module}] {d.assumption_type.value}: {d.detail} ({d.score:.2f})")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Steps & findings
# ---------------------------------------------------------------------------

ActionType = Literal[
    "plan", "gather", "analyze", "reflect", "evolve",
    "constitutional_halt", "budget_halt",
]


@dataclass(frozen=True)
class InvestigationStep:
    """A single step in the investigation loop."""

    step_number: int
    action: ActionType
    timestamp: datetime
    leads_generated: int = 0
    documents_gathered: int = 0
    hypotheses_evolved: int = 0
    findings_produced: int = 0
    pages_consumed: int = 0
    llm_calls: int = 0
    assumptions_detected: int = 0


@dataclass(frozen=True)
class Finding:
    """A high-confidence discovery from the investigation."""

    id: str
    description: str
    confidence: float
    supporting_hypothesis_ids: tuple[str, ...]
    supporting_document_urls: tuple[str, ...]
    threatened_constructs: tuple[str, ...] = ()
    welfare_relevance: float = 0.0
    is_injection_finding: bool = False
    is_assumption_finding: bool = False
    assumption_module: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @classmethod
    def create(
        cls,
        description: str,
        confidence: float,
        supporting_hypothesis_ids: tuple[str, ...] = (),
        supporting_document_urls: tuple[str, ...] = (),
        threatened_constructs: tuple[str, ...] = (),
        welfare_relevance: float = 0.0,
        is_injection_finding: bool = False,
        is_assumption_finding: bool = False,
        assumption_module: str = "",
    ) -> Finding:
        return cls(
            id=uuid.uuid4().hex[:12],
            description=description,
            confidence=confidence,
            supporting_hypothesis_ids=supporting_hypothesis_ids,
            supporting_document_urls=supporting_document_urls,
            threatened_constructs=threatened_constructs,
            welfare_relevance=welfare_relevance,
            is_injection_finding=is_injection_finding,
            is_assumption_finding=is_assumption_finding,
            assumption_module=assumption_module,
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

TerminationReason = Literal[
    "budget_max_steps",
    "budget_max_pages",
    "budget_max_llm_calls",
    "budget_max_time",
    "leads_exhausted",
    "constitutional_halt",
]


@dataclass(frozen=True)
class HypothesisSnapshot:
    """Serializable snapshot of a hypothesis for reports."""

    id: str
    text: str
    confidence: float
    parent_id: str | None
    welfare_relevance: float
    threatened_constructs: tuple[str, ...]
    combined_score: float


@dataclass(frozen=True)
class InvestigationReport:
    """Final report produced by an investigation run."""

    config: InvestigationConfig
    findings: tuple[Finding, ...]
    hypothesis_tree: tuple[HypothesisSnapshot, ...]
    steps: tuple[InvestigationStep, ...]
    total_pages: int
    total_llm_calls: int
    total_documents: int
    elapsed_seconds: float
    termination_reason: TerminationReason
    graph_edges_added: int = 0
    total_assumptions_detected: int = 0
