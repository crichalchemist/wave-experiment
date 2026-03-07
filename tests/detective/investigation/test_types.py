"""Tests for investigation frozen dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

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


# ---------------------------------------------------------------------------
# InvestigationBudget
# ---------------------------------------------------------------------------


class TestInvestigationBudget:
    def test_defaults(self):
        b = InvestigationBudget()
        assert b.max_steps == 50
        assert b.max_pages == 200
        assert b.max_llm_calls == 300
        assert b.max_time_seconds == 3600

    def test_custom_values(self):
        b = InvestigationBudget(max_steps=10, max_pages=20, max_llm_calls=30, max_time_seconds=60)
        assert b.max_steps == 10
        assert b.max_pages == 20

    def test_frozen(self):
        b = InvestigationBudget()
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.max_steps = 100  # type: ignore[misc]

    def test_invalid_zero_steps(self):
        with pytest.raises(ValueError, match="max_steps"):
            InvestigationBudget(max_steps=0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="max_pages"):
            InvestigationBudget(max_pages=-1)


# ---------------------------------------------------------------------------
# InvestigationConfig
# ---------------------------------------------------------------------------


class TestInvestigationConfig:
    def test_hypothesis_mode(self):
        c = InvestigationConfig(trigger_mode="hypothesis", seed="test hypothesis")
        assert c.trigger_mode == "hypothesis"
        assert c.seed == "test hypothesis"
        assert len(c.id) == 12

    def test_topic_mode(self):
        c = InvestigationConfig(trigger_mode="topic", seed="financial networks")
        assert c.trigger_mode == "topic"

    def test_reactive_mode(self):
        c = InvestigationConfig(trigger_mode="reactive", seed="new entity discovered")
        assert c.trigger_mode == "reactive"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid trigger_mode"):
            InvestigationConfig(trigger_mode="invalid", seed="test")  # type: ignore[arg-type]

    def test_empty_seed(self):
        with pytest.raises(ValueError, match="seed must be non-empty"):
            InvestigationConfig(trigger_mode="topic", seed="")

    def test_frozen(self):
        c = InvestigationConfig(trigger_mode="topic", seed="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.seed = "new"  # type: ignore[misc]

    def test_custom_budget(self):
        b = InvestigationBudget(max_steps=5)
        c = InvestigationConfig(trigger_mode="topic", seed="test", budget=b)
        assert c.budget.max_steps == 5


# ---------------------------------------------------------------------------
# Lead
# ---------------------------------------------------------------------------


class TestLead:
    def test_create_factory(self):
        lead = Lead.create(query="search term", source_id="foia_fbi_vault", priority=0.8)
        assert lead.query == "search term"
        assert lead.source_id == "foia_fbi_vault"
        assert lead.priority == 0.8
        assert len(lead.id) == 12

    def test_frozen(self):
        lead = Lead.create(query="q", source_id="s", priority=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            lead.query = "new"  # type: ignore[misc]

    def test_invalid_priority(self):
        with pytest.raises(ValueError, match="priority"):
            Lead(id="x", query="q", source_id="s", priority=1.5)

    def test_parent_hypothesis(self):
        lead = Lead.create(query="q", source_id="s", priority=0.5, parent_hypothesis_id="hyp-1")
        assert lead.parent_hypothesis_id == "hyp-1"


# ---------------------------------------------------------------------------
# DocumentEvidence
# ---------------------------------------------------------------------------


class TestDocumentEvidence:
    def test_basic(self):
        doc = DocumentEvidence(
            text="content",
            source_url="https://example.com",
            source_portal="fbi_vault",
            title="Doc Title",
        )
        assert doc.risk_level == "low"
        assert doc.metadata == ()


# ---------------------------------------------------------------------------
# SourceResult
# ---------------------------------------------------------------------------


class TestSourceResult:
    def test_basic(self):
        result = SourceResult(
            lead_id="l1",
            documents=(),
            pages_consumed=5,
        )
        assert result.injection_findings == ()
        assert result.pages_consumed == 5


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


class TestFinding:
    def test_create_factory(self):
        f = Finding.create(description="discovered link", confidence=0.85)
        assert f.description == "discovered link"
        assert f.confidence == 0.85
        assert f.is_injection_finding is False
        assert len(f.id) == 12

    def test_injection_finding(self):
        f = Finding.create(
            description="injection attempt",
            confidence=1.0,
            is_injection_finding=True,
        )
        assert f.is_injection_finding is True

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            Finding(
                id="x",
                description="d",
                confidence=1.5,
                supporting_hypothesis_ids=(),
                supporting_document_urls=(),
            )

    def test_frozen(self):
        f = Finding.create(description="d", confidence=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.confidence = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InvestigationStep
# ---------------------------------------------------------------------------


class TestInvestigationStep:
    def test_plan_step(self):
        from datetime import datetime, timezone
        step = InvestigationStep(
            step_number=0,
            action="plan",
            timestamp=datetime.now(timezone.utc),
            leads_generated=5,
        )
        assert step.action == "plan"
        assert step.leads_generated == 5
        assert step.documents_gathered == 0

    def test_reflect_action(self):
        from datetime import datetime, timezone
        step = InvestigationStep(
            step_number=1,
            action="reflect",
            timestamp=datetime.now(timezone.utc),
            llm_calls=3,
        )
        assert step.action == "reflect"


# ---------------------------------------------------------------------------
# HypothesisSnapshot & InvestigationReport
# ---------------------------------------------------------------------------


class TestHypothesisSnapshot:
    def test_basic(self):
        snap = HypothesisSnapshot(
            id="h1",
            text="test",
            confidence=0.7,
            parent_id=None,
            welfare_relevance=0.3,
            threatened_constructs=("c",),
            combined_score=0.6,
        )
        assert snap.combined_score == 0.6
        assert snap.threatened_constructs == ("c",)


class TestInvestigationReport:
    def test_basic(self):
        config = InvestigationConfig(trigger_mode="topic", seed="test")
        report = InvestigationReport(
            config=config,
            findings=(),
            hypothesis_tree=(),
            steps=(),
            total_pages=10,
            total_llm_calls=20,
            total_documents=5,
            elapsed_seconds=30.5,
            termination_reason="leads_exhausted",
        )
        assert report.termination_reason == "leads_exhausted"
        assert report.graph_edges_added == 0
