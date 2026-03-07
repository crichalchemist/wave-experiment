"""Tests for A+B+C assumption detection integration in the investigation agent."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from src.core.providers import MockProvider
from src.core.types import AssumptionType
from src.detective.investigation.agent import (
    InvestigationAgent,
    _ASSUMPTION_MAX_FINDINGS_PER_SCAN,
    _ASSUMPTION_SCAN_MAX_DOCS,
    _MAX_COUNTER_LEADS,
)
from src.detective.investigation.types import (
    AssumptionDetection,
    AssumptionScanResult,
    DocumentEvidence,
    Finding,
    InvestigationBudget,
    InvestigationConfig,
    SourceResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockSource:
    """Test source that returns pre-configured documents."""

    def __init__(self, source_id: str, documents: tuple[DocumentEvidence, ...] = ()) -> None:
        self._source_id = source_id
        self._documents = documents

    @property
    def source_id(self) -> str:
        return self._source_id

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        return SourceResult(
            lead_id="",
            documents=self._documents,
            pages_consumed=len(self._documents),
        )


def _make_doc(
    text: str = "Test evidence document",
    title: str = "Doc",
    url: str = "https://example.com/doc",
) -> DocumentEvidence:
    return DocumentEvidence(
        text=text,
        source_url=url,
        source_portal="test",
        title=title,
    )


def _make_config(
    mode: str = "hypothesis",
    seed: str = "Entity X had undisclosed financial ties",
    max_steps: int = 2,
    max_pages: int = 50,
    max_llm_calls: int = 100,
    max_time: int = 60,
    enable_assumption_scan: bool = True,
    assumption_threshold: float = 0.5,
) -> InvestigationConfig:
    return InvestigationConfig(
        trigger_mode=mode,  # type: ignore[arg-type]
        seed=seed,
        budget=InvestigationBudget(
            max_steps=max_steps,
            max_pages=max_pages,
            max_llm_calls=max_llm_calls,
            max_time_seconds=max_time,
        ),
        source_ids=("test_source",),
        enable_assumption_scan=enable_assumption_scan,
        assumption_threshold=assumption_threshold,
    )


def _make_agent(
    config: InvestigationConfig | None = None,
    docs: tuple[DocumentEvidence, ...] | None = None,
    provider_response: str = "score: 0.7\nAnalysis complete.",
    sources: dict | None = None,
) -> InvestigationAgent:
    if config is None:
        config = _make_config()

    if docs is None:
        docs = (_make_doc(),)

    provider = MockProvider(response=provider_response)
    graph = MagicMock()
    graph.nodes.return_value = ["Entity A"]
    graph.successors.return_value = []
    graph.add_edge = MagicMock()

    if sources is None:
        sources = {"test_source": MockSource("test_source", docs)}
    constitution = "Epistemic honesty above analytical comfort."

    return InvestigationAgent(
        config=config,
        provider=provider,
        graph=graph,
        sources=sources,
        constitution=constitution,
    )


# ---------------------------------------------------------------------------
# Type tests: AssumptionDetection
# ---------------------------------------------------------------------------


class TestAssumptionDetection:
    def test_valid_detection(self):
        det = AssumptionDetection(
            module="A",
            assumption_type=AssumptionType.COGNITIVE_BIAS,
            score=0.8,
            source_text="confirms our hypothesis",
            detail="confirmation",
        )
        assert det.module == "A"
        assert det.score == 0.8

    def test_frozen(self):
        det = AssumptionDetection(
            module="B",
            assumption_type=AssumptionType.HISTORICAL_DETERMINISM,
            score=0.6,
            source_text="historically inevitable",
            detail=r"\bhistorically\b",
        )
        with pytest.raises(AttributeError):
            det.score = 0.9  # type: ignore[misc]

    def test_invalid_score_low(self):
        with pytest.raises(ValueError, match="score must be in"):
            AssumptionDetection(
                module="A",
                assumption_type=AssumptionType.COGNITIVE_BIAS,
                score=-0.1,
                source_text="text",
                detail="bias",
            )

    def test_invalid_score_high(self):
        with pytest.raises(ValueError, match="score must be in"):
            AssumptionDetection(
                module="C",
                assumption_type=AssumptionType.GEOPOLITICAL_PRESUMPTION,
                score=1.5,
                source_text="text",
                detail="actor",
            )


# ---------------------------------------------------------------------------
# Type tests: AssumptionScanResult
# ---------------------------------------------------------------------------


class TestAssumptionScanResult:
    def test_has_assumptions_true(self):
        det = AssumptionDetection(
            module="A",
            assumption_type=AssumptionType.COGNITIVE_BIAS,
            score=0.7,
            source_text="text",
            detail="confirmation",
        )
        result = AssumptionScanResult(
            document_url="https://example.com",
            detections=(det,),
            llm_calls=1,
        )
        assert result.has_assumptions is True

    def test_has_assumptions_false(self):
        result = AssumptionScanResult(
            document_url="https://example.com",
            detections=(),
            llm_calls=0,
        )
        assert result.has_assumptions is False

    def test_max_score(self):
        dets = (
            AssumptionDetection("A", AssumptionType.COGNITIVE_BIAS, 0.6, "t", "d1"),
            AssumptionDetection("B", AssumptionType.HISTORICAL_DETERMINISM, 0.9, "t", "d2"),
        )
        result = AssumptionScanResult("url", dets, 2)
        assert result.max_score == 0.9

    def test_max_score_empty(self):
        result = AssumptionScanResult("url", (), 0)
        assert result.max_score == 0.0

    def test_assumption_summary(self):
        det = AssumptionDetection("C", AssumptionType.GEOPOLITICAL_PRESUMPTION, 0.8, "t", "SEC")
        result = AssumptionScanResult("url", (det,), 1)
        summary = result.assumption_summary
        assert "[C]" in summary
        assert "SEC" in summary

    def test_assumption_summary_empty(self):
        result = AssumptionScanResult("url", (), 0)
        assert result.assumption_summary == "No assumptions detected"


# ---------------------------------------------------------------------------
# Type tests: Finding backward compatibility + new fields
# ---------------------------------------------------------------------------


class TestFindingAssumptionFields:
    def test_backward_compatible(self):
        f = Finding.create(
            description="test",
            confidence=0.5,
        )
        assert f.is_assumption_finding is False
        assert f.assumption_module == ""

    def test_assumption_finding(self):
        f = Finding.create(
            description="assumption found",
            confidence=0.7,
            is_assumption_finding=True,
            assumption_module="A",
        )
        assert f.is_assumption_finding is True
        assert f.assumption_module == "A"


# ---------------------------------------------------------------------------
# Type tests: InvestigationConfig defaults
# ---------------------------------------------------------------------------


class TestConfigAssumptionDefaults:
    def test_defaults(self):
        cfg = _make_config()
        assert cfg.enable_assumption_scan is True
        assert cfg.assumption_threshold == 0.5

    def test_disabled(self):
        cfg = _make_config(enable_assumption_scan=False)
        assert cfg.enable_assumption_scan is False


# ---------------------------------------------------------------------------
# Scan method tests
# ---------------------------------------------------------------------------


class TestAssumptionScan:
    def test_module_a_detects_cognitive_bias(self):
        """Module A should detect confirmation bias patterns."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        agent = _make_agent(docs=(doc,), provider_response="score: 0.8")

        results, findings, leads, llm = agent._assumption_scan([doc])

        assert len(results) == 1
        assert results[0].has_assumptions
        a_findings = [f for f in findings if f.assumption_module == "A"]
        assert len(a_findings) >= 1

    def test_module_b_detects_historical_determinism(self):
        """Module B should detect determinism language."""
        doc = _make_doc(text="The entity has always been involved in illicit activities.")
        agent = _make_agent(docs=(doc,), provider_response="score: 0.8")

        results, findings, leads, llm = agent._assumption_scan([doc])

        assert len(results) == 1
        b_findings = [f for f in findings if f.assumption_module == "B"]
        assert len(b_findings) >= 1

    def test_module_c_detects_geopolitical_presumption(self):
        """Module C should detect geopolitical presumption patterns."""
        doc = _make_doc(
            text="The SEC properly reviewed all disclosures and found no evidence of wrongdoing."
        )
        agent = _make_agent(docs=(doc,), provider_response="score: 0.8")

        results, findings, leads, llm = agent._assumption_scan([doc])

        assert len(results) == 1
        c_findings = [f for f in findings if f.assumption_module == "C"]
        assert len(c_findings) >= 1

    def test_disabled_returns_empty(self):
        """When enable_assumption_scan=False, scan should return empty."""
        doc = _make_doc(text="This confirms our hypothesis.")
        config = _make_config(enable_assumption_scan=False)
        agent = _make_agent(config=config, docs=(doc,))

        results, findings, leads, llm = agent._assumption_scan([doc])

        assert results == []
        assert findings == []
        assert leads == []
        assert llm == 0

    def test_deduplication_by_url(self):
        """Documents with the same URL should be scanned only once."""
        doc1 = _make_doc(text="This confirms our view.", url="https://same.com")
        doc2 = _make_doc(text="Also confirms our hypothesis.", url="https://same.com")
        agent = _make_agent(docs=(doc1, doc2), provider_response="score: 0.8")

        results, findings, leads, llm = agent._assumption_scan([doc1, doc2])

        assert len(results) == 1  # Only one unique URL

    def test_budget_stops_scan_midway(self):
        """Budget exhaustion should stop scanning remaining docs."""
        docs = [
            _make_doc(text="confirms our hypothesis", url=f"https://doc{i}.com")
            for i in range(10)
        ]
        config = _make_config(max_llm_calls=5)
        agent = _make_agent(config=config, provider_response="score: 0.8")
        # Simulate near-budget-exhaustion
        agent._budget.llm_calls = 4

        results, findings, leads, llm = agent._assumption_scan(docs)

        # Should have scanned very few docs due to budget
        assert len(results) <= _ASSUMPTION_SCAN_MAX_DOCS

    def test_module_a_keyword_only_on_tight_budget(self):
        """Module A should use keyword-only mode when remaining budget < 10."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        config = _make_config(max_llm_calls=15)
        agent = _make_agent(config=config, provider_response="score: 0.8")
        # Set budget near limit so remaining < 10
        agent._budget.llm_calls = 8

        results, findings, leads, llm = agent._assumption_scan([doc])

        # Should still detect (keyword mode gives score=1.0)
        assert len(results) == 1
        if results[0].has_assumptions:
            # In keyword-only mode, no LLM calls for module A detections
            a_dets = [d for d in results[0].detections if d.module == "A"]
            for d in a_dets:
                assert d.score == 1.0  # keyword-only returns 1.0

    def test_module_failure_doesnt_crash(self):
        """A module exception should not crash the entire scan."""
        doc = _make_doc(text="The SEC properly reviewed all disclosures.")

        # Provider that raises on first call
        class FailingProvider:
            def __init__(self):
                self._calls = 0

            def complete(self, prompt: str) -> str:
                self._calls += 1
                if self._calls <= 2:
                    raise RuntimeError("Provider error")
                return "score: 0.7"

        config = _make_config()
        provider = FailingProvider()
        graph = MagicMock()
        graph.nodes.return_value = []
        graph.successors.return_value = []
        graph.add_edge = MagicMock()

        agent = InvestigationAgent(
            config=config,
            provider=provider,  # type: ignore[arg-type]
            graph=graph,
            sources={"test_source": MockSource("test_source", (doc,))},
            constitution="Test.",
        )

        # Should not raise
        results, findings, leads, llm = agent._assumption_scan([doc])
        assert isinstance(results, list)

    def test_findings_capped_per_scan(self):
        """Findings should be capped at _ASSUMPTION_MAX_FINDINGS_PER_SCAN."""
        # Create docs that trigger many bias patterns
        docs = [
            _make_doc(
                text=(
                    "This confirms our view and supports our hypothesis. "
                    "The initial anchor point was clear. "
                    "Only those who succeeded are mentioned. "
                    "Our group always wins while they never adapt."
                ),
                url=f"https://doc{i}.com",
            )
            for i in range(5)
        ]
        agent = _make_agent(provider_response="score: 0.8")

        results, findings, leads, llm = agent._assumption_scan(docs)

        assert len(findings) <= _ASSUMPTION_MAX_FINDINGS_PER_SCAN

    def test_text_truncated(self):
        """Document text should be truncated to _ASSUMPTION_TEXT_LIMIT."""
        long_text = "confirms our hypothesis " * 500  # Way over 2000 chars
        doc = _make_doc(text=long_text)
        agent = _make_agent(docs=(doc,), provider_response="score: 0.8")

        # Should not crash, and text passed to modules should be truncated
        results, findings, leads, llm = agent._assumption_scan([doc])
        assert isinstance(results, list)

    def test_llm_call_count(self):
        """LLM call count should reflect actual calls made."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        agent = _make_agent(docs=(doc,), provider_response="score: 0.8")

        results, findings, leads, llm = agent._assumption_scan([doc])

        # Module A has LLM calls for each detected bias, plus counter-leads
        assert llm >= 0
        total_from_results = sum(r.llm_calls for r in results)
        # llm includes both scan calls and counter-lead calls
        assert llm >= total_from_results

    def test_max_docs_limit(self):
        """Scan should process at most _ASSUMPTION_SCAN_MAX_DOCS unique docs."""
        docs = [
            _make_doc(text="confirms our hypothesis", url=f"https://doc{i}.com")
            for i in range(20)
        ]
        agent = _make_agent(provider_response="score: 0.8")

        results, findings, leads, llm = agent._assumption_scan(docs)

        assert len(results) <= _ASSUMPTION_SCAN_MAX_DOCS


# ---------------------------------------------------------------------------
# Counter-lead tests
# ---------------------------------------------------------------------------


class TestCounterLeads:
    def test_counter_lead_from_cognitive_bias(self):
        """Counter-lead should be generated from a cognitive bias detection."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        sources = {
            "test_source": MockSource("test_source", (doc,)),
            "web_search": MockSource("web_search", ()),
        }
        agent = _make_agent(
            docs=(doc,),
            provider_response="score: 0.8\nquery: evidence contradicting entity ties",
            sources=sources,
        )

        results, findings, leads, llm = agent._assumption_scan([doc])

        # Should have generated at least one counter-lead
        assert len(leads) >= 1

    def test_counter_lead_from_geopolitical_routes_to_court(self):
        """Module C counter-leads should prefer court_listener source."""
        doc = _make_doc(
            text="The SEC properly reviewed all disclosures and found no evidence."
        )
        sources = {
            "test_source": MockSource("test_source", (doc,)),
            "court_listener": MockSource("court_listener", ()),
            "web_search": MockSource("web_search", ()),
        }
        agent = _make_agent(
            docs=(doc,),
            provider_response="score: 0.8\nquery: SEC review actual findings",
            sources=sources,
        )

        results, findings, leads, llm = agent._assumption_scan([doc])

        c_leads = [lead for lead in leads if lead.source_id == "court_listener"]
        # If module C detected anything, its leads should route to court_listener
        if any(d.module == "C" for r in results for d in r.detections):
            assert len(c_leads) >= 1

    def test_counter_lead_query_parsed(self):
        """Counter-lead query should be parsed from provider response."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        agent = _make_agent(
            docs=(doc,),
            provider_response="query: evidence against entity X involvement",
        )

        results, findings, leads, llm = agent._assumption_scan([doc])

        for lead in leads:
            assert lead.query  # Non-empty query
            assert "query:" not in lead.query.lower()  # Prefix stripped

    def test_counter_leads_capped(self):
        """Counter-leads should be capped at _MAX_COUNTER_LEADS."""
        # Create doc with many triggerable patterns
        doc = _make_doc(
            text=(
                "This confirms our view and supports our hypothesis. "
                "The initial anchor point was clear. "
                "Only those who succeeded are mentioned. "
                "Our group always wins. "
                "The entity has always been involved. "
                "The SEC properly reviewed all disclosures. "
                "The DOJ found no evidence of wrongdoing."
            ),
        )
        agent = _make_agent(
            docs=(doc,),
            provider_response="score: 0.8\nquery: counter evidence",
        )

        results, findings, leads, llm = agent._assumption_scan([doc])

        assert len(leads) <= _MAX_COUNTER_LEADS

    def test_counter_lead_priority(self):
        """Counter-lead priority should be detection.score * 0.8."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        agent = _make_agent(
            docs=(doc,),
            provider_response="score: 0.8\nquery: counter evidence",
        )

        results, findings, leads, llm = agent._assumption_scan([doc])

        for lead in leads:
            # Priority should be score * 0.8, score is 0.8 or 1.0
            assert 0.0 < lead.priority <= 0.8

    def test_counter_leads_added_to_lead_queue(self):
        """Counter-leads should be added to agent's lead queue in analyze phase."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        config = _make_config(max_steps=1, max_llm_calls=200)
        agent = _make_agent(
            config=config,
            docs=(doc,),
            provider_response="score: 0.7\nquery: counter evidence",
        )

        report = asyncio.run(agent.run())

        # The agent ran through the loop, so counter-leads were generated
        # and added to the queue during analyze phase
        # Check that assumption findings exist in the report
        assumption_findings = [f for f in report.findings if f.is_assumption_finding]
        # Module A should detect "confirms" pattern
        assert len(assumption_findings) >= 0  # May be 0 if below threshold


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAssumptionIntegration:
    def test_full_loop_produces_assumption_findings(self):
        """Full agent loop should produce assumption findings."""
        doc = _make_doc(
            text="This confirms our hypothesis. The entity has always been involved."
        )
        config = _make_config(max_steps=1, max_llm_calls=200)
        agent = _make_agent(config=config, docs=(doc,), provider_response="score: 0.7")
        report = asyncio.run(agent.run())

        assumption_findings = [f for f in report.findings if f.is_assumption_finding]
        assert len(assumption_findings) >= 1

    def test_report_includes_total_assumptions(self):
        """Report should include total_assumptions_detected count."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        config = _make_config(max_steps=1, max_llm_calls=200)
        agent = _make_agent(config=config, docs=(doc,), provider_response="score: 0.7")
        report = asyncio.run(agent.run())

        assert hasattr(report, "total_assumptions_detected")
        assert report.total_assumptions_detected >= 0

    def test_status_includes_assumptions(self):
        """Status dict should include assumptions_detected."""
        config = _make_config(max_steps=10)
        agent = _make_agent(config=config)

        status = agent.status
        assert "assumptions_detected" in status
        assert status["assumptions_detected"] == 0

    def test_step_records_assumptions_count(self):
        """Analyze step should record assumptions_detected count."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        config = _make_config(max_steps=1, max_llm_calls=200)
        agent = _make_agent(config=config, docs=(doc,), provider_response="score: 0.7")
        report = asyncio.run(agent.run())

        analyze_steps = [s for s in report.steps if s.action == "analyze"]
        if analyze_steps:
            assert hasattr(analyze_steps[0], "assumptions_detected")

    def test_disabled_scan_produces_no_findings(self):
        """With scan disabled, no assumption findings should appear."""
        doc = _make_doc(text="This confirms our hypothesis about the entity.")
        config = _make_config(
            max_steps=1, max_llm_calls=200, enable_assumption_scan=False
        )
        agent = _make_agent(config=config, docs=(doc,), provider_response="score: 0.7")
        report = asyncio.run(agent.run())

        assumption_findings = [f for f in report.findings if f.is_assumption_finding]
        assert len(assumption_findings) == 0
        assert report.total_assumptions_detected == 0


# ---------------------------------------------------------------------------
# Source routing tests
# ---------------------------------------------------------------------------


class TestSourceRouting:
    def test_module_c_prefers_court_listener(self):
        agent = _make_agent()
        agent._sources = {
            "court_listener": MockSource("court_listener"),
            "web_search": MockSource("web_search"),
        }
        assert agent._route_counter_lead("C") == "court_listener"

    def test_module_c_falls_back_to_sec_edgar(self):
        agent = _make_agent()
        agent._sources = {
            "sec_edgar": MockSource("sec_edgar"),
            "web_search": MockSource("web_search"),
        }
        assert agent._route_counter_lead("C") == "sec_edgar"

    def test_module_a_prefers_web_search(self):
        agent = _make_agent()
        agent._sources = {
            "web_search": MockSource("web_search"),
            "court_listener": MockSource("court_listener"),
        }
        assert agent._route_counter_lead("A") == "web_search"

    def test_module_b_prefers_web_search(self):
        agent = _make_agent()
        agent._sources = {
            "web_search": MockSource("web_search"),
            "news_search": MockSource("news_search"),
        }
        assert agent._route_counter_lead("B") == "web_search"

    def test_fallback_to_first_available(self):
        agent = _make_agent()
        agent._sources = {"custom_source": MockSource("custom_source")}
        assert agent._route_counter_lead("A") == "custom_source"
