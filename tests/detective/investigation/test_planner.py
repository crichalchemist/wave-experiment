"""Tests for investigation planner — lead generation and hypothesis seeding."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.core.providers import MockProvider
from src.detective.hypothesis import Hypothesis
from src.detective.investigation.planner import (
    generate_leads,
    generate_seed_hypotheses,
    hypotheses_from_graph_event,
    spawn_alternatives,
)


# ---------------------------------------------------------------------------
# generate_seed_hypotheses
# ---------------------------------------------------------------------------


class TestGenerateSeedHypotheses:
    def test_parses_llm_response(self):
        provider = MockProvider(
            response=(
                "hypothesis: Entity X had secret financial ties\n"
                "confidence: 0.6\n"
                "hypothesis: Entity Y facilitated meetings\n"
                "confidence: 0.4\n"
                "hypothesis: Documents were systematically removed\n"
                "confidence: 0.5\n"
            )
        )
        hypotheses = generate_seed_hypotheses("financial networks", provider)

        assert len(hypotheses) == 3
        assert all(isinstance(h, Hypothesis) for h in hypotheses)
        assert hypotheses[0].text == "Entity X had secret financial ties"
        assert hypotheses[0].confidence == 0.6

    def test_fallback_when_no_parseable_response(self):
        provider = MockProvider(response="I cannot generate hypotheses.")
        hypotheses = generate_seed_hypotheses("test topic", provider)

        assert len(hypotheses) == 1
        assert hypotheses[0].text == "test topic"
        assert hypotheses[0].confidence == 0.3  # default

    def test_calls_provider(self):
        captured = []

        class _CapturingProvider(MockProvider):
            def complete(self, prompt: str, **kwargs) -> str:
                captured.append(prompt)
                return "hypothesis: Test\nconfidence: 0.5"

        provider = _CapturingProvider(response="")
        generate_seed_hypotheses("financial ties", provider)

        assert len(captured) == 1
        assert "financial ties" in captured[0]


# ---------------------------------------------------------------------------
# hypotheses_from_graph_event
# ---------------------------------------------------------------------------


class TestHypothesesFromGraphEvent:
    def test_parses_response(self):
        provider = MockProvider(
            response=(
                "hypothesis: The new connection suggests a cover-up\n"
                "confidence: 0.55\n"
                "hypothesis: This is a coincidental co-occurrence\n"
                "confidence: 0.3\n"
            )
        )
        hypotheses = hypotheses_from_graph_event("Entity A linked to Entity B", provider)

        assert len(hypotheses) == 2
        assert hypotheses[0].confidence == 0.55

    def test_fallback(self):
        provider = MockProvider(response="")
        hypotheses = hypotheses_from_graph_event("new entity found", provider)

        assert len(hypotheses) == 1
        assert hypotheses[0].text == "new entity found"


# ---------------------------------------------------------------------------
# generate_leads
# ---------------------------------------------------------------------------


class TestGenerateLeads:
    def test_parses_leads(self):
        provider = MockProvider(
            response=(
                "query: FBI vault documents about Entity X\n"
                "source: foia_fbi_vault\n"
                "priority: 0.9\n"
                "query: Graph connections of Entity X\n"
                "source: graph_neighbourhood\n"
                "priority: 0.7\n"
            )
        )
        graph = MagicMock()
        graph.nodes.return_value = ["Entity A", "Entity B"]

        hyp = Hypothesis.create(text="Entity X had ties", confidence=0.4)
        leads = generate_leads(
            hypotheses=[hyp],
            graph=graph,
            available_sources=("foia_fbi_vault", "graph_neighbourhood"),
            provider=provider,
            step=0,
        )

        assert len(leads) == 2
        assert leads[0].priority >= leads[1].priority  # sorted descending
        assert leads[0].query == "FBI vault documents about Entity X"

    def test_fallback_when_no_parseable_leads(self):
        provider = MockProvider(response="I cannot generate leads.")
        graph = MagicMock()
        graph.nodes.return_value = []

        hyp = Hypothesis.create(text="Test hypothesis", confidence=0.4)
        leads = generate_leads(
            hypotheses=[hyp],
            graph=graph,
            available_sources=("foia_fbi_vault",),
            provider=provider,
            step=0,
        )

        assert len(leads) >= 1
        assert leads[0].source_id == "foia_fbi_vault"

    def test_empty_hypotheses(self):
        provider = MockProvider(response="anything")
        graph = MagicMock()
        leads = generate_leads(
            hypotheses=[],
            graph=graph,
            available_sources=("foia_fbi_vault",),
            provider=provider,
            step=0,
        )
        assert leads == []

    def test_empty_sources(self):
        provider = MockProvider(response="anything")
        graph = MagicMock()
        hyp = Hypothesis.create(text="test", confidence=0.5)
        leads = generate_leads(
            hypotheses=[hyp],
            graph=graph,
            available_sources=(),
            provider=provider,
            step=0,
        )
        assert leads == []

    def test_invalid_source_falls_back(self):
        """If LLM returns a source not in available_sources, fall back to first."""
        provider = MockProvider(
            response=(
                "query: search something\n"
                "source: nonexistent_source\n"
                "priority: 0.8\n"
            )
        )
        graph = MagicMock()
        graph.nodes.return_value = []

        hyp = Hypothesis.create(text="test", confidence=0.5)
        leads = generate_leads(
            hypotheses=[hyp],
            graph=graph,
            available_sources=("foia_fbi_vault",),
            provider=provider,
            step=0,
        )

        assert all(lead.source_id == "foia_fbi_vault" for lead in leads)


# ---------------------------------------------------------------------------
# spawn_alternatives
# ---------------------------------------------------------------------------


class TestSpawnAlternatives:
    def test_parses_alternatives(self):
        provider = MockProvider(
            response=(
                "hypothesis: Alternative explanation 1\n"
                "confidence: 0.35\n"
                "hypothesis: Alternative explanation 2\n"
                "confidence: 0.25\n"
            )
        )
        hyp = Hypothesis.create(text="Original low-confidence hypothesis", confidence=0.2)
        alternatives = spawn_alternatives(hyp, "analysis was inconclusive", provider)

        assert len(alternatives) == 2
        assert alternatives[0].text == "Alternative explanation 1"

    def test_empty_response(self):
        provider = MockProvider(response="No alternatives.")
        hyp = Hypothesis.create(text="test", confidence=0.3)
        alternatives = spawn_alternatives(hyp, "inconclusive", provider)

        assert alternatives == []

    def test_prompt_contains_hypothesis(self):
        captured = []

        class _CapturingProvider(MockProvider):
            def complete(self, prompt: str, **kwargs) -> str:
                captured.append(prompt)
                return "hypothesis: Alt\nconfidence: 0.3"

        provider = _CapturingProvider(response="")
        hyp = Hypothesis.create(text="Financial ties between A and B", confidence=0.2)
        spawn_alternatives(hyp, "analysis summary", provider)

        assert "Financial ties between A and B" in captured[0]
        assert "0.20" in captured[0]
