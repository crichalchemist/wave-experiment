"""End-to-end smoke tests exercising all major subsystems together.

MockProvider is used throughout — no external LLM calls are made.
Each test wires subsystems together in the way a real analysis run would,
verifying integration contracts rather than unit-level behaviour.
"""
from __future__ import annotations

from src.core.providers import MockProvider
from src.core.types import RelationType
from src.data.graph_store import InMemoryGraph
from src.detective.experience import EMPTY_LIBRARY
from src.detective.evolution import evolve_hypothesis
from src.detective.hypothesis import Hypothesis
from src.inference.pipeline import (
    analyze,
    verify_inline,
    fuse_reasoning,
    parse_intent,
    Evidence,
    ReasoningChain,
    AnalysisResult,
    AnalyticalIntent,
)
from src.inference.reflection import inject_reflection_trigger
from evaluation.independent_discovery import DiscoveryEvaluation, summarise, EvaluationSummary


# ---------------------------------------------------------------------------
# Shared mock constitution — avoids dependency on file-system or real providers
# ---------------------------------------------------------------------------

class _MockConstitution:
    def critique(self, text: str) -> str:
        return "No issues found."

    def revise(self, text: str, critique: str) -> str:
        return text


# ---------------------------------------------------------------------------
# 1. Full pipeline smoke test
# ---------------------------------------------------------------------------

def test_full_pipeline_returns_analysis_result() -> None:
    """All 6 AnalysisResult fields are present after a full 4-layer pipeline run."""
    provider = MockProvider(response="Step one reasoning.\nConclusion: plausible.")
    graph = InMemoryGraph()
    constitution = _MockConstitution()

    result = analyze(
        claim="Entity A was active throughout the period",
        provider=provider,
        graph=graph,
        library=EMPTY_LIBRARY,
        constitution=constitution,
    )

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.claim, str) and result.claim
    assert isinstance(result.intent, AnalyticalIntent)
    assert isinstance(result.evidence, tuple)
    assert isinstance(result.reasoning, ReasoningChain)
    assert isinstance(result.verdict, str) and result.verdict
    assert isinstance(result.confidence, float)


# ---------------------------------------------------------------------------
# 2. Hypothesis evolution round-trip
# ---------------------------------------------------------------------------

def test_hypothesis_evolution_round_trip() -> None:
    """evolve_hypothesis returns a new Hypothesis whose parent_id is the original id."""
    provider = MockProvider(response="0.8")
    original = Hypothesis.create(
        text="Entity A was connected to Entity B via financial transactions.",
        confidence=0.5,
    )

    evolved, experience = evolve_hypothesis(
        hypothesis=original,
        new_evidence="Bank records confirm wire transfers between A and B in 2015.",
        library=EMPTY_LIBRARY,
        provider=provider,
    )

    assert evolved.id != original.id
    assert evolved.parent_id == original.id
    assert isinstance(evolved, Hypothesis)
    assert isinstance(evolved.confidence, float)


# ---------------------------------------------------------------------------
# 3. InMemoryGraph stores and retrieves edges
# ---------------------------------------------------------------------------

def test_knowledge_graph_stores_and_retrieves_edges() -> None:
    """add_edge followed by get_edge returns a KnowledgeEdge with correct fields."""
    from src.core.types import KnowledgeEdge

    graph = InMemoryGraph()
    graph.add_edge("A", "B", RelationType.CAUSAL, 0.9)

    edge = graph.get_edge("A", "B")

    assert edge is not None
    assert isinstance(edge, KnowledgeEdge)
    assert edge.source == "A"
    assert edge.target == "B"
    assert edge.relation == RelationType.CAUSAL
    assert edge.confidence == 0.9

    successors = graph.successors("A")
    assert successors == ["B"]


# ---------------------------------------------------------------------------
# 4. inject_reflection_trigger integrates with pipeline prompt construction
# ---------------------------------------------------------------------------

def test_reflection_trigger_integrates_with_pipeline() -> None:
    """inject_reflection_trigger embeds the trigger and principle into the prompt."""
    from src.inference.reflection import REFLECTION_TRIGGER

    original_text = "Analysis text."
    principle = "Be honest."

    result = inject_reflection_trigger(original_text, principle)

    assert "Wait" in result
    assert principle in result
    assert original_text[:len(original_text) - 1] in result  # body of text is preserved


# ---------------------------------------------------------------------------
# 5. No external calls — MockProvider is the only LLM used
# ---------------------------------------------------------------------------

def test_no_external_calls_in_pipeline() -> None:
    """Pipeline completes without error; verdict equals the MockProvider's configured response."""
    configured_response = "Definitive conclusion from mock."
    provider = MockProvider(response=configured_response)
    graph = InMemoryGraph()
    constitution = _MockConstitution()

    result = analyze(
        claim="Claim requiring no external data.",
        provider=provider,
        graph=graph,
        library=EMPTY_LIBRARY,
        constitution=constitution,
    )

    # The verdict is the raw output of provider.complete(), which is our configured response.
    assert result.verdict == configured_response


# ---------------------------------------------------------------------------
# 6. Pipeline with a populated graph surfaces Evidence
# ---------------------------------------------------------------------------

def test_pipeline_with_populated_graph() -> None:
    """Running analyze with populated graph yields at least one Evidence with relevance > 0."""
    provider = MockProvider(response="Step.\nConclusion.")
    graph = InMemoryGraph()
    graph.add_edge("EntityA", "EntityB", RelationType.CAUSAL, 0.9)
    graph.add_edge("EntityB", "EntityC", RelationType.SEQUENTIAL, 0.7)
    graph.add_edge("EntityA", "EntityC", RelationType.CONDITIONAL, 0.6)
    constitution = _MockConstitution()

    result = analyze(
        claim="EntityA related to EntityB and EntityC",
        provider=provider,
        graph=graph,
        library=EMPTY_LIBRARY,
        constitution=constitution,
    )

    assert len(result.evidence) >= 1
    assert all(isinstance(e, Evidence) for e in result.evidence)
    assert any(e.relevance > 0 for e in result.evidence)


# ---------------------------------------------------------------------------
# 7. Evaluation harness integrates — summarise returns correct counts and bounds
# ---------------------------------------------------------------------------

def test_evaluation_harness_integrates() -> None:
    """summarise over 3 DiscoveryEvaluation records yields total=3 and bounded floats."""
    evals = [
        DiscoveryEvaluation(
            gap_id="gap-001",
            independently_discovered=True,
            path_taken="Analyst A found this via temporal analysis.",
            matches_system=True,
        ),
        DiscoveryEvaluation(
            gap_id="gap-002",
            independently_discovered=True,
            path_taken="Analyst B confirmed via document review.",
            matches_system=False,
        ),
        DiscoveryEvaluation(
            gap_id="gap-003",
            independently_discovered=False,
            path_taken="Analyst C did not find this gap independently.",
            matches_system=True,
        ),
    ]

    summary = summarise(evals)

    assert isinstance(summary, EvaluationSummary)
    assert summary.total == 3
    assert 0.0 <= summary.discovery_rate <= 1.0
    assert 0.0 <= summary.precision <= 1.0
    assert 0.0 <= summary.recall <= 1.0
    assert 0.0 <= summary.f1 <= 1.0


# ---------------------------------------------------------------------------
# 8. Constitutional critique applies — critique text reaches the verdict prompt
# ---------------------------------------------------------------------------

def test_constitutional_critique_applies() -> None:
    """_MockConstitution.critique() return value is used as the reflection principle."""
    critique_text = "Verify temporal consistency before concluding."

    class _SpecificConstitution:
        def critique(self, text: str) -> str:
            return critique_text

        def revise(self, text: str, critique: str) -> str:
            return text

    # MockProvider records the prompt it receives via its complete() call.
    # We capture the prompt by subclassing MockProvider to store the last call.
    captured_prompts: list[str] = []

    class _CapturingProvider(MockProvider):
        def complete(self, prompt: str, **kwargs) -> str:
            captured_prompts.append(prompt)
            return super().complete(prompt, **kwargs)

    provider = _CapturingProvider(response="Verdict: plausible with caveats.")
    graph = InMemoryGraph()
    constitution = _SpecificConstitution()

    result = analyze(
        claim="Entity A acted consistently over the period.",
        provider=provider,
        graph=graph,
        library=EMPTY_LIBRARY,
        constitution=constitution,
    )

    # The constitution critique is injected into the verify_inline prompt.
    # At least one captured prompt must contain the critique text.
    assert any(critique_text in p for p in captured_prompts), (
        f"Expected {critique_text!r} to appear in at least one prompt, "
        f"but saw: {captured_prompts}"
    )
    assert isinstance(result.verdict, str) and result.verdict
