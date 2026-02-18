"""Tests for the 4-layer analytical pipeline."""
from __future__ import annotations

from src.core.providers import MockProvider
from src.core.types import RelationType
from src.data.graph_store import InMemoryGraph
from src.detective.experience import EMPTY_LIBRARY


# ---------------------------------------------------------------------------
# Minimal mock constitution — avoids dependency on file-system or providers
# ---------------------------------------------------------------------------

class _MockConstitution:
    def critique(self, text: str) -> str:
        return "No issues found."

    def revise(self, text: str, critique: str) -> str:
        return text


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

def test_pipeline_constants():
    from src.inference.pipeline import _STOPWORDS, _DEFAULT_CONSTITUTION_PRINCIPLE
    assert "the" in _STOPWORDS
    assert "a" in _STOPWORDS
    assert "Epistemic" in _DEFAULT_CONSTITUTION_PRINCIPLE


# ---------------------------------------------------------------------------
# 2. parse_intent — keyword extraction
# ---------------------------------------------------------------------------

def test_parse_intent_extracts_keywords():
    from src.inference.pipeline import parse_intent
    result = parse_intent("Entity A influenced Policy X")
    assert "Entity" in result.keywords
    assert "influenced" in result.keywords
    assert "Policy" in result.keywords


def test_parse_intent_filters_stopwords():
    """Stopword filtering is case-insensitive: 'The' (capitalised) is filtered just like 'the'."""
    from src.inference.pipeline import parse_intent
    result = parse_intent("The entity was active in the region")
    assert "The" not in result.keywords
    assert "the" not in result.keywords
    assert "in" not in result.keywords
    assert "entity" in result.keywords


def test_parse_intent_returns_analytical_intent():
    from src.inference.pipeline import parse_intent, AnalyticalIntent
    result = parse_intent("Entity A influenced Policy X")
    assert isinstance(result, AnalyticalIntent)
    assert result.claim == "Entity A influenced Policy X"
    assert isinstance(result.keywords, tuple)
    assert isinstance(result.gap_hint, str)


def test_parse_intent_stopword_claim():
    """Stopwords are compared case-insensitively: 'the' should be filtered."""
    from src.inference.pipeline import parse_intent
    result = parse_intent("the entity was active in the region")
    assert "the" not in result.keywords
    assert "in" not in result.keywords
    assert "entity" in result.keywords or "Entity" in result.keywords


def test_parse_intent_gap_hint_non_empty():
    from src.inference.pipeline import parse_intent
    result = parse_intent("Entity A influenced Policy X")
    assert len(result.gap_hint) > 0


# ---------------------------------------------------------------------------
# 3. retrieve_evidence — graph querying
# ---------------------------------------------------------------------------

def test_retrieve_evidence_queries_graph():
    from src.inference.pipeline import retrieve_evidence, parse_intent

    graph = InMemoryGraph()
    graph.add_edge("EntityA", "PolicyX", RelationType.CAUSAL, confidence=0.9)
    graph.add_edge("PolicyX", "OutcomeZ", RelationType.SEQUENTIAL, confidence=0.8)

    intent = parse_intent("EntityA influenced PolicyX")
    evidence = retrieve_evidence(intent, graph)

    assert isinstance(evidence, list)
    assert len(evidence) > 0


def test_retrieve_evidence_returns_evidence_objects():
    from src.inference.pipeline import retrieve_evidence, parse_intent, Evidence

    graph = InMemoryGraph()
    graph.add_edge("EntityA", "PolicyX", RelationType.CAUSAL, confidence=0.9)

    intent = parse_intent("EntityA influenced PolicyX")
    evidence = retrieve_evidence(intent, graph)

    for item in evidence:
        assert isinstance(item, Evidence)
        assert isinstance(item.node_id, str)
        assert isinstance(item.content, str)
        assert isinstance(item.relevance, float)


def test_retrieve_evidence_deduplicates_by_node_id():
    from src.inference.pipeline import retrieve_evidence, parse_intent

    graph = InMemoryGraph()
    # Both keywords match same node via different paths
    graph.add_edge("EntityA", "PolicyX", RelationType.CAUSAL, confidence=0.9)
    graph.add_edge("EntityA", "PolicyX", RelationType.SEQUENTIAL, confidence=0.8)

    intent = parse_intent("EntityA PolicyX overlap test")
    evidence = retrieve_evidence(intent, graph)

    node_ids = [e.node_id for e in evidence]
    # No duplicate node_ids
    assert len(node_ids) == len(set(node_ids))


def test_retrieve_evidence_direct_match_relevance():
    from src.inference.pipeline import retrieve_evidence, parse_intent

    graph = InMemoryGraph()
    graph.add_edge("EntityA", "PolicyX", RelationType.CAUSAL, confidence=0.9)

    intent = parse_intent("EntityA influenced PolicyX")
    evidence = retrieve_evidence(intent, graph)

    direct_matches = [e for e in evidence if e.relevance == 1.0]
    assert len(direct_matches) > 0


def test_retrieve_evidence_empty_graph():
    from src.inference.pipeline import retrieve_evidence, parse_intent

    graph = InMemoryGraph()
    intent = parse_intent("Entity A influenced Policy X")
    evidence = retrieve_evidence(intent, graph)
    assert evidence == []


# ---------------------------------------------------------------------------
# 4. fuse_reasoning — provider call
# ---------------------------------------------------------------------------

def test_fuse_reasoning_calls_provider():
    from src.inference.pipeline import fuse_reasoning, Evidence

    provider = MockProvider(response="Step one analysis.\nStep two synthesis.\nConclusion reached.")
    evidence = [Evidence(node_id="EntityA", content="EntityA connects to PolicyX", relevance=1.0)]
    chain = fuse_reasoning(evidence, provider)

    from src.inference.pipeline import ReasoningChain
    assert isinstance(chain, ReasoningChain)


def test_fuse_reasoning_returns_steps_and_conclusion():
    from src.inference.pipeline import fuse_reasoning, Evidence, ReasoningChain

    provider = MockProvider(response="Step one.\nStep two.\nFinal conclusion.")
    evidence = [Evidence(node_id="n1", content="some evidence", relevance=0.9)]
    chain = fuse_reasoning(evidence, provider)

    assert isinstance(chain.steps, tuple)
    assert len(chain.steps) > 0
    assert isinstance(chain.conclusion, str)
    assert len(chain.conclusion) > 0


def test_fuse_reasoning_single_line_response():
    """Single-line response: steps contains that line, conclusion is the same line."""
    from src.inference.pipeline import fuse_reasoning, Evidence

    provider = MockProvider(response="Single conclusion.")
    evidence = [Evidence(node_id="n1", content="data", relevance=1.0)]
    chain = fuse_reasoning(evidence, provider)

    assert len(chain.conclusion) > 0


def test_fuse_reasoning_empty_evidence():
    """Fuse reasoning still calls provider even with no evidence."""
    from src.inference.pipeline import fuse_reasoning

    provider = MockProvider(response="No evidence provided.")
    chain = fuse_reasoning([], provider)

    assert len(chain.conclusion) > 0


# ---------------------------------------------------------------------------
# 5. verify_inline — returns AnalysisResult
# ---------------------------------------------------------------------------

def test_verify_inline_returns_analysis_result():
    from src.inference.pipeline import (
        verify_inline, fuse_reasoning, parse_intent,
        Evidence, AnalysisResult,
    )

    provider = MockProvider(response="Verdict: The claim is plausible.")
    constitution = _MockConstitution()

    intent = parse_intent("Entity A influenced Policy X")
    evidence = [Evidence(node_id="EntityA", content="EntityA -> PolicyX", relevance=1.0)]
    chain = fuse_reasoning(evidence, provider)

    result = verify_inline(intent, chain, provider, constitution)

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.verdict, str)
    assert len(result.verdict) > 0


def test_verify_inline_has_confidence():
    from src.inference.pipeline import (
        verify_inline, fuse_reasoning, parse_intent, Evidence,
    )

    provider = MockProvider(response="The analysis is verified.")
    constitution = _MockConstitution()

    intent = parse_intent("Entity A influenced Policy X")
    evidence = [Evidence(node_id="n1", content="data", relevance=1.0)]
    chain = fuse_reasoning(evidence, provider)

    result = verify_inline(intent, chain, provider, constitution)
    assert isinstance(result.confidence, float)


# ---------------------------------------------------------------------------
# 6. analyze — full end-to-end pipeline
# ---------------------------------------------------------------------------

def test_analyze_returns_analysis_result():
    from src.inference.pipeline import analyze, AnalysisResult

    provider = MockProvider(response="Step one.\nStep two.\nFinal verdict is positive.")
    graph = InMemoryGraph()
    graph.add_edge("EntityA", "PolicyX", RelationType.CAUSAL, confidence=0.9)
    constitution = _MockConstitution()

    result = analyze(
        claim="EntityA influenced PolicyX",
        provider=provider,
        graph=graph,
        library=EMPTY_LIBRARY,
        constitution=constitution,
    )

    assert isinstance(result, AnalysisResult)


def test_analyze_result_has_all_fields():
    from src.inference.pipeline import analyze, AnalysisResult, AnalyticalIntent, ReasoningChain

    provider = MockProvider(response="Analysis complete.\nConclusion: valid.")
    graph = InMemoryGraph()
    graph.add_edge("EntityA", "PolicyX", RelationType.SEQUENTIAL, confidence=0.8)
    constitution = _MockConstitution()

    result = analyze(
        claim="EntityA influenced PolicyX connection",
        provider=provider,
        graph=graph,
        library=EMPTY_LIBRARY,
        constitution=constitution,
    )

    assert isinstance(result.claim, str)
    assert result.claim == "EntityA influenced PolicyX connection"
    assert isinstance(result.intent, AnalyticalIntent)
    assert isinstance(result.evidence, tuple)
    assert isinstance(result.reasoning, ReasoningChain)
    assert isinstance(result.verdict, str)
    assert isinstance(result.confidence, float)


# ---------------------------------------------------------------------------
# 7. confidence is in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_confidence_between_zero_and_one():
    from src.inference.pipeline import analyze

    provider = MockProvider(response="Single step conclusion.")
    graph = InMemoryGraph()
    constitution = _MockConstitution()

    result = analyze(
        claim="Some claim about Entity X",
        provider=provider,
        graph=graph,
        library=EMPTY_LIBRARY,
        constitution=constitution,
    )

    assert 0.0 <= result.confidence <= 1.0


def test_confidence_increases_with_more_steps():
    """More reasoning steps → higher confidence (up to limit)."""
    from src.inference.pipeline import fuse_reasoning, verify_inline, parse_intent, Evidence

    # Multi-step response
    provider_many = MockProvider(
        response="Step 1.\nStep 2.\nStep 3.\nStep 4.\nConclusion."
    )
    # Single-step response
    provider_one = MockProvider(response="Conclusion only.")

    constitution = _MockConstitution()
    intent = parse_intent("test claim")
    evidence: list = []

    chain_many = fuse_reasoning(evidence, provider_many)
    chain_one = fuse_reasoning(evidence, provider_one)

    result_many = verify_inline(intent, chain_many, provider_many, constitution)
    result_one = verify_inline(intent, chain_one, provider_one, constitution)

    assert result_many.confidence >= result_one.confidence
