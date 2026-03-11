"""
4-layer analytical pipeline for information gap analysis.

Layers:
  1. parse_intent   — pure keyword extraction, no LLM
  2. retrieve_evidence — graph neighbourhood query
  3. fuse_reasoning — LLM synthesis over collected evidence
  4. verify_inline  — constitutional reflection + confidence scoring
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from src.core.providers import ModelProvider
from src.data.graph_store import GraphStore
from src.detective.experience import ExperienceLibrary
from src.inference.reflection import inject_reflection_trigger
from src.inference.welfare_scoring import (
    compute_gap_urgency,
    infer_threatened_constructs,
)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {"a", "an", "the", "in", "of", "and", "or", "is", "was", "to", "for"}
)

# Relevance scores for evidence retrieval — named so the scoring policy is explicit
_RELEVANCE_DIRECT_MATCH: float = 1.0
_RELEVANCE_NEIGHBOUR: float = 0.5

_DEFAULT_CONSTITUTION_PRINCIPLE: str = "Epistemic honesty above analytical comfort."

# Denominator offset for confidence heuristic: confidence = steps / (steps + offset)
# Offset of 1 ensures confidence < 1.0 even for arbitrarily long chains,
# and > 0.0 for chains with at least 1 step.
_CONFIDENCE_DENOMINATOR_OFFSET: int = 1

# Gap hint template keys — checked against keywords to produce a contextual hint
_TEMPORAL_KEYWORDS: frozenset[str] = frozenset(
    {"year", "month", "period", "before", "after", "during", "since", "until"}
)
_NETWORK_KEYWORDS: frozenset[str] = frozenset(
    {"entity", "network", "connection", "linked", "related", "associated"}
)

_GAP_HINT_TEMPORAL: str = "Possible temporal gap — verify date ranges and silences."
_GAP_HINT_NETWORK: str = "Possible network gap — verify implied entity connections."
_GAP_HINT_DEFAULT: str = "Possible evidential gap — verify source coverage."

# Prompt templates
_FUSE_PROMPT_HEADER: str = "Analyze the following evidence and provide step-by-step reasoning:\n\n"
_FUSE_PROMPT_EVIDENCE_PREFIX: str = "Evidence:\n"
_FUSE_PROMPT_FOOTER: str = "\n\nProvide your analysis as numbered steps, one per line."

_VERIFY_PROMPT_TEMPLATE: str = (
    "Reasoning chain:\n{steps}\n\nConclusion: {conclusion}\n\n"
    "Provide a concise verdict on the validity of this analysis."
)


# ---------------------------------------------------------------------------
# Data types — all frozen dataclasses (immutable, hashable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticalIntent:
    """Parsed representation of the analyst's claim — extracted without LLM."""
    claim: str
    keywords: tuple[str, ...]
    gap_hint: str


@dataclass(frozen=True)
class Evidence:
    """A single piece of evidence retrieved from the knowledge graph."""
    node_id: str
    content: str
    relevance: float  # 1.0 for direct keyword matches; decayed for indirect


@dataclass(frozen=True)
class ReasoningChain:
    """LLM-synthesised reasoning over the collected evidence."""
    steps: tuple[str, ...]
    conclusion: str


@dataclass(frozen=True)
class AnalysisResult:
    """Full output of the 4-layer pipeline — immutable record for audit trails."""
    claim: str
    intent: AnalyticalIntent
    evidence: tuple[Evidence, ...]
    reasoning: ReasoningChain
    verdict: str
    confidence: float
    welfare_relevance: float = 0.0
    threatened_constructs: tuple[str, ...] = ()
    gaps: tuple = ()  # tuple[Gap, ...] — detected information gaps


# ---------------------------------------------------------------------------
# Layer 1 — parse_intent (pure, no LLM)
# ---------------------------------------------------------------------------


def parse_intent(claim: str) -> AnalyticalIntent:
    """
    Extract keywords and produce a gap hint without calling any external service.

    Stopword filtering is case-insensitive so "The" and "the" are both filtered;
    the original token casing is preserved in the output tuple for readability.
    """
    tokens = claim.split()
    keywords = tuple(t for t in tokens if t.lower() not in _STOPWORDS)

    lower_keywords = frozenset(k.lower() for k in keywords)
    gap_hint = _select_gap_hint(lower_keywords)

    return AnalyticalIntent(claim=claim, keywords=keywords, gap_hint=gap_hint)


def _select_gap_hint(lower_keywords: frozenset[str]) -> str:
    """
    Choose the most specific gap hint based on keyword semantics.

    Temporal > network > default matches in priority order —
    temporal signals are the most actionable gap type in investigative datasets.
    """
    if lower_keywords & _TEMPORAL_KEYWORDS:
        return _GAP_HINT_TEMPORAL
    if lower_keywords & _NETWORK_KEYWORDS:
        return _GAP_HINT_NETWORK
    return _GAP_HINT_DEFAULT


# ---------------------------------------------------------------------------
# Layer 2 — retrieve_evidence (graph query)
# ---------------------------------------------------------------------------


def retrieve_evidence(intent: AnalyticalIntent, graph: GraphStore) -> list[Evidence]:
    """
    Query the graph for nodes whose IDs contain any of the intent's keywords.

    The GraphStore Protocol (add_edge, get_edge, n_hop_paths) does not expose a
    generic node-listing API — that would bind all backends to an O(V) scan.
    InMemoryGraph exposes the raw networkx DiGraph via _graph for exactly this
    use case (local, in-process analysis runs).  For production backends, a
    more targeted query (e.g. full-text index on a persistent store) would
    replace this layer.

    Relevance is 1.0 for direct keyword-to-node_id matches and 0.5 for nodes
    discovered as neighbours of a matched node.  All results are deduplicated
    by node_id, keeping the highest relevance score seen.
    """
    # Collect node IDs from the backing store
    candidate_nodes = _extract_graph_nodes(graph)
    if not candidate_nodes:
        return []

    lower_keywords = frozenset(k.lower() for k in intent.keywords)

    # Score each node: direct keyword hit → 1.0, neighbour → 0.5
    scored: dict[str, float] = {}
    neighbour_map: dict[str, set[str]] = _build_neighbour_map(graph, candidate_nodes)

    for node in candidate_nodes:
        if node.lower() in lower_keywords or any(
            kw in node.lower() for kw in lower_keywords
        ):
            scored[node] = _RELEVANCE_DIRECT_MATCH

    # Expand to immediate neighbours of matched nodes
    for matched_node in list(scored.keys()):
        for neighbour in neighbour_map.get(matched_node, set()):
            if neighbour not in scored:
                scored[neighbour] = _RELEVANCE_NEIGHBOUR

    return [
        Evidence(
            node_id=node_id,
            content=f"{node_id} (graph node)",
            relevance=relevance,
        )
        for node_id, relevance in scored.items()
    ]


def _extract_graph_nodes(graph: GraphStore) -> list[str]:
    """
    Extract all node IDs via the GraphStore Protocol's nodes() method.

    Delegates to the Protocol rather than touching any private attribute —
    this keeps all backends (InMemoryGraph, KuzuGraph, etc.) on equal footing.
    """
    return graph.nodes()


def _build_neighbour_map(
    graph: GraphStore, nodes: list[str]
) -> dict[str, set[str]]:
    """
    Build a node → immediate successors mapping via the GraphStore Protocol.

    Uses successors() so all backends participate without private-attribute access.
    """
    return {node: set(graph.successors(node)) for node in nodes}


# ---------------------------------------------------------------------------
# Token budget — truncate evidence to fit vLLM context window
# ---------------------------------------------------------------------------

# Approximate chars-per-token for English text (conservative estimate).
# vLLM default max-model-len is 8192 tokens; we reserve headroom for the
# fuse prompt header/footer (~200 tokens) and the verify prompt (~2000 tokens).
_CHARS_PER_TOKEN: int = 4
_DEFAULT_EVIDENCE_TOKEN_BUDGET: int = 6000
_VERIFY_PROMPT_TOKEN_RESERVE: int = 2000


def _truncate_evidence(
    evidence: list[Evidence],
    token_budget: int = _DEFAULT_EVIDENCE_TOKEN_BUDGET,
) -> list[Evidence]:
    """Trim evidence list to fit within a token budget.

    Keeps evidence items in relevance order (highest first), dropping
    tail items when the cumulative character count exceeds the budget.
    This prevents silent truncation or 400 errors from vLLM when the
    fuse_reasoning prompt exceeds max-model-len.
    """
    char_budget = (token_budget - _VERIFY_PROMPT_TOKEN_RESERVE) * _CHARS_PER_TOKEN
    if char_budget <= 0:
        return evidence[:1] if evidence else []

    # Sort by relevance descending so we keep the best evidence
    sorted_ev = sorted(evidence, key=lambda e: e.relevance, reverse=True)

    kept: list[Evidence] = []
    total_chars = 0
    for ev in sorted_ev:
        item_chars = len(ev.node_id) + len(ev.content) + 30  # overhead per item
        if total_chars + item_chars > char_budget and kept:
            break
        kept.append(ev)
        total_chars += item_chars

    return kept


# ---------------------------------------------------------------------------
# Inline gap detection — lightweight scan after verification
# ---------------------------------------------------------------------------

_GAP_SCAN_PROMPT: str = (
    "You are an information gap analyst. Given the claim and evidence below, "
    "identify 0-3 information gaps — things that are absent, suppressed, or "
    "undocumented that should be present.\n\n"
    "For each gap, reply with one line in this exact format:\n"
    "  GAP|type|description\n"
    "Where type is one of: temporal, evidential, contradiction, normative, doctrinal\n"
    "If no gaps are found, reply with: NONE\n\n"
    "Claim: {claim}\n\n"
    "Evidence:\n{evidence}\n"
)


def _detect_gaps_inline(
    claim: str,
    evidence: list[Evidence],
    provider: ModelProvider,
) -> list:
    """Run a lightweight LLM-assisted gap scan on the claim and evidence.

    Returns a list of Gap objects. Fails gracefully (returns []) on parse
    errors or provider failures — gap detection is advisory, not blocking.
    """
    from src.core.types import Gap, GapType

    evidence_text = "\n".join(
        f"- [{e.node_id}] {e.content}" for e in evidence[:10]
    ) if evidence else "(no evidence)"

    prompt = _GAP_SCAN_PROMPT.format(claim=claim[:1000], evidence=evidence_text[:3000])

    try:
        response = provider.complete(prompt)
    except Exception:
        return []

    if "NONE" in response.strip().upper()[:10]:
        return []

    type_map = {
        "temporal": GapType.TEMPORAL,
        "evidential": GapType.EVIDENTIAL,
        "contradiction": GapType.CONTRADICTION,
        "normative": GapType.NORMATIVE,
        "doctrinal": GapType.DOCTRINAL,
    }

    gaps: list[Gap] = []
    for line in response.splitlines():
        line = line.strip()
        if not line.startswith("GAP|"):
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        gap_type_str = parts[1].strip().lower()
        description = parts[2].strip()
        gap_type = type_map.get(gap_type_str, GapType.EVIDENTIAL)
        if description:
            gaps.append(Gap(
                type=gap_type,
                description=description,
                confidence=0.5,  # LLM-detected, moderate confidence
                location=claim[:100],
            ))

    return gaps[:3]  # cap at 3


# Confidence threshold for running the gap scan — below this the analysis
# is too uncertain to meaningfully detect gaps in the evidence.
_GAP_SCAN_CONFIDENCE_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Layer 3 — fuse_reasoning (single LLM call)
# ---------------------------------------------------------------------------


def fuse_reasoning(evidence: list[Evidence], provider: ModelProvider) -> ReasoningChain:
    """
    Synthesise evidence into a step-by-step reasoning chain via one LLM completion.

    The prompt lists each evidence item, then asks for numbered step-by-step analysis.
    Response is split on newlines; the last non-empty line becomes the conclusion.
    """
    evidence_block = "\n".join(
        f"- [{e.node_id}] {e.content} (relevance: {e.relevance:.2f})"
        for e in evidence
    ) if evidence else "(no direct evidence found)"

    prompt = (
        _FUSE_PROMPT_HEADER
        + _FUSE_PROMPT_EVIDENCE_PREFIX
        + evidence_block
        + _FUSE_PROMPT_FOOTER
    )

    response = provider.complete(prompt)
    return _parse_reasoning_response(response)


def _parse_reasoning_response(response: str) -> ReasoningChain:
    """
    Split LLM response into individual steps and extract the conclusion.

    The last non-empty line is treated as the conclusion — this matches the
    natural structure of LLM outputs where the final line summarises the chain.
    """
    lines = [line.strip() for line in response.splitlines()]
    non_empty = [line for line in lines if line]

    if not non_empty:
        return ReasoningChain(steps=(), conclusion=response.strip())

    conclusion = non_empty[-1]
    # Exclude the conclusion from steps — the verify prompt renders them separately
    steps = tuple(non_empty[:-1]) if len(non_empty) > 1 else ()
    return ReasoningChain(steps=steps, conclusion=conclusion)


# ---------------------------------------------------------------------------
# Layer 4 — verify_inline (constitutional reflection + confidence)
# ---------------------------------------------------------------------------


def verify_inline(
    intent: AnalyticalIntent,
    chain: ReasoningChain,
    provider: ModelProvider,
    constitution: object,
    evidence: tuple[Evidence, ...] = (),
) -> AnalysisResult:
    """
    Apply a constitutional reflection trigger to the reasoning chain, then score confidence.

    The reflection trigger (from src.inference.reflection) injects a mid-prompt pause
    that nudges the model to reconsider its reasoning against the constitutional principle
    before producing a final verdict.

    Confidence heuristic: steps / (steps + _CONFIDENCE_DENOMINATOR_OFFSET).
    This is intentionally simple — a single step gives 0.5, infinite steps approach 1.0,
    and zero steps give 0.0.  A richer calibration would use calibration datasets.
    """
    steps_text = "\n".join(chain.steps) if chain.steps else "(no steps)"
    base_prompt = _VERIFY_PROMPT_TEMPLATE.format(
        steps=steps_text,
        conclusion=chain.conclusion,
    )

    # Derive the reflection principle from the constitution object when possible.
    # Duck-typed against `.critique()` so both ConstitutionClient (which loads a
    # full text constitution) and lightweight mock objects work without coupling
    # this layer to a concrete type.
    if hasattr(constitution, "critique"):
        principle = constitution.critique(chain.conclusion)  # type: ignore[union-attr]
    else:
        principle = _DEFAULT_CONSTITUTION_PRINCIPLE

    triggered_prompt = inject_reflection_trigger(base_prompt, principle)
    verdict = provider.complete(triggered_prompt)

    n_steps = len(chain.steps)
    confidence = n_steps / (n_steps + _CONFIDENCE_DENOMINATOR_OFFSET)

    return AnalysisResult(
        claim=intent.claim,
        intent=intent,
        evidence=evidence,
        reasoning=chain,
        verdict=verdict,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Gap welfare scoring utility
# ---------------------------------------------------------------------------


def score_gaps_welfare(gaps: list, phi_metrics: dict[str, float]) -> list:
    """
    Score welfare impact for a list of Gap objects and sort by urgency.

    Args:
        gaps: List of Gap objects to score
        phi_metrics: Current Φ construct levels

    Returns:
        Gaps sorted by welfare urgency (descending)

    Note:
        This utility is ready for integration when gap detection
        is wired into the analyze() pipeline. Current pipeline returns
        AnalysisResult without gaps field.
    """
    from src.core.types import Gap

    scored_gaps = []
    for gap in gaps:
        # Infer constructs if not already set
        if not gap.threatened_constructs:
            constructs = infer_threatened_constructs(gap.description)
            gap = replace(gap, threatened_constructs=constructs)

        # Compute welfare impact
        welfare_impact = compute_gap_urgency(gap, phi_metrics)
        gap = replace(gap, welfare_impact=welfare_impact)

        scored_gaps.append(gap)

    # Sort by welfare urgency (descending)
    return sorted(scored_gaps, key=lambda g: g.welfare_impact, reverse=True)


# ---------------------------------------------------------------------------
# Top-level — wire all 4 layers
# ---------------------------------------------------------------------------


def analyze(
    claim: str,
    provider: ModelProvider,
    graph: GraphStore,
    library: ExperienceLibrary,
    constitution: object,
    phi_metrics: dict[str, float] | None = None,
) -> AnalysisResult:
    """
    Run the full 4-layer analytical pipeline.

    When phi_metrics is provided, computes welfare relevance of the claim
    by identifying threatened constructs and scoring their Phi gradients.

    library is accepted per spec for future enrichment of Layer 2 retrieval
    (e.g. retrieving similar past hypotheses to weight evidence).
    It is not consumed in this iteration — the parameter is explicit so callers
    are not surprised when it is used in a future version.
    """
    intent = parse_intent(claim)
    evidence = retrieve_evidence(intent, graph)
    evidence = _truncate_evidence(evidence)
    chain = fuse_reasoning(evidence, provider)
    result = verify_inline(intent, chain, provider, constitution, evidence=tuple(evidence))

    # Gap detection — only when analysis has enough confidence to be meaningful
    if result.confidence >= _GAP_SCAN_CONFIDENCE_THRESHOLD:
        detected_gaps = _detect_gaps_inline(claim, evidence, provider)
        if detected_gaps and phi_metrics is not None:
            detected_gaps = score_gaps_welfare(detected_gaps, phi_metrics)
        if detected_gaps:
            result = replace(result, gaps=tuple(detected_gaps))

    # Welfare scoring when phi_metrics provided
    if phi_metrics is not None:
        from src.inference.welfare_scoring import (
            infer_threatened_constructs as _infer,
            score_hypothesis_welfare as _score_welfare,
        )
        from src.detective.hypothesis import Hypothesis

        constructs = _infer(claim)
        if constructs:
            # Create a temporary hypothesis to use the scoring infrastructure
            temp_h = Hypothesis.create(text=claim, confidence=result.confidence)
            temp_h = replace(temp_h, threatened_constructs=constructs)
            welfare_rel = _score_welfare(temp_h, phi_metrics)
        else:
            welfare_rel = 0.0

        result = replace(
            result,
            welfare_relevance=welfare_rel,
            threatened_constructs=constructs,
        )

    return result
