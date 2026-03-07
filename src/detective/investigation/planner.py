"""LLM-assisted lead generation and hypothesis seeding for investigations."""

from __future__ import annotations

import logging

from src.core.providers import ModelProvider
from src.core.scoring import parse_score
from src.data.graph_store import GraphStore
from src.detective.hypothesis import Hypothesis
from src.detective.investigation.types import Lead

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates — named constants per project convention
# ---------------------------------------------------------------------------

_SEED_HYPOTHESES_PROMPT = (
    "You are an investigative analyst. Given the topic below, generate 3 "
    "distinct hypotheses that could be investigated using public documents "
    "(FOIA releases, court records, government archives).\n\n"
    "Topic: {topic}\n\n"
    "For each hypothesis, provide:\n"
    "hypothesis: <text>\n"
    "confidence: <0.0-1.0>\n\n"
    "Respond with exactly 3 entries, nothing else."
)

_GRAPH_EVENT_PROMPT = (
    "A new entity relationship was discovered in the knowledge graph:\n"
    "{event}\n\n"
    "Generate 2 hypotheses that this relationship might indicate. "
    "For each:\n"
    "hypothesis: <text>\n"
    "confidence: <0.0-1.0>"
)

_LEADS_PROMPT = (
    "You are an investigative analyst planning the next step of an investigation.\n\n"
    "Current hypotheses:\n{hypotheses}\n\n"
    "Available sources: {sources}\n\n"
    "Graph context (known entities): {entities}\n\n"
    "Generate 3-5 search queries to gather evidence. For each:\n"
    "query: <search text>\n"
    "source: <source_id>\n"
    "priority: <0.0-1.0>\n\n"
    "Prioritise queries that test the weakest hypotheses."
)

_ALTERNATIVES_PROMPT = (
    "An investigation hypothesis has low confidence after analysis:\n\n"
    "Hypothesis: {hypothesis}\n"
    "Current confidence: {confidence}\n"
    "Analysis summary: {analysis}\n\n"
    "Generate 2 alternative hypotheses that could explain the same evidence "
    "differently. For each:\n"
    "hypothesis: <text>\n"
    "confidence: <0.0-1.0>"
)

# Default confidence for hypotheses when parsing fails
_DEFAULT_HYPOTHESIS_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_hypotheses(response: str) -> list[Hypothesis]:
    """Parse LLM response into Hypothesis objects."""
    hypotheses: list[Hypothesis] = []
    lines = response.strip().splitlines()

    current_text: str | None = None
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("hypothesis:"):
            current_text = stripped.split(":", 1)[1].strip()
        elif lower.startswith("confidence:") and current_text:
            conf = parse_score(stripped, default=_DEFAULT_HYPOTHESIS_CONFIDENCE)
            hypotheses.append(Hypothesis.create(text=current_text, confidence=conf))
            current_text = None

    # Handle trailing hypothesis without confidence
    if current_text:
        hypotheses.append(
            Hypothesis.create(text=current_text, confidence=_DEFAULT_HYPOTHESIS_CONFIDENCE)
        )

    return hypotheses


def _parse_leads(
    response: str,
    available_sources: tuple[str, ...],
    step: int,
    parent_hypothesis_id: str | None = None,
) -> list[Lead]:
    """Parse LLM response into Lead objects."""
    leads: list[Lead] = []
    lines = response.strip().splitlines()

    current_query: str | None = None
    current_source: str | None = None

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("query:"):
            current_query = stripped.split(":", 1)[1].strip()
        elif lower.startswith("source:"):
            current_source = stripped.split(":", 1)[1].strip()
        elif lower.startswith("priority:") and current_query:
            priority = parse_score(stripped, default=0.5)
            source_id = current_source if current_source in available_sources else available_sources[0]
            leads.append(
                Lead.create(
                    query=current_query,
                    source_id=source_id,
                    priority=priority,
                    parent_hypothesis_id=parent_hypothesis_id,
                    generation_step=step,
                )
            )
            current_query = None
            current_source = None

    return leads


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_seed_hypotheses(
    topic: str,
    provider: ModelProvider,
) -> list[Hypothesis]:
    """Generate initial hypotheses from a topic string (for 'topic' trigger mode)."""
    prompt = _SEED_HYPOTHESES_PROMPT.format(topic=topic)
    response = provider.complete(prompt)
    hypotheses = _parse_hypotheses(response)

    if not hypotheses:
        _logger.warning("LLM returned no parseable hypotheses; creating fallback")
        hypotheses = [Hypothesis.create(text=topic, confidence=_DEFAULT_HYPOTHESIS_CONFIDENCE)]

    return hypotheses


def hypotheses_from_graph_event(
    event: str,
    provider: ModelProvider,
) -> list[Hypothesis]:
    """Generate hypotheses from a graph event (for 'reactive' trigger mode)."""
    prompt = _GRAPH_EVENT_PROMPT.format(event=event)
    response = provider.complete(prompt)
    hypotheses = _parse_hypotheses(response)

    if not hypotheses:
        hypotheses = [Hypothesis.create(text=event, confidence=_DEFAULT_HYPOTHESIS_CONFIDENCE)]

    return hypotheses


def generate_leads(
    hypotheses: list[Hypothesis],
    graph: GraphStore,
    available_sources: tuple[str, ...],
    provider: ModelProvider,
    step: int,
) -> list[Lead]:
    """Generate prioritised leads from current hypotheses and graph context."""
    if not hypotheses or not available_sources:
        return []

    hyp_text = "\n".join(
        f"- [{h.confidence:.2f}] {h.text}" for h in hypotheses
    )
    entities = ", ".join(graph.nodes()[:50]) if graph.nodes() else "(empty graph)"
    sources_str = ", ".join(available_sources)

    prompt = _LEADS_PROMPT.format(
        hypotheses=hyp_text,
        sources=sources_str,
        entities=entities,
    )
    response = provider.complete(prompt)

    # Use weakest hypothesis as parent for lead lineage
    weakest = min(hypotheses, key=lambda h: h.confidence)
    leads = _parse_leads(response, available_sources, step, parent_hypothesis_id=weakest.id)

    if not leads:
        _logger.warning("LLM returned no parseable leads; creating fallback from hypotheses")
        for h in hypotheses[:3]:
            leads.append(
                Lead.create(
                    query=h.text,
                    source_id=available_sources[0],
                    priority=1.0 - h.confidence,
                    parent_hypothesis_id=h.id,
                    generation_step=step,
                )
            )

    # Sort by priority descending
    leads.sort(key=lambda lead: lead.priority, reverse=True)
    return leads


def spawn_alternatives(
    hypothesis: Hypothesis,
    analysis_summary: str,
    provider: ModelProvider,
) -> list[Hypothesis]:
    """Generate competing hypotheses for a low-confidence branch (breadth strategy)."""
    prompt = _ALTERNATIVES_PROMPT.format(
        hypothesis=hypothesis.text,
        confidence=f"{hypothesis.confidence:.2f}",
        analysis=analysis_summary,
    )
    response = provider.complete(prompt)
    alternatives = _parse_hypotheses(response)

    if not alternatives:
        _logger.warning("No alternatives parsed; returning empty list")

    return alternatives
