"""
Legal domain gap detection.

Detects gaps between how law is written (STATUTE, REGULATION, CASE_LAW)
and how it is applied (ENFORCEMENT_PRACTICE, COMMUNITY_EXPERIENCE) for
entities in the knowledge graph. These gaps are primary detection targets
per ADR-007: the absence of enforcement data for a statute, or the absence
of community experience data alongside regulatory data, is investigatively
meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.core.types import GapType, KnowledgeEdge, LegalDomain
from src.data.graph_store import GraphStore


# Legal domains that represent law-as-written vs law-as-applied
_WRITTEN_DOMAINS: frozenset[LegalDomain] = frozenset({
    LegalDomain.STATUTE,
    LegalDomain.REGULATION,
    LegalDomain.CASE_LAW,
    LegalDomain.TREATY,
    LegalDomain.TERRITORIAL,
})

_APPLIED_DOMAINS: frozenset[LegalDomain] = frozenset({
    LegalDomain.ENFORCEMENT_PRACTICE,
    LegalDomain.COMMUNITY_EXPERIENCE,
})


@dataclass(frozen=True)
class LegalGap:
    """A detected gap between legal domains for a given entity."""
    topic_entity: str
    written_edges: tuple[KnowledgeEdge, ...]
    applied_edges: tuple[KnowledgeEdge, ...]
    gap_type: GapType
    description: str


def _collect_edges_by_domain(
    graph: GraphStore,
    entity: str,
) -> tuple[list[KnowledgeEdge], list[KnowledgeEdge]]:
    """Collect edges touching entity, partitioned into written and applied domains."""
    written: list[KnowledgeEdge] = []
    applied: list[KnowledgeEdge] = []

    for successor in graph.successors(entity):
        edge = graph.get_edge(entity, successor)
        if edge is None or edge.legal_domain is None:
            continue
        if edge.legal_domain in _WRITTEN_DOMAINS:
            written.append(edge)
        elif edge.legal_domain in _APPLIED_DOMAINS:
            applied.append(edge)

    return written, applied


def detect_legal_domain_gaps(
    graph: GraphStore,
    entity: str,
) -> list[LegalGap]:
    """
    Find legal domain gaps for an entity in the knowledge graph.

    A gap exists when an entity appears in law-as-written contexts (STATUTE,
    REGULATION, CASE_LAW) but not in law-as-applied contexts (ENFORCEMENT_PRACTICE,
    COMMUNITY_EXPERIENCE), or vice versa.

    Returns:
        List of LegalGap findings. Empty if no legal domain edges exist or
        if both written and applied contexts are present.
    """
    written, applied = _collect_edges_by_domain(graph, entity)

    gaps: list[LegalGap] = []

    if written and not applied:
        written_domains = {e.legal_domain for e in written if e.legal_domain}
        gaps.append(LegalGap(
            topic_entity=entity,
            written_edges=tuple(written),
            applied_edges=(),
            gap_type=GapType.DOCTRINAL,
            description=(
                f"Entity '{entity}' appears in {', '.join(d.value for d in written_domains)} "
                f"contexts but has no enforcement or community experience data. "
                f"The law assumes its own enforcement, which may not have occurred."
            ),
        ))

    if applied and not written:
        applied_domains = {e.legal_domain for e in applied if e.legal_domain}
        gaps.append(LegalGap(
            topic_entity=entity,
            written_edges=(),
            applied_edges=tuple(applied),
            gap_type=GapType.NORMATIVE,
            description=(
                f"Entity '{entity}' appears in {', '.join(d.value for d in applied_domains)} "
                f"contexts but has no corresponding statutory or regulatory basis documented. "
                f"Enforcement without legal grounding is investigatively significant."
            ),
        ))

    return gaps
