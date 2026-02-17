from __future__ import annotations

import networkx as nx

from src.core.types import KnowledgeEdge, RelationType

KnowledgeGraph = nx.DiGraph

# Per-hop confidence decay by relation type.
# Causal chains decay faster than sequential chains (correlation != causation).
_HOP_DECAY: dict[RelationType, float] = {
    RelationType.CAUSAL: 0.7,
    RelationType.CONDITIONAL: 0.75,
    RelationType.INSTANTIATIVE: 0.85,
    RelationType.SEQUENTIAL: 0.9,
}

_EDGE_DATA_KEY: str = "edge"  # key under which KnowledgeEdge is stored in networkx edge attrs


def make_graph() -> KnowledgeGraph:
    """Start with an empty directed knowledge graph."""
    return nx.DiGraph()


def add_edge(
    graph: KnowledgeGraph,
    source: str,
    target: str,
    relation: RelationType,
    confidence: float,
) -> KnowledgeGraph:
    """
    Return a new graph with the typed edge added.
    Copy-on-write preserves the functional contract; the original graph is unchanged.
    """
    edge = KnowledgeEdge(
        source=source,
        target=target,
        relation=relation,
        confidence=confidence,
    )
    new_graph = graph.copy()
    new_graph.add_edge(source, target, **{_EDGE_DATA_KEY: edge})
    return new_graph


def get_edge(graph: KnowledgeGraph, source: str, target: str) -> KnowledgeEdge | None:
    """Return the edge between source and target, or None if it does not exist."""
    if not graph.has_edge(source, target):
        return None
    return graph[source][target][_EDGE_DATA_KEY]


def n_hop_paths(
    graph: KnowledgeGraph,
    source: str,
    target: str,
    max_hops: int = 3,
) -> list[dict]:
    """
    Find all simple paths from source to target within max_hops.
    Each result carries the path, aggregate confidence, and the weakest relation type
    — path confidence decays per hop according to the relation-specific decay factor.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops!r}")

    # nx.all_simple_paths returns a lazy generator; NodeNotFound may be raised
    # on the first iteration (newer networkx), not at call time.
    if source not in graph or target not in graph:
        return []

    results = []
    try:
        all_paths = nx.all_simple_paths(graph, source, target, cutoff=max_hops)
        for path in all_paths:
            edges = [get_edge(graph, path[i], path[i + 1]) for i in range(len(path) - 1)]
            if any(e is None for e in edges):
                continue
            # Compute aggregate confidence: product of each edge's confidence * hop decay
            aggregate = 1.0
            for hop_idx, edge in enumerate(edges):
                decay = _HOP_DECAY.get(edge.relation, 0.8)
                aggregate *= edge.confidence * (decay ** (hop_idx + 1))
            results.append({
                "path": path,
                "confidence": aggregate,
                "hops": len(path) - 1,
            })
    except nx.NodeNotFound:
        return []

    return sorted(results, key=lambda r: r["confidence"], reverse=True)
