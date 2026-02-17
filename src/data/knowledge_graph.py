from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PathResult:
    """A traversal path through the knowledge graph with aggregate confidence."""
    path: tuple[str, ...]
    confidence: float
    hops: int


def make_graph() -> KnowledgeGraph:
    """New graph per analysis run — no shared state between independent analyses."""
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
    """None return is a valid finding — absence of a direct edge between entities is investigatively meaningful."""
    if not graph.has_edge(source, target):
        return None
    return graph[source][target][_EDGE_DATA_KEY]


def n_hop_paths(
    graph: KnowledgeGraph,
    source: str,
    target: str,
    max_hops: int = 3,
) -> list[PathResult]:
    """
    Find all simple paths from source to target within max_hops.
    Each result carries the path, aggregate confidence, and hop count
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
            # Compute aggregate confidence: product of each edge's confidence * per-hop decay
            aggregate = 1.0
            for edge in edges:
                aggregate *= edge.confidence * _HOP_DECAY[edge.relation]
            results.append(PathResult(
                path=tuple(path),
                confidence=aggregate,
                hops=len(path) - 1,
            ))
    except nx.NodeNotFound:
        return []

    return sorted(results, key=lambda r: r.confidence, reverse=True)
