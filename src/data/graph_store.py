from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import networkx as nx

from src.core.types import KnowledgeEdge, RelationType
from src.data.knowledge_graph import PathResult, _EDGE_DATA_KEY, _HOP_DECAY, n_hop_paths

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

_GRAPH_BACKEND_ENV: str = "DETECTIVE_GRAPH_BACKEND"
_BACKEND_MEMORY: str = "memory"


# ---------------------------------------------------------------------------
# Protocol — structural interface for all graph backends
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphStore(Protocol):
    """
    Structural contract for graph persistence backends.

    Mutation semantics are intentional: a remote persistent store (KuzuGraph,
    Neo4j, etc.) cannot offer copy-on-write, so the Protocol matches the lowest
    common denominator of all realistic backends.
    """

    def add_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        confidence: float,
    ) -> None: ...

    def get_edge(self, source: str, target: str) -> KnowledgeEdge | None: ...

    def n_hop_paths(
        self, source: str, target: str, max_hops: int = 3
    ) -> list[PathResult]: ...


# ---------------------------------------------------------------------------
# InMemoryGraph — Protocol-conforming, mutable, networkx-backed
# ---------------------------------------------------------------------------


@dataclass
class InMemoryGraph:
    """
    In-process graph store backed by networkx DiGraph.

    Not frozen — mutation is required by GraphStore Protocol semantics and
    by the imperative nature of incremental edge ingestion during analysis runs.
    The leading underscore on _graph signals that callers must go through the
    Protocol methods rather than touching the raw graph directly.
    """

    _graph: nx.DiGraph = field(default_factory=nx.DiGraph, repr=False)

    def add_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        confidence: float,
    ) -> None:
        edge = KnowledgeEdge(
            source=source,
            target=target,
            relation=relation,
            confidence=confidence,
        )
        self._graph.add_edge(source, target, **{_EDGE_DATA_KEY: edge})

    def get_edge(self, source: str, target: str) -> KnowledgeEdge | None:
        """None signals investigatively meaningful absence, not an error condition."""
        if not self._graph.has_edge(source, target):
            return None
        return self._graph[source][target][_EDGE_DATA_KEY]

    def n_hop_paths(
        self, source: str, target: str, max_hops: int = 3
    ) -> list[PathResult]:
        """
        Delegate to the functional n_hop_paths implementation so the decay
        algorithm lives in exactly one place — knowledge_graph.py.
        """
        return n_hop_paths(self._graph, source, target, max_hops)


# ---------------------------------------------------------------------------
# Factory — reads DETECTIVE_GRAPH_BACKEND to select backend
# ---------------------------------------------------------------------------


def graph_store_from_env() -> GraphStore:
    """
    Instantiate the graph backend specified by the environment, defaulting to
    InMemoryGraph for zero-config local and CI runs.

    KuzuGraph (persistent, embedded) is Task 30 — setting the env var to "kuzu"
    now raises explicitly rather than silently falling back, so misconfiguration
    is caught at startup rather than after silent data loss.
    """
    backend = os.environ.get(_GRAPH_BACKEND_ENV, _BACKEND_MEMORY)
    match backend:
        case str(b) if b == _BACKEND_MEMORY:
            return InMemoryGraph()
        case _:
            raise ValueError(
                f"Unknown graph backend {backend!r} in {_GRAPH_BACKEND_ENV}. "
                f"Supported: {_BACKEND_MEMORY!r}. "
                f"KuzuGraph backend is not yet implemented (Task 30)."
            )
