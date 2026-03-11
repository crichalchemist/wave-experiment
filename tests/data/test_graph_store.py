from __future__ import annotations

import pytest

from src.core.types import LegalDomain, RelationType
from src.data.knowledge_graph import _HOP_DECAY
from src.data.graph_store import (
    GraphStore,
    InMemoryGraph,
    graph_store_from_env,
    _GRAPH_BACKEND_ENV,
    _BACKEND_MEMORY,
)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_in_memory_graph_satisfies_protocol() -> None:
    """runtime_checkable Protocol is the structural guarantee callers rely on."""
    assert isinstance(InMemoryGraph(), GraphStore)


# ---------------------------------------------------------------------------
# add_edge / get_edge round-trips
# ---------------------------------------------------------------------------


def test_add_and_get_edge() -> None:
    g = InMemoryGraph()
    g.add_edge("A", "B", RelationType.CAUSAL, 0.9)
    edge = g.get_edge("A", "B")
    assert edge is not None
    assert edge.source == "A"
    assert edge.target == "B"
    assert edge.relation == RelationType.CAUSAL
    assert edge.confidence == pytest.approx(0.9)


def test_get_nonexistent_edge_returns_none() -> None:
    g = InMemoryGraph()
    assert g.get_edge("X", "Y") is None


def test_multiple_edges_preserved() -> None:
    """All inserted edges must be independently retrievable — no eviction or collision."""
    g = InMemoryGraph()
    g.add_edge("A", "B", RelationType.CAUSAL, 0.9)
    g.add_edge("B", "C", RelationType.SEQUENTIAL, 0.8)
    g.add_edge("C", "D", RelationType.CONDITIONAL, 0.7)

    ab = g.get_edge("A", "B")
    bc = g.get_edge("B", "C")
    cd = g.get_edge("C", "D")

    assert ab is not None and ab.relation == RelationType.CAUSAL
    assert bc is not None and bc.relation == RelationType.SEQUENTIAL
    assert cd is not None and cd.relation == RelationType.CONDITIONAL


# ---------------------------------------------------------------------------
# n_hop_paths
# ---------------------------------------------------------------------------


def test_n_hop_paths_direct() -> None:
    g = InMemoryGraph()
    g.add_edge("A", "B", RelationType.SEQUENTIAL, 0.9)
    paths = g.n_hop_paths("A", "B", max_hops=1)
    assert len(paths) == 1
    assert paths[0].hops == 1
    assert paths[0].path == ("A", "B")
    expected = 0.9 * _HOP_DECAY[RelationType.SEQUENTIAL]
    assert paths[0].confidence == pytest.approx(expected)


def test_n_hop_paths_missing_nodes_returns_empty() -> None:
    """Absent source or target must never raise — absence is investigatively meaningful."""
    g = InMemoryGraph()
    assert g.n_hop_paths("ghost", "phantom", max_hops=3) == []


def test_n_hop_paths_confidence_decays() -> None:
    """
    Two CAUSAL hops — aggregate must be strictly less than the first edge's
    confidence alone, confirming per-hop decay is applied.
    """
    g = InMemoryGraph()
    g.add_edge("A", "B", RelationType.CAUSAL, 0.9)
    g.add_edge("B", "C", RelationType.CAUSAL, 0.9)

    paths = g.n_hop_paths("A", "C", max_hops=2)
    assert len(paths) == 1

    first_edge_confidence = 0.9
    # Two-hop aggregate: (0.9 * 0.7) * (0.9 * 0.7) = 0.3969
    expected = (0.9 * _HOP_DECAY[RelationType.CAUSAL]) * (0.9 * _HOP_DECAY[RelationType.CAUSAL])
    assert paths[0].confidence == pytest.approx(expected)
    assert paths[0].confidence < first_edge_confidence


def test_n_hop_paths_sorted_by_confidence_desc() -> None:
    """
    Higher-confidence paths surface first — investigators should see the
    strongest evidence chain before weaker indirect paths.
    """
    g = InMemoryGraph()
    # Direct high-confidence causal edge
    g.add_edge("A", "C", RelationType.CAUSAL, 0.95)
    # Longer indirect sequential path (lower aggregate)
    g.add_edge("A", "B", RelationType.SEQUENTIAL, 0.9)
    g.add_edge("B", "C", RelationType.SEQUENTIAL, 0.9)

    paths = g.n_hop_paths("A", "C", max_hops=3)
    assert len(paths) == 2
    # Sorted descending
    assert paths[0].confidence >= paths[1].confidence


# ---------------------------------------------------------------------------
# graph_store_from_env factory
# ---------------------------------------------------------------------------


def test_graph_store_from_env_default_is_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env var must default to InMemoryGraph so zero-config local runs work."""
    monkeypatch.delenv(_GRAPH_BACKEND_ENV, raising=False)
    store = graph_store_from_env()
    assert isinstance(store, InMemoryGraph)


def test_graph_store_from_env_memory_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_GRAPH_BACKEND_ENV, _BACKEND_MEMORY)
    store = graph_store_from_env()
    assert isinstance(store, InMemoryGraph)


def test_graph_store_from_env_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown backends must fail fast rather than silently fall back to memory."""
    monkeypatch.setenv(_GRAPH_BACKEND_ENV, "nosuchbackend")
    with pytest.raises(ValueError, match="nosuchbackend"):
        graph_store_from_env()


# ---------------------------------------------------------------------------
# legal_domain on edges
# ---------------------------------------------------------------------------


def test_add_edge_with_legal_domain() -> None:
    """Edges can carry an optional legal domain annotation."""
    g = InMemoryGraph()
    g.add_edge("A", "B", RelationType.CAUSAL, 0.9, legal_domain=LegalDomain.STATUTE)
    edge = g.get_edge("A", "B")
    assert edge is not None
    assert edge.legal_domain == LegalDomain.STATUTE


def test_add_edge_legal_domain_default_none() -> None:
    """Backward compatibility: edges without legal_domain default to None."""
    g = InMemoryGraph()
    g.add_edge("A", "B", RelationType.CAUSAL, 0.9)
    edge = g.get_edge("A", "B")
    assert edge is not None
    assert edge.legal_domain is None


def test_edges_with_different_legal_domains() -> None:
    """Same entity pair can appear in different legal domain contexts."""
    g = InMemoryGraph()
    g.add_edge("Corp", "Agency", RelationType.CAUSAL, 0.8, legal_domain=LegalDomain.STATUTE)
    # Note: networkx overwrites same-direction edge, so use different pairs
    g.add_edge("Agency", "Corp", RelationType.CAUSAL, 0.6, legal_domain=LegalDomain.ENFORCEMENT_PRACTICE)

    statute_edge = g.get_edge("Corp", "Agency")
    enforcement_edge = g.get_edge("Agency", "Corp")

    assert statute_edge is not None and statute_edge.legal_domain == LegalDomain.STATUTE
    assert enforcement_edge is not None and enforcement_edge.legal_domain == LegalDomain.ENFORCEMENT_PRACTICE
