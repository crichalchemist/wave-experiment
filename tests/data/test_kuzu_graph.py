"""
Tests for KuzuGraph persistent store.

Tests that mock kuzu (always run) verify ImportError behaviour when kuzu is absent.
Tests marked kuzu_available are skipped if kuzu is not installed — safe for CI environments
where the kuzu wheel may not be present.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.data.knowledge_graph import _HOP_DECAY
from src.core.types import RelationType

# Probe at import time so the skipif marker has a concrete bool
try:
    import kuzu as _kuzu
except ImportError:
    _kuzu = None  # type: ignore[assignment]

kuzu_available = pytest.mark.skipif(
    _kuzu is None,
    reason="kuzu not installed",
)

# ---------------------------------------------------------------------------
# Tests that always run (kuzu mocked out)
# ---------------------------------------------------------------------------


def test_missing_kuzu_raises_import_error(tmp_path: pytest.TempPathFactory) -> None:
    """Construction must fail fast when the kuzu wheel is absent."""
    from src.data import kuzu_graph

    with patch.object(kuzu_graph, "_kuzu", None):
        with pytest.raises(ImportError, match="kuzu is required"):
            kuzu_graph.KuzuGraph(db_path=str(tmp_path / "db"))


# ---------------------------------------------------------------------------
# Tests that require actual kuzu
# ---------------------------------------------------------------------------


@kuzu_available
def test_kuzu_graph_satisfies_protocol(tmp_path: pytest.TempPathFactory) -> None:
    """KuzuGraph must structurally satisfy the GraphStore Protocol."""
    from src.data.graph_store import GraphStore
    from src.data.kuzu_graph import KuzuGraph

    g = KuzuGraph(db_path=str(tmp_path / "db"))
    assert isinstance(g, GraphStore)


@kuzu_available
def test_add_and_get_edge_persisted(tmp_path: pytest.TempPathFactory) -> None:
    """add_edge then get_edge returns a KnowledgeEdge with correct fields."""
    from src.core.types import KnowledgeEdge, RelationType
    from src.data.kuzu_graph import KuzuGraph

    g = KuzuGraph(db_path=str(tmp_path / "db"))
    g.add_edge("Alice", "Bob", RelationType.CAUSAL, 0.85)

    edge = g.get_edge("Alice", "Bob")

    assert edge is not None
    assert isinstance(edge, KnowledgeEdge)
    assert edge.source == "Alice"
    assert edge.target == "Bob"
    assert edge.relation == RelationType.CAUSAL
    assert abs(edge.confidence - 0.85) < 1e-9
    assert edge.hop_count == 1


@kuzu_available
def test_get_missing_edge_returns_none(tmp_path: pytest.TempPathFactory) -> None:
    """get_edge returns None when no direct edge exists between two nodes."""
    from src.core.types import RelationType
    from src.data.kuzu_graph import KuzuGraph

    g = KuzuGraph(db_path=str(tmp_path / "db"))
    g.add_edge("Alice", "Bob", RelationType.SEQUENTIAL, 0.7)

    assert g.get_edge("Bob", "Alice") is None
    assert g.get_edge("Alice", "Carol") is None


@kuzu_available
def test_persistence_across_connections(tmp_path: pytest.TempPathFactory) -> None:
    """Data written through one KuzuGraph is readable by a second instance at same path."""
    from src.core.types import RelationType
    from src.data.kuzu_graph import KuzuGraph

    db_path = str(tmp_path / "persistent_db")
    g1 = KuzuGraph(db_path=db_path)
    g1.add_edge("X", "Y", RelationType.CONDITIONAL, 0.6)

    # Explicitly close the first connection before opening a second one
    g1.close()

    g2 = KuzuGraph(db_path=db_path)
    edge = g2.get_edge("X", "Y")

    assert edge is not None
    assert edge.source == "X"
    assert edge.target == "Y"
    assert edge.relation == RelationType.CONDITIONAL


@kuzu_available
def test_n_hop_paths_single_hop(tmp_path: pytest.TempPathFactory) -> None:
    """A direct edge produces a single PathResult with hops=1."""
    from src.core.types import RelationType
    from src.data.kuzu_graph import KuzuGraph

    g = KuzuGraph(db_path=str(tmp_path / "db"))
    g.add_edge("A", "B", RelationType.CAUSAL, 0.9)

    paths = g.n_hop_paths("A", "B", max_hops=1)

    assert len(paths) == 1
    assert paths[0].path == ("A", "B")
    assert paths[0].hops == 1
    expected = 0.9 * _HOP_DECAY[RelationType.CAUSAL]  # 0.9 * 0.7 = 0.63
    assert abs(paths[0].confidence - expected) < 1e-9


@kuzu_available
def test_n_hop_paths_multi_hop(tmp_path: pytest.TempPathFactory) -> None:
    """A->B->C chain with max_hops=2 returns the 2-hop path."""
    from src.core.types import RelationType
    from src.data.kuzu_graph import KuzuGraph

    g = KuzuGraph(db_path=str(tmp_path / "db"))
    g.add_edge("A", "B", RelationType.CAUSAL, 0.9)
    g.add_edge("B", "C", RelationType.SEQUENTIAL, 0.8)

    paths = g.n_hop_paths("A", "C", max_hops=2)

    assert len(paths) == 1
    assert paths[0].path == ("A", "B", "C")
    assert paths[0].hops == 2
    # Confidence must be strictly less than the minimum individual edge confidence
    assert paths[0].confidence < 0.8


@kuzu_available
def test_n_hop_paths_missing_node_returns_empty(tmp_path: pytest.TempPathFactory) -> None:
    """n_hop_paths returns [] when source or target is not in the graph."""
    from src.core.types import RelationType
    from src.data.kuzu_graph import KuzuGraph

    g = KuzuGraph(db_path=str(tmp_path / "db"))
    g.add_edge("A", "B", RelationType.CAUSAL, 0.9)

    assert g.n_hop_paths("Z", "B", max_hops=3) == []
    assert g.n_hop_paths("A", "Z", max_hops=3) == []


@kuzu_available
def test_graph_store_from_env_kuzu_backend(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """graph_store_from_env returns a KuzuGraph when DETECTIVE_GRAPH_BACKEND=kuzu."""
    from src.data.graph_store import GraphStore, graph_store_from_env
    from src.data.kuzu_graph import KuzuGraph, _KUZU_DB_PATH_ENV

    db_path = str(tmp_path / "env_db")
    monkeypatch.setenv("DETECTIVE_GRAPH_BACKEND", "kuzu")
    monkeypatch.setenv(_KUZU_DB_PATH_ENV, db_path)

    store = graph_store_from_env()

    assert isinstance(store, KuzuGraph)
    assert isinstance(store, GraphStore)
