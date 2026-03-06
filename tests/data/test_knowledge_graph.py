import pytest
import networkx as nx

from src.core.types import KnowledgeEdge, RelationType
from src.data.knowledge_graph import (
    PathResult,
    _EDGE_DATA_KEY,
    get_edge,
    n_hop_paths,
    _HOP_DECAY,
)


def _make_graph() -> nx.DiGraph:
    """Test helper: create a fresh empty DiGraph."""
    return nx.DiGraph()


def _add_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    relation: RelationType,
    confidence: float,
) -> nx.DiGraph:
    """Test helper: add typed edge (copy-on-write)."""
    edge = KnowledgeEdge(
        source=source,
        target=target,
        relation=relation,
        confidence=confidence,
    )
    new_graph = graph.copy()
    new_graph.add_edge(source, target, **{_EDGE_DATA_KEY: edge})
    return new_graph


def test_get_edge_returns_none_for_missing() -> None:
    g = _make_graph()
    assert get_edge(g, "X", "Y") is None


def test_n_hop_paths_finds_direct_path() -> None:
    g = _make_graph()
    g = _add_edge(g, "A", "B", RelationType.CAUSAL, 0.9)
    paths = n_hop_paths(g, "A", "B", max_hops=1)
    assert len(paths) == 1
    assert paths[0].path == ("A", "B")
    assert paths[0].hops == 1


def test_n_hop_paths_finds_multi_hop() -> None:
    g = _make_graph()
    g = _add_edge(g, "A", "B", RelationType.SEQUENTIAL, 0.9)
    g = _add_edge(g, "B", "C", RelationType.SEQUENTIAL, 0.9)
    paths = n_hop_paths(g, "A", "C", max_hops=2)
    assert len(paths) == 1
    assert paths[0].path == ("A", "B", "C")
    assert paths[0].hops == 2


def test_n_hop_paths_empty_for_missing_node() -> None:
    g = _make_graph()
    paths = n_hop_paths(g, "X", "Y")
    assert paths == []


def test_n_hop_paths_raises_on_invalid_max_hops() -> None:
    g = _make_graph()
    with pytest.raises(ValueError, match="max_hops must be"):
        n_hop_paths(g, "A", "B", max_hops=0)


def test_n_hop_paths_sorted_by_confidence_descending() -> None:
    g = _make_graph()
    # Direct causal path (high confidence, one hop)
    g = _add_edge(g, "A", "C", RelationType.CAUSAL, 0.95)
    # Indirect sequential path (lower aggregate confidence)
    g = _add_edge(g, "A", "B", RelationType.SEQUENTIAL, 0.9)
    g = _add_edge(g, "B", "C", RelationType.SEQUENTIAL, 0.9)
    paths = n_hop_paths(g, "A", "C", max_hops=3)
    assert len(paths) == 2
    assert paths[0].confidence >= paths[1].confidence


def test_n_hop_paths_direct_causal_confidence() -> None:
    g = _make_graph()
    g = _add_edge(g, "A", "B", RelationType.CAUSAL, 0.9)
    paths = n_hop_paths(g, "A", "B", max_hops=1)
    # One hop: 0.9 * _HOP_DECAY[CAUSAL] = 0.9 * 0.7 = 0.63
    expected = 0.9 * 0.7
    assert paths[0].confidence == pytest.approx(expected)
