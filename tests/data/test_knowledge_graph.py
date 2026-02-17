import pytest
import networkx as nx

from src.core.types import RelationType
from src.data.knowledge_graph import (
    PathResult,
    add_edge,
    get_edge,
    make_graph,
    n_hop_paths,
    _HOP_DECAY,
)


def test_make_graph_returns_empty_digraph() -> None:
    g = make_graph()
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0


def test_add_edge_returns_new_graph() -> None:
    g = make_graph()
    g2 = add_edge(g, "A", "B", RelationType.CAUSAL, 0.9)
    # Original unchanged
    assert g.number_of_edges() == 0
    # New graph has the edge
    assert g2.number_of_edges() == 1


def test_add_edge_stores_typed_edge() -> None:
    g = make_graph()
    g = add_edge(g, "A", "B", RelationType.SEQUENTIAL, 0.8)
    edge = get_edge(g, "A", "B")
    assert edge is not None
    assert edge.relation == RelationType.SEQUENTIAL
    assert edge.confidence == pytest.approx(0.8)
    assert edge.source == "A"
    assert edge.target == "B"


def test_get_edge_returns_none_for_missing() -> None:
    g = make_graph()
    assert get_edge(g, "X", "Y") is None


def test_n_hop_paths_finds_direct_path() -> None:
    g = make_graph()
    g = add_edge(g, "A", "B", RelationType.CAUSAL, 0.9)
    paths = n_hop_paths(g, "A", "B", max_hops=1)
    assert len(paths) == 1
    assert paths[0].path == ("A", "B")
    assert paths[0].hops == 1


def test_n_hop_paths_finds_multi_hop() -> None:
    g = make_graph()
    g = add_edge(g, "A", "B", RelationType.SEQUENTIAL, 0.9)
    g = add_edge(g, "B", "C", RelationType.SEQUENTIAL, 0.9)
    paths = n_hop_paths(g, "A", "C", max_hops=2)
    assert len(paths) == 1
    assert paths[0].path == ("A", "B", "C")
    assert paths[0].hops == 2


def test_n_hop_paths_empty_for_missing_node() -> None:
    g = make_graph()
    paths = n_hop_paths(g, "X", "Y")
    assert paths == []


def test_n_hop_paths_raises_on_invalid_max_hops() -> None:
    g = make_graph()
    with pytest.raises(ValueError, match="max_hops must be"):
        n_hop_paths(g, "A", "B", max_hops=0)


def test_n_hop_paths_sorted_by_confidence_descending() -> None:
    g = make_graph()
    # Direct causal path (high confidence, one hop)
    g = add_edge(g, "A", "C", RelationType.CAUSAL, 0.95)
    # Indirect sequential path (lower aggregate confidence)
    g = add_edge(g, "A", "B", RelationType.SEQUENTIAL, 0.9)
    g = add_edge(g, "B", "C", RelationType.SEQUENTIAL, 0.9)
    paths = n_hop_paths(g, "A", "C", max_hops=3)
    assert len(paths) == 2
    assert paths[0].confidence >= paths[1].confidence


def test_n_hop_paths_direct_causal_confidence() -> None:
    g = make_graph()
    g = add_edge(g, "A", "B", RelationType.CAUSAL, 0.9)
    paths = n_hop_paths(g, "A", "B", max_hops=1)
    # One hop: 0.9 * _HOP_DECAY[CAUSAL] = 0.9 * 0.7 = 0.63
    expected = 0.9 * 0.7
    assert paths[0].confidence == pytest.approx(expected)
