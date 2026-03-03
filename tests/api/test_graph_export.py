"""
Tests for GET /graph/export endpoint.

Verifies response structure, edge format, and empty-graph behaviour.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes import create_app  # noqa: E402
from src.core.types import RelationType  # noqa: E402
from src.data.graph_store import InMemoryGraph  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_graph() -> InMemoryGraph:
    """Graph with 3 nodes and 4 edges for testing export shape."""
    g = InMemoryGraph()
    g.add_edge("Alice", "Bob", RelationType.ASSOCIATED, confidence=0.9)
    g.add_edge("Bob", "Carol", RelationType.CO_MENTIONED, confidence=0.5)
    g.add_edge("Alice", "Carol", RelationType.CAUSAL, confidence=0.7)
    g.add_edge("Carol", "Alice", RelationType.SEQUENTIAL, confidence=0.6)
    return g


@pytest.fixture
def client_with_graph(populated_graph: InMemoryGraph) -> TestClient:
    app = create_app(graph=populated_graph)
    return TestClient(app)


@pytest.fixture
def empty_client() -> TestClient:
    """Client backed by an empty graph."""
    app = create_app(graph=InMemoryGraph())
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_export_returns_correct_structure(client_with_graph: TestClient) -> None:
    """Response has nodes, edges, stats with correct types."""
    response = client_with_graph.get("/graph/export")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"nodes", "edges", "stats"}
    assert isinstance(body["nodes"], list)
    assert isinstance(body["edges"], list)
    assert isinstance(body["stats"], dict)
    assert body["stats"]["node_count"] == 3
    assert body["stats"]["edge_count"] == 4


def test_graph_export_edge_format(client_with_graph: TestClient) -> None:
    """Each edge includes source, target, relation, and confidence."""
    body = client_with_graph.get("/graph/export").json()
    for edge in body["edges"]:
        assert set(edge.keys()) == {"source", "target", "relation", "confidence"}
        assert isinstance(edge["source"], str)
        assert isinstance(edge["target"], str)
        assert isinstance(edge["relation"], str)
        assert isinstance(edge["confidence"], float)
        assert 0.0 <= edge["confidence"] <= 1.0


def test_graph_export_contains_expected_entities(client_with_graph: TestClient) -> None:
    """All three populated entities appear in nodes list."""
    body = client_with_graph.get("/graph/export").json()
    assert set(body["nodes"]) == {"Alice", "Bob", "Carol"}


def test_graph_export_empty_graph(empty_client: TestClient) -> None:
    """Empty graph returns empty lists and zero counts."""
    body = empty_client.get("/graph/export").json()
    assert body["nodes"] == []
    assert body["edges"] == []
    assert body["stats"]["node_count"] == 0
    assert body["stats"]["edge_count"] == 0


def test_graph_export_relation_values(client_with_graph: TestClient) -> None:
    """Relation values match RelationType enum string values."""
    body = client_with_graph.get("/graph/export").json()
    valid_relations = {rt.value for rt in RelationType}
    for edge in body["edges"]:
        assert edge["relation"] in valid_relations
