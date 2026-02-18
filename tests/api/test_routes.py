"""
Tests for FastAPI routes — data minimization, correct HTTP status codes, response shape.

All tests are skipped when fastapi is not installed (CI-safe for local Python 3.14
environments where fastapi is unavailable).
"""
from __future__ import annotations

import pytest

# Skip entire module when fastapi is not installed.
fastapi = pytest.importorskip("fastapi")


from fastapi.testclient import TestClient  # noqa: E402  (after importorskip guard)

from src.api.routes import create_app  # noqa: E402


# ---------------------------------------------------------------------------
# Shared client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    """Fresh app per test — avoids cross-test state bleed from the InMemoryGraph."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------

def test_analyze_status_200(client: TestClient) -> None:
    """Valid claim returns HTTP 200."""
    response = client.post("/analyze", json={"claim": "Entity A was active in 2013"})
    assert response.status_code == 200


def test_analyze_response_shape_and_value_types(client: TestClient) -> None:
    """Data minimization: exactly verdict (str), confidence (float in [0,1]), evidence_count (int)."""
    response = client.post("/analyze", json={"claim": "Entity A was active in 2013"})
    body = response.json()
    assert set(body.keys()) == {"verdict", "confidence", "evidence_count"}
    assert isinstance(body["verdict"], str) and body["verdict"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["evidence_count"], int) and body["evidence_count"] >= 0


def test_analyze_missing_claim_returns_422(client: TestClient) -> None:
    """Missing required field triggers Pydantic validation → 422 Unprocessable Entity."""
    response = client.post("/analyze", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /network/{entity}
# ---------------------------------------------------------------------------

def test_network_endpoint_returns_relationships(client: TestClient) -> None:
    """Response body contains entity and relationships list."""
    response = client.get("/network/EntityA")
    assert response.status_code == 200
    body = response.json()
    assert "entity" in body
    assert "relationships" in body
    assert isinstance(body["relationships"], list)


def test_network_unknown_entity_returns_empty_list(client: TestClient) -> None:
    """Entity absent from graph yields an empty relationships list, not an error."""
    response = client.get("/network/UnknownEntity")
    assert response.status_code == 200
    body = response.json()
    assert body["relationships"] == []


# ---------------------------------------------------------------------------
# /evolve
# ---------------------------------------------------------------------------

def test_evolve_response_shape(client: TestClient) -> None:
    """Data minimization: /evolve response carries exactly hypothesis_id, statement, confidence."""
    response = client.post("/evolve", json={"evidence_path": "new evidence text"})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"hypothesis_id", "statement", "confidence"}


def test_evolve_missing_evidence_path_returns_422(client: TestClient) -> None:
    """Missing required field triggers Pydantic validation → 422 Unprocessable Entity."""
    response = client.post("/evolve", json={})
    assert response.status_code == 422
