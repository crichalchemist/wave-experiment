"""Tests for trace API endpoints — recent, history, SSE streaming."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes import create_app  # noqa: E402
from src.core.reasoning_trace import ReasoningTrace  # noqa: E402
from src.core.trace_store import TraceStore  # noqa: E402


def _make_trace(module: str = "module_a", score: float = 0.5) -> ReasoningTrace:
    return ReasoningTrace.create(
        prompt=f"Rate the cognitive bias: {module}",
        raw_response=f"analysis... score: {score}",
        model="deepseek-test",
        route="scoring",
        duration_ms=100,
    )


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(path=tmp_path / "traces" / "reasoning.jsonl")


@pytest.fixture
def client(store: TraceStore) -> TestClient:
    """App with a trace store wired in."""
    from src.data.graph_store import InMemoryGraph

    app = create_app(trace_store=store, graph=InMemoryGraph())
    return TestClient(app)


@pytest.fixture
def client_no_store() -> TestClient:
    """App without a trace store — trace endpoints degrade gracefully."""
    from src.data.graph_store import InMemoryGraph

    app = create_app(graph=InMemoryGraph())
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /traces/recent
# ---------------------------------------------------------------------------


class TestTracesRecent:
    def test_returns_empty_when_no_traces(self, client: TestClient) -> None:
        response = client.get("/traces/recent")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_traces_after_recording(
        self, client: TestClient, store: TraceStore
    ) -> None:
        store.record(_make_trace())
        response = client.get("/traces/recent")
        data = response.json()
        assert len(data) == 1
        assert "id" in data[0]
        assert data[0]["module"] == "module_a"

    def test_respects_n_parameter(
        self, client: TestClient, store: TraceStore
    ) -> None:
        for i in range(5):
            store.record(_make_trace(score=i * 0.1))
        response = client.get("/traces/recent?n=2")
        assert len(response.json()) == 2

    def test_returns_empty_without_store(self, client_no_store: TestClient) -> None:
        response = client_no_store.get("/traces/recent")
        assert response.status_code == 200
        assert response.json() == []

    def test_response_contains_expected_fields(
        self, client: TestClient, store: TraceStore
    ) -> None:
        store.record(_make_trace())
        data = client.get("/traces/recent").json()
        trace = data[0]
        expected_keys = {
            "id", "timestamp", "module", "prompt", "raw_response",
            "parsed_score", "model", "route", "duration_ms",
        }
        assert set(trace.keys()) == expected_keys


# ---------------------------------------------------------------------------
# GET /traces/history
# ---------------------------------------------------------------------------


class TestTracesHistory:
    def test_returns_empty_when_no_traces(self, client: TestClient) -> None:
        response = client.get("/traces/history")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_newest_first(
        self, client: TestClient, store: TraceStore
    ) -> None:
        t1 = _make_trace(score=0.1)
        t2 = _make_trace(score=0.2)
        store.record(t1)
        store.record(t2)
        response = client.get("/traces/history")
        data = response.json()
        assert data[0]["id"] == t2.id
        assert data[1]["id"] == t1.id

    def test_pagination(self, client: TestClient, store: TraceStore) -> None:
        for i in range(5):
            store.record(_make_trace(score=i * 0.1))
        response = client.get("/traces/history?offset=2&limit=2")
        assert len(response.json()) == 2

    def test_returns_empty_without_store(self, client_no_store: TestClient) -> None:
        response = client_no_store.get("/traces/history")
        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# GET /traces/stream (SSE)
# ---------------------------------------------------------------------------


class TestTracesStream:
    def test_stream_without_store_returns_comment(
        self, client_no_store: TestClient
    ) -> None:
        """When no trace store, SSE returns a single comment and closes."""
        response = client_no_store.get("/traces/stream")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "no trace store" in response.text


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------


class TestCORS:
    def test_cors_allows_crichalchemist_origin(self, client: TestClient) -> None:
        response = client.get(
            "/traces/recent",
            headers={"Origin": "https://www.crichalchemist.com"},
        )
        assert response.headers.get("access-control-allow-origin") == "https://www.crichalchemist.com"

    def test_cors_rejects_unknown_origin(self, client: TestClient) -> None:
        response = client.get(
            "/traces/recent",
            headers={"Origin": "https://evil.com"},
        )
        assert "access-control-allow-origin" not in response.headers
