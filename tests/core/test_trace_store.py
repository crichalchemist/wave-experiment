"""Tests for TraceStore: JSONL persistence, deque, SSE subscribers."""

import asyncio
import json
from pathlib import Path

import pytest

from src.core.reasoning_trace import ReasoningTrace
from src.core.trace_store import TraceStore


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


# ---------------------------------------------------------------------------
# record + recent
# ---------------------------------------------------------------------------


class TestRecordAndRecent:
    def test_record_appends_to_deque(self, store: TraceStore) -> None:
        trace = _make_trace()
        store.record(trace)
        recent = store.recent()
        assert len(recent) == 1
        assert recent[0].id == trace.id

    def test_recent_returns_last_n(self, store: TraceStore) -> None:
        for i in range(10):
            store.record(_make_trace(score=i * 0.1))
        assert len(store.recent(n=3)) == 3

    def test_record_persists_to_jsonl(self, store: TraceStore) -> None:
        trace = _make_trace()
        store.record(trace)
        lines = store.path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["id"] == trace.id
        assert data["module"] == trace.module

    def test_multiple_records_append_lines(self, store: TraceStore) -> None:
        store.record(_make_trace(score=0.1))
        store.record(_make_trace(score=0.2))
        store.record(_make_trace(score=0.3))
        lines = store.path.read_text().strip().splitlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# historical (JSONL pagination)
# ---------------------------------------------------------------------------


class TestHistorical:
    def test_empty_file_returns_empty_list(self, store: TraceStore) -> None:
        assert store.historical() == []

    def test_newest_first_ordering(self, store: TraceStore) -> None:
        t1 = _make_trace(score=0.1)
        t2 = _make_trace(score=0.2)
        store.record(t1)
        store.record(t2)
        history = store.historical()
        assert history[0].id == t2.id  # newest first
        assert history[1].id == t1.id

    def test_pagination_offset_limit(self, store: TraceStore) -> None:
        traces = []
        for i in range(5):
            t = _make_trace(score=i * 0.1)
            store.record(t)
            traces.append(t)
        # Get second page (offset=2, limit=2)
        page = store.historical(offset=2, limit=2)
        assert len(page) == 2
        # Reversed order: newest is traces[4], so offset=2 is traces[2]
        assert page[0].id == traces[2].id
        assert page[1].id == traces[1].id


# ---------------------------------------------------------------------------
# SSE subscribers
# ---------------------------------------------------------------------------


class TestSubscribers:
    def test_subscribe_receives_new_traces(self, store: TraceStore) -> None:
        q = store.subscribe()
        trace = _make_trace()
        store.record(trace)
        received = q.get_nowait()
        assert received.id == trace.id

    def test_unsubscribe_stops_receiving(self, store: TraceStore) -> None:
        q = store.subscribe()
        store.unsubscribe(q)
        store.record(_make_trace())
        assert q.empty()

    def test_multiple_subscribers(self, store: TraceStore) -> None:
        q1 = store.subscribe()
        q2 = store.subscribe()
        trace = _make_trace()
        store.record(trace)
        assert q1.get_nowait().id == trace.id
        assert q2.get_nowait().id == trace.id

    def test_unsubscribe_nonexistent_is_noop(self, store: TraceStore) -> None:
        q: asyncio.Queue[ReasoningTrace] = asyncio.Queue()
        store.unsubscribe(q)  # should not raise


# ---------------------------------------------------------------------------
# directory creation
# ---------------------------------------------------------------------------


class TestDirectoryCreation:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "traces.jsonl"
        store = TraceStore(path=deep_path)
        store.record(_make_trace())
        assert deep_path.exists()
