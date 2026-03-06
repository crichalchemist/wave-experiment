"""Persistent trace store with JSONL backing and SSE subscriber support.

Traces are appended to a JSONL file for durability, kept in a bounded
in-memory deque for fast recent-query access, and pushed to asyncio.Queue
subscribers for Server-Sent Events streaming.

Thread-safe: parallel_evolution dispatches LLM calls via run_in_executor,
so record() may be called from multiple threads concurrently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.core.reasoning_trace import ReasoningTrace

_logger = logging.getLogger(__name__)

_DEFAULT_DEQUE_SIZE: int = 500
_DEFAULT_RECENT: int = 50


@dataclass
class TraceStore:
    """Append-only trace store backed by JSONL on disk."""

    path: Path
    _deque: deque[ReasoningTrace] = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)
    _subscribers: list[asyncio.Queue[ReasoningTrace]] = field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        self._deque = deque(maxlen=_DEFAULT_DEQUE_SIZE)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load_recent_from_disk()

    def _load_recent_from_disk(self) -> None:
        """Pre-populate deque from the tail of the JSONL file so /traces/recent isn't cold."""
        if not self.path.exists():
            return
        try:
            lines: list[str] = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        lines.append(stripped)
            for line in lines[-_DEFAULT_DEQUE_SIZE:]:
                data = json.loads(line)
                self._deque.append(ReasoningTrace(**data))
            _logger.info("Loaded %d traces from disk into deque", len(self._deque))
        except Exception:
            _logger.warning("Failed to pre-load traces from %s", self.path, exc_info=True)

    def record(self, trace: ReasoningTrace) -> None:
        """Persist trace to JSONL, push to deque, notify SSE subscribers."""
        line = json.dumps(asdict(trace), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._deque.append(trace)
        # Notify subscribers (non-blocking put from any thread)
        for q in list(self._subscribers):
            try:
                q.put_nowait(trace)
            except asyncio.QueueFull:
                _logger.debug("SSE subscriber queue full, dropping trace %s", trace.id)

    def recent(self, n: int = _DEFAULT_RECENT) -> list[ReasoningTrace]:
        """Return the last *n* traces from the in-memory deque."""
        with self._lock:
            items = list(self._deque)
        return items[-n:]

    def subscribe(self) -> asyncio.Queue[ReasoningTrace]:
        """Create a new SSE subscriber queue."""
        q: asyncio.Queue[ReasoningTrace] = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[ReasoningTrace]) -> None:
        """Remove an SSE subscriber queue."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def historical(self, offset: int = 0, limit: int = 50) -> list[ReasoningTrace]:
        """Read traces from JSONL with pagination (newest first)."""
        if not self.path.exists():
            return []
        with self._lock:
            # First pass: collect byte positions of non-empty lines
            positions: list[int] = []
            with open(self.path, encoding="utf-8") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        positions.append(pos)

            # Reverse for newest-first, then slice for pagination
            positions.reverse()
            page_positions = positions[offset : offset + limit]

            # Second pass: read only the lines we need
            lines: list[str] = []
            with open(self.path, encoding="utf-8") as f:
                for pos in page_positions:
                    f.seek(pos)
                    lines.append(f.readline().strip())

        traces: list[ReasoningTrace] = []
        for line in lines:
            if not line:
                continue
            data = json.loads(line)
            traces.append(ReasoningTrace(**data))
        return traces
