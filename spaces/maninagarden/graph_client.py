"""
Thin HTTP client for fetching the knowledge graph from the detective API.

Uses stdlib only (no requests dependency). Caches graph data for 5 minutes
to avoid hammering the API on every tab interaction. The DETECTIVE_API_URL
environment variable should be set as a Space secret.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)

DETECTIVE_API_URL = os.environ.get("DETECTIVE_API_URL", "http://localhost:8000")
_CACHE_TTL = 300  # 5 minutes

_graph_cache: dict | None = None
_cache_time: float = 0.0


def fetch_graph() -> dict:
    """Fetch full graph from detective API. Returns {nodes, edges, stats}.

    Caches for 5 minutes. On failure returns empty graph structure so
    callers never crash — the Space degrades gracefully when the API is down.
    """
    global _graph_cache, _cache_time
    if _graph_cache and (time.time() - _cache_time) < _CACHE_TTL:
        return _graph_cache

    url = f"{DETECTIVE_API_URL}/graph/export"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        _graph_cache = data
        _cache_time = time.time()
        logger.info(
            "Graph fetched: %d nodes, %d edges",
            data.get("stats", {}).get("node_count", 0),
            data.get("stats", {}).get("edge_count", 0),
        )
        return data
    except Exception as e:
        logger.warning("Failed to fetch graph from %s: %s", url, e)
        return {"nodes": [], "edges": [], "stats": {"node_count": 0, "edge_count": 0}}


def invalidate_cache() -> None:
    """Clear the graph cache so the next fetch_graph() call hits the API."""
    global _graph_cache, _cache_time
    _graph_cache = None
    _cache_time = 0.0
