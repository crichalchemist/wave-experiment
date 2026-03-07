---
id: ADR-008
title: Persistent graph store (replacing networkx in-memory copy-on-write)
status: accepted
date: 2026-02-17
tags: [architecture, graph, persistence, kuzu]
---

# ADR-008: Persistent Knowledge Graph Store

## Decision

Replace the current `networkx.DiGraph` copy-on-write implementation with a persistent graph database. The current in-memory approach is suitable for development and small corpora; it cannot survive process restarts, cannot handle large investigation corpora, and cannot support concurrent analysis sessions.

## Candidate stores

| Store | Type | Notes |
|---|---|---|
| **Kuzu** | Embedded (like SQLite for graphs) | No server, ACID, Cypher-compatible, Python-first, open-source. Preferred for single-machine deployments. |
| **Neo4j Community** | Server-based | Industry standard Cypher; requires running a server; free tier available. Better for multi-user deployments. |
| **SQLite + adjacency tables** | Relational fallback | No graph-native queries; acceptable for small corpora if graph libraries are unavailable. |
| **NetworkX (current)** | In-memory | Retained for tests and ephemeral analysis. NOT for production persistence. |

## Recommended approach: Kuzu

Kuzu is an embedded graph database -- it runs in-process like SQLite, requires no server, is ACID-compliant, and supports a Cypher-compatible query language. It is the most appropriate choice for a single-machine investigation tool that needs persistence without operational overhead.

Why not Neo4j as primary: requires a running server, adds operational complexity, and the community edition has limits that affect production use.

Why not SQLite: graph traversal (n-hop paths, pattern matching) requires recursive CTEs which are slower and harder to reason about than native graph queries.

## Schema (Kuzu)

```cypher
-- Node types
CREATE NODE TABLE IF NOT EXISTS Entity (
    id STRING,
    PRIMARY KEY (id)
);

-- Edge (relationship) types
CREATE REL TABLE IF NOT EXISTS Relationship (
    FROM Entity TO Entity,
    relation STRING,
    confidence DOUBLE
);
```

Entity has only an `id STRING` primary key. Relationship carries `relation STRING` (RelationType enum value) and `confidence DOUBLE`. No `name`, `legal_domain`, `hop_count`, or `GapFinding` table -- those were from an earlier design that was never implemented.

## Abstraction strategy

Two concrete implementations behind a `GraphStore` Protocol defined in `src/data/graph_store.py`:

1. `InMemoryGraph` -- wraps `networkx.DiGraph` (tests, ephemeral analysis)
2. `KuzuGraph` -- wraps a Kuzu database connection (production, persistent)

Both implement the `GraphStore` Protocol:
```python
@runtime_checkable
class GraphStore(Protocol):
    def add_edge(self, source: str, target: str, relation: RelationType, confidence: float) -> None: ...
    def get_edge(self, source: str, target: str) -> KnowledgeEdge | None: ...
    def n_hop_paths(self, source: str, target: str, max_hops: int = 3) -> list[PathResult]: ...
    def successors(self, entity: str) -> list[str]: ...
    def nodes(self) -> list[str]: ...
```

Note: no `persist()` method. Kuzu persists automatically (embedded database); InMemoryGraph is ephemeral by design.

`KuzuGraph` additionally provides:
- `bulk_add_edges(edges)` -- batch insert for faster ingestion (collects unique nodes, MERGEs all, then creates edges)
- `close()` -- release database/connection handles

`graph_store_from_env()` factory function in `src/data/graph_store.py` selects the backend based on `DETECTIVE_GRAPH_BACKEND`.

## Dependency note

`kuzu` is imported with `try/except ImportError` in `src/data/kuzu_graph.py`. It is not declared in `pyproject.toml` optional dependencies. Users must install it manually (`pip install kuzu`) to use the Kuzu backend.

## Environment configuration

`DETECTIVE_GRAPH_BACKEND=memory|kuzu` selects the implementation (default: `memory`).
`DETECTIVE_KUZU_PATH` sets the Kuzu database directory. **Required** when backend is `kuzu` -- raises `ValueError` if not set (no default).

## Files

- `src/data/graph_store.py` -- GraphStore Protocol, InMemoryGraph, `graph_store_from_env()` factory
- `src/data/kuzu_graph.py` -- KuzuGraph implementation (persistent, embedded)
- `src/data/knowledge_graph.py` -- shared `PathResult`, `n_hop_paths()`, `_HOP_DECAY` (used by both backends)
