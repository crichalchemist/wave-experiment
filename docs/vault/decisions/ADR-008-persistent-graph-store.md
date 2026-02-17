---
id: ADR-008
title: Persistent graph store (replacing networkx in-memory copy-on-write)
status: proposed
date: 2026-02-17
tags: [architecture, graph, persistence, neo4j, kuzu]
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

Kuzu is an embedded graph database — it runs in-process like SQLite, requires no server, is ACID-compliant, and supports a Cypher-compatible query language. It is the most appropriate choice for a single-machine investigation tool that needs persistence without operational overhead.

Why not Neo4j as primary: requires a running server, adds operational complexity, and the community edition has limits that affect production use.

Why not SQLite: graph traversal (n-hop paths, pattern matching) requires recursive CTEs which are slower and harder to reason about than native graph queries.

## Schema (Kuzu)

```cypher
-- Node types
CREATE NODE TABLE Entity (
    id STRING,
    name STRING,
    legal_domain STRING,  -- LegalDomain enum value
    PRIMARY KEY (id)
);

-- Edge (relationship) types
CREATE REL TABLE KnowledgeEdge (
    FROM Entity TO Entity,
    relation STRING,          -- RelationType enum value
    confidence DOUBLE,
    hop_count INT64
);

-- Gap findings
CREATE NODE TABLE GapFinding (
    id STRING,
    gap_type STRING,          -- GapType enum value
    description STRING,
    confidence DOUBLE,
    location STRING,
    run_id STRING,            -- links gap to analysis run
    PRIMARY KEY (id)
);
```

## Abstraction strategy

The `KnowledgeGraph` type alias and all public functions in `src/data/knowledge_graph.py` must remain unchanged from the caller's perspective. The Kuzu implementation is a drop-in replacement behind the same functional interface.

Two concrete implementations:
1. `InMemoryGraph` — wraps `networkx.DiGraph` (tests, ephemeral analysis)
2. `KuzuGraph` — wraps a Kuzu database connection (production, persistent)

Both implement a `GraphStore` Protocol:
```python
@runtime_checkable
class GraphStore(Protocol):
    def add_edge(self, source: str, target: str, relation: RelationType, confidence: float) -> None: ...
    def get_edge(self, source: str, target: str) -> KnowledgeEdge | None: ...
    def n_hop_paths(self, source: str, target: str, max_hops: int) -> list[PathResult]: ...
    def persist(self) -> None: ...  # no-op for in-memory
```

Note: The persistent store moves from pure functional (copy-on-write) to mutable-with-protocol because true graph persistence requires mutation. The functional wrapper functions in `knowledge_graph.py` become thin adapters over the mutable `GraphStore`.

## Environment configuration

`GRAPH_STORE=memory|kuzu` selects the implementation.
`GRAPH_DB_PATH` sets the Kuzu database directory (default: `data/graph.kuzu`).

## Migration plan

Phase 1 (current): `InMemoryGraph` (networkx) — all tests pass, no persistence
Phase 2 (Task 29): Add `GraphStore` Protocol + `InMemoryGraph` wrapper — refactor `knowledge_graph.py` to use it
Phase 3 (Task 30): Implement `KuzuGraph` — add kuzu as optional dependency, wire to `graph_store_from_env()`

## Files

- `src/data/knowledge_graph.py` — refactor to use GraphStore Protocol (Task 29)
- `src/data/kuzu_graph.py` — KuzuGraph implementation (Task 30)
- `pyproject.toml` — add `kuzu>=0.5.0` to optional dependencies
