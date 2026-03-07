---
id: ADR-005
title: Typed semantic relations on knowledge graph edges
status: accepted
date: 2026-02-16
tags: [knowledge-graph, relations, networkx, SRA]
---

# ADR-005: Typed Semantic Relations on Knowledge Graph Edges

## Decision

Knowledge graph edges carry one of six typed semantic relations: CONDITIONAL, CAUSAL, INSTANTIATIVE, SEQUENTIAL, CO_MENTIONED, ASSOCIATED. Edges are NOT generic weighted connections.

## Context

The SRA FlowchartQA paper showed that a 2-hop **causal** chain (A caused B caused C) is investigatively different from a 2-hop **sequential** chain (A preceded B preceded C). With untyped edges, n-hop confidence decay is generic; with typed relations, it reflects the actual inferential strength of the chain.

A CAUSAL hop degrades confidence faster than a SEQUENTIAL hop (correlation != causation). A CONDITIONAL hop depends on a separate premise being true. CO_MENTIONED and ASSOCIATED were introduced by ADR-015 for the epstein-docs ingestion pipeline.

## Confidence decay

`KnowledgeEdge.hop_count` tracks position in a multi-hop path. Per-hop decay coefficients are defined in `_HOP_DECAY` in `src/data/knowledge_graph.py`:

| RelationType | Decay | Rationale |
|---|---|---|
| CAUSAL | 0.7 | Causation claims degrade fastest |
| CONDITIONAL | 0.75 | Contingent on separate premise |
| CO_MENTIONED | 0.6 | Same-page co-occurrence is correlative, not causal |
| INSTANTIATIVE | 0.85 | Specific instance of a general pattern |
| ASSOCIATED | 0.8 | Role-based association from structured analysis |
| SEQUENTIAL | 0.9 | Chronological sequence is the weakest decay |

## PathResult

`PathResult` is a frozen dataclass in `src/data/knowledge_graph.py` representing a traversal path with aggregate confidence:

```python
@dataclass(frozen=True)
class PathResult:
    path: tuple[str, ...]
    confidence: float
    hops: int
```

## Graph representation

The underlying representation is `networkx.DiGraph` (aliased as `KnowledgeGraph`). Edge data is stored under the `_EDGE_DATA_KEY = "edge"` attribute as a `KnowledgeEdge` frozen dataclass.

## Validation

`KnowledgeEdge` is a `frozen=True` dataclass that validates:
- `confidence` in [0.0, 1.0]
- `hop_count >= 1`
- `source` and `target` are non-empty

## Files

- `src/core/types.py` -- RelationType enum (6 values), KnowledgeEdge dataclass
- `src/data/knowledge_graph.py` -- PathResult, n_hop_paths(), _HOP_DECAY, graph operations
- `src/data/graph_store.py` -- GraphStore Protocol, InMemoryGraph (wraps networkx.DiGraph)
