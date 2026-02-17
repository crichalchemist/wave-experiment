---
id: ADR-005
title: Typed semantic relations on knowledge graph edges
status: accepted
date: 2026-02-16
tags: [knowledge-graph, relations, networkx, SRA]
---

# ADR-005: Typed Semantic Relations on Knowledge Graph Edges

## Decision

Knowledge graph edges carry one of four typed semantic relations: CONDITIONAL, CAUSAL, INSTANTIATIVE, SEQUENTIAL. Edges are NOT generic weighted connections.

## Context

The SRA FlowchartQA paper showed that a 2-hop **causal** chain (A caused B caused C) is investigatively different from a 2-hop **sequential** chain (A preceded B preceded C). With untyped edges, n-hop confidence decay is generic; with typed relations, it reflects the actual inferential strength of the chain.

A CAUSAL hop degrades confidence faster than a SEQUENTIAL hop (correlation ≠ causation). A CONDITIONAL hop depends on a separate premise being true.

## Confidence decay

`KnowledgeEdge.hop_count` tracks position in a multi-hop path. Decay coefficient is applied per hop based on relation type (to be defined in `src/data/knowledge_graph.py`).

## Validation

`KnowledgeEdge` is a `frozen=True` dataclass that validates:
- `confidence` in [0.0, 1.0]
- `hop_count >= 1`
- `source` and `target` are non-empty

## Files

- `src/core/types.py` — RelationType enum, KnowledgeEdge dataclass
- `src/data/knowledge_graph.py` — graph operations (planned)
