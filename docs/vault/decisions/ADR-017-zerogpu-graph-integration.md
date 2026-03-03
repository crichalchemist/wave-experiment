---
id: ADR-017
title: ZeroGPU + graph-enhanced welfare forecasting
status: accepted
date: 2026-03-03
tags: [space, gpu, graph, forecasting, visualization, zerogpu, cugraph]
---

# ADR-017: ZeroGPU + Graph-Enhanced Welfare Forecasting

## Decision

Integrate ZeroGPU H200 acceleration, interactive entity network exploration, and graph topology features into the Phi Research Workbench (`crichalchemist/maninagarden` HuggingFace Space).

## Context

The Space was converted from 1 vCPU to ZeroGPU (H200, 70GB VRAM) but ran pure time-series forecasting with no GPU acceleration and no graph operations. Meanwhile, the detective project has rich graph infrastructure: ~3K entities from epstein-docs ingestion (ADR-015/016), GATv2Conv in HybridGraphLayer, and confidence-decayed n-hop traversal. Bridging these systems enables GPU-accelerated inference, interactive graph exploration, and graph-informed welfare forecasting.

## Implementation

### Phase 1: ZeroGPU acceleration

- `@spaces.GPU(duration=30)` on `run_inference()`, `@spaces.GPU(duration=60)` on `compare_experiments()`
- Core logic extracted to undecorated `_run_inference_core()` to avoid nested decorator errors (ZeroGPU constraint)
- Model moved to CUDA in `load_model()` when available; `.cpu().numpy()` for GPU→CPU tensor transfer
- `input_size` read from `training_metadata.json` for variable-width model support (36 or 43 features)

### Phase 2: Entity Network Explorer (Tab 7)

**API bridge:**
- `GET /graph/export` endpoint in `src/api/routes.py` returns `{nodes, edges, stats}`
- CORS expanded to allow `crichalchemist-maninagarden.hf.space`
- `graph_client.py` fetches graph via stdlib urllib with 5-minute cache TTL

**Graph analytics:**
- `graph_analytics.py` with cuGraph primary / networkx fallback backend detection
- Louvain community detection, PageRank centrality, degree, clustering, ego subgraph
- cuGraph accelerates on ZeroGPU H200; networkx works on CPU for local dev

**Entity Network tab:**
- Admin-gated (same ADMIN_KEY pattern as Training/Data Workshop)
- Overview sub-tab: Plotly force-directed layout (`spring_layout` with k=1/sqrt(n)), nodes colored by community, sized by PageRank, stats sidebar with top-10 entities
- Explorer sub-tab: entity search + k-hop slider, ego subgraph visualization, connection details table

### Phase 3: Graph-enhanced forecaster

**Feature engineering:**
- `graph_features.py` extracts 7 normalized [0,1] features: graph_density, entity_pagerank, entity_degree, entity_clustering, community_size, avg_neighbor_conf, hub_score
- Features are static per trajectory (graph topology is a snapshot), broadcast as constant columns
- All-zeros when graph data unavailable (graceful degradation)

**Model extension:**
- `INPUT_SIZE_GRAPH = 43` (36 base + 7 graph)
- `load_model()` reads `input_size` from metadata; `_run_inference_core()` introspects `cnn.net[0].in_channels` to auto-select pipeline
- Separate `REFERENCE_SCALER_GRAPH` built at startup for 43-feature normalization

**Training:**
- `get_training_script(graph_features_enabled=True)` generates script with 7 simulated graph features per scenario
- Simulation uses Beta distributions mimicking real graph statistics (sparse density, power-law centrality)
- Training metadata includes `input_size: 43` and `graph_features_enabled: true`
- UI checkbox: "Include graph topology features (43-dim input)"

## Consequences

**Positive:**
- Inference runs on H200 GPU instead of CPU — faster predictions
- Entity network exploration enables interactive investigation of the knowledge graph
- Graph topology signals can improve welfare forecasting when trained on graph-aware data
- Full backward compatibility: 36-feature models work unchanged, graph features degrade to zeros when API is down

**Negative:**
- Space now depends on detective API for graph data (mitigated by cache + graceful degradation)
- cuGraph only available on NVIDIA hardware (mitigated by networkx fallback)
- Graph features are simulated during synthetic training rather than derived from real data (mitigated by Beta distributions matching realistic statistics)

## Files

### New files (space)
- `spaces/maninagarden/graph_client.py` — API bridge with cache
- `spaces/maninagarden/graph_analytics.py` — cuGraph/networkx dual backend
- `spaces/maninagarden/graph_features.py` — 7-feature topology extraction

### Modified files (space)
- `spaces/maninagarden/app.py` — ZeroGPU decorators, Entity Network tab, graph-aware inference
- `spaces/maninagarden/model.py` — CUDA support, variable input_size, INPUT_SIZE_GRAPH
- `spaces/maninagarden/scenarios.py` — compute_all_signals_with_graph()
- `spaces/maninagarden/training.py` — graph_features_enabled flag, simulated features
- `spaces/maninagarden/requirements.txt` — spaces package

### Modified files (API)
- `src/api/routes.py` — /graph/export endpoint, expanded CORS

### New test files
- `tests/api/test_graph_export.py` — 5 tests for graph export endpoint
