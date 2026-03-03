"""
Extract 7 graph topology features from the knowledge graph for welfare forecasting.

All features are normalized to [0, 1] for compatibility with the existing
RobustScaler pipeline. When graph data is unavailable or the focal entity
is not in the graph, returns all zeros — the model learns to treat zeros
as "no graph signal" (graceful degradation).
"""
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)

GRAPH_FEATURE_NAMES = (
    "graph_density",
    "entity_pagerank",
    "entity_degree",
    "entity_clustering",
    "community_size",
    "avg_neighbor_conf",
    "hub_score",
)


def _zeros() -> dict[str, float]:
    """Return all-zero feature dict (no graph signal)."""
    return {k: 0.0 for k in GRAPH_FEATURE_NAMES}


def extract_graph_features(
    graph_data: dict,
    focal_entity: str | None = None,
) -> dict[str, float]:
    """Extract 7 graph topology features.

    Parameters
    ----------
    graph_data : dict
        Output from graph_client.fetch_graph(): {nodes, edges, stats}.
    focal_entity : str or None
        Entity to compute ego-centric features for. When None, ego features
        are zero and only global features (density) are computed.

    Returns
    -------
    dict[str, float]
        Keys match GRAPH_FEATURE_NAMES. All values in [0, 1].
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        return _zeros()

    # Build networkx graph once for all computations
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(
            e["source"], e["target"],
            confidence=e.get("confidence", 0.5),
        )

    n = len(nodes)
    features = _zeros()

    # --- Global features ---
    max_possible_edges = n * (n - 1) if n > 1 else 1
    features["graph_density"] = min(1.0, len(edges) / max_possible_edges)

    # Early return if no focal entity or entity not in graph
    if not focal_entity or focal_entity not in G:
        return features

    # --- Ego-centric features ---

    # PageRank (already normalized to sum=1, but individual values are small)
    pagerank = nx.pagerank(G)
    max_pr = max(pagerank.values()) if pagerank else 1.0
    features["entity_pagerank"] = pagerank.get(focal_entity, 0.0) / max(max_pr, 1e-9)

    # Degree (normalized by max degree)
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    features["entity_degree"] = degrees.get(focal_entity, 0) / max(max_deg, 1)

    # Clustering coefficient (on undirected projection)
    G_undirected = G.to_undirected()
    features["entity_clustering"] = nx.clustering(G_undirected, focal_entity)

    # Community size (fraction of total nodes in the same community)
    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
        for comm in communities:
            if focal_entity in comm:
                features["community_size"] = len(comm) / n
                break
    except Exception:
        pass  # community detection can fail on degenerate graphs

    # Average neighbour confidence
    neighbors = list(G.successors(focal_entity)) + list(G.predecessors(focal_entity))
    if neighbors:
        confs = []
        for nb in neighbors:
            if G.has_edge(focal_entity, nb):
                confs.append(G[focal_entity][nb].get("confidence", 0.5))
            if G.has_edge(nb, focal_entity):
                confs.append(G[nb][focal_entity].get("confidence", 0.5))
        features["avg_neighbor_conf"] = sum(confs) / len(confs) if confs else 0.0

    # HITS hub score (normalized)
    try:
        hubs, _ = nx.hits(G, max_iter=100, tol=1e-6)
        max_hub = max(hubs.values()) if hubs else 1.0
        features["hub_score"] = hubs.get(focal_entity, 0.0) / max(max_hub, 1e-9)
    except nx.PowerIterationFailedConvergence:
        pass  # degenerate graph, leave as 0.0

    return features
