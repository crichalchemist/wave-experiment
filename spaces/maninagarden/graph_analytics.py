"""
GPU-accelerated graph analytics with cuGraph primary, networkx fallback.

All public functions accept the raw (nodes, edges) lists from
graph_client.fetch_graph() — no pre-built graph object required.
This keeps the API boundary clean: the caller never needs to know
which backend is active.
"""
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    import cugraph  # noqa: F401
    import cudf  # noqa: F401
    BACKEND = "cugraph"
    logger.info("Graph analytics backend: cuGraph (GPU)")
except ImportError:
    BACKEND = "networkx"
    logger.info("Graph analytics backend: networkx (CPU)")


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------


def _build_networkx(nodes: list[str], edges: list[dict]) -> nx.DiGraph:
    """Build networkx DiGraph from export data."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(
            e["source"], e["target"],
            relation=e["relation"], confidence=e["confidence"],
        )
    return G


def _build_cugraph(nodes: list[str], edges: list[dict]):
    """Build cuGraph graph from export data.

    Returns (graph, node_to_id, id_to_node) since cuGraph uses integer vertex IDs.
    """
    import cudf as _cudf
    import cugraph as _cugraph

    node_to_id = {n: i for i, n in enumerate(nodes)}
    valid = [
        e for e in edges
        if e["source"] in node_to_id and e["target"] in node_to_id
    ]
    src = [node_to_id[e["source"]] for e in valid]
    dst = [node_to_id[e["target"]] for e in valid]
    wgt = [e["confidence"] for e in valid]

    df = _cudf.DataFrame({"src": src, "dst": dst, "weight": wgt})
    G = _cugraph.Graph(directed=True)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
    return G, node_to_id, {v: k for k, v in node_to_id.items()}


# ---------------------------------------------------------------------------
# Public analytics functions
# ---------------------------------------------------------------------------


def detect_communities(nodes: list[str], edges: list[dict]) -> dict[str, int]:
    """Louvain community detection. Returns {node_name: community_id}."""
    if not nodes:
        return {}

    if BACKEND == "cugraph":
        import cugraph as _cugraph
        G, node_to_id, id_to_node = _build_cugraph(nodes, edges)
        parts, _ = _cugraph.louvain(G)
        return {
            id_to_node[int(row["vertex"])]: int(row["partition"])
            for _, row in parts.to_pandas().iterrows()
        }

    G = _build_networkx(nodes, edges)
    communities = nx.community.louvain_communities(G.to_undirected(), seed=42)
    result = {}
    for i, comm in enumerate(communities):
        for node in comm:
            result[node] = i
    return result


def compute_centrality(nodes: list[str], edges: list[dict]) -> dict[str, float]:
    """PageRank centrality. Returns {node_name: pagerank_score}."""
    if not nodes:
        return {}

    if BACKEND == "cugraph":
        import cugraph as _cugraph
        G, node_to_id, id_to_node = _build_cugraph(nodes, edges)
        pr = _cugraph.pagerank(G)
        return {
            id_to_node[int(row["vertex"])]: float(row["pagerank"])
            for _, row in pr.to_pandas().iterrows()
        }

    G = _build_networkx(nodes, edges)
    return nx.pagerank(G)


def compute_degree(nodes: list[str], edges: list[dict]) -> dict[str, int]:
    """Node degree (in + out). Returns {node_name: degree}."""
    if not nodes:
        return {}
    G = _build_networkx(nodes, edges)
    return dict(G.degree())


def compute_clustering(nodes: list[str], edges: list[dict]) -> dict[str, float]:
    """Clustering coefficient on undirected projection. Returns {node_name: coeff}."""
    if not nodes:
        return {}
    G = _build_networkx(nodes, edges).to_undirected()
    return nx.clustering(G)


def ego_subgraph(
    nodes: list[str], edges: list[dict], center: str, hops: int = 2,
) -> dict:
    """Extract ego-centered subgraph for the Explorer view.

    Returns {nodes: [...], edges: [...]} containing only the k-hop
    neighbourhood of the center entity.
    """
    G = _build_networkx(nodes, edges)
    if center not in G:
        return {"nodes": [], "edges": []}
    ego_nodes = set(nx.ego_graph(G, center, radius=hops).nodes())
    ego_edges = [
        e for e in edges
        if e["source"] in ego_nodes and e["target"] in ego_nodes
    ]
    return {"nodes": list(ego_nodes), "edges": ego_edges}
