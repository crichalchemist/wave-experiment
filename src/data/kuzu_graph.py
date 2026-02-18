"""
KuzuGraph: persistent, embedded graph store backed by the Kuzu property-graph database.

Kuzu is chosen over SQLite-backed alternatives because it speaks Cypher natively,
making variable-length path queries (n-hop reasoning) straightforward without custom
graph traversal logic. No server process is required — the database lives in a
local directory, similar to SQLite.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.core.types import KnowledgeEdge, RelationType
from src.data.knowledge_graph import PathResult, _HOP_DECAY

try:
    import kuzu as _kuzu
except ImportError:
    _kuzu = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_NODE_TABLE: str = "Entity"
_EDGE_TABLE: str = "Relationship"
_KUZU_DB_PATH_ENV: str = "DETECTIVE_KUZU_PATH"


# ---------------------------------------------------------------------------
# Schema bootstrap — idempotent
# ---------------------------------------------------------------------------


def _init_schema(conn: object) -> None:
    """
    Create node/rel tables if they do not yet exist.

    Idempotent: safe to call on an already-initialised database. Kuzu supports
    IF NOT EXISTS for both node and rel table DDL as of 0.4+.
    """
    conn.execute(  # type: ignore[union-attr]
        f"CREATE NODE TABLE IF NOT EXISTS {_NODE_TABLE} (id STRING, PRIMARY KEY (id))"
    )
    conn.execute(  # type: ignore[union-attr]
        f"CREATE REL TABLE IF NOT EXISTS {_EDGE_TABLE} "
        f"(FROM {_NODE_TABLE} TO {_NODE_TABLE}, "
        f"relation STRING, confidence DOUBLE)"
    )


# ---------------------------------------------------------------------------
# KuzuGraph — GraphStore-conforming dataclass
# ---------------------------------------------------------------------------


@dataclass
class KuzuGraph:
    """
    Persistent graph store backed by an embedded Kuzu database.

    Not frozen because we hold mutable database/connection handles. The leading
    underscore attributes (_db, _conn) signal that callers must go through the
    GraphStore Protocol methods rather than touching the handles directly.

    db_path is the filesystem directory where Kuzu will store its data files.
    The directory is created by Kuzu on first open; it must not already exist as
    a file.
    """

    db_path: str

    def __post_init__(self) -> None:
        if _kuzu is None:
            raise ImportError(
                "kuzu is required for KuzuGraph. "
                "Install it with: pip install kuzu"
            )
        db = _kuzu.Database(self.db_path)
        conn = _kuzu.Connection(db)
        _init_schema(conn)
        # object.__setattr__ sidesteps any accidental frozen-dataclass guard
        # and makes the mutation intent explicit at the cost of brevity.
        object.__setattr__(self, "_db", db)
        object.__setattr__(self, "_conn", conn)

    # ------------------------------------------------------------------
    # GraphStore Protocol implementation
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        confidence: float,
    ) -> None:
        """
        Upsert source node, target node, and the directed relationship between them.

        MERGE semantics make repeated calls idempotent; re-adding the same edge
        with the same properties is a no-op at the database level.
        """
        self._conn.execute(
            f"MERGE (n:{_NODE_TABLE} {{id: $id}})",
            {"id": source},
        )
        self._conn.execute(
            f"MERGE (n:{_NODE_TABLE} {{id: $id}})",
            {"id": target},
        )
        self._conn.execute(
            f"MATCH (s:{_NODE_TABLE}), (t:{_NODE_TABLE}) "
            f"WHERE s.id = $src AND t.id = $tgt "
            f"MERGE (s)-[r:{_EDGE_TABLE} {{relation: $rel, confidence: $conf}}]->(t)",
            {
                "src": source,
                "tgt": target,
                "rel": relation.value,
                "conf": confidence,
            },
        )

    def get_edge(self, source: str, target: str) -> KnowledgeEdge | None:
        """
        None return is investigatively meaningful — absence of a direct link is data.
        hop_count is always 1 for a direct edge regardless of how it was stored.
        """
        result = self._conn.execute(
            f"MATCH (s:{_NODE_TABLE})-[r:{_EDGE_TABLE}]->(t:{_NODE_TABLE}) "
            f"WHERE s.id = $src AND t.id = $tgt "
            f"RETURN r.relation, r.confidence",
            {"src": source, "tgt": target},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        relation_str: str = row[0]
        confidence: float = row[1]
        return KnowledgeEdge(
            source=source,
            target=target,
            relation=RelationType(relation_str),
            confidence=confidence,
            hop_count=1,
        )

    def successors(self, entity: str) -> list[str]:
        """
        Return immediate successor node IDs (1-hop outgoing neighbours).

        Probes for entity existence first — Kuzu raises on missing nodes rather
        than returning an empty result, so a cheap existence check is cheaper
        than catching the error.
        """
        probe = self._conn.execute(
            f"MATCH (n:{_NODE_TABLE}) WHERE n.id = $id RETURN n.id",
            {"id": entity},
        )
        if not probe.has_next():
            return []
        result = self._conn.execute(
            f"MATCH (s:{_NODE_TABLE})-[:{_EDGE_TABLE}]->(t:{_NODE_TABLE}) "
            f"WHERE s.id = $src RETURN t.id",
            {"src": entity},
        )
        out: list[str] = []
        while result.has_next():
            out.append(result.get_next()[0])
        return out

    def nodes(self) -> list[str]:
        """Return all node IDs currently stored in the database."""
        result = self._conn.execute(f"MATCH (n:{_NODE_TABLE}) RETURN n.id")
        out: list[str] = []
        while result.has_next():
            out.append(result.get_next()[0])
        return out

    def close(self) -> None:
        """Release the Kuzu connection and database handles."""
        self._conn.close()

    def n_hop_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
    ) -> list[PathResult]:
        """
        Return all simple paths from source to target within max_hops, sorted by
        aggregate confidence descending.

        Uses Kuzu's variable-length relationship pattern for efficient graph traversal;
        confidence decays per hop using the same _HOP_DECAY table as InMemoryGraph so
        both backends produce identical numeric results for identical graphs.
        """
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops!r}")

        # Guard: Kuzu raises an error (not returns empty) if the node doesn't exist,
        # so we probe first with a cheap existence check.
        for node_id in (source, target):
            probe = self._conn.execute(
                f"MATCH (n:{_NODE_TABLE}) WHERE n.id = $id RETURN n.id",
                {"id": node_id},
            )
            if not probe.has_next():
                return []

        result = self._conn.execute(
            f"MATCH p=(s:{_NODE_TABLE})-[r:{_EDGE_TABLE}*1..{max_hops}]->(t:{_NODE_TABLE}) "
            f"WHERE s.id = $src AND t.id = $tgt "
            f"RETURN nodes(p), rels(p)",
            {"src": source, "tgt": target},
        )

        paths: list[PathResult] = []
        while result.has_next():
            row = result.get_next()
            nodes: list[dict] = row[0]   # list of node dicts with 'id' key
            rels: list[dict] = row[1]    # list of rel dicts with 'relation', 'confidence'

            node_ids = tuple(n["id"] for n in nodes)

            # Aggregate confidence: product of per-edge (confidence × hop_decay)
            aggregate = 1.0
            for rel in rels:
                rel_type = RelationType(rel["relation"])
                aggregate *= rel["confidence"] * _HOP_DECAY[rel_type]

            paths.append(PathResult(
                path=node_ids,
                confidence=aggregate,
                hops=len(rels),
            ))

        return sorted(paths, key=lambda p: p.confidence, reverse=True)
