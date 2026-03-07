"""InvestigationSource Protocol and adapters wrapping existing infrastructure."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.data.graph_store import GraphStore
from src.detective.investigation.types import (
    DocumentEvidence,
    SourceResult,
)
from src.inference.pipeline import Evidence, parse_intent, retrieve_evidence
from src.security.sanitizer import sanitize_document

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class InvestigationSource(Protocol):
    """Structural protocol for pluggable investigation sources."""

    @property
    def source_id(self) -> str: ...

    def search(self, query: str, max_pages: int = 10) -> SourceResult: ...


# ---------------------------------------------------------------------------
# FOIA adapter
# ---------------------------------------------------------------------------

class FOIAInvestigationSource:
    """Wraps FOIAScraper to satisfy InvestigationSource."""

    def __init__(self, portal: str = "fbi_vault", output_dir: str = "data/foia") -> None:
        from src.data.sourcing.foia_scraper import FOIAScraper

        self._scraper = FOIAScraper(portal=portal, output_dir=output_dir)
        self._portal = portal

    @property
    def source_id(self) -> str:
        return f"foia_{self._portal}"

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        raw_docs = self._scraper.crawl(query=query, max_pages=max_pages)
        ingested = self._scraper.download_and_ingest(raw_docs, max_documents=max_pages)

        documents: list[DocumentEvidence] = []
        injection_findings: list[str] = []

        for doc in ingested:
            result = sanitize_document(doc.text)
            if result.injection_detected:
                injection_findings.extend(result.findings)
            documents.append(
                DocumentEvidence(
                    text=result.safe_text,
                    source_url=doc.url,
                    source_portal=doc.source_portal,
                    title=doc.title,
                    risk_level=result.risk_level,
                )
            )

        return SourceResult(
            lead_id="",  # filled by caller
            documents=tuple(documents),
            pages_consumed=len(ingested),
            injection_findings=tuple(injection_findings),
        )


# ---------------------------------------------------------------------------
# Graph neighbourhood adapter
# ---------------------------------------------------------------------------

class GraphNeighbourhoodSource:
    """Wraps the pipeline's parse_intent + retrieve_evidence for graph queries."""

    def __init__(self, graph: GraphStore) -> None:
        self._graph = graph

    @property
    def source_id(self) -> str:
        return "graph_neighbourhood"

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        intent = parse_intent(query)
        evidence: list[Evidence] = retrieve_evidence(intent, self._graph)

        documents = tuple(
            DocumentEvidence(
                text=ev.content,
                source_url=f"graph://{ev.node_id}",
                source_portal="knowledge_graph",
                title=ev.node_id,
                risk_level="low",
            )
            for ev in evidence[:max_pages]
        )

        return SourceResult(
            lead_id="",
            documents=documents,
            pages_consumed=len(documents),
        )


def build_sources(
    source_ids: tuple[str, ...],
    graph: GraphStore,
) -> dict[str, InvestigationSource]:
    """Construct source adapters from configuration IDs."""
    sources: dict[str, InvestigationSource] = {}

    for sid in source_ids:
        if sid.startswith("foia_"):
            portal = sid.removeprefix("foia_")
            sources[sid] = FOIAInvestigationSource(portal=portal)
        elif sid == "graph_neighbourhood":
            sources[sid] = GraphNeighbourhoodSource(graph=graph)
        else:
            _logger.warning("Unknown source_id %r — skipped", sid)

    return sources
