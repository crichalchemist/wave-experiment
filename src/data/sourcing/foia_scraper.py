"""
FOIA portal scraper: crawl FBI Vault, NARA, and State Dept reading rooms.

This module provides a Scrapling-based web scraper that discovers and downloads
declassified FOIA documents, then feeds them into the existing document_ingestion
pipeline for MIME detection, OCR, and redaction analysis.

Portal coverage:
  - FBI Vault (vault.fbi.gov) — declassified FBI investigations
  - NARA (archives.gov) — National Archives declassified collections
  - State Dept (foia.state.gov) — diplomatic cables and embassy reports

All documents are public record released under the Freedom of Information Act.
Metadata is preserved for standpoint transparency (constitution.md).

Usage:
    scraper = FOIAScraper("fbi_vault", output_dir="data/foia")
    docs = scraper.crawl(collection="jeffrey-epstein", max_pages=10)
    ingested = scraper.download_and_ingest(docs, max_documents=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FOIADocument:
    """
    Metadata and text for a single FOIA document.

    Frozen per project convention — evidence artifacts are immutable.
    Use ``dataclasses.replace(doc, text=new_text)`` to create updated copies.
    """

    source_portal: str  # 'fbi_vault', 'nara', 'state_dept'
    title: str
    url: str
    date: str | None
    collection: str | None
    text: str  # extracted text (post-OCR if PDF)
    pdf_path: Path | None  # local path to downloaded PDF


PORTAL_CONFIGS: dict[str, dict[str, Any]] = {
    "fbi_vault": {
        "base_url": "https://vault.fbi.gov",
        "description": "FBI Vault FOIA reading room — declassified investigations",
        "collections_path": "/vault/",
    },
    "nara": {
        "base_url": "https://www.archives.gov",
        "description": "National Archives — declassified government documents",
        "search_path": "/research/catalog/",
    },
    "state_dept": {
        "base_url": "https://foia.state.gov",
        "description": "State Department FOIA — diplomatic cables and embassy reports",
        "reading_room_path": "/Search/Collections",
    },
}


class FOIAScraper:
    """
    Crawl a FOIA portal for declassified documents.

    Dispatches to portal-specific crawlers that use Scrapling for
    JavaScript-rendered pages and pagination handling.

    Args:
        portal: One of the keys in PORTAL_CONFIGS.
        output_dir: Root directory for downloaded PDFs. A portal-specific
            subdirectory is created automatically.
    """

    def __init__(self, portal: str, output_dir: Path | str = "data/foia") -> None:
        if portal not in PORTAL_CONFIGS:
            raise ValueError(
                f"Unknown portal: {portal}. "
                f"Choose from: {list(PORTAL_CONFIGS.keys())}"
            )
        self.portal = portal
        self.config = PORTAL_CONFIGS[portal]
        self.output_dir = Path(output_dir) / portal
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def crawl(
        self,
        collection: str | None = None,
        query: str | None = None,
        max_pages: int = 100,
    ) -> list[FOIADocument]:
        """
        Crawl portal, return document metadata.

        Dispatches to ``_crawl_{portal}`` for portal-specific logic.

        Args:
            collection: Restrict to a named collection (portal-specific).
            query: Free-text search query.
            max_pages: Maximum number of index pages to crawl.

        Returns:
            List of FOIADocument with metadata populated. Text field
            may be empty until ``download_and_ingest`` is called.
        """
        method = getattr(self, f"_crawl_{self.portal}", None)
        if method is None:
            raise NotImplementedError(
                f"Crawler for {self.portal} not yet implemented"
            )
        return method(collection=collection, query=query, max_pages=max_pages)

    # ------------------------------------------------------------------
    # Portal-specific crawlers
    # ------------------------------------------------------------------

    def _crawl_fbi_vault(
        self,
        collection: str | None = None,
        query: str | None = None,
        max_pages: int = 100,
    ) -> list[FOIADocument]:
        """
        FBI Vault crawler using Scrapling Fetcher.

        The Vault organizes documents into collection pages (e.g.,
        ``/vault/jeffrey-epstein/``). Each collection lists multi-part
        PDFs. This crawler:
          1. Fetches the collection index page
          2. Extracts document links and titles
          3. Returns FOIADocument stubs (text empty, pdf_path None)
        """
        try:
            from scrapling import Fetcher
        except ImportError as e:
            raise ImportError(
                "Scrapling not installed. Run: pip install 'detective-llm[scraping]'"
            ) from e

        fetcher = Fetcher()
        base = self.config["base_url"]
        collections_path = self.config.get("collections_path", "/vault/")
        target = f"{base}{collections_path}"
        if collection:
            target = f"{base}{collections_path}{collection}/"

        documents: list[FOIADocument] = []
        pages_crawled = 0

        while target and pages_crawled < max_pages:
            logger.info("Crawling FBI Vault page: %s", target)
            page = fetcher.get(target)

            # Extract document links from the vault index
            for link in page.css("a[href$='.pdf'], a[href*='/vault/']"):
                href = link.attrib.get("href", "")
                title = link.text or href.split("/")[-1]

                if not href:
                    continue

                # Normalize relative URLs
                if href.startswith("/"):
                    href = f"{base}{href}"

                documents.append(
                    FOIADocument(
                        source_portal="fbi_vault",
                        title=title.strip(),
                        url=href,
                        date=None,
                        collection=collection,
                        text="",
                        pdf_path=None,
                    )
                )

            # Pagination: look for a "next" link
            next_link = page.css_first("a.next, a[rel='next']")
            if next_link:
                next_href = next_link.attrib.get("href", "")
                if next_href.startswith("/"):
                    next_href = f"{base}{next_href}"
                target = next_href
            else:
                target = ""  # type: ignore[assignment]

            pages_crawled += 1

        logger.info(
            "FBI Vault crawl complete: %d documents from %d pages",
            len(documents),
            pages_crawled,
        )
        return documents

    def _crawl_nara(
        self,
        collection: str | None = None,
        query: str | None = None,
        max_pages: int = 100,
    ) -> list[FOIADocument]:
        """
        National Archives catalog crawler using Scrapling Fetcher.

        NARA provides a search-based catalog. This crawler:
          1. Constructs a search URL from query/collection
          2. Paginates through search results
          3. Returns FOIADocument stubs for each catalog entry
        """
        try:
            from scrapling import Fetcher
        except ImportError as e:
            raise ImportError(
                "Scrapling not installed. Run: pip install 'detective-llm[scraping]'"
            ) from e

        fetcher = Fetcher()
        base = self.config["base_url"]
        search_path = self.config.get("search_path", "/research/catalog/")
        search_term = query or collection or "declassified"
        target = f"{base}{search_path}?q={search_term}"

        documents: list[FOIADocument] = []
        pages_crawled = 0

        while target and pages_crawled < max_pages:
            logger.info("Crawling NARA page: %s", target)
            page = fetcher.get(target)

            for result in page.css(".result-item, .search-result"):
                title_el = result.css_first("h3, .title, a")
                link_el = result.css_first("a[href]")

                title = title_el.text.strip() if title_el and title_el.text else "Untitled"
                href = link_el.attrib.get("href", "") if link_el else ""

                if href.startswith("/"):
                    href = f"{base}{href}"

                date_el = result.css_first(".date, time")
                date = date_el.text.strip() if date_el and date_el.text else None

                documents.append(
                    FOIADocument(
                        source_portal="nara",
                        title=title,
                        url=href,
                        date=date,
                        collection=collection,
                        text="",
                        pdf_path=None,
                    )
                )

            next_link = page.css_first("a.next, a[rel='next'], .pagination a:last-child")
            if next_link:
                next_href = next_link.attrib.get("href", "")
                if next_href.startswith("/"):
                    next_href = f"{base}{next_href}"
                target = next_href
            else:
                target = ""  # type: ignore[assignment]

            pages_crawled += 1

        logger.info(
            "NARA crawl complete: %d documents from %d pages",
            len(documents),
            pages_crawled,
        )
        return documents

    def _crawl_state_dept(
        self,
        collection: str | None = None,
        query: str | None = None,
        max_pages: int = 100,
    ) -> list[FOIADocument]:
        """
        State Department FOIA reading room crawler using Scrapling Fetcher.

        The State Dept FOIA site organizes documents by collections.
        This crawler:
          1. Navigates to the reading room search/collections page
          2. Extracts document listings with dates and titles
          3. Returns FOIADocument stubs
        """
        try:
            from scrapling import Fetcher
        except ImportError as e:
            raise ImportError(
                "Scrapling not installed. Run: pip install 'detective-llm[scraping]'"
            ) from e

        fetcher = Fetcher()
        base = self.config["base_url"]
        reading_room_path = self.config.get("reading_room_path", "/Search/Collections")
        target = f"{base}{reading_room_path}"
        if query:
            target = f"{base}/Search/Results?query={query}"

        documents: list[FOIADocument] = []
        pages_crawled = 0

        while target and pages_crawled < max_pages:
            logger.info("Crawling State Dept page: %s", target)
            page = fetcher.get(target)

            for row in page.css(".document-row, .search-result, tr.result"):
                title_el = row.css_first("a, .doc-title, td:first-child")
                link_el = row.css_first("a[href]")

                title = title_el.text.strip() if title_el and title_el.text else "Untitled"
                href = link_el.attrib.get("href", "") if link_el else ""

                if href.startswith("/"):
                    href = f"{base}{href}"

                date_el = row.css_first(".doc-date, td.date, time")
                date = date_el.text.strip() if date_el and date_el.text else None

                coll_el = row.css_first(".collection, td.collection")
                coll = coll_el.text.strip() if coll_el and coll_el.text else collection

                documents.append(
                    FOIADocument(
                        source_portal="state_dept",
                        title=title,
                        url=href,
                        date=date,
                        collection=coll,
                        text="",
                        pdf_path=None,
                    )
                )

            next_link = page.css_first("a.next, a[rel='next'], .pagination a:last-child")
            if next_link:
                next_href = next_link.attrib.get("href", "")
                if next_href.startswith("/"):
                    next_href = f"{base}{next_href}"
                target = next_href
            else:
                target = ""  # type: ignore[assignment]

            pages_crawled += 1

        logger.info(
            "State Dept crawl complete: %d documents from %d pages",
            len(documents),
            pages_crawled,
        )
        return documents

    # ------------------------------------------------------------------
    # Download + OCR ingestion
    # ------------------------------------------------------------------

    def download_and_ingest(
        self,
        documents: list[FOIADocument],
        max_documents: int | None = None,
    ) -> list[FOIADocument]:
        """
        Download PDFs and run through the document_ingestion OCR pipeline.

        For each document whose URL ends with ``.pdf``, downloads the file
        to ``self.output_dir`` and calls ``ingest_document`` to produce OCR
        text. Returns new FOIADocument instances with ``text`` and ``pdf_path``
        populated.

        Args:
            documents: FOIADocument stubs from ``crawl()``.
            max_documents: Limit the number of documents to download.

        Returns:
            Updated FOIADocument list with text extracted from PDFs.
        """
        from src.data.sourcing.document_ingestion import ingest_document

        try:
            import httpx
        except ImportError as e:
            raise ImportError("pip install httpx") from e

        to_process = documents[:max_documents] if max_documents else documents
        ingested: list[FOIADocument] = []

        for doc in to_process:
            if not doc.url.endswith(".pdf"):
                ingested.append(doc)
                continue

            # Derive a safe filename
            filename = doc.url.split("/")[-1]
            if not filename.endswith(".pdf"):
                filename = f"{hash(doc.url) & 0xFFFFFFFF:08x}.pdf"
            local_path = self.output_dir / filename

            try:
                logger.info("Downloading %s -> %s", doc.url, local_path)
                response = httpx.get(doc.url, timeout=60, follow_redirects=True)
                response.raise_for_status()
                local_path.write_bytes(response.content)

                record = ingest_document(local_path, source_id=doc.source_portal)
                ingested.append(
                    replace(
                        doc,
                        text=record.text,
                        pdf_path=local_path,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to download/ingest %s: %s", doc.url, exc)
                ingested.append(doc)

        return ingested
