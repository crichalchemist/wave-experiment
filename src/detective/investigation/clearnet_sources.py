"""
Clearnet investigation sources for the autonomous investigation agent.

Six source adapters covering web search, news, court records, SEC filings,
and investigative journalism organisations. Each implements InvestigationSource
Protocol with rate limiting, sanitization, and injection detection.

All sources target public, open-access data:
  - DuckDuckGo HTML search (web + news tabs)
  - CourtListener REST v4 (Free Law Project, no auth)
  - SEC EDGAR EFTS (public filings, no auth)
  - OCCRP published investigations (open access)
  - IICSA published reports (Crown Copyright, OGL v3)

Dependencies: Scrapling (HTML sources), httpx (REST API sources). Both are
existing project dependencies — no new packages required.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

from src.detective.investigation.types import DocumentEvidence, SourceResult
from src.security.sanitizer import sanitize_document

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded imports — Scrapling optional, httpx always available
# ---------------------------------------------------------------------------

try:
    from scrapling import Fetcher
except ImportError:
    Fetcher = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# DuckDuckGo
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_DDG_RESULT_LINK_SELECTOR = ".result__a"
_DDG_RESULT_SNIPPET_SELECTOR = ".result__snippet"

# CourtListener
_COURTLISTENER_SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"

# SEC EDGAR
_SEC_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
_SEC_DEFAULT_USER_AGENT = "detective-llm research@example.com"
_SEC_FORM_TYPES = ("10-K", "10-Q", "8-K", "DEF 14A")

# OCCRP
_OCCRP_SEARCH_URL = "https://www.occrp.org/en/search"

# IICSA
_IICSA_REPORTS_URL = "https://www.iicsa.org.uk/reports-recommendations"

# Rate limits (seconds between requests)
_RATE_WEB = 2.0
_RATE_NEWS = 2.0
_RATE_COURTLISTENER = 1.0
_RATE_SEC = 0.5
_RATE_OCCRP = 2.0
_RATE_IICSA = 2.0

# Text truncation for evidence
_MAX_EVIDENCE_CHARS = 10_000

# Article body fallback selectors (most specific first)
_ARTICLE_BODY_SELECTORS = ("article", "[role='main']", ".article-body", "main")

# Tags to strip from scraped pages
_STRIP_TAGS = re.compile(
    r"<\s*(script|style|nav|header|footer|aside)[^>]*>.*?</\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_COLLAPSE_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Thread-safe fixed-interval delay between requests."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._last: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last = time.monotonic()


def _httpx_get(url: str, **kwargs: Any) -> Any:
    """Thin wrapper enabling monkeypatching in tests."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("pip install httpx") from e
    return httpx.get(url, timeout=30, follow_redirects=True, **kwargs)


def _extract_text_from_page(page: Any) -> str:
    """Strip scripts/styles/nav and collapse whitespace from a Scrapling page."""
    html = str(page.html) if hasattr(page, "html") else str(page)
    cleaned = _STRIP_TAGS.sub("", html)
    # Strip remaining tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Collapse whitespace
    cleaned = _COLLAPSE_WS.sub(" ", cleaned).strip()
    return cleaned


def _css_first(element: Any, selector: str) -> Any:
    """Return first CSS match or None. Scrapling compatibility helper."""
    results = element.css(selector)
    return results[0] if results else None


def _to_evidence(
    text: str,
    url: str,
    portal: str,
    title: str,
    metadata: dict[str, str] | None = None,
) -> tuple[DocumentEvidence | None, list[str]]:
    """
    Sanitize text and return (DocumentEvidence | None, injection_findings).

    Returns None for empty/whitespace-only text. Truncates to _MAX_EVIDENCE_CHARS.
    """
    text = text.strip()
    if not text:
        return None, []

    text = text[:_MAX_EVIDENCE_CHARS]
    result = sanitize_document(text)

    injection_findings = list(result.findings) if result.injection_detected else []

    meta_tuples = tuple((k, v) for k, v in metadata.items()) if metadata else ()

    evidence = DocumentEvidence(
        text=result.safe_text,
        source_url=url,
        source_portal=portal,
        title=title,
        risk_level=result.risk_level,
        metadata=meta_tuples,
    )
    return evidence, injection_findings


# ---------------------------------------------------------------------------
# Source 1: WebSearchSource (DuckDuckGo HTML)
# ---------------------------------------------------------------------------


class WebSearchSource:
    """General web search via DuckDuckGo HTML interface."""

    def __init__(self, fetch_pages: bool = False) -> None:
        self._fetch_pages = fetch_pages
        self._limiter = _RateLimiter(_RATE_WEB)

    @property
    def source_id(self) -> str:
        return "web_search"

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        if Fetcher is None:
            _logger.warning("Scrapling not installed — web_search unavailable")
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        fetcher = Fetcher()
        self._limiter.wait()

        try:
            page = fetcher.get(_DDG_HTML_URL, params={"q": query})
        except Exception as exc:
            _logger.warning("DuckDuckGo search failed: %s", exc)
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        documents: list[DocumentEvidence] = []
        injection_findings: list[str] = []

        links = page.css(_DDG_RESULT_LINK_SELECTOR)
        snippets = page.css(_DDG_RESULT_SNIPPET_SELECTOR)

        for rank, link in enumerate(links[:max_pages]):
            href = link.attrib.get("href", "")
            title = (link.text or "").strip()
            snippet = ""
            if rank < len(snippets):
                snippet = (snippets[rank].text or "").strip()

            if not href:
                continue

            text = snippet
            if self._fetch_pages and href.startswith("http"):
                self._limiter.wait()
                try:
                    full_page = fetcher.get(href)
                    text = _extract_text_from_page(full_page) or snippet
                except Exception:
                    _logger.debug("Failed to fetch result page: %s", href)

            metadata = {
                "query": query,
                "rank": str(rank),
                "search_engine": "duckduckgo",
            }

            evidence, findings = _to_evidence(
                text, href, "web_search", title or href, metadata
            )
            if evidence:
                documents.append(evidence)
            injection_findings.extend(findings)

        return SourceResult(
            lead_id="",
            documents=tuple(documents),
            pages_consumed=len(documents),
            injection_findings=tuple(injection_findings),
        )


# ---------------------------------------------------------------------------
# Source 2: NewsSearchSource (DuckDuckGo news tab)
# ---------------------------------------------------------------------------


class NewsSearchSource:
    """News search via DuckDuckGo news tab with article fetching."""

    def __init__(self) -> None:
        self._limiter = _RateLimiter(_RATE_NEWS)

    @property
    def source_id(self) -> str:
        return "news_search"

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        if Fetcher is None:
            _logger.warning("Scrapling not installed — news_search unavailable")
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        fetcher = Fetcher()
        self._limiter.wait()

        try:
            page = fetcher.get(_DDG_HTML_URL, params={"q": query, "iar": "news"})
        except Exception as exc:
            _logger.warning("DuckDuckGo news search failed: %s", exc)
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        documents: list[DocumentEvidence] = []
        injection_findings: list[str] = []

        links = page.css(_DDG_RESULT_LINK_SELECTOR)

        for rank, link in enumerate(links[:max_pages]):
            href = link.attrib.get("href", "")
            title = (link.text or "").strip()

            if not href or not href.startswith("http"):
                continue

            # Fetch the actual article page
            self._limiter.wait()
            text = ""
            try:
                article_page = fetcher.get(href)
                # Try fallback selectors for article body
                for selector in _ARTICLE_BODY_SELECTORS:
                    body = _css_first(article_page, selector)
                    if body:
                        text = _extract_text_from_page(body)
                        break
                if not text:
                    text = _extract_text_from_page(article_page)
            except Exception:
                _logger.debug("Failed to fetch news article: %s", href)
                continue

            metadata = {
                "query": query,
                "rank": str(rank),
                "source_type": "news",
            }

            evidence, findings = _to_evidence(
                text, href, "news_search", title or href, metadata
            )
            if evidence:
                documents.append(evidence)
            injection_findings.extend(findings)

        return SourceResult(
            lead_id="",
            documents=tuple(documents),
            pages_consumed=len(documents),
            injection_findings=tuple(injection_findings),
        )


# ---------------------------------------------------------------------------
# Source 3: CourtListenerSource (REST API)
# ---------------------------------------------------------------------------


class CourtListenerSource:
    """Public court records via CourtListener REST v4 API."""

    def __init__(self) -> None:
        self._limiter = _RateLimiter(_RATE_COURTLISTENER)

    @property
    def source_id(self) -> str:
        return "court_listener"

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        self._limiter.wait()

        try:
            response = _httpx_get(
                _COURTLISTENER_SEARCH_URL,
                params={
                    "q": query,
                    "type": "o",
                    "page_size": min(max_pages, 20),
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            _logger.warning("CourtListener search failed: %s", exc)
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        documents: list[DocumentEvidence] = []
        injection_findings: list[str] = []

        for item in data.get("results", [])[:max_pages]:
            text = item.get("plain_text", "").strip()
            if not text:
                text = item.get("snippet", "").strip()
            if not text:
                continue

            case_name = item.get("caseName", item.get("case_name", ""))
            docket_id = str(item.get("docket_id", ""))
            date_filed = item.get("dateFiled", item.get("date_filed", ""))
            court = item.get("court", "")

            url = f"https://www.courtlistener.com/opinion/{docket_id}/"
            metadata = {
                "case_name": case_name,
                "docket_id": docket_id,
                "date_filed": date_filed,
                "court": court,
            }

            evidence, findings = _to_evidence(
                text, url, "court_listener", case_name or query, metadata
            )
            if evidence:
                documents.append(evidence)
            injection_findings.extend(findings)

        return SourceResult(
            lead_id="",
            documents=tuple(documents),
            pages_consumed=len(documents),
            injection_findings=tuple(injection_findings),
        )


# ---------------------------------------------------------------------------
# Source 4: SECEdgarSource (EFTS API)
# ---------------------------------------------------------------------------


class SECEdgarSource:
    """SEC EDGAR full-text search for public filings."""

    def __init__(self, user_agent: str | None = None) -> None:
        self._user_agent = (
            user_agent
            or os.environ.get("SEC_EDGAR_USER_AGENT")
            or _SEC_DEFAULT_USER_AGENT
        )
        self._limiter = _RateLimiter(_RATE_SEC)

    @property
    def source_id(self) -> str:
        return "sec_edgar"

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        self._limiter.wait()

        try:
            response = _httpx_get(
                _SEC_EFTS_URL,
                params={
                    "q": query,
                    "dateRange": "custom",
                    "forms": ",".join(_SEC_FORM_TYPES),
                    "from": "0",
                    "size": str(min(max_pages, 40)),
                },
                headers={"User-Agent": self._user_agent},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            _logger.warning("SEC EDGAR search failed: %s", exc)
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        documents: list[DocumentEvidence] = []
        injection_findings: list[str] = []

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits[:max_pages]:
            source = hit.get("_source", {})
            text = source.get("file_description", "")
            if not text:
                text = source.get("display_names", [""])[0] if source.get("display_names") else ""

            form_type = source.get("form_type", "")
            filing_date = source.get("file_date", "")
            cik = str(source.get("entity_id", ""))
            company = source.get("entity_name", "")
            accession = source.get("file_num", "")

            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
            metadata = {
                "form_type": form_type,
                "filing_date": filing_date,
                "cik": cik,
                "company_name": company,
                "accession_number": accession,
            }

            evidence, findings = _to_evidence(
                text, url, "sec_edgar", f"{company} {form_type}" or query, metadata
            )
            if evidence:
                documents.append(evidence)
            injection_findings.extend(findings)

        return SourceResult(
            lead_id="",
            documents=tuple(documents),
            pages_consumed=len(documents),
            injection_findings=tuple(injection_findings),
        )


# ---------------------------------------------------------------------------
# Source 5: OCCRPSource (Scrapling)
# ---------------------------------------------------------------------------


class OCCRPSource:
    """OCCRP published investigative journalism (open access)."""

    def __init__(self) -> None:
        self._limiter = _RateLimiter(_RATE_OCCRP)

    @property
    def source_id(self) -> str:
        return "web_occrp"

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        if Fetcher is None:
            _logger.warning("Scrapling not installed — web_occrp unavailable")
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        fetcher = Fetcher()
        self._limiter.wait()

        try:
            page = fetcher.get(_OCCRP_SEARCH_URL, params={"q": query})
        except Exception as exc:
            _logger.warning("OCCRP search failed: %s", exc)
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        documents: list[DocumentEvidence] = []
        injection_findings: list[str] = []

        # OCCRP search results are article links
        for rank, link in enumerate(page.css("a.article-link, .story a, a[href*='/en/']")[:max_pages]):
            href = link.attrib.get("href", "")
            title = (link.text or "").strip()

            if not href:
                continue
            if href.startswith("/"):
                href = f"https://www.occrp.org{href}"
            if not href.startswith("http"):
                continue

            # Fetch article body
            self._limiter.wait()
            text = ""
            try:
                article_page = fetcher.get(href)
                for selector in _ARTICLE_BODY_SELECTORS:
                    body = _css_first(article_page, selector)
                    if body:
                        text = _extract_text_from_page(body)
                        break
                if not text:
                    text = _extract_text_from_page(article_page)
            except Exception:
                _logger.debug("Failed to fetch OCCRP article: %s", href)
                continue

            metadata = {
                "query": query,
                "rank": str(rank),
                "source_org": "OCCRP",
            }

            evidence, findings = _to_evidence(
                text, href, "web_occrp", title or href, metadata
            )
            if evidence:
                documents.append(evidence)
            injection_findings.extend(findings)

        return SourceResult(
            lead_id="",
            documents=tuple(documents),
            pages_consumed=len(documents),
            injection_findings=tuple(injection_findings),
        )


# ---------------------------------------------------------------------------
# Source 6: IICSASource (Scrapling)
# ---------------------------------------------------------------------------


class IICSASource:
    """UK IICSA published reports (Crown Copyright, OGL v3)."""

    def __init__(self) -> None:
        self._limiter = _RateLimiter(_RATE_IICSA)

    @property
    def source_id(self) -> str:
        return "web_iicsa"

    def search(self, query: str, max_pages: int = 10) -> SourceResult:
        if Fetcher is None:
            _logger.warning("Scrapling not installed — web_iicsa unavailable")
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        fetcher = Fetcher()
        self._limiter.wait()

        try:
            page = fetcher.get(_IICSA_REPORTS_URL)
        except Exception as exc:
            _logger.warning("IICSA reports fetch failed: %s", exc)
            return SourceResult(lead_id="", documents=(), pages_consumed=0)

        documents: list[DocumentEvidence] = []
        injection_findings: list[str] = []

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for link in page.css("a[href]"):
            title = (link.text or "").strip()
            href = link.attrib.get("href", "")

            if not title or not href:
                continue

            # Filter by query keyword match in title
            title_lower = title.lower()
            if not any(term in title_lower for term in query_terms):
                continue

            if href.startswith("/"):
                href = f"https://www.iicsa.org.uk{href}"
            if not href.startswith("http"):
                continue

            if len(documents) >= max_pages:
                break

            # Fetch report page
            self._limiter.wait()
            text = ""
            try:
                report_page = fetcher.get(href)
                for selector in _ARTICLE_BODY_SELECTORS:
                    body = _css_first(report_page, selector)
                    if body:
                        text = _extract_text_from_page(body)
                        break
                if not text:
                    text = _extract_text_from_page(report_page)
            except Exception:
                _logger.debug("Failed to fetch IICSA report: %s", href)
                continue

            metadata = {
                "report_title": title,
                "source_org": "IICSA",
                "licence": "OGL_v3",
            }

            evidence, findings = _to_evidence(
                text, href, "web_iicsa", title, metadata
            )
            if evidence:
                documents.append(evidence)
            injection_findings.extend(findings)

        return SourceResult(
            lead_id="",
            documents=tuple(documents),
            pages_consumed=len(documents),
            injection_findings=tuple(injection_findings),
        )
