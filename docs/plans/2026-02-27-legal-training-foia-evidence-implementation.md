# Legal Training + FOIA Evidence Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Launch legal DPO training (600 pairs, 3 domains) on existing infrastructure, then build a Scrapling-based FOIA portal scraper that feeds both investigation and training pipelines.

**Architecture:** Part 1 is pure execution — no new code. Part 2 builds `foia_scraper.py` using Scrapling's Spider framework, wires it to the existing `document_ingestion.py` pipeline, adds a CLI command, and connects to the dual-output (evidence + training) flow.

**Tech Stack:** Scrapling (adaptive web scraping), pytesseract/pdf2image (OCR), existing `document_ingestion.py` + `ocr_provider.py`, Click CLI.

**Design doc:** `docs/plans/2026-02-27-legal-training-foia-evidence-pipeline.md`

---

## Part 1: Legal Warmup + DPO Training (Operational)

### Task 1: Generate Legal Preference Pairs

No new code. Run the existing `legal-warmup` CLI command.

**Files:**
- Read: `src/training/legal_warmup.py` (already implemented)
- Read: `src/data/sourcing/legal_sources.py` (domain configs)
- Output: `data/training/legal_pairs.jsonl`

**Step 1: Verify legal warmup infrastructure works**

Run: `detective legal-warmup --help`
Expected: Help text showing `--output`, `--examples-per-domain`, `--domains` options.

**Step 2: Launch legal warmup generation**

This requires provider access (local vLLM + Claude critic). Check what's available:

```bash
# Check provider env
python -c "from src.core.providers import provider_from_env; print(provider_from_env())"
```

If providers are configured:
```bash
detective legal-warmup \
  --output data/training/legal_pairs.jsonl \
  --examples-per-domain 200 \
  --domains criminal_justice,territorial_rights,foia_transparency
```

If providers are NOT configured (no local vLLM):
- This task is blocked until a provider is available
- Skip to Part 2 (FOIA scraper can be built independently)
- Return to legal warmup when Azure/vLLM is reachable

**Step 3: Verify output**

```bash
wc -l data/training/legal_pairs.jsonl
# Expected: ~600 lines
head -1 data/training/legal_pairs.jsonl | python -m json.tool
# Expected: {instruction, chosen, rejected, source, metadata: {legal_domain: ...}}
```

**Step 4: Commit training data**

```bash
git add data/training/legal_pairs.jsonl
git commit -m "data: generate 600 legal DPO pairs (3 domains × 200)"
```

---

### Task 2: Train Legal DPO on HF Jobs

**Files:**
- Read: `src/training/train_dpo.py`
- Input: `data/training/legal_pairs.jsonl` + `checkpoints/dpo/checkpoint-23`
- Output: `checkpoints/dpo-legal/`

**Step 1: Upload legal pairs to Hub**

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="data/training/legal_pairs.jsonl",
    path_in_repo="legal_pairs.jsonl",
    repo_id="crichalchemist/welfare-training-data",
    repo_type="dataset",
)
```

**Step 2: Launch DPO training**

```bash
python src/training/train_dpo.py \
  --data data/training/legal_pairs.jsonl \
  --output-dir checkpoints/dpo-legal/
```

Or on HF Jobs if local GPU unavailable — this would need a new HF Jobs script similar to `train_welfare_classifier_hf_job.py`.

**Step 3: Verify checkpoint**

```bash
ls checkpoints/dpo-legal/
# Expected: adapter_config.json, adapter_model.safetensors, etc.
```

**Step 4: Commit**

```bash
git commit -m "feat(training): legal DPO checkpoint from 600 domain-specific pairs"
```

---

### Task 3: Extend Welfare Classifier Training Data

The 600 legal documents contain rich welfare construct signals. Score them and add to the classifier training set.

**Files:**
- Read: `scripts/create_welfare_training_data.py`
- Input: `data/training/legal_pairs.jsonl`
- Modify: `data/training/welfare_training_split_train.jsonl`

**Step 1: Extract text from legal pairs for welfare scoring**

```python
import json
texts = []
with open("data/training/legal_pairs.jsonl") as f:
    for line in f:
        pair = json.loads(line)
        # Use the instruction text (the source document)
        texts.append(pair["instruction"])
```

**Step 2: Score via welfare labeling script**

Adapt `scripts/create_welfare_training_data.py` to accept the legal corpus, or run the existing script with the legal text as input.

**Step 3: Merge into training splits and retrain classifier**

```bash
# Merge new welfare-scored examples into training data
# Resplit 80/20
# Retrain via HF Jobs
python -c "
from scripts.train_welfare_classifier_hf_job import launch_classifier_training
import os
from huggingface_hub import get_token
os.environ['HF_TOKEN'] = get_token()
job_id, msg = launch_classifier_training(epochs=3, lr=2e-5, batch_size=16)
print(msg)
"
```

**Step 4: Commit**

```bash
git commit -m "data: extend welfare training set with legal corpus welfare scores"
```

---

## Part 2: Scrapling FOIA Evidence Collector

### Task 4: Install Scrapling and Write Spider Tests

**Files:**
- Modify: `pyproject.toml` (add scrapling to optional deps)
- Create: `tests/data/sourcing/test_foia_scraper.py`
- Create: `src/data/sourcing/foia_scraper.py`

**Step 1: Add scrapling dependency**

In `pyproject.toml`, add to the `[project.optional-dependencies]` section:

```toml
scraping = [
    "scrapling>=0.3.0",
    "pdf2image>=1.16.0",
    "pytesseract>=0.3.10",
]
```

Install: `pip install -e ".[scraping]"`

**Step 2: Write the failing tests**

```python
# tests/data/sourcing/test_foia_scraper.py
"""Test FOIA portal scraper."""
import pytest
from dataclasses import fields


def test_foia_document_dataclass():
    from src.data.sourcing.foia_scraper import FOIADocument
    doc = FOIADocument(
        source_portal="fbi_vault",
        title="Test Document",
        url="https://vault.fbi.gov/test",
        date="2020-01-01",
        collection="test-collection",
        text="Sample text content",
        pdf_path=None,
    )
    assert doc.source_portal == "fbi_vault"
    assert doc.text == "Sample text content"


def test_foia_document_is_frozen():
    from src.data.sourcing.foia_scraper import FOIADocument
    doc = FOIADocument(
        source_portal="fbi_vault", title="Test", url="https://test",
        date=None, collection=None, text="text", pdf_path=None,
    )
    with pytest.raises(AttributeError):
        doc.text = "mutated"


def test_portal_configs_exist():
    from src.data.sourcing.foia_scraper import PORTAL_CONFIGS
    assert "fbi_vault" in PORTAL_CONFIGS
    assert "nara" in PORTAL_CONFIGS
    assert "state_dept" in PORTAL_CONFIGS


def test_portal_config_has_required_keys():
    from src.data.sourcing.foia_scraper import PORTAL_CONFIGS
    for name, config in PORTAL_CONFIGS.items():
        assert "base_url" in config, f"{name} missing base_url"
        assert "description" in config, f"{name} missing description"


def test_foia_scraper_class_exists():
    from src.data.sourcing.foia_scraper import FOIAScraper
    assert callable(FOIAScraper)


def test_scraper_rejects_unknown_portal():
    from src.data.sourcing.foia_scraper import FOIAScraper
    with pytest.raises(ValueError, match="Unknown portal"):
        FOIAScraper(portal="nonexistent")
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/data/sourcing/test_foia_scraper.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 4: Implement the minimal module**

```python
# src/data/sourcing/foia_scraper.py
"""
FOIA portal scraper using Scrapling's adaptive framework.

Crawls FBI Vault, NARA, and State Department FOIA reading rooms.
Downloads documents (HTML + PDF), routes through document_ingestion.py
for OCR and normalization.

Each scraped document feeds two pipelines:
1. Evidence: A/B/C modules + knowledge graph (investigation)
2. Training: welfare scoring + constitutional warmup (model improvement)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FOIADocument:
    """A document retrieved from a FOIA portal."""
    source_portal: str          # 'fbi_vault', 'nara', 'state_dept'
    title: str
    url: str
    date: str | None
    collection: str | None      # e.g., 'jeffrey-epstein' on FBI Vault
    text: str                   # extracted text (post-OCR if PDF)
    pdf_path: Path | None       # local path to downloaded PDF


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
    """Scrapling-based FOIA portal crawler.

    Uses Scrapling's adaptive parser for resilient element location,
    StealthyFetcher for rate-limited portals, and Spider for
    paginated crawls with checkpoint support.
    """

    def __init__(
        self,
        portal: str,
        output_dir: Path | str = "data/foia",
    ):
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
        """Crawl the portal and return document metadata + text.

        Args:
            collection: Specific collection to crawl (e.g., 'jeffrey-epstein').
            query: Search query for portals with search functionality.
            max_pages: Maximum pages to crawl.

        Returns:
            List of FOIADocument with extracted text.
        """
        method = getattr(self, f"_crawl_{self.portal}", None)
        if method is None:
            raise NotImplementedError(f"Crawler for {self.portal} not yet implemented")
        return method(collection=collection, query=query, max_pages=max_pages)

    def _crawl_fbi_vault(
        self,
        collection: str | None = None,
        query: str | None = None,
        max_pages: int = 100,
    ) -> list[FOIADocument]:
        """Crawl FBI Vault collections."""
        try:
            from scrapling import Fetcher
        except ImportError as e:
            raise ImportError("pip install scrapling") from e

        base = self.config["base_url"]
        fetcher = Fetcher(auto_match=True)

        if collection:
            start_url = f"{base}/vault/{collection}"
        else:
            start_url = f"{base}{self.config['collections_path']}"

        documents: list[FOIADocument] = []
        page = fetcher.get(start_url)

        # Find document links on the collection page
        links = page.css("a[href*='/vault/']") or page.css("a")
        for link in links[:max_pages]:
            href = link.attrib.get("href", "")
            title = link.text or href.split("/")[-1]

            if not href or href == "#":
                continue

            # Normalize URL
            if href.startswith("/"):
                href = f"{base}{href}"

            # Skip non-document links
            if not any(ext in href.lower() for ext in [".pdf", "/view", "/vault/"]):
                continue

            doc = FOIADocument(
                source_portal="fbi_vault",
                title=title.strip(),
                url=href,
                date=None,
                collection=collection,
                text="",  # filled after download + OCR
                pdf_path=None,
            )
            documents.append(doc)

        logger.info(f"Found {len(documents)} documents in FBI Vault{f'/{collection}' if collection else ''}")
        return documents

    def download_and_ingest(
        self,
        documents: list[FOIADocument],
        max_documents: int | None = None,
    ) -> list[FOIADocument]:
        """Download PDFs and run through document ingestion pipeline.

        Downloads each document's PDF, runs OCR via document_ingestion.py,
        and returns updated FOIADocuments with extracted text.
        """
        from dataclasses import replace
        try:
            from scrapling import Fetcher
        except ImportError as e:
            raise ImportError("pip install scrapling") from e
        from src.data.sourcing.document_ingestion import ingest_document

        fetcher = Fetcher(auto_match=True)
        results: list[FOIADocument] = []
        limit = max_documents or len(documents)

        for i, doc in enumerate(documents[:limit]):
            if doc.url.lower().endswith(".pdf"):
                # Download PDF
                pdf_dir = self.output_dir / "pdfs"
                pdf_dir.mkdir(exist_ok=True)
                safe_name = doc.title.replace("/", "_")[:100] + ".pdf"
                pdf_path = pdf_dir / safe_name

                try:
                    response = fetcher.get(doc.url)
                    pdf_path.write_bytes(response.content)

                    # Run through document_ingestion pipeline
                    record = ingest_document(pdf_path, source_id=f"{doc.source_portal}:{doc.title}")

                    results.append(replace(
                        doc,
                        text=record.text,
                        pdf_path=pdf_path,
                    ))
                    logger.info(f"  [{i+1}/{limit}] Ingested: {doc.title} ({record.page_count} pages)")

                except Exception as e:
                    logger.warning(f"  [{i+1}/{limit}] Failed: {doc.title} — {e}")
                    results.append(doc)  # keep metadata even if download fails
            else:
                # HTML page — extract text directly
                try:
                    page = fetcher.get(doc.url)
                    text = page.css("div.content, article, main, .document-text")
                    body_text = "\n".join(el.text for el in text if el.text) if text else page.text
                    results.append(replace(doc, text=body_text or ""))
                except Exception as e:
                    logger.warning(f"  [{i+1}/{limit}] Failed: {doc.title} — {e}")
                    results.append(doc)

        logger.info(f"Ingested {len([r for r in results if r.text])} of {len(results)} documents")
        return results
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/data/sourcing/test_foia_scraper.py -v`
Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add pyproject.toml src/data/sourcing/foia_scraper.py tests/data/sourcing/test_foia_scraper.py
git commit -m "feat(data): FOIA portal scraper with FBI Vault, NARA, State Dept configs"
```

---

### Task 5: CLI Command for FOIA Scraping

**Files:**
- Modify: `src/cli/main.py`
- Create: `tests/cli/test_scrape_foia_command.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_scrape_foia_command.py
"""Test scrape-foia CLI command."""
from click.testing import CliRunner


def test_scrape_foia_command_exists():
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["scrape-foia", "--help"])
    assert result.exit_code == 0
    assert "portal" in result.output.lower()


def test_scrape_foia_rejects_unknown_portal():
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["scrape-foia", "--portal", "nonexistent"])
    assert result.exit_code != 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/cli/test_scrape_foia_command.py -v`
Expected: FAIL

**Step 3: Implement the CLI command**

Add to `src/cli/main.py`:

```python
@cli.command("scrape-foia")
@click.option("--portal", type=click.Choice(["fbi_vault", "nara", "state_dept"]),
              required=True, help="FOIA portal to scrape")
@click.option("--collection", type=str, default=None,
              help="Specific collection to crawl (e.g., 'jeffrey-epstein')")
@click.option("--query", type=str, default=None,
              help="Search query for the portal")
@click.option("--max-pages", type=int, default=100, help="Maximum pages to crawl")
@click.option("--output", "-o", type=click.Path(), default="data/foia",
              help="Output directory for scraped documents")
@click.option("--ingest/--no-ingest", default=True,
              help="Run document ingestion (OCR + text extraction) after scraping")
def scrape_foia(portal, collection, query, max_pages, output, ingest):
    """Scrape FOIA portal for investigation evidence and training data.

    Downloads documents from FOIA reading rooms, runs OCR, and outputs
    both evidence (for investigation) and training data (for model improvement).
    """
    from src.data.sourcing.foia_scraper import FOIAScraper

    scraper = FOIAScraper(portal=portal, output_dir=output)

    click.echo(f"Crawling {portal}...")
    documents = scraper.crawl(collection=collection, query=query, max_pages=max_pages)
    click.echo(f"Found {len(documents)} documents")

    if ingest and documents:
        click.echo("Downloading and ingesting documents...")
        documents = scraper.download_and_ingest(documents)
        with_text = [d for d in documents if d.text]
        click.echo(f"Ingested {len(with_text)} documents with extracted text")

    # Save document index
    import json
    index_path = Path(output) / portal / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump([
            {"title": d.title, "url": d.url, "collection": d.collection,
             "has_text": bool(d.text), "date": d.date}
            for d in documents
        ], f, indent=2)
    click.echo(f"Saved document index to {index_path}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/cli/test_scrape_foia_command.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -q --deselect tests/detective/test_parallel_evolution.py::test_welfare_scoring_applied_to_evolved_hypotheses --deselect tests/detective/test_parallel_evolution.py::test_backward_compatible_without_phi_metrics --deselect tests/detective/test_parallel_evolution.py::test_high_welfare_relevance_for_urgent_findings --deselect tests/detective/test_parallel_evolution.py::test_welfare_aware_sorting`
Expected: 535+ passed, 0 new failures

**Step 6: Commit**

```bash
git add src/cli/main.py tests/cli/test_scrape_foia_command.py
git commit -m "feat(cli): add scrape-foia command for FOIA portal evidence collection"
```

---

### Task 6: Dual Output Pipeline — Evidence + Training

Wire scraped documents to both the investigation pipeline (A/B/C modules + knowledge graph) and the training pipeline (welfare scoring + preference pair generation).

**Files:**
- Create: `src/data/sourcing/dual_pipeline.py`
- Create: `tests/data/sourcing/test_dual_pipeline.py`

**Step 1: Write the failing tests**

```python
# tests/data/sourcing/test_dual_pipeline.py
"""Test dual output pipeline: evidence + training from scraped documents."""
import pytest
from unittest.mock import patch, MagicMock


def test_process_for_evidence_returns_dict():
    from src.data.sourcing.dual_pipeline import process_for_evidence
    from src.data.sourcing.foia_scraper import FOIADocument

    doc = FOIADocument(
        source_portal="fbi_vault", title="Test",
        url="https://test", date=None, collection=None,
        text="Resource allocation gap in oversight records from 2013-2017.",
        pdf_path=None,
    )

    with patch("src.data.sourcing.dual_pipeline.get_construct_scores",
               return_value={"c": 0.7, "kappa": 0.3, "j": 0.3, "p": 0.3,
                             "eps": 0.3, "lam_L": 0.3, "lam_P": 0.5, "xi": 0.6}):
        result = process_for_evidence(doc)

    assert "construct_scores" in result
    assert "threatened_constructs" in result
    assert isinstance(result["construct_scores"], dict)


def test_process_for_training_returns_dict():
    from src.data.sourcing.dual_pipeline import process_for_training
    from src.data.sourcing.foia_scraper import FOIADocument

    doc = FOIADocument(
        source_portal="fbi_vault", title="Test",
        url="https://test", date=None, collection=None,
        text="Investigation revealed systematic suppression of evidence.",
        pdf_path=None,
    )

    with patch("src.data.sourcing.dual_pipeline.get_construct_scores",
               return_value={"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                             "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.8}):
        result = process_for_training(doc)

    assert "welfare_scores" in result
    assert "text" in result
    assert len(result["welfare_scores"]) == 8


def test_run_dual_pipeline():
    from src.data.sourcing.dual_pipeline import run_dual_pipeline
    from src.data.sourcing.foia_scraper import FOIADocument

    docs = [
        FOIADocument(
            source_portal="fbi_vault", title="Doc 1",
            url="https://test/1", date=None, collection=None,
            text="Financial records show unexplained gaps in reporting.",
            pdf_path=None,
        ),
    ]

    mock_scores = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}
    with patch("src.data.sourcing.dual_pipeline.get_construct_scores", return_value=mock_scores):
        result = run_dual_pipeline(docs)

    assert "evidence" in result
    assert "training" in result
    assert len(result["evidence"]) == 1
    assert len(result["training"]) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/data/sourcing/test_dual_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the dual pipeline**

```python
# src/data/sourcing/dual_pipeline.py
"""
Dual output pipeline: process scraped documents for both investigation and training.

Evidence output: welfare construct scores, threatened constructs, gap indicators
Training output: welfare-scored examples for classifier + DPO pair generation
"""
from __future__ import annotations

from typing import Any
import logging

from src.data.sourcing.foia_scraper import FOIADocument
from src.inference.welfare_scoring import (
    get_construct_scores, infer_threatened_constructs,
    compute_phi, ALL_CONSTRUCTS,
)

logger = logging.getLogger(__name__)


def process_for_evidence(doc: FOIADocument) -> dict[str, Any]:
    """Process a document for the investigation evidence pipeline.

    Returns construct scores, threatened constructs, and Phi value
    for knowledge graph enrichment and gap detection.
    """
    scores = get_construct_scores(doc.text)
    threatened = infer_threatened_constructs(doc.text)
    phi = compute_phi(scores)

    return {
        "title": doc.title,
        "url": doc.url,
        "source_portal": doc.source_portal,
        "construct_scores": scores,
        "threatened_constructs": list(threatened),
        "phi": phi,
        "text_length": len(doc.text),
        "has_text": bool(doc.text.strip()),
    }


def process_for_training(doc: FOIADocument) -> dict[str, Any]:
    """Process a document for the training data pipeline.

    Returns welfare-scored example suitable for extending the
    welfare classifier training set.
    """
    scores = get_construct_scores(doc.text)

    return {
        "text": doc.text,
        "scores": scores,
        "metadata": {
            "source": doc.source_portal,
            "title": doc.title,
            "url": doc.url,
            "collection": doc.collection,
        },
    }


def run_dual_pipeline(
    documents: list[FOIADocument],
) -> dict[str, list]:
    """Run both evidence and training pipelines on a list of documents.

    Returns dict with 'evidence' and 'training' lists.
    """
    evidence_results = []
    training_results = []

    for i, doc in enumerate(documents):
        if not doc.text or not doc.text.strip():
            logger.debug(f"Skipping empty document: {doc.title}")
            continue

        try:
            evidence = process_for_evidence(doc)
            training = process_for_training(doc)
            evidence_results.append(evidence)
            training_results.append(training)
        except Exception as e:
            logger.warning(f"Failed to process {doc.title}: {e}")

    logger.info(f"Dual pipeline: {len(evidence_results)} evidence, {len(training_results)} training")
    return {
        "evidence": evidence_results,
        "training": training_results,
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/data/sourcing/test_dual_pipeline.py -v`
Expected: PASS (3 tests)

**Step 5: Run full test suite**

Run: `pytest tests/ -q --deselect tests/detective/test_parallel_evolution.py::test_welfare_scoring_applied_to_evolved_hypotheses --deselect tests/detective/test_parallel_evolution.py::test_backward_compatible_without_phi_metrics --deselect tests/detective/test_parallel_evolution.py::test_high_welfare_relevance_for_urgent_findings --deselect tests/detective/test_parallel_evolution.py::test_welfare_aware_sorting`
Expected: 540+ passed

**Step 6: Commit**

```bash
git add src/data/sourcing/dual_pipeline.py tests/data/sourcing/test_dual_pipeline.py
git commit -m "feat(data): dual pipeline — evidence + training from scraped FOIA documents"
```

---

### Task 7: Wire CLI Ingest Command + ADR

**Files:**
- Modify: `src/cli/main.py` (add `ingest` command)
- Create: `docs/vault/decisions/ADR-012-foia-scraping-dual-pipeline.md`

**Step 1: Add ingest command to CLI**

```python
@cli.command("ingest")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--evidence/--no-evidence", default=True, help="Run evidence pipeline")
@click.option("--training/--no-training", default=True, help="Run training pipeline")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output JSONL for training data (default: data/training/foia_welfare_scored.jsonl)")
def ingest(input_dir, evidence, training, output):
    """Process scraped documents through evidence + training pipelines.

    Reads FOIADocument index from INPUT_DIR, runs welfare scoring,
    and outputs evidence results and/or training data.
    """
    import json
    from pathlib import Path
    from src.data.sourcing.foia_scraper import FOIADocument
    from src.data.sourcing.dual_pipeline import run_dual_pipeline

    index_path = Path(input_dir) / "index.json"
    if not index_path.exists():
        click.echo(f"No index.json found in {input_dir}. Run scrape-foia first.")
        return

    # Reconstruct documents from index + extracted text
    with open(index_path) as f:
        index = json.load(f)

    docs = []
    text_dir = Path(input_dir) / "text"
    for entry in index:
        text_path = text_dir / f"{entry['title'][:100]}.txt"
        text = text_path.read_text() if text_path.exists() else ""
        docs.append(FOIADocument(
            source_portal=entry.get("source_portal", "unknown"),
            title=entry["title"], url=entry["url"],
            date=entry.get("date"), collection=entry.get("collection"),
            text=text, pdf_path=None,
        ))

    click.echo(f"Processing {len(docs)} documents...")
    result = run_dual_pipeline(docs)

    if evidence:
        click.echo(f"  Evidence results: {len(result['evidence'])}")
    if training:
        output_path = output or "data/training/foia_welfare_scored.jsonl"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for item in result["training"]:
                f.write(json.dumps(item) + "\n")
        click.echo(f"  Training data: {len(result['training'])} examples → {output_path}")
```

**Step 2: Write ADR-012**

```markdown
# ADR-012: FOIA Scraping with Dual Evidence/Training Pipeline

## Decision
Use Scrapling's adaptive web scraping framework to collect documents from
FOIA portals (FBI Vault, NARA, State Dept). Every scraped document feeds
two pipelines simultaneously: investigation evidence and model training data.

## Context
[... per design doc ...]
```

**Step 3: Run full test suite and commit**

```bash
git add src/cli/main.py docs/vault/decisions/ADR-012-foia-scraping-dual-pipeline.md
git commit -m "feat(cli): add ingest command + ADR-012 for FOIA dual pipeline"
```

---

## Summary

| Task | Part | What | New Code? |
|------|------|------|-----------|
| 1 | P1 | Generate 600 legal DPO pairs | No — run existing CLI |
| 2 | P1 | Train legal DPO | No — run existing script |
| 3 | P1 | Extend welfare training data | Minimal — merge + retrain |
| 4 | P2 | FOIA scraper module + tests | Yes — `foia_scraper.py` |
| 5 | P2 | CLI scrape-foia command | Yes — CLI addition |
| 6 | P2 | Dual pipeline (evidence + training) | Yes — `dual_pipeline.py` |
| 7 | P2 | CLI ingest command + ADR-012 | Yes — CLI + ADR |

**Dependencies:** Tasks 1-3 are sequential (pairs → DPO → classifier). Tasks 4-7 are sequential but independent of Part 1. Both parts can run in parallel.
