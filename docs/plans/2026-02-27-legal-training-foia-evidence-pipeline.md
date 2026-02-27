# Legal Training + FOIA Evidence Pipeline

**Goal:** Launch the legal DPO training curriculum (Phase 3-4) using existing infrastructure, then build a Scrapling-based FOIA evidence collector that feeds both the detective's knowledge graph and the training data pipeline.

**Architecture:** Two parts, sequentially. Part 1 is operational — run existing scripts. Part 2 builds a new scraping module that closes the evidence → training data loop.

---

## Part 1: Legal Warmup + DPO Training

### What exists

Everything needed for legal training is already built and tested:

- `src/training/legal_warmup.py` — `run_legal_warmup()` generates DPO preference pairs from legal documents
- `src/data/sourcing/legal_sources.py` — domain configs with keyword filters for 3 legal domains
- `src/data/sourcing/hf_loader.py` — streams from HuggingFace pile-of-law, legalbench
- `src/data/sourcing/doj_loader.py` — CourtListener REST API + FBI Vault FOIA
- `src/training/train_dpo.py` — DPO trainer with LoRA (r=16, α=32, β=0.1)
- `checkpoints/dpo/checkpoint-23` — existing DPO checkpoint from 200 constitutional pairs

### The run

**Step 1: Generate legal pairs** (3 domains × 200 = 600 preference pairs)

```bash
detective legal-warmup \
  --output data/training/legal_pairs.jsonl \
  --examples-per-domain 200 \
  --domains criminal_justice,territorial_rights,foia_transparency
```

Domains from `legal_sources.py`:
- **Criminal justice**: sentencing, plea bargains, counsel, bail, 4th/5th/6th Amendment gaps
- **Territorial rights**: Puerto Rico, Guam, tribal sovereignty, Insular Cases, self-determination
- **FOIA/Transparency**: disclosure mandates, redaction patterns, Glomar responses, exemption abuse

Each pair: `{instruction, chosen, rejected, metadata}` — the "chosen" response identifies doctrinal gaps; the "rejected" response misses them.

**Step 2: Train legal DPO** (on HF Jobs or local)

```bash
python src/training/train_dpo.py \
  --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --resume-from checkpoints/dpo/checkpoint-23 \
  --data data/training/legal_pairs.jsonl \
  --output checkpoints/dpo-legal/
```

**Step 3: Welfare-score the legal pairs** (side effect: training data for classifier)

The 600 legal documents are rich welfare material — constitutional violations threaten multiple constructs (c, lam_P, xi, eps). Run welfare scoring to extend the classifier training set from 991 → ~1591 examples.

### Success criteria

- 600 legal pairs generated (200 per domain)
- Legal DPO checkpoint trained with combined constitutional + legal pairs
- Welfare classifier retrained on extended dataset (MAE still < 0.20)

---

## Part 2: Scrapling FOIA Evidence Collector

### Why Scrapling

FOIA portals are the primary source of declassified evidence for the detective's investigation. They share common challenges:

| Challenge | Scrapling feature |
|-----------|------------------|
| Layouts change without notice | Adaptive element relocation — parser learns structure changes |
| Rate limiting / anti-bot | StealthyFetcher with Cloudflare bypass |
| Paginated document lists | Spider framework with pause/resume checkpoints |
| PDF-heavy content | Fetches PDFs; OCR pipeline extracts text |
| Inconsistent structure across portals | Per-portal Spider configs with shared post-processing |

### Target FOIA portals

**Priority 1 — FBI Vault** (`vault.fbi.gov`)
- Already partially covered in `doj_loader.py` (API-based)
- Scrapling extends to: full collection crawling, PDF download, OCR extraction
- ~2,500 document sets across investigations

**Priority 2 — NARA** (`www.archives.gov/research`)
- National Archives declassified documents
- Catalog search + document page scraping
- Includes JFK files, various declassified collections

**Priority 3 — State Department FOIA** (`foia.state.gov`)
- Virtual Reading Room with case-organized documents
- Diplomatic cables, embassy reports, FOIA responses

### Architecture

```
src/data/sourcing/foia_scraper.py     ← New: Scrapling Spider configs per portal
src/data/sourcing/ocr_provider.py     ← Existing stub: wire to pytesseract/deepseek-ocr
src/data/sourcing/document_ingestion.py ← Existing stub: complete the orchestrator

Flow:
  foia_scraper.py → raw HTML/PDFs
    → ocr_provider.py → extracted text
    → document_ingestion.py → Document objects
    → DUAL OUTPUT:
      1. Evidence: A/B/C modules + knowledge graph (investigation)
      2. Training: welfare scoring + constitutional warmup (model improvement)
```

### Module: `src/data/sourcing/foia_scraper.py`

```python
class FOIAScraper:
    """Scrapling-based FOIA portal crawler."""

    def __init__(self, portal: str, output_dir: Path):
        """portal: 'fbi_vault' | 'nara' | 'state_dept'"""

    def crawl(self, query: str | None = None, max_pages: int = 100) -> list[FOIADocument]:
        """Crawl a FOIA portal, return document metadata + content."""

    def download_pdfs(self, documents: list[FOIADocument], output_dir: Path) -> list[Path]:
        """Download PDFs for OCR processing."""

@dataclass(frozen=True)
class FOIADocument:
    source_portal: str          # 'fbi_vault', 'nara', 'state_dept'
    title: str
    url: str
    date: str | None
    collection: str | None      # e.g., 'jeffrey-epstein' on FBI Vault
    text: str                   # extracted text (post-OCR if PDF)
    pdf_path: Path | None       # local path to downloaded PDF
```

### Portal-specific Spider configs

**FBI Vault**: Navigate collection pages → document list → individual PDFs. FBI Vault uses a REST-like structure: `/vault/collections/{collection}/pages/{page}`.

**NARA**: Catalog search → result pages → document detail → file downloads. NARA uses a search API with pagination.

**State Dept**: Reading Room → case folders → individual documents. Paginated HTML with embedded document viewers.

### Dual output pipeline

Every scraped document produces:

1. **Evidence output** (investigation):
   - Run A/B/C modules → `BiasDetection`, `DeterminismDetection`, `GeopoliticalDetection`
   - Extract entities → add to knowledge graph
   - Detect gaps → create `Gap` objects
   - Score welfare impact → prioritize by Φ gradient

2. **Training output** (model improvement):
   - Welfare classifier scores → extend training set
   - Constitutional warmup → generate new DPO preference pairs
   - Scenario extraction → forecaster training scenarios
   - Gap annotations → future SFT training data

### CLI integration

```bash
# Scrape FBI Vault Epstein collection
detective scrape-foia --portal fbi_vault --collection jeffrey-epstein --output data/foia/fbi_vault/

# Process scraped documents through both pipelines
detective ingest --input data/foia/fbi_vault/ --evidence --training

# Extract scenarios from scraped corpus for forecaster
detective extract-scenarios data/foia/fbi_vault/extracted_text/ --output spaces/maninagarden/extracted_scenarios.json
```

### Dependencies

```
scrapling>=0.3.0    # Core scraping framework
pytesseract>=0.3.10 # OCR for PDF extraction (or deepseek-ocr if available)
pdf2image>=1.16.0   # PDF → images for OCR
```

### What this does NOT include

- Robin integration (Part 3, separate design doc)
- Dark web scraping (requires Tor infrastructure)
- PACER/CourtListener scraping (already covered by `doj_loader.py` API)
- Automated retraining loop (flywheel is manual for v1)

---

## Execution order

1. **Part 1, Step 1**: Generate legal pairs (existing infrastructure, ~1-2hr)
2. **Part 1, Step 2**: Train legal DPO on HF Jobs (~30min GPU)
3. **Part 1, Step 3**: Welfare-score legal corpus, retrain classifier (~15min GPU)
4. **Part 2**: Build FOIA scraper module + wire OCR + complete document_ingestion
5. **Part 2**: Scrape FBI Vault Epstein collection as first test
6. **Part 2**: Run dual pipeline on scraped documents
7. **Part 3** (future): Robin integration design doc
