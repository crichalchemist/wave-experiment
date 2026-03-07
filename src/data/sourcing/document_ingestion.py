"""
Document ingestion pipeline: MIME detection → rasterization → OCR → DocumentRecord.

Handles the common FOIA/court document problem: files with wrong extensions.
Examples seen in practice:
  - exhibit_47.pdf  → actual MIME: image/jpeg (scanned page saved with wrong ext)
  - deposition.pdf  → actual MIME: application/pdf (real PDF, needs rasterization)
  - recording.mov   → actual MIME: video/mp4  (out of scope, logged and skipped)

Redaction detection: pages with black-pixel ratio > REDACTION_THRESHOLD are
annotated with <REDACTED_REGION> markers — the extent and pattern of redaction
is itself an information gap per docs/constitution.md (normative gaps).

Output: DocumentRecord (frozen dataclass) with full provenance for training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Black pixel ratio above this marks a region as redacted
_REDACTION_THRESHOLD: float = 0.85

# MIME types this pipeline can process
_SUPPORTED_IMAGE_MIMES = frozenset({
    "image/jpeg", "image/jpg", "image/png", "image/tiff", "image/bmp",
})
_SUPPORTED_PDF_MIME = "application/pdf"


@dataclass(frozen=True)
class DocumentRecord:
    """
    Normalized output from the document ingestion pipeline.

    All fields are immutable — DocumentRecord is an evidence artifact.
    Mutating an ingested document after the fact would violate the
    chain of custody principle (constitution: standpoint transparency).
    """
    text: str               # extracted plain text (OCR'd or parsed)
    true_mime: str          # MIME type from file header, not extension
    declared_suffix: str    # original file extension (may differ from true_mime)
    source_path: str        # original file path for provenance
    page_count: int         # number of pages processed
    redaction_ratio: float  # 0.0 (no redaction) to 1.0 (fully redacted)
    ocr_backend: str        # "tesseract", "deepseek-ocr", or "fallback_chain"
    ocr_confidence: float = 0.0  # [0.0, 1.0] heuristic quality score

    def __post_init__(self) -> None:
        if not (0.0 <= self.redaction_ratio <= 1.0):
            raise ValueError(f"redaction_ratio must be [0,1], got {self.redaction_ratio}")
        if not (0.0 <= self.ocr_confidence <= 1.0):
            raise ValueError(f"ocr_confidence must be [0,1], got {self.ocr_confidence}")

    @property
    def has_redactions(self) -> bool:
        return self.redaction_ratio > _REDACTION_THRESHOLD * 0.5

    @property
    def suffix_mismatch(self) -> bool:
        """True if the file extension doesn't match the true MIME type."""
        mime_to_ext = {
            "image/jpeg": {".jpg", ".jpeg"},
            "image/png": {".png"},
            "application/pdf": {".pdf"},
        }
        expected = mime_to_ext.get(self.true_mime, set())
        return bool(expected) and self.declared_suffix.lower() not in expected


def detect_true_mime(path: Path) -> str:
    """
    Read file header bytes to determine true MIME type.

    Does NOT trust the file extension — FOIA documents are commonly
    mislabeled. Falls back to extension-guessing if python-magic unavailable.
    """
    try:
        import magic
        return magic.from_file(str(path), mime=True)
    except ImportError:
        pass

    # Fallback: read magic bytes manually
    header = path.read_bytes()[:16]
    if header[:4] == b"%PDF":
        return "application/pdf"
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    # Unknown — return based on extension
    suffix = path.suffix.lower()
    return {".pdf": "application/pdf", ".jpg": "image/jpeg", ".png": "image/png"}.get(suffix, "application/octet-stream")


def estimate_redaction_ratio(image: "object") -> float:
    """
    Estimate what fraction of a page image is redacted (black regions).

    Returns 0.0–1.0. A ratio > 0.85 indicates heavy redaction.
    Redaction pattern is preserved in DocumentRecord as a gap signal.
    """
    try:
        import numpy as np
        from PIL import Image as PILImage
        if not isinstance(image, PILImage.Image):
            return 0.0
        gray = image.convert("L")
        arr = np.array(gray)
        black_pixels = (arr < 30).sum()  # pixels darker than near-black
        return float(black_pixels) / arr.size
    except ImportError:
        return 0.0


def _rasterize_pdf(path: Path, dpi: int = 200) -> list["object"]:
    """Convert each PDF page to a PIL Image for OCR."""
    try:
        from pdf2image import convert_from_path
        return convert_from_path(str(path), dpi=dpi)
    except ImportError as e:
        raise ImportError("pip install pdf2image && apt install poppler-utils") from e


def _get_ocr_chain() -> object:
    """Build the default OCR fallback chain from available backends."""
    from src.data.sourcing.ocr_provider import (
        OcrFallbackChain,
        _DeepSeekOcrBackend,
        _TesseractBackend,
        default_backend,
    )
    # Use the auto-selected backend first, then add the other as fallback
    backends = [default_backend]
    if isinstance(default_backend, _DeepSeekOcrBackend):
        backends.append(_TesseractBackend())
    elif isinstance(default_backend, _TesseractBackend):
        # DeepSeek is expensive to add as fallback — only if explicitly requested
        pass
    return OcrFallbackChain(backends=backends)


def _ocr_image_file(image: "object") -> object:
    """OCR a single PIL Image using the fallback chain."""
    chain = _get_ocr_chain()
    return chain.extract_text_with_confidence(image)  # type: ignore[attr-defined,arg-type]


def ingest_document(
    path: Path,
    source_id: str = "",
    max_pages: int = 50,
) -> DocumentRecord:
    """
    Ingest a document file regardless of its declared extension.

    Pipeline:
      1. MIME-sniff the file header (don't trust extension)
      2. Route: image → OCR directly | PDF → rasterize → OCR page by page
      3. Estimate redaction ratio per page
      4. Return DocumentRecord with full provenance

    Args:
        path: Path to the document file.
        source_id: Provenance identifier (e.g., "fbi_vault_001", "maxwell_trial").
        max_pages: Limit page processing for very long documents.

    Returns:
        DocumentRecord with extracted text and metadata.
    """
    true_mime = detect_true_mime(path)
    declared_suffix = path.suffix.lower()

    pages_text: list[str] = []
    total_redaction = 0.0
    page_count = 0
    backend_name = "unknown"
    confidence_sum = 0.0

    if true_mime in _SUPPORTED_IMAGE_MIMES:
        # Single-page image (including mislabeled PDFs)
        try:
            from PIL import Image as PILImage
            img = PILImage.open(path)
            ocr_result = _ocr_image_file(img)
            pages_text.append(ocr_result.text)
            total_redaction = estimate_redaction_ratio(img)
            page_count = 1
            backend_name = ocr_result.backend_name
            confidence_sum = ocr_result.confidence
        except Exception as e:
            pages_text.append(f"[OCR FAILED: {e}]")
            page_count = 1

    elif _SUPPORTED_PDF_MIME in true_mime:
        # Real PDF: rasterize each page
        try:
            images = _rasterize_pdf(path)[:max_pages]
            redaction_sum = 0.0
            for img in images:
                ocr_result = _ocr_image_file(img)
                redaction = estimate_redaction_ratio(img)
                redaction_sum += redaction
                confidence_sum += ocr_result.confidence
                backend_name = ocr_result.backend_name
                if redaction > _REDACTION_THRESHOLD:
                    pages_text.append(f"<REDACTED_REGION redaction_ratio={redaction:.2f}>")
                else:
                    pages_text.append(ocr_result.text)
            page_count = len(images)
            total_redaction = redaction_sum / page_count if page_count else 0.0
        except Exception as e:
            pages_text.append(f"[PDF PROCESSING FAILED: {e}]")
            page_count = 1

    else:
        # Unsupported MIME (video, audio, binary) — log and skip
        pages_text.append(
            f"[UNSUPPORTED MIME: {true_mime} — declared suffix: {declared_suffix}]"
        )
        page_count = 0

    avg_confidence = confidence_sum / page_count if page_count else 0.0

    return DocumentRecord(
        text="\n\n".join(pages_text),
        true_mime=true_mime,
        declared_suffix=declared_suffix,
        source_path=str(path),
        page_count=page_count,
        redaction_ratio=min(1.0, total_redaction),
        ocr_backend=backend_name,
        ocr_confidence=min(1.0, avg_confidence),
    )
