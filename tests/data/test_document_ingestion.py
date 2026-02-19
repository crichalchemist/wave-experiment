"""Tests for MIME-sniffing document ingestion pipeline."""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_import():
    from src.data.sourcing.document_ingestion import ingest_document, DocumentRecord
    assert callable(ingest_document)


def test_document_record_is_frozen():
    from src.data.sourcing.document_ingestion import DocumentRecord
    import pytest
    rec = DocumentRecord(
        text="Hello",
        true_mime="image/jpeg",
        declared_suffix=".pdf",
        source_path="exhibit_1.pdf",
        page_count=1,
        redaction_ratio=0.0,
        ocr_backend="tesseract",
    )
    with pytest.raises(Exception):
        rec.text = "mutated"


def test_mime_mismatch_detected(tmp_path):
    """A file with .pdf suffix but JPEG header must be detected as image/jpeg."""
    from src.data.sourcing.document_ingestion import detect_true_mime

    # Write a minimal JPEG header (SOI marker: FF D8 FF)
    fake_pdf = tmp_path / "exhibit.pdf"
    fake_pdf.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    true_mime = detect_true_mime(fake_pdf)
    assert true_mime.startswith("image/"), f"Expected image/*, got {true_mime}"


def test_real_pdf_detected(tmp_path):
    """A real PDF file must be detected as application/pdf."""
    from src.data.sourcing.document_ingestion import detect_true_mime

    # Minimal valid PDF header
    fake_pdf = tmp_path / "document.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n" + b"\x00" * 100)

    true_mime = detect_true_mime(fake_pdf)
    assert true_mime == "application/pdf" or "pdf" in true_mime.lower()


def test_ingest_image_routes_to_ocr(tmp_path):
    """A JPEG file (even if named .pdf) routes to OCR backend."""
    from src.data.sourcing.document_ingestion import ingest_document

    fake_jpg = tmp_path / "exhibit.pdf"  # mislabeled
    fake_jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    mock_ocr = MagicMock(return_value="Extracted text from image.")
    mock_pil_image = MagicMock()

    with patch("src.data.sourcing.document_ingestion._ocr_image_file", mock_ocr):
        with patch("src.data.sourcing.document_ingestion.detect_true_mime",
                   return_value="image/jpeg"):
            with patch("PIL.Image.open", return_value=mock_pil_image):
                record = ingest_document(fake_jpg, source_id="test")

    assert record.text == "Extracted text from image."
    assert record.true_mime == "image/jpeg"
    assert record.declared_suffix == ".pdf"


def test_ingest_pdf_rasterizes_and_ocrs(tmp_path):
    """A real PDF is rasterized to images then OCR'd page by page."""
    from src.data.sourcing.document_ingestion import ingest_document

    fake_pdf = tmp_path / "document.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n" + b"\x00" * 100)

    mock_rasterize = MagicMock(return_value=[MagicMock()])  # one page
    mock_ocr = MagicMock(return_value="Page 1 text.")

    with patch("src.data.sourcing.document_ingestion._rasterize_pdf", mock_rasterize):
        with patch("src.data.sourcing.document_ingestion._ocr_image_file", mock_ocr):
            with patch("src.data.sourcing.document_ingestion.detect_true_mime",
                       return_value="application/pdf"):
                record = ingest_document(fake_pdf, source_id="test")

    assert "Page 1 text." in record.text
    assert record.page_count == 1


def test_redaction_ratio_reported(tmp_path):
    """Heavily redacted pages have redaction_ratio > 0."""
    from src.data.sourcing.document_ingestion import estimate_redaction_ratio
    import PIL.Image

    # Create a mostly-black image (simulating heavy redaction)
    img = PIL.Image.new("L", (100, 100), color=0)  # all black
    ratio = estimate_redaction_ratio(img)
    assert ratio > 0.9


def test_lightly_redacted_ratio(tmp_path):
    from src.data.sourcing.document_ingestion import estimate_redaction_ratio
    import PIL.Image

    img = PIL.Image.new("L", (100, 100), color=255)  # all white
    ratio = estimate_redaction_ratio(img)
    assert ratio < 0.1


def test_ingest_returns_document_record(tmp_path):
    from src.data.sourcing.document_ingestion import ingest_document, DocumentRecord

    f = tmp_path / "doc.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    with patch("src.data.sourcing.document_ingestion.detect_true_mime",
               return_value="image/jpeg"):
        with patch("src.data.sourcing.document_ingestion._ocr_image_file",
                   return_value="Some extracted text"):
            record = ingest_document(f, source_id="fbi_vault_001")

    assert isinstance(record, DocumentRecord)
    assert record.source_path == str(f)
