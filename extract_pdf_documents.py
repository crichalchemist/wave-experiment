#!/usr/bin/env python3
"""
Extract text from PDFs in 'smiles and cries' for constitutional warmup.

Uses the existing OCR infrastructure (document_ingestion.py) which handles:
- MIME detection (doesn't trust file extensions)
- Dual-backend OCR (DeepSeek-OCR for GPU, pytesseract for CPU)
- Redaction detection (identifies heavily blacked-out pages)
- Immutable DocumentRecord with full provenance

Chunks documents into meaningful segments and analyzes welfare relevance.
"""

import os
import sys
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Hybrid PDF text extraction: PyPDF2 first, OCR fallback.

    Strategy:
    1. Try PyPDF2 for embedded text (fast - seconds)
    2. If text is too short (<500 chars), fallback to OCR
    3. OCR is only used for actual scans or failed extractions
    """
    path = Path(pdf_path)

    # Step 1: Try fast PyPDF2 extraction
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            max_pages = min(50, len(reader.pages))

            for page_num in range(max_pages):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"

            # Check if we got meaningful text
            if len(text.strip()) > 500:
                print(f"  ✓ PyPDF2: {len(text)} chars from {max_pages} pages", file=sys.stderr)
                return text
            else:
                print(f"  ⚠ PyPDF2 extracted only {len(text.strip())} chars - trying OCR", file=sys.stderr)

    except ImportError:
        print(f"  ⚠ PyPDF2 not available - trying OCR", file=sys.stderr)
    except Exception as e:
        print(f"  ⚠ PyPDF2 failed: {e} - trying OCR", file=sys.stderr)

    # Step 2: Fallback to OCR for scanned PDFs
    try:
        from src.data.sourcing.document_ingestion import ingest_document

        doc = ingest_document(
            path=path,
            source_id=f"smiles_cries_{path.stem}",
            max_pages=50
        )

        print(f"  ✓ OCR ({doc.ocr_backend}): {len(doc.text)} chars", file=sys.stderr)
        if doc.has_redactions:
            print(f"  ⚠ Redactions: {doc.redaction_ratio:.1%}", file=sys.stderr)

        return doc.text

    except Exception as e:
        print(f"  ✗ OCR failed: {e}", file=sys.stderr)
        return ""


def chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
    """Split text into chunks of ~chunk_size characters at sentence boundaries."""
    if not text:
        return []

    # Simple sentence splitting (can be improved with nltk)
    sentences = text.replace('\n', ' ').split('. ')

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def analyze_welfare_relevance(text: str) -> tuple[tuple[str, ...], float]:
    """Analyze text for welfare constructs and score."""
    from src.inference.welfare_scoring import infer_threatened_constructs, score_hypothesis_welfare
    from src.detective.hypothesis import Hypothesis

    constructs = infer_threatened_constructs(text)

    if not constructs:
        return (), 0.0

    # Use baseline phi_metrics
    phi_metrics = {
        "c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
        "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5
    }

    h = Hypothesis.create(text[:500], 0.5)
    welfare_score = score_hypothesis_welfare(h, phi_metrics)

    return constructs, welfare_score


def main():
    pdf_dir = Path("src/data/smiles and cries")
    output_file = Path("data/training/smiles_and_cries_extracted.txt")

    print("\n" + "="*70)
    print("PDF DOCUMENT EXTRACTION FOR CONSTITUTIONAL WARMUP")
    print("="*70)

    if not pdf_dir.exists():
        print(f"\n❌ Directory not found: {pdf_dir}")
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"\nFound {len(pdfs)} PDF files")
    print(f"Output: {output_file}\n")

    all_chunks = []
    welfare_stats = {"high": 0, "medium": 0, "low": 0, "none": 0}

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Processing: {pdf_path.name[:50]}...")

        # Extract text
        text = extract_text_from_pdf(str(pdf_path))
        if not text:
            print("  ⚠ No text extracted")
            continue

        # Chunk into segments
        chunks = chunk_text(text, chunk_size=2000)
        print(f"  Extracted {len(chunks)} chunks")

        # Analyze first few chunks for welfare relevance
        sample_chunks = chunks[:5]  # Check first 5 chunks
        high_welfare_chunks = 0

        for chunk in sample_chunks:
            constructs, score = analyze_welfare_relevance(chunk)
            if score >= 0.5:
                high_welfare_chunks += 1
                welfare_stats["high"] += 1
            elif score >= 0.3:
                welfare_stats["medium"] += 1
            elif score > 0:
                welfare_stats["low"] += 1
            else:
                welfare_stats["none"] += 1

        # Add chunks to collection
        for j, chunk in enumerate(chunks):
            # Add metadata comment
            metadata = f"# Source: {pdf_path.name} | Chunk: {j+1}/{len(chunks)}"
            all_chunks.append(f"{metadata}\n{chunk}\n")

        print(f"  Welfare analysis (first 5 chunks):")
        print(f"    High relevance (≥0.5): {high_welfare_chunks}/5")

    # Write to output file
    print(f"\n" + "="*70)
    print("WRITING EXTRACTED TEXT")
    print("="*70)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Extracted text from 'smiles and cries' PDF collection\n")
        f.write("# For constitutional warmup with welfare filtering\n")
        f.write(f"# Total chunks: {len(all_chunks)}\n\n")

        for chunk in all_chunks:
            f.write(chunk + "\n" + "="*70 + "\n\n")

    print(f"\nWrote {len(all_chunks)} chunks to: {output_file}")
    print(f"File size: {output_file.stat().st_size:,} bytes")

    print(f"\n" + "="*70)
    print("WELFARE RELEVANCE STATISTICS (sampled chunks)")
    print("="*70)
    total = sum(welfare_stats.values())
    print(f"\nTotal sampled: {total}")
    print(f"  High (≥0.5):   {welfare_stats['high']:3d} ({welfare_stats['high']/total*100:.1f}%)")
    print(f"  Medium (≥0.3): {welfare_stats['medium']:3d} ({welfare_stats['medium']/total*100:.1f}%)")
    print(f"  Low (>0):      {welfare_stats['low']:3d} ({welfare_stats['low']/total*100:.1f}%)")
    print(f"  None (0):      {welfare_stats['none']:3d} ({welfare_stats['none']/total*100:.1f}%)")

    print(f"\n✅ Extraction complete! Use this file in run_warmup.py:")
    print(f"   config.document_file = '{output_file}'")
    print()


if __name__ == "__main__":
    main()
