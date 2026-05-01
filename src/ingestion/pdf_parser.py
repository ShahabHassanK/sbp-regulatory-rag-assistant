"""PDF parsing pipeline — triages each PDF as digital or scanned,
extracts text page-by-page, and writes raw_pages.jsonl."""

import json
import sys
from pathlib import Path

import pdfplumber
from loguru import logger
from tqdm import tqdm

# Make `src` importable when this file is run directly as __main__
sys.path.insert(0, str(Path(__file__).parents[2]))

# Configure loguru: INFO to console + rotating file
logger.add(
    Path(__file__).parents[2] / "logs" / "ingestion.log",
    rotation="10 MB",
    level="INFO",
)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
RAW_PAGES_PATH = PROCESSED_DIR / "raw_pages.jsonl"


def triage_pdf(pdf_path: Path) -> str:
    """Decide whether a PDF is digital (text-based) or scanned (image-based).

    Samples the first 3 pages. If their combined extractable text exceeds
    150 characters the PDF has a real text layer → 'digital'. Otherwise the
    pages are likely scanned images → 'scanned'.

    Args:
        pdf_path: Absolute path to the PDF.

    Returns:
        'digital' or 'scanned'.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            sample_pages = pdf.pages[:3]
            combined = "".join((p.extract_text() or "") for p in sample_pages)
        return "digital" if len(combined.strip()) > 150 else "scanned"
    except Exception as exc:
        logger.warning(f"Triage failed for {pdf_path.name}: {exc} — defaulting to digital")
        return "digital"


def get_document_metadata(pdf_path: Path) -> dict:
    """Infer doc_type and category labels from the PDF's directory path.

    These labels control which Qdrant collection a chunk lands in and
    how the evaluation set is stratified.

    Path conventions:
        data/laws/              → doc_type='law',        category='law'
        data/regulations/
          prudential .../       → doc_type='regulation', category='prudential'
          AML-CFT-CPF .../      → doc_type='aml',        category='aml'
          other .../            → doc_type='regulation', category='other'
        data/notifications/     → doc_type='regulation', category='notification'

    Args:
        pdf_path: Absolute path to the PDF.

    Returns:
        Dict with keys: source_file, doc_type, category, file_path.
    """
    # Lowercase all path parts for case-insensitive matching
    parts = [p.lower() for p in pdf_path.parts]

    if "laws" in parts:
        doc_type, category = "law", "law"
    elif any("aml-cft-cpf" in p for p in parts):
        doc_type, category = "aml", "aml"
    elif "notifications" in parts:
        doc_type, category = "regulation", "notification"
    else:
        doc_type = "regulation"
        if any("prudential" in p for p in parts):
            category = "prudential"
        elif any("other" in p for p in parts):
            category = "other"
        else:
            category = "regulation"

    return {
        "source_file": pdf_path.stem,
        "doc_type": doc_type,
        "category": category,
        "file_path": str(pdf_path),
    }


def _table_to_text(table: list) -> str:
    """Flatten a pdfplumber table (list[list]) into pipe-delimited text rows."""
    if not table:
        return ""
    rows = [" | ".join(str(cell or "").strip() for cell in row) for row in table]
    return "\n".join(rows)


def parse_digital_pdf(pdf_path: Path, metadata: dict) -> list[dict]:
    """Extract text and tables from a digital PDF using pdfplumber.

    Processes every page individually. Tables are converted to
    pipe-delimited text and appended after the page's prose. Pages
    with fewer than 50 characters after extraction are skipped (they
    are usually blank pages or decorative covers).

    Args:
        pdf_path: Path to the PDF.
        metadata: Dict from get_document_metadata().

    Returns:
        List of page dicts: [{"text": str, "metadata": {..., "page": int}}]
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                # Append any tables as structured text
                for table in (page.extract_tables() or []):
                    table_text = _table_to_text(table)
                    if table_text:
                        text += f"\n\n[TABLE]\n{table_text}"

                text = text.strip()
                if len(text) < 50:
                    continue

                pages.append({"text": text, "metadata": {**metadata, "page": page_num}})

    except Exception as exc:
        logger.error(f"pdfplumber failed on {pdf_path.name}: {exc}")

    return pages


def parse_scanned_pdf(pdf_path: Path, metadata: dict) -> list[dict]:
    """Extract text from a scanned PDF via Tesseract OCR (unstructured library).

    Uses strategy='ocr_only': converts each page to a PNG image then
    runs Tesseract. Requires tesseract.exe and poppler/pdfinfo on PATH.

    Falls back to parse_digital_pdf if OCR fails (some 'scanned' PDFs
    still have a partial text layer that pdfplumber can extract).

    Args:
        pdf_path: Path to the PDF.
        metadata: Dict from get_document_metadata().

    Returns:
        List of page dicts in the same format as parse_digital_pdf.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        logger.error("unstructured not installed — cannot OCR scanned PDF")
        return []

    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="ocr_only",
            languages=["eng"],
        )
    except Exception as exc:
        logger.warning(f"OCR failed for {pdf_path.name}: {exc} — falling back to digital parse")
        return parse_digital_pdf(pdf_path, metadata)

    # Group unstructured elements by their page number
    pages_dict: dict[int, list[str]] = {}
    for el in elements:
        page_num = (el.metadata.page_number or 1)
        pages_dict.setdefault(page_num, []).append(str(el))

    pages = []
    for page_num in sorted(pages_dict):
        text = "\n".join(pages_dict[page_num]).strip()
        if len(text) < 50:
            continue
        pages.append({"text": text, "metadata": {**metadata, "page": page_num}})

    return pages


def parse_all_pdfs(data_dir: str | Path) -> list[dict]:
    """Walk data_dir, parse every PDF, and write results to raw_pages.jsonl.

    Skips the 'processed/' subdirectory to avoid re-parsing outputs.
    Saves one JSON object per line (JSONL format) for streaming-friendly
    reads in later pipeline stages.

    Args:
        data_dir: Root of the data directory (contains laws/, regulations/, etc.)

    Returns:
        List of all page dicts from all documents.
    """
    data_dir = Path(data_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Recursively find all PDFs, excluding anything already in processed/
    pdf_files = sorted(
        p for p in data_dir.rglob("*.pdf")
        if "processed" not in [part.lower() for part in p.parts]
    )

    if not pdf_files:
        logger.warning(f"No PDFs found under {data_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) to parse")

    all_pages: list[dict] = []
    stats = {"digital": 0, "scanned": 0, "failed": 0}

    with open(RAW_PAGES_PATH, "w", encoding="utf-8") as out_file:
        for pdf_path in tqdm(pdf_files, desc="Parsing PDFs", unit="file"):
            metadata = get_document_metadata(pdf_path)
            pdf_type = triage_pdf(pdf_path)
            stats[pdf_type] += 1

            logger.info(
                f"{pdf_path.name}  type={pdf_type}  doc_type={metadata['doc_type']}"
            )

            pages = (
                parse_digital_pdf(pdf_path, metadata)
                if pdf_type == "digital"
                else parse_scanned_pdf(pdf_path, metadata)
            )

            if not pages:
                stats["failed"] += 1
                logger.warning(f"No pages extracted from {pdf_path.name}")
                continue

            for page in pages:
                out_file.write(json.dumps(page, ensure_ascii=False) + "\n")

            all_pages.extend(pages)
            total_chars = sum(len(p["text"]) for p in pages)
            logger.info(f"  → {len(pages)} pages | {total_chars:,} chars")

    print(f"\n✓ Parsing complete")
    print(f"  Total pages   : {len(all_pages)}")
    print(f"  Digital PDFs  : {stats['digital']}")
    print(f"  Scanned PDFs  : {stats['scanned']}")
    print(f"  Failed PDFs   : {stats['failed']}")
    print(f"  Output        : {RAW_PAGES_PATH}")

    return all_pages


if __name__ == "__main__":
    project_root = Path(__file__).parents[2]
    parse_all_pdfs(project_root / "data")
