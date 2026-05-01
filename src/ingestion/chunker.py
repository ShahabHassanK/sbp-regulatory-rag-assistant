"""Chunking pipeline — splits parsed pages into overlapping chunks
and writes chunks.jsonl for embedding in the next phase."""

import json
import random
import re
import sys
from pathlib import Path

from loguru import logger
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.utils.config import load_config

logger.add(
    Path(__file__).parents[2] / "logs" / "ingestion.log",
    rotation="10 MB",
    level="INFO",
)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
RAW_PAGES_PATH = PROCESSED_DIR / "raw_pages.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# Section header patterns found in SBP regulatory documents.
# Order matters: more specific patterns come first.
_HEADER_PATTERNS = [
    r"^(PR|R|M|O|G)-\d+[^\n]*",            # Pakistani reg codes: PR-1, M-3
    r"^Regulation\s+\d+[^\n]*",             # Regulation 5 – Capital Requirements
    r"^Section\s+\d+[\.\d]*[^\n]*",         # Section 17, Section 3.2
    r"^Article\s+\d+[^\n]*",                # Article 12 – Definitions
    r"^CHAPTER\s+[IVXLCDM]+[^\n]*",         # CHAPTER IV
    r"^Chapter\s+[IVXLCDM\d]+[^\n]*",       # Chapter 3
    r"^PART\s+[IVXLCDM]+[^\n]*",            # PART III
    r"^Part\s+[IVXLCDM\d]+[^\n]*",          # Part 2
    r"^\d+\.\s+[A-Z][A-Za-z][^\n]*",        # 3. Definitions and Scope
]

_COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in _HEADER_PATTERNS]


def extract_section_header(text: str) -> str:
    """Find the first legal section header within the first 10 lines of text.

    Tries each pattern in order of specificity. Returns the matched string
    so it can be prepended to child chunks, keeping citation context intact
    even after the page text is split into smaller pieces.

    Args:
        text: Raw page or chunk text.

    Returns:
        The matched header string, or empty string if none found.
    """
    # Only scan the first 10 lines — headers appear at the top
    first_lines = "\n".join(text.splitlines()[:10])
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(first_lines)
        if match:
            return match.group(0).strip()
    return ""


def chunk_pages(pages: list[dict], cfg: dict) -> list[dict]:
    """Split each page's text into overlapping token chunks.

    Uses LlamaIndex SentenceSplitter which respects sentence boundaries —
    it won't cut a sentence in half even to hit the exact chunk_size.
    The section header is prepended to every chunk from that page so
    the retriever always knows which section a chunk came from.

    Args:
        pages: List of page dicts from the parsing phase.
        cfg: Loaded config dict (uses chunking sub-key).

    Returns:
        List of chunk dicts, each with 'text' and 'metadata'.
    """
    chunk_cfg = cfg["chunking"]
    splitter = SentenceSplitter(
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
    )

    all_chunks: list[dict] = []

    for page in pages:
        raw_text: str = page["text"]
        page_meta: dict = page["metadata"]
        section_header = extract_section_header(raw_text)

        # Prepend section header so every child chunk inherits it
        text_to_split = (
            f"{section_header}\n{raw_text}" if section_header else raw_text
        )

        try:
            splits = splitter.split_text(text_to_split)
        except Exception as exc:
            logger.warning(f"SentenceSplitter failed on page {page_meta.get('page')}: {exc}")
            splits = [text_to_split]

        for idx, chunk_text in enumerate(splits):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    **page_meta,
                    "section_header": section_header,
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                },
            })

    return all_chunks


def filter_chunks(chunks: list[dict], min_length: int) -> list[dict]:
    """Remove chunks that are too short to be useful for retrieval.

    Very short chunks (navigation text, page numbers, headers-only) add
    noise to search results without contributing meaningful content.

    Args:
        chunks: List of chunk dicts from chunk_pages().
        min_length: Minimum character count (from config chunking.min_chunk_length).

    Returns:
        Filtered list with only chunks meeting the minimum length.
    """
    before = len(chunks)
    filtered = [c for c in chunks if c["metadata"]["char_count"] >= min_length]
    removed = before - len(filtered)
    if removed:
        logger.info(f"Filtered out {removed} chunks below {min_length} chars")
    return filtered


def run_chunking_pipeline(raw_pages_path: Path = RAW_PAGES_PATH) -> list[dict]:
    """Load parsed pages, chunk them, filter, and save chunks.jsonl.

    Args:
        raw_pages_path: Path to the raw_pages.jsonl written by pdf_parser.py.

    Returns:
        Final list of chunk dicts saved to chunks.jsonl.
    """
    cfg = load_config()

    if not raw_pages_path.exists():
        raise FileNotFoundError(
            f"raw_pages.jsonl not found at {raw_pages_path}. "
            "Run pdf_parser.py first."
        )

    # Load all pages from JSONL
    pages: list[dict] = []
    with open(raw_pages_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pages.append(json.loads(line))

    logger.info(f"Loaded {len(pages)} pages from {raw_pages_path}")

    # Chunk and filter
    logger.info("Splitting pages into chunks...")
    chunks = chunk_pages(pages, cfg)
    chunks = filter_chunks(chunks, cfg["chunking"]["min_chunk_length"])

    # Save to JSONL
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for chunk in tqdm(chunks, desc="Writing chunks", unit="chunk"):
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # --- Stats ---
    total_chars = sum(c["metadata"]["char_count"] for c in chunks)
    avg_chars = total_chars // len(chunks) if chunks else 0

    # Per doc_type breakdown
    type_counts: dict[str, int] = {}
    for c in chunks:
        dt = c["metadata"].get("doc_type", "unknown")
        type_counts[dt] = type_counts.get(dt, 0) + 1

    print(f"\n✓ Chunking complete")
    print(f"  Total chunks  : {len(chunks)}")
    print(f"  Avg length    : {avg_chars} chars")
    print(f"  By doc_type   :")
    for dtype, count in sorted(type_counts.items()):
        print(f"    {dtype:<15} {count}")
    print(f"  Output        : {CHUNKS_PATH}")

    # Print 3 random samples for visual inspection
    print("\n--- 3 random sample chunks ---")
    for sample in random.sample(chunks, min(3, len(chunks))):
        meta = sample["metadata"]
        print(
            f"\n  [{meta['doc_type']}] {meta['source_file']} | "
            f"Section: '{meta['section_header']}' | Page {meta['page']} | "
            f"Chunk {meta['chunk_index']} | {meta['char_count']} chars"
        )
        print(f"  Text preview: {sample['text'][:200].strip()}...")

    return chunks


if __name__ == "__main__":
    run_chunking_pipeline()
