"""Integration tests for the hybrid retrieval pipeline.

Requires a populated Qdrant store and chunks.jsonl. Run after ingestion:

    python src/ingestion/embedder.py
    pytest tests/test_retrieval.py -v
"""

import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.retrieval.retriever import load_bm25_index, retrieve
from src.utils.config import load_config

CHUNKS_PATH = Path(__file__).parents[1] / "data" / "processed" / "chunks.jsonl"
REQUIRED_METADATA_KEYS = {"source_file", "page", "section_header"}

QUERIES = [
    "What is the minimum capital requirement for a microfinance bank?",
    "Define a Suspicious Transaction Report under AML regulations",
    "What does Section 17 of the SBP Act say?",
    "Prudential regulations for SME financing",
    "banking license",
]


@pytest.fixture(scope="module")
def retrieval_components():
    """Load BM25 index and Qdrant client once for the full test module."""
    if not CHUNKS_PATH.exists():
        pytest.skip(f"chunks.jsonl not found at {CHUNKS_PATH} — run ingestion first")

    cfg = load_config()
    bm25, chunks = load_bm25_index(CHUNKS_PATH)
    qdrant_client = QdrantClient(path=cfg["qdrant"]["path"])
    return bm25, chunks, qdrant_client


@pytest.mark.parametrize("query", QUERIES)
def test_retrieve(retrieval_components, query):
    """Each query must return at least one result with the required metadata keys."""
    bm25, chunks, qdrant_client = retrieval_components

    results = retrieve(query, bm25, chunks, qdrant_client)

    assert len(results) > 0, f"No results returned for: '{query}'"

    for result in results:
        metadata = result["chunk"]["metadata"]
        missing = REQUIRED_METADATA_KEYS - set(metadata.keys())
        assert not missing, (
            f"Result missing metadata keys {missing} for query: '{query}'"
        )
