"""Hybrid retrieval: BM25 + Qdrant vector search, fused with RRF, re-ranked by Cohere.

Pipeline per query:
    1. classify_query  → select relevant Qdrant collections
    2. bm25_search + vector_search  run in parallel via ThreadPoolExecutor
    3. reciprocal_rank_fusion  → merge & deduplicate
    4. rerank (Cohere rerank-v3.5, falls back to RRF top-n on rate limit)
    5. return top-5 results
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cohere
from dotenv import load_dotenv
from loguru import logger
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

load_dotenv()

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.ingestion.embedder import embed_batch
from src.utils.config import load_config

logger.add(
    Path(__file__).parents[2] / "logs" / "sbp_rag.log",
    rotation="10 MB",
    level="INFO",
)

CHUNKS_PATH = Path(__file__).parents[2] / "data" / "processed" / "chunks.jsonl"

# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def load_bm25_index(chunks_path: Path = CHUNKS_PATH) -> tuple[BM25Okapi, list[dict]]:
    """Load chunks.jsonl and build an in-memory BM25Okapi index.

    Args:
        chunks_path: Path to the chunks.jsonl file produced by chunker.py.

    Returns:
        Tuple of (fitted BM25Okapi index, list of chunk dicts in index order).
    """
    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    logger.info(f"BM25 index built — {len(chunks)} documents")
    return bm25, chunks


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """Score all chunks with BM25 and return the top-k.

    Args:
        query: Raw query string.
        bm25: Fitted BM25Okapi index.
        chunks: Chunk dicts in the same order as the index.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys ``chunk``, ``rank``, and ``score``.
    """
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"chunk": chunks[idx], "rank": rank, "score": float(scores[idx])}
        for rank, idx in enumerate(ranked_indices)
    ]


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def vector_search(
    query: str,
    qdrant_client: QdrantClient,
    collections: list[str],
    top_k: int = 20,
) -> list[dict]:
    """Embed the query and search the specified Qdrant collections.

    Results from all collections are merged and sorted by descending cosine
    score; only the top_k are returned.

    Args:
        query: Raw query string.
        qdrant_client: Connected QdrantClient instance.
        collections: Qdrant collection names to search.
        top_k: Number of merged results to return.

    Returns:
        List of dicts with keys ``chunk``, ``rank``, and ``score``.
    """
    query_vector = embed_batch([query], task_type="retrieval_query")[0]

    all_hits: list[dict] = []
    for collection_name in collections:
        try:
            hits = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
            for hit in hits:
                payload = dict(hit.payload or {})
                text = payload.pop("text", "")
                all_hits.append({
                    "chunk": {"text": text, "metadata": payload},
                    "score": hit.score,
                })
        except Exception as exc:
            logger.warning(f"Vector search failed for '{collection_name}': {exc}")

    all_hits.sort(key=lambda h: h["score"], reverse=True)
    top = all_hits[:top_k]
    for rank, hit in enumerate(top):
        hit["rank"] = rank
    return top


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge ranked result lists using Reciprocal Rank Fusion.

    Deduplication key is the first 100 characters of each chunk's text.
    Scores are summed across all lists; the merged set is sorted by
    descending RRF score and truncated to 20 results.

    Args:
        results_lists: Sequence of ranked result lists (each item has
            ``chunk`` and ``rank`` keys).
        k: RRF smoothing constant (higher = less top-rank bias).

    Returns:
        Up to 20 merged dicts with keys ``chunk`` and ``rrf_score``.
    """
    scores: dict[str, float] = {}
    chunks_by_key: dict[str, dict] = {}

    for result_list in results_lists:
        for item in result_list:
            key = item["chunk"]["text"][:100]
            rrf_score = 1.0 / (k + item["rank"])
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in chunks_by_key:
                chunks_by_key[key] = item["chunk"]

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
    return [{"chunk": chunks_by_key[key], "rrf_score": score} for key, score in merged]


# ---------------------------------------------------------------------------
# Cohere reranking
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Re-rank candidates with the Cohere rerank-v3.5 model.

    Falls back to returning the RRF top_n on any rate-limit or API error so
    that the pipeline degrades gracefully on trial-key throttling.

    Args:
        query: Raw query string.
        candidates: RRF-merged results, each containing a ``chunk`` key.
        top_n: Number of results to return after reranking.

    Returns:
        Top ``top_n`` candidates in reranked order, each augmented with a
        ``rerank_score`` key when Cohere succeeded.
    """
    if not candidates:
        return []

    api_key = os.getenv("COHERE_API_KEY", "")
    documents = [c["chunk"]["text"] for c in candidates]

    try:
        co = cohere.ClientV2(api_key=api_key)
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=top_n,
        )
        reranked = []
        for result in response.results:
            candidate = dict(candidates[result.index])
            candidate["rerank_score"] = result.relevance_score
            reranked.append(candidate)
        return reranked
    except Exception as exc:
        exc_name = type(exc).__name__
        if "TooManyRequests" in exc_name or "RateLimit" in exc_name or "429" in str(exc):
            logger.warning("Cohere rate limit hit — falling back to RRF top-n")
        else:
            logger.warning(f"Cohere rerank failed ({exc_name}: {exc}) — falling back to RRF top-n")
        return candidates[:top_n]


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

_AML_KEYWORDS = {
    "aml", "cft", "cpf", "kyc", "cdd", "suspicious", "str",
    "money laundering", "terrorist financing", "beneficial owner",
    "politically exposed", "pep", "sanctions", "targeted financial",
    "financial crime", "due diligence", "risk based approach",
}

_LAW_KEYWORDS = {
    "sbp act", "banking companies ordinance",
    "microfinance institutions ordinance", "state bank of pakistan act",
    "foreign exchange regulation act", "schedule to the act",
}


def classify_query(query: str) -> list[str]:
    """Route a query to the relevant Qdrant collections via keyword matching.

    Returns all three collections when no strong keyword signal is detected
    so that no relevant content is missed.

    Args:
        query: Raw query string.

    Returns:
        List of Qdrant collection name strings to search.
    """
    cfg = load_config()
    cols = cfg["qdrant"]["collections"]
    q_lower = query.lower()

    is_aml = any(kw in q_lower for kw in _AML_KEYWORDS)
    is_law = any(kw in q_lower for kw in _LAW_KEYWORDS)

    if is_aml and not is_law:
        return [cols["aml"]]
    if is_law and not is_aml:
        return [cols["laws"]]
    return list(cols.values())


# ---------------------------------------------------------------------------
# Main retrieval orchestrator
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    qdrant_client: QdrantClient,
    collections_override: list[str] | None = None,
) -> list[dict]:
    """Run the full hybrid retrieval pipeline for a single query.

    BM25 and vector search run in parallel; results are fused with RRF and
    then re-ranked by Cohere to produce the final top-5.

    Args:
        query: Raw query string.
        bm25: Fitted BM25Okapi index (built by load_bm25_index).
        chunks: Chunk dicts in the same order as the BM25 index.
        qdrant_client: Connected QdrantClient instance.
        collections_override: When provided, skips classify_query and searches
            only these collections. Used by the API to honour doc_type filters.

    Returns:
        Up to 5 result dicts, each with a ``chunk`` key containing
        ``text`` and ``metadata`` sub-keys (and optionally ``rerank_score``).
    """
    cfg = load_config()
    r = cfg["retrieval"]

    collections = collections_override or classify_query(query)
    logger.info(f"Query classified → collections: {collections}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(bm25_search, query, bm25, chunks, r["bm25_top_k"])
        vector_future = executor.submit(
            vector_search, query, qdrant_client, collections, r["vector_top_k"]
        )
        bm25_results = bm25_future.result()
        vector_results = vector_future.result()

    fused = reciprocal_rank_fusion([bm25_results, vector_results], k=r["rrf_k"])
    final = rerank(query, fused, top_n=r["rerank_top_n"])

    logger.info(f"Retrieved {len(final)} chunks for: '{query[:60]}'")
    return final
