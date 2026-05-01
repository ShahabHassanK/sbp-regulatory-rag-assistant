"""FastAPI application for the SBP Regulatory RAG Assistant.

Endpoints:
    POST /query          — single-shot RAG query, returns JSON
    POST /query/stream   — streaming SSE RAG query
    GET  /health         — liveness check + collection point counts
    GET  /collections/stats — per-collection stats with a sample payload

BM25 index and Qdrant client are initialised once at startup and stored in
app.state to avoid repeated cold-start costs per request.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import RateLimitError as GroqRateLimitError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from qdrant_client import QdrantClient

load_dotenv()

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.generation.rag_chain import get_answer_streaming, rag_query, retrieve
from src.retrieval.retriever import load_bm25_index
from src.utils.config import load_config

logger.add(
    Path(__file__).parents[1] / "logs" / "sbp_rag.log",
    rotation="10 MB",
    level="INFO",
)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    cfg = load_config()
    chunks_path = Path(__file__).parents[1] / "data" / "processed" / "chunks.jsonl"

    logger.info("Loading BM25 index and Qdrant client at startup…")
    bm25, chunks = load_bm25_index(chunks_path)
    app.state.bm25 = bm25
    app.state.chunks = chunks
    app.state.qdrant = QdrantClient(path=cfg["qdrant"]["path"])
    app.state.cfg = cfg
    logger.info(f"Startup complete — {len(chunks)} chunks in BM25 index")

    yield  # application runs here

    # Nothing to teardown for local-path Qdrant


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SBP Regulatory Assistant",
    description="Hybrid RAG over State Bank of Pakistan regulatory documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    doc_type: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query")
def query(req: QueryRequest) -> dict:
    """Run a single-shot RAG query and return the full answer with sources.

    Args:
        req: Request body with ``question`` and optional ``doc_type`` filter.

    Returns:
        Dict with ``question``, ``answer``, and ``sources`` keys.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    logger.info(f"POST /query — '{req.question[:80]}'")
    try:
        return rag_query(
            question=req.question,
            bm25=app.state.bm25,
            chunks=app.state.chunks,
            qdrant_client=app.state.qdrant,
            doc_type=req.doc_type,
        )
    except GroqRateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit reached (30 RPM on free tier). Wait a few seconds and retry.",
        )


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Stream a RAG answer token-by-token as Server-Sent Events.

    Args:
        req: Request body with ``question`` and optional ``doc_type`` filter.

    Returns:
        StreamingResponse yielding ``data: <token>\\n\\n`` SSE events.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    logger.info(f"POST /query/stream — '{req.question[:80]}'")

    from src.generation.rag_chain import _resolve_collections
    collections_override = _resolve_collections(req.doc_type)
    results = retrieve(
        req.question,
        app.state.bm25,
        app.state.chunks,
        app.state.qdrant,
        collections_override,
    )

    def event_stream():
        for token in get_answer_streaming(req.question, results):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/retrieve")
def retrieve_chunks(question: str, doc_type: str | None = None) -> dict:
    """Retrieve relevant chunks for a question without calling the LLM.

    Used by the evaluator to get contexts while generating answers via a
    separate LLM (e.g. Gemini Flash) to avoid Groq TPM limits during eval.

    Args:
        question: The regulatory question.
        doc_type: Optional document-type filter.

    Returns:
        Dict with ``chunks`` list (text + metadata) and ``question``.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    from src.generation.rag_chain import _resolve_collections
    collections_override = _resolve_collections(doc_type)
    results = retrieve(
        question,
        app.state.bm25,
        app.state.chunks,
        app.state.qdrant,
        collections_override,
    )
    chunks_out = [
        {
            "text": r["chunk"]["text"],
            "metadata": r["chunk"]["metadata"],
        }
        for r in results
    ]
    return {"question": question, "chunks": chunks_out}


@app.get("/health")
def health() -> dict:
    """Return service liveness status and per-collection point counts."""
    cfg = app.state.cfg
    collections_info = []
    for col_name in cfg["qdrant"]["collections"].values():
        try:
            info = app.state.qdrant.get_collection(col_name)
            collections_info.append({"name": col_name, "points_count": info.points_count})
        except Exception as exc:
            collections_info.append({"name": col_name, "error": str(exc)})

    return {"status": "ok", "collections": collections_info}


@app.get("/collections/stats")
def collections_stats() -> dict:
    """Return per-collection point counts and a sample payload for inspection."""
    cfg = app.state.cfg
    stats = {}
    for label, col_name in cfg["qdrant"]["collections"].items():
        try:
            info = app.state.qdrant.get_collection(col_name)
            # Scroll one point for a sample payload
            points, _ = app.state.qdrant.scroll(
                collection_name=col_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            sample = None
            if points:
                payload = dict(points[0].payload or {})
                # Truncate text for readability
                if "text" in payload:
                    payload["text"] = payload["text"][:200] + "…"
                sample = payload
            stats[label] = {
                "collection": col_name,
                "points_count": info.points_count,
                "sample": sample,
            }
        except Exception as exc:
            stats[label] = {"collection": col_name, "error": str(exc)}

    return stats
