"""RAG generation chain: context formatting, Groq LLM calls, and query orchestration.

Imports retrieve() from the retrieval layer and wraps it with a strict
regulatory-assistant prompt before calling the Groq LLM.
"""

import os
import sys
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from groq import Groq, RateLimitError as GroqRateLimitError
from loguru import logger
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

load_dotenv()

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.retrieval.retriever import retrieve
from src.utils.config import load_config

logger.add(
    Path(__file__).parents[2] / "logs" / "sbp_rag.log",
    rotation="10 MB",
    level="INFO",
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a regulatory compliance assistant specializing in \
State Bank of Pakistan (SBP) regulations.

Your role is to answer questions strictly based on the regulatory documents \
provided as context.

Rules you must follow without exception:
1. Answer ONLY from the provided context. Do not use any outside knowledge.
2. Cite every factual claim with: \
[Source: <source_file>, Section: <section_header>, Page: <page>]
3. If the context does not contain sufficient information to answer, respond \
exactly: "I cannot find this in the provided regulatory documents."
4. Never invent, estimate, or extrapolate regulatory requirements, thresholds, \
or procedures.
5. Never provide legal advice or interpretations beyond what is literally stated \
in the documents."""


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM.

    Args:
        results: Retrieval results, each with a ``chunk`` key containing
            ``text`` and ``metadata`` sub-keys.

    Returns:
        Multi-line string with one numbered block per chunk.
    """
    blocks = []
    for i, result in enumerate(results, 1):
        meta = result["chunk"]["metadata"]
        text = result["chunk"]["text"]
        blocks.append(
            f"[{i}] Document: {meta.get('source_file', 'Unknown')} "
            f"| Section: {meta.get('section_header', 'N/A')} "
            f"| Page: {meta.get('page', 'N/A')}\n{text}\n---"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=3, max=10),
    retry=retry_if_exception_type(GroqRateLimitError),
    before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
    reraise=True,
)
def get_answer(question: str, results: list[dict]) -> str:
    """Generate a grounded answer using the Groq LLM.

    Retries up to 4 times with exponential back-off on Groq 429 rate-limit
    errors. Raises on the final attempt so the API returns a 500 rather than
    hanging indefinitely.

    Args:
        question: The user's regulatory question.
        results: Retrieved chunks from the retrieval pipeline.

    Returns:
        LLM-generated answer string with inline citations.
    """
    cfg = load_config()
    client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    context = format_context(results)

    response = client.chat.completions.create(
        model=cfg["llm"]["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    answer = response.choices[0].message.content or ""
    logger.info(f"Generated answer ({len(answer)} chars) for: '{question[:60]}'")
    return answer


def get_answer_streaming(
    question: str,
    results: list[dict],
) -> Generator[str, None, None]:
    """Stream a grounded answer token-by-token from the Groq LLM.

    On Groq 429 rate-limit errors the generator yields a user-friendly error
    token rather than crashing the SSE stream.

    Args:
        question: The user's regulatory question.
        results: Retrieved chunks from the retrieval pipeline.

    Yields:
        Content delta strings as they arrive from the API.
    """
    cfg = load_config()
    client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    context = format_context(results)

    try:
        stream = client.chat.completions.create(
            model=cfg["llm"]["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except GroqRateLimitError:
        logger.warning("Groq rate limit hit in streaming endpoint — yielding error message")
        yield ("\n\n⚠️ The language model is temporarily rate-limited (Groq free tier: 30 RPM). "
               "Please wait a few seconds and try again.")


# ---------------------------------------------------------------------------
# Doc-type to collections mapping
# ---------------------------------------------------------------------------

_DOC_TYPE_MAP = {
    "laws":        "laws",
    "law":         "laws",
    "regulations": "regulations",
    "regulation":  "regulations",
    "aml":         "aml",
    "aml-cft":     "aml",
}


def _resolve_collections(doc_type: str | None) -> list[str] | None:
    """Map a doc_type string from the API to a Qdrant collection list.

    Returns None when doc_type is absent or unrecognised so that
    retrieve() falls back to classify_query().

    Args:
        doc_type: Optional filter string from the API request body.

    Returns:
        List of Qdrant collection names, or None.
    """
    if not doc_type:
        return None
    cfg = load_config()
    cols = cfg["qdrant"]["collections"]
    key = _DOC_TYPE_MAP.get(doc_type.lower())
    return [cols[key]] if key else None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def rag_query(
    question: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    qdrant_client: QdrantClient,
    doc_type: str | None = None,
) -> dict:
    """Run the full RAG pipeline: retrieve → format → generate.

    Args:
        question: The user's regulatory question.
        bm25: Fitted BM25Okapi index.
        chunks: Chunk dicts in the same order as the BM25 index.
        qdrant_client: Connected QdrantClient instance.
        doc_type: Optional document-type filter ("laws", "regulations", "aml").

    Returns:
        Dict with keys ``question``, ``answer``, and ``sources`` (list of
        dicts with ``document``, ``section``, and ``page`` keys).
    """
    collections_override = _resolve_collections(doc_type)
    results = retrieve(question, bm25, chunks, qdrant_client, collections_override)
    answer = get_answer(question, results)

    sources = [
        {
            "document": r["chunk"]["metadata"].get("source_file", ""),
            "section": r["chunk"]["metadata"].get("section_header", ""),
            "page": r["chunk"]["metadata"].get("page", ""),
        }
        for r in results
    ]

    return {"question": question, "answer": answer, "sources": sources}
