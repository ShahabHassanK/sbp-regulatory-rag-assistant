"""Embedding and Qdrant ingestion pipeline.

Uses sentence-transformers (all-mpnet-base-v2, 768-dim) running locally —
no API key or rate limits required. Loads chunks.jsonl, embeds each chunk,
and upserts vectors + metadata into three Qdrant collections.

The model is downloaded once on first run (~420 MB) and cached at:
    ~/.cache/huggingface/hub/
Subsequent runs are instant (loaded from disk).
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from uuid import uuid4

load_dotenv()

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.utils.config import load_config

logger.add(
    Path(__file__).parents[2] / "logs" / "ingestion.log",
    rotation="10 MB",
    level="INFO",
)

CHUNKS_PATH = Path(__file__).parents[2] / "data" / "processed" / "chunks.jsonl"

# Module-level singleton — model is loaded once and reused by all callers,
# including the retriever (which imports embed_batch at API startup).
_model: SentenceTransformer | None = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_embedding_model() -> SentenceTransformer:
    """Load (or return cached) the sentence-transformers model from config.

    The model is downloaded on first call and cached by HuggingFace in
    ~/.cache/huggingface/hub/. All subsequent calls return the same object.

    Returns:
        A loaded SentenceTransformer ready for .encode() calls.
    """
    global _model
    if _model is None:
        cfg = load_config()
        model_name = cfg["embedding"]["model"]
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info(f"Model loaded — dim={_model.get_embedding_dimension()}")
    return _model


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    """Embed a list of texts using the local sentence-transformers model.

    The task_type parameter is accepted for API compatibility with retriever.py
    but has no effect — all-mpnet-base-v2 handles query/document similarity
    symmetrically (no separate task-type encoding needed).

    Args:
        texts: List of strings to embed.
        task_type: Ignored; kept for interface compatibility.

    Returns:
        List of float lists, one 768-dim vector per input text.
    """
    model = get_embedding_model()
    # convert_to_numpy=False gives plain Python lists directly
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False,
        convert_to_numpy=False,
    )
    return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# Qdrant collection setup
# ---------------------------------------------------------------------------

def setup_qdrant_collections(client: QdrantClient, cfg: dict) -> None:
    """Recreate the three Qdrant collections defined in config.yaml.

    Deletes existing collections first so ingestion is always a clean slate.

    Args:
        client: Initialised QdrantClient.
        cfg: Loaded config dict.
    """
    dim = cfg["embedding"]["dimensions"]
    for _label, collection_name in cfg["qdrant"]["collections"].items():
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {collection_name} (dim={dim}, cosine)")


def get_collection_for_chunk(chunk: dict, cfg: dict) -> str:
    """Map a chunk's doc_type to its Qdrant collection name.

    Args:
        chunk: Chunk dict with metadata.doc_type set by the parser.
        cfg: Loaded config dict.

    Returns:
        Qdrant collection name string.
    """
    doc_type = chunk["metadata"].get("doc_type", "regulation")
    cols = cfg["qdrant"]["collections"]
    mapping = {
        "law":          cols["laws"],
        "aml":          cols["aml"],
        "regulation":   cols["regulations"],
        "notification": cols["regulations"],
    }
    return mapping.get(doc_type, cols["regulations"])


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_chunks(chunks: list[dict]) -> QdrantClient:
    """Embed all chunks and upsert them into Qdrant.

    Groups chunks by their target collection, then encodes them in batches
    using the local sentence-transformers model. Each Qdrant point stores:
        - A UUID string as the point id
        - The 768-dim embedding vector
        - Full metadata + original text as the payload (used for citations)

    Args:
        chunks: List of chunk dicts from the chunking phase.

    Returns:
        The initialised QdrantClient (passed to verify_ingestion).
    """
    cfg = load_config()

    # Load model before opening Qdrant (model load can take a few seconds)
    get_embedding_model()

    client = QdrantClient(path=cfg["qdrant"]["path"])
    setup_qdrant_collections(client, cfg)

    batch_size = cfg["embedding"]["batch_size"]

    # Group chunks by collection
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        col = get_collection_for_chunk(chunk, cfg)
        groups.setdefault(col, []).append(chunk)

    total_upserted = 0

    for collection_name, col_chunks in groups.items():
        logger.info(f"Embedding + ingesting {len(col_chunks)} chunks → '{collection_name}'")

        for batch_start in tqdm(
            range(0, len(col_chunks), batch_size),
            desc=f"  {collection_name}",
            unit="batch",
        ):
            batch = col_chunks[batch_start : batch_start + batch_size]
            texts = [c["text"] for c in batch]

            try:
                vectors = embed_batch(texts)
            except Exception as exc:
                logger.error(f"Embedding failed at offset {batch_start}: {exc}")
                continue

            points = [
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={**chunk["metadata"], "text": chunk["text"]},
                )
                for chunk, vector in zip(batch, vectors)
            ]
            client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)

    print(f"\n✓ Ingestion complete — {total_upserted} points upserted")
    for col_name in cfg["qdrant"]["collections"].values():
        info = client.get_collection(col_name)
        print(f"  {col_name:<25} {info.points_count} points")

    return client


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_ingestion(client: QdrantClient) -> None:
    """Run a test query against all three collections to confirm retrieval works.

    Args:
        client: The QdrantClient returned by ingest_chunks().
    """
    cfg = load_config()
    test_query = "capital adequacy requirements for banks"
    logger.info(f"Verifying with test query: '{test_query}'")

    query_vector = embed_batch([test_query], task_type="retrieval_query")[0]

    print(f"\n--- Verification: '{test_query}' ---")
    for col_name in cfg["qdrant"]["collections"].values():
        results = client.search(
            collection_name=col_name,
            query_vector=query_vector,
            limit=3,
            with_payload=True,
        )
        print(f"\n  [{col_name}]")
        for i, hit in enumerate(results, 1):
            p = hit.payload or {}
            print(
                f"    {i}. {p.get('source_file', 'unknown')} "
                f"| p.{p.get('page', '?')} "
                f"| score={hit.score:.4f}"
            )
            print(f"       {p.get('text', '')[:120].strip()}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found. Run chunker.py first.")
        sys.exit(1)

    chunks: list[dict] = []
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks")
    client = ingest_chunks(chunks)
    verify_ingestion(client)
