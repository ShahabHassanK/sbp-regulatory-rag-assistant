"""Verify the Groq and Cohere API keys are valid (Gemini no longer used for embeddings).

Run from the project root with the venv active:
    python tests/test_gemini_key.py
"""

import os
import sys
from dotenv import load_dotenv
import requests

load_dotenv()


def check_groq() -> None:
    print("--- Groq API key ---")
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        print("FAIL  GROQ_API_KEY not set in .env")
        return
    r = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=10,
    )
    if r.ok:
        models = [m["id"] for m in r.json().get("data", [])]
        llama = [m for m in models if "llama" in m.lower()]
        print(f"OK    Key valid. Llama models available: {llama[:3]}")
    else:
        print(f"FAIL  {r.status_code}: {r.text[:200]}")


def check_cohere() -> None:
    print("\n--- Cohere API key ---")
    key = os.getenv("COHERE_API_KEY", "")
    if not key:
        print("FAIL  COHERE_API_KEY not set in .env")
        return
    try:
        import cohere
        co = cohere.ClientV2(api_key=key)
        # Embed one sentence to verify access
        resp = co.embed(
            texts=["test"],
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )
        dims = len(resp.embeddings.float[0])
        print(f"OK    Key valid. embed-english-v3.0 returned {dims}-dim vector.")
    except Exception as exc:
        print(f"FAIL  {exc}")


def check_sentence_transformers() -> None:
    print("\n--- sentence-transformers (local embedding model) ---")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")
        vec = model.encode(["test sentence"])
        print(f"OK    Model loaded. Embedding dim = {vec.shape[1]}")
    except Exception as exc:
        print(f"FAIL  {exc}")


if __name__ == "__main__":
    check_groq()
    check_cohere()
    check_sentence_transformers()
    print("\nDone.")
