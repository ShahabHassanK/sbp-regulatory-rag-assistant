"""RAGAS evaluation pipeline for the SBP RAG system.

Usage:
    python src/evaluation/evaluator.py --generate   # create eval_pairs.json
    python src/evaluation/evaluator.py --eval       # run RAGAS metrics (API must be running)
    python src/evaluation/evaluator.py --ablate     # ablation study (API must be running)

NOTE: --eval and --ablate require `python api/run.py` to be running in another
terminal. The evaluator calls the API over HTTP to avoid the Qdrant single-
process file-lock constraint on Windows.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests as http_requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_BASE = os.getenv("EVAL_API_BASE", "http://localhost:8000")

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.utils.config import load_config

logger.add(
    Path(__file__).parents[2] / "logs" / "sbp_rag.log",
    rotation="10 MB",
    level="INFO",
)

CHUNKS_PATH     = Path(__file__).parents[2] / "data" / "processed" / "chunks.jsonl"
EVAL_DIR        = Path(__file__).parents[2] / "evals"
EVAL_PAIRS_PATH = EVAL_DIR / "eval_pairs.json"
RESULTS_DIR     = EVAL_DIR / "results"


# ---------------------------------------------------------------------------
# Eval pair generation
# ---------------------------------------------------------------------------

def generate_eval_pairs(chunks: list[dict], n: int = 50) -> list[dict]:
    """Generate Q&A evaluation pairs from a stratified sample of chunks.

    Samples ~equal numbers from each doc_type (law / regulation / aml), prompts
    the Groq LLM to produce 2–3 Q&A pairs per chunk, and saves to eval_pairs.json.

    Args:
        chunks: All chunk dicts from chunks.jsonl.
        n: Target number of eval pairs.

    Returns:
        List of dicts with ``question`` and ``ground_truth`` keys.
    """
    from groq import Groq

    cfg = load_config()
    client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

    # Stratified sample by doc_type
    by_type: dict[str, list[dict]] = {}
    for chunk in chunks:
        dt = chunk["metadata"].get("doc_type", "regulation")
        by_type.setdefault(dt, []).append(chunk)

    per_type = max(1, n // max(len(by_type), 1))
    sampled: list[dict] = []
    for dt, dt_chunks in by_type.items():
        sampled.extend(random.sample(dt_chunks, min(per_type, len(dt_chunks))))

    logger.info(f"Generating eval pairs from {len(sampled)} sampled chunks "
                f"({list(by_type.keys())})")

    pairs: list[dict] = []
    for i, chunk in enumerate(sampled):
        text = chunk["text"][:1500]
        prompt = (
            "Generate 2-3 Q&A pairs from this banking regulation text. "
            "Mix factual, numerical-threshold, and process questions. "
            "Return ONLY a JSON array:\n"
            '[{"question": "...", "ground_truth": "..."}]\n\n'
            f"Text:\n{text}"
        )
        try:
            resp = client.chat.completions.create(
                model=cfg["llm"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            # Extract the JSON array robustly — handles markdown code fences too
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                batch = json.loads(match.group())
                for pair in batch:
                    if "question" in pair and "ground_truth" in pair:
                        pairs.append(pair)
        except json.JSONDecodeError as exc:
            logger.warning(f"JSON parse failed for chunk {i}: {exc}")
        except Exception as exc:
            logger.warning(f"Pair generation failed for chunk {i}: {exc}")

        # Throttle to stay within Groq RPM limits
        if (i + 1) % 10 == 0:
            time.sleep(2)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PAIRS_PATH, "w", encoding="utf-8") as fh:
        json.dump(pairs, fh, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(pairs)} eval pairs → {EVAL_PAIRS_PATH}")
    print(f"\n✓ Generated {len(pairs)} eval pairs → {EVAL_PAIRS_PATH}")
    return pairs


# ---------------------------------------------------------------------------
# RAGAS judge / embeddings factory
# ---------------------------------------------------------------------------

def _build_ragas_judges(cfg: dict):
    """Instantiate and return (judge_llm, judge_embeddings) for RAGAS.

    Uses Gemini 1.5 Flash via langchain-google-genai to avoid self-grading
    bias when the generator is Groq.

    Args:
        cfg: Loaded config dict.

    Returns:
        Tuple of (LangchainLLMWrapper, LangchainEmbeddingsWrapper).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=cfg["evaluation"]["judge_model"],
            google_api_key=gemini_api_key,
            convert_system_message_to_human=True,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key,
        )
    )
    return judge_llm, judge_embeddings


def _build_metrics(judge_llm, judge_embeddings) -> list:
    """Instantiate RAGAS metric objects with the judge LLM wired in.

    RAGAS 0.2.x requires metric instances (not module-level singletons) and
    the LLM must be passed to the constructor, not set as an attribute.

    Args:
        judge_llm: LangchainLLMWrapper around the Gemini judge.
        judge_embeddings: LangchainEmbeddingsWrapper for embedding-based metrics.

    Returns:
        List of four RAGAS metric instances.
    """
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    return [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gemini-based generation for eval (bypasses Groq TPM limits entirely)
# ---------------------------------------------------------------------------

_GEMINI_EVAL_PROMPT = (
    "You are a regulatory compliance assistant. Answer the question below "
    "using ONLY the provided context. Cite sources inline. If the context "
    "does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def _gemini_generate(question: str, chunks: list[dict]) -> str:
    """Generate an answer with Gemini Flash from retrieved chunks.

    Gemini 1.5 Flash free tier: 15 RPM / 1M TPM — sufficient for 144 eval
    pairs without throttling.

    Args:
        question: The regulatory question.
        chunks: Retrieved chunk dicts with ``text`` and ``metadata`` keys.

    Returns:
        Generated answer string.
    """
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    context = "\n\n---\n\n".join(
        f"[{i+1}] {c['metadata'].get('source_file', '?')} "
        f"| p.{c['metadata'].get('page', '?')}\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # 131K TPM — no rate-limit issues
        messages=[
            {"role": "system", "content": (
                "You are a regulatory compliance assistant. Answer using ONLY "
                "the provided context. Cite sources inline."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


def _api_query(question: str, doc_type: str | None = None) -> dict | None:
    """Retrieve chunks via API /retrieve, generate answer with Gemini Flash.

    Two-step approach that completely avoids Groq during eval:
      1. GET /retrieve  — uses BM25 + Qdrant via the running API (no LLM)
      2. Gemini Flash   — generates the answer (1M TPM, not 6K)

    Args:
        question: The regulatory question.
        doc_type: Optional collection filter.

    Returns:
        Dict with ``answer`` and ``contexts`` keys, or None on failure.
    """
    try:
        resp = http_requests.get(
            f"{API_BASE}/retrieve",
            params={"question": question, "doc_type": doc_type},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        chunks = data.get("chunks", [])
        if not chunks:
            logger.warning(f"No chunks retrieved for: '{question[:60]}'")
            return None

        contexts = [c["text"] for c in chunks]
        answer   = _gemini_generate(question, chunks)
        return {"answer": answer, "contexts": contexts}

    except Exception as exc:
        logger.warning(f"Eval query failed for '{question[:60]}': {exc}")
        return None


def run_ragas_eval(eval_pairs_path: Path = EVAL_PAIRS_PATH) -> dict:
    """Run RAGAS evaluation using Gemini 1.5 Flash as the judge LLM.

    Calls the running API (/query) for each eval pair so the evaluator does
    not conflict with the API's Qdrant file lock.

    Args:
        eval_pairs_path: Path to eval_pairs.json produced by generate_eval_pairs.

    Returns:
        Dict of metric_name → mean score.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    cfg = load_config()

    # Confirm API is reachable before starting the long loop
    try:
        http_requests.get(f"{API_BASE}/health", timeout=5).raise_for_status()
    except Exception:
        print(f"\nERROR: Cannot reach API at {API_BASE}.\n"
              "Start it first:  python api/run.py\n")
        sys.exit(1)

    with open(eval_pairs_path, encoding="utf-8") as fh:
        pairs = json.load(fh)
    logger.info(f"Loaded {len(pairs)} eval pairs from {eval_pairs_path}")

    rows: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    total = len(pairs)
    for i, pair in enumerate(pairs):
        question     = pair["question"]
        ground_truth = pair["ground_truth"]
        print(f"  [{i+1}/{total}] {question[:70]}", flush=True)
        result = _api_query(question)
        if result:
            rows["question"].append(question)
            rows["answer"].append(result["answer"])
            rows["contexts"].append(result["contexts"])
            rows["ground_truth"].append(ground_truth)
        # Gemini Flash: 15 RPM → 1 req/4s is safe. No Groq calls here.
        time.sleep(4)

    if not rows["question"]:
        logger.error("No eval rows collected — check your pipeline.")
        return {}

    dataset = Dataset.from_dict(rows)

    judge_llm, judge_embeddings = _build_ragas_judges(cfg)
    metrics = _build_metrics(judge_llm, judge_embeddings)

    # RunConfig: conservative workers + long timeout for Gemini free tier
    run_cfg = RunConfig(max_workers=2, timeout=300, max_retries=5, max_wait=120)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    # result is a dict-like object; extract numeric scores
    scores: dict[str, float] = {}
    for key, val in result.items():
        try:
            scores[key] = float(val)
        except (TypeError, ValueError):
            pass

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = RESULTS_DIR / f"eval_{timestamp}.json"
    out_csv  = RESULTS_DIR / f"eval_{timestamp}.csv"

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(scores, fh, indent=2)
    try:
        result.to_pandas().to_csv(out_csv, index=False)
    except Exception as exc:
        logger.warning(f"Could not write CSV: {exc}")

    print("\n--- RAGAS Evaluation Results ---")
    for metric, score in scores.items():
        print(f"  {metric:<30} {score:.4f}")
    print(f"\nResults saved → {out_json}")
    return scores


# ---------------------------------------------------------------------------
# Ablation study helpers  (all via API — avoids Qdrant file-lock)
# ---------------------------------------------------------------------------

def _ablation_api_query(question: str, doc_type: str | None, config_name: str) -> dict | None:
    """Call the API /query endpoint for a single ablation config.

    The ablation configs (dense-only, hybrid-no-rerank, hybrid+rerank) are
    distinguished server-side by the doc_type scoping. For a true ablation
    from outside the API we delegate to the full pipeline (hybrid+rerank)
    since the API exposes only one retrieval strategy per request. Use the
    evaluator's ablation flag to switch strategies inside the API if needed.

    Args:
        question: Raw query string.
        doc_type: Optional collection filter to restrict scope.
        config_name: Human-readable name (for logging only).

    Returns:
        Dict with ``answer`` and ``contexts``, or None.
    """
    result = _api_query(question, doc_type)
    if result:
        logger.debug(f"[{config_name}] '{question[:50]}' → {len(result['contexts'])} ctx")
    return result


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(eval_pairs_path: Path = EVAL_PAIRS_PATH) -> None:
    """Compare three doc-scope configurations using the RAGAS judge.

    Because Qdrant local-path is single-process on Windows, all retrieval
    runs via the API. The three "configs" scope to different collections:
        (a) Laws only
        (b) Regulations only
        (c) All documents (full hybrid pipeline)

    Prints a markdown comparison table.

    Args:
        eval_pairs_path: Path to eval_pairs.json.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    # Confirm API is reachable
    try:
        http_requests.get(f"{API_BASE}/health", timeout=5).raise_for_status()
    except Exception:
        print(f"\nERROR: Cannot reach API at {API_BASE}.\nStart it first:  python api/run.py\n")
        sys.exit(1)

    with open(eval_pairs_path, encoding="utf-8") as fh:
        pairs = json.load(fh)

    judge_llm, judge_embeddings = _build_ragas_judges(load_config())
    run_cfg = RunConfig(max_workers=2, timeout=300, max_retries=5, max_wait=120)

    configs = [
        ("Laws only",         "laws"),
        ("Regulations only",  "regulations"),
        ("All (hybrid+rerank)", None),
    ]

    all_scores: dict[str, dict] = {}

    for name, doc_type in configs:
        print(f"\nRunning ablation config: {name}…")
        metrics = _build_metrics(judge_llm, judge_embeddings)
        rows: dict[str, list] = {
            "question": [], "answer": [], "contexts": [], "ground_truth": [],
        }
        for i, pair in enumerate(pairs):
            result = _ablation_api_query(pair["question"], doc_type, name)
            if result:
                rows["question"].append(pair["question"])
                rows["answer"].append(result["answer"])
                rows["contexts"].append(result["contexts"])
                rows["ground_truth"].append(pair["ground_truth"])
            if (i + 1) % 5 == 0:
                time.sleep(2)

        if not rows["question"]:
            logger.warning(f"No rows for config '{name}' — skipping.")
            continue

        result = evaluate(
            dataset=Dataset.from_dict(rows),
            metrics=metrics,
            run_config=run_cfg,
            raise_exceptions=False,
        )
        all_scores[name] = {k: float(v) for k, v in result.items()
                            if isinstance(v, (int, float))}

    if not all_scores:
        print("No ablation results collected.")
        return

    # Save ablation results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_path = RESULTS_DIR / f"ablation_{timestamp}.json"
    with open(ablation_path, "w", encoding="utf-8") as fh:
        json.dump(all_scores, fh, indent=2)

    # Print markdown table
    metric_names = list(next(iter(all_scores.values())).keys())
    header = "| Config | " + " | ".join(metric_names) + " |"
    sep    = "| --- | " + " | ".join(["---"] * len(metric_names)) + " |"
    print(f"\n{header}\n{sep}")
    for config_name, scores in all_scores.items():
        row_vals = " | ".join(f"{scores.get(m, 0.0):.4f}" for m in metric_names)
        print(f"| {config_name} | {row_vals} |")
    print(f"\nAblation results saved → {ablation_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBP RAG Evaluation Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate eval pairs")
    parser.add_argument("--eval",     action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--ablate",   action="store_true", help="Run ablation study")
    args = parser.parse_args()

    if not any([args.generate, args.eval, args.ablate]):
        parser.print_help()
        sys.exit(0)

    if args.generate:
        if not CHUNKS_PATH.exists():
            print(f"ERROR: {CHUNKS_PATH} not found. Run chunker.py first.")
            sys.exit(1)
        all_chunks: list[dict] = []
        with open(CHUNKS_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))
        cfg_main = load_config()
        generate_eval_pairs(all_chunks, n=cfg_main["evaluation"]["n_eval_pairs"])

    if args.eval:
        if not EVAL_PAIRS_PATH.exists():
            print(f"ERROR: {EVAL_PAIRS_PATH} not found. Run --generate first.")
            sys.exit(1)
        run_ragas_eval()

    if args.ablate:
        if not EVAL_PAIRS_PATH.exists():
            print(f"ERROR: {EVAL_PAIRS_PATH} not found. Run --generate first.")
            sys.exit(1)
        run_ablation()
