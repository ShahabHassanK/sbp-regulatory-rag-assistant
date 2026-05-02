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

    Uses Gemini Flash (model set in config) as judge to avoid self-grading
    bias.  The ``_GeminiNoTemp`` subclass:
      1. Strips the ``temperature`` kwarg (unsupported by google-generativeai
         0.7.x's GenerativeServiceClient).
      2. Enforces a **thread-safe rate limiter** that guarantees a minimum
         13 s between consecutive calls — keeping throughput below the
         free-tier cap of 5 RPM (= 12 s/req) with 1 s headroom.  This is
         applied globally across all RAGAS worker threads via a class-level
         lock + timestamp.

    Args:
        cfg: Loaded config dict.

    Returns:
        Tuple of (LangchainLLMWrapper, LangchainEmbeddingsWrapper).
    """
    import threading
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    # Free-tier limit for every Gemini model: 5 RPM = 12 s per request.
    # We enforce 13 s minimum interval (1 s safety margin).
    _RPM_INTERVAL = 13.0

    class _GeminiNoTemp(ChatGoogleGenerativeAI):
        """Gemini wrapper that strips temperature and enforces 5-RPM rate limit."""

        _rate_lock: threading.Lock = threading.Lock()
        _last_call_at: float = 0.0

        def _wait_for_slot(self) -> None:
            """Block until at least _RPM_INTERVAL seconds have passed since the
            previous call.  Thread-safe via a class-level lock."""
            with self.__class__._rate_lock:
                elapsed = time.time() - self.__class__._last_call_at
                remaining = _RPM_INTERVAL - elapsed
                if remaining > 0:
                    logger.debug(f"Rate-limiter: sleeping {remaining:.1f}s")
                    time.sleep(remaining)
                self.__class__._last_call_at = time.time()

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.pop("temperature", None)
            self._wait_for_slot()
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.pop("temperature", None)
            self._wait_for_slot()   # still blocks; RAGAS uses threads not asyncio
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    judge_llm = LangchainLLMWrapper(
        _GeminiNoTemp(
            model=cfg["evaluation"]["judge_model"],
            google_api_key=gemini_api_key,
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
# Groq model auto-discovery + token-minimal eval generation
# ---------------------------------------------------------------------------

# Candidate models in preference order (each has its own daily TPD bucket).
# We probe availability at startup and only use models that respond to a
# 1-token test call — this avoids hard-coding names that get decommissioned.
_EVAL_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",          # 100K TPD, high quality
    "llama-3.1-8b-instant",             # 500K TPD, fast
    "llama-3.3-70b-specdec",            # separate budget, speculative decoding
    "llama3-groq-8b-8192-tool-use-preview",   # tool-use preview, separate budget
    "llama3-groq-70b-8192-tool-use-preview",  # 70b tool-use, separate budget
    "llama-3.2-11b-vision-preview",     # vision model, text still works
    "llama-3.2-90b-vision-preview",     # largest vision, text still works
]

# Module-level cache: populated once per process by _discover_eval_models().
_EVAL_MODEL_CACHE: list[str] | None = None


def _discover_eval_models() -> list[str]:
    """Auto-discover which Groq models are live and have remaining TPD budget.

    Strategy:
      1. Fetch the live model list from Groq's /models endpoint (avoids using
         decommissioned names entirely).
      2. Probe each candidate with a 1-token call:
         - Success or RPM/TPM error  → model is active, keep it.
         - TPD-exhausted error       → model active but budget empty; put last.
         - Any other error           → skip (decommissioned or unavailable).
      3. Cache the result for the rest of the process.

    Returns:
        Ordered list of model IDs to try for generation; working models first,
        TPD-exhausted ones last as a last-resort fallback.
    """
    global _EVAL_MODEL_CACHE
    if _EVAL_MODEL_CACHE is not None:
        return _EVAL_MODEL_CACHE

    from groq import Groq, RateLimitError

    client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

    # Step 1: fetch the live model roster from Groq's own API.
    try:
        api_model_ids = {m.id for m in client.models.list().data}
        logger.info(f"Groq reports {len(api_model_ids)} live models")
    except Exception as exc:
        logger.warning(f"Could not fetch Groq model list ({exc}); using candidate list")
        api_model_ids = set(_EVAL_MODEL_CANDIDATES)

    # Intersect candidates with live roster, preserving preference order.
    candidates = [m for m in _EVAL_MODEL_CANDIDATES if m in api_model_ids]
    if not candidates:
        logger.warning("No candidates in live roster — falling back to full candidate list")
        candidates = list(_EVAL_MODEL_CANDIDATES)

    # Step 2: probe each candidate.
    working: list[str] = []
    tpd_exhausted: list[str] = []

    probe_msg = [{"role": "user", "content": "1"}]
    for model in candidates:
        try:
            client.chat.completions.create(
                model=model,
                messages=probe_msg,
                max_tokens=1,
                temperature=0.0,
            )
            working.append(model)
            logger.info(f"  ✓ {model} — ready")
        except RateLimitError as exc:
            err = str(exc).lower()
            if "per day" in err or "tpd" in err:
                # Budget empty but model is live — use as last resort.
                logger.warning(f"  ! {model} — TPD exhausted (keeping as fallback)")
                tpd_exhausted.append(model)
            else:
                # RPM/TPM at probe time means the model IS active right now.
                logger.warning(f"  ~ {model} — RPM limited at probe, treating as working")
                working.append(model)
        except Exception as exc:
            logger.warning(f"  ✗ {model} — unavailable ({type(exc).__name__})")

    ordered = working + tpd_exhausted
    if not ordered:
        logger.error(
            "No Groq models available for eval generation. "
            "All candidates are decommissioned or unreachable."
        )
    else:
        logger.info(f"Eval model order: {ordered}")

    _EVAL_MODEL_CACHE = ordered
    return ordered


def _gemini_generate(question: str, chunks: list[dict]) -> str:
    """Generate a concise answer for RAGAS eval using available Groq models.

    Uses auto-discovered models (probed once at startup). Context is
    intentionally minimal — 2 chunks × 150 characters — to keep token
    usage around 300 per question (~43K for all 144 questions), which fits
    comfortably within the smallest free-tier daily budget (100K TPD).

    Error handling:
      - ``model_decommissioned``:  skip immediately (already filtered at probe,
        but guard against mid-run deprecations).
      - TPD exhausted:             skip to next model.
      - RPM / TPM exceeded:        sleep 20 s then retry the same model once.
      - Any other exception:       skip to next model.

    Args:
        question: The regulatory question.
        chunks: Retrieved chunk dicts with ``text`` and ``metadata`` keys.

    Returns:
        Generated answer string, or empty string if every model fails.
    """
    from groq import Groq, RateLimitError, BadRequestError

    models = _discover_eval_models()
    if not models:
        return ""

    client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

    # Ultra-minimal context: top-2 chunks, 150 chars each.
    # ~75 tokens input per chunk + ~50 token question + 100 output ≈ 300 tokens total.
    top_chunks = chunks[:2]
    context = "\n---\n".join(
        f"[{c['metadata'].get('source_file', '?')} p.{c['metadata'].get('page', '?')}] "
        f"{c['text'][:150]}"
        for c in top_chunks
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"Answer in 1–2 sentences using ONLY the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            ),
        }
    ]

    def _single_call(model: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=100,
        )
        return resp.choices[0].message.content or ""

    for model in models:
        try:
            return _single_call(model)

        except BadRequestError as exc:
            # Catches model_decommissioned 400s that slipped past the probe.
            if "decommissioned" in str(exc).lower():
                logger.warning(f"  ✗ {model} decommissioned mid-run, skipping")
            else:
                logger.warning(f"  ✗ {model} bad request: {exc}")
            continue

        except RateLimitError as exc:
            err = str(exc).lower()
            if "per day" in err or "tpd" in err:
                logger.warning(f"  ! {model} TPD exhausted mid-run, trying next")
                continue
            # RPM / TPM — short sleep then one retry on the same model.
            wait = 20
            logger.warning(f"  ~ {model} RPM/TPM limit, sleeping {wait}s then retrying")
            time.sleep(wait)
            try:
                return _single_call(model)
            except Exception:
                continue

        except Exception as exc:
            logger.warning(f"  ✗ {model} unexpected error: {type(exc).__name__}: {exc}")
            continue

    logger.warning(f"All eval models failed for: '{question[:60]}'")
    return ""


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


# Path where collected rows are cached so scoring can be re-run independently.
EVAL_CHECKPOINT_PATH = EVAL_PAIRS_PATH.parent / "eval_checkpoint.json"


def _collect_eval_rows(
    eval_pairs_path: Path = EVAL_PAIRS_PATH,
    checkpoint_path: Path = EVAL_CHECKPOINT_PATH,
) -> dict[str, list]:
    """Retrieve answers + contexts for every eval pair and cache to disk.

    If ``checkpoint_path`` already exists it is loaded directly, skipping all
    API calls.  Delete the file to force a fresh collection run.

    Args:
        eval_pairs_path: Source eval pairs produced by ``--generate``.
        checkpoint_path: Where to save/load the collected rows.

    Returns:
        Dict with keys ``question``, ``answer``, ``contexts``, ``ground_truth``.
    """
    # ------------------------------------------------------------------ #
    # Fast path: checkpoint already on disk
    # ------------------------------------------------------------------ #
    if checkpoint_path.exists():
        logger.info(f"Loading eval checkpoint from {checkpoint_path} (skipping collection)")
        with open(checkpoint_path, encoding="utf-8") as fh:
            rows = json.load(fh)
        logger.info(f"  Checkpoint contains {len(rows['question'])} rows")
        return rows

    # ------------------------------------------------------------------ #
    # Slow path: call the API for each question
    # ------------------------------------------------------------------ #
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
        "question": [], "answer": [], "contexts": [], "ground_truth": []
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
        time.sleep(4)   # stay within free-tier RPM

    if not rows["question"]:
        logger.error("No eval rows collected — check your pipeline.")
        return rows

    # Save checkpoint so scoring can be re-run without re-collecting.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    logger.info(f"Checkpoint saved → {checkpoint_path}  ({len(rows['question'])} rows)")
    return rows


def _score_eval_rows(rows: dict[str, list]) -> dict[str, float]:
    """Run RAGAS scoring on pre-collected rows and write results to disk.

    Subsamples to ``_SCORE_SAMPLE`` rows (default 50) before scoring so the
    run completes in a predictable time window:

        50 rows × 4 metrics × 13 s/call (5-RPM rate limit) ≈ 44 minutes.

    Uses ``max_workers=1`` so all judge calls go through the thread-safe
    rate-limiter in ``_GeminiNoTemp`` without races.

    Args:
        rows: Dict produced by ``_collect_eval_rows``.

    Returns:
        Dict of metric_name → mean score.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    import random

    cfg = load_config()

    # Subsample for tractable runtime under the 5-RPM free-tier cap.
    _SCORE_SAMPLE = 50
    n = len(rows["question"])
    if n > _SCORE_SAMPLE:
        logger.info(f"Subsampling {_SCORE_SAMPLE}/{n} rows for RAGAS scoring")
        indices = random.sample(range(n), _SCORE_SAMPLE)
        rows = {k: [v[i] for i in indices] for k, v in rows.items()}
    logger.info(f"Scoring {len(rows['question'])} rows with RAGAS")

    dataset = Dataset.from_dict(rows)
    judge_llm, judge_embeddings = _build_ragas_judges(cfg)
    metrics = _build_metrics(judge_llm, judge_embeddings)

    # max_workers=1: serial execution so the class-level rate-limiter in
    # _GeminiNoTemp is the single choke-point — no parallel 429 storms.
    run_cfg = RunConfig(max_workers=1, timeout=180, max_retries=3, max_wait=60)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_cfg,
        raise_exceptions=False,
    )

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


def run_ragas_eval(eval_pairs_path: Path = EVAL_PAIRS_PATH) -> dict:
    """Full pipeline: collect answers then run RAGAS scoring.

    Saves a checkpoint after collection so ``--score`` can re-run scoring
    independently without repeating API calls.

    Args:
        eval_pairs_path: Path to eval_pairs.json produced by ``--generate``.

    Returns:
        Dict of metric_name → mean score.
    """
    rows = _collect_eval_rows(eval_pairs_path)
    if not rows["question"]:
        return {}
    return _score_eval_rows(rows)


def run_ragas_score_only() -> dict:
    """Re-run RAGAS scoring from the saved checkpoint (no API calls).

    Use this after ``--eval`` completed collection but scoring failed.
    Requires ``evals/eval_checkpoint.json`` to exist.

    Returns:
        Dict of metric_name → mean score.
    """
    if not EVAL_CHECKPOINT_PATH.exists():
        print(
            f"ERROR: Checkpoint not found at {EVAL_CHECKPOINT_PATH}.\n"
            "Run  python evaluator.py --eval  first to collect data."
        )
        sys.exit(1)
    logger.info(f"Loading checkpoint from {EVAL_CHECKPOINT_PATH}")
    with open(EVAL_CHECKPOINT_PATH, encoding="utf-8") as fh:
        rows = json.load(fh)
    logger.info(f"  {len(rows['question'])} rows loaded — starting RAGAS scoring")
    return _score_eval_rows(rows)


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
    parser = argparse.ArgumentParser(
        description="SBP RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical workflow:\n"
            "  1. python evaluator.py --generate          # build eval_pairs.json\n"
            "  2. python evaluator.py --eval              # collect answers + score\n"
            "     (if scoring fails, fix the judge and re-run without re-collecting)\n"
            "  3. python evaluator.py --score             # re-score from checkpoint\n"
            "  4. python evaluator.py --ablate            # ablation study\n"
        ),
    )
    parser.add_argument("--generate", action="store_true", help="Generate eval pairs from chunks")
    parser.add_argument("--eval",     action="store_true", help="Collect answers + run RAGAS scoring (saves checkpoint)")
    parser.add_argument("--score",    action="store_true", help="Re-run RAGAS scoring from saved checkpoint (no API calls)")
    parser.add_argument("--ablate",   action="store_true", help="Run ablation study")
    args = parser.parse_args()

    if not any([args.generate, args.eval, args.score, args.ablate]):
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

    if args.score:
        run_ragas_score_only()

    if args.ablate:
        if not EVAL_PAIRS_PATH.exists():
            print(f"ERROR: {EVAL_PAIRS_PATH} not found. Run --generate first.")
            sys.exit(1)
        run_ablation()
