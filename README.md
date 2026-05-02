# SBP Regulatory RAG Assistant

A production-grade Retrieval-Augmented Generation system over 42 State Bank of Pakistan regulatory PDFs.

## Prerequisites

### 1. Python 3.11
```bash
python --version   # must be 3.11.x
```

### 2. Tesseract OCR (required for scanned PDFs)
Download the Windows installer from:
https://github.com/UB-Mannheim/tesseract/wiki

During installation, **check "Add Tesseract to system PATH"**.

Verify:
```bash
tesseract --version
```

### 3. Poppler (required by unstructured for PDF→image conversion)
1. Download Windows binaries: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\poppler`
3. Add `C:\poppler\Library\bin` to your system PATH (System Properties → Advanced → Environment Variables)

Verify:
```bash
pdftotext -v
```

### 4. API Keys
Copy `.env.example` to `.env` and fill in your keys:
```bash
copy .env.example .env
```
- **Gemini**: https://aistudio.google.com/apikey (free)
- **Groq**: https://console.groq.com/keys (free)
- **Cohere**: https://dashboard.cohere.com/api-keys (trial)

---

## Setup

```bash
# Create venv inside api/
python -m venv api/.venv

# Activate it
api\.venv\Scripts\activate      # Windows CMD/PowerShell
# source api/.venv/bin/activate  # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

> **Note:** All project scripts must be run with this venv active and from the project root (`E:/Personal Projects/sbp_rag/`).

---

## Run Order (first time)

Run each step from the project root with the venv activated:

```bash
# Phase 2 — Parse all 42 PDFs
python src/ingestion/pdf_parser.py

# Phase 3 — Chunk parsed pages
python src/ingestion/chunker.py

# Phase 4 — Embed chunks and ingest into Qdrant
python src/ingestion/embedder.py

# Phase 5 — Run retrieval tests
pytest tests/test_retrieval.py -v

# Phase 6 — Start the API backend
python api/run.py

# Phase 7 — Start the UI (in a separate terminal)
streamlit run ui/app.py

# Phase 7 — Generate eval pairs and run RAGAS evaluation
python src/evaluation/evaluator.py --generate
python src/evaluation/evaluator.py --eval
python src/evaluation/evaluator.py --score
python src/evaluation/evaluator.py --ablate
```

---

## Architecture

```
User question
    │
    ▼
Streamlit UI  ──HTTP──▶  FastAPI  (/query)
                              │
                    ┌─────────▼──────────┐
                    │   Hybrid Retrieval  │
                    │  BM25 + Vector RRF  │
                    │  Cohere Reranker    │
                    └─────────┬──────────┘
                              │ top 5 chunks
                    ┌─────────▼──────────┐
                    │  Groq LLM          │
                    │  llama-3.3-70b     │
                    └─────────┬──────────┘
                              │
                    Cited answer + sources
```

### Qdrant Collections
| Collection | Contents |
|---|---|
| `sbp_laws` | SBP Act, Banking Companies Ordinance, Credit Bureau Act, etc. |
| `sbp_regulations` | Prudential Regulations (SME/Housing/Corporate/Microfinance), Other Regulations |
| `sbp_aml` | AML/CFT/CPF Regulations, TFS Guidelines, Risk-Based Approach |

### Important Constraints
- **Qdrant runs in local-path mode** (`E:/qdrant_storage`) — single process only. Do not run ingestion while the API is running, and do not run uvicorn with `--workers > 1`.
- **Gemini free tier:** 1500 embedding requests/day. If ingestion fails mid-way, wait 24h before re-running (or the next run will quota-error immediately).
- **Cohere trial:** 5 RPM / 1000 calls/month. Reranker falls back to RRF top-5 if rate-limited.

---

## Project Structure

```
sbp_rag/
├── api/
│   ├── .venv/          ← virtual environment (all packages installed here)
│   ├── main.py         ← FastAPI application
│   └── run.py          ← entry point: python api/run.py
├── config/
│   └── config.yaml     ← all tunable parameters
├── data/
│   ├── laws/           ← SBP Act, ordinances, etc. (19 PDFs)
│   ├── regulations/
│   │   ├── prudential regulations/   (11 PDFs)
│   │   ├── AML-CFT-CPF regulations/ (5 PDFs)
│   │   └── other regulations/       (6 PDFs)
│   ├── notifications/  (1 PDF)
│   └── processed/      ← generated: raw_pages.jsonl, chunks.jsonl
├── evals/
│   ├── eval_pairs.json
│   └── results/        ← RAGAS eval outputs
├── logs/               ← loguru rotating log files
├── src/
│   ├── ingestion/      ← pdf_parser.py, chunker.py, embedder.py
│   ├── retrieval/      ← retriever.py
│   ├── generation/     ← rag_chain.py
│   ├── evaluation/     ← evaluator.py
│   └── utils/          ← config.py
├── tests/
│   └── test_retrieval.py
├── ui/
│   └── app.py          ← Streamlit chat interface
├── .env                ← your API keys (never commit)
├── .env.example
├── .gitignore
└── requirements.txt
```
