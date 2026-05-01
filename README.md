# 🏛️ SBP Regulatory RAG Assistant

> A production-grade **Retrieval-Augmented Generation** system over 42 State Bank of Pakistan regulatory documents — featuring hybrid BM25 + vector search, Cohere reranking, a FastAPI backend, a Streamlit chat UI, and an automated RAGAS evaluation pipeline.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Screenshots

### Landing Page
![Landing Page](screenshots/landingpage.PNG)

### Conversation — Regulatory Query
![Conversation Log 1](screenshots/conversationlog1.PNG)

### Conversation — AML / Compliance Query
![Conversation Log 2](screenshots/conversationlog2.PNG)

---

## 📋 Overview

The **SBP Regulatory RAG Assistant** enables banking professionals and compliance teams to query Pakistan's banking regulatory corpus in natural language and receive cited, grounded answers. Every response is traceable to a specific document, section, and page number.

**Corpus coverage:**
| Category | Examples |
|---|---|
| Banking Laws | SBP Act, Banking Companies Ordinance, FICA |
| Prudential Regulations | Corporate, SME, Consumer, Housing Finance, MFBs |
| AML-CFT-CPF Regulations | Customer Due Diligence, Wire Transfers, PEPs, Sanctions |
| Notifications | SBP Circulars, Policy Directives |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│               FastAPI Backend               │
│                                             │
│  ┌──────────┐   ┌──────────┐  ┌─────────┐  │
│  │  BM25    │   │  Qdrant  │  │ Cohere  │  │
│  │ (sparse) │ + │ (dense)  │→ │ Rerank  │  │
│  └──────────┘   └──────────┘  └────┬────┘  │
│       └──────── RRF Fusion ────────┘        │
│                                             │
│  Top-K Chunks → Groq LLaMA 3.3 70B         │
│                   → Grounded Answer         │
└─────────────────────────────────────────────┘
    │
    ▼
Streamlit UI  (chat history · source expanders · doc filters)
```

**Retrieval pipeline (3 stages):**
1. **BM25 lexical search** — exact keyword matching over all chunks
2. **Dense vector search** — semantic similarity via `all-mpnet-base-v2` embeddings in Qdrant
3. **Reciprocal Rank Fusion → Cohere Rerank** — fuses both lists, then Cohere re-scores top candidates

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM (Generation)** | Groq · `llama-3.3-70b-versatile` |
| **Embeddings** | `sentence-transformers/all-mpnet-base-v2` (local) |
| **Vector Store** | Qdrant (local-path mode) |
| **Sparse Retrieval** | BM25Okapi (`rank-bm25`) |
| **Reranking** | Cohere Rerank v3 |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit (custom CSS, SBP green theme) |
| **Evaluation** | RAGAS 0.2.x · Gemini 1.5 Flash (judge) |
| **PDF Parsing** | pdfplumber + pytesseract (OCR fallback) |
| **Config** | YAML (`config/config.yaml`) |

---

## 📁 Project Structure

```
sbp_rag/
├── api/
│   ├── main.py          # FastAPI app — /query, /retrieve, /health endpoints
│   └── run.py           # Uvicorn entry point
├── config/
│   └── config.yaml      # All tunable parameters (models, paths, retrieval k)
├── data/
│   ├── laws/            # SBP Act, BCO, FICA, NAB Ordinance, etc.
│   ├── regulations/     # Prudential regs, AML-CFT-CPF, other
│   └── notifications/   # SBP circulars
├── evals/
│   ├── eval_pairs.json  # 144 auto-generated Q&A evaluation pairs
│   └── results/         # RAGAS metric outputs (gitignored)
├── screenshots/         # UI screenshots for README
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py   # PDF → raw pages JSONL
│   │   ├── chunker.py      # Sliding-window sentence chunker
│   │   └── embedder.py     # Encode chunks → Qdrant collections
│   ├── retrieval/
│   │   └── retriever.py    # BM25 + vector search + RRF + Cohere rerank
│   ├── generation/
│   │   └── rag_chain.py    # Prompt builder + Groq generation + streaming
│   ├── evaluation/
│   │   └── evaluator.py    # RAGAS eval pipeline (generate / eval / ablate)
│   └── utils/
│       └── config.py       # YAML config loader
├── ui/
│   └── app.py           # Streamlit chat application
├── .env.example         # Required environment variable template
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.11 |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Add to system PATH |
| [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) | Add `bin/` to system PATH |
| Groq API key | [console.groq.com](https://console.groq.com) — free tier |
| Cohere API key | [dashboard.cohere.com](https://dashboard.cohere.com) — free tier |
| Gemini API key | [aistudio.google.com](https://aistudio.google.com) — for RAGAS judge |

### 1. Clone & install

```bash
git clone https://github.com/ShahabHassanK/sbp-regulatory-rag-assistant.git
cd sbp-regulatory-rag-assistant

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
GEMINI_API_KEY=AIza...
```

### 3. Ingest documents

> Run once. Wipes and rebuilds Qdrant collections each time.

```bash
python src/ingestion/pdf_parser.py   # PDF → raw_pages.jsonl  (~5 min)
python src/ingestion/chunker.py      # Pages → chunks.jsonl   (~1 min)
python src/ingestion/embedder.py     # Chunks → Qdrant        (~10 min)
```

---

## 🚀 Running the Application

Open **two terminals** in the project root with the venv activated.

**Terminal 1 — API backend:**
```bash
python api/run.py
# Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run ui/app.py
# Open http://localhost:8501
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Full RAG query — retrieves + generates answer |
| `POST` | `/query/stream` | Streaming version (SSE) |
| `GET` | `/retrieve` | Retrieval-only — returns raw chunks, no LLM |
| `GET` | `/health` | Service liveness + per-collection vector counts |

**Example `/query` request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum capital requirement for a microfinance bank?", "doc_type": null}'
```

**Example response:**
```json
{
  "answer": "According to SBP Prudential Regulations for MFBs, the minimum paid-up capital ...",
  "sources": [
    { "document": "PR_MFBs.pdf", "section": "Regulation R-1", "page": 3 }
  ]
}
```

---

## 📊 Evaluation (RAGAS)

The evaluation pipeline uses **Gemini 1.5 Flash** as a judge LLM (separate from the Groq generator to avoid self-grading bias) and scores 4 RAGAS metrics across 144 auto-generated Q&A pairs.

> ⚠️ **Requires the API to be running** (`python api/run.py`) before executing eval/ablate commands.

```bash
# Step 1 — Generate 144 Q&A eval pairs from your corpus (~5 min, uses Groq)
python src/evaluation/evaluator.py --generate

# Step 2 — Run RAGAS scoring (~30 min, uses Gemini as judge)
python src/evaluation/evaluator.py --eval

# Step 3 — Ablation study: Laws-only vs Regulations-only vs All docs
python src/evaluation/evaluator.py --ablate
```

### RAGAS Metrics

| Metric | Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |
| **Context Recall** | Does the context cover what's needed to answer? |

> Results are saved to `evals/results/eval_YYYYMMDD_HHMMSS.json`.

---

## 🔧 Configuration

All parameters are in [`config/config.yaml`](config/config.yaml):

```yaml
llm:
  model: "llama-3.3-70b-versatile"   # Groq model for generation
  temperature: 0.1
  max_tokens: 1024

retrieval:
  bm25_top_k: 20          # BM25 candidates
  vector_top_k: 20        # Dense vector candidates
  rrf_k: 60               # RRF fusion constant
  rerank_top_n: 5         # Final chunks after Cohere rerank

evaluation:
  n_eval_pairs: 50        # Pairs per --generate run
  judge_model: "gemini-1.5-flash-latest"
```

---

## ⚠️ Known Constraints

| Constraint | Detail |
|---|---|
| **Qdrant single-process lock** | Local-path Qdrant cannot be opened by two processes simultaneously. The evaluator uses `/retrieve` HTTP endpoint to avoid conflict. |
| **Groq free tier** | 30 RPM, 6K TPM for 70B model. Use `llama-3.1-8b-instant` (131K TPM) for bulk eval. |
| **Gemini v1beta deprecation** | `langchain-google-genai==1.0.x` uses v1beta; use model name `gemini-1.5-flash-latest`. |
| **Single Uvicorn worker** | Do not run with `--workers > 1` — Qdrant local-path is not thread-safe. |

---

## 📄 License

MIT © 2026 [Shahab Hassan](https://github.com/ShahabHassanK)
