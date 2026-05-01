"""Entry point: boots the SBP RAG FastAPI server with uvicorn.

Run with:
    python api/run.py

Notes:
- workers=1 is required — Qdrant local-path mode is single-process only.
- reload=False: the app loads a sentence-transformer model at startup (~30s),
  so auto-reload on file change is impractical. Restart manually after edits.
"""

import sys
from pathlib import Path

# Ensure project root is importable before uvicorn imports api.main
sys.path.insert(0, str(Path(__file__).parents[1]))

import uvicorn
from api.main import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
