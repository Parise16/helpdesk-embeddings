from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/helpdesk.db"))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = ROOT_DIR / DATABASE_PATH
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

SPACY_MODEL_NAME = os.getenv("SPACY_MODEL_NAME", "pt_core_news_sm")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:4b-instruct")

AI_SEMANTIC_HIGH_SCORE = float(os.getenv("AI_SEMANTIC_HIGH_SCORE", "0.70"))
AI_SEMANTIC_MIN_SCORE = float(os.getenv("AI_SEMANTIC_MIN_SCORE", "0.50"))
AI_SEMANTIC_MIN_MARGIN = float(os.getenv("AI_SEMANTIC_MIN_MARGIN", "0.15"))
AI_NLP_WEIGHT = float(os.getenv("AI_NLP_WEIGHT", "0.10"))
AI_SEMANTIC_WEIGHT = 1.0 - AI_NLP_WEIGHT

OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
