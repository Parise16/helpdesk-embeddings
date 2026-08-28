from __future__ import annotations

import json
import urllib.request

import spacy
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, SPACY_MODEL_NAME


def check_spacy() -> None:
    nlp = spacy.load(SPACY_MODEL_NAME)
    doc = nlp("Notebook sem Wi-Fi no laboratório 704.")
    print(f"spaCy OK: {SPACY_MODEL_NAME} | tokens={len(doc)}")


def check_embedding() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector = model.encode("Teste de embedding.")
    print(f"Embedding OK: {EMBEDDING_MODEL_NAME} | dimensão={len(vector)}")


def check_ollama() -> None:
    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names = {item.get("name") for item in payload.get("models", [])}
    if OLLAMA_LLM_MODEL not in names:
        raise RuntimeError(f"Modelo {OLLAMA_LLM_MODEL} não encontrado no Ollama.")
    print(f"Ollama OK: {OLLAMA_LLM_MODEL}")


if __name__ == "__main__":
    check_spacy()
    check_embedding()
    check_ollama()
    print("Todos os modelos estão disponíveis.")
