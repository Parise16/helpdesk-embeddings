from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.config import AI_NLP_WEIGHT, AI_SEMANTIC_WEIGHT, EMBEDDING_MODEL_NAME
from app.repositories import get_classifier_data
from app.services.nlp_service import NLPAnalysis


@dataclass
class Candidate:
    category_id: int
    category_name: str
    semantic_score: float
    nlp_score: float
    hybrid_score: float
    closest_example: str

    def as_dict(self) -> dict:
        return {
            "category_id": self.category_id,
            "category_name": self.category_name,
            "semantic_score": self.semantic_score,
            "nlp_score": self.nlp_score,
            "hybrid_score": self.hybrid_score,
            "closest_example": self.closest_example,
        }


def normalize_match(text: str) -> str:
    text = text.lower().replace("‑", "-").replace("–", "-")
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


class SemanticClassifier:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._fingerprint: tuple | None = None
        self._rows: list[dict] = []
        self._vectors: np.ndarray | None = None
        self._keywords: dict[str, list[tuple[str, float]]] = {}

    def _refresh(self, session: Session) -> None:
        rows, keywords = get_classifier_data(session)
        fingerprint = tuple((row["id"], row["category_id"], row["text"]) for row in rows)

        if fingerprint == self._fingerprint:
            self._keywords = keywords
            return

        if not rows:
            raise RuntimeError("Nenhum exemplo ativo de classificação foi encontrado.")

        self._rows = rows
        self._keywords = keywords
        self._vectors = self.model.encode(
            [row["text"] for row in rows],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._fingerprint = fingerprint

    def _nlp_score(self, category_name: str, analysis: NLPAnalysis) -> float:
        evidence = normalize_match(
            " ".join(
                [
                    analysis.original_text,
                    analysis.normalized_text,
                    *analysis.locations,
                    *analysis.technologies,
                    *analysis.devices,
                    *analysis.systems,
                ]
            )
        )

        points = 0.0
        for keyword, weight in self._keywords.get(category_name, []):
            if normalize_match(keyword) in evidence:
                points += float(weight)

        return min(points / 3.0, 1.0)

    def rank(self, session: Session, text: str, analysis: NLPAnalysis, top_k: int = 3) -> list[Candidate]:
        self._refresh(session)

        query = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        similarities = self._vectors @ query
        best: dict[int, tuple[float, dict]] = {}

        for row, score in zip(self._rows, similarities):
            category_id = int(row["category_id"])
            score = float(score)
            current = best.get(category_id)
            if current is None or score > current[0]:
                best[category_id] = (score, row)

        candidates: list[Candidate] = []

        for semantic_score, row in best.values():
            nlp_score = self._nlp_score(row["category_name"], analysis)
            hybrid_score = semantic_score * AI_SEMANTIC_WEIGHT + nlp_score * AI_NLP_WEIGHT

            candidates.append(
                Candidate(
                    category_id=int(row["category_id"]),
                    category_name=row["category_name"],
                    semantic_score=semantic_score,
                    nlp_score=nlp_score,
                    hybrid_score=hybrid_score,
                    closest_example=row["text"],
                )
            )

        candidates.sort(key=lambda item: item.hybrid_score, reverse=True)
        return candidates[:top_k]
