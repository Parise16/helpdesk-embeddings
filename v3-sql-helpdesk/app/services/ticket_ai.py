from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import (
    AI_SEMANTIC_HIGH_SCORE,
    AI_SEMANTIC_MIN_MARGIN,
    AI_SEMANTIC_MIN_SCORE,
)
from app.services.nlp_service import NLPAnalysis, NLPService
from app.services.ollama_client import OllamaClient
from app.services.semantic_classifier import Candidate, SemanticClassifier


@dataclass
class AIResult:
    category_id: int
    category_name: str
    semantic_score: float
    nlp_score: float
    hybrid_score: float
    margin: float
    decision_source: str
    qwen_used: bool
    needs_review: bool
    problem_summary: str
    priority: str
    analysis: NLPAnalysis
    candidates: list[Candidate]


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


def infer_problem(text: str, analysis: NLPAnalysis) -> str:
    normalized = normalize(text)
    rules = [
        (("nao conecta", "sem conexao", "nao consigo conectar"), "Falha de conexão"),
        (("desconecta", "conexao cai", "cai toda hora"), "Conexão instável ou interrompida"),
        (("nao autentica", "nao consigo autenticar", "nao consigo fazer login"), "Falha de autenticação"),
        (("senha nao funciona", "conta bloqueada"), "Problema de acesso à conta"),
        (("fecha ao iniciar", "fecha sozinho", "fecha quando abre"), "Aplicação encerra durante a inicialização"),
        (("nao reconhece",), "Componente ou software não reconhecido"),
        (("nao funciona", "parou de funcionar", "nao esta funcionando"), "Componente não está funcionando"),
        (("nao liga", "nao esta ligando"), "Equipamento não liga"),
        (("erro",), "Erro reportado pelo usuário"),
    ]

    objects = analysis.technologies + analysis.devices + analysis.systems

    for phrases, summary in rules:
        if any(phrase in normalized for phrase in phrases):
            if objects:
                return f"{summary}: {', '.join(dict.fromkeys(objects))}"
            return summary

    if analysis.problem_terms:
        return "Problema envolvendo " + ", ".join(analysis.problem_terms[:5])

    return "Problema técnico informado pelo usuário"


def infer_priority(text: str) -> str:
    normalized = normalize(text)

    if any(
        signal in normalized
        for signal in (
            "empresa inteira",
            "escritorio inteiro",
            "todos os usuarios",
            "todo mundo",
            "producao parada",
            "sistema fora do ar",
            "urgente",
        )
    ):
        return "HIGH"

    if any(signal in normalized for signal in ("quando puder", "sem urgencia", "nao e urgente")):
        return "LOW"

    return "MEDIUM"


class TicketAIService:
    def __init__(self) -> None:
        self.nlp = NLPService()
        self.semantic = SemanticClassifier()
        self.ollama = OllamaClient()

    @staticmethod
    def _decision(candidates: list[Candidate]) -> tuple[bool, float, str]:
        if not candidates:
            return True, 0.0, "sem candidatos"

        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) > 1 else None
        margin = top1.hybrid_score - top2.hybrid_score if top2 else top1.hybrid_score

        if top1.semantic_score >= AI_SEMANTIC_HIGH_SCORE:
            return False, margin, "semantic_high_score"

        if top1.semantic_score >= AI_SEMANTIC_MIN_SCORE and margin >= AI_SEMANTIC_MIN_MARGIN:
            return False, margin, "semantic_clear_margin"

        return True, margin, "ambiguous"

    def analyze(self, session: Session, title: str, description: str) -> AIResult:
        text = f"{title}. {description}".strip()
        analysis = self.nlp.analyze(text)
        candidates = self.semantic.rank(session, text, analysis, top_k=3)

        use_llm, margin, reason = self._decision(candidates)
        selected = candidates[0]
        problem_summary = infer_problem(text, analysis)
        priority = infer_priority(text)
        decision_source = reason
        qwen_used = False
        needs_review = False

        if use_llm:
            try:
                result = self.ollama.classify(text, analysis, candidates)
                selected = next(candidate for candidate in candidates if candidate.category_name == result["category"])
                problem_summary = result["problem_summary"] or problem_summary
                priority = result["priority"]
                decision_source = "qwen_fallback"
                qwen_used = True
            except Exception:
                decision_source = "semantic_fallback_after_llm_error"
                needs_review = True

        return AIResult(
            category_id=selected.category_id,
            category_name=selected.category_name,
            semantic_score=selected.semantic_score,
            nlp_score=selected.nlp_score,
            hybrid_score=selected.hybrid_score,
            margin=margin,
            decision_source=decision_source,
            qwen_used=qwen_used,
            needs_review=needs_review,
            problem_summary=problem_summary,
            priority=priority,
            analysis=analysis,
            candidates=candidates,
        )
