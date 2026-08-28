from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from app.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TIMEOUT_SECONDS
from app.services.nlp_service import NLPAnalysis
from app.services.semantic_classifier import Candidate


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("O Qwen não retornou JSON válido.")
        return json.loads(match.group(0))


class OllamaClient:
    def classify(self, ticket_text: str, analysis: NLPAnalysis, candidates: list[Candidate]) -> dict:
        allowed = [candidate.category_name for candidate in candidates]

        prompt = f"""
Você atua como fallback de um classificador de chamados de TI.
Escolha exatamente uma categoria permitida e não invente dados.

Categorias permitidas:
{json.dumps(allowed, ensure_ascii=False)}

Chamado:
{ticket_text}

NLP estruturado:
{json.dumps(analysis.as_dict(), ensure_ascii=False)}

Retorne somente JSON válido:
{{
  "category": "categoria permitida",
  "problem_summary": "descrição curta e objetiva do problema",
  "priority": "LOW, MEDIUM ou HIGH"
}}

Use o chamado original e o NLP como contexto.
Não invente local, responsável, departamento ou tamanho de fila.
""".strip()

        body = {
            "model": OLLAMA_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "format": "json",
            "options": {"temperature": 0},
        }

        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama indisponível em {OLLAMA_BASE_URL}.") from exc

        result = _extract_json(payload.get("response", ""))

        if result.get("category") not in allowed:
            raise ValueError("O Qwen retornou uma categoria fora das opções permitidas.")

        priority = str(result.get("priority", "MEDIUM")).upper()
        if priority not in {"LOW", "MEDIUM", "HIGH"}:
            priority = "MEDIUM"

        return {
            "category": result["category"],
            "problem_summary": str(result.get("problem_summary", "")).strip(),
            "priority": priority,
        }
