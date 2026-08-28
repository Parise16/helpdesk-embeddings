from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from sentence_transformers import SentenceTransformer
import argparse
import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
import numpy as np
import spacy

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")

SEMANTIC_HIGH_SCORE = 0.70
SEMANTIC_MIN_SCORE = 0.50
MIN_MARGIN = 0.15
SEMANTIC_WEIGHT = 0.90
NLP_WEIGHT = 0.10

CATEGORY_EXAMPLES = {
    "Rede e Internet": [
        "A internet do laboratório caiu.",
        "Estou conectado no Wi-Fi, mas nenhum site abre.",
        "Meu notebook não consegue conectar na rede sem fio.",
        "A rede está conectada, porém estou sem acesso à internet.",
    ],
    "VPN": [
        "A VPN desconecta depois de alguns minutos.",
        "Não consigo conectar na VPN corporativa.",
        "O acesso remoto pela VPN apresenta erro.",
    ],
    "Hardware": [
        "O teclado do computador não está funcionando.",
        "O monitor não liga.",
        "Meu notebook não reconhece o mouse.",
        "O computador não está ligando.",
    ],
    "Acesso e Autenticação": [
        "Não consigo autenticar no GitHub com minha conta.",
        "Minha senha não funciona no login.",
        "Minha conta está bloqueada.",
        "Não consigo fazer login no sistema.",
    ],
    "Software e Aplicações": [
        "O Docker Desktop fecha ao iniciar no Windows.",
        "O VS Code não reconhece o Python.",
        "O aplicativo fecha sozinho quando abre.",
        "O programa apresenta erro durante a inicialização.",
    ],
    "Banco de Dados": [
        "Não consigo conectar no banco de dados.",
        "A consulta SQL está retornando erro.",
        "O Oracle não aceita a conexão.",
        "O PostgreSQL está indisponível.",
    ],
    "Segurança": [
        "Recebi um e-mail de phishing.",
        "Minha conta apresentou um login que não reconheço.",
        "O antivírus detectou um arquivo suspeito.",
        "Acredito que minha conta foi comprometida.",
    ],
}

CATEGORY_HINTS = {
    "Rede e Internet": {
        "wi-fi": 1.5,
        "wifi": 1.5,
        "internet": 1.5,
        "rede": 1.2,
        "dns": 1.5,
        "roteador": 1.4,
        "conexao": 0.8,
        "site": 0.6,
    },
    "VPN": {
        "vpn": 2.0,
        "tunel": 1.5,
        "acesso remoto": 1.3,
    },
    "Hardware": {
        "teclado": 1.4,
        "mouse": 1.3,
        "monitor": 1.4,
        "impressora": 1.4,
        "hardware": 1.5,
        "computador": 0.6,
        "notebook": 0.6,
        "pc": 0.5,
    },
    "Acesso e Autenticação": {
        "senha": 1.5,
        "login": 1.5,
        "autenticar": 1.5,
        "autenticacao": 1.5,
        "credencial": 1.4,
        "conta": 0.8,
        "bloqueada": 1.0,
        "bloqueado": 1.0,
    },
    "Software e Aplicações": {
        "docker": 1.6,
        "vs code": 1.6,
        "vscode": 1.6,
        "python": 1.0,
        "windows": 0.8,
        "aplicativo": 1.1,
        "programa": 1.0,
        "software": 1.3,
    },
    "Banco de Dados": {
        "sql": 1.5,
        "oracle": 1.6,
        "postgresql": 1.6,
        "postgres": 1.6,
        "mysql": 1.6,
        "banco de dados": 1.5,
        "database": 1.4,
        "consulta": 0.7,
    },
    "Segurança": {
        "phishing": 1.8,
        "malware": 1.8,
        "virus": 1.6,
        "invasao": 1.6,
        "suspeito": 1.1,
        "comprometida": 1.5,
        "comprometido": 1.5,
        "seguranca": 1.2,
    },
}

@dataclass
class Entity:
    text: str
    label: str


@dataclass
class NLPAnalysis:
    original_text: str
    relevant_lemmas: list[str]
    normalized_text: str
    entities: list[Entity]
    locations: list[str]
    technologies: list[str]
    devices: list[str]
    systems: list[str]
    time_expressions: list[str]
    problem_terms: list[str]

    def as_dict(self) -> dict:
        return {
            "texto_original": self.original_text,
            "lemmas_relevantes": self.relevant_lemmas,
            "texto_normalizado": self.normalized_text,
            "entidades": [
                {"texto": entity.text, "tipo": entity.label}
                for entity in self.entities
            ],
            "locais": self.locations,
            "tecnologias": self.technologies,
            "dispositivos": self.devices,
            "sistemas": self.systems,
            "tempo": self.time_expressions,
            "termos_problema": self.problem_terms,
        }


@dataclass
class Candidate:
    category: str
    semantic_score: float
    nlp_score: float
    hybrid_score: float
    closest_example: str

def normalize_for_match(text: str) -> str:
    text = text.lower().replace("‑", "-").replace("–", "-")
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TechnicalNLP:
    def __init__(self) -> None:
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError as exc:
            raise RuntimeError(
                "Modelo pt_core_news_sm não encontrado.\n"
                "Instale com:\n"
                "python -m spacy download pt_core_news_sm"
            ) from exc

        if "technical_entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("technical_entity_ruler")

        ruler = self.nlp.add_pipe(
            "entity_ruler",
            name="technical_entity_ruler",
            after="ner" if "ner" in self.nlp.pipe_names else None,
            config={
                "overwrite_ents": True,
                "phrase_matcher_attr": "LOWER",
            },
        )

        ruler.add_patterns(self._patterns())

    @staticmethod
    def _patterns() -> list[dict]:
        return [
            # Locais
            {
                "label": "LOCAL",
                "pattern": [
                    {"LOWER": {"IN": ["laboratório", "laboratorio", "lab"]}},
                    {"LIKE_NUM": True},
                ],
            },
            {
                "label": "LOCAL",
                "pattern": [
                    {"LOWER": "sala"},
                    {"LIKE_NUM": True},
                ],
            },
            {
                "label": "LOCAL",
                "pattern": [
                    {"LOWER": "andar"},
                    {"LIKE_NUM": True},
                ],
            },
            {
                "label": "LOCAL",
                "pattern": [
                    {"LOWER": "bloco"},
                    {"IS_ALPHA": True},
                ],
            },
            {"label": "LOCAL", "pattern": "escritório"},
            {"label": "LOCAL", "pattern": "escritorio"},

            # Tecnologias / rede
            {"label": "TECNOLOGIA", "pattern": "Wi-Fi"},
            {"label": "TECNOLOGIA", "pattern": "Wifi"},
            {"label": "TECNOLOGIA", "pattern": "VPN"},
            {"label": "TECNOLOGIA", "pattern": "Docker"},
            {"label": "TECNOLOGIA", "pattern": "Docker Desktop"},
            {"label": "TECNOLOGIA", "pattern": "GitHub"},
            {"label": "TECNOLOGIA", "pattern": "VS Code"},
            {"label": "TECNOLOGIA", "pattern": "Python"},
            {"label": "TECNOLOGIA", "pattern": "SQL"},
            {"label": "TECNOLOGIA", "pattern": "Oracle"},
            {"label": "TECNOLOGIA", "pattern": "PostgreSQL"},
            {"label": "TECNOLOGIA", "pattern": "MySQL"},

            # Dispositivos
            {"label": "DISPOSITIVO", "pattern": "notebook"},
            {"label": "DISPOSITIVO", "pattern": "computador"},
            {"label": "DISPOSITIVO", "pattern": "PC"},
            {"label": "DISPOSITIVO", "pattern": "teclado"},
            {"label": "DISPOSITIVO", "pattern": "mouse"},
            {"label": "DISPOSITIVO", "pattern": "monitor"},
            {"label": "DISPOSITIVO", "pattern": "impressora"},

            # Sistemas
            {"label": "SISTEMA", "pattern": "Windows 11"},
            {"label": "SISTEMA", "pattern": "Windows 10"},
            {"label": "SISTEMA", "pattern": "Windows"},
            {"label": "SISTEMA", "pattern": "Linux"},

            # Tempo
            {"label": "TEMPO", "pattern": "desde ontem"},
            {"label": "TEMPO", "pattern": "desde hoje"},
            {"label": "TEMPO", "pattern": "ontem"},
            {"label": "TEMPO", "pattern": "hoje"},
        ]

    def analyze(self, text: str) -> NLPAnalysis:
        doc = self.nlp(text)

        relevant_lemmas: list[str] = []
        problem_terms: list[str] = []

        for token in doc:
            lemma = token.lemma_.strip().lower()

            if (
                lemma
                and not token.is_space
                and not token.is_punct
                and (
                    not token.is_stop
                    or token.lower_ in {"não", "sem"}
                )
            ):
                relevant_lemmas.append(lemma)

            if (
                lemma
                and not token.is_space
                and not token.is_punct
                and token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
                and not token.is_stop
            ):
                problem_terms.append(lemma)

        entities = [
            Entity(text=ent.text, label=ent.label_)
            for ent in doc.ents
            if ent.label_ in {
                "LOCAL",
                "TECNOLOGIA",
                "DISPOSITIVO",
                "SISTEMA",
                "TEMPO",
            }
        ]

        def values(label: str) -> list[str]:
            return list(dict.fromkeys(
                entity.text
                for entity in entities
                if entity.label == label
            ))

        normalized = " ".join(dict.fromkeys(relevant_lemmas))

        return NLPAnalysis(
            original_text=text,
            relevant_lemmas=relevant_lemmas,
            normalized_text=normalized,
            entities=entities,
            locations=values("LOCAL"),
            technologies=values("TECNOLOGIA"),
            devices=values("DISPOSITIVO"),
            systems=values("SISTEMA"),
            time_expressions=values("TEMPO"),
            problem_terms=list(dict.fromkeys(problem_terms))[:10],
        )

class HybridClassifier:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self._example_rows: list[tuple[str, str]] = []
        texts: list[str] = []

        for category, examples in CATEGORY_EXAMPLES.items():
            for example in examples:
                self._example_rows.append((category, example))
                texts.append(example)

        self._example_vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _nlp_signal(self, category: str, analysis: NLPAnalysis) -> float:
        normalized_original = normalize_for_match(analysis.original_text)
        normalized_lemmas = normalize_for_match(analysis.normalized_text)

        evidence_text = f"{normalized_original} {normalized_lemmas}"
        hints = CATEGORY_HINTS[category]

        points = 0.0
        for phrase, weight in hints.items():
            if normalize_for_match(phrase) in evidence_text:
                points += weight

        return min(points / 3.0, 1.0)

    def rank(
        self,
        text: str,
        analysis: NLPAnalysis,
        top_k: int = 3,
    ) -> list[Candidate]:
        query = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        similarities = self._example_vectors @ query

        best_by_category: dict[str, tuple[float, str]] = {}

        for (category, example), score in zip(
            self._example_rows,
            similarities,
        ):
            score = float(score)
            current = best_by_category.get(category)

            if current is None or score > current[0]:
                best_by_category[category] = (score, example)

        candidates = []

        for category, (semantic_score, closest_example) in best_by_category.items():
            nlp_score = self._nlp_signal(category, analysis)

            hybrid_score = (
                semantic_score * SEMANTIC_WEIGHT
                + nlp_score * NLP_WEIGHT
            )

            candidates.append(
                Candidate(
                    category=category,
                    semantic_score=semantic_score,
                    nlp_score=nlp_score,
                    hybrid_score=hybrid_score,
                    closest_example=closest_example,
                )
            )

        candidates.sort(
            key=lambda candidate: candidate.hybrid_score,
            reverse=True,
        )

        return candidates[:top_k]

def extract_problem_phrase(text: str, analysis: NLPAnalysis) -> str:
    normalized = normalize_for_match(text)

    rules = [
        (
            ("nao conecta", "sem conexao", "nao consigo conectar"),
            "Falha de conexão",
        ),
        (
            ("desconecta", "cai toda hora", "conexao cai"),
            "Conexão instável ou interrompida",
        ),
        (
            ("nao autentica", "nao consigo autenticar", "nao consigo fazer login"),
            "Falha de autenticação",
        ),
        (
            ("senha nao funciona", "conta bloqueada", "conta bloqueado"),
            "Problema de acesso à conta",
        ),
        (
            ("fecha ao iniciar", "fecha sozinho", "fecha quando abre"),
            "Aplicação encerra durante a inicialização",
        ),
        (
            ("nao reconhece",),
            "Componente ou software não reconhecido",
        ),
        (
            ("nao funciona", "parou de funcionar", "nao esta funcionando"),
            "Componente não está funcionando",
        ),
        (
            ("nao liga", "nao esta ligando"),
            "Equipamento não liga",
        ),
        (
            ("erro",),
            "Erro reportado pelo usuário",
        ),
    ]

    for phrases, result in rules:
        if any(phrase in normalized for phrase in phrases):
            objects = (
                analysis.technologies
                + analysis.devices
                + analysis.systems
            )
            if objects:
                return f"{result}: {', '.join(dict.fromkeys(objects))}"
            return result

    if analysis.problem_terms:
        return "Termos principais: " + ", ".join(analysis.problem_terms[:6])

    return "Problema técnico não estruturado"


def infer_priority(text: str) -> str:
    normalized = normalize_for_match(text)

    high_signals = (
        "empresa inteira",
        "escritorio inteiro",
        "todos os usuarios",
        "todo mundo",
        "producao parada",
        "sistema fora do ar",
        "urgente",
    )

    low_signals = (
        "quando puder",
        "sem urgencia",
        "nao e urgente",
    )

    if any(signal in normalized for signal in high_signals):
        return "HIGH"

    if any(signal in normalized for signal in low_signals):
        return "LOW"

    return "MEDIUM"

def _extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("O modelo não retornou JSON válido.")

    return json.loads(match.group(0))


def call_qwen(
    text: str,
    analysis: NLPAnalysis,
    candidates: list[Candidate],
) -> dict:
    candidate_names = [candidate.category for candidate in candidates]

    prompt = f"""
Você é o fallback de um classificador de chamados técnicos.

Escolha EXATAMENTE UMA categoria da lista permitida:
{json.dumps(candidate_names, ensure_ascii=False)}

Chamado:
{text}

NLP estruturado:
{json.dumps(analysis.as_dict(), ensure_ascii=False)}

Retorne SOMENTE JSON válido no formato:
{{
  "category": "uma categoria permitida",
  "problem": "descrição curta e objetiva do problema",
  "priority": "LOW, MEDIUM ou HIGH"
}}

Regras:
- Não crie categorias novas.
- Não invente um local.
- Use o contexto completo do chamado.
- Seja conciso.
""".strip()

    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Não foi possível acessar o Ollama em {OLLAMA_URL}."
        ) from exc

    result = _extract_json(payload.get("response", ""))

    if result.get("category") not in candidate_names:
        raise ValueError(
            "O Qwen retornou uma categoria fora das opções permitidas."
        )

    priority = str(result.get("priority", "MEDIUM")).upper()
    if priority not in {"LOW", "MEDIUM", "HIGH"}:
        priority = "MEDIUM"

    return {
        "category": result["category"],
        "problem": str(result.get("problem", "")).strip(),
        "priority": priority,
    }

def should_use_llm(candidates: list[Candidate]) -> tuple[bool, str]:
    if not candidates:
        return True, "sem candidatos"

    top1 = candidates[0]
    top2 = candidates[1] if len(candidates) > 1 else None

    margin = (
        top1.hybrid_score - top2.hybrid_score
        if top2
        else top1.hybrid_score
    )

    if top1.semantic_score >= SEMANTIC_HIGH_SCORE:
        return False, "score semântico alto"

    if (
        top1.semantic_score >= SEMANTIC_MIN_SCORE
        and margin >= MIN_MARGIN
    ):
        return False, "margem híbrida clara"

    return True, "classificação ambígua"


def find_candidate(
    candidates: Iterable[Candidate],
    category: str,
) -> Candidate:
    for candidate in candidates:
        if candidate.category == category:
            return candidate

    raise ValueError("Categoria selecionada não encontrada no ranking.")


def run_ticket(
    nlp_engine: TechnicalNLP,
    classifier: HybridClassifier,
    text: str,
    debug_nlp: bool = False,
) -> None:
    analysis = nlp_engine.analyze(text)
    candidates = classifier.rank(text, analysis, top_k=3)

    use_llm, decision_reason = should_use_llm(candidates)

    selected = candidates[0]
    problem = extract_problem_phrase(text, analysis)
    priority = infer_priority(text)
    decision_source = "hybrid_nlp_embeddings"
    qwen_used = False

    if use_llm:
        try:
            llm_result = call_qwen(text, analysis, candidates)
            selected = find_candidate(
                candidates,
                llm_result["category"],
            )
            problem = llm_result["problem"] or problem
            priority = llm_result["priority"]
            decision_source = "qwen_fallback"
            qwen_used = True

        except Exception as exc:
            decision_source = "manual_review"
            decision_reason = (
                f"{decision_reason}; fallback indisponível: {exc}"
            )

    top2 = candidates[1] if len(candidates) > 1 else None
    margin = (
        candidates[0].hybrid_score - top2.hybrid_score
        if top2
        else candidates[0].hybrid_score
    )

    print("\n=== ANÁLISE NLP ===")
    print(
        "Texto normalizado:",
        analysis.normalized_text or "—",
    )
    print(
        "Local:",
        ", ".join(analysis.locations) if analysis.locations else "não identificado",
    )
    print(
        "Tecnologia:",
        ", ".join(analysis.technologies) if analysis.technologies else "—",
    )
    print(
        "Dispositivo:",
        ", ".join(analysis.devices) if analysis.devices else "—",
    )
    print(
        "Sistema:",
        ", ".join(analysis.systems) if analysis.systems else "—",
    )

    if debug_nlp:
        print("\nNLP completo:")
        print(
            json.dumps(
                analysis.as_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )

    print("\n=== TOP 3 ===")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"{index}. {candidate.category}\n"
            f"   embedding: {candidate.semantic_score * 100:.1f}%\n"
            f"   sinal NLP: {candidate.nlp_score * 100:.1f}%\n"
            f"   score híbrido: {candidate.hybrid_score * 100:.1f}%\n"
            f"   exemplo próximo: {candidate.closest_example}"
        )

    print("\n=== RESULTADO ===")
    print("Categoria:", selected.category)
    print(
        "Score semântico da categoria:",
        f"{selected.semantic_score * 100:.1f}%",
    )
    print(
        "Score híbrido da categoria:",
        f"{selected.hybrid_score * 100:.1f}%",
    )
    print("Margem Top1-Top2:", f"{margin * 100:.1f}%")
    print("Problema:", problem)
    print("Prioridade:", priority)
    print(
        "Local:",
        ", ".join(analysis.locations) if analysis.locations else "não identificado",
    )
    print("Origem da decisão:", decision_source)
    print("Motivo:", decision_reason)
    print("Qwen utilizado:", "SIM" if qwen_used else "NÃO")

    print(
        "\nObservação: os scores são medidas de similaridade/evidência "
        "e não probabilidades calibradas."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classificador híbrido de chamados: "
            "spaCy NLP + MiniLM embeddings + Qwen fallback."
        )
    )
    parser.add_argument(
        "--debug-nlp",
        action="store_true",
        help="Mostra tokens/entidades estruturados em JSON.",
    )
    args = parser.parse_args()

    print("=== HelpDesk Embeddings + NLP V2 ===")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Fallback: {OLLAMA_MODEL}\n")

    print("Carregando NLP...")
    nlp_engine = TechnicalNLP()

    print("Carregando embeddings...")
    classifier = HybridClassifier()

    print("\nModelos prontos.\n")

    while True:
        text = input(
            "Digite o chamado "
            "(ou 'sair' para encerrar): "
        ).strip()

        if text.lower() in {"sair", "exit", "quit"}:
            break

        if not text:
            continue

        run_ticket(
            nlp_engine,
            classifier,
            text,
            debug_nlp=args.debug_nlp,
        )
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
