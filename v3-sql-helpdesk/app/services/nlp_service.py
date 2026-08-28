from __future__ import annotations

import re
from dataclasses import dataclass

import spacy

from app.config import SPACY_MODEL_NAME


@dataclass
class NLPAnalysis:
    original_text: str
    normalized_text: str
    relevant_lemmas: list[str]
    locations: list[str]
    technologies: list[str]
    devices: list[str]
    systems: list[str]
    time_expressions: list[str]
    problem_terms: list[str]

    def as_dict(self) -> dict:
        return {
            "texto_original": self.original_text,
            "texto_normalizado": self.normalized_text,
            "lemmas_relevantes": self.relevant_lemmas,
            "locais": self.locations,
            "tecnologias": self.technologies,
            "dispositivos": self.devices,
            "sistemas": self.systems,
            "tempo": self.time_expressions,
            "termos_problema": self.problem_terms,
        }


class NLPService:
    def __init__(self) -> None:
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError as exc:
            raise RuntimeError(
                f"Modelo spaCy '{SPACY_MODEL_NAME}' não encontrado. "
                f"Execute: python -m spacy download {SPACY_MODEL_NAME}"
            ) from exc

        if "technical_entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("technical_entity_ruler")

        ruler = self.nlp.add_pipe(
            "entity_ruler",
            name="technical_entity_ruler",
            after="ner" if "ner" in self.nlp.pipe_names else None,
            config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"},
        )
        ruler.add_patterns(self._patterns())

    @staticmethod
    def _patterns() -> list[dict]:
        return [
            {"label": "LOCAL", "pattern": [{"LOWER": {"IN": ["laboratório", "laboratorio", "lab"]}}, {"LIKE_NUM": True}]},
            {"label": "LOCAL", "pattern": [{"LOWER": "sala"}, {"LIKE_NUM": True}]},
            {"label": "LOCAL", "pattern": [{"LOWER": "andar"}, {"LIKE_NUM": True}]},
            {"label": "LOCAL", "pattern": [{"LOWER": "bloco"}, {"IS_ALPHA": True}]},
            {"label": "LOCAL", "pattern": "escritório"},
            {"label": "LOCAL", "pattern": "escritorio"},
            {"label": "TECNOLOGIA", "pattern": "Wi-Fi"},
            {"label": "TECNOLOGIA", "pattern": "Wifi"},
            {"label": "TECNOLOGIA", "pattern": "VPN"},
            {"label": "TECNOLOGIA", "pattern": "Docker Desktop"},
            {"label": "TECNOLOGIA", "pattern": "Docker"},
            {"label": "TECNOLOGIA", "pattern": "GitHub"},
            {"label": "TECNOLOGIA", "pattern": "VS Code"},
            {"label": "TECNOLOGIA", "pattern": "Python"},
            {"label": "TECNOLOGIA", "pattern": "SQL"},
            {"label": "TECNOLOGIA", "pattern": "Oracle"},
            {"label": "TECNOLOGIA", "pattern": "PostgreSQL"},
            {"label": "TECNOLOGIA", "pattern": "MySQL"},
            {"label": "DISPOSITIVO", "pattern": "notebook"},
            {"label": "DISPOSITIVO", "pattern": "computador"},
            {"label": "DISPOSITIVO", "pattern": "PC"},
            {"label": "DISPOSITIVO", "pattern": "teclado"},
            {"label": "DISPOSITIVO", "pattern": "mouse"},
            {"label": "DISPOSITIVO", "pattern": "monitor"},
            {"label": "DISPOSITIVO", "pattern": "impressora"},
            {"label": "SISTEMA", "pattern": "Windows 11"},
            {"label": "SISTEMA", "pattern": "Windows 10"},
            {"label": "SISTEMA", "pattern": "Windows"},
            {"label": "SISTEMA", "pattern": "Linux"},
            {"label": "TEMPO", "pattern": "desde ontem"},
            {"label": "TEMPO", "pattern": "desde hoje"},
            {"label": "TEMPO", "pattern": "ontem"},
            {"label": "TEMPO", "pattern": "hoje"},
        ]

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))

    def analyze(self, text: str) -> NLPAnalysis:
        doc = self.nlp(text)
        lemmas: list[str] = []
        problem_terms: list[str] = []

        for token in doc:
            lemma = token.lemma_.strip().lower()
            if lemma and not token.is_space and not token.is_punct and (not token.is_stop or token.lower_ in {"não", "sem"}):
                lemmas.append(lemma)
            if lemma and not token.is_space and not token.is_punct and token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"} and not token.is_stop:
                problem_terms.append(lemma)

        entities = [
            (ent.text, ent.label_)
            for ent in doc.ents
            if ent.label_ in {"LOCAL", "TECNOLOGIA", "DISPOSITIVO", "SISTEMA", "TEMPO"}
        ]

        def values(label: str) -> list[str]:
            return self._unique([text for text, current_label in entities if current_label == label])

        normalized = re.sub(r"\s+", " ", " ".join(self._unique(lemmas))).strip()

        return NLPAnalysis(
            original_text=text,
            normalized_text=normalized,
            relevant_lemmas=lemmas,
            locations=values("LOCAL"),
            technologies=values("TECNOLOGIA"),
            devices=values("DISPOSITIVO"),
            systems=values("SISTEMA"),
            time_expressions=values("TEMPO"),
            problem_terms=self._unique(problem_terms)[:12],
        )
