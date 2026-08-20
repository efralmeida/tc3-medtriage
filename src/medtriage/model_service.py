"""Camada de servico para inferencia de triagem medica."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib


def _softmax_confidence(scores: list[float]) -> float:
    """Converte scores em uma confianca no intervalo [0, 1]."""
    max_score = max(scores)
    shifted = [exp(score - max_score) for score in scores]
    total = sum(shifted)
    return max(shifted) / total if total else 0.0


@dataclass
class ModelService:
    """Encapsula o pipeline de classificacao e sua inferencia."""

    model: Any

    def predict(self, text: str) -> dict[str, float | str]:
        """Classifica um laudo e retorna classe, confianca e latencia."""
        start = perf_counter()
        urgency = str(self.model.predict([text])[0])
        confidence = self._resolve_confidence(text)
        latency_ms = (perf_counter() - start) * 1000
        return {
            "urgency": urgency,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }

    def _resolve_confidence(self, text: str) -> float:
        """Recupera confianca usando predict_proba ou decision_function."""
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([text])[0]
            return float(max(probabilities))

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function([text])[0]
            scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            return float(_softmax_confidence(scores_list))

        return 1.0


class ModelServiceFactory:
    """Factory para criar instancias de ModelService a partir de artefatos."""

    @staticmethod
    def create(model_path: Path) -> ModelService:
        """Carrega o artefato serializado e retorna o servico pronto."""
        model = joblib.load(model_path)
        return ModelService(model=model)
