"""Factory da aplicacao FastAPI."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from medtriage.model_service import ModelServiceFactory
from medtriage.schemas import TriageRequest, TriageResponse

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CANDIDATES = [
    PROJECT_DIR / "models" / "text_classifier_best.joblib",
    PROJECT_DIR / "models" / "text_classifier.joblib",
]


def _resolve_model_path() -> Path:
    """Resolve caminho do modelo via env ou fallback local."""
    model_path_env = os.getenv("MODEL_PATH")
    if model_path_env:
        path = Path(model_path_env)
        if path.exists():
            return path

    for path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Nenhum modelo encontrado. Defina MODEL_PATH ou gere models/text_classifier_best.joblib."
    )


def create_app() -> FastAPI:
    """Cria a app com endpoints de health e inferencia."""
    model_service = ModelServiceFactory.create(model_path=_resolve_model_path())

    app = FastAPI(
        title="MedTriage API",
        version="0.1.0",
        description="Classificacao de urgencia de laudos medicos",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=TriageResponse)
    def predict(payload: TriageRequest) -> TriageResponse:
        cleaned_text = payload.text.strip()
        if not cleaned_text:
            raise HTTPException(status_code=422, detail="Texto do laudo vazio.")

        prediction = model_service.predict(cleaned_text)
        return TriageResponse(**prediction)

    return app
