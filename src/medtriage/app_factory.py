"""Factory da aplicacao FastAPI."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from medtriage.model_service import ModelServiceFactory
from medtriage.schemas import TriageRequest, TriageResponse

REQUEST_COUNT = Counter(
    "medtriage_http_requests_total",
    "Total de requisicoes HTTP recebidas pela API",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "medtriage_http_request_latency_seconds",
    "Latencia das requisicoes HTTP por endpoint",
    ["method", "endpoint"],
)

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
        "Nenhum modelo encontrado. Defina MODEL_PATH ou gere "
        "models/text_classifier_best.joblib."
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

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.middleware("http")
    async def track_metrics(request: Request, call_next):
        """Registra contagem de chamadas e latencia por endpoint."""
        start = perf_counter()
        response = await call_next(request)
        duration = perf_counter() - start
        endpoint = request.url.path
        REQUEST_LATENCY.labels(request.method, endpoint).observe(duration)
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        return response

    @app.post("/predict", response_model=TriageResponse)
    def predict(payload: TriageRequest) -> TriageResponse:
        cleaned_text = payload.text.strip()
        if not cleaned_text:
            raise HTTPException(status_code=422, detail="Texto do laudo vazio.")

        prediction = model_service.predict(cleaned_text)
        return TriageResponse(**prediction)

    return app
