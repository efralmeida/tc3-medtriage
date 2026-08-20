"""Schemas de entrada e saida da API."""

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """Entrada para classificacao de urgencia de laudo."""

    text: str = Field(..., min_length=5, description="Texto do laudo medico")


class TriageResponse(BaseModel):
    """Saida da predicao de urgencia."""

    urgency: str
    confidence: float
    latency_ms: float
