from dataclasses import dataclass
from pathlib import Path

import numpy as np

import medtriage.model_service as model_service
from medtriage.model_service import ModelService, ModelServiceFactory


@dataclass
class FakeModel:
    def predict(self, texts: list[str]) -> list[str]:
        return ["urgente"]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.7]]


@dataclass
class FakeOnnxInput:
    name: str


class FakeOnnxSession:
    def __init__(self, model_path: str, providers: list[str]) -> None:
        self.model_path = model_path
        self.providers = providers
        self.received_inputs: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[FakeOnnxInput]:
        return [FakeOnnxInput(name="text_input")]

    def run(
        self, output_names: None, inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        self.received_inputs = inputs
        return [
            np.asarray([b"urgente"], dtype=object),
            np.asarray([[0.05, 0.15, 0.8]]),
        ]


def test_predict_returns_class_confidence_and_latency() -> None:
    result = ModelService(model=FakeModel()).predict("dor toracica")

    assert result["urgency"] == "urgente"
    assert result["confidence"] == 0.7
    assert result["latency_ms"] >= 0


def test_predict_returns_expected_fields() -> None:
    result = ModelService(model=FakeModel()).predict("sintoma")

    assert set(result) == {"urgency", "confidence", "latency_ms"}


def test_factory_uses_onnx_runtime_for_onnx_model(monkeypatch) -> None:
    sessions: list[FakeOnnxSession] = []

    def create_session(model_path: str, providers: list[str]) -> FakeOnnxSession:
        session = FakeOnnxSession(model_path, providers)
        sessions.append(session)
        return session

    monkeypatch.setattr(model_service.ort, "InferenceSession", create_session)

    service = ModelServiceFactory.create(Path("models/text_classifier_best.onnx"))
    result = service.predict("dor torácica")

    assert result["urgency"] == "urgente"
    assert result["confidence"] == 0.8
    assert sessions[0].model_path == "models/text_classifier_best.onnx"
    assert sessions[0].providers == ["CPUExecutionProvider"]
    assert sessions[0].received_inputs["text_input"].tolist() == [["dor toracica"]]