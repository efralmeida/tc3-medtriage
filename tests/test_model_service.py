from dataclasses import dataclass

from medtriage.model_service import ModelService


@dataclass
class FakeModel:
    def predict(self, texts: list[str]) -> list[str]:
        return ["urgente"]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.7]]


def test_predict_returns_class_confidence_and_latency() -> None:
    result = ModelService(model=FakeModel()).predict("dor toracica")

    assert result["urgency"] == "urgente"
    assert result["confidence"] == 0.7
    assert result["latency_ms"] >= 0


def test_predict_returns_expected_fields() -> None:
    result = ModelService(model=FakeModel()).predict("sintoma")

    assert set(result) == {"urgency", "confidence", "latency_ms"}