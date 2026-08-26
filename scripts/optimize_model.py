"""Exporta o modelo para ONNX e compara latencias de inferencia locais."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import statistics
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from medtriage.model_service import ModelService, ModelServiceFactory

DEFAULT_JOBLIB = Path("models/text_classifier_best.joblib")
DEFAULT_ONNX = Path("models/text_classifier_best.onnx")
DEFAULT_REPORT = Path("models/model_latency_comparison.csv")


def _export_model(joblib_path: Path, onnx_path: Path) -> None:
    """Converte o pipeline sklearn para um modelo ONNX de entrada textual."""
    pipeline = deepcopy(joblib.load(joblib_path))
    vectorizer = pipeline.named_steps["tfidf"]
    if vectorizer.strip_accents is not None:
        vectorizer.strip_accents = None
    initial_types = [("text", StringTensorType([None, 1]))]
    classifier = pipeline.named_steps["classifier"]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        options={id(classifier): {"zipmap": False}},
        target_opset=17,
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())


def _measure(
    service: ModelService, texts: list[str], iterations: int
) -> dict[str, float]:
    """Mede latencia por predicao depois de aquecer o servico."""
    for text in texts[:10]:
        service.predict(text)

    latencies = []
    for index in range(iterations):
        start = time.perf_counter()
        service.predict(texts[index % len(texts)])
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "avg_latency_ms": statistics.mean(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": (
            statistics.quantiles(latencies, n=100)[94]
            if len(latencies) >= 20
            else max(latencies)
        ),
    }


def _accuracy(service: ModelService, texts: list[str], targets: list[str]) -> float:
    """Calcula acuracia do servico no conjunto de teste."""
    predictions = [str(service.predict(text)["urgency"]) for text in texts]
    return sum(
        prediction == target for prediction, target in zip(predictions, targets)
    ) / len(targets)


def _write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    """Salva os resultados para uso na documentacao e auditoria."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Exporta, valida e compara os dois backends de inferencia."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--joblib", type=Path, default=DEFAULT_JOBLIB)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations deve ser maior que zero")

    _export_model(args.joblib, args.onnx)
    test_data = pd.read_csv(args.test_data)
    texts = test_data["text"].astype(str).tolist()
    targets = (
        test_data["urgency"].astype(str).tolist()
        if "urgency" in test_data.columns
        else []
    )
    services = {
        "sklearn_joblib": ModelServiceFactory.create(args.joblib),
        "onnxruntime": ModelServiceFactory.create(args.onnx),
    }
    rows = []
    for name, service in services.items():
        metrics = _measure(service, texts, args.iterations)
        rows.append(
            {
                "backend": name,
                "accuracy": _accuracy(service, texts, targets) if targets else "n/a",
                **metrics,
            }
        )

    _write_report(args.report, rows)
    for row in rows:
        formatted = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items()
        )
        print(formatted)
    print(f"report={args.report}")


if __name__ == "__main__":
    main()