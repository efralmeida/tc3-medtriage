"""Benchmark simples de latencia da API de triagem."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def _post_predict(url: str, text: str) -> tuple[int, float]:
    """Envia um POST e retorna status code e latencia em ms."""
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request) as response:  # nosec: B310
        _ = response.read()
        status_code = response.getcode()
    latency_ms = (time.perf_counter() - start) * 1000
    return status_code, latency_ms


def main() -> None:
    """Executa benchmark de n requisicoes e imprime p50/p95/media."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument(
        "--text",
        default="Paciente com dor toracica intensa e sudorese fria nas ultimas 2 horas.",
    )
    args = parser.parse_args()

    latencies = []
    errors = 0
    for _ in range(args.requests):
        status_code, latency_ms = _post_predict(url=args.url, text=args.text)
        if status_code != 200:
            errors += 1
        latencies.append(latency_ms)

    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 20 else max(latencies)
    avg = statistics.mean(latencies)

    print(f"requests={args.requests}")
    print(f"errors={errors}")
    print(f"avg_latency_ms={avg:.3f}")
    print(f"p50_latency_ms={p50:.3f}")
    print(f"p95_latency_ms={p95:.3f}")


if __name__ == "__main__":
    main()
