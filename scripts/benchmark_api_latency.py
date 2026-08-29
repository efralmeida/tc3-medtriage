"""Benchmark simples de latencia da API de triagem."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def _post_predict(url: str, text: str) -> tuple[int, float]:
    """Chama ``POST /predict`` e mede a latencia total percebida pelo cliente.

    Args:
        url: URL base da API, sem a necessidade de informar ``/predict``.
        text: Texto do laudo enviado para classificação.

    Returns:
        Tupla com o código HTTP da resposta e a latencia total em milissegundos.
    """
    # A API espera um corpo JSON com a chave "text" codificado em UTF-8.
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Mede também serialização, rede e leitura da resposta, não apenas a inferência.
    start = time.perf_counter()
    with urllib.request.urlopen(request) as response:  # nosec: B310
        _ = response.read()
        status_code = response.getcode()
    latency_ms = (time.perf_counter() - start) * 1000
    return status_code, latency_ms


def main() -> None:
    """Executa requisições sequenciais e imprime estatísticas de latência.

    Os tempos coletados são do ponto de vista do cliente que executa o script.
    """
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
    # Cada iteração produz uma amostra independente de latência para a mesma entrada.
    for _ in range(args.requests):
        status_code, latency_ms = _post_predict(url=args.url, text=args.text)
        if status_code != 200:
            errors += 1
        latencies.append(latency_ms)

    p50 = statistics.median(latencies)
    # Com menos de 20 amostras, usa o pior tempo para evitar percentil instavel.
    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 20 else max(latencies)
    avg = statistics.mean(latencies)

    print(f"requests={args.requests}")
    print(f"errors={errors}")
    print(f"avg_latency_ms={avg:.3f}")
    print(f"p50_latency_ms={p50:.3f}")
    print(f"p95_latency_ms={p95:.3f}")


if __name__ == "__main__":
    main()
