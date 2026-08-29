"""Separa treino e validação para avaliar o modelo em dados não vistos.

O dataset original possui os rótulos de urgência. Avaliar no mesmo conjunto
usado no treinamento produziria uma medida otimista demais. Este script cria
dois CSVs com uma divisão estratificada e reproduzível, permitindo medir a
qualidade do modelo com ground truth antes do benchmark Joblib versus ONNX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"text", "urgency"}


def create_split(
    source_path: Path,
    train_path: Path,
    validation_path: Path,
    validation_size: float,
    random_state: int,
) -> None:
    """Divide o dataset rotulado em treino e validação estratificados.

    Args:
        source_path: CSV rotulado de origem, com as colunas ``text`` e
            ``urgency``.
        train_path: Destino do subconjunto usado para treinar o modelo.
        validation_path: Destino do subconjunto reservado para a avaliação.
        validation_size: Fração do dataset destinada à validação.
        random_state: Semente que torna a mesma divisão repetível.

    Raises:
        ValueError: Se o CSV de origem não tiver as colunas obrigatórias.
    """
    data = pd.read_csv(source_path)
    # A avaliação exige o texto de entrada e o rótulo real de cada exemplo.
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes: {sorted(missing_columns)}")

    # Estratificar preserva a proporção de normal, atenção e urgente nos dois
    # arquivos, evitando que classes menos frequentes desapareçam da validação.
    train_data, validation_data = train_test_split(
        data,
        test_size=validation_size,
        random_state=random_state,
        stratify=data["urgency"],
    )
    # Cria o diretório de saída também quando o comando é executado do zero.
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path, index=False)
    validation_data.to_csv(validation_path, index=False)
    # A distribuição permite conferir rapidamente se a estratificação ocorreu.
    print(f"train_rows={len(train_data)}")
    print(f"validation_rows={len(validation_data)}")
    print(f"validation_distribution={validation_data['urgency'].value_counts().to_dict()}")


def main() -> None:
    """Lê argumentos da linha de comando e gera os CSVs de treino e validação."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/processed/train_split.csv"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("data/processed/validation.csv"),
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    # ``train_test_split`` exige uma proporção estritamente entre zero e um.
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size deve estar entre 0 e 1")
    create_split(
        args.source,
        args.train_output,
        args.validation_output,
        args.validation_size,
        args.random_state,
    )


if __name__ == "__main__":
    main()
