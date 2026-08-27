"""Cria conjuntos reproduziveis de treino e validacao a partir do train.csv."""

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
    """Divide o dataset estratificando pela classe de urgencia."""
    data = pd.read_csv(source_path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes: {sorted(missing_columns)}")

    train_data, validation_data = train_test_split(
        data,
        test_size=validation_size,
        random_state=random_state,
        stratify=data["urgency"],
    )
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path, index=False)
    validation_data.to_csv(validation_path, index=False)
    print(f"train_rows={len(train_data)}")
    print(f"validation_rows={len(validation_data)}")
    print(f"validation_distribution={validation_data['urgency'].value_counts().to_dict()}")


def main() -> None:
    """Processa argumentos e cria os arquivos de treino e validacao."""
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
