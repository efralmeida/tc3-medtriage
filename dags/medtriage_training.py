"""DAG de ingestao, treinamento e salvamento do classificador MedTriage."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from airflow.decorators import dag, task # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DATA = PROJECT_DIR / "data" / "processed" / "train.csv"
STAGED_DATA = PROJECT_DIR / "data" / "airflow" / "train.csv"
MODEL_PATH = PROJECT_DIR / "models" / "text_classifier_best.joblib"
REQUIRED_COLUMNS = {"text", "urgency"}


def _validate_dataset(data: pd.DataFrame) -> None:
    """Valida o contrato minimo usado pelo treinamento."""
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(missing_columns)}")
    if data[["text", "urgency"]].isna().any().any():
        raise ValueError("O dataset contem valores nulos em text ou urgency.")
    if data["urgency"].nunique() < 2:
        raise ValueError("O dataset precisa conter ao menos duas classes.")


@dag(
    dag_id="medtriage_training",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["medtriage", "training"],
)
def medtriage_training():
    """Executa ingestao do CSV e treina o pipeline de classificacao."""

    @task
    def ingest_data() -> str:
        """Copia o dataset de treino para a area de trabalho do Airflow."""
        data = pd.read_csv(SOURCE_DATA)
        _validate_dataset(data)
        STAGED_DATA.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_DATA, STAGED_DATA)
        return str(STAGED_DATA)

    @task
    def train_and_save(data_path: str) -> str:
        """Treina e salva o pipeline TF-IDF + regressao logistica."""
        data = pd.read_csv(data_path)
        _validate_dataset(data)
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20_000)),
                ("classifier", LogisticRegression(max_iter=1_000)),
            ]
        )
        pipeline.fit(data["text"], data["urgency"])
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        return str(MODEL_PATH)

    train_and_save(ingest_data())


medtriage_training()