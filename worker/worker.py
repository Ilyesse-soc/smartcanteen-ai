from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
from autogluon.tabular import TabularPredictor
from celery import Celery


BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
SHARED_DATA_DIR = Path(os.getenv("SHARED_DATA_DIR", "/shared_data")).resolve()
MODELS_BASE_DIR = Path(os.getenv("WORKER_MODELS_DIR", str(SHARED_DATA_DIR / "models"))).resolve()

app = Celery("smartcanteen_worker", broker=BROKER_URL, backend=RESULT_BACKEND)


@app.task(name="train_model_task")
def train_model_task(file_path: str, target_column: str) -> dict:
    resolved_file_path = Path(file_path)
    if not resolved_file_path.is_absolute():
        resolved_file_path = (SHARED_DATA_DIR / resolved_file_path).resolve()
    else:
        resolved_file_path = resolved_file_path.resolve()

    if not str(resolved_file_path).startswith(str(SHARED_DATA_DIR)):
        raise ValueError("Le fichier CSV doit être situé dans le volume partagé.")
    if not resolved_file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {resolved_file_path}")

    df = pd.read_csv(resolved_file_path)
    if target_column not in df.columns:
        raise ValueError(f"Colonne cible absente du CSV: {target_column}")

    # Protection mémoire: si le CSV dépasse 100 000 lignes, on force un échantillonnage strict.
    if len(df) > 100_000:
        df = df.sample(n=100_000, random_state=42)

    rows, cols = df.shape
    # Temps dynamique: 30s + (lignes * colonnes * 0.001), puis borné entre 60s et 1800s.
    computed_time_limit = 30 + (rows * cols * 0.001)
    time_limit = int(max(60, min(1800, computed_time_limit)))

    job_dir = None
    for _ in range(3):
        candidate = MODELS_BASE_DIR / f"job_{uuid4().hex}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            job_dir = candidate
            break
        except FileExistsError:
            continue
    if job_dir is None:
        raise RuntimeError("Impossible de créer un dossier de job unique.")

    predictor = TabularPredictor(label=target_column, path=str(job_dir)).fit(
        train_data=df,
        presets="medium_quality",
        time_limit=time_limit,
    )

    leaderboard_records = predictor.leaderboard(silent=True).to_dict(orient="records")
    return {
        "job_dir": str(job_dir),
        "time_limit": time_limit,
        "rows_used": rows,
        "columns_used": cols,
        "leaderboard": leaderboard_records,
    }
