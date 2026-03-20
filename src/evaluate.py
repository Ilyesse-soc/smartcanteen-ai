from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt

try:
    from autogluon.tabular import TabularPredictor
except Exception:  # pragma: no cover
    TabularPredictor = None  # type: ignore

from src.config import PATHS
from src.feature_engineering import FeatureSpec, build_feature_frame, split_time_based
from src.preprocess import split_features_target
from src.metrics import compute_metrics
from src.utils import cast_nullable_int_to_float


def _plot_real_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Réel", linewidth=2)
    plt.plot(y_pred, label="Prédit", linewidth=2)
    plt.title("Réel vs Prédit (test)")
    plt.xlabel("Index temporel (test)")
    plt.ylabel("Nb repas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_error_over_time(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    err = y_pred - y_true
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(dates), err, linewidth=1)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Erreur (prédit - réel) dans le temps")
    plt.xlabel("Date")
    plt.ylabel("Erreur")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_feature_importance(artifact: dict[str, Any], out_path: Path, top_n: int = 15) -> None:
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    model_name = artifact.get("model_name", "model")

    if not hasattr(model, "feature_importances_"):
        return

    importances = np.asarray(model.feature_importances_)
    feature_names = preprocessor.get_feature_names_out()
    if len(importances) != len(feature_names):
        return

    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(idx)), importances[idx][::-1])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1])
    plt.title(f"Top {top_n} features — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def evaluate_and_report(df_raw: pd.DataFrame, artifacts_dir: Path | None = None) -> pd.DataFrame:
    if artifacts_dir is None:
        artifacts_dir = PATHS.models_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not PATHS.model_path.exists():
        from src.train import train_and_select_best

        train_and_select_best(df_raw)

    artifact = load(PATHS.model_path)
    spec: FeatureSpec = artifact["feature_spec"]
    leakage_cols = artifact.get("leakage_cols_dropped", [])

    df_feat = build_feature_frame(df_raw, spec)
    df_feat = df_feat.dropna(subset=[spec.date_col])
    _, test_df = split_time_based(df_feat, date_col=spec.date_col, test_size=0.2)

    test_ml = test_df.drop(columns=[spec.date_col] + leakage_cols, errors="ignore")
    X_test, y_test = split_features_target(test_ml, spec.target_col)

    if artifact.get("model_framework") == "autogluon":
        if TabularPredictor is None:
            raise RuntimeError("AutoGluon n'est pas importable pour l'évaluation.")
        ag_model_path = artifact.get("autogluon_model_path")
        if not ag_model_path:
            raise RuntimeError("Chemin du modèle AutoGluon introuvable dans l'artifact.")
        predictor = TabularPredictor.load(ag_model_path)
        test_ag = cast_nullable_int_to_float(test_ml)
        y_pred = np.clip(predictor.predict(test_ag).to_numpy(dtype=float), 0, None)
    else:
        preprocessor = artifact["preprocessor"]
        model = artifact["model"]
        X_test_pp = preprocessor.transform(X_test)
        y_pred = np.clip(model.predict(X_test_pp), 0, None)

    metrics = compute_metrics(y_test.values, y_pred)
    report = pd.DataFrame([{"model": artifact.get("model_name", "best_model"), **metrics}])

    _plot_real_vs_pred(y_test.values, y_pred, artifacts_dir / "real_vs_pred.png")
    _plot_error_over_time(test_df[spec.date_col], y_test.values, y_pred, artifacts_dir / "error_over_time.png")
    _plot_feature_importance(artifact, artifacts_dir / "feature_importance.png", top_n=15)

    print("\n=== Rapport évaluation (best model) ===")
    print(report.to_string(index=False))
    print("\n=== Comparatif (train) ===")
    metrics_table = pd.DataFrame(
        [
            {"model": k, **v}
            for k, v in artifact.get("metrics", {}).items()
            if isinstance(v, dict) and {"MAE", "RMSE", "R2", "MAPE"}.issubset(v.keys())
        ]
    ).sort_values("MAE")
    if len(metrics_table) > 0:
        print(metrics_table.to_string(index=False))
    return report
