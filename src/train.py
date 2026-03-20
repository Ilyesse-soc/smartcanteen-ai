from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception as e:  # pragma: no cover
    XGBRegressor = None  # type: ignore
    _XGB_IMPORT_ERROR = e

try:
    from autogluon.tabular import TabularPredictor
except Exception as e:  # pragma: no cover
    TabularPredictor = None  # type: ignore
    _AUTOGLUON_IMPORT_ERROR = e

from src.config import PATHS, RANDOM_SEED
from src.metrics import compute_metrics
from src.feature_engineering import FeatureSpec, build_feature_frame, split_time_based
from src.preprocess import PreprocessSpec, build_preprocessor, split_features_target
from src.utils import cast_nullable_int_to_float


@dataclass(frozen=True)
class TrainingResult:
    best_model_name: str
    model_path: Path
    metrics_table: pd.DataFrame
    artifacts: dict[str, Any]


def _baseline_rolling7_predict(df_all: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """Baseline: moyenne des 7 derniers jours (par cantine), sans fuite."""
    df_sorted = df_all.sort_values([spec.group_col, spec.date_col]).copy()
    pred = (
        df_sorted.groupby(spec.group_col)[spec.target_col]
        .apply(lambda s: s.shift(1).rolling(window=7, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )
    fallback = (
        df_sorted.groupby([spec.group_col, "jour_semaine"])[spec.target_col]
        .apply(lambda s: s.shift(1).expanding(min_periods=5).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    pred = pred.fillna(fallback)
    pred = pred.fillna(df_sorted[spec.target_col].mean())
    return pred.loc[df_all.index]


def train_and_select_best(
    df_raw: pd.DataFrame,
    *,
    use_autogluon: bool = False,
    autogluon_time_limit: int = 120,
    autogluon_presets: str = "medium_quality",
) -> TrainingResult:
    spec = FeatureSpec()
    df_feat = build_feature_frame(df_raw, spec)
    df_feat = df_feat.dropna(subset=[spec.date_col])

    train_df, test_df = split_time_based(df_feat, date_col=spec.date_col, test_size=0.2)

    all_for_baseline = pd.concat([train_df, test_df], axis=0)
    baseline_pred_all = _baseline_rolling7_predict(all_for_baseline, spec)
    baseline_pred = baseline_pred_all.loc[test_df.index]
    baseline_metrics = compute_metrics(test_df[spec.target_col].values, baseline_pred.values)

    drop_cols = [spec.date_col]
    leakage_cols = ["quantite_consommee_kg", "gaspillage_pct"]
    leakage_cols = [c for c in leakage_cols if c in df_feat.columns]

    train_ml = train_df.drop(columns=drop_cols + leakage_cols, errors="ignore")
    test_ml = test_df.drop(columns=drop_cols + leakage_cols, errors="ignore")

    X_train, y_train = split_features_target(train_ml, spec.target_col)
    X_test, y_test = split_features_target(test_ml, spec.target_col)

    categorical_cols = ["type_cantine", "jour_semaine", "menu_type", "meteo"]
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    preprocess_spec = PreprocessSpec(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    preprocessor = build_preprocessor(preprocess_spec)

    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp = preprocessor.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=450,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )
    rf.fit(X_train_pp, y_train)
    rf_pred = np.clip(rf.predict(X_test_pp), 0, None)
    rf_metrics = compute_metrics(y_test.values, rf_pred)

    if XGBRegressor is None:
        raise RuntimeError(
            "xgboost n'est pas importable. Installez les dépendances via requirements.txt. "
            f"Détail import: {_XGB_IMPORT_ERROR}"
        )

    xgb = XGBRegressor(
        n_estimators=900,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb.fit(X_train_pp, y_train)
    xgb_pred = np.clip(xgb.predict(X_test_pp), 0, None)
    xgb_metrics = compute_metrics(y_test.values, xgb_pred)

    autogluon_metrics: dict[str, float] | None = None
    autogluon_model_path: str | None = None
    if use_autogluon:
        if TabularPredictor is None:
            raise RuntimeError(
                "AutoGluon n'est pas importable. Installez les dépendances via requirements.txt. "
                f"Détail import: {_AUTOGLUON_IMPORT_ERROR}"
            )
        train_ag = cast_nullable_int_to_float(train_ml)
        test_ag = cast_nullable_int_to_float(test_ml)

        ag_path = PATHS.models_dir / "autogluon"
        ag_path.mkdir(parents=True, exist_ok=True)
        ag = TabularPredictor(
            label=spec.target_col,
            problem_type="regression",
            eval_metric="mean_absolute_error",
            path=str(ag_path),
        )
        ag.fit(
            train_data=train_ag,
            presets=autogluon_presets,
            time_limit=autogluon_time_limit,
            verbosity=0,
        )
        ag_pred = np.clip(ag.predict(test_ag).to_numpy(dtype=float), 0, None)
        autogluon_metrics = compute_metrics(y_test.values, ag_pred)
        autogluon_model_path = str(ag.path)

    rows = [
        {"model": "baseline_roll7", **baseline_metrics},
        {"model": "random_forest", **rf_metrics},
        {"model": "xgboost", **xgb_metrics},
    ]
    if autogluon_metrics is not None:
        rows.append({"model": "autogluon", **autogluon_metrics})

    metrics_table = pd.DataFrame(rows).sort_values("MAE")

    best_ml_name, best_model = ("random_forest", rf)
    best_ml_metrics = rf_metrics
    if xgb_metrics["MAE"] < rf_metrics["MAE"]:
        best_ml_name, best_model = ("xgboost", xgb)
        best_ml_metrics = xgb_metrics
    if autogluon_metrics is not None and autogluon_metrics["MAE"] < best_ml_metrics["MAE"]:
        best_ml_name, best_model = ("autogluon", None)
        best_ml_metrics = autogluon_metrics

    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": best_ml_name,
        "model": best_model,
        "model_framework": "autogluon" if best_ml_name == "autogluon" else "sklearn",
        "autogluon_model_path": autogluon_model_path,
        "preprocessor": preprocessor,
        "preprocess_spec": preprocess_spec,
        "feature_spec": spec,
        "leakage_cols_dropped": leakage_cols,
        "metrics": {
            "baseline_roll7": baseline_metrics,
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
            "best_ml": best_ml_metrics,
        },
    }
    if autogluon_metrics is not None:
        artifact["metrics"]["autogluon"] = autogluon_metrics
    dump(artifact, PATHS.model_path)

    return TrainingResult(
        best_model_name=best_ml_name,
        model_path=PATHS.model_path,
        metrics_table=metrics_table,
        artifacts=artifact,
    )
