from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from src.business import BusinessResult, build_business_result
from src.config import DEFAULT_SAFETY_MARGIN, PATHS
from src.feature_engineering import FeatureSpec, build_feature_frame


@dataclass(frozen=True)
class PredictionOutput:
    nb_repas_predits: int
    quantite_recommandee_kg: float
    gaspillage_estime_kg: float
    alertes: list[str]
    kpis: dict[str, float]
    message: str


def load_artifact(model_path: Path | None = None) -> dict[str, Any]:
    if model_path is None:
        model_path = PATHS.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {model_path}. Lancez d'abord `python main.py --all`."
        )
    return load(model_path)


def _normalize_streamlit_input(user_input: dict[str, Any]) -> dict[str, Any]:
    x = dict(user_input)
    x.setdefault("cantine_id", 1)
    x.setdefault("type_cantine", "scolaire")

    x.setdefault("temperature", 12.0)
    x.setdefault("pluie", 0)

    menu_type = str(x.get("menu_type", "standard"))
    if menu_type == "poisson":
        x.setdefault("viande", 0)
        x.setdefault("poisson", 1)
        x.setdefault("vegetarien", 0)
        x.setdefault("dessert_populaire", 1)
    elif menu_type == "vegetarien":
        x.setdefault("viande", 0)
        x.setdefault("poisson", 0)
        x.setdefault("vegetarien", 1)
        x.setdefault("dessert_populaire", 1)
    elif menu_type in {"pizza", "pates", "fete"}:
        x.setdefault("viande", 1)
        x.setdefault("poisson", 0)
        x.setdefault("vegetarien", 0)
        x.setdefault("dessert_populaire", 1)
    else:
        x.setdefault("viande", 1)
        x.setdefault("poisson", 0)
        x.setdefault("vegetarien", 0)
        x.setdefault("dessert_populaire", 0)

    x.setdefault("vacances", 0)
    x.setdefault("jour_ferie", 0)

    if "semaine_annee" not in x:
        m = int(x.get("mois", 1))
        x["semaine_annee"] = int(np.clip((m - 1) * 4 + 2, 1, 52))

    return x


def predict_from_dataframe(
    artifact: dict[str, Any],
    input_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> PredictionOutput:
    spec: FeatureSpec = artifact["feature_spec"]
    leakage_cols = artifact.get("leakage_cols_dropped", [])

    if history_df is not None and len(history_df) > 0:
        hist = history_df.copy()
        hist["_is_input"] = 0
        inp = input_df.copy()
        inp["_is_input"] = 1
        df_for_feat = pd.concat([hist, inp], axis=0, ignore_index=True)
        df_feat_all = build_feature_frame(df_for_feat, spec)
        df_feat = df_feat_all[df_feat_all["_is_input"] == 1].copy()
    else:
        df_feat = build_feature_frame(input_df, spec)

    df_feat = df_feat.drop(columns=["_is_input"], errors="ignore")

    df_feat = df_feat.drop(columns=[spec.date_col], errors="ignore")
    df_feat = df_feat.drop(columns=leakage_cols, errors="ignore")

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    X_pp = preprocessor.transform(df_feat)
    y_pred = float(np.clip(model.predict(X_pp)[0], 0, None))

    stock_disponible_kg = float(input_df.iloc[0].get("stock_disponible_kg", 0.0))
    quantite_produite_kg = float(input_df.iloc[0].get("quantite_produite_kg", 0.0))
    portion_moyenne_kg = float(input_df.iloc[0].get("portion_moyenne_kg", 0.52))
    nb_inscrits = float(input_df.iloc[0].get("nb_inscrits", 0.0))
    nb_absents_prevus = float(input_df.iloc[0].get("nb_absents_prevus", 0.0))

    # Contrainte métier: on centre autour de (inscrits - absents) et on laisse le modèle ajuster ±10%,
    # tout en imposant 0 <= repas <= inscrits.
    repas_model = int(max(0, round(y_pred)))
    base = float(max(0.0, nb_inscrits - nb_absents_prevus))
    lower = int(max(0, round(base * 0.90)))
    upper = int(min(round(nb_inscrits), round(base * 1.10)))
    if upper < lower:
        upper = lower
    repas_predits = int(np.clip(repas_model, lower, upper))

    business: BusinessResult = build_business_result(
        nb_repas_predits=float(repas_predits),
        portion_moyenne_kg=portion_moyenne_kg,
        marge_securite=safety_margin,
        stock_disponible_kg=stock_disponible_kg,
        quantite_produite_kg=quantite_produite_kg,
        nb_inscrits=nb_inscrits,
        nb_absents_prevus=nb_absents_prevus,
    )

    return PredictionOutput(
        nb_repas_predits=business.nb_repas_predits,
        quantite_recommandee_kg=business.quantite_recommandee_kg,
        gaspillage_estime_kg=business.gaspillage_estime_kg,
        alertes=business.alertes,
        kpis=business.kpis,
        message=business.message,
    )


def predict_from_dict(
    artifact: dict[str, Any],
    user_input: dict[str, Any],
    history_df: pd.DataFrame | None = None,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> dict[str, Any]:
    normalized = _normalize_streamlit_input(user_input)
    normalized.setdefault("date", pd.Timestamp.today().normalize().date().isoformat())

    input_df = pd.DataFrame([normalized])
    out = predict_from_dataframe(artifact, input_df, history_df=history_df, safety_margin=safety_margin)
    payload = {
        "nb_repas_predits": out.nb_repas_predits,
        "quantite_recommandee_kg": out.quantite_recommandee_kg,
        "gaspillage_estime_kg": out.gaspillage_estime_kg,
        "alertes": out.alertes,
        "kpis": out.kpis,
        "message": out.message,
    }
    # Alias “métier” (checklist) — ne pas casser les clés existantes
    payload.update(
        {
            "repas_prevus": payload["nb_repas_predits"],
            "quantite_recommandee": payload["quantite_recommandee_kg"],
            "gaspillage_estime": payload["gaspillage_estime_kg"],
        }
    )
    return payload
