from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COL = "nb_repas_consommés"


@dataclass(frozen=True)
class FeatureSpec:
    date_col: str = "date"
    group_col: str = "cantine_id"
    target_col: str = TARGET_COL
    waste_col: str = "gaspillage_kg"


def parse_and_cast_types(df: pd.DataFrame, spec: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    out = df.copy()
    out[spec.date_col] = pd.to_datetime(out[spec.date_col], errors="coerce")

    int_cols = [
        "cantine_id",
        "mois",
        "semaine_annee",
        "vacances",
        "jour_ferie",
        "nb_inscrits",
        "nb_absents_prevus",
        "viande",
        "poisson",
        "vegetarien",
        "dessert_populaire",
        "pluie",
        "evenement_special",
    ]
    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    float_cols = [
        "temperature",
        "stock_disponible_kg",
        "quantite_produite_kg",
        "portion_moyenne_kg",
        "quantite_consommee_kg",
        "gaspillage_kg",
        "gaspillage_pct",
    ]
    for c in float_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if spec.target_col in out.columns:
        out[spec.target_col] = pd.to_numeric(out[spec.target_col], errors="coerce")
    return out


def add_time_features(df: pd.DataFrame, spec: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    out = df.copy()
    out["weekend"] = out["jour_semaine"].isin(["Samedi", "Dimanche"]).astype(int)

    # Feature métier: météo (catégorielle) dérivée de température + pluie.
    # On garde aussi les variables brutes (temperature/pluie) pour la partie numérique.
    pluie = out.get("pluie")
    temp = out.get("temperature")
    if pluie is not None and temp is not None:
        pluie_i = pd.to_numeric(pluie, errors="coerce").fillna(0).astype(int)
        temp_f = pd.to_numeric(temp, errors="coerce")

        def _meteo_label(t: float | None, p: int) -> str:
            if p == 1:
                if t is not None and not np.isnan(t) and t <= 5:
                    return "pluie_froid"
                if t is not None and not np.isnan(t) and t >= 25:
                    return "pluie_chaud"
                return "pluie"
            if t is not None and not np.isnan(t) and t <= 5:
                return "sec_froid"
            if t is not None and not np.isnan(t) and t >= 25:
                return "sec_chaud"
            return "sec"

        out["meteo"] = [
            _meteo_label(None if pd.isna(t) else float(t), int(p))
            for t, p in zip(temp_f.to_numpy(), pluie_i.to_numpy(), strict=False)
        ]

    if "semaine_annee" not in out.columns or out["semaine_annee"].isna().any():
        out["semaine_annee"] = (
            pd.to_datetime(out[spec.date_col], errors="coerce").dt.isocalendar().week.astype("Int64")
        )
    if "mois" not in out.columns or out["mois"].isna().any():
        out["mois"] = pd.to_datetime(out[spec.date_col], errors="coerce").dt.month.astype("Int64")
    return out


def add_lag_and_rolling_features(df: pd.DataFrame, spec: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    """Crée des lags et moyennes glissantes SANS fuite.

    Les lags/rolling sont calculés par cantine, triés par date, puis shift(1).
    """
    out = df.copy()
    out = out.sort_values([spec.group_col, spec.date_col]).reset_index(drop=True)

    # Lags (vectorisés)
    grp_target = out.groupby(spec.group_col)[spec.target_col]
    out["lag_1_repas"] = grp_target.shift(1)
    out["lag_7_repas"] = grp_target.shift(7)

    # Alias attendus (checklist): conserve compatibilité (on n'enlève rien)
    out["lag_1"] = out["lag_1_repas"]
    out["lag_7"] = out["lag_7_repas"]

    if spec.waste_col in out.columns:
        grp_waste = out.groupby(spec.group_col)[spec.waste_col]
        out["lag_1_gaspillage"] = grp_waste.shift(1)
        out["lag_7_gaspillage"] = grp_waste.shift(7)

    # Rolling 7 jours uniquement passé: shift(1) puis rolling
    out["roll7_repas"] = (
        out.groupby(spec.group_col)[spec.target_col]
        .apply(lambda s: s.shift(1).rolling(window=7, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )

    # Alias attendu (checklist)
    out["rolling_mean_7"] = out["roll7_repas"]
    if spec.waste_col in out.columns:
        out["roll7_gaspillage"] = (
            out.groupby(spec.group_col)[spec.waste_col]
            .apply(lambda s: s.shift(1).rolling(window=7, min_periods=3).mean())
            .reset_index(level=0, drop=True)
        )

    return out


def build_feature_frame(df_raw: pd.DataFrame, spec: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    df = df_raw.copy()
    # Robustesse inférence: si target/gaspillage n'existent pas, on les crée en NaN
    if spec.target_col not in df.columns:
        df[spec.target_col] = np.nan
    if spec.waste_col not in df.columns:
        df[spec.waste_col] = np.nan

    df = parse_and_cast_types(df, spec)
    df = add_time_features(df, spec)
    df = add_lag_and_rolling_features(df, spec)
    return df


def split_time_based(
    df: pd.DataFrame,
    date_col: str = "date",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(date_col)
    unique_dates = df_sorted[date_col].dropna().sort_values().unique()
    if len(unique_dates) < 30:
        raise ValueError("Pas assez de dates pour un split temporel.")
    cutoff_idx = int(np.floor((1.0 - test_size) * len(unique_dates)))
    cutoff_date = unique_dates[cutoff_idx]
    train_df = df_sorted[df_sorted[date_col] < cutoff_date].copy()
    test_df = df_sorted[df_sorted[date_col] >= cutoff_date].copy()
    return train_df, test_df
