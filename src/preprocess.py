from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class PreprocessSpec:
    categorical_cols: list[str]
    numeric_cols: list[str]


def build_preprocessor(spec: PreprocessSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric_cols),
            ("cat", categorical_pipe, spec.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target absente: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
