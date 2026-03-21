from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def to_jsonable(d: Any) -> Any:
    """Convertit quelques types courants (Path, dataclass) en JSON-serializable."""
    if hasattr(d, "__dataclass_fields__"):
        return asdict(d)
    if isinstance(d, Path):
        return str(d)
    return d


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=to_jsonable)


def cast_nullable_int_to_float(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    nullable_int_cols = [
        c for c, dt in out.dtypes.items() if is_integer_dtype(dt) and hasattr(dt, "na_value")
    ]
    if nullable_int_cols:
        out[nullable_int_cols] = out[nullable_int_cols].astype("float64")
    return out
