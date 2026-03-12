from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métriques standard pour la régression.

    - MAPE est calculée en excluant les jours à 0 repas.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    # scikit-learn 1.8+ a retiré l'argument `squared`; on calcule RMSE = sqrt(MSE)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    mask = y_true > 0
    if mask.sum() >= 10:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
