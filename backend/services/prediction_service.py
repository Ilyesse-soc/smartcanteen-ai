from __future__ import annotations

from functools import lru_cache

from src.config import DEFAULT_SAFETY_MARGIN, PATHS
from src.predict import load_artifact, predict_from_dict

from backend.models.prediction import Prediction
from backend.repositories.prediction_repository import PredictionRepository
from backend.schemas.prediction import PredictionRequest, PredictionResponse


class ModelUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _get_artifact() -> dict:
    return load_artifact(PATHS.model_path)


class PredictionService:
    def __init__(self, repository: PredictionRepository) -> None:
        self._repository = repository

    def create_prediction(self, request: PredictionRequest) -> PredictionResponse:
        try:
            artifact = _get_artifact()
        except FileNotFoundError as exc:
            raise ModelUnavailableError(
                "Modele ML introuvable. Lancez `python main.py --all` pour generer models/trained_model.joblib."
            ) from exc

        user_input = {
            "jour_semaine": request.jour_semaine,
            "mois": request.mois,
            "nb_inscrits": request.nb_inscrits,
            "nb_absents_prevus": request.nb_absents_prevus,
            "menu_type": request.menu_type,
            "stock_disponible_kg": request.stock_disponible_kg,
            "quantite_produite_kg": request.quantite_produite_kg,
            "portion_moyenne_kg": request.portion_moyenne_kg,
        }
        prediction = predict_from_dict(
            artifact=artifact,
            user_input=user_input,
            history_df=None,
            safety_margin=DEFAULT_SAFETY_MARGIN,
        )

        record = Prediction(
            jour_semaine=request.jour_semaine,
            mois=request.mois,
            nb_inscrits=request.nb_inscrits,
            nb_absents_prevus=request.nb_absents_prevus,
            menu_type=request.menu_type,
            stock_disponible_kg=request.stock_disponible_kg,
            quantite_produite_kg=request.quantite_produite_kg,
            portion_moyenne_kg=request.portion_moyenne_kg,
            repas_prevus=int(prediction["repas_prevus"]),
            quantite_recommandee=float(prediction["quantite_recommandee"]),
            gaspillage_estime=float(prediction["gaspillage_estime"]),
        )
        self._repository.create(record)

        return PredictionResponse(
            repas_prevus=int(prediction["repas_prevus"]),
            quantite_recommandee=float(prediction["quantite_recommandee"]),
            gaspillage_estime=float(prediction["gaspillage_estime"]),
            alertes=list(prediction["alertes"]),
            message=str(prediction["message"]),
        )
