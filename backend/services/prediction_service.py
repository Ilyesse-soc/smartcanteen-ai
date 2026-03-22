from __future__ import annotations

from src.business import build_business_result
from src.config import DEFAULT_SAFETY_MARGIN

from backend.models.prediction import Prediction
from backend.repositories.prediction_repository import PredictionRepository
from backend.schemas.prediction import PredictionRequest, PredictionResponse


class PredictionService:
    def __init__(self, repository: PredictionRepository) -> None:
        self._repository = repository

    def create_prediction(self, request: PredictionRequest) -> PredictionResponse:
        nb_repas_attendus = max(0, request.nb_inscrits - request.nb_absents_prevus)
        business_result = build_business_result(
            nb_repas_predits=float(nb_repas_attendus),
            portion_moyenne_kg=request.portion_moyenne_kg,
            marge_securite=DEFAULT_SAFETY_MARGIN,
            stock_disponible_kg=request.stock_disponible_kg,
            quantite_produite_kg=request.quantite_produite_kg,
            nb_inscrits=float(request.nb_inscrits),
            nb_absents_prevus=float(request.nb_absents_prevus),
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
            repas_prevus=business_result.nb_repas_predits,
            quantite_recommandee=business_result.quantite_recommandee_kg,
            gaspillage_estime=business_result.gaspillage_estime_kg,
        )
        self._repository.create(record)

        return PredictionResponse(
            repas_prevus=business_result.nb_repas_predits,
            quantite_recommandee=business_result.quantite_recommandee_kg,
            gaspillage_estime=business_result.gaspillage_estime_kg,
            alertes=business_result.alertes,
            message=business_result.message,
        )
