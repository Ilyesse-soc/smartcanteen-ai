from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_prediction_service
from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.services.prediction_service import PredictionService


router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def create_prediction(
    payload: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return service.create_prediction(payload)
