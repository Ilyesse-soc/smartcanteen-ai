from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.dependencies import get_prediction_service
from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.services.prediction_service import ModelUnavailableError, PredictionService


router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def create_prediction(
    payload: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    try:
        return service.create_prediction(payload)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
