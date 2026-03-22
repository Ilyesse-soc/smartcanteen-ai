from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.core.database import get_db_session
from backend.repositories.prediction_repository import PredictionRepository
from backend.services.prediction_service import PredictionService


def get_prediction_repository(session: Session = Depends(get_db_session)) -> PredictionRepository:
    return PredictionRepository(session=session)


def get_prediction_service(
    repository: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionService:
    return PredictionService(repository=repository)
