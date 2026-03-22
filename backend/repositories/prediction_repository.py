from __future__ import annotations

from sqlalchemy.orm import Session

from backend.models.prediction import Prediction


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, prediction: Prediction) -> Prediction:
        self._session.add(prediction)
        self._session.commit()
        self._session.refresh(prediction)
        return prediction
