from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    jour_semaine: Mapped[str] = mapped_column(String(32), nullable=False)
    mois: Mapped[int] = mapped_column(Integer, nullable=False)
    nb_inscrits: Mapped[int] = mapped_column(Integer, nullable=False)
    nb_absents_prevus: Mapped[int] = mapped_column(Integer, nullable=False)
    menu_type: Mapped[str] = mapped_column(String(32), nullable=False)
    stock_disponible_kg: Mapped[float] = mapped_column(Float, nullable=False)
    quantite_produite_kg: Mapped[float] = mapped_column(Float, nullable=False)
    portion_moyenne_kg: Mapped[float] = mapped_column(Float, nullable=False)

    repas_prevus: Mapped[int] = mapped_column(Integer, nullable=False)
    quantite_recommandee: Mapped[float] = mapped_column(Float, nullable=False)
    gaspillage_estime: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow, nullable=False)
