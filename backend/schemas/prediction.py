from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jour_semaine: Literal[
        "Lundi",
        "Mardi",
        "Mercredi",
        "Jeudi",
        "Vendredi",
        "Samedi",
        "Dimanche",
    ]
    mois: int = Field(ge=1, le=12)
    nb_inscrits: int = Field(ge=0)
    nb_absents_prevus: int = Field(ge=0)
    menu_type: Literal["standard", "poisson", "vegetarien", "pizza", "pates", "fete"]
    stock_disponible_kg: float = Field(ge=0)
    quantite_produite_kg: float = Field(ge=0)
    portion_moyenne_kg: float = Field(gt=0)


class PredictionResponse(BaseModel):
    repas_prevus: int
    quantite_recommandee: float
    gaspillage_estime: float
    alertes: list[str]
    message: str
