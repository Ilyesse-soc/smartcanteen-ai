from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.utils import set_global_seed


@dataclass(frozen=True)
class CantineProfile:
    cantine_id: int
    type_cantine: str
    base_inscrits: int


JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def _is_weekend(day_name_fr: str) -> int:
    return int(day_name_fr in {"Samedi", "Dimanche"})


def _jours_feries_fr(year: int) -> set[date]:
    # Liste simplifiée (jours fixes). Suffisant pour un prototype.
    fixed = [
        (1, 1),   # Jour de l'an
        (5, 1),   # Fête du travail
        (5, 8),   # Victoire 1945
        (7, 14),  # Fête nationale
        (8, 15),  # Assomption
        (11, 1),  # Toussaint
        (11, 11), # Armistice
        (12, 25), # Noël
    ]
    return {date(year, m, d) for (m, d) in fixed}


def _is_vacances_fr(d: date) -> int:
    # Heuristique simple: été (juil/août) + fin décembre + 1ère semaine de janvier.
    if d.month in {7, 8}:
        return 1
    if d.month == 12 and d.day >= 20:
        return 1
    if d.month == 1 and d.day <= 7:
        return 1
    return 0


def _seasonal_temperature(d: date, rng: np.random.Generator) -> float:
    # Température saisonnière: sinus + bruit
    day_of_year = d.timetuple().tm_yday
    base = 12.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    noise = rng.normal(0, 2.5)
    return float(np.clip(base + noise, -5, 38))


def _rain_indicator(d: date, rng: np.random.Generator) -> int:
    # Probabilité de pluie un peu plus élevée en hiver/printemps
    if d.month in {11, 12, 1, 2, 3, 4}:
        p = 0.45
    elif d.month in {5, 6, 10}:
        p = 0.35
    else:
        p = 0.25
    return int(rng.random() < p)


def _sample_menu_type(d: date, rng: np.random.Generator) -> str:
    # Quelques types de menus plausibles
    # Jour "spécial": plus de chances de menu "fete" en décembre
    if d.month == 12 and rng.random() < 0.12:
        return "fete"

    menus = ["standard", "eco", "pizza", "pates", "poisson", "vegetarien", "gourmet"]
    probs = np.array([0.40, 0.16, 0.10, 0.10, 0.08, 0.10, 0.06])
    return str(rng.choice(menus, p=probs))


def _menu_flags(menu_type: str) -> tuple[int, int, int, int]:
    # viande, poisson, vegetarien, dessert_populaire
    if menu_type == "poisson":
        return (0, 1, 0, 1)
    if menu_type == "vegetarien":
        return (0, 0, 1, 1)
    if menu_type in {"pizza", "pates"}:
        return (1, 0, 0, 1)
    if menu_type == "fete":
        return (1, 0, 0, 1)
    if menu_type == "gourmet":
        return (1, 0, 0, 0)
    # standard / eco
    return (1, 0, 0, 0)


def _event_special(d: date, rng: np.random.Generator) -> int:
    # Événements plus probables en juin et décembre
    p = 0.02
    if d.month in {6, 12}:
        p = 0.05
    return int(rng.random() < p)


def generate_synthetic_canteen_dataset(
    n_days: int = 365,
    start_date: date | None = None,
    cantines: Iterable[CantineProfile] | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Génère un dataset synthétique "1 ligne = 1 jour pour une cantine".

    Logique métier incluse:
    - baisse de fréquentation vacances / jours fériés
    - variations jour de semaine
    - impact météo (température, pluie)
    - menus + événements spéciaux
    - gaspillage dépendant de la surproduction
    """

    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    if start_date is None:
        # Historique qui finit à J-1 (utile pour lags)
        start_date = date.today() - timedelta(days=n_days)

    if cantines is None:
        cantines = [
            CantineProfile(cantine_id=1, type_cantine="scolaire", base_inscrits=480),
            CantineProfile(cantine_id=2, type_cantine="entreprise", base_inscrits=240),
        ]

    all_rows: list[dict] = []
    for day_offset in range(n_days):
        d = start_date + timedelta(days=day_offset)
        weekday = d.weekday()  # 0 lundi
        jour_semaine = JOURS[weekday]
        mois = d.month
        semaine_annee = int(pd.Timestamp(d).isocalendar().week)

        feries = _jours_feries_fr(d.year)
        jour_ferie = int(d in feries)
        vacances = _is_vacances_fr(d)
        weekend = _is_weekend(jour_semaine)

        temperature = _seasonal_temperature(d, rng)
        pluie = _rain_indicator(d, rng)
        evenement_special = _event_special(d, rng)

        menu_type = _sample_menu_type(d, rng)
        viande, poisson, vegetarien, dessert_populaire = _menu_flags(menu_type)

        for c in cantines:
            # Base inscrits avec légère variabilité (tendance annuelle)
            trend = 1.0 + 0.03 * np.sin(2 * np.pi * (day_offset / max(1, n_days)))
            base_inscrits = int(max(40, c.base_inscrits * trend + rng.normal(0, c.base_inscrits * 0.03)))

            # Fermeture weekend pour scolaire/entreprise
            open_factor = 0.0 if weekend == 1 else 1.0

            # Impact vacances / fériés
            holiday_factor = 0.55 if vacances == 1 else 1.0
            if jour_ferie == 1:
                holiday_factor *= 0.15

            # Jour de semaine
            weekday_factor = {
                "Lundi": 0.96,
                "Mardi": 1.02,
                "Mercredi": 1.00,
                "Jeudi": 1.01,
                "Vendredi": 0.94,
                "Samedi": 0.10,
                "Dimanche": 0.05,
            }[jour_semaine]

            # Menu: effet de popularité
            menu_factor = {
                "standard": 1.00,
                "eco": 0.97,
                "pizza": 1.06,
                "pates": 1.04,
                "poisson": 0.96,
                "vegetarien": 0.98,
                "gourmet": 1.03,
                "fete": 1.08,
            }.get(menu_type, 1.0)

            # Météo
            rain_factor = 0.985 if pluie == 1 else 1.0
            temp_factor = 0.995
            if temperature < 0 or temperature > 30:
                temp_factor = 0.985

            # Événement spécial
            event_factor = 1.12 if evenement_special == 1 else 1.0

            # Demande latente
            latent_meals = base_inscrits * open_factor * holiday_factor * weekday_factor
            latent_meals *= menu_factor * rain_factor * temp_factor * event_factor

            # Absents prévus
            abs_base = 0.06 + (0.03 if vacances == 1 else 0.0) + (0.02 if pluie == 1 else 0.0)
            abs_base += 0.02 if jour_semaine in {"Lundi", "Vendredi"} else 0.0
            nb_absents_prevus = int(
                np.clip(rng.normal(abs_base * base_inscrits, base_inscrits * 0.02), 0, base_inscrits)
            )

            # Repas consommés
            dessert_factor = 1.01 if dessert_populaire == 1 else 1.0
            meals_mean = max(0.0, latent_meals * dessert_factor - nb_absents_prevus)
            nb_repas = int(
                np.clip(rng.normal(meals_mean, max(5.0, meals_mean * 0.04)), 0, base_inscrits)
            )

            # Portion moyenne (kg)
            portion_base = 0.52 if c.type_cantine == "scolaire" else 0.56
            if menu_type == "eco":
                portion_base -= 0.02
            if menu_type in {"gourmet", "fete"}:
                portion_base += 0.02
            portion_moyenne_kg = float(np.clip(rng.normal(portion_base, 0.02), 0.38, 0.75))

            # Planification production (pas basée sur nb_repas réel)
            safety_planning = 0.04 + (0.03 if evenement_special == 1 else 0.0)
            production_uncertainty = rng.normal(0.0, 0.03)
            planned_meals = max(0.0, latent_meals - (0.8 * nb_absents_prevus))
            quantite_produite_kg = float(
                np.clip(
                    planned_meals * portion_moyenne_kg * (1.0 + safety_planning + production_uncertainty),
                    0.0,
                    5000.0,
                )
            )

            # Stock disponible (kg)
            stock_noise = rng.normal(0.0, 12.0)
            stock_disponible_kg = float(
                np.clip(quantite_produite_kg + stock_noise + rng.normal(0, 8.0), 0.0, 6000.0)
            )

            # Consommation et gaspillage
            quantite_consommee_kg = float(nb_repas * portion_moyenne_kg)
            prep_losses = float(np.clip(rng.normal(1.2, 0.6), 0.0, 6.0))
            gaspillage_kg = float(max(0.0, quantite_produite_kg - quantite_consommee_kg) + prep_losses)
            gaspillage_pct = float(
                0.0
                if quantite_produite_kg <= 0
                else np.clip(gaspillage_kg / quantite_produite_kg, 0.0, 1.0)
            )

            all_rows.append(
                {
                    "date": d.isoformat(),
                    "cantine_id": c.cantine_id,
                    "type_cantine": c.type_cantine,
                    "jour_semaine": jour_semaine,
                    "mois": mois,
                    "semaine_annee": semaine_annee,
                    "vacances": vacances,
                    "jour_ferie": jour_ferie,
                    "nb_inscrits": base_inscrits,
                    "nb_absents_prevus": nb_absents_prevus,
                    "menu_type": menu_type,
                    "viande": viande,
                    "poisson": poisson,
                    "vegetarien": vegetarien,
                    "dessert_populaire": dessert_populaire,
                    "temperature": temperature,
                    "pluie": pluie,
                    "evenement_special": evenement_special,
                    "stock_disponible_kg": stock_disponible_kg,
                    "quantite_produite_kg": quantite_produite_kg,
                    "nb_repas_consommés": nb_repas,
                    "portion_moyenne_kg": portion_moyenne_kg,
                    "quantite_consommee_kg": quantite_consommee_kg,
                    "gaspillage_kg": gaspillage_kg,
                    "gaspillage_pct": gaspillage_pct,
                }
            )

    return pd.DataFrame(all_rows)
