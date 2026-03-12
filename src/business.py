from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BusinessResult:
    nb_repas_predits: int
    quantite_recommandee_kg: float
    gaspillage_estime_kg: float
    alertes: list[str]
    kpis: dict[str, float]
    message: str


def calcul_quantite_recommandee(
    nb_repas_predits: float,
    portion_moyenne_kg: float,
    marge_securite: float = 0.05,
) -> float:
    q = float(nb_repas_predits) * float(portion_moyenne_kg) * (1.0 + float(marge_securite))
    return float(max(0.0, q))


def calcul_gaspillage_estime(quantite_produite_kg: float, quantite_recommandee_kg: float) -> float:
    return float(max(0.0, float(quantite_produite_kg) - float(quantite_recommandee_kg)))


def calcul_kpis(
    nb_inscrits: float,
    nb_absents_prevus: float,
    nb_repas_predits: float,
    quantite_produite_kg: float,
    quantite_recommandee_kg: float,
    gaspillage_estime_kg: float,
) -> dict[str, float]:
    nb_inscrits = float(nb_inscrits)
    nb_absents_prevus = float(nb_absents_prevus)
    nb_repas_predits = float(nb_repas_predits)

    taux_presence = 0.0 if nb_inscrits <= 0 else float(
        np.clip((nb_inscrits - nb_absents_prevus) / nb_inscrits, 0.0, 1.0)
    )
    surproduction_kg = float(max(0.0, quantite_produite_kg - quantite_recommandee_kg))
    gaspillage_pct_estime = 0.0 if quantite_produite_kg <= 0 else float(
        np.clip(gaspillage_estime_kg / quantite_produite_kg, 0.0, 1.0)
    )

    return {
        "taux_presence_estime": taux_presence,
        "surproduction_kg": surproduction_kg,
        "gaspillage_pct_estime": gaspillage_pct_estime,
        "repas_predits": nb_repas_predits,
    }


def generation_message_alerte(
    nb_repas_predits: float,
    nb_inscrits: float,
    stock_disponible_kg: float,
    quantite_recommandee_kg: float,
    quantite_produite_kg: float,
    gaspillage_estime_kg: float,
) -> list[str]:
    alertes: list[str] = []

    if nb_inscrits > 0 and (nb_repas_predits / nb_inscrits) > 0.90:
        alertes.append("Journée à forte fréquentation prévue (proche du maximum d'inscrits).")
    if stock_disponible_kg < quantite_recommandee_kg:
        alertes.append("Stock potentiellement insuffisant vs quantité recommandée.")
    if quantite_produite_kg > quantite_recommandee_kg * 1.08:
        alertes.append("Risque de surproduction (production significativement au-dessus du recommandé).")
    if gaspillage_estime_kg > 12:
        alertes.append("Gaspillage élevé estimé (considérez une réduction de production).")

    if len(alertes) == 0:
        alertes.append("Situation nominale: pas d'alerte critique.")
    return alertes


def build_business_result(
    nb_repas_predits: float,
    portion_moyenne_kg: float,
    marge_securite: float,
    stock_disponible_kg: float,
    quantite_produite_kg: float,
    nb_inscrits: float,
    nb_absents_prevus: float,
) -> BusinessResult:
    q_rec = calcul_quantite_recommandee(nb_repas_predits, portion_moyenne_kg, marge_securite)
    g_est = calcul_gaspillage_estime(quantite_produite_kg, q_rec)
    alertes = generation_message_alerte(
        nb_repas_predits=nb_repas_predits,
        nb_inscrits=nb_inscrits,
        stock_disponible_kg=stock_disponible_kg,
        quantite_recommandee_kg=q_rec,
        quantite_produite_kg=quantite_produite_kg,
        gaspillage_estime_kg=g_est,
    )
    kpis = calcul_kpis(
        nb_inscrits=nb_inscrits,
        nb_absents_prevus=nb_absents_prevus,
        nb_repas_predits=nb_repas_predits,
        quantite_produite_kg=quantite_produite_kg,
        quantite_recommandee_kg=q_rec,
        gaspillage_estime_kg=g_est,
    )

    msg = (
        f"Prévision: ~{int(round(nb_repas_predits))} repas. "
        f"Recommandation: {q_rec:.1f} kg (marge {marge_securite*100:.0f}%). "
        f"Gaspillage estimé: {g_est:.1f} kg."
    )
    return BusinessResult(
        nb_repas_predits=int(max(0, round(nb_repas_predits))),
        quantite_recommandee_kg=float(q_rec),
        gaspillage_estime_kg=float(g_est),
        alertes=alertes,
        kpis=kpis,
        message=msg,
    )
