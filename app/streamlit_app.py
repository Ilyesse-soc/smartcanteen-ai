from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Permet `import src.*` même si Streamlit est lancé depuis un autre CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SAFETY_MARGIN, PATHS
from src.predict import load_artifact, predict_from_dict


st.set_page_config(page_title="SmartCanteen — AI", layout="wide")

st.title("SmartCanteen — Prédiction & Recommandation")
st.caption("Prototype: prévision de repas, recommandation de production, estimation du gaspillage.")


def _load_history(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists() or csv_path.stat().st_size < 50:
        return None
    return pd.read_csv(csv_path)


model_ready = PATHS.model_path.exists()
data_ready = PATHS.raw_data_path.exists() and PATHS.raw_data_path.stat().st_size > 50

if not model_ready:
    st.warning(
        "Modèle non trouvé. Lancez d'abord: `python main.py --all` (génère les données, entraîne, sauvegarde le modèle)."
    )
    st.stop()

artifact = load_artifact(PATHS.model_path)
history_df = _load_history(PATHS.raw_data_path) if data_ready else None


with st.sidebar:
    st.header("Entrées")

    jour_semaine = st.selectbox(
        "Jour de la semaine",
        ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
        index=1,
    )
    mois = st.selectbox("Mois", list(range(1, 13)), index=2)
    nb_inscrits = st.number_input("Nb inscrits", min_value=0, max_value=5000, value=420, step=10)
    nb_absents_prevus = st.number_input(
        "Nb absents prévus", min_value=0, max_value=5000, value=35, step=5
    )
    menu_type = st.selectbox(
        "Type de menu",
        ["standard", "eco", "pizza", "pates", "poisson", "vegetarien", "gourmet", "fete"],
        index=0,
    )
    temperature = st.number_input(
        "Température (°C)", min_value=-10.0, max_value=45.0, value=12.0, step=0.5
    )
    pluie = st.selectbox("Pluie", [0, 1], index=0, format_func=lambda x: "Oui" if x == 1 else "Non")
    evenement_special = st.selectbox(
        "Événement spécial",
        [0, 1],
        index=0,
        format_func=lambda x: "Oui" if x == 1 else "Non",
    )
    stock_disponible_kg = st.number_input(
        "Stock disponible (kg)", min_value=0.0, max_value=8000.0, value=250.0, step=5.0
    )
    quantite_produite_kg = st.number_input(
        "Quantité produite (kg)", min_value=0.0, max_value=8000.0, value=240.0, step=5.0
    )
    portion_moyenne_kg = st.number_input(
        "Portion moyenne (kg)", min_value=0.2, max_value=1.2, value=0.52, step=0.01
    )

    safety_margin = st.slider(
        "Marge de sécurité",
        min_value=0.0,
        max_value=0.20,
        value=float(DEFAULT_SAFETY_MARGIN),
        step=0.01,
        help="Ex: 0.05 = +5% sur la quantité recommandée.",
    )

    run_btn = st.button("Prédire")


if run_btn:
    user_input = {
        "jour_semaine": jour_semaine,
        "mois": int(mois),
        "nb_inscrits": int(nb_inscrits),
        "nb_absents_prevus": int(nb_absents_prevus),
        "menu_type": menu_type,
        "temperature": float(temperature),
        "pluie": int(pluie),
        "evenement_special": int(evenement_special),
        "stock_disponible_kg": float(stock_disponible_kg),
        "quantite_produite_kg": float(quantite_produite_kg),
        "portion_moyenne_kg": float(portion_moyenne_kg),
    }

    result = predict_from_dict(artifact, user_input, history_df=history_df, safety_margin=float(safety_margin))

    c1, c2, c3 = st.columns(3)
    c1.metric("Repas prédits", f"{result['nb_repas_predits']}")
    c2.metric("Quantité recommandée", f"{result['quantite_recommandee_kg']:.1f} kg")
    c3.metric("Gaspillage estimé", f"{result['gaspillage_estime_kg']:.1f} kg")

    st.subheader("Message métier")
    st.info(result["message"])

    st.subheader("Alertes")
    for a in result["alertes"]:
        st.write(f"- {a}")

    st.subheader("KPI")
    kpi = result["kpis"]
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Taux de présence estimé", f"{kpi['taux_presence_estime']*100:.1f}%")
    kc2.metric("Surproduction (kg)", f"{kpi['surproduction_kg']:.1f}")
    kc3.metric("Gaspillage estimé (%)", f"{kpi['gaspillage_pct_estime']*100:.1f}%")

else:
    st.write("Renseignez les entrées à gauche puis cliquez sur **Prédire**.")
