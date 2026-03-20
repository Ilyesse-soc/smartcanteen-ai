# SmartCanteen-AI — Prédiction du gaspillage alimentaire (prototype)

Prototype SaaS (démonstration/concours) pour aider une cantine à **anticiper la fréquentation**, **recommander une quantité à produire**, et **estimer le gaspillage**.

Ce dépôt est volontairement **end-to-end** : il génère un dataset synthétique réaliste, entraîne plusieurs modèles, compare leurs performances, sauvegarde le meilleur modèle, puis expose une app Streamlit simple.

## Objectif métier

Pour un jour donné, le système vise à :

1. Prédire le **nombre de repas consommés** (`nb_repas_consommés`) — *target principale*
2. Recommander la **quantité à produire** (kg)
3. Estimer le **gaspillage** (kg)
4. Fournir des **KPI** simples et des alertes métier (surproduction, stock insuffisant, forte affluence…)

Important : ce projet ne promet pas 100% de précision (variabilité réelle, bruit, décisions humaines, données imparfaites).

## Architecture

Arborescence principale :

```
smartcanteen-ai/
  data/
    raw/            # dataset brut (CSV synthétique)
    processed/      # exports optionnels
  models/           # modèle entraîné + figures d'évaluation
  notebooks/        # exploration
  src/              # logique data/ML/feature/business
  app/              # app Streamlit
  main.py           # CLI: génération, entraînement, évaluation
  requirements.txt
  README.md
```

### Modules clés

- `src/data_generation.py` : génération d’un dataset synthétique réaliste (2 cantines, ~365 jours → ~730 lignes)
- `src/feature_engineering.py` : features temporelles, menu, lags (J-1, J-7) et moyenne glissante 7 jours (sans fuite)
- `src/preprocess.py` : preprocessing scikit-learn (types, imputation, encodage catégoriel)
- `src/train.py` : baseline + RandomForest + XGBoost (split temporel)
- `src/evaluate.py` : métriques + graphiques (réel vs prédit, erreur temps, top features)
- `src/predict.py` : prédiction + dérivés métier (quantité recommandée, gaspillage estimé)
- `src/business.py` : fonctions métier + alertes
- `app/streamlit_app.py` : interface de démonstration

## Installation

### Pré-requis

- Python 3.11+
- VS Code (optionnel mais recommandé)

### Installer les dépendances

Depuis le dossier `smartcanteen-ai/` :

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lancer un run complet (données → entraînement → évaluation → modèle)

```bash
python main.py --all
```

Cela va :

- générer `data/raw/cantine_data.csv`
- entraîner baseline / RandomForest / XGBoost
- évaluer sur un **split temporel**
- sauvegarder le meilleur modèle dans `models/trained_model.joblib`
- sauvegarder quelques figures dans `models/`

### Option AutoML avec AutoGluon

Pour intégrer AutoGluon dans le pipeline (sans casser le flux existant), activez l'option suivante :

```bash
python main.py --all --use-autogluon --autogluon-time-limit 300 --autogluon-presets medium_quality
```

- `--use-autogluon` : ajoute AutoGluon comme candidat en plus de baseline / RandomForest / XGBoost
- `--autogluon-time-limit` : budget temps (en secondes) pour la recherche AutoML
- `--autogluon-presets` : niveau de qualité/temps AutoGluon

Le meilleur modèle (y compris AutoGluon s'il gagne) est sauvegardé dans `models/trained_model.joblib` et reste utilisable par l'app Streamlit.

## Lancer l’app Streamlit

Après avoir entraîné le modèle :

```bash
streamlit run app/streamlit_app.py
```

Si `models/trained_model.joblib` n’existe pas, l’app vous indiquera de lancer `python main.py --all`.

## Exemple de prédiction (CLI / Python)

Exemple minimal en Python (après entraînement) :

```python
from src.predict import load_artifact, predict_from_dict

artifact = load_artifact()

user_input = {
  "jour_semaine": "Mardi",
  "mois": 3,
  "nb_inscrits": 420,
  "nb_absents_prevus": 35,
  "menu_type": "standard",
  "temperature": 12.0,
  "pluie": 1,
  "evenement_special": 0,
  "stock_disponible_kg": 250.0,
  "quantite_produite_kg": 240.0,
  "portion_moyenne_kg": 0.52,
}

result = predict_from_dict(artifact, user_input, safety_margin=0.05)
print(result)
```

## Métriques (rappel)

- **MAE** : erreur absolue moyenne (plus petit = mieux)
- **RMSE** : pénalise davantage les grosses erreurs (plus petit = mieux)
- **R²** : part de variance expliquée (plus grand = mieux)
- **MAPE** : erreur relative (%) — calculée en évitant les jours à 0 repas

## Limites du prototype

- Données synthétiques : la performance n’est pas une garantie sur données réelles.
- Variables manquantes possibles dans la vraie vie (retards, annulations, changements de menu…)
- Les lags nécessitent de l’historique : en démo, l’app utilise l’historique du CSV généré.

## Pistes d’amélioration

- Ajout de données réelles (capteurs, réservations, retours cuisine)
- Modèles de séries temporelles dédiés (LightGBM, CatBoost, Prophet… selon contraintes)
- Explicabilité (SHAP) et monitoring (drift, qualité des données)
- Déploiement (API FastAPI + stockage modèle/versioning)
