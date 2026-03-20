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

## Plan d’attaque stratégique — intégration FastAPI + Next.js (sans code)

Ce plan est conçu pour passer proprement du prototype Data Science actuel vers une application web scalable, en minimisant les risques de régression et de data leakage.

### Étape 1 — Cadrage des contrats et des responsabilités

Objectif : figer les frontières entre entraînement, inférence, API et UI avant d’implémenter.

- Définir les cas d’usage prioritaires (prédiction unitaire, batch futur, simulation).
- Lister les champs **obligatoires/optionnels** côté interface.
- Décider les SLO techniques (latence cible, disponibilité, volumétrie).
- Valider les règles métier non négociables (bornes sur `repas_predits`, marges sécurité, alertes).

Critères de sortie :
- Liste des endpoints cible, schémas d’entrée/sortie attendus, et règles de validation métier validées.

---

### Étape 2 — Pilier 1 : refactorisation du pipeline ML

Objectif : séparer strictement entraînement et inférence, tout en garantissant un preprocessing identique.

1. **Isoler les modules**
   - `training`: préparation dataset, split temporel, entraînement, comparaison, sérialisation.
   - `inference`: chargement artefact, validation des entrées, transformation, prédiction, post-traitement métier.

2. **Stabiliser l’artefact de modèle**
   - Conserver dans un artefact unique : modèle, préprocesseur, spec de features, colonnes exclues (anti-fuite), métadonnées (version, date, métriques, signature des features).
   - Versionner l’artefact pour garantir compatibilité descendante API.

3. **Garantir l’identité preprocessing entraînement/inférence**
   - Interdire toute logique de transformation “manuelle” côté API qui diverge du pipeline entraîné.
   - Centraliser les conversions de type, features temporelles, lags/rolling et imputations dans une même chaîne réutilisable.
   - Définir une politique explicite pour les features dépendantes d’historique (source d’historique, fallback, stratégie cold-start).

4. **Sécurité ML et robustesse**
   - Contrôler explicitement les colonnes de fuite (features calculées à partir de la cible réelle).
   - Ajouter des garde-fous de domaine (bornes plausibles, valeurs négatives interdites, gestion des NaN/inconnus).

Critères de sortie :
- Artefact d’inférence autonome et reproductible.
- Règles anti-fuite documentées.
- Comportement défini pour données incomplètes et absence d’historique.

---

### Étape 3 — Pilier 2 : contrat d’API (Pydantic)

Objectif : normaliser les échanges Front/Back avec un contrat strict, explicite et versionné.

1. **Schémas d’entrée**
   - Définir un modèle `PredictionRequest` avec types stricts, bornes numériques et enums (ex. jours/menu).
   - Prévoir des defaults seulement quand ils sont métierement justifiés.
   - Séparer les champs “utilisateur” des champs techniques (traçabilité, version API, id requête).

2. **Schémas de sortie**
   - Définir `PredictionResponse` : prédiction, quantités recommandées, gaspillage estimé, alertes, KPI.
   - Inclure un bloc `meta` (version modèle, timestamp, durée inférence, source historique).

3. **Gestion des erreurs**
   - Contrat d’erreur uniforme (`code`, `message`, `details`) pour faciliter l’UX.
   - Distinguer erreurs de validation (422), indisponibilité modèle (503), et erreurs internes (500).

4. **Versionnement**
   - Préfixer les routes (`/api/v1`) et planifier les évolutions non rétrocompatibles (`v2`).

Critères de sortie :
- Contrat API lisible et stable pour l’équipe front.
- Messages d’erreur cohérents et exploitables UI.

---

### Étape 4 — Pilier 3 : architecture backend FastAPI

Objectif : backend modulaire, faible latence, non bloquant, prêt pour industrialisation.

1. **Structure de dossiers recommandée**
   - `api/` (routes + dépendances)
   - `schemas/` (Pydantic request/response)
   - `services/` (orchestration inférence, règles métier)
   - `ml/` (chargement artefact, adaptateurs pipeline)
   - `core/` (config, logging, settings)

2. **Cycle de vie modèle**
   - Charger l’artefact **une seule fois au démarrage** via lifecycle hooks.
   - Stocker une référence thread-safe en mémoire (singleton applicatif).
   - Exposer un endpoint de health/readiness qui vérifie la disponibilité du modèle.

3. **Non-blocage event-loop**
   - Si inférence CPU lourde, l’exécuter hors event-loop principal (threadpool/process pool) pour préserver la réactivité FastAPI.
   - Encadrer timeouts et limiter la concurrence pour éviter saturation CPU.

4. **Observabilité et exploitation**
   - Journaliser requêtes/réponses de manière anonymisée (pas de données sensibles).
   - Ajouter métriques applicatives (latence p95, taux d’erreur, débit).
   - Prévoir traces corrélées (request-id) entre front et back.

Critères de sortie :
- API prête production avec démarrage rapide, latence maîtrisée et monitoring minimal.

---

### Étape 5 — Pilier 4 : intégration frontend Next.js

Objectif : UX fiable et pédagogique autour de la prédiction.

1. **Organisation UI**
   - Composants de formulaire (input métier), composant résultat, composant alertes/erreurs.
   - Séparer logique d’appel API et logique d’affichage.

2. **Stratégie d’appel asynchrone**
   - Encapsuler les appels dans une couche dédiée (client API).
   - Gérer annulation/retry contrôlé pour éviter doubles soumissions.

3. **Gestion d’états UX**
   - `idle`: formulaire prêt
   - `loading`: bouton désactivé + indicateur de progression
   - `success`: résultats + KPI + alertes métier
   - `error`: message actionnable (validation ou indisponibilité service)

4. **Robustesse produit**
   - Validation côté front alignée avec Pydantic (sans dupliquer toutes les règles serveur).
   - Accessibilité et messages clairs pour les champs invalides.

Critères de sortie :
- Parcours utilisateur fluide de la saisie à l’interprétation des résultats.

---

### Étape 6 — Ordonnancement recommandé (roadmap exécutable)

1. Finaliser contrat API + exemples payloads.
2. Refactoriser pipeline ML (entraînement/inférence) et figer artefact.
3. Implémenter backend FastAPI (routes v1, lifecycle modèle, erreurs standardisées).
4. Connecter Next.js au backend (états loading/error/success).
5. Ajouter observabilité, tests d’intégration API, tests E2E du parcours UI.
6. Préparer déploiement (env vars, conteneurisation, stratégie de versionnement modèle).

### Risques clés à piloter dès le départ

- **Data leakage** : vérifier qu’aucune feature de vérité terrain n’entre en inférence.
- **Drift de schéma** : détecter les payloads front incompatibles avec le modèle courant.
- **Blocage CPU** : empêcher qu’une inférence lente dégrade toute l’API.
- **Incohérence métier** : harmoniser les règles de bornes/priorités entre modèle et UX.
