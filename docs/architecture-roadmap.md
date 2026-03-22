# Architecture et Roadmap

## Architecture actuelle

- src/: logique data science et ML.
- backend/: API FastAPI (routes, services, persistence SQLAlchemy).
- app/: interface Streamlit de demonstration.
- models/: artefacts entraines et outputs d'evaluation.

## Principes de conception

- Separation entrainement/inference.
- Validation stricte des entrees API.
- Reutilisation du meme pipeline de prediction entre UI et API.
- Persistance des predictions pour tracabilite.

## Roadmap recommandee

1. Durcir le contrat API (metadonnees de modele, trace-id, timings).
2. Charger le modele explicitement au demarrage pour readiness fiable.
3. Standardiser les erreurs (code/message/details).
4. Ajouter observabilite (latence p95, erreurs, debit).
5. Ajouter tests d'integration API et tests E2E front/back.
6. Industrialiser le deploiement et la strategie de versionning modele.

## Risques a piloter

- Data leakage entre features d'entrainement et inference.
- Drift de schema entree API vs artefact courant.
- Saturation CPU en cas de charge inference.
- Incoherences metier entre API et front.
