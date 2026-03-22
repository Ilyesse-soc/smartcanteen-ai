# SmartCanteen-AI: Vue d'ensemble

Ce projet vise a predire la frequentation d'une cantine,
recommander la quantite de production et estimer le gaspillage alimentaire.

## Objectif metier

Pour un jour donne, le systeme permet de:

1. Predire le nombre de repas consommes.
2. Recommander la quantite a produire (kg).
3. Estimer le gaspillage (kg).
4. Produire des KPI et alertes metier.

## Perimetre

Le depot couvre un flux end-to-end:

- generation de donnees synthetiques,
- entrainement et comparaison de modeles,
- evaluation,
- sauvegarde d'un artefact de modele,
- inference via API FastAPI,
- interface de demonstration Streamlit.

## Arborescence principale

```text
smartcanteen-ai/
  app/         # Interface Streamlit
  backend/     # API FastAPI (routes, services, repository)
  data/        # Donnees brutes/traitees
  docs/        # Documentation fonctionnelle et technique
  models/      # Artefacts et figures d'evaluation
  notebooks/   # Exploration
  src/         # Modules data science / ML
```

## Liens utiles

- Setup et execution: [setup-and-run.md](setup-and-run.md)
- Pipeline ML: [ml-pipeline.md](ml-pipeline.md)
- API backend: [backend-api.md](backend-api.md)
- Architecture et roadmap: [architecture-roadmap.md](architecture-roadmap.md)
