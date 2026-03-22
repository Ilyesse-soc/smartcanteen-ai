# SmartCanteen-AI

Documentation principale du projet SmartCanteen-AI.

## Objectif

Predire la frequentation d'une cantine, recommander la quantite de production et estimer le gaspillage alimentaire.

## Documentation par responsabilite

- Vue d'ensemble produit et perimetre: [docs/overview.md](docs/overview.md)
- Installation et execution locale/Docker: [docs/setup-and-run.md](docs/setup-and-run.md)
- Pipeline data science et ML: [docs/ml-pipeline.md](docs/ml-pipeline.md)
- API FastAPI (contrats, endpoint, erreurs): [docs/backend-api.md](docs/backend-api.md)
- Architecture cible et roadmap: [docs/architecture-roadmap.md](docs/architecture-roadmap.md)

## Demarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --all
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
