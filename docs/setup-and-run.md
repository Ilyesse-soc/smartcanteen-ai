# Setup et Execution

## Prerequis

- Python 3.11+
- Docker et Docker Compose (optionnel,
  pour execution conteneurisee)

## Installation locale

Depuis le dossier smartcanteen-ai:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sur Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run complet data/ML

```bash
python main.py --all
```

Ce run:

- genere data/raw/cantine_data.csv,
- entraine plusieurs modeles,
- compare leurs performances,
- sauvegarde le meilleur artefact dans models/trained_model.joblib,
- exporte des figures dans models/.

### Option AutoGluon

```bash
python main.py --all \
	--use-autogluon \
	--autogluon-time-limit 300 \
	--autogluon-presets medium_quality
```

## Lancer Streamlit

```bash
streamlit run app/streamlit_app.py
```

Si models/trained_model.joblib est absent,
il faut executer python main.py --all.

## Lancer FastAPI

### Local

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Healthcheck:

```bash
curl http://localhost:8000/health
```

### Docker Compose

```bash
docker-compose up --build
```

API disponible sur [http://localhost:8000](http://localhost:8000).
