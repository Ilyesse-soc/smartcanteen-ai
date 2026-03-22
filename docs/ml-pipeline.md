# Pipeline ML

## Flux principal

Le pipeline ML est pilote par main.py.

1. Generation des donnees synthetiques.
2. Feature engineering.
3. Entrainement de plusieurs modeles.
4. Evaluation sur split temporel.
5. Selection et sauvegarde du meilleur modele.

## Modules

- src/data_generation.py: generation du dataset synthetique.
- src/feature_engineering.py: features temporelles et derivees.
- src/preprocess.py: preprocessing scikit-learn.
- src/train.py: entrainement et selection du meilleur modele.
- src/evaluate.py: metriques et visualisations.
- src/predict.py: inference a partir de l'artefact sauvegarde.

## Artefact de modele

Le modele est sauvegarde dans:

- models/trained_model.joblib

Cet artefact est utilise par Streamlit et par l'API backend.

## Exemple d'inference Python

```python
from src.predict import load_artifact, predict_from_dict

artifact = load_artifact()

result = predict_from_dict(
    artifact,
    {
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
    },
    safety_margin=0.05,
)

print(result)
```

## Limites connues

- Donnees synthetiques: performance non garantie en production reelle.
- Variables operationnelles manquantes possibles selon le contexte terrain.
- Certaines features temporelles peuvent dependre de l'historique disponible.
