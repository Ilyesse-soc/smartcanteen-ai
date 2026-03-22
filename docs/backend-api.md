# API Backend FastAPI

## Base API

- Prefixe: /api/v1
- Healthcheck: GET /health
- Endpoint prediction: POST /api/v1/predictions

## Contrat d'entree

Le payload est valide via Pydantic.

Contraintes principales:

- jour_semaine: enum des jours.
- mois: 1..12.
- nb_inscrits et nb_absents_prevus: >= 0.
- menu_type: enum des menus supportes.
- stock_disponible_kg et quantite_produite_kg: >= 0.
- portion_moyenne_kg: > 0.
- champs inconnus interdits.

## Contrat de sortie

Reponse JSON:

- repas_prevus (int)
- quantite_recommandee (float)
- gaspillage_estime (float)
- alertes (list[str])
- message (str)

## Fonctionnement de la prediction

1. Le service charge l'artefact ML (cache en memoire).
2. Le service construit l'entree metier depuis la requete.
3. L'inference est executee via src.predict.predict_from_dict.
4. Le resultat est persiste en base (table predictions).
5. Le resultat est renvoye au client.

## Gestion des erreurs

- Erreur de validation: HTTP 422 (Pydantic).
- Modele indisponible: HTTP 503.
- Erreurs internes non gerees: HTTP 500.

## Exemple de requete

```bash
curl -X POST localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "jour_semaine": "Mardi",
    "mois": 3,
    "nb_inscrits": 420,
    "nb_absents_prevus": 35,
    "menu_type": "standard",
    "stock_disponible_kg": 250.0,
    "quantite_produite_kg": 240.0,
    "portion_moyenne_kg": 0.52
  }'
```
