from __future__ import annotations

import argparse

import pandas as pd

from src.config import DEFAULT_N_DAYS, DEFAULT_SAFETY_MARGIN, PATHS
from src.data_generation import generate_synthetic_canteen_dataset
from src.evaluate import evaluate_and_report
from src.train import train_and_select_best


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartCanteen-AI: prototype end-to-end")

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Génère le dataset synthétique dans data/raw/cantine_data.csv",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Entraîne les modèles et sauvegarde le meilleur dans models/trained_model.joblib",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Évalue les modèles (nécessite l'entraînement) et exporte des graphiques dans models/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exécute tout: generate-data + train + evaluate",
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=DEFAULT_N_DAYS,
        help="Nombre de jours à générer (par cantine). Défaut: %(default)s",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=DEFAULT_SAFETY_MARGIN,
        help="Marge de sécurité métier (ex: 0.05 pour 5%%). Défaut: %(default)s",
    )
    parser.add_argument(
        "--use-autogluon",
        action="store_true",
        help="Active AutoGluon AutoML comme candidat supplémentaire pendant l'entraînement.",
    )
    parser.add_argument(
        "--autogluon-time-limit",
        type=int,
        default=120,
        help="Temps max (secondes) pour AutoGluon. Défaut: %(default)s",
    )
    parser.add_argument(
        "--autogluon-presets",
        type=str,
        default="medium_quality",
        help="Preset AutoGluon (best_quality, high_quality, good_quality, medium_quality...).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    do_generate = args.generate_data or args.all
    do_train = args.train or args.all
    do_eval = args.evaluate or args.all

    if not (do_generate or do_train or do_eval):
        raise SystemExit("Aucune action. Utilisez --all ou une option (voir --help).")

    if do_generate:
        PATHS.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_canteen_dataset(n_days=args.n_days)
        df.to_csv(PATHS.raw_data_path, index=False)
        print(f"✅ Dataset généré: {PATHS.raw_data_path}")

    if do_train:
        if not PATHS.raw_data_path.exists() or PATHS.raw_data_path.stat().st_size < 50:
            raise SystemExit(
                f"Dataset introuvable ou vide: {PATHS.raw_data_path}. Lancez d'abord --generate-data."
            )
        df = pd.read_csv(PATHS.raw_data_path)
        training_result = train_and_select_best(
            df,
            use_autogluon=args.use_autogluon,
            autogluon_time_limit=args.autogluon_time_limit,
            autogluon_presets=args.autogluon_presets,
        )
        print(f"✅ Meilleur modèle: {training_result.best_model_name}")
        print("\n=== Comparatif modèles (test) ===")
        print(training_result.metrics_table.to_string(index=False))
        print(f"✅ Modèle sauvegardé: {training_result.model_path}")

    if do_eval:
        if not PATHS.raw_data_path.exists() or PATHS.raw_data_path.stat().st_size < 50:
            raise SystemExit(
                f"Dataset introuvable ou vide: {PATHS.raw_data_path}. Lancez d'abord --generate-data."
            )
        df = pd.read_csv(PATHS.raw_data_path)
        evaluate_and_report(df, artifacts_dir=PATHS.models_dir)
        print(f"✅ Figures exportées dans: {PATHS.models_dir}")


if __name__ == "__main__":
    main()
