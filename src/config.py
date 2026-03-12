from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_NAME = "smartcanteen-ai"
RANDOM_SEED = 42

# Dataset: ~365 jours * 2 cantines => ~730 lignes
DEFAULT_N_DAYS = 365

# Marge de sécurité métier (5% par défaut)
DEFAULT_SAFETY_MARGIN = 0.05


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path
    data_dir: Path
    raw_data_path: Path
    processed_dir: Path
    models_dir: Path
    model_path: Path


def _resolve_root_dir() -> Path:
    # src/config.py -> smartcanteen-ai/
    return Path(__file__).resolve().parents[1]


ROOT_DIR = _resolve_root_dir()
PATHS = ProjectPaths(
    root_dir=ROOT_DIR,
    data_dir=ROOT_DIR / "data",
    raw_data_path=ROOT_DIR / "data" / "raw" / "cantine_data.csv",
    processed_dir=ROOT_DIR / "data" / "processed",
    models_dir=ROOT_DIR / "models",
    model_path=ROOT_DIR / "models" / "trained_model.joblib",
)
