"""Paths and fixed settings of the VAEP method.

Everything stochastic is seeded with ``SEED`` from ``src.contract``. The lightgbm
settings below are the single place where the model configuration lives, so that
``train.py`` (fitting), ``rate.py`` (scoring) and ``report.py`` (documentation) agree.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.contract import SEED

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
REPORTS_DIR = ROOT / "reports"

# inputs (written by src.ingest.build)
MATCHES_PATH = PROCESSED_DIR / "matches.parquet"
LINEUPS_PATH = PROCESSED_DIR / "lineups.parquet"
ACTIONS_PATH = PROCESSED_DIR / "actions.parquet"
EVENTS_PATH = PROCESSED_DIR / "events.parquet"  # StatsBomb shot xG (`shot_statsbomb_xg`) for the xG labels
BEFORE_DIR = PROCESSED_DIR / "before"  # frozen pre-xG outputs kept by the orchestrator for before/after comparisons

# outputs (written by src.vaep.train)
FEATURES_PATH = PROCESSED_DIR / "vaep_features.parquet"  # cache: game_id, action_id + one column per feature
LABELS_PATH = PROCESSED_DIR / "vaep_labels.parquet"  # cache: game_id, action_id + LABEL_CACHE_COLUMNS
SPLITS_PATH = PROCESSED_DIR / "vaep_splits.json"
ACTION_VALUES_PATH = PROCESSED_DIR / "action_values.parquet"
RAW_VAEP_PATH = PROCESSED_DIR / "raw_vaep.parquet"
RAW_VAEP_HOLDOUT_PATH = PROCESSED_DIR / "raw_vaep_holdout.parquet"
HOLDOUT_MODEL_PATH = MODELS_DIR / "vaep_holdout.pkl"
METRICS_PATH = MODELS_DIR / "vaep_metrics.json"  # settings + model quality, consumed by report.py
REPORT_PATH = REPORTS_DIR / "vaep_splits.md"


def fold_model_path(fold_id: int) -> Path:
    """Where the model of CV fold ``fold_id`` is pickled."""
    return MODELS_DIR / f"vaep_fold{int(fold_id)}.pkl"


# model configuration
NB_PREV_ACTIONS = 3  # a game state = the action plus its 2 predecessors (socceraction's default)
LABELS = ("scores", "concedes")  # socceraction's VAEP labels: goal for / against within the next 10 actions
NR_LABEL_ACTIONS = 10  # label window: the action and the next 9 (socceraction's default)
LABEL_MODES = ("xg", "outcome")  # xg: soft labels from StatsBomb shot xG (default); outcome: socceraction's binary labels
DEFAULT_LABEL_MODE = "xg"
SOFT_LABELS = {"scores": "scores_xg", "concedes": "concedes_xg"}  # label cache columns of the xG variant
LABEL_CACHE_COLUMNS = ["scores", "concedes", "scores_xg", "concedes_xg"]
LEARNER_OBJECTIVE = "cross_entropy"  # lightgbm objective that accepts targets in [0, 1]
N_FOLDS = 5
VALIDATION_SHARE = 0.15  # share of each split's *training* games set aside for early stopping
EARLY_STOPPING_ROUNDS = 30
N_JOBS = max(1, (os.cpu_count() or 2) - 2)
TREE_PARAMS = {
    "n_estimators": 600,  # upper bound; early stopping on the validation games picks the round count
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "max_bin": 63,
    "random_state": SEED,
    "deterministic": True,  # with force_row_wise: the same model for any thread count
    "force_row_wise": True,
    "n_jobs": N_JOBS,
    "verbose": -1,
}

ACTION_VALUE_COLUMNS = [
    "game_id",
    "action_id",
    "player_id",
    "team_id",
    "period_id",
    "time_seconds",
    "type_name",
    "result_name",
    "offensive_value",
    "defensive_value",
    "vaep_value",
    "fold_id",
]
