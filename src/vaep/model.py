"""Fitting, saving and loading the VAEP models (one ``socceraction.vaep.VAEP`` per split).

How a model is fitted
---------------------
``VAEP.fit`` trains one lightgbm classifier per label (``scores``, ``concedes``). We
call it once per label so that each classifier gets its own ``eval_set`` for early
stopping (the two calls accumulate in the same VAEP object). The early-stopping set is
a seeded 15 % of the split's *training games* (whole games, never individual rows,
because consecutive game states share their labels); the remaining training games fit
the trees. Scored games are never seen during fitting or round selection.

Why not socceraction's own ``val_size``: it draws a row-level validation sample and
passes lightgbm the ``early_stopping_rounds`` fit argument that lightgbm 4 removed, so
on this environment that path raises. ``val_size=0`` plus explicit ``fit_params`` uses
only documented arguments of both libraries.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from socceraction.vaep import VAEP

from src.contract import SEED
from src.vaep.config import EARLY_STOPPING_ROUNDS, LABELS, NB_PREV_ACTIONS, TREE_PARAMS, VALIDATION_SHARE

log = logging.getLogger(__name__)


def new_model() -> VAEP:
    """An unfitted VAEP model with the project settings (default features, 3-action game states)."""
    return VAEP(nb_prev_actions=NB_PREV_ACTIONS)


def validation_games(train_ids: list[int], share: float = VALIDATION_SHARE, seed: int = SEED) -> list[int]:
    """A seeded sample of the training games used only for early stopping."""
    ids = sorted(int(m) for m in train_ids)
    n_val = min(len(ids) - 1, max(1, int(round(share * len(ids)))))
    chosen = np.random.default_rng(seed).choice(ids, size=n_val, replace=False)
    return sorted(int(m) for m in chosen)


def fit_model(
    X_fit: pd.DataFrame,
    y_fit: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    seed: int = SEED,
) -> tuple[VAEP, dict]:
    """Fit both VAEP classifiers with early stopping on ``(X_val, y_val)``.

    ``X_*`` hold exactly the feature columns, ``y_*`` the ``scores`` and ``concedes``
    labels. Returns the fitted model and, per label, the chosen number of rounds, the
    validation log loss and the fit time.
    """
    model = new_model()
    info: dict[str, dict] = {}
    for label in LABELS:
        curve: dict = {}
        fit_params = {
            "eval_set": [(X_val, y_val[label])],
            "eval_metric": "binary_logloss",
            "callbacks": [
                lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lightgbm.record_evaluation(curve),
            ],
        }
        t0 = time.perf_counter()
        np.random.seed(seed)  # VAEP.fit shuffles the rows with numpy's global RNG
        model.fit(
            X_fit,
            y_fit[[label]],
            learner="lightgbm",
            val_size=0.0,
            tree_params=dict(TREE_PARAMS),
            fit_params=fit_params,
        )
        losses = curve["valid_0"]["binary_logloss"]
        info[label] = {
            "best_iteration": int(np.argmin(losses)) + 1,
            "rounds_trained": len(losses),
            "hit_round_cap": len(losses) >= TREE_PARAMS["n_estimators"],
            "val_logloss": float(np.min(losses)),
            "fit_seconds": round(time.perf_counter() - t0, 1),
        }
        log.info(
            "fitted %s: best iteration %d of %d trained, validation logloss %.5f (%.0fs)",
            label,
            info[label]["best_iteration"],
            info[label]["rounds_trained"],
            info[label]["val_logloss"],
            info[label]["fit_seconds"],
        )
    return model, info


def save_model(model: VAEP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> VAEP:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing. Run `python -m src.vaep.train` first.")
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    if not isinstance(model, VAEP):
        raise TypeError(f"{path} does not contain a socceraction VAEP model")
    return model
