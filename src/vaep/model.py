"""Fitting, saving and loading the VAEP models (one ``SoftLabelVAEP`` per split).

``SoftLabelVAEP`` is a ``socceraction.vaep.VAEP`` whose two probability models
(P(scores), P(concedes) within the next 10 actions) are lightgbm regressors with the
``cross_entropy`` objective, so they accept the xG-based soft labels in [0, 1] as well
as the binary outcome labels. socceraction's ``VAEP.fit`` builds ``LGBMClassifier``s,
which reject non-binary targets, hence this thin fit/predict path; the game-state
features (``compute_features``) and the value formula (``socceraction.vaep.formula``)
are socceraction's, unchanged, so ``rate(game, game_actions)`` keeps its contract.

How a model is fitted: one regressor per label, early stopping on the validation games
-- a seeded 15 % of the split's *training games* (whole games, never individual rows,
because consecutive game states share their labels); the remaining training games fit
the trees. Scored games are never seen during fitting or round selection.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import brier_score_loss, roc_auc_score
from socceraction import spadl
from socceraction.vaep import VAEP
from socceraction.vaep import features as fs
from socceraction.vaep import formula

from src.contract import SEED
from src.vaep.config import (
    DEFAULT_LABEL_MODE,
    EARLY_STOPPING_ROUNDS,
    LABEL_MODES,
    LABELS,
    LEARNER_OBJECTIVE,
    NB_PREV_ACTIONS,
    TREE_PARAMS,
    VALIDATION_SHARE,
)

log = logging.getLogger(__name__)


class SoftLabelVAEP(VAEP):
    """VAEP with lightgbm ``cross_entropy`` regressors, fitted on soft or binary labels."""

    def __init__(self, xfns=None, nb_prev_actions: int = NB_PREV_ACTIONS, label_mode: str = DEFAULT_LABEL_MODE) -> None:
        super().__init__(xfns, nb_prev_actions)
        if label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}")
        self.label_mode = label_mode
        self.models: dict[str, lightgbm.LGBMRegressor] = {}

    # -- fitting ----------------------------------------------------------------------
    def feature_names(self) -> list[str]:
        return list(fs.feature_column_names(self.xfns, self.nb_prev_actions))

    def fit_label(
        self,
        label: str,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tree_params: dict = TREE_PARAMS,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    ) -> dict:
        """Fit the model of one label with early stopping on ``(X_val, y_val)``; returns the fit record."""
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        cols = self.feature_names()
        curve: dict = {}
        t0 = time.perf_counter()
        model = lightgbm.LGBMRegressor(objective=LEARNER_OBJECTIVE, **tree_params)
        model.fit(
            X_fit[cols],
            y_fit.to_numpy(dtype="float64"),
            eval_set=[(X_val[cols], y_val.to_numpy(dtype="float64"))],
            eval_metric=LEARNER_OBJECTIVE,
            callbacks=[lightgbm.early_stopping(early_stopping_rounds, verbose=False), lightgbm.record_evaluation(curve)],
        )
        losses = curve["valid_0"][LEARNER_OBJECTIVE]
        self.models[label] = model
        record = {
            "best_iteration": int(np.argmin(losses)) + 1,
            "rounds_trained": len(losses),
            "hit_round_cap": len(losses) >= tree_params["n_estimators"],
            "val_logloss": float(np.min(losses)),
            "fit_seconds": round(time.perf_counter() - t0, 1),
        }
        log.info(
            "fitted %s (%s labels): best iteration %d of %d trained, validation cross-entropy %.5f (%.0fs)",
            label, self.label_mode, record["best_iteration"], record["rounds_trained"], record["val_logloss"], record["fit_seconds"],
        )
        return record

    # -- prediction -------------------------------------------------------------------
    def predict_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """P(scores) and P(concedes) for game states ``X`` (any frame holding the feature columns)."""
        if len(self.models) != len(LABELS):
            raise NotFittedError("both label models must be fitted first")
        cols = self.feature_names()
        missing = set(cols) - set(X.columns)
        if missing:
            raise ValueError(f"{len(missing)} feature columns are missing from the game states")
        features = X[cols]
        return pd.DataFrame({label: self.models[label].predict(features) for label in LABELS}, index=X.index)

    def rate(self, game: pd.Series, game_actions: pd.DataFrame, game_states: pd.DataFrame | None = None) -> pd.DataFrame:
        """Offensive, defensive and VAEP value of each action (socceraction's formula on our probabilities)."""
        actions_with_names = spadl.add_names(game_actions)
        if game_states is None:
            game_states = self.compute_features(game, game_actions)
        probabilities = self.predict_probabilities(game_states)
        return formula.value(actions_with_names, probabilities["scores"], probabilities["concedes"])

    def score(self, X: pd.DataFrame, y: pd.DataFrame, y_soft: pd.DataFrame | None = None) -> dict:
        """Brier and AUROC against the binary labels ``y``; plus Brier against ``y_soft`` when given."""
        probabilities = self.predict_probabilities(X)
        out: dict = {}
        for label in LABELS:
            p = probabilities[label].to_numpy()
            truth = y[label].to_numpy(dtype="float64")
            out[label] = {"brier": float(brier_score_loss(truth, p)), "auroc": float(roc_auc_score(truth, p))}
            if y_soft is not None:
                out[label]["brier_soft"] = float(np.mean((y_soft[label].to_numpy(dtype="float64") - p) ** 2))
        return out


def new_model(label_mode: str = DEFAULT_LABEL_MODE) -> SoftLabelVAEP:
    """An unfitted model with the project settings (default features, 3-action game states)."""
    return SoftLabelVAEP(nb_prev_actions=NB_PREV_ACTIONS, label_mode=label_mode)


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
    label_mode: str = DEFAULT_LABEL_MODE,
) -> tuple[SoftLabelVAEP, dict]:
    """Fit both label models. ``y_*`` hold the ``scores`` / ``concedes`` targets (soft or binary)."""
    model = new_model(label_mode)
    info = {label: model.fit_label(label, X_fit, y_fit[label], X_val, y_val[label]) for label in LABELS}
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
