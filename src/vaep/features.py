"""Game-state features and labels for the VAEP models, computed per game and cached.

socceraction describes every action by the action itself and its ``NB_PREV_ACTIONS - 1``
predecessors (type, result, body part, location, movement, timing, score line, ...),
with the pitch flipped so that the acting team always attacks left to right -- which is
why the home team id of the match is needed. Labels: did the acting team score
(``scores``) or concede (``concedes``) within the next 10 actions. Both are pure
functions of one game's actions, so the cache is keyed by ``(game_id, action_id)``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from socceraction.vaep import VAEP
from socceraction.vaep import features as fs

from src.vaep.config import FEATURES_PATH, LABELS, LABELS_PATH

log = logging.getLogger(__name__)

KEY_COLUMNS = ["game_id", "action_id"]
GAME_COLUMNS = ["game_id", "match_week", "match_date", "home_team_id", "away_team_id", "home_team", "away_team"]


def games_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """socceraction's ``games`` table: one row per match with ``game_id`` (= match_id), indexed by game id."""
    games = matches.rename(columns={"match_id": "game_id"})
    missing = [c for c in GAME_COLUMNS if c not in games.columns]
    if missing:
        raise ValueError(f"matches is missing columns {missing}")
    games = games[GAME_COLUMNS].copy()
    games.index = pd.Index(games["game_id"].to_numpy(dtype="int64"))
    return games


def game_row(match_id: int, home_team_id: int) -> pd.Series:
    """The minimal ``game`` Series socceraction needs (it only reads ``home_team_id``)."""
    return pd.Series({"game_id": int(match_id), "home_team_id": int(home_team_id)})


def game_actions(actions: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """One match's SPADL actions in action order with a fresh RangeIndex.

    socceraction aligns its per-action Series on the index, so the actions of a game
    must carry a RangeIndex 0..n-1. Raises ``ValueError`` when the match is absent.
    """
    ga = actions.loc[actions["game_id"] == int(match_id)]
    if ga.empty:
        raise ValueError(f"no actions for match {match_id}")
    return ga.sort_values("action_id", kind="mergesort").reset_index(drop=True)


def feature_columns(model: VAEP) -> list[str]:
    """The feature names the model's transformers generate, in socceraction's order."""
    return list(fs.feature_column_names(model.xfns, model.nb_prev_actions))


def compute_game_features(model: VAEP, game: pd.Series, ga: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and labels of one game's actions, each prefixed with the key columns."""
    X = model.compute_features(game, ga)
    y = model.compute_labels(game, ga)
    for frame in (X, y):
        frame.insert(0, "action_id", ga["action_id"].to_numpy(dtype="int64"))
        frame.insert(0, "game_id", np.int64(game["game_id"]))
    return X, y


def build_feature_cache(
    model: VAEP,
    games: pd.DataFrame,
    actions: pd.DataFrame,
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
    log_every: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute features and labels game by game, concatenate once and cache both as parquet."""
    X_parts, y_parts = [], []
    t0 = time.perf_counter()
    groups = actions.groupby("game_id", sort=True)
    for i, (gid, ga) in enumerate(groups, 1):
        gid = int(gid)
        X, y = compute_game_features(model, games.loc[gid], game_actions(ga, gid))
        X_parts.append(X)
        y_parts.append(y)
        if i % log_every == 0 or i == groups.ngroups:
            log.info("features: %d/%d games (%.0fs)", i, groups.ngroups, time.perf_counter() - t0)
    X_all = pd.concat(X_parts, ignore_index=True)
    y_all = pd.concat(y_parts, ignore_index=True)
    del X_parts, y_parts
    features_path.parent.mkdir(parents=True, exist_ok=True)
    X_all.to_parquet(features_path, index=False)
    y_all.to_parquet(labels_path, index=False)
    log.info("cached %d game states x %d features in %s", len(X_all), X_all.shape[1] - len(KEY_COLUMNS), features_path)
    return X_all, y_all


def cache_is_valid(
    model: VAEP, actions: pd.DataFrame, features_path: Path = FEATURES_PATH, labels_path: Path = LABELS_PATH
) -> bool:
    """True when both cache files exist, have the expected columns and cover exactly these actions."""
    if not (features_path.exists() and labels_path.exists()):
        return False
    if pq.ParquetFile(features_path).schema_arrow.names != KEY_COLUMNS + feature_columns(model):
        return False
    if pq.ParquetFile(labels_path).schema_arrow.names != KEY_COLUMNS + list(LABELS):
        return False
    expected = actions[KEY_COLUMNS].sort_values(KEY_COLUMNS).to_numpy(dtype="int64")
    for path in (features_path, labels_path):
        keys = pd.read_parquet(path, columns=KEY_COLUMNS).to_numpy(dtype="int64")
        if keys.shape != expected.shape or not np.array_equal(keys, expected):
            return False
    return True


def load_or_build_features(
    model: VAEP,
    games: pd.DataFrame,
    actions: pd.DataFrame,
    force: bool = False,
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cached features and labels, rebuilding them when stale or ``force``."""
    if not force and cache_is_valid(model, actions, features_path, labels_path):
        t0 = time.perf_counter()
        X = pd.read_parquet(features_path)
        y = pd.read_parquet(labels_path)
        log.info("loaded cached features for %d game states from %s (%.0fs)", len(X), features_path, time.perf_counter() - t0)
        return X, y
    log.info("computing features and labels for %d games", actions["game_id"].nunique())
    return build_feature_cache(model, games, actions, features_path, labels_path)
