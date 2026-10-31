"""Game-state features and labels for the VAEP models, computed per game and cached.

socceraction describes every action by the action itself and its ``NB_PREV_ACTIONS - 1``
predecessors (type, result, body part, location, movement, timing, score line, ...),
with the pitch flipped so that the acting team always attacks left to right -- which is
why the home team id of the match is needed. Labels (``src.vaep.labels``): the binary
outcome labels (``scores``, ``concedes``) and the xG-based soft labels (``scores_xg``,
``concedes_xg``). Features and labels are pure functions of one game's actions, so both
caches are keyed by ``(game_id, action_id)``.
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

from src.vaep.config import FEATURES_PATH, LABEL_CACHE_COLUMNS, LABELS, LABELS_PATH, SOFT_LABELS
from src.vaep.labels import outcome_labels, xg_labels

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


def _with_keys(frame: pd.DataFrame, game_id: int, ga: pd.DataFrame) -> pd.DataFrame:
    frame.insert(0, "action_id", ga["action_id"].to_numpy(dtype="int64"))
    frame.insert(0, "game_id", np.int64(game_id))
    return frame


def compute_game_features(model: VAEP, game: pd.Series, ga: pd.DataFrame) -> pd.DataFrame:
    """Features of one game's actions, prefixed with the key columns."""
    return _with_keys(model.compute_features(game, ga), int(game["game_id"]), ga)


def compute_game_labels(game_id: int, ga: pd.DataFrame, xg_by_event: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Binary and xG labels of one game's actions (``LABEL_CACHE_COLUMNS``), plus the xG coverage counts."""
    binary = outcome_labels(ga)
    soft, coverage = xg_labels(ga, xg_by_event)
    y = pd.concat([binary[list(LABELS)], soft[list(LABELS)].rename(columns=SOFT_LABELS)], axis=1)
    return _with_keys(y, game_id, ga), coverage


def _keys_match(path: Path, actions: pd.DataFrame) -> bool:
    expected = actions[KEY_COLUMNS].sort_values(KEY_COLUMNS).to_numpy(dtype="int64")
    keys = pd.read_parquet(path, columns=KEY_COLUMNS).to_numpy(dtype="int64")
    return keys.shape == expected.shape and bool(np.array_equal(keys, expected))


def features_cache_is_valid(model: VAEP, actions: pd.DataFrame, features_path: Path = FEATURES_PATH) -> bool:
    """True when the feature cache exists, has the expected columns and covers exactly these actions."""
    if not features_path.exists():
        return False
    if pq.ParquetFile(features_path).schema_arrow.names != KEY_COLUMNS + feature_columns(model):
        return False
    return _keys_match(features_path, actions)


def labels_cache_is_valid(actions: pd.DataFrame, labels_path: Path = LABELS_PATH) -> bool:
    """True when the label cache exists with the binary and xG columns and covers exactly these actions."""
    if not labels_path.exists():
        return False
    if pq.ParquetFile(labels_path).schema_arrow.names != KEY_COLUMNS + LABEL_CACHE_COLUMNS:
        return False
    return _keys_match(labels_path, actions)


def build_feature_cache(
    model: VAEP, games: pd.DataFrame, actions: pd.DataFrame, features_path: Path = FEATURES_PATH, log_every: int = 20
) -> pd.DataFrame:
    """Compute the features game by game, concatenate once and cache them as parquet."""
    parts = []
    t0 = time.perf_counter()
    groups = actions.groupby("game_id", sort=True)
    for i, (gid, ga) in enumerate(groups, 1):
        gid = int(gid)
        parts.append(compute_game_features(model, games.loc[gid], game_actions(ga, gid)))
        if i % log_every == 0 or i == groups.ngroups:
            log.info("features: %d/%d games (%.0fs)", i, groups.ngroups, time.perf_counter() - t0)
    X = pd.concat(parts, ignore_index=True)
    del parts
    features_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(features_path, index=False)
    log.info("cached %d game states x %d features in %s", len(X), X.shape[1] - len(KEY_COLUMNS), features_path)
    return X


def build_labels_cache(
    actions: pd.DataFrame, xg_by_event: pd.Series, labels_path: Path = LABELS_PATH, log_every: int = 100
) -> tuple[pd.DataFrame, dict]:
    """Compute binary and xG labels game by game and cache them; returns the labels and the xG coverage."""
    parts, coverage = [], {"shots": 0, "shots_without_xg": 0, "own_goals": 0}
    t0 = time.perf_counter()
    groups = actions.groupby("game_id", sort=True)
    for i, (gid, ga) in enumerate(groups, 1):
        gid = int(gid)
        y, cov = compute_game_labels(gid, game_actions(ga, gid), xg_by_event)
        parts.append(y)
        for key in coverage:
            coverage[key] += cov[key]
        if i % log_every == 0 or i == groups.ngroups:
            log.info("labels: %d/%d games (%.0fs)", i, groups.ngroups, time.perf_counter() - t0)
    y_all = pd.concat(parts, ignore_index=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    y_all.to_parquet(labels_path, index=False)
    log.info("cached labels for %d game states in %s; xG coverage %s", len(y_all), labels_path, coverage)
    return y_all, coverage


def xg_coverage(actions: pd.DataFrame, xg_by_event: pd.Series) -> dict:
    """xG coverage counts of the whole action set (used when the label cache is reused)."""
    from src.vaep.labels import shot_goal_probability

    return shot_goal_probability(actions, xg_by_event)[1]


def load_or_build_features(
    model: VAEP, games: pd.DataFrame, actions: pd.DataFrame, force: bool = False, features_path: Path = FEATURES_PATH
) -> pd.DataFrame:
    """Return the cached features, rebuilding them when stale or ``force``."""
    if not force and features_cache_is_valid(model, actions, features_path):
        t0 = time.perf_counter()
        X = pd.read_parquet(features_path)
        log.info("loaded cached features for %d game states from %s (%.0fs)", len(X), features_path, time.perf_counter() - t0)
        return X
    log.info("computing features for %d games", actions["game_id"].nunique())
    return build_feature_cache(model, games, actions, features_path)


def load_or_build_labels(
    actions: pd.DataFrame, xg_by_event: pd.Series, force: bool = False, labels_path: Path = LABELS_PATH
) -> tuple[pd.DataFrame, dict]:
    """Return the cached binary + xG labels and the xG coverage, rebuilding when stale or ``force``."""
    if not force and labels_cache_is_valid(actions, labels_path):
        y = pd.read_parquet(labels_path)
        log.info("loaded cached labels for %d game states from %s", len(y), labels_path)
        return y, xg_coverage(actions, xg_by_event)
    log.info("computing labels for %d games", actions["game_id"].nunique())
    return build_labels_cache(actions, xg_by_event, labels_path)
