"""Per-match VAEP rating: the pure function behind ``raw_vaep.parquet``.

``rate_match`` is a per-match rating function in the sense of CLAUDE.md's isolation
rule: it selects the requested match from whatever actions and lineups it is given,
raises ``ValueError`` when the match is absent, and reads nothing else except the
frozen model it is handed and two things about that match only: its row of
``matches.parquet`` unless ``home_team_id`` is passed (socceraction flips the pitch for the
away team, so it needs to know who played at home), and its shots in ``events.parquet``
unless ``shot_xg`` is passed (their StatsBomb xG feeds the option-3 value rules in
``src.vaep.value``).
``shot_xg`` may be a season-wide lookup: only this match's event ids are looked up in it.
``train.py`` and ``score.py`` build the cached season output through this same function, so
rating a match on its own reproduces the cache exactly.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from socceraction.vaep import VAEP

from src.contract import CONTRACT_COLUMNS, validate_contract
from src.vaep.config import HOLDOUT_MODEL_PATH, MATCHES_PATH, SPLITS_PATH, fold_model_path
from src.vaep.features import game_actions, game_row
from src.vaep.labels import shot_xg_lookup
from src.vaep.model import load_model
from src.vaep.splits import fold_for_match
from src.vaep.value import SHOT_TYPES

log = logging.getLogger(__name__)

ACTION_KEY_COLUMNS = ["game_id", "action_id", "player_id", "team_id", "period_id", "time_seconds", "type_name", "result_name"]
VALUE_COLUMNS = ["offensive_value", "defensive_value", "vaep_value"]


def home_team_id_for(match_id: int) -> int:
    """The home team of one match, read from that match's row of matches.parquet only."""
    row = pd.read_parquet(MATCHES_PATH, columns=["match_id", "home_team_id"], filters=[("match_id", "==", int(match_id))])
    if row.empty:
        raise ValueError(f"match {match_id} is not in {MATCHES_PATH}")
    return int(row["home_team_id"].iloc[0])


def with_shot_xg(ga: pd.DataFrame, match_id: int, shot_xg: pd.Series | None = None) -> pd.DataFrame:
    """The game's actions plus a ``shot_xg`` column: StatsBomb xG of each shot action, NaN elsewhere.

    ``shot_xg`` maps event id -> xG (any superset of the match's shots); when it is None the match's
    own shots are read from ``events.parquet``. Actions without ``original_event_id`` (synthetic
    frames) get no column, and the value rules then run without xG.
    """
    if "original_event_id" not in ga.columns:
        return ga
    lookup = shot_xg if shot_xg is not None else shot_xg_lookup(match_id=match_id)
    is_shot = ga["type_name"].isin(SHOT_TYPES).to_numpy()
    xg = ga["original_event_id"].map(lookup).astype("float64").to_numpy()
    return ga.assign(shot_xg=np.where(is_shot, xg, np.nan))


def action_values_for_match(
    match_id: int, actions: pd.DataFrame, model: VAEP, home_team_id: int | None = None, shot_xg: pd.Series | None = None
) -> pd.DataFrame:
    """Offensive, defensive and VAEP value of every action of ``match_id``.

    Returns one row per action (in action order) with the action's keys and the three
    values; raises ``ValueError`` when the match has no actions in ``actions``.
    """
    match_id = int(match_id)
    ga = game_actions(actions, match_id)
    if ga["player_id"].isna().any():
        raise ValueError(f"match {match_id} has actions without a player_id")
    if home_team_id is None:
        home_team_id = home_team_id_for(match_id)
    values = model.rate(game_row(match_id, home_team_id), with_shot_xg(ga, match_id, shot_xg))
    out = ga[ACTION_KEY_COLUMNS].copy()
    out["player_id"] = out["player_id"].astype("int64")  # SPADL stores it as a (nullable) float
    for col in VALUE_COLUMNS:
        out[col] = values[col].to_numpy(dtype="float64")
    return out


def rate_match(
    match_id: int,
    actions: pd.DataFrame,
    lineups: pd.DataFrame,
    model: VAEP,
    home_team_id: int | None = None,
    shot_xg: pd.Series | None = None,
) -> pd.DataFrame:
    """Contract-format raw scores for one match: ``raw_score`` = sum of the player's VAEP values.

    Every player in the match's lineups gets a row; players without actions get 0.0.
    Rows are sorted by ``match_id, team, player_id``.
    """
    match_id = int(match_id)
    lineup = lineups.loc[lineups["match_id"] == match_id]
    if lineup.empty:
        raise ValueError(f"match {match_id} is not in the given lineups")
    values = action_values_for_match(match_id, actions, model, home_team_id=home_team_id, shot_xg=shot_xg)
    per_player = values.groupby("player_id")["vaep_value"].sum()
    unknown = set(per_player.index) - set(lineup["player_id"].astype("int64"))
    if unknown:
        log.warning("match %d: %d players with actions are missing from the lineups; their values are dropped", match_id, len(unknown))
    out = lineup[["match_id", "player_id", "player_name", "team", "position_group", "minutes"]].copy()
    out["match_id"] = out["match_id"].astype("int64")
    out["player_id"] = out["player_id"].astype("int64")
    out["minutes"] = out["minutes"].astype("float64")
    out["raw_score"] = out["player_id"].map(per_player).fillna(0.0).astype("float64")
    out = out.sort_values(["match_id", "team", "player_id"], kind="mergesort").reset_index(drop=True)
    return validate_contract(out[CONTRACT_COLUMNS])


@lru_cache(maxsize=1)
def load_splits() -> dict:
    """The recorded splits (``vaep_splits.json``)."""
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"{SPLITS_PATH} is missing. Run `python -m src.vaep.train` first.")
    return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _fold_model(fold_id: int) -> VAEP:
    return load_model(fold_model_path(fold_id))


def load_model_for_match(match_id: int) -> VAEP:
    """The cross-fitting fold model whose training set excludes ``match_id``."""
    fold = fold_for_match(load_splits(), int(match_id))
    return _fold_model(int(fold["fold_id"]))


def load_holdout_model() -> VAEP:
    """The model trained on matchweeks 1-19 only."""
    return load_model(HOLDOUT_MODEL_PATH)
