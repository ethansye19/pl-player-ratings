"""Per-match player stats for the baseline method, computed from ONE match's events.

This implements the stat definitions documented in ``src.baseline.weights`` (module
docstring). ``player_match_stats`` is a per-match rating function under the isolation
rule (CLAUDE.md): it selects the rows of the requested ``match_id`` from whatever it is
given, raises ``ValueError`` if that match is absent, and produces identical output
whether or not other matches' rows are present. The only inputs are that match's
events and that match's lineups; the team-context stats (goals conceded, xG faced, clean
sheet) are derived from the same match.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.baseline.weights import FLOAT_STATS, STATS

log = logging.getLogger(__name__)

# StatsBomb pitch: 120 x 80 yards, every team attacks left -> right (x increases).
DEFENSIVE_40_X = 48.0        # progressive passes / carries must start beyond the own defensive 40%
PROGRESSIVE_YARDS = 10.0     # ... and move the ball at least this far towards the opponent's goal
CLEAN_SHEET_MINUTES = 60.0   # FPL rule: a clean sheet needs 60+ minutes on the pitch

SET_PIECE_PASS_TYPES = ("Throw-in", "Goal Kick", "Kick Off", "Corner")
INCOMPLETE_PASS_OUTCOMES = ("Incomplete", "Out", "Unknown", "Pass Offside")
ON_TARGET_OUTCOMES = ("Goal", "Saved", "Saved to Post")
WON_OUTCOMES = ("Won", "Success In Play", "Success Out")      # duel (tackle) and interception outcomes
RED_CARD_VALUES = ("Red Card", "Second Yellow")
SAVE_TYPES = ("Shot Saved", "Shot Saved to Post", "Save", "Penalty Saved", "Penalty Saved to Post")
PENALTY_SAVE_TYPES = ("Penalty Saved", "Penalty Saved to Post")
CLAIM_OUTCOMES = {
    "Collected": ("Success", "Collected Twice"),
    "Punch": ("Punched out", "In Play Safe"),
    "Keeper Sweeper": ("Claim", "Clear"),
    "Smother": ("Success In Play", "Won"),
}
AERIAL_WON_FLAGS = ("clearance_aerial_won", "pass_aerial_won", "shot_aerial_won", "miscontrol_aerial_won")

# The only events.parquet columns this module reads (rate_season loads just these).
EVENT_COLUMNS: list[str] = [
    "match_id", "team_id", "player_id", "period", "timestamp", "type",
    "shot_outcome", "shot_type", "shot_statsbomb_xg",
    "pass_outcome", "pass_type", "pass_goal_assist", "pass_shot_assist",
    "location_x", "pass_end_x", "carry_end_x",
    "dribble_outcome", "duel_type", "duel_outcome", "interception_outcome",
    "ball_recovery_recovery_failure", *AERIAL_WON_FLAGS,
    "foul_committed_card", "bad_behaviour_card", "foul_committed_penalty", "foul_won_penalty",
    "goalkeeper_type", "goalkeeper_outcome",
]
LINEUP_COLUMNS: list[str] = ["match_id", "team_id", "player_id", "position_group", "minutes", "minute_on", "minute_off"]
KEY_COLUMNS: list[str] = ["match_id", "player_id", "position_group", "minutes"]


# ----------------------------------------------------------------------------- helpers
def select_match(df: pd.DataFrame, match_id: int, name: str = "data") -> pd.DataFrame:
    """Rows of ``df`` for ``match_id``; ``ValueError`` if that match is absent."""
    if "match_id" not in df.columns:
        raise ValueError(f"{name} has no 'match_id' column")
    out = df.loc[df["match_id"] == int(match_id)]
    if out.empty:
        raise ValueError(f"match {match_id} is absent from {name}")
    return out


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")


def _flag(series: pd.Series) -> pd.Series:
    """Boolean view of a StatsBomb flag column (missing -> False)."""
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(bool)


def absolute_minutes(events: pd.DataFrame) -> pd.Series:
    """Minutes since kick-off, stoppage time included, for every event of ONE match.

    Event timestamps restart at 00:00 every period; the real length of each period is the
    timestamp of its "Half End" event, exactly as the ingest step computed the lineups'
    ``minute_on`` / ``minute_off``.
    """
    seconds = pd.to_timedelta(events["timestamp"]).dt.total_seconds()
    half_end = events["type"].eq("Half End")
    if half_end.any():
        lengths = seconds[half_end].groupby(events.loc[half_end, "period"]).max()
    else:  # defensive fallback: the last event of each period
        lengths = seconds.groupby(events["period"]).max()
    lengths = lengths.sort_index()
    offsets = lengths.cumsum().shift(fill_value=0.0)
    return (events["period"].map(offsets).fillna(0.0) + seconds) / 60.0


# ------------------------------------------------------------------- player event stats
def _player_event_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Per-player counts of the event-based stats (index: player_id) for one match."""
    pe = events.loc[events["player_id"].notna()]
    player = pe["player_id"].astype("int64").to_numpy()

    typ = pe["type"]
    is_pass = typ.eq("Pass")
    is_shot = typ.eq("Shot")
    is_duel = typ.eq("Duel")
    is_gk = typ.eq("Goal Keeper")
    pass_completed = is_pass & pe["pass_outcome"].isna()
    penalty = is_shot & pe["shot_type"].eq("Penalty")
    goal = is_shot & pe["shot_outcome"].eq("Goal")
    foul_card = pe["foul_committed_card"]
    behaviour_card = pe["bad_behaviour_card"]
    gk_type = pe["goalkeeper_type"]
    gk_outcome = pe["goalkeeper_outcome"]
    claims = pd.Series(False, index=pe.index)
    for kind, outcomes in CLAIM_OUTCOMES.items():
        claims |= gk_type.eq(kind) & gk_outcome.isin(outcomes)
    aerial_won = pd.Series(False, index=pe.index)
    for column in AERIAL_WON_FLAGS:
        aerial_won |= _flag(pe[column])
    forward_pass = pe["pass_end_x"] - pe["location_x"]
    forward_carry = pe["carry_end_x"] - pe["location_x"]

    flags = {
        "goals": goal,
        "assists": is_pass & _flag(pe["pass_goal_assist"]),
        "shots_on_target": is_shot & pe["shot_outcome"].isin(ON_TARGET_OUTCOMES),
        "key_passes": is_pass & (_flag(pe["pass_shot_assist"]) | _flag(pe["pass_goal_assist"])),
        "dribbles_completed": typ.eq("Dribble") & pe["dribble_outcome"].eq("Complete"),
        "progressive_carries": typ.eq("Carry")
        & (pe["location_x"] >= DEFENSIVE_40_X)
        & (forward_carry >= PROGRESSIVE_YARDS),
        "penalties_won": typ.eq("Foul Won") & _flag(pe["foul_won_penalty"]),
        "penalties_missed": penalty & ~goal,
        "passes_completed": pass_completed,
        "passes_incomplete": is_pass & pe["pass_outcome"].isin(INCOMPLETE_PASS_OUTCOMES),
        "progressive_passes": pass_completed
        & ~pe["pass_type"].isin(SET_PIECE_PASS_TYPES)
        & (pe["location_x"] >= DEFENSIVE_40_X)
        & (forward_pass >= PROGRESSIVE_YARDS),
        "possession_lost": typ.isin(["Dispossessed", "Miscontrol"]),
        "tackles_won": is_duel & pe["duel_type"].eq("Tackle") & pe["duel_outcome"].isin(WON_OUTCOMES),
        "dribbled_past": typ.eq("Dribbled Past"),
        "interceptions": typ.eq("Interception") & pe["interception_outcome"].isin(WON_OUTCOMES),
        "clearances": typ.eq("Clearance"),
        "blocks": typ.eq("Block"),
        "ball_recoveries": typ.eq("Ball Recovery") & ~_flag(pe["ball_recovery_recovery_failure"]),
        "aerials_won": aerial_won,
        "aerials_lost": is_duel & pe["duel_type"].eq("Aerial Lost"),
        "pressures": typ.eq("Pressure"),
        "fouls_committed": typ.eq("Foul Committed"),
        "fouls_won": typ.eq("Foul Won"),
        "yellow_cards": foul_card.eq("Yellow Card") | behaviour_card.eq("Yellow Card"),
        "red_cards": foul_card.isin(RED_CARD_VALUES) | behaviour_card.isin(RED_CARD_VALUES),
        "errors": typ.eq("Error"),
        "own_goals": typ.eq("Own Goal Against"),
        "penalties_conceded": typ.eq("Foul Committed") & _flag(pe["foul_committed_penalty"]),
        "saves": is_gk & gk_type.isin(SAVE_TYPES),
        "penalties_saved": is_gk & gk_type.isin(PENALTY_SAVE_TYPES),
        "keeper_claims": is_gk & claims,
    }
    table = pd.DataFrame(
        {name: mask.to_numpy(dtype=bool).astype(np.int64) for name, mask in flags.items()}, index=pe.index
    )
    table["xg"] = pe["shot_statsbomb_xg"].where(is_shot & ~penalty, 0.0).fillna(0.0).to_numpy(dtype="float64")
    return table.groupby(player, sort=True).sum()


# ------------------------------------------------------------------------ team context
def _team_context(events: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """goals_conceded, xg_faced and clean_sheet per lineup row, counted while on the pitch."""
    minute = absolute_minutes(events)
    shots = events.loc[events["type"].eq("Shot"), ["team_id", "shot_outcome", "shot_statsbomb_xg"]]
    own_goals = events.loc[events["type"].eq("Own Goal Against"), ["team_id"]]

    on = lineups["minute_on"].fillna(0.0).to_numpy(dtype="float64")
    off = lineups["minute_off"].fillna(np.inf).to_numpy(dtype="float64")
    goals_conceded = np.zeros(len(lineups), dtype=np.int64)
    xg_faced = np.zeros(len(lineups), dtype=np.float64)

    for team_id in pd.unique(lineups["team_id"]):
        rows = (lineups["team_id"] == team_id).to_numpy()
        faced = shots[shots["team_id"] != team_id]
        shot_minute = minute.loc[faced.index].to_numpy(dtype="float64")
        shot_xg = faced["shot_statsbomb_xg"].fillna(0.0).to_numpy(dtype="float64")
        goal_minute = np.concatenate(
            [
                shot_minute[faced["shot_outcome"].eq("Goal").to_numpy()],
                minute.loc[own_goals.index[own_goals["team_id"] == team_id]].to_numpy(dtype="float64"),
            ]
        )
        on_pitch_goal = (on[rows, None] <= goal_minute[None, :]) & (goal_minute[None, :] <= off[rows, None])
        on_pitch_shot = (on[rows, None] <= shot_minute[None, :]) & (shot_minute[None, :] <= off[rows, None])
        goals_conceded[rows] = on_pitch_goal.sum(axis=1)
        xg_faced[rows] = (on_pitch_shot * shot_xg[None, :]).sum(axis=1)

    clean_sheet = ((goals_conceded == 0) & (lineups["minutes"].to_numpy(dtype="float64") >= CLEAN_SHEET_MINUTES)).astype(np.int64)
    return pd.DataFrame(
        {"goals_conceded": goals_conceded, "xg_faced": xg_faced, "clean_sheet": clean_sheet}, index=lineups.index
    )


# ---------------------------------------------------------------------------- public
def player_match_stats(match_id: int, events: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """One row per player in that match's lineups: ``match_id, player_id, position_group,
    minutes`` + one column per stat in ``STATS`` (zeros for players with no events).

    Rows are ordered by team then player_id. Raises ``ValueError`` when ``match_id`` is
    absent from ``events`` or ``lineups``.
    """
    match_id = int(match_id)
    _require_columns(events, EVENT_COLUMNS, "events")
    _require_columns(lineups, LINEUP_COLUMNS, "lineups")
    ev = select_match(events, match_id, "events")
    lu = select_match(lineups, match_id, "lineups")
    sort_by = ["team", "player_id"] if "team" in lu.columns else ["team_id", "player_id"]
    lu = lu.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    if lu["player_id"].duplicated().any():
        raise ValueError(f"match {match_id}: duplicate player_id rows in lineups")

    per_player = _player_event_stats(ev).reindex(lu["player_id"].astype("int64").to_numpy()).fillna(0.0)
    context = _team_context(ev, lu)

    out = pd.DataFrame(
        {
            "match_id": np.full(len(lu), match_id, dtype=np.int64),
            "player_id": lu["player_id"].to_numpy(dtype="int64"),
            "position_group": lu["position_group"].astype(str).to_numpy(),
            "minutes": lu["minutes"].to_numpy(dtype="float64"),
        }
    )
    for stat in STATS:
        if stat in context.columns:
            values = context[stat].to_numpy()
        else:
            values = per_player[stat].to_numpy()
        out[stat] = values.astype(np.float64 if stat in FLOAT_STATS else np.int64)
    return out
