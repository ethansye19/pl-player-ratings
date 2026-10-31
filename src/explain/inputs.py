"""Structured input for a "why this rating?" explanation.

``explanation_input(match_id, player_id, method, k)`` returns a dict holding ONLY that
player's data for that match: identity, the match context (opponent, score, minutes),
the rating and raw score, and the player's top-k positive and top-k negative items:

- baseline: stat contributions ``count x weight = value`` from ``baseline_stats.parquet``
  and ``src.baseline.weights`` (only stats in the player's position-group profile);
- vaep: the player's individual actions and their ``vaep_value`` from
  ``action_values.parquet``.

Per-match function: it reads only rows of the requested match (isolation rule) and
raises ValueError if the match or the player is absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline.weights import FLOAT_STATS, LABELS, WEIGHTS
from src.contract import POSITION_GROUPS

PROCESSED = Path("data") / "processed"
METHODS = ("baseline", "vaep")
PERIOD_CLOCK_BASE = {1: 0, 2: 45, 3: 90, 4: 105}


def _read(processed_dir: Path, name: str, filters: list) -> pd.DataFrame:
    path = processed_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the rating steps first")
    return pd.read_parquet(path, filters=filters)


def _number(value: Any) -> float | int:
    value = float(value)
    return int(value) if value.is_integer() else value


def baseline_items(match_id: int, player_id: int, group: str, k: int, processed_dir: Path) -> tuple[list[dict], list[dict]]:
    stats = _read(processed_dir, "baseline_stats.parquet", [("match_id", "==", match_id), ("player_id", "==", player_id)])
    if stats.empty:
        raise ValueError(f"no baseline stats for player {player_id} in match {match_id}")
    row = stats.iloc[0]
    items = []
    for stat, weight in WEIGHTS[group].items():
        count = float(row[stat])
        value = count * float(weight.weight)
        if count == 0 or value == 0:
            continue
        items.append(
            {
                "stat": stat,
                "label": LABELS.get(stat, stat.replace("_", " ")),
                "count": count if stat in FLOAT_STATS else _number(count),
                "weight": float(weight.weight),
                "value": value,
            }
        )
    positive = sorted((i for i in items if i["value"] > 0), key=lambda i: -i["value"])[:k]
    negative = sorted((i for i in items if i["value"] < 0), key=lambda i: i["value"])[:k]
    return positive, negative


def vaep_items(match_id: int, player_id: int, k: int, processed_dir: Path) -> tuple[list[dict], list[dict]]:
    values = _read(processed_dir, "action_values.parquet", [("game_id", "==", match_id), ("player_id", "==", player_id)])
    items = []
    for a in values.itertuples(index=False):
        minute = PERIOD_CLOCK_BASE.get(int(a.period_id), 0) + int(a.time_seconds // 60)
        label = str(a.type_name).replace("_", " ")
        if isinstance(a.result_name, str) and a.result_name:
            label = f"{label} ({a.result_name})"
        items.append({"action_id": int(a.action_id), "label": label, "minute": minute, "value": round(float(a.vaep_value), 6)})
    positive = sorted((i for i in items if i["value"] > 0), key=lambda i: -i["value"])[:k]
    negative = sorted((i for i in items if i["value"] < 0), key=lambda i: i["value"])[:k]
    return positive, negative


def explanation_input(match_id: int, player_id: int, method: str, k: int = 5, processed_dir: Path | str = PROCESSED) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    match_id, player_id, processed_dir = int(match_id), int(player_id), Path(processed_dir)

    matches = _read(processed_dir, "matches.parquet", [("match_id", "==", match_id)])
    if matches.empty:
        raise ValueError(f"match {match_id} is not in matches.parquet")
    match = matches.iloc[0]
    lineups = _read(processed_dir, "lineups.parquet", [("match_id", "==", match_id), ("player_id", "==", player_id)])
    if lineups.empty:
        raise ValueError(f"player {player_id} did not appear in match {match_id}")
    player = lineups.iloc[0]
    group = str(player["position_group"])
    if group not in POSITION_GROUPS:
        raise ValueError(f"unknown position group {group!r}")

    is_home = int(player["team_id"]) == int(match["home_team_id"])
    opponent = str(match["away_team"] if is_home else match["home_team"])

    raw = _read(processed_dir, f"raw_{method}.parquet", [("match_id", "==", match_id), ("player_id", "==", player_id)])
    rated = _read(processed_dir, f"ratings_{method}.parquet", [("match_id", "==", match_id), ("player_id", "==", player_id)])
    raw_score = float(raw["raw_score"].iloc[0]) if not raw.empty else None
    rating = float(rated["rating"].iloc[0]) if not rated.empty else None

    if method == "baseline":
        positive, negative = baseline_items(match_id, player_id, group, k, processed_dir)
    else:
        positive, negative = vaep_items(match_id, player_id, k, processed_dir)

    return {
        "method": method,
        "match_id": match_id,
        "player_id": player_id,
        "player_name": str(player["player_name"]),
        "team": str(player["team"]),
        "opponent": opponent,
        "home_team": str(match["home_team"]),
        "away_team": str(match["away_team"]),
        "home_score": int(match["home_score"]),
        "away_score": int(match["away_score"]),
        "position_group": group,
        "position": str(player["position"]),
        "is_starter": bool(player["is_starter"]),
        "minutes": float(player["minutes"]),
        "raw_score": None if raw_score is None else round(raw_score, 6),
        "rating": rating,
        "top_positive": positive,
        "top_negative": negative,
    }
