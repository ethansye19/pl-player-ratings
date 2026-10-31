"""Web export: build the JSON payloads the match-ratings page (``web/``) loads.

``python -m src.pipeline.export_web [--out web/data]`` writes, for the 2015/16 season::

    <out>/index.json                  one entry per match (the page's match picker)
    <out>/2015-16/<match_id>.json     one payload per match (both lineups, both methods)

The payload schema is the "Web payload" section of CLAUDE.md. Two optional per-player
keys are added on top of it, ``minute_on`` and ``minute_off`` (clock minutes, from
``lineups.parquet``), so the page can show when a substitute came on. Consumers must treat
them as optional; ``validate_payload`` accepts payloads with or without them.

Rules enforced by :func:`validate_payload` (and therefore by :func:`export_season`, which
validates every payload before it writes anything):

* a player has a rating for a method iff he played ``MIN_MINUTES`` or more;
* ratings are within ``[RATING_MIN, RATING_MAX]`` with one decimal;
* exactly 11 starters per side, unique ``player_id``, ``position_group`` from the contract;
* a StatsBomb payload must credit StatsBomb.

Nothing is invented: a rated player who is missing from a method's ratings file yields a
``null`` rating, the payload then fails validation and the export stops with a clear error.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import numbers
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.contract import MIN_MINUTES, POSITION_GROUPS, RATING_DECIMALS, RATING_MAX, RATING_MIN

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SOURCE_STATSBOMB = "StatsBomb Open Data"
SEASON = "2015/16"
COMPETITION = "Premier League"
METHODS = ("baseline", "vaep")  # presentation order; a method is exported only if its ratings file exists
CREDITS = "Data: StatsBomb Open Data (https://github.com/statsbomb/open-data)"
SIDES = ("home", "away")
STARTERS_PER_SIDE = 11
RAW_SCORE_DECIMALS = 4

_TOP_KEYS = (
    "schema_version", "source", "season", "competition", "match_id", "match_date", "matchweek",
    "home", "away", "methods", "players", "credits",
)
_TEAM_KEYS = ("team_id", "team", "score", "formation")
_PLAYER_KEYS = (
    "player_id", "player_name", "team_id", "team", "side", "position_id", "position",
    "position_group", "is_starter", "jersey_number", "minutes", "ratings", "raw_scores",
)
_OPTIONAL_TIME_KEYS = ("minute_on", "minute_off")

_MATCH_COLUMNS = (
    "match_id", "match_date", "match_week", "home_team_id", "home_team", "away_team_id", "away_team",
    "home_score", "away_score", "home_formation", "away_formation",
)
_LINEUP_COLUMNS = (
    "match_id", "team_id", "team", "player_id", "player_name", "jersey_number", "is_starter",
    "position_id", "position", "position_group", "minutes",
)
_RATING_COLUMNS = ("match_id", "player_id", "rating", "raw_score")


# --------------------------------------------------------------------------- validation
def season_dir(season: str) -> str:
    """Folder name for a season label: ``"2015/16"`` -> ``"2015-16"``."""
    return str(season).replace("/", "-")


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def validate_payload(payload: Any) -> None:
    """Raise ``ValueError`` unless ``payload`` follows the web payload schema and rating rules."""
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    missing = [key for key in _TOP_KEYS if key not in payload]
    if missing:
        raise ValueError(f"payload is missing required keys: {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {payload['schema_version']!r} (expected {SCHEMA_VERSION})")

    source, credits = payload["source"], payload["credits"]
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    if not isinstance(credits, str) or not credits.strip():
        raise ValueError("credits must be a non-empty string")
    if source == SOURCE_STATSBOMB and "StatsBomb" not in credits:
        raise ValueError(f"credits must mention StatsBomb when source is {SOURCE_STATSBOMB!r}, got {credits!r}")

    for side in SIDES:
        team = payload[side]
        if not isinstance(team, dict):
            raise ValueError(f"'{side}' must be a dict")
        missing = [key for key in _TEAM_KEYS if key not in team]
        if missing:
            raise ValueError(f"'{side}' is missing required keys: {missing}")

    methods = payload["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list")
    if not all(isinstance(m, str) and m for m in methods) or len(set(methods)) != len(methods):
        raise ValueError(f"methods must be unique non-empty strings, got {methods!r}")

    players = payload["players"]
    if not isinstance(players, list) or not players:
        raise ValueError("players must be a non-empty list")

    seen: set = set()
    starters = {side: 0 for side in SIDES}
    for index, player in enumerate(players):
        if not isinstance(player, dict):
            raise ValueError(f"players[{index}] must be a dict")
        missing = [key for key in _PLAYER_KEYS if key not in player]
        if missing:
            raise ValueError(f"players[{index}] is missing required keys: {missing}")
        label = f"{player['player_name']!r} (player_id {player['player_id']!r})"

        if player["side"] not in SIDES:
            raise ValueError(f"{label}: side must be one of {SIDES}, got {player['side']!r}")
        if player["position_group"] not in POSITION_GROUPS:
            raise ValueError(f"{label}: position_group must be one of {POSITION_GROUPS}, got {player['position_group']!r}")
        if player["player_id"] in seen:
            raise ValueError(f"duplicate player_id {player['player_id']!r} ({label})")
        seen.add(player["player_id"])
        if not _is_bool(player["is_starter"]):
            raise ValueError(f"{label}: is_starter must be a bool, got {player['is_starter']!r}")
        if player["is_starter"]:
            starters[player["side"]] += 1

        minutes = player["minutes"]
        if not _is_number(minutes) or minutes < 0:
            raise ValueError(f"{label}: minutes must be a non-negative number, got {minutes!r}")
        for key in _OPTIONAL_TIME_KEYS:
            if key in player and player[key] is not None and (not _is_number(player[key]) or player[key] < 0):
                raise ValueError(f"{label}: {key} must be null or a non-negative number, got {player[key]!r}")

        ratings, raw_scores = player["ratings"], player["raw_scores"]
        if not isinstance(ratings, dict) or set(ratings) != set(methods):
            raise ValueError(f"{label}: ratings keys {sorted(ratings) if isinstance(ratings, dict) else ratings!r} "
                             f"must equal methods {sorted(methods)}")
        if not isinstance(raw_scores, dict) or set(raw_scores) != set(methods):
            raise ValueError(f"{label}: raw_scores keys must equal methods {sorted(methods)}")

        rated = minutes >= MIN_MINUTES
        for method in methods:
            rating = ratings[method]
            if rating is None:
                if rated:
                    raise ValueError(f"{label}: played {minutes} min (>= {MIN_MINUTES}) but has no {method} rating")
            else:
                if not rated:
                    raise ValueError(f"{label}: played only {minutes} min (< {MIN_MINUTES}) "
                                     f"but has a {method} rating of {rating!r}")
                if not _is_number(rating):
                    raise ValueError(f"{label}: {method} rating must be a finite number, got {rating!r}")
                if not RATING_MIN <= rating <= RATING_MAX:
                    raise ValueError(f"{label}: {method} rating {rating!r} is outside [{RATING_MIN}, {RATING_MAX}]")
                scaled = float(rating) * 10 ** RATING_DECIMALS
                if abs(scaled - round(scaled)) > 1e-6:
                    raise ValueError(f"{label}: {method} rating {rating!r} must have {RATING_DECIMALS} decimal place")
            raw = raw_scores[method]
            if raw is not None and not _is_number(raw):
                raise ValueError(f"{label}: {method} raw score must be null or a finite number, got {raw!r}")

    for side in SIDES:
        if starters[side] != STARTERS_PER_SIDE:
            raise ValueError(f"{side} side has {starters[side]} starters, expected {STARTERS_PER_SIDE}")


# --------------------------------------------------------------------------- loading
@dataclass
class _Tables:
    matches: pd.DataFrame  # one row per match
    by_id: dict  # match_id -> that match's row (namedtuple)
    lineups: dict  # match_id -> DataFrame of that match's lineup rows
    ratings: dict  # method -> {match_id -> {player_id -> (rating, raw_score)}}
    methods: list  # exported methods, in presentation order
    has_times: bool  # lineups carry minute_on / minute_off


def _read(path: pathlib.Path, required: tuple, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing ({what}). Run the earlier pipeline steps first (see CLAUDE.md).")
    frame = pd.read_parquet(path)
    absent = [col for col in required if col not in frame.columns]
    if absent:
        raise ValueError(f"{path} is missing columns {absent}")
    return frame


def _load_tables(processed: pathlib.Path) -> _Tables:
    matches = _read(processed / "matches.parquet", _MATCH_COLUMNS, "match table")
    lineups = _read(processed / "lineups.parquet", _LINEUP_COLUMNS, "lineups")
    if matches["match_id"].duplicated().any():
        raise ValueError("matches.parquet has duplicate match_id rows")

    ratings: dict = {}
    for method in METHODS:
        path = processed / f"ratings_{method}.parquet"
        if not path.exists():
            continue
        frame = _read(path, _RATING_COLUMNS, f"{method} ratings")
        if frame.duplicated(subset=["match_id", "player_id"]).any():
            raise ValueError(f"{path} has duplicate (match_id, player_id) rows")
        per_match: dict = {}
        for row in frame.itertuples(index=False):
            per_match.setdefault(int(row.match_id), {})[int(row.player_id)] = (row.rating, row.raw_score)
        ratings[method] = per_match
    if not ratings:
        names = ", ".join(f"ratings_{m}.parquet" for m in METHODS)
        raise FileNotFoundError(f"no ratings files in {processed} (looked for {names}). "
                                "Run `python -m src.scale.apply --method baseline|vaep` first.")

    return _Tables(
        matches=matches,
        by_id={int(row.match_id): row for row in matches.itertuples(index=False)},
        lineups={int(mid): frame for mid, frame in lineups.groupby("match_id", sort=False)},
        ratings=ratings,
        methods=[m for m in METHODS if m in ratings],
        has_times=all(col in lineups.columns for col in _OPTIONAL_TIME_KEYS),
    )


# --------------------------------------------------------------------------- building
def _clean_number(value: Any, decimals: int):
    """float rounded to ``decimals``, or None for missing / NaN values."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), decimals)


def _formation(value: Any):
    """Formation as a string such as ``"4231"``; the parquet stores it as an integer."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return str(value)


def _date(value: Any) -> str:
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)[:10]


def _build_payload(tables: _Tables, match_id: int) -> dict:
    match = tables.by_id.get(match_id)
    if match is None:
        raise ValueError(f"unknown match_id {match_id}")
    lineup = tables.lineups.get(match_id)
    if lineup is None or lineup.empty:
        raise ValueError(f"no lineup rows for match_id {match_id}")

    team_side = {int(match.home_team_id): "home", int(match.away_team_id): "away"}
    players = []
    for row in lineup.itertuples(index=False):
        side = team_side.get(int(row.team_id))
        if side is None:
            raise ValueError(f"match {match_id}: lineup team_id {row.team_id} is neither the home nor the away team")
        minutes = round(float(row.minutes), 1)
        ratings, raw_scores = {}, {}
        for method in tables.methods:
            found = tables.ratings[method].get(match_id, {}).get(int(row.player_id))
            if minutes < MIN_MINUTES or found is None:
                ratings[method] = raw_scores[method] = None
            else:
                ratings[method] = _clean_number(found[0], RATING_DECIMALS)
                raw_scores[method] = _clean_number(found[1], RAW_SCORE_DECIMALS)
        player = {
            "player_id": int(row.player_id),
            "player_name": str(row.player_name),
            "team_id": int(row.team_id),
            "team": str(row.team),
            "side": side,
            "position_id": int(row.position_id),
            "position": str(row.position),
            "position_group": str(row.position_group),
            "is_starter": bool(row.is_starter),
            "jersey_number": int(row.jersey_number),
            "minutes": minutes,
        }
        if tables.has_times:
            player["minute_on"] = _clean_number(row.minute_on, 1)
            player["minute_off"] = _clean_number(row.minute_off, 1)
        player["ratings"] = ratings
        player["raw_scores"] = raw_scores
        players.append(player)

    def order(p: dict) -> tuple:
        came_on = p.get("minute_on")
        return (
            SIDES.index(p["side"]),
            0 if p["is_starter"] else 1,
            p["position_id"] if p["is_starter"] else (math.inf if came_on is None else came_on),
            p["jersey_number"],
            p["player_id"],
        )

    players.sort(key=order)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_STATSBOMB,
        "season": SEASON,
        "competition": COMPETITION,
        "match_id": int(match_id),
        "match_date": _date(match.match_date),
        "matchweek": int(match.match_week),
        "home": {
            "team_id": int(match.home_team_id), "team": str(match.home_team),
            "score": int(match.home_score), "formation": _formation(match.home_formation),
        },
        "away": {
            "team_id": int(match.away_team_id), "team": str(match.away_team),
            "score": int(match.away_score), "formation": _formation(match.away_formation),
        },
        "methods": list(tables.methods),
        "players": players,
        "credits": CREDITS,
    }


def match_payload(match_id: int, processed_dir: str | pathlib.Path = "data/processed") -> dict:
    """The web payload for one match, built from ``matches``, ``lineups`` and every
    ``ratings_<method>.parquet`` present in ``processed_dir`` (``baseline`` before ``vaep``).

    The result is *not* validated here; run :func:`validate_payload` (``export_season`` does).
    Raises ``ValueError`` for an unknown ``match_id`` and ``FileNotFoundError`` when an input
    file is missing.
    """
    return _build_payload(_load_tables(pathlib.Path(processed_dir)), int(match_id))


def index_entry(payload: dict) -> dict:
    """The ``index.json`` entry for a payload; ``path`` is relative to the index's folder."""
    return {
        "season": payload["season"],
        "match_id": payload["match_id"],
        "match_date": payload["match_date"],
        "matchweek": payload["matchweek"],
        "home": payload["home"]["team"],
        "away": payload["away"]["team"],
        "home_score": payload["home"]["score"],
        "away_score": payload["away"]["score"],
        "methods": list(payload["methods"]),
        "path": f"{season_dir(payload['season'])}/{payload['match_id']}.json",
    }


# --------------------------------------------------------------------------- export
def _dump(obj: Any, *, indent: int | None = None) -> str:
    separators = None if indent else (",", ":")
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=indent, separators=separators)


def export_season(out_dir: str | pathlib.Path = "web/data", processed_dir: str | pathlib.Path = "data/processed") -> list:
    """Write ``<out_dir>/index.json`` and one validated payload per match; return the index entries.

    Every payload is built and validated *before* anything is written, so a failure leaves
    ``out_dir`` untouched (``ValueError`` listing the first problems).
    """
    tables = _load_tables(pathlib.Path(processed_dir))
    matches = tables.matches
    sort_columns = ["match_week", "match_date"] + (["kick_off"] if "kick_off" in matches.columns else []) + ["match_id"]
    match_ids = [int(m) for m in matches.sort_values(sort_columns)["match_id"]]

    payloads, problems = [], []
    for match_id in match_ids:
        try:
            payload = _build_payload(tables, match_id)
            validate_payload(payload)
        except ValueError as exc:
            problems.append(f"match {match_id}: {exc}")
        else:
            payloads.append(payload)
    if problems:
        shown = "\n  ".join(problems[:10])
        raise ValueError(f"{len(problems)} of {len(match_ids)} matches failed validation; nothing was written. "
                         f"First problems:\n  {shown}")

    out = pathlib.Path(out_dir)
    entries = [index_entry(p) for p in payloads]
    for payload, entry in zip(payloads, entries):
        target = out / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_dump(payload), encoding="utf-8")
    (out / "index.json").write_text(_dump(entries, indent=1) + "\n", encoding="utf-8")
    log.info("wrote %d match payloads and index.json to %s", len(entries), out)
    return entries


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.export_web",
                                     description="Export the per-match JSON payloads the match-ratings page loads.")
    parser.add_argument("--out", default="web/data", help="output folder (default: web/data)")
    parser.add_argument("--processed", default="data/processed", help="folder with the processed parquet files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        entries = export_season(args.out, args.processed)
    except (ValueError, FileNotFoundError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(entries)} matches to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
