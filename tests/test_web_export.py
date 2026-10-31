"""Tests for src/pipeline/export_web.py.

Everything here runs on a small SYNTHETIC processed directory built in ``tmp_path`` (real column
names from CLAUDE.md, made-up players and ratings). Nothing here is real data.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.contract import MIN_MINUTES, RATING_MAX, RATING_MIN
from src.pipeline.export_web import (
    CREDITS, SEASON, export_season, index_entry, main, match_payload, validate_payload,
)
from tests.conftest import ROOT

HOME = (10, "Synthetic Home FC")
AWAY = (20, "Synthetic Away FC")
XI = [  # (position_id, position, group): 11 starters
    (1, "Goalkeeper", "GK"), (2, "Right Back", "DEF"), (3, "Right Center Back", "DEF"),
    (5, "Left Center Back", "DEF"), (6, "Left Back", "DEF"), (10, "Center Defensive Midfield", "MID"),
    (13, "Right Center Midfield", "MID"), (15, "Left Center Midfield", "MID"), (17, "Right Wing", "FWD"),
    (23, "Center Forward", "FWD"), (21, "Left Wing", "FWD"),
]
SUBS = [  # (position_id, position, group, minutes); the last one is under 10 minutes
    (14, "Center Midfield", "MID", 25.0), (23, "Center Forward", "FWD", 15.0), (4, "Center Back", "DEF", 6.5),
]
MATCH_LENGTH = 94.0
LINEUP_COLUMNS = [
    "match_id", "team_id", "team", "player_id", "player_name", "player_full_name", "jersey_number", "is_starter",
    "starting_position_id", "position_id", "position", "position_group", "minutes", "minute_on", "minute_off",
    "red_card_minute",
]


def player_id(team_id: int, slot: int) -> int:
    return 1000 * team_id + slot


def make_matches() -> pd.DataFrame:
    """Two matches, listed out of matchweek order on purpose (the export sorts them)."""
    return pd.DataFrame({
        "match_id": np.array([102, 101], dtype="int64"),
        "match_date": ["2015-08-15", "2015-08-08"],
        "kick_off": ["16:00:00.000", "12:45:00.000"],
        "match_week": np.array([2, 1], dtype="int64"),
        "home_team_id": np.array([AWAY[0], HOME[0]], dtype="int64"),
        "home_team": [AWAY[1], HOME[1]],
        "away_team_id": np.array([HOME[0], AWAY[0]], dtype="int64"),
        "away_team": [HOME[1], AWAY[1]],
        "home_score": np.array([0, 2], dtype="int64"),
        "away_score": np.array([0, 1], dtype="int64"),
        "stadium": ["Synthetic Park", "Synthetic Park"],
        "referee": ["Synthetic Ref", "Synthetic Ref"],
        "home_formation": np.array([442, 433], dtype="int64"),
        "away_formation": np.array([4411, 4231], dtype="int64"),
    })


def make_lineups(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in matches.itertuples():
        for team_id, team in ((m.home_team_id, m.home_team), (m.away_team_id, m.away_team)):
            for slot, (position_id, position, group) in enumerate(XI):
                minutes = 70.0 if slot in (5, 9) else MATCH_LENGTH  # two starters are substituted off
                rows.append((m.match_id, team_id, team, player_id(team_id, slot), f"{team} Player {slot}",
                             f"{team} Player {slot} Full", slot + 1, True, position_id, position_id, position,
                             group, minutes, 0.0, minutes, np.nan))
            for j, (position_id, position, group, minutes) in enumerate(SUBS):
                slot = len(XI) + j
                rows.append((m.match_id, team_id, team, player_id(team_id, slot), f"{team} Player {slot}",
                             f"{team} Player {slot} Full", slot + 1, False, position_id, position_id, position,
                             group, minutes, MATCH_LENGTH - minutes, MATCH_LENGTH, np.nan))
    frame = pd.DataFrame(rows, columns=LINEUP_COLUMNS)
    for col in ("match_id", "team_id", "player_id", "jersey_number", "starting_position_id", "position_id"):
        frame[col] = frame[col].astype("int64")
    return frame


def make_ratings(lineups: pd.DataFrame, method: str) -> pd.DataFrame:
    """Contract columns + rating, rated rows only (minutes >= MIN_MINUTES), as the scale module writes them."""
    rated = lineups[lineups["minutes"] >= MIN_MINUTES]
    frame = rated[["match_id", "player_id", "player_name", "team", "position_group", "minutes"]].copy()
    k = 1 if method == "baseline" else 3
    frame["raw_score"] = ((frame["player_id"] * k) % 17) / 4.0 - 2.0
    frame["rating"] = np.round(RATING_MIN + ((frame["player_id"] * k) % 71) / 10.0, 1)  # 3.0 .. 10.0
    return frame.reset_index(drop=True)


def build_processed(directory, *, methods=("baseline", "vaep"), drop_from_vaep=()) -> str:
    """Write matches / lineups / ratings parquet files into ``directory``; return its path as a string.

    The baseline file also carries a STRAY row for the under-10-minute substitute of match 101 (the scale
    module would never write one): the export must still leave that player unrated. ``drop_from_vaep``
    removes (match_id, player_id) pairs of rated players from the VAEP file."""
    directory.mkdir(parents=True, exist_ok=True)
    matches = make_matches()
    lineups = make_lineups(matches)
    matches.to_parquet(directory / "matches.parquet", index=False)
    lineups.to_parquet(directory / "lineups.parquet", index=False)
    for method in methods:
        frame = make_ratings(lineups, method)
        if method == "baseline":
            short = lineups[(lineups["match_id"] == 101) & (lineups["minutes"] < MIN_MINUTES)].iloc[0]
            stray = frame.iloc[[0]].copy()
            stray[["match_id", "player_id", "player_name", "team", "position_group", "minutes"]] = [
                short.match_id, short.player_id, short.player_name, short.team, short.position_group, short.minutes]
            stray[["raw_score", "rating"]] = [0.5, 6.0]
            frame = pd.concat([frame, stray], ignore_index=True)
        if method == "vaep":
            for match_id, pid in drop_from_vaep:
                frame = frame[~((frame["match_id"] == match_id) & (frame["player_id"] == pid))]
        frame["match_id"] = frame["match_id"].astype("int64")
        frame["player_id"] = frame["player_id"].astype("int64")
        frame.to_parquet(directory / f"ratings_{method}.parquet", index=False)
    return str(directory)


@pytest.fixture
def processed(tmp_path) -> str:
    return build_processed(tmp_path / "processed")


@pytest.fixture
def payload(processed) -> dict:
    return match_payload(101, processed)


def player(payload: dict, pid: int) -> dict:
    return next(p for p in payload["players"] if p["player_id"] == pid)


# ------------------------------------------------------------------ match_payload
def test_match_payload_follows_the_web_schema(payload):
    assert payload["schema_version"] == 1
    assert payload["source"] == "StatsBomb Open Data"
    assert payload["season"] == SEASON == "2015/16"
    assert payload["competition"] == "Premier League"
    assert payload["credits"] == CREDITS and "StatsBomb" in payload["credits"]
    assert payload["match_id"] == 101 and payload["matchweek"] == 1 and payload["match_date"] == "2015-08-08"
    assert payload["home"] == {"team_id": 10, "team": "Synthetic Home FC", "score": 2, "formation": "433"}
    assert payload["away"] == {"team_id": 20, "team": "Synthetic Away FC", "score": 1, "formation": "4231"}
    assert isinstance(payload["home"]["formation"], str)
    assert payload["methods"] == ["baseline", "vaep"]
    assert len(payload["players"]) == 2 * (len(XI) + len(SUBS))
    validate_payload(payload)


def test_match_payload_players_carry_the_documented_fields(payload):
    keys = {"player_id", "player_name", "team_id", "team", "side", "position_id", "position", "position_group",
            "is_starter", "jersey_number", "minutes", "minute_on", "minute_off", "ratings", "raw_scores"}
    for p in payload["players"]:
        assert set(p) == keys
        assert p["team_id"] == payload[p["side"]]["team_id"], "side must follow the team"
        assert p["team"] == payload[p["side"]]["team"]
    for side in ("home", "away"):
        assert sum(p["is_starter"] for p in payload["players"] if p["side"] == side) == 11
    gk = player(payload, player_id(10, 0))
    assert (gk["side"], gk["position_id"], gk["position"], gk["position_group"], gk["jersey_number"]) == \
        ("home", 1, "Goalkeeper", "GK", 1)
    sub = player(payload, player_id(20, len(XI)))
    assert sub["is_starter"] is False and sub["minute_on"] == pytest.approx(MATCH_LENGTH - 25.0)
    assert sub["minute_off"] == MATCH_LENGTH


def test_match_payload_is_plain_json_and_ordered_home_then_away_starters_first(payload):
    json.dumps(payload, allow_nan=False)  # raises on numpy scalars or NaN
    sides = [p["side"] for p in payload["players"]]
    assert sides == sorted(sides, key=("home", "away").index)
    for side in ("home", "away"):
        flags = [p["is_starter"] for p in payload["players"] if p["side"] == side]
        assert flags == sorted(flags, reverse=True), "starters come before substitutes"


def test_ratings_and_raw_scores_come_from_the_ratings_files(processed, payload):
    for method in payload["methods"]:
        frame = pd.read_parquet(f"{processed}/ratings_{method}.parquet")
        frame = frame[frame["match_id"] == 101].set_index("player_id")
        for p in payload["players"]:
            rating, raw = p["ratings"][method], p["raw_scores"][method]
            if p["minutes"] < MIN_MINUTES:
                assert rating is None and raw is None
            else:
                assert rating == frame.loc[p["player_id"], "rating"] and round(rating, 1) == rating
                assert RATING_MIN <= rating <= RATING_MAX
                assert raw == pytest.approx(frame.loc[p["player_id"], "raw_score"], abs=1e-4)


def test_under_ten_minute_sub_is_unrated_even_if_a_ratings_file_has_a_row_for_him(processed, payload):
    short = player(payload, player_id(10, len(XI) + 2))
    assert short["minutes"] == 6.5
    baseline = pd.read_parquet(f"{processed}/ratings_baseline.parquet")
    assert short["player_id"] in set(baseline["player_id"]), "the fixture must contain the stray row"
    assert short["ratings"] == {"baseline": None, "vaep": None}
    assert short["raw_scores"] == {"baseline": None, "vaep": None}


def test_rated_player_missing_from_one_ratings_file_gets_null_there_and_fails_validation(tmp_path):
    missing = player_id(10, 1)
    processed = build_processed(tmp_path / "p", drop_from_vaep=[(101, missing)])
    payload = match_payload(101, processed)
    p = player(payload, missing)
    assert p["minutes"] >= MIN_MINUTES
    assert p["ratings"]["vaep"] is None and p["raw_scores"]["vaep"] is None
    assert p["ratings"]["baseline"] is not None
    with pytest.raises(ValueError, match="vaep"):
        validate_payload(payload)


@pytest.mark.parametrize("methods", [("baseline",), ("vaep",), ("baseline", "vaep")])
def test_methods_are_exactly_those_with_a_ratings_file(tmp_path, methods):
    processed = build_processed(tmp_path / "p", methods=methods)
    payload = match_payload(101, processed)
    assert payload["methods"] == list(methods)
    for p in payload["players"]:
        assert list(p["ratings"]) == list(methods) and list(p["raw_scores"]) == list(methods)
    validate_payload(payload)


def test_method_order_is_baseline_then_vaep_whatever_the_file_order(tmp_path):
    processed = build_processed(tmp_path / "p", methods=("vaep", "baseline"))
    assert match_payload(101, processed)["methods"] == ["baseline", "vaep"]


def test_missing_input_files_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):  # no ratings at all
        match_payload(101, build_processed(tmp_path / "none", methods=()))
    with pytest.raises(FileNotFoundError):
        export_season(tmp_path / "out", build_processed(tmp_path / "none2", methods=()))
    with pytest.raises(FileNotFoundError):  # not even a processed folder
        match_payload(101, tmp_path / "does-not-exist")


@pytest.mark.parametrize("bad_id", [0, 999, -1])
def test_unknown_match_id_raises_value_error(processed, bad_id):
    with pytest.raises(ValueError, match=str(bad_id)):
        match_payload(bad_id, processed)


def test_numpy_integer_match_ids_are_accepted(processed):
    assert match_payload(np.int64(102), processed)["match_id"] == 102


# ------------------------------------------------------------------ export_season
def test_export_season_writes_the_index_and_one_validated_payload_per_match(processed, tmp_path):
    out = tmp_path / "site" / "data"
    entries = export_season(str(out), processed)
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index == entries
    assert [e["match_id"] for e in index] == [101, 102], "sorted by matchweek, not by parquet row order"
    assert index[0] == {
        "season": "2015/16", "match_id": 101, "match_date": "2015-08-08", "matchweek": 1,
        "home": "Synthetic Home FC", "away": "Synthetic Away FC", "home_score": 2, "away_score": 1,
        "methods": ["baseline", "vaep"], "path": "2015-16/101.json",
    }
    assert set(index[1]) == set(index[0])
    for entry in index:
        written = json.loads((out / entry["path"]).read_text(encoding="utf-8"))
        validate_payload(written)
        assert written == json.loads(json.dumps(match_payload(entry["match_id"], processed)))
        assert index_entry(written) == entry
    assert sorted(p.name for p in (out / "2015-16").iterdir()) == ["101.json", "102.json"]


def test_export_season_writes_nothing_when_any_payload_is_invalid(tmp_path):
    processed = build_processed(tmp_path / "p", drop_from_vaep=[(102, player_id(20, 3))])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="102"):
        export_season(out, processed)
    assert not out.exists() or not list(out.rglob("*.json")), "a failed export must not leave partial output"


def test_export_is_deterministic_and_never_writes_nan(processed, tmp_path):
    export_season(tmp_path / "a", processed)
    export_season(tmp_path / "b", processed)
    for path in sorted((tmp_path / "a").rglob("*.json")):
        twin = tmp_path / "b" / path.relative_to(tmp_path / "a")
        assert path.read_bytes() == twin.read_bytes()
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text


def test_export_works_when_lineups_have_no_minute_on_off_columns(tmp_path):
    processed = tmp_path / "p"
    build_processed(processed)
    lineups = pd.read_parquet(processed / "lineups.parquet").drop(columns=["minute_on", "minute_off"])
    lineups.to_parquet(processed / "lineups.parquet", index=False)
    payload = match_payload(101, processed)
    assert all("minute_on" not in p and "minute_off" not in p for p in payload["players"])
    validate_payload(payload)


def test_export_sorts_matches_without_a_kick_off_column(tmp_path):
    processed = tmp_path / "p"
    build_processed(processed)
    matches = pd.read_parquet(processed / "matches.parquet").drop(columns=["kick_off", "stadium", "referee"])
    matches.to_parquet(processed / "matches.parquet", index=False)
    assert [e["match_id"] for e in export_season(tmp_path / "out", processed)] == [101, 102]


def test_season_label_is_fixed_whatever_the_matches_table_says(tmp_path):
    """The real matches.parquet carries season='2015/2016' and competition columns; the payload uses 2015/16."""
    processed = tmp_path / "p"
    build_processed(processed)
    matches = pd.read_parquet(processed / "matches.parquet")
    matches["season"], matches["competition"] = "2015/2016", "Premier League"
    matches.to_parquet(processed / "matches.parquet", index=False)
    payload = match_payload(101, processed)
    assert payload["season"] == "2015/16" and payload["competition"] == "Premier League"
    assert index_entry(payload)["path"] == "2015-16/101.json"


def test_formation_is_a_string_even_for_float_or_missing_values(tmp_path):
    processed = tmp_path / "p"
    build_processed(processed)
    matches = pd.read_parquet(processed / "matches.parquet")
    matches["home_formation"] = matches["home_formation"].astype("float64")
    matches.loc[matches["match_id"] == 101, "away_formation"] = np.nan
    matches["away_formation"] = matches["away_formation"].astype("float64")
    matches.to_parquet(processed / "matches.parquet", index=False)
    payload = match_payload(101, processed)
    assert payload["home"]["formation"] == "433" and payload["away"]["formation"] is None
    validate_payload(payload)


# ------------------------------------------------------------------ validate_payload
def mutated(payload: dict, change) -> dict:
    bad = copy.deepcopy(payload)
    change(bad)
    return bad


def first_rated(p: dict) -> dict:
    return next(x for x in p["players"] if x["minutes"] >= MIN_MINUTES)


def first_unrated(p: dict) -> dict:
    return next(x for x in p["players"] if x["minutes"] < MIN_MINUTES)


BAD_PAYLOADS = {
    "missing top-level key": lambda p: p.pop("matchweek"),
    "missing credits": lambda p: p.pop("credits"),
    "missing home key": lambda p: p["home"].pop("formation"),
    "missing player key": lambda p: p["players"][0].pop("position_group"),
    "missing raw_scores": lambda p: p["players"][0].pop("raw_scores"),
    "unsupported schema_version": lambda p: p.__setitem__("schema_version", 2),
    "bad side": lambda p: p["players"][0].__setitem__("side", "left"),
    "bad position group": lambda p: p["players"][0].__setitem__("position_group", "STRIKER"),
    "lowercase position group": lambda p: p["players"][0].__setitem__("position_group", "gk"),
    "duplicate player_id": lambda p: p["players"][1].__setitem__("player_id", p["players"][0]["player_id"]),
    "empty methods": lambda p: p.__setitem__("methods", []),
    "methods is a string": lambda p: p.__setitem__("methods", "baseline"),
    "duplicate methods": lambda p: p.__setitem__("methods", ["baseline", "baseline"]),
    "ratings lack a method": lambda p: p["players"][0]["ratings"].pop("vaep"),
    "ratings have an extra method": lambda p: p["players"][0]["ratings"].__setitem__("elo", 6.5),
    "raw_scores lack a method": lambda p: p["players"][0]["raw_scores"].pop("vaep"),
    "rated player under 10 minutes": lambda p: first_unrated(p).__setitem__("ratings", {"baseline": 6.5, "vaep": None}),
    "unrated player with 10+ minutes": lambda p: first_rated(p)["ratings"].__setitem__("baseline", None),
    "rating below 3.0": lambda p: first_rated(p)["ratings"].__setitem__("baseline", 2.9),
    "rating above 10.0": lambda p: first_rated(p)["ratings"].__setitem__("baseline", 10.1),
    "rating with two decimals": lambda p: first_rated(p)["ratings"].__setitem__("baseline", 6.55),
    "rating NaN": lambda p: first_rated(p)["ratings"].__setitem__("baseline", float("nan")),
    "rating as a string": lambda p: first_rated(p)["ratings"].__setitem__("baseline", "6.5"),
    "rating as a bool": lambda p: first_rated(p)["ratings"].__setitem__("baseline", True),
    "raw score NaN": lambda p: first_rated(p)["raw_scores"].__setitem__("baseline", float("nan")),
    "negative minutes": lambda p: p["players"][0].__setitem__("minutes", -1.0),
    "minutes not a number": lambda p: p["players"][0].__setitem__("minutes", "94"),
    "is_starter not a bool": lambda p: p["players"][0].__setitem__("is_starter", "yes"),
    "negative minute_on": lambda p: p["players"][0].__setitem__("minute_on", -5.0),
    "ten home starters": lambda p: p["players"][0].__setitem__("is_starter", False),
    "twelve home starters": lambda p: next(x for x in p["players"] if x["side"] == "home" and not x["is_starter"]).__setitem__("is_starter", True),
    "ten away starters": lambda p: next(x for x in p["players"] if x["side"] == "away" and x["is_starter"]).__setitem__("is_starter", False),
    "no away players": lambda p: p.__setitem__("players", [x for x in p["players"] if x["side"] == "home"]),
    "credits do not mention StatsBomb": lambda p: p.__setitem__("credits", "Data: somebody else"),
    "credits empty": lambda p: p.__setitem__("credits", ""),
    "payload is not a dict": None,
}


@pytest.mark.parametrize("name", sorted(BAD_PAYLOADS))
def test_validate_payload_rejects(payload, name):
    change = BAD_PAYLOADS[name]
    bad = mutated(payload, change) if change else ["not", "a", "payload"]
    with pytest.raises(ValueError):
        validate_payload(bad)


def test_validate_payload_accepts_the_unmodified_payload(payload):
    validate_payload(payload)


def test_validate_payload_rating_boundaries(payload):
    ok = mutated(payload, lambda p: first_rated(p)["ratings"].update(baseline=RATING_MIN, vaep=RATING_MAX))
    validate_payload(ok)

    def at_exactly_ten_minutes(p):
        first_rated(p)["minutes"] = MIN_MINUTES  # 10.0 minutes is rated
    validate_payload(mutated(payload, at_exactly_ten_minutes))

    def just_under_ten_minutes(p):
        first_rated(p)["minutes"] = MIN_MINUTES - 0.1  # 9.9 minutes must not be
    with pytest.raises(ValueError):
        validate_payload(mutated(payload, just_under_ten_minutes))


def test_validate_payload_accepts_other_sources_without_a_statsbomb_credit(payload):
    def fpl(p):
        p["source"] = "Fantasy Premier League API"
        p["credits"] = "Data: Fantasy Premier League API (https://fantasy.premierleague.com/)"
    validate_payload(mutated(payload, fpl))


def test_validate_payload_accepts_a_single_method_and_missing_minute_keys(tmp_path):
    single = match_payload(101, build_processed(tmp_path / "p", methods=("baseline",)))
    validate_payload(single)

    def drop_times(p):
        for x in p["players"]:
            x.pop("minute_on"), x.pop("minute_off")
    validate_payload(mutated(single, drop_times))


def test_validate_payload_messages_name_the_problem(payload):
    bad = mutated(payload, lambda p: first_unrated(p).__setitem__("ratings", {"baseline": 6.5, "vaep": None}))
    with pytest.raises(ValueError, match="baseline"):
        validate_payload(bad)
    bad = mutated(payload, lambda p: p["players"][0].__setitem__("is_starter", False))
    with pytest.raises(ValueError, match="10 starters"):
        validate_payload(bad)


# ------------------------------------------------------------------ CLI
def test_cli_prints_how_many_matches_were_written(processed, tmp_path, capsys):
    out = tmp_path / "web-data"
    assert main(["--out", str(out), "--processed", processed]) == 0
    assert "wrote 2 matches" in capsys.readouterr().out
    assert (out / "index.json").exists()


def test_cli_reports_failures_and_exits_non_zero(tmp_path, capsys):
    out = tmp_path / "web-data"
    assert main(["--out", str(out), "--processed", str(tmp_path / "missing")]) == 1
    assert "export failed" in capsys.readouterr().err
    assert not out.exists()


def test_module_runs_with_python_dash_m(processed, tmp_path):
    out = tmp_path / "web-data"
    run = subprocess.run([sys.executable, "-m", "src.pipeline.export_web", "--out", str(out), "--processed", processed],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert "wrote 2 matches" in run.stdout
    assert json.loads((out / "index.json").read_text(encoding="utf-8"))[0]["match_id"] == 101
