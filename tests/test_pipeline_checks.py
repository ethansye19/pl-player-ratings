"""Gameweek validity checks for blank gameweeks and missing starters. Network-free: everything below
uses the SYNTHETIC FPL payloads from tests/test_pipeline.py."""
import json

import numpy as np
import pandas as pd
import pytest

from src.contract import SEED
from src.pipeline.gameweek import run, run_checks
from tests.test_pipeline import FakeFplClient, _synthetic_gameweek_ratings


def _blank_gameweek(n_fixtures: int = 8):
    """SYNTHETIC rated gameweek in which only 2 * n_fixtures of the 20 teams play."""
    df, fixtures, _ = _synthetic_gameweek_ratings(n_teams=2 * n_fixtures)
    teams = [{"id": t, "name": f"Synthetic Team {t}"} for t in range(1, 21)]
    return df, fixtures, teams


def test_a_blank_gameweek_passes_the_checks():
    df, fixtures, teams = _blank_gameweek()
    assert df["team_id"].nunique() == 16
    assert run_checks(df, fixtures, teams) == []


def test_a_team_with_a_fixture_still_needs_rated_players():
    df, fixtures, teams = _blank_gameweek()
    failures = run_checks(df[df["team_id"] != 3], fixtures, teams)
    assert any("Synthetic Team 3 has a fixture but no rated players" in f for f in failures)
    assert not any("Synthetic Team 17" in f for f in failures), "teams without a fixture are not missing"


def test_a_starter_with_10_plus_minutes_must_be_rated():
    df, fixtures, teams = _synthetic_gameweek_ratings()
    victim = df[df["is_starter"]].iloc[3]
    rated = df.drop(index=victim.name)
    failures = run_checks(rated, fixtures, teams, appearances=df)
    assert failures == [f"fixture {victim.match_id}: {victim.team} starter {victim.player_name} (90 min) is not rated"]


def test_a_starter_under_10_minutes_may_be_unrated():
    df, fixtures, teams = _synthetic_gameweek_ratings()
    appearances = df.copy()
    appearances.loc[3, "minutes"] = 6.0  # e.g. an early injury
    rated = appearances.drop(index=3)
    assert run_checks(rated, fixtures, teams, appearances=appearances) == []


def test_a_team_without_11_starters_is_a_check_failure():
    df, fixtures, teams = _synthetic_gameweek_ratings()
    victim = df[df["is_starter"]].iloc[0]
    short = df.drop(index=victim.name)  # the starter is absent from the FPL data altogether
    failures = run_checks(short, fixtures, teams, appearances=short)
    assert f"fixture {victim.match_id}: {victim.team} has 10 starters in the FPL data (expected 11)" in failures


def _drop_fixture(client: FakeFplClient, gw: int) -> int:
    fixture = client._fixtures[gw].pop(0)
    client._live[gw]["elements"] = [e for e in client._live[gw]["elements"] if e["explain"][0]["fixture"] != fixture["id"]]
    return fixture["id"]


def test_run_publishes_a_blank_gameweek(tmp_path):
    client = FakeFplClient()
    dropped = _drop_fixture(client, 2)
    result = run(client, out_dir=tmp_path)
    assert result.ok is True, result.failures
    index = json.loads((tmp_path / "index.json").read_text())
    assert len(index) == 9 and dropped not in {e["match_id"] for e in index}
    assert result.ratings["team_id"].nunique() == 18


def test_run_reports_a_missing_starter_instead_of_crashing(tmp_path):
    client = FakeFplClient()
    elements = {e["id"]: e for e in client._bootstrap["elements"]}
    live = client._live[2]["elements"]
    victim = next(e for e in live if e["stats"]["starts"] == 1 and elements[e["id"]]["team"] == 5)
    live.remove(victim)
    result = run(client, out_dir=tmp_path)  # used to raise ValueError from validate_payload
    assert result.ok is False
    assert any("Synthetic Team 5 has 10 starters in the FPL data (expected 11)" in f for f in result.failures), result.failures
    assert list(tmp_path.rglob("*")) == [], "no output may be written when a check fails"


def test_run_publishes_a_starter_under_10_minutes_as_unrated(tmp_path):
    client = FakeFplClient()
    live = client._live[2]["elements"]
    victim = next(e for e in live if e["stats"]["starts"] == 1)
    victim["stats"]["minutes"] = 7
    victim["explain"][0]["stats"][0]["value"] = 7
    result = run(client, out_dir=tmp_path)
    assert result.ok is True, result.failures
    fixture_id = victim["explain"][0]["fixture"]
    payload = json.loads((tmp_path / "2026-27" / f"{fixture_id}.json").read_text())
    row = next(p for p in payload["players"] if p["player_id"] == victim["id"])
    assert row["is_starter"] is True and row["minutes"] == 7.0 and row["ratings"]["baseline"] is None
