"""Current-season pipeline (FPL API, baseline method only). Network-free: a fake client
serves SYNTHETIC payloads shaped like the real API. The pipeline agent may extend the
fake's stat fields; the assertions stay."""
import json

import numpy as np
import pandas as pd
import pytest

from src.contract import POSITION_GROUPS, RATING_MAX, RATING_MIN, SEED
from tests.conftest import import_or_fail

# ------------------------------------------------------------------ synthetic FPL API
FPL_STAT_FIELDS = {
    "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 0, "own_goals": 0,
    "penalties_saved": 0, "penalties_missed": 0, "yellow_cards": 0, "red_cards": 0, "saves": 0,
    "bonus": 0, "bps": 0, "influence": "0.0", "creativity": "0.0", "threat": "0.0",
    "ict_index": "0.0", "starts": 0, "expected_goals": "0.00", "expected_assists": "0.00",
    "expected_goal_involvements": "0.00", "expected_goals_conceded": "0.00",
    "clearances_blocks_interceptions": 0, "recoveries": 0, "tackles": 0,
    "defensive_contribution": 0, "total_points": 0, "in_dreamteam": False,
}
SQUAD = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3          # element_type per squad slot (15 players)
XI = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]               # squad slots that start


def make_fake_fpl(n_teams: int = 20, gameweeks=(1, 2), seed: int = SEED, drop_team=None):
    """SYNTHETIC bootstrap/fixtures/live payloads mirroring the FPL API shapes."""
    rng = np.random.default_rng(seed)
    teams = [{"id": t, "name": f"Synthetic Team {t}", "short_name": f"S{t:02d}"} for t in range(1, n_teams + 1)]
    elements, eid = [], 1
    for t in range(1, n_teams + 1):
        for slot, etype in enumerate(SQUAD):
            elements.append({"id": eid, "web_name": f"Synth{eid}", "first_name": "Synthetic",
                             "second_name": f"Player {eid}", "team": t, "element_type": etype,
                             "status": "a", "squad_slot": slot})
            eid += 1
    bootstrap = {
        "events": [{"id": gw, "name": f"Gameweek {gw}", "finished": True, "data_checked": True} for gw in gameweeks]
        + [{"id": max(gameweeks) + 1, "name": "Gameweek next", "finished": False, "data_checked": False}],
        "teams": teams,
        "elements": elements,
        "element_types": [{"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
                          {"id": 3, "singular_name_short": "MID"}, {"id": 4, "singular_name_short": "FWD"}],
    }
    fixtures, live = {}, {}
    fid = 1
    for gw in gameweeks:
        order = rng.permutation(np.arange(1, n_teams + 1))
        fixtures[gw] = []
        pairs = [(int(order[i]), int(order[i + 1])) for i in range(0, n_teams - 1, 2)]
        team_fixture = {}
        for h, a in pairs:
            fixtures[gw].append({"id": fid, "event": gw, "team_h": h, "team_a": a, "finished": True,
                                 "team_h_score": int(rng.integers(0, 4)), "team_a_score": int(rng.integers(0, 4)),
                                 "kickoff_time": f"2026-10-{10 + gw:02d}T14:00:00Z"})
            team_fixture[h] = team_fixture[a] = fid
            fid += 1
        rows = []
        for el in elements:
            if drop_team is not None and el["team"] == drop_team:
                continue
            slot = el["squad_slot"]
            if slot in XI:
                minutes = 90 if rng.random() < 0.8 else int(rng.integers(60, 85))
                starts = 1
            elif slot == 14:
                minutes, starts = 0, 0
            else:
                minutes, starts = int(rng.integers(3, 30)), 0
            stats = dict(FPL_STAT_FIELDS)
            stats.update({
                "minutes": minutes, "starts": starts,
                "goals_scored": int(rng.random() < 0.08 * (minutes / 90)),
                "assists": int(rng.random() < 0.08 * (minutes / 90)),
                "saves": int(rng.integers(0, 6)) if el["element_type"] == 1 and minutes else 0,
                "bps": int(rng.integers(-5, 40)) if minutes else 0,
                "tackles": int(rng.integers(0, 5)) if minutes else 0,
                "recoveries": int(rng.integers(0, 9)) if minutes else 0,
                "clearances_blocks_interceptions": int(rng.integers(0, 8)) if minutes else 0,
                "yellow_cards": int(rng.random() < 0.1) if minutes else 0,
                "goals_conceded": int(rng.integers(0, 3)) if minutes else 0,
                "expected_goals": f"{rng.random() * 0.6:.2f}" if minutes else "0.00",
                "expected_assists": f"{rng.random() * 0.3:.2f}" if minutes else "0.00",
            })
            rows.append({"id": el["id"], "stats": stats,
                         "explain": [{"fixture": team_fixture[el["team"]],
                                      "stats": [{"identifier": "minutes", "points": 2 if minutes else 0,
                                                 "value": minutes}]}]})
        live[gw] = {"elements": rows}
    return bootstrap, fixtures, live


class FakeFplClient:
    """Implements the FplClient protocol from CLAUDE.md with synthetic payloads."""

    def __init__(self, **kwargs):
        self._bootstrap, self._fixtures, self._live = make_fake_fpl(**kwargs)

    def bootstrap(self) -> dict:
        return self._bootstrap

    def fixtures(self, gw: int) -> list:
        return self._fixtures[gw]

    def live(self, gw: int) -> dict:
        return self._live[gw]


def _synthetic_gameweek_ratings(seed=SEED, n_teams=20):
    rng = np.random.default_rng(seed)
    rows = []
    pid = 1
    fixtures = [{"id": f + 1, "event": 1, "team_h": 2 * f + 1, "team_a": 2 * f + 2} for f in range(n_teams // 2)]
    for f in fixtures:
        for team_id in (f["team_h"], f["team_a"]):
            for group in ["GK"] + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 2:
                rows.append((f["id"], pid, f"Synth{pid}", f"Synthetic Team {team_id}", team_id, group, 90.0,
                             float(rng.normal()), float(np.clip(np.round(rng.normal(6.5, 0.8), 1), 3.0, 10.0)), True))
                pid += 1
    df = pd.DataFrame(rows, columns=["match_id", "player_id", "player_name", "team", "team_id", "position_group",
                                     "minutes", "raw_score", "rating", "is_starter"])
    teams = [{"id": t, "name": f"Synthetic Team {t}"} for t in range(1, n_teams + 1)]
    return df, fixtures, teams


# ---------------------------------------------------------------------------- tests
def test_fpl_element_types_map_to_position_groups():
    FPL_POSITION_GROUPS = import_or_fail("src.pipeline.fpl_mapping", "FPL_POSITION_GROUPS")
    assert FPL_POSITION_GROUPS == {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    assert set(FPL_POSITION_GROUPS.values()) == set(POSITION_GROUPS)


def test_every_baseline_stat_is_mapped_or_documented_as_unavailable():
    STATS = import_or_fail("src.baseline.weights", "STATS")
    FPL_STAT_MAP, UNAVAILABLE_STATS = import_or_fail("src.pipeline.fpl_mapping", "FPL_STAT_MAP", "UNAVAILABLE_STATS")
    assert set(FPL_STAT_MAP) | set(UNAVAILABLE_STATS) == set(STATS)
    assert set(FPL_STAT_MAP).isdisjoint(UNAVAILABLE_STATS)
    assert all(isinstance(v, str) and v for v in FPL_STAT_MAP.values()), "map to a named FPL field"
    assert all(isinstance(note, str) and len(note) >= 15 for note in UNAVAILABLE_STATS.values())


def test_fpl_weights_only_use_available_stats_and_cover_all_groups():
    FPL_STAT_MAP, FPL_WEIGHTS = import_or_fail("src.pipeline.fpl_mapping", "FPL_STAT_MAP", "FPL_WEIGHTS")
    assert set(FPL_WEIGHTS) == set(POSITION_GROUPS)
    for group, ws in FPL_WEIGHTS.items():
        assert set(ws) <= set(FPL_STAT_MAP), f"{group} uses stats FPL does not provide"
        for stat, w in ws.items():
            assert isinstance(w.why, str) and len(w.why.strip()) >= 15, (group, stat)


def test_checks_pass_for_a_valid_synthetic_gameweek():
    run_checks = import_or_fail("src.pipeline.gameweek", "run_checks")
    df, fixtures, teams = _synthetic_gameweek_ratings()
    assert run_checks(df, fixtures, teams) == []


def test_checks_fail_when_a_team_is_missing():
    run_checks = import_or_fail("src.pipeline.gameweek", "run_checks")
    df, fixtures, teams = _synthetic_gameweek_ratings()
    failures = run_checks(df[df["team_id"] != 7], fixtures, teams)
    assert failures and any("Synthetic Team 7" in f for f in failures)


def test_checks_fail_when_a_rating_is_out_of_bounds():
    run_checks = import_or_fail("src.pipeline.gameweek", "run_checks")
    df, fixtures, teams = _synthetic_gameweek_ratings()
    df.loc[0, "rating"] = RATING_MAX + 0.1
    assert run_checks(df, fixtures, teams)
    df, fixtures, teams = _synthetic_gameweek_ratings()
    df.loc[0, "rating"] = RATING_MIN - 0.1
    assert run_checks(df, fixtures, teams)


def test_checks_fail_when_the_gameweek_mean_is_off():
    run_checks = import_or_fail("src.pipeline.gameweek", "run_checks")
    df, fixtures, teams = _synthetic_gameweek_ratings()
    df["rating"] = np.clip(df["rating"] + 1.0, RATING_MIN, RATING_MAX).round(1)
    assert run_checks(df, fixtures, teams)


def test_latest_finished_gameweek_is_detected():
    latest = import_or_fail("src.pipeline.fpl", "latest_finished_gameweek")
    assert latest(FakeFplClient().bootstrap()) == 2


def test_nothing_is_written_when_checks_fail(tmp_path):
    run = import_or_fail("src.pipeline.gameweek", "run")
    result = run(FakeFplClient(drop_team=7), out_dir=tmp_path)
    assert result.ok is False
    assert result.failures
    assert list(tmp_path.rglob("*")) == [], "no output may be written when a check fails"


def test_output_is_written_when_checks_pass(tmp_path):
    run = import_or_fail("src.pipeline.gameweek", "run")
    validate_payload = import_or_fail("src.pipeline.export_web", "validate_payload")
    result = run(FakeFplClient(), out_dir=tmp_path)
    assert result.ok is True, result.failures
    assert result.gameweek == 2
    index = json.loads((tmp_path / "index.json").read_text())
    assert len(index) == 10, "one entry per fixture of the gameweek"
    for entry in index:
        payload = json.loads((tmp_path / entry["path"]).read_text())
        validate_payload(payload)
        assert payload["methods"] == ["baseline"]
        assert payload["source"] == "Fantasy Premier League API"
        starters = [p for p in payload["players"] if p["is_starter"]]
        assert len(starters) == 22
