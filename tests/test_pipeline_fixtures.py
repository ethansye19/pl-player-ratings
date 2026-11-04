"""Per-fixture numbers and clubs in the FPL pipeline: double gameweeks, transfers, and the check that a
player's per-fixture rows add up to his gameweek totals. Network-free: every payload here is SYNTHETIC,
shaped like the real bootstrap-static, fixtures, event/<gw>/live and element-summary responses."""
import copy
import json

import pytest

from src.pipeline.gameweek import run
from tests.test_pipeline import XI, FakeFplClient, fake_history_rows

GW = 2
XI_SECOND = [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]  # squad slots that start the extra fixture: rotation against XI
SUB_SECOND = {2: 20}  # slot 2 started the first fixture and comes off the bench for 20 minutes in the second


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _second_fixture_stats(slot: int, etype: int, conceded: int) -> dict:
    """SYNTHETIC numbers for the extra fixture: who plays, and a few stats FPL itemises only when they score points."""
    minutes = 90 if slot in XI_SECOND else SUB_SECOND.get(slot, 0)
    if not minutes:
        return {"minutes": 0, "starts": 0}
    return {"minutes": minutes, "starts": int(slot in XI_SECOND), "saves": 2 if etype == 1 else 0,  # 2 saves earn no FPL points
            "goals_conceded": conceded, "clean_sheets": int(conceded == 0 and minutes >= 60), "tackles": 1, "recoveries": 2,
            "clearances_blocks_interceptions": 1, "expected_goals": "0.10", "expected_goals_conceded": "0.45"}


def _history_row(element: dict, fixture: dict, gw: int, stats: dict) -> dict:
    was_home = fixture["team_h"] == element["team"]
    row = {"element": element["id"], "fixture": fixture["id"], "round": gw, "kickoff_time": fixture["kickoff_time"],
           "opponent_team": fixture["team_a"] if was_home else fixture["team_h"], "was_home": was_home,
           "team_h_score": fixture["team_h_score"], "team_a_score": fixture["team_a_score"], "modified": False,
           "value": 50, "selected": 1000, "transfers_balance": 0, "transfers_in": 0, "transfers_out": 0}
    row.update({k: v for k, v in stats.items() if k not in ("in_dreamteam", "played")})
    return row


def double_gameweek_client() -> tuple[FakeFplClient, dict]:
    """A SYNTHETIC season in which, in gameweek 2, the home sides of the first two fixtures meet again."""
    client = FakeFplClient()
    fixtures, live, boot = client._fixtures, client._live, client._bootstrap
    a, b = fixtures[GW][0], fixtures[GW][1]
    club_x, club_y = a["team_h"], b["team_h"]
    extra = {"id": max(f["id"] for g in fixtures for f in fixtures[g]) + 1, "event": GW, "team_h": club_x, "team_a": club_y,
             "finished": True, "team_h_score": 0, "team_a_score": 1, "kickoff_time": "2026-10-15T19:45:00Z"}
    fixtures[GW].append(extra)
    first_of = {club_x: a, club_y: b}
    elements = {e["id"]: e for e in boot["elements"]}
    for item in live[GW]["elements"]:
        element = elements[item["id"]]
        if element["team"] not in first_of:
            continue
        first = copy.deepcopy(item["stats"])
        conceded = 1 if element["team"] == club_x else 0
        second = dict.fromkeys(first, 0) | {"expected_goals": "0.00", "expected_assists": "0.00", "expected_goal_involvements": "0.00",
                                            "expected_goals_conceded": "0.00", "influence": "0.0", "creativity": "0.0", "threat": "0.0",
                                            "ict_index": "0.0", "in_dreamteam": False} | _second_fixture_stats(element["squad_slot"], element["element_type"], conceded)
        total = {}
        for k, v in first.items():
            if isinstance(v, str):
                total[k] = _fmt(float(v) + float(second[k]))
            elif isinstance(v, bool):
                total[k] = v
            else:
                total[k] = v + second[k]
        item["stats"] = total
        item["explain"] = item["explain"] + [{"fixture": extra["id"], "stats": [{"identifier": "minutes", "points": 2 if second["minutes"] else 0, "value": second["minutes"]}]}]
        earlier = fake_history_rows(element, {1: fixtures[1]}, {1: live[1]})
        client.histories[element["id"]] = {"fixtures": [], "history_past": [], "history": earlier + [
            _history_row(element, first_of[element["team"]], GW, first), _history_row(element, extra, GW, second)]}
    by_slot = {(e["team"], e["squad_slot"]): e["id"] for e in boot["elements"]}
    return client, {"x": club_x, "y": club_y, "first": a["id"], "extra": extra["id"], "by_slot": by_slot}


def _rows(result, player_id):
    r = result.ratings
    return r[r["player_id"] == player_id].set_index("match_id")


def _payload(out_dir, fixture_id):
    return json.loads(next(out_dir.rglob(f"{fixture_id}.json")).read_text())


# ------------------------------------------------------------------------- single gameweek
def test_a_normal_single_gameweek_needs_no_element_summary(tmp_path):
    client = FakeFplClient()
    result = run(client, out_dir=tmp_path)
    assert result.ok, result.failures
    assert client.summary_calls == [], "one fixture per player, for his club: the live feed is enough"
    assert set(result.ratings["per_fixture_source"]) == {"live"}


def test_a_missing_live_field_fails_instead_of_counting_as_zero(tmp_path):
    client = FakeFplClient()
    victim = next(e for e in client._live[GW]["elements"] if e["stats"]["minutes"] > 0)
    del victim["stats"]["saves"]
    result = run(client, out_dir=tmp_path)
    assert not result.ok and any(f"(player {victim['id']}): 'saves' is missing" in f for f in result.failures), result.failures
    assert list(tmp_path.rglob("*")) == []


# ------------------------------------------------------------------------- double gameweek
def test_a_double_gameweek_takes_starts_minutes_and_stats_per_fixture(tmp_path):
    client, dg = double_gameweek_client()
    result = run(client, out_dir=tmp_path)
    assert result.ok, result.failures
    x, first, extra, slot = dg["x"], dg["first"], dg["extra"], dg["by_slot"]

    rotated = _rows(result, slot[(x, 2)])  # started the first fixture, came off the bench in the second
    assert bool(rotated.loc[first, "is_starter"]) is True and bool(rotated.loc[extra, "is_starter"]) is False
    assert rotated.loc[extra, "minutes"] == 20.0
    promoted = _rows(result, slot[(x, 6)])  # a substitute in the first fixture, a starter in the second
    assert bool(promoted.loc[first, "is_starter"]) is False and bool(promoted.loc[extra, "is_starter"]) is True
    keeper = _rows(result, slot[(x, 0)])
    first_saves = next(e for e in client.histories[slot[(x, 0)]]["history"] if e["fixture"] == first)["saves"]
    assert keeper.loc[extra, "saves"] == 2 and keeper.loc[first, "saves"] == first_saves, "per fixture, not split"
    assert keeper.loc[extra, "goals_conceded"] == 1 and keeper.loc[extra, "xg_faced"] == pytest.approx(0.45)
    assert set(result.ratings.loc[result.ratings["match_id"] == extra, "per_fixture_source"]) == {"element-summary"}

    for fixture_id in (first, extra):
        payload = _payload(tmp_path, fixture_id)
        for side in ("home", "away"):
            assert sum(p["is_starter"] for p in payload["players"] if p["side"] == side) == 11
    # element-summary only for the two clubs that played twice, once per player across both gameweeks of the run
    needed = {e["id"] for e in client._bootstrap["elements"] if e["team"] in (dg["x"], dg["y"])}
    assert set(client.summary_calls) <= needed and len(client.summary_calls) == len(set(client.summary_calls))


def test_a_player_who_misses_one_of_two_fixtures_has_one_row(tmp_path):
    client, dg = double_gameweek_client()
    result = run(client, out_dir=tmp_path)
    assert result.ok, result.failures
    x, first, extra, slot = dg["x"], dg["first"], dg["extra"], dg["by_slot"]
    rested = _rows(result, slot[(x, 7)])  # started the first fixture, did not play the second
    assert list(rested.index) == [first] and bool(rested.loc[first, "is_starter"]) is True
    only_second = _rows(result, slot[(x, 14)])  # never played the first fixture, started the second
    assert list(only_second.index) == [extra] and only_second.loc[extra, "minutes"] == 90.0


def test_per_fixture_rows_that_do_not_add_up_fail_the_run(tmp_path):
    client, dg = double_gameweek_client()
    pid = dg["by_slot"][(dg["x"], 2)]
    next(item for item in client._live[GW]["elements"] if item["id"] == pid)["stats"]["minutes"] += 5
    result = run(client, out_dir=tmp_path)
    assert not result.ok
    assert any(f"(player {pid}): the per-fixture 'minutes' add up to" in f for f in result.failures), result.failures
    assert list(tmp_path.rglob("*")) == [], "nothing is written"


@pytest.mark.parametrize("damage, message", [
    ("fetch fails", "element-summary is needed (more than one fixture in the gameweek) but could not be read"),
    ("row missing", "element-summary has no history row for fixture"),
    ("field missing", "'saves' is missing"),
    ("field null", "'tackles' is missing"),
    ("wrong opponent", "does not match fixture"),
])
def test_missing_or_inconsistent_history_fails_the_run(tmp_path, damage, message):
    client, dg = double_gameweek_client()
    pid = dg["by_slot"][(dg["x"], 0)]
    history = client.histories[pid]["history"]
    extra_row = next(h for h in history if h["fixture"] == dg["extra"])
    if damage == "fetch fails":
        client.histories[pid] = RuntimeError("FPL API request failed: timeout")
    elif damage == "row missing":
        history.remove(extra_row)
    elif damage == "field missing":
        del extra_row["saves"]
    elif damage == "field null":
        extra_row["tackles"] = None
    else:
        extra_row["opponent_team"] = 99
    result = run(client, out_dir=tmp_path)
    assert not result.ok
    assert any(f"(player {pid})" in f and message in f for f in result.failures), result.failures
    assert list(tmp_path.rglob("*")) == [], "nothing is written"


# ------------------------------------------------------------------------- transfers
@pytest.mark.parametrize("future_club", ["another club", "the opponent"])
def test_a_transferred_player_counts_for_the_club_he_played_for(tmp_path, future_club):
    """He started gameweek 1 for his old club and moved afterwards; bootstrap-static only knows the new club."""
    client = FakeFplClient(gameweeks=(1,))
    fixture = client._fixtures[1][0]
    old_club, opponent = fixture["team_h"], fixture["team_a"]
    player = next(e for e in client._bootstrap["elements"] if e["team"] == old_club and e["squad_slot"] == XI[3])
    history = fake_history_rows(player, client._fixtures, client._live)  # written while he was at his old club
    client.histories[player["id"]] = {"fixtures": [], "history": history, "history_past": []}
    new_club = opponent if future_club == "the opponent" else next(t["id"] for t in client._bootstrap["teams"] if t["id"] not in (old_club, opponent))
    player["team"], player["team_join_date"] = new_club, "2026-10-20"
    result = run(client, out_dir=tmp_path, gw=1)
    assert result.ok, result.failures
    assert client.summary_calls == [player["id"]]
    row = _rows(result, player["id"]).loc[fixture["id"]]
    assert row["team_id"] == old_club and row["side"] == "home" and row["per_fixture_source"] == "element-summary"
    payload = _payload(tmp_path, fixture["id"])
    assert next(p for p in payload["players"] if p["player_id"] == player["id"])["team_id"] == old_club
    assert sum(p["is_starter"] for p in payload["players"] if p["side"] == "home") == 11
