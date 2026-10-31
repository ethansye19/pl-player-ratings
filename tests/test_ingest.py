"""Step 2: the processed 2015/16 data. 380 matches, 20 teams, lineups with minutes and
position groups, flattened events and SPADL actions covering every match."""
import pandas as pd
import pytest

from src.contract import MIN_MINUTES, POSITION_GROUPS
from tests.conftest import INGEST_HINT, PROCESSED, load_parquet, require_file

pytestmark = pytest.mark.data


def test_380_matches(matches):
    assert len(matches) == 380
    assert matches["match_id"].is_unique
    assert set(matches["match_week"]) == set(range(1, 39))


def test_20_teams(matches, lineups):
    teams_in_matches = set(matches["home_team"]) | set(matches["away_team"])
    assert len(teams_in_matches) == 20
    assert set(lineups["team"]) == teams_in_matches
    assert load_parquet("teams.parquet", INGEST_HINT)["team_id"].nunique() == 20


def test_every_team_plays_38_matches(matches):
    played = pd.concat([matches["home_team"], matches["away_team"]]).value_counts()
    assert (played == 38).all(), played[played != 38].to_dict()


def test_lineups_cover_every_match_with_two_teams(matches, lineups):
    assert set(lineups["match_id"]) == set(matches["match_id"])
    teams_per_match = lineups.groupby("match_id")["team"].nunique()
    assert (teams_per_match == 2).all()


def test_eleven_starters_per_team_per_match(lineups):
    starters = lineups[lineups["is_starter"]].groupby(["match_id", "team"]).size()
    assert len(starters) == 760
    assert (starters == 11).all(), starters[starters != 11].to_dict()


def test_position_groups_are_valid(lineups):
    assert set(lineups["position_group"].unique()) <= set(POSITION_GROUPS)
    starters = lineups[lineups["is_starter"]]
    one_gk = starters[starters["position_group"] == "GK"].groupby(["match_id", "team"]).size()
    assert len(one_gk) == 760 and (one_gk == 1).all(), "every starting XI has exactly one goalkeeper"


def test_minutes_are_plausible(lineups):
    """Only players who appeared are listed; minutes include stoppage time but a
    Premier League match has no extra time."""
    assert (lineups["minutes"] > 0).all()
    assert lineups["minutes"].max() <= 105
    assert (lineups["minutes"].round(1) == lineups["minutes"]).all()
    starters = lineups[lineups["is_starter"]]
    full_xi = (starters["minutes"] >= MIN_MINUTES).groupby([starters["match_id"], starters["team"]]).all()
    assert full_xi.mean() >= 0.95, "early injuries/red cards are rare; most XIs play 10+ minutes"


def test_lineups_have_one_row_per_player_per_match(lineups):
    assert not lineups.duplicated(subset=["match_id", "player_id"]).any()


def test_events_cover_all_matches(matches):
    ids = load_parquet("events.parquet", INGEST_HINT, columns=["match_id"])
    assert set(ids["match_id"]) == set(matches["match_id"])
    assert len(ids) > 380 * 2000, "a Premier League match has well over 2000 StatsBomb events"


def test_event_players_appear_in_the_lineups(lineups):
    ev = load_parquet("events.parquet", INGEST_HINT, columns=["match_id", "player_id"]).dropna()
    ev["player_id"] = ev["player_id"].astype("int64")
    known = set(map(tuple, lineups[["match_id", "player_id"]].to_numpy()))
    seen = set(map(tuple, ev.drop_duplicates().to_numpy()))
    assert seen <= known, f"{len(seen - known)} (match, player) pairs in events are missing from lineups"


def test_actions_cover_all_matches(matches):
    actions = load_parquet("actions.parquet", INGEST_HINT, columns=["game_id", "action_id", "type_id", "result_id"])
    assert set(actions["game_id"]) == set(matches["match_id"])
    assert not actions.duplicated(subset=["game_id", "action_id"]).any()


def test_raw_statsbomb_json_was_kept(matches):
    raw = require_file(PROCESSED.parent / "raw" / "statsbomb", "Ingestion must keep the downloaded JSON.")
    require_file(raw / "matches" / "2" / "27.json", "competition 2 / season 27 match list")
    for match_id in matches["match_id"].head(3):
        require_file(raw / "events" / f"{match_id}.json", "per-match events")
        require_file(raw / "lineups" / f"{match_id}.json", "per-match lineups")
