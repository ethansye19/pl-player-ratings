"""Baseline stat definitions and the per-match stats table.

The synthetic match below is NOT real data: it is a hand-built set of events whose
expected stat counts are known, so each definition in ``src/baseline/weights.py`` can be
checked exactly (on-pitch goals conceded, clean sheets, aerials, cards, saves, ...).
The data-backed tests reconcile the real stats with the match scores."""
import numpy as np
import pandas as pd
import pytest

from src.contract import CONTRACT_COLUMNS, POSITION_GROUPS, validate_contract
from tests.conftest import events_for, import_or_fail, load_parquet

BOOL_COLUMNS = {
    "pass_goal_assist", "pass_shot_assist", "ball_recovery_recovery_failure", "clearance_aerial_won",
    "pass_aerial_won", "shot_aerial_won", "miscontrol_aerial_won", "foul_committed_penalty", "foul_won_penalty",
    "pass_cross",
}
FLOAT_COLUMNS = {"player_id", "shot_statsbomb_xg", "location_x", "pass_end_x", "carry_end_x"}
HOME, AWAY = 1, 2


def _event(match_id=1, **kw) -> dict:
    """One SYNTHETIC event row with every column the stats module reads."""
    from src.baseline.stats import EVENT_COLUMNS

    row = {}
    for col in EVENT_COLUMNS:
        if col in BOOL_COLUMNS:
            row[col] = False
        elif col in FLOAT_COLUMNS:
            row[col] = np.nan
        elif col in ("match_id", "team_id", "period"):
            row[col] = 0
        else:
            row[col] = None
    row.update(match_id=match_id, period=1, timestamp="00:00:00.000")
    row.update(kw)
    return row


def synthetic_events(match_id: int = 1) -> pd.DataFrame:
    """SYNTHETIC two-period match; period 1 lasts 46:00, period 2 lasts 49:00 (95 minutes).

    Home (team 1): 101 GK, 102 DEF (full match), 103 FWD (sub on at 70.0).
    Away (team 2): 201 GK, 203 MID (full match), 202 DEF (off at 50.0).
    """
    e = [
        _event(match_id, team_id=HOME, type="Starting XI"),
        _event(match_id, team_id=HOME, type="Half Start"),
        # away MID: completed progressive key pass at 10:00, then away DEF scores (xG 0.3)
        _event(match_id, team_id=AWAY, player_id=203, type="Pass", timestamp="00:10:00.000",
               location_x=50.0, pass_end_x=70.0, pass_shot_assist=True),
        _event(match_id, team_id=AWAY, player_id=202, type="Shot", timestamp="00:10:05.000",
               shot_outcome="Goal", shot_type="Open Play", shot_statsbomb_xg=0.3),
        # home DEF: one incomplete pass, one completed aerial pass, one tackle won, a failed interception
        _event(match_id, team_id=HOME, player_id=102, type="Pass", timestamp="00:12:00.000", pass_outcome="Incomplete"),
        _event(match_id, team_id=HOME, player_id=102, type="Pass", timestamp="00:13:00.000", pass_aerial_won=True),
        _event(match_id, team_id=HOME, player_id=102, type="Duel", timestamp="00:14:00.000", duel_type="Tackle", duel_outcome="Won"),
        _event(match_id, team_id=HOME, player_id=102, type="Interception", timestamp="00:15:00.000", interception_outcome="Lost In Play"),
        _event(match_id, team_id=AWAY, player_id=202, type="Duel", timestamp="00:16:00.000", duel_type="Aerial Lost"),
        # away DEF own goal at 20:00 (home benefit)
        _event(match_id, team_id=AWAY, player_id=202, type="Own Goal Against", timestamp="00:20:00.000"),
        _event(match_id, team_id=HOME, type="Own Goal For", timestamp="00:20:00.000"),
        # home GK: a claim and an injury clearance pass (neither completed nor incomplete)
        _event(match_id, team_id=HOME, player_id=101, type="Goal Keeper", timestamp="00:25:00.000",
               goalkeeper_type="Collected", goalkeeper_outcome="Success"),
        _event(match_id, team_id=HOME, player_id=101, type="Pass", timestamp="00:26:00.000", pass_outcome="Injury Clearance"),
        # away MID: progressive carry, a failed and a real recovery, a pressure, a foul with a yellow card
        _event(match_id, team_id=AWAY, player_id=203, type="Carry", timestamp="00:30:00.000", location_x=60.0, carry_end_x=75.0),
        _event(match_id, team_id=AWAY, player_id=203, type="Ball Recovery", timestamp="00:31:00.000", ball_recovery_recovery_failure=True),
        _event(match_id, team_id=AWAY, player_id=203, type="Ball Recovery", timestamp="00:32:00.000"),
        _event(match_id, team_id=AWAY, player_id=203, type="Pressure", timestamp="00:33:00.000"),
        _event(match_id, team_id=AWAY, player_id=203, type="Foul Committed", timestamp="00:34:00.000", foul_committed_card="Yellow Card"),
        # a failed cross is not an incomplete pass
        _event(match_id, team_id=AWAY, player_id=203, type="Pass", timestamp="00:35:00.000", pass_outcome="Incomplete", pass_cross=True),
        # home DEF second yellow (bad behaviour)
        _event(match_id, team_id=HOME, player_id=102, type="Bad Behaviour", timestamp="00:40:00.000", bad_behaviour_card="Second Yellow"),
        _event(match_id, team_id=HOME, type="Half End", timestamp="00:46:00.000"),
        _event(match_id, team_id=AWAY, type="Half End", timestamp="00:46:00.000"),
        # second half: home sub FWD shoots (saved, xG 0.2) at 76.0 abs and misses a penalty at 86.0 abs
        _event(match_id, team_id=HOME, player_id=103, type="Shot", period=2, timestamp="00:30:00.000",
               shot_outcome="Saved", shot_type="Open Play", shot_statsbomb_xg=0.2),
        _event(match_id, team_id=AWAY, player_id=201, type="Goal Keeper", period=2, timestamp="00:30:00.500",
               goalkeeper_type="Shot Saved", goalkeeper_outcome="Success"),
        _event(match_id, team_id=HOME, player_id=103, type="Shot", period=2, timestamp="00:40:00.000",
               shot_outcome="Off T", shot_type="Penalty", shot_statsbomb_xg=0.7835),
        _event(match_id, team_id=HOME, type="Half End", period=2, timestamp="00:49:00.000"),
        _event(match_id, team_id=AWAY, type="Half End", period=2, timestamp="00:49:00.000"),
    ]
    return pd.DataFrame(e)


def synthetic_lineups(match_id: int = 1) -> pd.DataFrame:
    rows = [
        (HOME, "Synthetic Home", 101, "Synthetic GK H", "GK", 95.0, 0.0, 95.0, True),
        (HOME, "Synthetic Home", 102, "Synthetic DEF H", "DEF", 95.0, 0.0, 95.0, True),
        (HOME, "Synthetic Home", 103, "Synthetic FWD H", "FWD", 25.0, 70.0, 95.0, False),
        (AWAY, "Synthetic Away", 201, "Synthetic GK A", "GK", 95.0, 0.0, 95.0, True),
        (AWAY, "Synthetic Away", 202, "Synthetic DEF A", "DEF", 50.0, 0.0, 50.0, True),
        (AWAY, "Synthetic Away", 203, "Synthetic MID A", "MID", 95.0, 0.0, 95.0, True),
    ]
    df = pd.DataFrame(rows, columns=["team_id", "team", "player_id", "player_name", "position_group", "minutes",
                                     "minute_on", "minute_off", "is_starter"])
    df.insert(0, "match_id", match_id)
    df["match_id"] = df["match_id"].astype("int64")
    df["player_id"] = df["player_id"].astype("int64")
    return df


@pytest.fixture(scope="module")
def synthetic_stats() -> pd.DataFrame:
    player_match_stats = import_or_fail("src.baseline.stats", "player_match_stats")
    return player_match_stats(1, synthetic_events(), synthetic_lineups()).set_index("player_id")


def test_stats_table_shape_and_dtypes(synthetic_stats):
    STATS = import_or_fail("src.baseline.weights", "STATS")
    assert list(synthetic_stats.columns) == ["match_id", "position_group", "minutes"] + list(STATS)
    assert list(synthetic_stats.index) == [201, 202, 203, 101, 102, 103]  # team name ("Synthetic Away" first), then player_id
    assert not synthetic_stats.isna().any().any()
    assert synthetic_stats["xg"].dtype == "float64" and synthetic_stats["goals"].dtype == "int64"


def test_attacking_and_possession_definitions(synthetic_stats):
    s = synthetic_stats
    assert s.loc[202, "goals"] == 1 and s.loc[202, "shots_on_target"] == 1 and s.loc[202, "xg"] == pytest.approx(0.3)
    assert s.loc[203, "key_passes"] == 1 and s.loc[203, "progressive_passes"] == 1 and s.loc[203, "passes_completed"] == 1
    assert s.loc[203, "progressive_carries"] == 1
    assert s.loc[103, "shots_on_target"] == 1 and s.loc[103, "penalties_missed"] == 1
    assert s.loc[103, "xg"] == pytest.approx(0.2), "penalty xG is excluded from xg"
    assert s.loc[102, "passes_completed"] == 1 and s.loc[102, "passes_incomplete"] == 1
    assert s.loc[101, "passes_completed"] == 0 and s.loc[101, "passes_incomplete"] == 0, "injury clearances are ignored"
    assert s.loc[203, "passes_incomplete"] == 0, "failed crosses are not counted as incomplete passes"


def test_defensive_and_discipline_definitions(synthetic_stats):
    s = synthetic_stats
    assert s.loc[102, "tackles_won"] == 1 and s.loc[102, "aerials_won"] == 1 and s.loc[102, "interceptions"] == 0
    assert s.loc[202, "aerials_lost"] == 1 and s.loc[202, "own_goals"] == 1
    assert s.loc[203, "ball_recoveries"] == 1 and s.loc[203, "pressures"] == 1
    assert s.loc[203, "fouls_committed"] == 1 and s.loc[203, "yellow_cards"] == 1 and s.loc[203, "red_cards"] == 0
    assert s.loc[102, "red_cards"] == 1 and s.loc[102, "yellow_cards"] == 0, "a second yellow counts as a red card only"
    assert s.loc[101, "keeper_claims"] == 1 and s.loc[201, "saves"] == 1 and s.loc[201, "penalties_saved"] == 0


def test_team_context_counts_only_while_on_the_pitch(synthetic_stats):
    s = synthetic_stats
    # home conceded at 10:05 (before the 70th-minute sub came on); away conceded the own goal at 20:00
    assert s.loc[101, "goals_conceded"] == 1 and s.loc[102, "goals_conceded"] == 1 and s.loc[103, "goals_conceded"] == 0
    assert s.loc[201, "goals_conceded"] == 1 and s.loc[202, "goals_conceded"] == 1 and s.loc[203, "goals_conceded"] == 1
    # xG faced: away DEF left at 50.0, before both home shots (76.0 and 86.0); penalties count
    assert s.loc[201, "xg_faced"] == pytest.approx(0.2 + 0.7835)
    assert s.loc[202, "xg_faced"] == pytest.approx(0.0)
    assert s.loc[101, "xg_faced"] == pytest.approx(0.3) and s.loc[103, "xg_faced"] == pytest.approx(0.0)
    assert (s["clean_sheet"] == 0).all()


def test_clean_sheet_needs_sixty_minutes_and_nothing_conceded():
    player_match_stats = import_or_fail("src.baseline.stats", "player_match_stats")
    events = synthetic_events()
    quiet = events[~events["type"].isin(["Own Goal Against", "Own Goal For"]) & ~(events["shot_outcome"] == "Goal")]
    s = player_match_stats(1, quiet, synthetic_lineups()).set_index("player_id")
    assert (s["goals_conceded"] == 0).all()
    assert s.loc[[101, 102, 201, 203], "clean_sheet"].tolist() == [1, 1, 1, 1]
    assert s.loc[103, "clean_sheet"] == 0 and s.loc[202, "clean_sheet"] == 0, "25 and 50 minutes are under the 60-minute rule"


def test_player_with_no_events_gets_zeros():
    player_match_stats = import_or_fail("src.baseline.stats", "player_match_stats")
    STATS = import_or_fail("src.baseline.weights", "STATS")
    lineups = synthetic_lineups()
    extra = lineups.iloc[[2]].copy()
    extra["player_id"] = 104
    extra["player_name"] = "Synthetic Ghost"
    s = player_match_stats(1, synthetic_events(), pd.concat([lineups, extra], ignore_index=True)).set_index("player_id")
    assert 104 in s.index
    assert (s.loc[104, [c for c in STATS if c not in ("goals_conceded", "xg_faced")]] == 0).all()


def test_other_matches_are_ignored_and_absent_match_raises():
    player_match_stats = import_or_fail("src.baseline.stats", "player_match_stats")
    rate_match = import_or_fail("src.baseline.rate", "rate_match")
    alone = player_match_stats(1, synthetic_events(1), synthetic_lineups(1))
    mixed_events = pd.concat([synthetic_events(2), synthetic_events(1)], ignore_index=True)
    mixed_lineups = pd.concat([synthetic_lineups(2), synthetic_lineups(1)], ignore_index=True)
    pd.testing.assert_frame_equal(player_match_stats(1, mixed_events, mixed_lineups), alone)
    with pytest.raises(ValueError):
        player_match_stats(3, mixed_events, mixed_lineups)
    with pytest.raises(ValueError):
        rate_match(1, synthetic_events(2), synthetic_lineups(2))


def test_rate_match_on_synthetic_data_is_the_weighted_sum():
    rate_match = import_or_fail("src.baseline.rate", "rate_match")
    STATS, WEIGHTS = import_or_fail("src.baseline.weights", "STATS", "WEIGHTS")
    player_match_stats = import_or_fail("src.baseline.stats", "player_match_stats")
    out = rate_match(1, synthetic_events(), synthetic_lineups())
    validate_contract(out)
    assert list(out.columns) == CONTRACT_COLUMNS
    stats = player_match_stats(1, synthetic_events(), synthetic_lineups()).set_index("player_id")
    for row in out.itertuples():
        expected = sum(float(w.weight) * float(stats.loc[row.player_id, s]) for s, w in WEIGHTS[row.position_group].items())
        assert row.raw_score == pytest.approx(expected, abs=1e-9)
    # the away defender scored but also put through his own net and lost an aerial: goal credit dominates
    assert out.set_index("player_id").loc[202, "raw_score"] > 0


def test_group_profiles_are_position_specific():
    STATS, WEIGHTS, LABELS = import_or_fail("src.baseline.weights", "STATS", "WEIGHTS", "LABELS")
    assert set(LABELS) == set(STATS)
    assert {"saves", "penalties_saved", "keeper_claims"} <= set(WEIGHTS["GK"])
    for group in ("DEF", "MID", "FWD"):
        assert "saves" not in WEIGHTS[group]
    for group in POSITION_GROUPS:
        assert WEIGHTS[group]["goals"].weight > 0 and WEIGHTS[group]["red_cards"].weight < 0
        assert WEIGHTS[group]["own_goals"].weight < 0 and WEIGHTS[group]["errors"].weight < 0
    assert WEIGHTS["GK"]["goals_conceded"].weight < WEIGHTS["DEF"]["goals_conceded"].weight < WEIGHTS["MID"]["goals_conceded"].weight < 0
    assert WEIGHTS["GK"]["xg_faced"].weight > 0 > WEIGHTS["DEF"]["xg_faced"].weight


# ------------------------------------------------------------------------------ real data
@pytest.mark.data
def test_goals_and_goals_conceded_reconcile_with_the_match_scores(matches, lineups):
    player_match_stats = import_or_fail("src.baseline.stats", "player_match_stats")
    sample = matches.sort_values("match_id").iloc[::60]  # 7 matches spread over the season
    for m in sample.itertuples():
        events = events_for([m.match_id])
        lu = lineups[lineups["match_id"] == m.match_id]
        stats = player_match_stats(m.match_id, events, lu).merge(lu[["player_id", "team_id"]], on="player_id")
        by_team = stats.groupby("team_id")[["goals", "own_goals"]].sum()
        scores = {m.home_team_id: m.home_score, m.away_team_id: m.away_score}
        for team_id, opp_id in ((m.home_team_id, m.away_team_id), (m.away_team_id, m.home_team_id)):
            assert by_team.loc[team_id, "goals"] + by_team.loc[opp_id, "own_goals"] == scores[team_id], m.match_id
            full = stats[(stats["team_id"] == team_id) & (stats["minutes"] >= lu["minutes"].max() - 0.05)]
            assert (full["goals_conceded"] == scores[opp_id]).all(), f"{m.match_id}: full-match players must be charged every goal"


@pytest.mark.data
def test_cached_tables_are_complete_consistent_and_sorted(lineups):
    STATS = import_or_fail("src.baseline.weights", "STATS")
    stats = load_parquet("baseline_stats.parquet", "Run `python -m src.baseline.rate` first.")
    raw = load_parquet("raw_baseline.parquet", "Run `python -m src.baseline.rate` first.")
    validate_contract(raw)
    assert len(raw) == len(lineups) == len(stats)
    keys = ["match_id", "player_id"]
    assert set(map(tuple, raw[keys].to_numpy())) == set(map(tuple, lineups[keys].to_numpy()))
    assert raw[["match_id", "team", "player_id"]].equals(raw.sort_values(["match_id", "team", "player_id"], kind="mergesort")[["match_id", "team", "player_id"]].reset_index(drop=True))
    assert stats[keys].equals(raw[keys]), "stats and raw tables are aligned row for row"
    counts = [s for s in STATS if s not in ("xg", "xg_faced")]
    assert (stats[counts] >= 0).all().all() and (stats[["xg", "xg_faced"]] >= 0).all().all()
    assert set(stats["clean_sheet"].unique()) <= {0, 1}
    assert (stats["goals"] <= stats["shots_on_target"]).all()
    assert (stats["penalties_saved"] <= stats["saves"]).all()
    assert (stats.loc[stats["clean_sheet"] == 1, "goals_conceded"] == 0).all()
    assert (stats.loc[stats["clean_sheet"] == 1, "minutes"] >= 60).all()
    assert stats["goals"].sum() + stats["own_goals"].sum() == 1026, "2015/16 had 1,026 Premier League goals"
