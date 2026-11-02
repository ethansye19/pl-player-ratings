"""The PROMPT.md rating rules, checked on the real full-season ratings of BOTH methods
(the `method` fixture parametrises every test over baseline and vaep)."""
import numpy as np
import pytest

from src.contract import (
    MAX_SHARE_OF_TENS,
    MEAN_TOLERANCE,
    MIN_MINUTES,
    POSITION_GROUPS,
    RATING_DECIMALS,
    RATING_MAX,
    RATING_MIN,
    TARGET_MEAN,
    validate_contract,
)

pytestmark = pytest.mark.data


def test_raw_output_conforms_to_contract(season_raw):
    validate_contract(season_raw)


def test_rated_output_conforms_to_contract(season_ratings):
    validate_contract(season_ratings, with_rating=True)


def test_only_players_with_10_plus_minutes_are_rated(season_ratings):
    assert (season_ratings["minutes"] >= MIN_MINUTES).all()


def test_every_player_with_10_plus_minutes_is_rated(season_ratings, lineups):
    eligible = lineups[lineups["minutes"] >= MIN_MINUTES]
    expected = set(map(tuple, eligible[["match_id", "player_id"]].to_numpy()))
    got = set(map(tuple, season_ratings[["match_id", "player_id"]].to_numpy()))
    assert got == expected, f"missing {len(expected - got)}, unexpected {len(got - expected)}"


def test_ratings_bounded_3_to_10(season_ratings):
    assert season_ratings["rating"].min() >= RATING_MIN
    assert season_ratings["rating"].max() <= RATING_MAX


def test_season_mean_is_6_5(season_ratings):
    mean = season_ratings["rating"].mean()
    assert abs(mean - TARGET_MEAN) <= MEAN_TOLERANCE, f"season mean {mean:.3f}"


def test_each_position_group_is_anchored_separately(season_ratings):
    means = season_ratings.groupby("position_group")["rating"].mean()
    assert set(means.index) == set(POSITION_GROUPS)
    for group, mean in means.items():
        assert abs(mean - TARGET_MEAN) <= MEAN_TOLERANCE, f"{group} mean {mean:.3f}"


def test_tens_are_extremely_rare(season_ratings):
    share = float((season_ratings["rating"] == RATING_MAX).mean())
    assert share < MAX_SHARE_OF_TENS, f"{share:.4%} of ratings are 10.0"


def test_ratings_have_one_decimal(season_ratings):
    scaled = season_ratings["rating"].to_numpy() * 10 ** RATING_DECIMALS
    assert np.allclose(scaled, np.round(scaled), atol=1e-9)


def test_all_380_matches_are_rated(season_ratings, matches):
    assert set(season_ratings["match_id"]) == set(matches["match_id"])


def test_eleven_rated_starters_per_team_unless_under_10_minutes(season_ratings, lineups):
    starters = lineups[lineups["is_starter"]][["match_id", "team", "player_id", "minutes"]]
    expected = (
        starters[starters["minutes"] >= MIN_MINUTES].groupby(["match_id", "team"]).size().rename("expected")
    )
    rated = season_ratings.merge(starters[["match_id", "player_id"]], on=["match_id", "player_id"])
    got = rated.groupby(["match_id", "team"]).size().rename("got")
    table = expected.to_frame().join(got, how="outer").fillna(0).astype(int)
    assert len(table) == 760
    mismatched = table[table["expected"] != table["got"]]
    assert mismatched.empty, mismatched.head(10).to_dict()
    # Whenever every starter played 10+ minutes, exactly 11 starters are rated.
    all_played = starters.groupby(["match_id", "team"])["minutes"].min() >= MIN_MINUTES
    assert (table.loc[all_played[all_played].index, "got"] == 11).all()
    assert table["got"].max() == 11


def test_rating_is_monotonic_in_raw_score_within_group(season_ratings):
    for group, part in season_ratings.groupby("position_group"):
        part = part.sort_values("raw_score", kind="mergesort")
        assert (np.diff(part["rating"].to_numpy()) >= 0).all(), f"{group}: rating not monotonic in raw_score"
