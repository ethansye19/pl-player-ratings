"""Scale module rules, checked on a SYNTHETIC season in the contract format so the scale
agent can build independently of the rating methods. Every rule in PROMPT.md that
concerns the rating distribution is here; the same rules are re-checked on real season
ratings in test_rules_season.py."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

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
from tests.conftest import import_or_fail


@pytest.fixture(scope="module")
def Scaler():
    return import_or_fail("src.scale.scaler", "PercentileScaler")


@pytest.fixture(scope="module")
def rated(fake_season, Scaler) -> pd.DataFrame:
    return Scaler().fit(fake_season).transform(fake_season)


def test_output_conforms_to_contract_with_rating(rated):
    validate_contract(rated, with_rating=True)


def test_players_under_10_minutes_are_not_rated(rated):
    assert (rated["minutes"] >= MIN_MINUTES).all()
    assert rated["rating"].notna().all()


def test_all_players_with_10_plus_minutes_are_rated(fake_season, rated):
    eligible = fake_season[fake_season["minutes"] >= MIN_MINUTES]
    assert len(eligible) < len(fake_season), "synthetic season should contain sub-10-minute players"
    expected = set(map(tuple, eligible[["match_id", "player_id"]].to_numpy()))
    got = set(map(tuple, rated[["match_id", "player_id"]].to_numpy()))
    assert got == expected


def test_ratings_bounded_3_to_10(rated):
    assert rated["rating"].min() >= RATING_MIN
    assert rated["rating"].max() <= RATING_MAX


def test_season_mean_is_6_5(rated):
    assert abs(rated["rating"].mean() - TARGET_MEAN) <= MEAN_TOLERANCE


def test_each_position_group_mean_is_6_5(rated):
    """Within-group scaling anchors every group separately; a cross-group comparison
    would push the GK group (raw ~N(0,1)) far below the DEF group (raw ~N(50,20))."""
    means = rated.groupby("position_group")["rating"].mean()
    assert set(means.index) == set(POSITION_GROUPS)
    for group, mean in means.items():
        assert abs(mean - TARGET_MEAN) <= MEAN_TOLERANCE, f"{group} mean {mean:.3f}"


def test_tens_are_extremely_rare(rated):
    share = float((rated["rating"] == RATING_MAX).mean())
    assert share < MAX_SHARE_OF_TENS, f"{share:.4%} of ratings are 10.0"


def test_ratings_have_one_decimal(rated):
    scaled = rated["rating"].to_numpy() * 10 ** RATING_DECIMALS
    assert np.allclose(scaled, np.round(scaled), atol=1e-9)


def test_rating_is_monotonic_in_raw_score_within_group(rated):
    for group, part in rated.groupby("position_group"):
        part = part.sort_values("raw_score", kind="mergesort")
        assert (np.diff(part["rating"].to_numpy()) >= 0).all(), f"{group}: rating not monotonic in raw_score"


def test_equal_raw_scores_get_equal_ratings(rated):
    ties = rated[rated["raw_score"] == 0.0]
    assert len(ties) > 50, "synthetic season should contain many exact ties"
    per_group = ties.groupby("position_group")["rating"].nunique()
    assert (per_group == 1).all(), per_group.to_dict()


def test_groups_are_scaled_independently(fake_season, rated, Scaler):
    """Shifting one group's raw scores by a constant must not change any rating."""
    shifted = fake_season.copy()
    mask = shifted["position_group"] == "FWD"
    shifted.loc[mask, "raw_score"] += 1000.0
    rated_shifted = Scaler().fit(shifted).transform(shifted)
    assert_series_equal(rated["rating"].reset_index(drop=True), rated_shifted["rating"].reset_index(drop=True))


def test_same_raw_score_rates_differently_in_different_groups(rated):
    """A raw score of 0.0 is median-ish for the synthetic GK group but far below the
    synthetic DEF group, so it must rate higher for a GK than for a DEF."""
    zero = rated[rated["raw_score"] == 0.0].groupby("position_group")["rating"].first()
    assert zero["GK"] > zero["DEF"]


def test_transform_is_row_wise(fake_season, rated, Scaler):
    """Transforming one match on its own must give the same ratings as transforming
    the whole season (the isolation rule for the scale step)."""
    scaler = Scaler().fit(fake_season)
    for match_id in (1, 57, 200, 380):
        one = scaler.transform(fake_season[fake_season["match_id"] == match_id])
        full = rated[rated["match_id"] == match_id]
        assert_frame_equal(one.reset_index(drop=True), full.reset_index(drop=True))


def test_raw_scores_beyond_the_reference_range_clip_to_the_bounds(fake_season, Scaler):
    scaler = Scaler().fit(fake_season)
    rows = []
    for i, group in enumerate(POSITION_GROUPS):
        ref = fake_season.loc[fake_season["position_group"] == group, "raw_score"]
        rows.append((900_000 + 2 * i, 1, "Synthetic Hi", "Synthetic FC", group, 90.0, ref.max() + 1e6))
        rows.append((900_000 + 2 * i + 1, 1, "Synthetic Lo", "Synthetic FC", group, 90.0, ref.min() - 1e6))
    probe = pd.DataFrame(rows, columns=list(fake_season.columns))
    out = scaler.transform(probe)
    assert (out.loc[out["player_name"] == "Synthetic Hi", "rating"] == RATING_MAX).all()
    assert (out.loc[out["player_name"] == "Synthetic Lo", "rating"] == RATING_MIN).all()


def test_scaler_round_trips_through_save_and_load(fake_season, rated, Scaler, tmp_path):
    path = tmp_path / "scaler.json"
    Scaler().fit(fake_season).save(path)
    reloaded = Scaler.load(path)
    assert_frame_equal(reloaded.transform(fake_season).reset_index(drop=True), rated.reset_index(drop=True))


def test_unknown_position_group_is_rejected(fake_season, Scaler):
    scaler = Scaler().fit(fake_season)
    bad = fake_season.head(3).copy()
    bad["position_group"] = "WING"
    with pytest.raises(ValueError):
        scaler.transform(bad)


def test_fit_rejects_frames_that_break_the_contract(fake_season, Scaler):
    with pytest.raises(ValueError):
        Scaler().fit(fake_season.drop(columns=["raw_score"]))
