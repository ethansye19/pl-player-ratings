"""No rating function may read data from a different match than the one it is rating.

Strategy: pick two consecutive matches of the same team (so they share players).
Rating match A with match B's rows mixed in must give exactly the same output as
rating match A alone; and a per-match call must reproduce the cached season output."""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.contract import CONTRACT_COLUMNS
from tests.conftest import PROCESSED, actions_for, events_for, import_or_fail, rating_hint, require_file

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def match_pair(matches) -> tuple:
    """(A, B): the first two matches of the season's first home team, chronologically."""
    first = matches.sort_values(["match_date", "kick_off", "match_id"]).iloc[0]
    team = first["home_team"]
    own = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    own = own.sort_values(["match_date", "kick_off", "match_id"])
    return int(own.iloc[0]["match_id"]), int(own.iloc[1]["match_id"])


def _match_data(method: str, match_ids) -> pd.DataFrame:
    return events_for(match_ids) if method == "baseline" else actions_for(match_ids)


def _rate(method: str, match_id: int, data: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    if method == "baseline":
        rate_match = import_or_fail("src.baseline.rate", "rate_match")
        out = rate_match(match_id, events=data, lineups=lineups)
    else:
        rate_match, load_model = import_or_fail("src.vaep.rate", "rate_match", "load_model_for_match")
        out = rate_match(match_id, actions=data, lineups=lineups, model=load_model(match_id))
    return out[CONTRACT_COLUMNS].sort_values(["team", "player_id"]).reset_index(drop=True)


def test_rating_one_match_alone_reproduces_the_season_output(method, match_pair, lineups, season_raw):
    a, _ = match_pair
    alone = _rate(method, a, _match_data(method, [a]), lineups[lineups["match_id"] == a])
    cached = season_raw[season_raw["match_id"] == a][CONTRACT_COLUMNS]
    cached = cached.sort_values(["team", "player_id"]).reset_index(drop=True)
    assert_frame_equal(alone, cached, check_exact=False, atol=1e-9)


def test_other_matches_rows_do_not_change_the_rating(method, match_pair, lineups):
    a, b = match_pair
    both_data = _match_data(method, [a, b])
    both_lineups = lineups[lineups["match_id"].isin([a, b])]
    shared = set(both_lineups.loc[both_lineups["match_id"] == a, "player_id"]) & set(
        both_lineups.loc[both_lineups["match_id"] == b, "player_id"]
    )
    assert shared, "the two matches must share players for this test to bite"
    with_noise = _rate(method, a, both_data, both_lineups)
    alone = _rate(method, a, _match_data(method, [a]), lineups[lineups["match_id"] == a])
    assert_frame_equal(with_noise, alone)
    assert set(with_noise["match_id"]) == {a}


def test_rating_a_match_absent_from_the_data_raises(method, match_pair, lineups):
    a, b = match_pair
    with pytest.raises(ValueError):
        _rate(method, a, _match_data(method, [b]), lineups[lineups["match_id"] == b])


def test_scaler_transform_of_one_match_matches_the_season_ratings(method, match_pair, season_raw, season_ratings):
    Scaler = import_or_fail("src.scale.scaler", "PercentileScaler")
    scaler = Scaler.load(require_file(PROCESSED / f"scaler_{method}.json", rating_hint(method)))
    a, _ = match_pair
    one = scaler.transform(season_raw[season_raw["match_id"] == a])
    one = one.sort_values(["team", "player_id"]).reset_index(drop=True)
    full = season_ratings[season_ratings["match_id"] == a]
    full = full[one.columns].sort_values(["team", "player_id"]).reset_index(drop=True)
    assert_frame_equal(one, full, check_dtype=False)
