"""Unit tests for the VAEP module on SYNTHETIC data (no processed data or trained model needed).

The split helpers are tested on made-up match ids; ``rate_match`` is tested with a fake
model object that stands in for a fitted socceraction ``VAEP`` (it only needs ``.rate``).
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.contract import CONTRACT_COLUMNS, SEED, validate_contract
from src.vaep.rate import action_values_for_match, rate_match
from src.vaep.splits import Split, fold_for_match, make_cv_splits, make_holdout_split, splits_payload


# ------------------------------------------------------------------------- splits
def test_cv_splits_handle_fold_counts_that_do_not_divide_evenly():
    folds = make_cv_splits(range(7), n_folds=3, seed=SEED)
    assert sorted(len(f.score) for f in folds) == [2, 2, 3]
    assert len({f.name for f in folds}) == 3
    for fold in folds:
        assert sorted(fold.train + fold.score) == list(range(7))


def test_cv_splits_reject_duplicates_and_impossible_fold_counts():
    with pytest.raises(ValueError):
        make_cv_splits([1, 2, 2, 3], n_folds=2)
    with pytest.raises(ValueError):
        make_cv_splits([1, 2, 3], n_folds=4)
    with pytest.raises(ValueError):
        make_cv_splits([1, 2, 3], n_folds=1)


def test_cv_splits_depend_on_the_seed_but_not_on_the_input_order():
    ids = list(range(100, 480))
    a = make_cv_splits(ids, seed=SEED)
    b = make_cv_splits(ids[::-1], seed=SEED)
    c = make_cv_splits(ids, seed=SEED + 1)
    assert [f.score for f in a] == [f.score for f in b]
    assert [f.score for f in a] != [f.score for f in c]


def test_holdout_split_boundary_is_matchweek_19_inclusive():
    synthetic = pd.DataFrame({"match_id": [1, 2, 3, 4], "match_week": [1, 19, 20, 38]})
    split = make_holdout_split(synthetic)
    assert split.name == "holdout"
    assert split.train == [1, 2] and split.score == [3, 4]


def test_holdout_split_needs_both_sides_and_the_right_columns():
    with pytest.raises(ValueError):
        make_holdout_split(pd.DataFrame({"match_id": [1, 2], "match_week": [3, 4]}))
    with pytest.raises(ValueError):
        make_holdout_split(pd.DataFrame({"match_id": [1, 2], "week": [3, 30]}))


def test_split_rejects_overlap_and_empty_sides():
    with pytest.raises(ValueError):
        Split("bad", train=[1, 2], score=[2, 3])
    with pytest.raises(ValueError):
        Split("bad", train=[1, 2], score=[])


def test_splits_payload_has_the_documented_layout_and_plain_ints():
    holdout = Split("holdout", train=[np.int64(1), np.int64(2)], score=[np.int64(3)])
    folds = make_cv_splits([1, 2, 3, 4], n_folds=2)
    payload = splits_payload(holdout, folds)
    assert set(payload) == {"holdout", "folds"}
    assert set(payload["holdout"]) == {"train", "score"}
    assert [set(f) for f in payload["folds"]] == [{"fold_id", "train", "score"}] * 2
    assert [f["fold_id"] for f in payload["folds"]] == [0, 1]
    assert all(type(m) is int for f in payload["folds"] for m in f["train"] + f["score"])
    assert json.loads(json.dumps(payload)) == payload
    for match_id in (1, 2, 3, 4):
        fold = fold_for_match(payload, match_id)
        assert match_id in fold["score"] and match_id not in fold["train"]
    with pytest.raises(ValueError):
        fold_for_match(payload, 99)


# --------------------------------------------------------------------- rate_match
class _FakeModel:
    """SYNTHETIC stand-in for a fitted socceraction VAEP: every action is worth (action_id + 1) / 100."""

    def __init__(self):
        self.calls = []

    def rate(self, game, game_actions):
        self.calls.append((int(game["game_id"]), int(game["home_team_id"]), list(game_actions.index)))
        value = (game_actions["action_id"].to_numpy(dtype="float64") + 1) / 100
        return pd.DataFrame(
            {"offensive_value": value, "defensive_value": np.zeros_like(value), "vaep_value": value},
            index=game_actions.index,
        )


def _synthetic_match(match_id: int, offset: int) -> tuple:
    """Actions and lineups of one synthetic match; player 3 appears but never acts."""
    players = [offset + 1, offset + 2, offset + 3, offset + 4]
    actions = pd.DataFrame(
        {
            "game_id": [match_id] * 4,
            "action_id": [2, 0, 1, 3],  # deliberately out of order
            "player_id": [float(players[1]), float(players[0]), float(players[0]), float(players[3])],
            "team_id": [10, 10, 10, 20],
            "period_id": [1, 1, 1, 2],
            "time_seconds": [30.0, 1.0, 12.5, 100.0],
            "type_name": ["pass", "pass", "dribble", "shot"],
            "result_name": ["success", "success", "success", "fail"],
        },
        index=[7, 8, 9, 10],  # a non-RangeIndex, as a slice of the season frame would have
    )
    lineups = pd.DataFrame(
        {
            "match_id": [match_id] * 4,
            "team_id": [10, 10, 10, 20],
            "team": ["Synthetic Home FC", "Synthetic Home FC", "Synthetic Home FC", "Synthetic Away FC"],
            "player_id": players,
            "player_name": [f"Synthetic Player {p}" for p in players],
            "position_group": ["GK", "DEF", "MID", "FWD"],
            "minutes": [95.0, 90.5, 3.2, 95.0],
        }
    )
    return actions, lineups


def test_rate_match_sums_the_players_values_and_zero_fills_players_without_actions():
    actions, lineups = _synthetic_match(1, offset=0)
    model = _FakeModel()
    out = rate_match(1, actions, lineups, model, home_team_id=10)
    validate_contract(out)
    assert list(out.columns) == CONTRACT_COLUMNS
    assert out["player_id"].dtype == "int64" and out["match_id"].dtype == "int64"
    scores = dict(zip(out["player_id"], out["raw_score"]))
    assert scores == pytest.approx({1: 0.01 + 0.02, 2: 0.03, 3: 0.0, 4: 0.04})
    # sorted by match_id, team, player_id and the actions were handed over in order on a RangeIndex
    assert list(out["team"]) == ["Synthetic Away FC"] + ["Synthetic Home FC"] * 3
    assert list(out["player_id"]) == [4, 1, 2, 3]
    assert model.calls == [(1, 10, [0, 1, 2, 3])]


def test_rate_match_ignores_other_matches_and_raises_when_the_match_is_absent():
    actions_a, lineups_a = _synthetic_match(1, offset=0)
    actions_b, lineups_b = _synthetic_match(2, offset=2)  # players 3 and 4 appear in both matches
    model = _FakeModel()
    alone = rate_match(1, actions_a, lineups_a, model, home_team_id=10)
    mixed = rate_match(1, pd.concat([actions_b, actions_a]), pd.concat([lineups_b, lineups_a]), model, home_team_id=10)
    pd.testing.assert_frame_equal(alone, mixed)
    assert set(mixed["match_id"]) == {1}
    with pytest.raises(ValueError):
        rate_match(1, actions_b, lineups_b, model, home_team_id=10)
    with pytest.raises(ValueError):  # lineups present but no actions for the match
        rate_match(1, actions_b, lineups_a, model, home_team_id=10)


def test_action_values_for_match_returns_one_row_per_action_in_order():
    actions, _ = _synthetic_match(1, offset=0)
    out = action_values_for_match(1, actions, _FakeModel(), home_team_id=10)
    assert list(out["action_id"]) == [0, 1, 2, 3]
    assert out["player_id"].dtype == "int64"
    assert list(out.columns) == [
        "game_id", "action_id", "player_id", "team_id", "period_id", "time_seconds", "type_name", "result_name",
        "offensive_value", "defensive_value", "vaep_value",
    ]
    assert out["vaep_value"].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04])
