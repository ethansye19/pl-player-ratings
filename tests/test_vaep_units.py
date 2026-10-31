"""Unit tests for the VAEP module on SYNTHETIC data (no processed data or trained model needed).

The split helpers are tested on made-up match ids; ``rate_match`` is tested with a fake
model object that stands in for a fitted socceraction ``VAEP`` (it only needs ``.rate``).
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.contract import CONTRACT_COLUMNS, SEED, validate_contract
from src.vaep.labels import outcome_labels, shot_goal_probability, xg_labels
from src.vaep.rate import action_values_for_match, rate_match
from src.vaep.splits import Split, fold_for_match, make_cv_splits, make_holdout_split, splits_payload

RESULT_IDS = {"fail": 0, "success": 1, "offside": 2, "owngoal": 3}  # socceraction.spadl.config.results order


def _actions(rows) -> pd.DataFrame:
    """SYNTHETIC one-game action frame from (team_id, type_name, result_name, event_id) tuples."""
    frame = pd.DataFrame(rows, columns=["team_id", "type_name", "result_name", "original_event_id"])
    frame["result_id"] = frame["result_name"].map(RESULT_IDS)
    frame["game_id"] = 1
    frame["action_id"] = range(len(frame))
    return frame


# ------------------------------------------------------------------------- labels
def test_xg_labels_with_outcomes_reproduce_socceraction_labels():
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(300):
        team = int(rng.integers(1, 3))
        kind = rng.choice(["pass", "pass", "dribble", "shot", "shot_penalty", "shot_freekick"], p=[0.4, 0.25, 0.15, 0.12, 0.04, 0.04])
        if kind.startswith("shot"):
            result = rng.choice(["fail", "success", "owngoal"], p=[0.6, 0.3, 0.1])  # socceraction only sees own goals on shots
        else:
            result = "success"
        rows.append((team, kind, result, f"ev{i}"))
    actions = _actions(rows)
    soft, coverage = xg_labels(actions, pd.Series(dtype="float64"))  # no xG anywhere -> outcome fallback
    binary = outcome_labels(actions)
    assert coverage["shots_without_xg"] == coverage["shots"] > 0
    assert set(np.unique(soft.to_numpy())) <= {0.0, 1.0}
    assert np.array_equal(soft["scores"].to_numpy() == 1.0, binary["scores"].to_numpy())
    assert np.array_equal(soft["concedes"].to_numpy() == 1.0, binary["concedes"].to_numpy())


def test_xg_labels_combine_chances_within_the_window_and_respect_teams():
    actions = _actions(
        [(1, "pass", "success", "e0"), (1, "shot", "fail", "e1"), (2, "pass", "success", "e2"),
         (1, "shot", "success", "e3"), (2, "shot", "fail", "e4")]
    )
    xg = pd.Series({"e1": 0.3, "e3": 0.5, "e4": 0.2})
    soft, coverage = xg_labels(actions, xg)
    assert coverage == {"shots": 3, "shots_without_xg": 0, "own_goals": 0}
    # team 1's first action: its two shots (0.3, 0.5) combine; team 2's shot (0.2) is a conceding chance
    assert soft.loc[0, "scores"] == pytest.approx(1 - 0.7 * 0.5)
    assert soft.loc[0, "concedes"] == pytest.approx(0.2)
    # the goal itself is worth its xG, not 1.0; team 2's pass sees only team 2's chance
    assert soft.loc[3, "scores"] == pytest.approx(0.5)
    assert soft.loc[2, "scores"] == pytest.approx(0.2) and soft.loc[2, "concedes"] == pytest.approx(0.5)
    # the last action's own shot counts for itself and nothing after it exists
    assert soft.loc[4, "scores"] == pytest.approx(0.2) and soft.loc[4, "concedes"] == 0.0


def test_xg_labels_window_is_ten_actions_and_own_goals_count_one():
    rows = [(1, "pass", "success", f"p{i}") for i in range(12)]
    rows[9] = (1, "shot", "fail", "late")  # 10th action of the window of action 0 -> inside
    rows[11] = (2, "bad_touch", "owngoal", "og")  # own goal by team 2: 1.0 for team 1, conceded by team 2
    rows.insert(0, (1, "shot", "fail", "first"))  # shifts everything by one: 'late' becomes action 10 -> outside for action 0
    actions = _actions(rows)
    soft, coverage = xg_labels(actions, pd.Series({"first": 0.1, "late": 0.4}))
    assert coverage["own_goals"] == 1
    assert soft.loc[0, "scores"] == pytest.approx(0.1)  # 'late' (index 10) is outside action 0's window
    assert soft.loc[1, "scores"] == pytest.approx(0.4)  # ... but inside action 1's
    assert soft.loc[3, "scores"] == pytest.approx(1.0)  # team 1 state 10 actions before the own goal benefits fully
    assert soft.loc[2, "scores"] == pytest.approx(0.4)  # one action earlier the own goal is outside the window
    assert soft.loc[12, "concedes"] == pytest.approx(1.0)  # team 2's own action: an own goal is a goal conceded
    assert soft.loc[12, "scores"] == 0.0


def test_shot_goal_probability_falls_back_to_the_outcome_only_without_xg():
    actions = _actions([(1, "shot", "success", "known"), (1, "shot", "success", "unknown"), (1, "shot", "fail", "none")])
    prob, coverage = shot_goal_probability(actions, pd.Series({"known": 0.25}))
    assert prob.tolist() == pytest.approx([0.25, 1.0, 0.0])
    assert coverage == {"shots": 3, "shots_without_xg": 2, "own_goals": 0}


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
