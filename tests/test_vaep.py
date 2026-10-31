"""VAEP method: documented train/score splits such that no match is scored by a model
trained on it, full-season cross-fitted action values, raw_score = sum of a player's
action values in that match."""
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.contract import SEED, validate_contract
from tests.conftest import PROCESSED, actions_for, import_or_fail, load_parquet, require_file

VAEP_HINT = "Run `.venv/Scripts/python -m src.vaep.train` (Step 3) first."


def test_cv_splits_partition_matches_with_no_leakage():
    make_cv_splits = import_or_fail("src.vaep.splits", "make_cv_splits")
    ids = list(range(1000, 1380))
    folds = make_cv_splits(ids, n_folds=5, seed=SEED)
    assert len(folds) == 5
    scored = Counter()
    for fold in folds:
        assert set(fold.train).isdisjoint(fold.score), fold.name
        assert set(fold.train) | set(fold.score) == set(ids), fold.name
        assert len(fold.score) >= 60, "folds should be roughly equal"
        scored.update(fold.score)
    assert all(scored[i] == 1 for i in ids), "every match is scored exactly once"


def test_cv_splits_are_deterministic():
    make_cv_splits = import_or_fail("src.vaep.splits", "make_cv_splits")
    ids = list(range(1, 381))
    a = make_cv_splits(ids, n_folds=5, seed=SEED)
    b = make_cv_splits(ids, n_folds=5, seed=SEED)
    assert [sorted(f.score) for f in a] == [sorted(f.score) for f in b]


def test_holdout_split_trains_on_earlier_matchweeks_only():
    make_holdout_split = import_or_fail("src.vaep.splits", "make_holdout_split")
    synthetic = pd.DataFrame(
        {"match_id": np.arange(1, 381, dtype="int64"), "match_week": np.repeat(np.arange(1, 39), 10)}
    )
    split = make_holdout_split(synthetic)
    weeks = synthetic.set_index("match_id")["match_week"]
    assert split.train and split.score
    assert weeks[split.train].max() < weeks[split.score].min()
    assert set(split.train) | set(split.score) == set(synthetic["match_id"])
    assert set(split.train).isdisjoint(split.score)


@pytest.mark.data
def test_recorded_splits_exclude_each_scored_match_from_its_training_set(matches):
    splits = json.loads(require_file(PROCESSED / "vaep_splits.json", VAEP_HINT).read_text())
    values = load_parquet("action_values.parquet", VAEP_HINT, columns=["game_id", "fold_id"]).drop_duplicates()
    assert values["game_id"].is_unique, "a match must be scored by exactly one fold"
    folds = {f["fold_id"]: f for f in splits["folds"]}
    for game_id, fold_id in values.itertuples(index=False):
        fold = folds[int(fold_id)]
        assert int(game_id) in set(fold["score"]), (game_id, fold_id)
        assert int(game_id) not in set(fold["train"]), f"match {game_id} scored by a model trained on it"
    hold = splits["holdout"]
    assert set(hold["train"]).isdisjoint(hold["score"])
    assert set(hold["train"]) | set(hold["score"]) == set(matches["match_id"].astype(int))


@pytest.mark.data
def test_action_values_cover_all_matches(matches):
    values = load_parquet("action_values.parquet", VAEP_HINT, columns=["game_id", "action_id", "vaep_value"])
    assert set(values["game_id"]) == set(matches["match_id"])
    assert np.isfinite(values["vaep_value"]).all()
    assert not values.duplicated(subset=["game_id", "action_id"]).any()


@pytest.mark.data
def test_holdout_scores_only_the_holdout_matches():
    splits = json.loads(require_file(PROCESSED / "vaep_splits.json", VAEP_HINT).read_text())
    holdout = load_parquet("raw_vaep_holdout.parquet", VAEP_HINT)
    validate_contract(holdout)
    assert set(holdout["match_id"]) == set(splits["holdout"]["score"])


@pytest.mark.data
def test_raw_score_is_the_sum_of_the_players_action_values(lineups):
    raw = load_parquet("raw_vaep.parquet", VAEP_HINT)
    validate_contract(raw)
    values = load_parquet("action_values.parquet", VAEP_HINT, columns=["game_id", "player_id", "vaep_value"])
    sums = values.groupby(["game_id", "player_id"])["vaep_value"].sum()
    for row in raw.sample(n=200, random_state=SEED).itertuples(index=False):
        expected = float(sums.get((row.match_id, row.player_id), 0.0))
        assert abs(row.raw_score - expected) < 1e-6, (row.match_id, row.player_id)
    assert set(map(tuple, raw[["match_id", "player_id"]].to_numpy())) == set(
        map(tuple, lineups[["match_id", "player_id"]].to_numpy())
    ), "every player who appeared gets a raw score (0.0 if they had no actions)"


@pytest.mark.data
def test_rate_match_returns_the_contract(matches, lineups):
    rate_match, load_model = import_or_fail("src.vaep.rate", "rate_match", "load_model_for_match")
    match_id = int(matches["match_id"].iloc[0])
    out = rate_match(
        match_id,
        actions=actions_for([match_id]),
        lineups=lineups[lineups["match_id"] == match_id],
        model=load_model(match_id),
    )
    validate_contract(out)
    assert set(out["player_id"]) == set(lineups.loc[lineups["match_id"] == match_id, "player_id"])
