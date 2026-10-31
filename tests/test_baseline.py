"""Baseline method: 25-35 documented stats, position-specific weights with a one-line
justification each, and raw_score = weighted sum of a player's stats in that match."""
import numpy as np
import pytest

from src.contract import POSITION_GROUPS, validate_contract
from tests.conftest import events_for, import_or_fail, load_parquet


@pytest.fixture(scope="module")
def weights():
    STATS, WEIGHTS = import_or_fail("src.baseline.weights", "STATS", "WEIGHTS")
    return STATS, WEIGHTS


def test_weights_defined_for_every_position_group(weights):
    _, WEIGHTS = weights
    assert set(WEIGHTS) == set(POSITION_GROUPS)
    assert all(len(ws) >= 8 for ws in WEIGHTS.values()), "each group needs a real stat profile"


def test_between_25_and_35_stats(weights):
    STATS, _ = weights
    assert 25 <= len(STATS) <= 35
    assert len(set(STATS)) == len(STATS)


def test_every_weight_has_a_one_line_justification(weights):
    _, WEIGHTS = weights
    for group, ws in WEIGHTS.items():
        for stat, w in ws.items():
            assert isinstance(float(w.weight), float), (group, stat)
            assert isinstance(w.why, str) and len(w.why.strip()) >= 15, (group, stat)
            assert "\n" not in w.why.strip(), f"{group}/{stat}: justification must be one line"


def test_weights_only_reference_known_stats(weights):
    STATS, WEIGHTS = weights
    for group, ws in WEIGHTS.items():
        unknown = set(ws) - set(STATS)
        assert not unknown, f"{group}: {unknown}"


def test_weights_differ_between_position_groups(weights):
    STATS, WEIGHTS = weights
    vectors = {
        g: np.array([float(WEIGHTS[g][s].weight) if s in WEIGHTS[g] else 0.0 for s in STATS]) for g in WEIGHTS
    }
    groups = list(vectors)
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            assert not np.allclose(vectors[g1], vectors[g2]), f"{g1} and {g2} share identical weights"


@pytest.mark.data
def test_rate_match_returns_the_contract(matches, lineups):
    rate_match = import_or_fail("src.baseline.rate", "rate_match")
    match_id = int(matches["match_id"].iloc[0])
    out = rate_match(match_id, events=events_for([match_id]), lineups=lineups[lineups["match_id"] == match_id])
    validate_contract(out)
    appeared = lineups[lineups["match_id"] == match_id]
    assert set(out["player_id"]) == set(appeared["player_id"]), "every player who appeared gets a raw score"


@pytest.mark.data
def test_raw_score_is_the_weighted_sum_of_the_stats(matches, lineups, weights):
    STATS, WEIGHTS = weights
    rate_match = import_or_fail("src.baseline.rate", "rate_match")
    player_match_stats = import_or_fail("src.baseline.stats", "player_match_stats")
    match_id = int(matches["match_id"].iloc[1])
    events = events_for([match_id])
    lu = lineups[lineups["match_id"] == match_id]
    stats = player_match_stats(match_id, events, lu)
    assert set(STATS) <= set(stats.columns)
    out = rate_match(match_id, events=events, lineups=lu).set_index("player_id")
    for _, row in stats.iterrows():
        ws = WEIGHTS[row["position_group"]]
        expected = sum(float(ws[s].weight) * float(row[s]) for s in ws)
        assert abs(out.loc[row["player_id"], "raw_score"] - expected) < 1e-6, row["player_id"]


@pytest.mark.data
def test_cached_stats_table_covers_all_matches_and_stats(matches, weights):
    STATS, _ = weights
    stats = load_parquet("baseline_stats.parquet", "Run `python -m src.baseline.rate` (Step 3) first.")
    assert set(stats["match_id"]) == set(matches["match_id"])
    assert set(STATS) <= set(stats.columns)
    assert not stats[STATS].isna().any().any()
    assert not stats.duplicated(subset=["match_id", "player_id"]).any()
