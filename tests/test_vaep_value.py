"""Option 3 (src/vaep/value.py): socceraction's VAEP formula plus the penalty override and the
after-shot rule. The unit tests use SYNTHETIC action sequences and probabilities; the data tests
check the season outputs and that rating one match alone still reproduces them."""
import numpy as np
import pandas as pd
import pytest
from socceraction.vaep import formula

from src.contract import SEED
from src.vaep.value import PENALTY_PRIOR, PENALTY_XG_FALLBACK, SHOT_TYPES, formula_value, value
from tests.conftest import INGEST_HINT, actions_for, load_parquet

VAEP_HINT = "Run `python -m src.vaep.score` (or src.vaep.train) first."


def _game(rows):
    """SYNTHETIC one-game action frame from (team_id, seconds, type_name, result_name) tuples."""
    return pd.DataFrame(rows, columns=["team_id", "time_seconds", "type_name", "result_name"])


def _random_game(rng, n=400, with_shots=True):
    """SYNTHETIC game: random actions, teams, gaps (some over 10 s), shots, penalties, corners and saves."""
    types = ["pass", "dribble", "cross", "clearance", "tackle", "interception", "corner_crossed", "corner_short", "foul", "keeper_save"]
    if with_shots:
        types += list(SHOT_TYPES) * 2
    rows, t = [], 0.0
    for _ in range(n):
        t += float(rng.choice([1.0, 3.0, 12.0], p=[0.6, 0.3, 0.1]))
        kind = str(rng.choice(types))
        rows.append((int(rng.choice([1, 2])), t, kind, str(rng.choice(["success", "fail"]))))
    game = _game(rows)
    p_s, p_c = rng.uniform(0.0, 0.6, n), rng.uniform(0.0, 0.3, n)
    xg = np.where(game["type_name"].isin(SHOT_TYPES), rng.uniform(0.02, 0.9, n), np.nan)
    return game, p_s, p_c, xg


def test_formula_value_equals_socceraction():
    rng = np.random.default_rng(SEED)
    for _ in range(5):
        game, p_s, p_c, _ = _random_game(rng)
        ours = formula_value(game, p_s, p_c)
        theirs = formula.value(game, pd.Series(p_s), pd.Series(p_c))
        np.testing.assert_allclose(ours.to_numpy(), theirs[["offensive_value", "defensive_value", "vaep_value"]].to_numpy(), atol=1e-12)


def test_option_3_equals_socceraction_when_a_game_has_no_shots():
    rng = np.random.default_rng(SEED + 1)
    game, p_s, p_c, xg = _random_game(rng, with_shots=False)
    np.testing.assert_allclose(value(game, p_s, p_c, xg).to_numpy(), formula.value(game, pd.Series(p_s), pd.Series(p_c)).to_numpy(), atol=1e-12)


@pytest.mark.parametrize("result", ["success", "fail"])
def test_a_penalty_is_valued_at_its_xg_whatever_happens(result):
    """The models under-predict penalty states (0.50 here); the override uses the penalty's xG instead."""
    game = _game([(2, 10.0, "foul", "success"), (1, 70.0, "shot_penalty", result), (2, 75.0, "pass", "success")])
    p_s, p_c = [0.05, 0.50, 0.02], [0.01, 0.01, 0.01]
    for xg, expected in (([np.nan, 0.7835, np.nan], 0.7835), (None, PENALTY_XG_FALLBACK), ([np.nan, 0.80, np.nan], 0.80)):
        out = value(game, p_s, p_c, xg)
        assert out["offensive_value"].iloc[1] == pytest.approx(expected - PENALTY_PRIOR)
    assert formula.value(game, pd.Series(p_s), pd.Series(p_c))["offensive_value"].iloc[1] == pytest.approx(0.50 - PENALTY_PRIOR)


def test_the_action_after_a_missed_shot_does_not_inherit_the_whole_chance():
    """A 0.5-xG miss in a state the model values at 0.55: only the 0.10 left after the shot is at stake."""
    game = _game([(1, 10.0, "shot", "fail"), (2, 12.0, "clearance", "success"), (2, 14.0, "pass", "success")])
    p_s, p_c, xg = [0.55, 0.01, 0.02], [0.01, 0.05, 0.03], [0.5, np.nan, np.nan]
    residual = 1 - (1 - 0.55) / (1 - 0.5)
    out = value(game, p_s, p_c, xg)
    assert out["defensive_value"].iloc[1] == pytest.approx(residual - 0.05)
    assert formula.value(game, pd.Series(p_s), pd.Series(p_c))["defensive_value"].iloc[1] == pytest.approx(0.55 - 0.05)
    assert out["vaep_value"].iloc[0] == pytest.approx(formula.value(game, pd.Series(p_s), pd.Series(p_c))["vaep_value"].iloc[0]), "the shot itself is unchanged"
    # a teammate's rebound is no longer charged with the miss either
    rebound = _game([(1, 10.0, "shot", "fail"), (1, 11.0, "shot", "success")])
    out = value(rebound, [0.55, 0.70], [0.01, 0.01], [0.5, 0.6])
    assert out["offensive_value"].iloc[1] == pytest.approx(0.70 - residual)


def test_a_keeper_save_keeps_full_credit_for_the_chance():
    game = _game([(1, 10.0, "shot", "fail"), (2, 11.0, "keeper_save", "success"), (2, 15.0, "goalkick", "success")])
    p_s, p_c, xg = [0.55, 0.01, 0.01], [0.01, 0.02, 0.01], [0.5, np.nan, np.nan]
    out = value(game, p_s, p_c, xg)
    assert out["defensive_value"].iloc[1] == pytest.approx(0.55 - 0.02)
    np.testing.assert_allclose(out.to_numpy(), formula.value(game, pd.Series(p_s), pd.Series(p_c)).to_numpy(), atol=1e-12)


def test_goals_long_gaps_and_shots_without_xg_follow_socceraction():
    game = _game([(1, 10.0, "shot", "success"), (2, 60.0, "pass", "success"), (2, 62.0, "shot", "fail"), (1, 63.0, "clearance", "success"),
                  (1, 90.0, "shot", "fail"), (2, 120.0, "goalkick", "success")])
    p_s, p_c = [0.40, 0.02, 0.30, 0.01, 0.25, 0.01], [0.01, 0.01, 0.02, 0.03, 0.01, 0.01]
    xg = [0.35, np.nan, np.nan, np.nan, 0.2, np.nan]  # the second shot has no xG; the third is followed by a 30 s gap
    np.testing.assert_allclose(value(game, p_s, p_c, xg).to_numpy(), formula.value(game, pd.Series(p_s), pd.Series(p_c)).to_numpy(), atol=1e-12)


def test_the_residual_stays_between_zero_and_the_state_value():
    game = _game([(1, 10.0, "shot", "fail"), (2, 11.0, "clearance", "success")])
    out = value(game, [0.30, 0.01], [0.01, 0.02], [0.5, np.nan])  # the model valued the state below the shot's own xG
    assert out["defensive_value"].iloc[1] == pytest.approx(0.0 - 0.02), "residual clipped at 0"


# ------------------------------------------------------------------------- real data
def _xg_of_shots(match_ids):
    from src.vaep.labels import shot_xg_lookup
    return shot_xg_lookup(match_id=None) if match_ids is None else pd.concat([shot_xg_lookup(match_id=m) for m in match_ids])


@pytest.mark.data
def test_every_penalty_in_the_season_output_is_worth_its_xg_minus_the_prior():
    values = load_parquet("action_values.parquet", VAEP_HINT, columns=["game_id", "action_id", "type_name", "offensive_value"])
    actions = load_parquet("actions.parquet", INGEST_HINT, columns=["game_id", "action_id", "original_event_id"])
    pens = values[values["type_name"] == "shot_penalty"].merge(actions, on=["game_id", "action_id"])
    xg = pens["original_event_id"].map(_xg_of_shots(None))
    assert len(pens) == 91 and xg.notna().all()
    np.testing.assert_allclose(pens["offensive_value"].to_numpy(), (xg - PENALTY_PRIOR).to_numpy(), atol=1e-9)


def _option_3_matches(lineups):
    """Steve Cook's clearance after a missed penalty (3754225), the first match with a converted penalty, Aguero's 6-1 (3754079)."""
    values = load_parquet("action_values.parquet", VAEP_HINT, columns=["game_id", "type_name", "result_name"])
    scored = values[(values["type_name"] == "shot_penalty") & (values["result_name"] == "success")]["game_id"]
    assert 3754225 in set(lineups.loc[lineups["player_name"] == "Steve Cook", "match_id"])
    return sorted({3754225, int(scored.min()), 3754079})


@pytest.mark.data
def test_rating_a_match_alone_still_reproduces_the_season_output(lineups):
    from src.vaep.rate import action_values_for_match, load_model_for_match, rate_match

    raw = load_parquet("raw_vaep.parquet", VAEP_HINT).set_index(["match_id", "player_id"])["raw_score"]
    values = load_parquet("action_values.parquet", VAEP_HINT)
    season_xg = _xg_of_shots(None)
    for match_id in _option_3_matches(lineups):
        actions = actions_for([match_id])
        model = load_model_for_match(match_id)
        alone = rate_match(match_id, actions, lineups[lineups["match_id"] == match_id], model)  # reads its own shots' xG
        with_lookup = rate_match(match_id, actions, lineups[lineups["match_id"] == match_id], model, shot_xg=season_xg)
        pd.testing.assert_frame_equal(alone, with_lookup)
        got = alone.set_index(["match_id", "player_id"])["raw_score"]
        assert float((got - raw.reindex(got.index)).abs().max()) < 1e-12, match_id
        av = action_values_for_match(match_id, actions, model)
        cached = values[values["game_id"] == match_id].sort_values("action_id")
        np.testing.assert_allclose(av["vaep_value"].to_numpy(), cached["vaep_value"].to_numpy(), atol=1e-12)


@pytest.mark.data
def test_the_clearance_after_a_missed_penalty_no_longer_takes_the_whole_chance(lineups):
    """Steve Cook v West Brom (3754225): his clearance two seconds after a missed penalty was worth +0.50."""
    values = load_parquet("action_values.parquet", VAEP_HINT).sort_values(["game_id", "action_id"]).reset_index(drop=True)
    game = values[values["game_id"] == 3754225].reset_index(drop=True)
    after = game.index[(game["type_name"].shift(1) == "shot_penalty") & (game["result_name"].shift(1) != "success")]
    assert len(after) >= 1
    nxt = game.loc[after[0]]
    assert nxt["type_name"] == "clearance"
    assert nxt["vaep_value"] < 0.1
