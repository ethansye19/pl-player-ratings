"""PercentileScaler behaviour beyond the 17 gate tests in test_scale.py: the exact
mid-rank / interpolation / clipping arithmetic, input handling, error cases and the JSON
format. All frames are SYNTHETIC (built here), never real data."""
import copy
import json
import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.contract import CONTRACT_COLUMNS, MIN_MINUTES, POSITION_GROUPS, RATING_DECIMALS, RATING_MAX, RATING_MIN
from src.scale.curve import ANCHORS, curve
from src.scale.scaler import PercentileScaler


def make_frame(raw, group="MID", minutes=90.0) -> pd.DataFrame:
    """SYNTHETIC contract-format frame, one row per raw score (one match per row)."""
    raw = np.asarray(raw, dtype="float64")
    n = len(raw)
    groups = np.broadcast_to(np.asarray(group, dtype=object), (n,))
    return pd.DataFrame(
        {
            "match_id": np.arange(1, n + 1, dtype="int64"),
            "player_id": np.arange(1, n + 1, dtype="int64"),
            "player_name": [f"Synthetic Player {i}" for i in range(n)],
            "team": "Synthetic FC",
            "position_group": groups,
            "minutes": np.broadcast_to(np.asarray(minutes, dtype="float64"), (n,)),
            "raw_score": raw,
        }
    )


def expected_rating(p) -> np.ndarray:
    return np.round(curve(np.asarray(p, dtype="float64")), RATING_DECIMALS)


# ------------------------------------------------------------------ the arithmetic
def test_mid_rank_percentiles_for_distinct_reference_scores():
    df = make_frame(np.arange(1.0, 11.0))
    out = PercentileScaler().fit_transform(df)
    np.testing.assert_array_equal(out["rating"].to_numpy(), expected_rating((np.arange(10) + 0.5) / 10))


def test_tied_scores_share_the_mid_rank_of_their_block():
    """Reference [1, 1, 2, 3]: below/equal counts give p = 0.25, 0.625, 0.875."""
    scaler = PercentileScaler().fit(make_frame([1.0, 1.0, 2.0, 3.0]))
    out = scaler.transform(make_frame([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(out["rating"].to_numpy(), expected_rating([0.25, 0.625, 0.875]))


def test_scores_between_reference_points_interpolate_linearly():
    """Reference [0, 10] has p(0) = 0.25 and p(10) = 0.75; the percentile is linear between."""
    scaler = PercentileScaler().fit(make_frame([0.0, 10.0]))
    probe = make_frame([0.0, 2.5, 5.0, 7.5, 10.0])
    out = scaler.transform(probe)
    np.testing.assert_array_equal(out["rating"].to_numpy(), expected_rating([0.25, 0.375, 0.5, 0.625, 0.75]))


def test_scores_outside_the_reference_range_clip_and_the_extremes_do_not():
    scaler = PercentileScaler().fit(make_frame(np.arange(1.0, 11.0)))
    below, lowest, highest, above = scaler.transform(make_frame([0.999, 1.0, 10.0, 10.001]))["rating"]
    assert below == RATING_MIN and above == RATING_MAX
    assert lowest == expected_rating(0.05) > RATING_MIN
    assert highest == expected_rating(0.95) < RATING_MAX


def test_single_distinct_reference_score_rates_6_5_and_clips_either_side():
    scaler = PercentileScaler().fit(make_frame([4.0, 4.0, 4.0]))
    out = scaler.transform(make_frame([3.9, 4.0, 4.1]))
    assert out["rating"].tolist() == [RATING_MIN, float(expected_rating(0.5)), RATING_MAX]
    assert float(expected_rating(0.5)) == 6.5


def test_rating_uses_each_rows_own_group_table():
    ref = pd.concat([make_frame(np.arange(100.0), "GK"), make_frame(np.arange(1000.0, 1100.0), "FWD")], ignore_index=True)
    ref["match_id"] = ref["player_id"] = np.arange(len(ref), dtype="int64")
    scaler = PercentileScaler().fit(ref)
    probe = pd.concat([make_frame([50.0], "GK"), make_frame([50.0], "FWD")], ignore_index=True)
    gk, fwd = scaler.transform(probe)["rating"]
    assert gk == expected_rating(0.505)          # inside the GK range (rank 50 of 100)
    assert fwd == RATING_MIN                     # far below the FWD range


@pytest.mark.parametrize(
    "name, draw",
    [
        ("normal", lambda rng, n: rng.normal(size=n)),
        ("lognormal", lambda rng, n: rng.lognormal(size=n)),
        ("exponential", lambda rng, n: rng.exponential(size=n)),
        ("uniform", lambda rng, n: rng.uniform(size=n)),
        ("student-t(2)", lambda rng, n: rng.standard_t(2, size=n)),
    ],
)
@pytest.mark.parametrize("n", [750, 3000])
def test_group_mean_is_6_5_whatever_the_shape_of_the_raw_scores(name, draw, n):
    """The percentile transform removes the raw distribution's shape entirely."""
    df = make_frame(draw(np.random.default_rng(11), n))
    assert abs(PercentileScaler().fit_transform(df)["rating"].mean() - 6.5) <= 0.002


@pytest.mark.parametrize("where", ["bottom", "middle", "top"])
def test_a_large_block_of_tied_scores_barely_moves_the_group_mean(where):
    """20 percent of a group sharing one raw score is a bad case for mid-rank ties: the
    whole block gets the curve value at its middle. The mean still stays within 0.04."""
    n, k = 2000, 400
    raw = np.sort(np.random.default_rng(5).normal(size=n))
    start = {"bottom": 0, "middle": (n - k) // 2, "top": n - k}[where]
    raw[start : start + k] = raw[start + k // 2]
    mean = PercentileScaler().fit_transform(make_frame(raw))["rating"].mean()
    assert abs(mean - 6.5) <= 0.04, mean


# ---------------------------------------------------------------- minutes handling
def test_rows_under_10_minutes_neither_enter_the_reference_nor_the_output():
    raw = np.arange(1.0, 21.0)
    minutes = np.full(20, 90.0)
    cameo = np.array([5, 6, 7])
    minutes[cameo] = [9.9, 0.1, 9.99]
    df = make_frame(raw, minutes=minutes)
    df.loc[cameo, "raw_score"] = [1e9, -1e9, 12345.0]            # would wreck the table if used
    scaler = PercentileScaler().fit(df)
    assert_frame_equal(df.loc[df["minutes"] >= MIN_MINUTES].reset_index(drop=True), scaler.transform(df).drop(columns="rating"))
    assert scaler.to_dict() == PercentileScaler().fit(df[df["minutes"] >= MIN_MINUTES]).to_dict()
    assert scaler.reference_sizes() == {"MID": 17}


def test_exactly_ten_minutes_is_rated():
    df = make_frame([1.0, 2.0, 3.0], minutes=[MIN_MINUTES, 9.9, 90.0])
    out = PercentileScaler().fit(df).transform(df)
    assert out["minutes"].tolist() == [MIN_MINUTES, 90.0]


def test_transform_ignores_garbage_in_rows_it_drops_but_not_in_rows_it_rates():
    scaler = PercentileScaler().fit(make_frame(np.arange(100.0)))
    probe = make_frame([1.0, np.nan, 3.0], minutes=[90.0, 5.0, 90.0])
    probe.loc[1, "position_group"] = "NOT A GROUP"
    assert scaler.transform(probe)["raw_score"].tolist() == [1.0, 3.0]
    probe.loc[1, "minutes"] = 45.0
    with pytest.raises(ValueError):
        scaler.transform(probe)


def test_all_rows_under_10_minutes_gives_an_empty_frame_with_the_rating_column():
    scaler = PercentileScaler().fit(make_frame(np.arange(10.0)))
    out = scaler.transform(make_frame([1.0, 2.0], minutes=[3.0, 9.0]))
    assert out.empty and list(out.columns) == CONTRACT_COLUMNS + ["rating"]
    assert out["rating"].dtype == np.float64


# ------------------------------------------------------------ frame handling
def test_output_keeps_input_columns_order_dtypes_and_resets_the_index():
    df = make_frame(np.arange(1.0, 9.0), minutes=[90, 90, 4, 90, 90, 90, 90, 90]).assign(team_id=7, is_starter=True)
    df.index = np.arange(100, 108)[::-1]
    before = df.copy(deep=True)
    out = PercentileScaler().fit(df).transform(df)
    assert_frame_equal(df, before)                                   # input untouched
    assert list(out.columns) == list(df.columns) + ["rating"]
    assert out.index.tolist() == list(range(7))
    assert out["rating"].dtype == np.float64
    assert out["match_id"].dtype == np.int64 and out["is_starter"].dtype == bool
    assert out["player_id"].tolist() == [1, 2, 4, 5, 6, 7, 8]        # input row order, row 3 dropped


def test_an_existing_rating_column_is_replaced():
    df = make_frame(np.arange(1.0, 6.0))
    df["rating"] = -1.0
    out = PercentileScaler().fit(df).transform(df)
    assert (out["rating"] >= RATING_MIN).all() and list(out.columns).count("rating") == 1


def test_categorical_position_group_is_accepted():
    df = make_frame(np.arange(1.0, 21.0))
    df["position_group"] = df["position_group"].astype("category")
    out = PercentileScaler().fit_transform(df)
    assert len(out) == 20


def test_transform_is_permutation_equivariant():
    rng = np.random.default_rng(2)
    df = make_frame(rng.normal(size=300))
    scaler = PercentileScaler().fit(df)
    order = rng.permutation(len(df))
    shuffled = scaler.transform(df.iloc[order])
    straight = scaler.transform(df).iloc[order].reset_index(drop=True)
    assert_frame_equal(shuffled, straight)


def test_fit_does_not_depend_on_row_order_and_refit_replaces_the_tables():
    rng = np.random.default_rng(4)
    a, b = make_frame(rng.normal(size=200)), make_frame(rng.normal(size=300, loc=5))
    assert PercentileScaler().fit(a).to_dict() == PercentileScaler().fit(a.sample(frac=1, random_state=1)).to_dict()
    scaler = PercentileScaler().fit(a)
    assert scaler.fit(b) is scaler
    assert scaler.to_dict() == PercentileScaler().fit(b).to_dict()


def test_fit_transform_is_fit_then_transform():
    df = make_frame(np.random.default_rng(9).normal(size=120))
    assert_frame_equal(PercentileScaler().fit_transform(df), PercentileScaler().fit(df).transform(df))


def test_state_accessors_and_repr():
    scaler = PercentileScaler()
    assert not scaler.is_fitted and scaler.groups == () and repr(scaler) == "PercentileScaler(unfitted)"
    df = pd.concat([make_frame(np.arange(60.0), "FWD"), make_frame(np.arange(70.0), "GK")], ignore_index=True)
    df["match_id"] = df["player_id"] = np.arange(len(df), dtype="int64")
    scaler.fit(df)
    assert scaler.is_fitted and scaler.groups == ("GK", "FWD")        # POSITION_GROUPS order
    assert scaler.reference_sizes() == {"GK": 70, "FWD": 60}
    assert repr(scaler) == "PercentileScaler(GK=70, FWD=60)"


def test_small_reference_groups_log_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="src.scale.scaler"):
        PercentileScaler().fit(make_frame(np.arange(10.0)))
    assert any("MID" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------- errors
def test_unfitted_scaler_refuses_to_work(tmp_path):
    scaler = PercentileScaler()
    for call in (lambda: scaler.transform(make_frame([1.0])), scaler.to_dict, lambda: scaler.save(tmp_path / "x.json")):
        with pytest.raises(RuntimeError):
            call()
    assert not (tmp_path / "x.json").exists()


def test_a_group_absent_from_the_fit_is_rejected_by_name():
    scaler = PercentileScaler().fit(make_frame(np.arange(60.0), "DEF"))
    with pytest.raises(ValueError, match="GK"):
        scaler.transform(make_frame([1.0], "GK"))
    with pytest.raises(ValueError, match="WING"):
        scaler.transform(make_frame([1.0], "WING"))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_raw_scores_are_rejected_on_rated_rows(bad):
    scaler = PercentileScaler().fit(make_frame(np.arange(60.0)))
    with pytest.raises(ValueError, match="raw_score"):
        scaler.transform(make_frame([1.0, bad]))


def test_bad_minutes_and_missing_or_non_numeric_columns_are_rejected():
    scaler = PercentileScaler().fit(make_frame(np.arange(60.0)))
    with pytest.raises(ValueError, match="minutes"):
        scaler.transform(make_frame([1.0, 2.0], minutes=[90.0, np.nan]))
    with pytest.raises(ValueError, match="missing contract columns"):
        scaler.transform(make_frame([1.0]).drop(columns="raw_score"))
    text_scores = make_frame([1.0]).assign(raw_score="high")
    with pytest.raises(ValueError, match="numeric"):
        scaler.transform(text_scores)


def test_fit_needs_at_least_one_rated_row_and_a_valid_contract():
    with pytest.raises(ValueError, match="nothing to fit"):
        PercentileScaler().fit(make_frame([1.0, 2.0], minutes=[3.0, 9.9]))
    with pytest.raises(ValueError):
        PercentileScaler().fit(make_frame([1.0, np.nan]))
    with pytest.raises(ValueError):
        PercentileScaler().fit(make_frame([1.0, 2.0], group="WING"))
    duplicated = pd.concat([make_frame([1.0]), make_frame([2.0])], ignore_index=True)
    with pytest.raises(ValueError):
        PercentileScaler().fit(duplicated)


# ------------------------------------------------------------------ JSON format
@pytest.fixture()
def small_scaler() -> PercentileScaler:
    return PercentileScaler().fit(make_frame([0.1 + 0.2, 1 / 3, 1 / 3, 2.5, 7.0]))


def test_saved_file_is_strict_json_with_the_documented_keys(small_scaler, tmp_path):
    path = small_scaler.save(tmp_path / "nested" / "scaler.json")
    assert path == tmp_path / "nested" / "scaler.json" and path.is_file()
    assert [p.name for p in path.parent.iterdir()] == ["scaler.json"], "no temporary file left behind"
    assert b"\r" not in path.read_bytes()

    def forbid(name):
        raise AssertionError(f"non-finite constant {name} in the saved file")

    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=forbid)
    assert set(data) == {"format", "version", "min_minutes", "rating_decimals", "curve", "groups"}
    assert data["format"] == "PercentileScaler" and data["version"] == 1
    assert data["min_minutes"] == MIN_MINUTES and data["rating_decimals"] == RATING_DECIMALS
    assert data["curve"] == {"percentiles": [p for p, _ in ANCHORS], "ratings": [r for _, r in ANCHORS]}
    assert set(data["groups"]) == {"MID"} and set(data["groups"]["MID"]) == {"n", "values", "counts"}
    assert data["groups"]["MID"]["n"] == 5 and data["groups"]["MID"]["counts"] == [1, 2, 1, 1]


def test_round_trip_is_exact_including_awkward_floats(small_scaler, tmp_path):
    reloaded = PercentileScaler.load(small_scaler.save(tmp_path / "s.json"))
    assert reloaded.to_dict() == small_scaler.to_dict()
    assert reloaded.to_dict()["groups"]["MID"]["values"][:2] == [0.1 + 0.2, 1 / 3]   # bit-exact floats
    assert PercentileScaler.from_dict(small_scaler.to_dict()).to_dict() == small_scaler.to_dict()
    probe = make_frame([0.1 + 0.2, 1 / 3, 0.3333333333333334, 0.30000000000000004, 8.0])
    assert_frame_equal(reloaded.transform(probe), small_scaler.transform(probe))


def test_to_dict_does_not_share_state_with_the_scaler(small_scaler):
    snapshot = copy.deepcopy(small_scaler.to_dict())
    small_scaler.to_dict()["groups"]["MID"]["values"].clear()
    assert small_scaler.to_dict() == snapshot


_MUTATIONS = {
    "wrong format": lambda d: d.update(format="Other"),
    "wrong version": lambda d: d.update(version=99),
    "different MIN_MINUTES": lambda d: d.update(min_minutes=5.0),
    "different RATING_DECIMALS": lambda d: d.update(rating_decimals=2),
    "different curve": lambda d: d["curve"]["ratings"].__setitem__(5, d["curve"]["ratings"][5] + 0.001),
    "missing curve": lambda d: d.pop("curve"),
    "no groups": lambda d: d.update(groups={}),
    "unknown group": lambda d: d["groups"].update(WING=d["groups"]["MID"]),
    "values out of order": lambda d: d["groups"]["MID"]["values"].reverse(),
    "duplicate values": lambda d: d["groups"]["MID"]["values"].__setitem__(1, d["groups"]["MID"]["values"][0]),
    "non-finite value": lambda d: d["groups"]["MID"]["values"].__setitem__(0, float("nan")),
    "counts too short": lambda d: d["groups"]["MID"]["counts"].pop(),
    "zero count": lambda d: d["groups"]["MID"]["counts"].__setitem__(0, 0),
    "fractional count": lambda d: d["groups"]["MID"]["counts"].__setitem__(0, 1.5),
    "n disagrees with counts": lambda d: d["groups"]["MID"].update(n=999),
    "missing values": lambda d: d["groups"]["MID"].pop("values"),
    "missing n": lambda d: d["groups"]["MID"].pop("n"),
}


@pytest.mark.parametrize("name", sorted(_MUTATIONS))
def test_from_dict_rejects_malformed_or_stale_data(small_scaler, name):
    data = copy.deepcopy(small_scaler.to_dict())
    _MUTATIONS[name](data)
    with pytest.raises(ValueError):
        PercentileScaler.from_dict(data)


def test_from_dict_rejects_non_mappings():
    for junk in ([], "scaler", None, 3):
        with pytest.raises(ValueError):
            PercentileScaler.from_dict(junk)


def test_load_rejects_unreadable_or_non_finite_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        PercentileScaler.load(bad)
    bad.write_text('{"format": "PercentileScaler", "values": [NaN]}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        PercentileScaler.load(bad)
    with pytest.raises(FileNotFoundError):
        PercentileScaler.load(tmp_path / "missing.json")


def test_every_position_group_can_be_saved_and_reloaded(tmp_path):
    parts = [make_frame(np.arange(80.0) * (i + 1), group) for i, group in enumerate(POSITION_GROUPS)]
    df = pd.concat(parts, ignore_index=True)
    df["match_id"] = df["player_id"] = np.arange(len(df), dtype="int64")
    scaler = PercentileScaler().fit(df)
    reloaded = PercentileScaler.load(scaler.save(tmp_path / "all.json"))
    assert reloaded.groups == POSITION_GROUPS
    assert_frame_equal(reloaded.transform(df), scaler.transform(df))
