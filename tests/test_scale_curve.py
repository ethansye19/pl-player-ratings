"""Numerical properties of the fixed percentile -> rating curve (src/scale/curve.py).

The curve is what makes the season mean 6.5 and tens rare, so its properties are checked
on a fine uniform percentile grid, not just at a few points. "Uniform percentiles" is
what a fitted scaler produces on its own reference rows (mid-rank: (i + 0.5) / n)."""
import numpy as np
import pytest

from src.contract import MAX_SHARE_OF_TENS, RATING_DECIMALS, RATING_MAX, RATING_MIN, TARGET_MEAN
from src.scale.curve import ANCHORS, curve

GRID_N = 1_000_000


@pytest.fixture(scope="module")
def p_grid() -> np.ndarray:
    return (np.arange(GRID_N) + 0.5) / GRID_N


@pytest.fixture(scope="module")
def ratings(p_grid) -> np.ndarray:
    """Rounded ratings at uniformly spread percentiles."""
    return np.round(curve(p_grid), RATING_DECIMALS)


def _anchor_arrays():
    return (
        np.array([p for p, _ in ANCHORS], dtype="float64"),
        np.array([r for _, r in ANCHORS], dtype="float64"),
    )


def _share(ratings: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(mask))


# --------------------------------------------------------------------- the anchor table
def test_anchor_table_is_well_formed():
    p, r = _anchor_arrays()
    assert len(ANCHORS) >= 20, "too few anchors for the histogram to be smooth"
    assert p[0] == 0.0 and p[-1] == 1.0
    assert (np.diff(p) > 0).all(), "anchor percentiles must strictly increase"
    assert (np.diff(r) > 0).all(), "anchor ratings must strictly increase (no flat stretch)"
    assert r[0] == RATING_MIN and r[-1] == RATING_MAX


def test_no_anchor_sits_on_a_rounding_boundary():
    """A rating exactly at x.x5 would let floating point decide which way it rounds."""
    _, r = _anchor_arrays()
    scaled = r * 10 ** RATING_DECIMALS
    distance_to_boundary = np.abs((scaled % 1.0) - 0.5)
    assert (distance_to_boundary > 1e-6).all(), r[distance_to_boundary <= 1e-6]


def test_curve_is_linear_between_anchors():
    p, r = _anchor_arrays()
    mid_p = (p[:-1] + p[1:]) / 2
    np.testing.assert_allclose(curve(mid_p), (r[:-1] + r[1:]) / 2, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(curve(p), r)


def test_core_has_no_sharp_kinks():
    """Between 2.5 and 95 percent the slope changes by at most 60 percent from one
    segment to the next, so the 0.1-wide histogram has no cliffs or spikes."""
    p, r = _anchor_arrays()
    slopes = np.diff(r) / np.diff(p)
    ratio = slopes[:-1] / slopes[1:]          # slope before / slope after anchor p[1:-1]
    inside = (p[1:-1] >= 0.025) & (p[1:-1] <= 0.95)
    assert inside.sum() >= 20
    assert ((ratio[inside] > 1 / 1.6) & (ratio[inside] < 1.6)).all(), ratio[inside]


# ------------------------------------------------------------------- end points, bounds
def test_curve_end_points_are_exact():
    assert curve(0.0) == RATING_MIN
    assert curve(1.0) == RATING_MAX
    assert round(float(curve(0.0)), RATING_DECIMALS) == 3.0
    assert round(float(curve(1.0)), RATING_DECIMALS) == 10.0


def test_curve_is_monotone_non_decreasing(p_grid):
    grid = np.concatenate([[0.0], p_grid, [1.0]])
    assert (np.diff(curve(grid)) >= 0).all()


def test_curve_stays_within_the_rating_bounds(p_grid):
    values = curve(np.concatenate([[0.0], p_grid, [1.0]]))
    assert values.min() >= RATING_MIN and values.max() <= RATING_MAX


def test_curve_clamps_outside_unit_interval_and_passes_nan_through():
    assert curve(-0.5) == RATING_MIN
    assert curve(1.5) == RATING_MAX
    assert np.isnan(curve(np.nan))
    out = curve(np.array([[0.1, 0.5], [0.9, 2.0]]))
    assert out.shape == (2, 2) and out[1, 1] == RATING_MAX
    assert curve([0.25, 0.75]).shape == (2,)


# ------------------------------------------------------------------------------- tens
def test_ten_needs_a_percentile_of_about_0_9995_or_more(p_grid, ratings):
    tens = ratings == RATING_MAX
    assert tens.any(), "10.0 must be reachable for the very best performances"
    assert p_grid[tens].min() >= 0.9995
    assert ratings[p_grid < 0.9995].max() < RATING_MAX


def test_tens_are_well_below_the_one_in_a_thousand_limit(ratings):
    share = _share(ratings, ratings == RATING_MAX)
    assert share < MAX_SHARE_OF_TENS / 2, f"{share:.4%} of ratings are 10.0"


def test_three_point_zero_is_reserved_for_the_very_bottom(ratings):
    assert _share(ratings, ratings == RATING_MIN) < 0.0002
    assert _share(ratings, ratings < 4.0) <= 0.002


# ------------------------------------------------------------------------------ mean
def test_mean_of_rounded_ratings_over_uniform_percentiles_is_6_5(ratings):
    assert abs(ratings.mean() - TARGET_MEAN) <= 0.005, ratings.mean()


@pytest.mark.parametrize("n", [300, 500, 750, 1000, 2000, 3600, 10_000, 100_000])
def test_mean_is_6_5_for_realistic_reference_group_sizes(n):
    """A fitted scaler maps its own n reference rows to percentiles (i + 0.5) / n; the
    brief asks for 6.5 +/- 0.02 per group on as few as ~750 rows."""
    ratings_n = np.round(curve((np.arange(n) + 0.5) / n), RATING_DECIMALS)
    assert abs(ratings_n.mean() - TARGET_MEAN) <= 0.01, ratings_n.mean()


def test_mean_is_6_5_for_every_group_size_from_200_to_12000():
    worst_small, worst_large = 0.0, 0.0
    for n in range(200, 12_001):
        dev = abs(np.round(curve((np.arange(n) + 0.5) / n), RATING_DECIMALS).mean() - TARGET_MEAN)
        if n < 750:
            worst_small = max(worst_small, dev)
        else:
            worst_large = max(worst_large, dev)
    assert worst_large <= 0.001, worst_large
    assert worst_small <= 0.003, worst_small


# ----------------------------------------------------------------------------- shape
def test_most_ratings_fall_between_6_and_7_5(ratings):
    assert _share(ratings, (ratings >= 6.0) & (ratings <= 7.5)) >= 0.60


def test_top_tail_shares_match_a_football_rating_scale(ratings):
    assert 0.05 <= _share(ratings, ratings >= 8.0) <= 0.07
    assert 0.004 <= _share(ratings, ratings >= 9.0) <= 0.006


def test_bottom_tail_shares_match_a_football_rating_scale(ratings):
    assert 0.02 <= _share(ratings, ratings < 5.0) <= 0.03


def test_median_rating_is_near_the_anchor_and_the_mean_is_lifted_by_the_right_tail(ratings):
    assert 6.3 <= np.median(ratings) <= 6.7
    assert ratings.mean() >= np.median(ratings) - 0.1


def test_histogram_is_unimodal_at_0_1_resolution(ratings):
    """Rising to a single peak between 6.2 and 6.7, then falling; 10.0 is excluded
    because it collects the whole capped tail."""
    values = np.round(np.arange(RATING_MIN, 9.95, 0.1), 1)
    shares = np.array([np.mean(ratings == v) for v in values])
    assert (shares > 0).all(), "every rating from 3.0 to 9.9 must be reachable"
    peak = int(shares.argmax())
    assert 6.2 <= values[peak] <= 6.7
    noise = 5.0 / GRID_N        # a share is a count over GRID_N points: allow a few points of slack
    assert (np.diff(shares[: peak + 1]) >= -noise).all()
    assert (np.diff(shares[peak:]) <= noise).all()
    assert shares[peak] <= 0.075, "no single rating should hold more than 7.5 percent"
