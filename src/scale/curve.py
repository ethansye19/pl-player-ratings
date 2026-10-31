"""The fixed percentile -> rating target curve.

``PercentileScaler`` (scaler.py) turns a raw score into a percentile ``p`` in [0, 1] within
its position group; this module turns ``p`` into the (unrounded) rating the player is
shown. The curve is a table of anchors joined by straight lines (``numpy.interp``), so
there is nothing to fit and the same ``p`` always gives the same rating.

Why a curve and not ``3 + 7 * p``
---------------------------------
A linear map would make a 3.0 exactly as common as a 6.5 and give one rating in seven a 9
or better. A match rating is only useful if it behaves like the football scale people know:
ordinary performances bunch up, and the extremes stay extreme. The curve is therefore a
right-skewed S-shape, fixed in advance, with these properties (shares of *rounded*
ratings, exact on a uniform percentile grid):

========================  ==========================================================
mean                      6.500 (median 6.5, most common rating 6.4 at 6.0 percent,
                          standard deviation 0.83). The long right tail is what lifts
                          the mean slightly above the median.
6.0 to 7.5                66.0 percent of ratings ("most ratings").
7.0 or better             25.2 percent; 6.0 or better 76.4 percent.
8.0 or better             6.0 percent (the top 6 percent: p >= 0.94).
9.0 or better             0.50 percent (p >= 0.995).
10.0                      0.040 percent (p >= 0.9996), i.e. about 4 in 10,000.
below 5.0                 2.5 percent (the worst 2.5 percent: p < 0.025).
below 4.0                 0.10 percent.
3.0                       only at p = 0 and below the lowest reference score; the
                          lowest rating a season actually contains is about 3.4 to 3.8
                          because the lowest of n reference rows sits at p = 0.5 / n.
========================  ==========================================================

The mean is the point of the whole exercise: PROMPT.md asks for a full-season average of
6.5, and the scaler fits and applies on the same reference rows, so the season's ratings
are the curve sampled at (almost) uniformly spread percentiles. The anchors are tuned so
that the mean of ``round(curve(p), 1)`` over uniform ``p`` is 6.5000 +/- 0.0001. For a
reference group of n rows the percentiles are (i + 0.5) / n, and the mean of the rounded
ratings is within 0.001 of 6.5 for every n from 750 to 12,000 (each n was checked), within
0.002 from 300 rows and within 0.003 from 200. ``tests/test_scale_curve.py`` checks all of
this numerically.

Where the numbers come from
---------------------------
The anchors are frozen literals; do not nudge single entries by hand. They were read off a
smooth quantile function (a Johnson SU curve fitted to the targets above, with its core
tightened by 20 percent and the percentile axis eased slightly near the two design
points), tabulated to three decimals, and the centre was then shifted by 0.02 so that the
rounded mean is exactly 6.5. Three decimals matter: with two, the 0.1-wide histogram
jitters by 10 to 20 percent from one rating to the next. Anchors are denser (every 0.025)
where the curve bends most, so the piecewise-linear histogram stays within about 12 percent
of the smooth one at every rating that holds at least 0.4 percent of the rows.

Four *design points* carry the shape promises: (0.025, 4.951), (0.94, 7.951),
(0.995, 8.951) and (0.9996, 9.951). They sit at x.x51, not x.x5, on purpose: a rating of
5.0 needs the curve to reach 4.95, and landing exactly on a rounding boundary would let
floating point decide between 4.9 and 5.0 for a row whose percentile equals the anchor.

Changing the curve
------------------
Edit ``ANCHORS`` and run ``tests/test_scale_curve.py`` (mean, tens, shape) and
``tests/test_scale_report.py`` (``reports/scale.md`` quotes the table). A saved
``scaler_<method>.json`` embeds the anchors it was fitted with and refuses to load under a
different curve, so re-run ``python -m src.scale.apply --method <method>`` afterwards.
"""
from __future__ import annotations

import numpy as np

from src.contract import RATING_MAX, RATING_MIN

# (percentile, unrounded rating). Percentiles strictly increase from 0.0 to 1.0.
ANCHORS: tuple[tuple[float, float], ...] = (
    # lower tail: logarithmic, 3.0 only at p = 0
    (0.0, 3.000), (0.0001, 3.349), (0.0003, 3.633), (0.001, 3.956), (0.003, 4.267),
    (0.006, 4.475), (0.010, 4.636), (0.015, 4.770), (0.020, 4.868),
    # design point: the worst 2.5 percent rate below 5.0
    (0.025, 4.951),
    # lower shoulder and core
    (0.0375, 5.094), (0.050, 5.203), (0.075, 5.373), (0.100, 5.497), (0.125, 5.595),
    (0.150, 5.687), (0.175, 5.775), (0.200, 5.853), (0.225, 5.922), (0.250, 5.986),
    (0.300, 6.098), (0.350, 6.196), (0.400, 6.285), (0.450, 6.370), (0.500, 6.453),
    (0.550, 6.537), (0.600, 6.625), (0.650, 6.721), (0.700, 6.828), (0.750, 6.954),
    # upper shoulder
    (0.775, 7.025), (0.800, 7.105), (0.825, 7.195), (0.850, 7.298), (0.875, 7.418),
    (0.900, 7.574), (0.920, 7.754),
    # design point: the top 6 percent rate 8.0 or better
    (0.940, 7.951),
    (0.955, 8.100), (0.970, 8.270), (0.980, 8.421), (0.990, 8.679),
    # design point: the top 0.5 percent rate 9.0 or better
    (0.995, 8.951), (0.9985, 9.424),
    # design point: the top 0.04 percent rate 10.0
    (0.9996, 9.951), (1.0, 10.000),
)

_P = np.array([p for p, _ in ANCHORS], dtype="float64")
_R = np.array([r for _, r in ANCHORS], dtype="float64")


def _validate_anchors(percentiles: np.ndarray, ratings: np.ndarray) -> None:
    if percentiles[0] != 0.0 or percentiles[-1] != 1.0:
        raise ValueError("curve anchors must start at percentile 0.0 and end at 1.0")
    if not (np.diff(percentiles) > 0).all():
        raise ValueError("curve anchor percentiles must be strictly increasing")
    if not (np.diff(ratings) >= 0).all():
        raise ValueError("curve anchor ratings must be non-decreasing")
    if ratings[0] != RATING_MIN or ratings[-1] != RATING_MAX:
        raise ValueError(f"curve must run from {RATING_MIN} (p=0) to {RATING_MAX} (p=1)")


_validate_anchors(_P, _R)


def curve(p):
    """Unrounded rating for percentile ``p`` (scalar or array-like).

    Linear between anchors. ``p`` outside [0, 1] is clamped (so the result is always in
    [RATING_MIN, RATING_MAX]); NaN stays NaN. ``curve(0) == 3.0`` and ``curve(1) == 10.0``.
    Rounding to ``RATING_DECIMALS`` is the scaler's job, not this function's.
    """
    return np.interp(np.asarray(p, dtype="float64"), _P, _R)
