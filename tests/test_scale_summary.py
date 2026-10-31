"""The text summaries in src/scale/summary.py (histogram, group table, curve statistics, CLI
summary). Hand-built SYNTHETIC rating frames; the numbers are small enough to check by eye."""
import numpy as np
import pandas as pd
import pytest

from src.contract import RATING_MAX
from src.scale.curve import ANCHORS
from src.scale.scaler import PercentileScaler
from src.scale.summary import (
    curve_statistics,
    format_anchor_table,
    format_cli_summary,
    format_curve_statistics,
    format_group_table,
    format_histogram,
    group_summary,
    rating_histogram,
)


def rated_frame(groups, ratings) -> pd.DataFrame:
    return pd.DataFrame({"position_group": groups, "rating": np.asarray(ratings, dtype="float64")})


def test_histogram_has_fifteen_half_point_buckets_and_a_separate_ten_bucket():
    hist = rating_histogram([6.5])
    assert len(hist) == 15
    assert hist["bucket"].iloc[0] == "3.0-3.4" and hist["bucket"].iloc[-2] == "9.5-9.9"
    assert hist["bucket"].iloc[-1] == "10.0"


def test_histogram_counts_land_in_the_right_buckets():
    ratings = [3.0, 3.4, 3.5, 6.4, 6.5, 9.9, 10.0, 10.0]
    hist = rating_histogram(ratings).set_index("bucket")
    assert hist.loc["3.0-3.4", "count"] == 2
    assert hist.loc["3.5-3.9", "count"] == 1
    assert hist.loc["6.0-6.4", "count"] == 1 and hist.loc["6.5-6.9", "count"] == 1
    assert hist.loc["9.5-9.9", "count"] == 1 and hist.loc["10.0", "count"] == 2
    assert hist["count"].sum() == len(ratings)
    assert hist["share"].sum() == pytest.approx(1.0)
    assert hist.loc["10.0", "share"] == pytest.approx(0.25)


def test_histogram_buckets_are_immune_to_float_representation_noise():
    noisy = [0.1 * 35, 3.4999999999999996, 0.1 * 34, 6.499999999999999, 6.500000000000001]
    hist = rating_histogram(noisy).set_index("bucket")["count"]
    assert hist["3.5-3.9"] == 2          # 3.5000000000000004 and 3.4999999999999996 are both 3.5
    assert hist["3.0-3.4"] == 1          # 3.4000000000000004 is 3.4
    assert hist["6.0-6.4"] == 0 and hist["6.5-6.9"] == 2


def test_histogram_of_nothing_is_all_zero():
    hist = rating_histogram([])
    assert (hist["count"] == 0).all() and (hist["share"] == 0).all()
    assert "#" not in format_histogram([])


def test_format_histogram_layout():
    text = format_histogram([6.4] * 6 + [6.5] * 3 + [7.0])
    lines = text.splitlines()
    assert lines[0] == "rating    count   share"
    assert len(lines) == 16
    by_bucket = {line.split()[0]: line for line in lines[1:]}
    assert by_bucket["6.0-6.4"].endswith("#" * 40)                       # the largest bucket gets the full bar
    assert by_bucket["6.0-6.4"] == f"6.0-6.4       6  60.00%  {'#' * 40}"
    assert by_bucket["6.5-6.9"].count("#") == 20 and by_bucket["7.0-7.4"].count("#") == 7
    assert by_bucket["3.0-3.4"] == "3.0-3.4       0   0.00%"            # empty bucket: no bar, no trailing space
    assert all(line == line.rstrip() for line in lines)


def test_group_summary_and_markdown_table():
    rated = rated_frame(["GK", "GK", "DEF", "DEF", "DEF"], [6.0, 7.0, 5.5, 6.5, 10.0])
    table = group_summary(rated)
    assert list(table.index) == ["GK", "DEF", "all"]
    assert table.loc["all", "rows"] == 5 and table.loc["all", "tens"] == 1
    assert table.loc["DEF", "mean"] == pytest.approx(22 / 3)
    lines = format_group_table(rated).splitlines()
    assert lines[0] == "| group | rated rows | mean rating | 10.0s | min | max |"
    assert lines[2:] == [
        "| GK | 2 | 6.500 | 0 | 6.0 | 7.0 |",
        "| DEF | 3 | 7.333 | 1 | 5.5 | 10.0 |",
        "| all | 5 | 7.000 | 1 | 5.5 | 10.0 |",
    ]


def test_anchor_table_lists_every_anchor():
    lines = format_anchor_table().splitlines()
    assert len(lines) == len(ANCHORS) + 2
    assert lines[2] == "| 0.0000 | 3.000 |" and lines[-1] == "| 1.0000 | 10.000 |"
    assert "| 0.9996 | 9.951 |" in lines


def test_curve_statistics_table_has_the_promised_rows():
    stats = dict(curve_statistics())
    assert float(stats["mean rating"]) == pytest.approx(6.5, abs=0.001)
    assert stats["median rating"] == "6.5"
    assert stats["most common rating"].startswith("6.4 (")
    assert set(stats) >= {"6.0 to 7.5", "7.0 or better", "8.0 or better", "9.0 or better", f"{RATING_MAX:.1f}", "below 5.0", "below 4.0"}
    lines = format_curve_statistics().splitlines()
    assert lines[0] == "| statistic | value |" and len(lines) == len(stats) + 2


def test_cli_summary_for_a_conforming_season_has_no_warning(fake_season):
    rated = PercentileScaler().fit(fake_season).transform(fake_season)
    lines = format_cli_summary("test", len(fake_season), rated)
    text = "\n".join(lines)
    assert lines[0] == "scale[test]"
    assert f"{len(rated)} of {len(fake_season)} raw rows" in text
    assert "WARNING" not in text


def test_cli_summary_warns_about_a_mean_off_target_and_too_many_tens():
    ratings = [7.5] * 90 + [10.0] * 10
    rated = rated_frame(["GK"] * 50 + ["DEF"] * 50, ratings)
    text = "\n".join(format_cli_summary("bad", 123, rated))
    assert "WARNING: mean rating 7.750 is outside 6.5 +/- 0.1" in text
    assert "WARNING: GK mean rating" in text and "WARNING: DEF mean rating" in text
    assert "10.000% of ratings are 10.0; the limit is < 0.1%" in text
