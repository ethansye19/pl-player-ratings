"""Plain-text summaries of ratings and of the curve.

Pure functions (no I/O), shared by the CLI (``apply.py``) and ``reports/scale.md``: the
report quotes these blocks verbatim and ``tests/test_scale_report.py`` regenerates them, so
the numbers in the report cannot drift from the code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.contract import (
    MAX_SHARE_OF_TENS,
    MEAN_TOLERANCE,
    MIN_MINUTES,
    POSITION_GROUPS,
    RATING_COLUMN,
    RATING_DECIMALS,
    RATING_MAX,
    RATING_MIN,
    TARGET_MEAN,
)
from src.scale.curve import ANCHORS, curve

BUCKET_TENTHS = 5          # histogram buckets are 0.5 wide = 5 tenths of a rating point
CURVE_GRID_N = 1_000_000   # uniform percentile grid used for the curve statistics


def _tenths(ratings: pd.Series | np.ndarray) -> np.ndarray:
    """Ratings as integer tenths (6.4 -> 64), immune to float representation noise."""
    return np.rint(np.asarray(ratings, dtype="float64") * 10 ** RATING_DECIMALS).astype("int64")


def _pct(share: float) -> str:
    return f"{share * 100:.3f}%" if share < 0.01 else f"{share * 100:.1f}%"


# --------------------------------------------------------------------- histogram
def rating_histogram(ratings: pd.Series | np.ndarray) -> pd.DataFrame:
    """Counts and shares in 0.5-wide buckets (3.0-3.4, 3.5-3.9, ... 9.5-9.9) plus a 10.0 bucket."""
    tenths = _tenths(ratings)
    top = int(round(RATING_MAX * 10 ** RATING_DECIMALS))
    low = int(round(RATING_MIN * 10 ** RATING_DECIMALS))
    rows = []
    for start in range(low, top, BUCKET_TENTHS):
        rows.append((f"{start / 10:.1f}-{(start + BUCKET_TENTHS - 1) / 10:.1f}", ((tenths >= start) & (tenths < start + BUCKET_TENTHS)).sum()))
    rows.append((f"{RATING_MAX:.1f}", (tenths >= top).sum()))
    out = pd.DataFrame(rows, columns=["bucket", "count"])
    out["count"] = out["count"].astype("int64")
    out["share"] = out["count"] / max(len(tenths), 1)
    return out


def format_histogram(ratings: pd.Series | np.ndarray, bar_width: int = 40) -> str:
    """Text histogram: bucket, count, share and a bar scaled to the largest bucket."""
    hist = rating_histogram(ratings)
    peak = hist["share"].max() or 1.0
    lines = [f"{'rating':<8} {'count':>6} {'share':>7}"]
    for row in hist.itertuples(index=False):
        bar = "#" * (max(1, round(row.share / peak * bar_width)) if row.count else 0)
        lines.append(f"{row.bucket:<8} {row.count:>6d} {row.share * 100:>6.2f}%  {bar}".rstrip())
    return "\n".join(lines)


# ----------------------------------------------------------------- group summary
def group_summary(rated: pd.DataFrame) -> pd.DataFrame:
    """Rated rows, mean rating, number of 10.0s, min and max per position group, plus an 'all' row."""
    parts = {g: rated.loc[rated["position_group"] == g, RATING_COLUMN] for g in POSITION_GROUPS}
    parts = {g: r for g, r in parts.items() if len(r)}
    parts["all"] = rated[RATING_COLUMN]
    return pd.DataFrame(
        {
            "rows": {g: len(r) for g, r in parts.items()},
            "mean": {g: float(r.mean()) for g, r in parts.items()},
            "tens": {g: int((r == RATING_MAX).sum()) for g, r in parts.items()},
            "min": {g: float(r.min()) for g, r in parts.items()},
            "max": {g: float(r.max()) for g, r in parts.items()},
        }
    )


def format_group_table(rated: pd.DataFrame) -> str:
    """Markdown table of ``group_summary``."""
    lines = [
        "| group | rated rows | mean rating | 10.0s | min | max |",
        "|-------|-----------:|------------:|------:|----:|----:|",
    ]
    table = group_summary(rated)
    for group in table.index:
        lines.append(
            f"| {group} | {int(table.at[group, 'rows'])} | {table.at[group, 'mean']:.3f} | {int(table.at[group, 'tens'])} "
            f"| {table.at[group, 'min']:.1f} | {table.at[group, 'max']:.1f} |"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------ curve
def format_anchor_table() -> str:
    """Markdown table of the curve anchors."""
    lines = ["| percentile p | curve(p) |", "|-------------:|---------:|"]
    lines += [f"| {p:.4f} | {r:.3f} |" for p, r in ANCHORS]
    return "\n".join(lines)


def curve_statistics() -> list[tuple[str, str]]:
    """(statistic, value) pairs for the rounded ratings at uniformly spread percentiles."""
    p = (np.arange(CURVE_GRID_N) + 0.5) / CURVE_GRID_N
    tenths = _tenths(np.round(curve(p), RATING_DECIMALS))
    ratings = tenths / 10 ** RATING_DECIMALS
    values, counts = np.unique(tenths, return_counts=True)
    mode = values[counts.argmax()] / 10 ** RATING_DECIMALS

    def share(mask: np.ndarray) -> float:
        return float(mask.mean())

    return [
        ("mean rating", f"{ratings.mean():.3f}"),
        ("median rating", f"{np.median(ratings):.1f}"),
        ("most common rating", f"{mode:.1f} ({_pct(counts.max() / CURVE_GRID_N)} of ratings)"),
        ("standard deviation", f"{ratings.std():.2f}"),
        ("6.0 to 7.5", _pct(share((ratings >= 6.0) & (ratings <= 7.5)))),
        ("7.0 or better", _pct(share(ratings >= 7.0))),
        ("8.0 or better", _pct(share(ratings >= 8.0))),
        ("9.0 or better", _pct(share(ratings >= 9.0))),
        (f"{RATING_MAX:.1f}", _pct(share(ratings == RATING_MAX))),
        ("below 5.0", _pct(share(ratings < 5.0))),
        ("below 4.0", _pct(share(ratings < 4.0))),
    ]


def format_curve_statistics() -> str:
    """Markdown table of ``curve_statistics``."""
    lines = ["| statistic | value |", "|-----------|------:|"]
    lines += [f"| {name} | {value} |" for name, value in curve_statistics()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def format_cli_summary(method: str, raw_rows: int, rated: pd.DataFrame) -> list[str]:
    """The lines ``python -m src.scale.apply`` prints after writing its outputs."""
    table = group_summary(rated)
    total = int(table.loc["all", "rows"])
    ratings = rated[RATING_COLUMN]
    tens = int(table.loc["all", "tens"])
    ten_share = tens / total
    lines = [
        f"scale[{method}]",
        f"  rows rated     : {total} of {raw_rows} raw rows (minutes >= {MIN_MINUTES:g})",
        f"  mean rating    : {ratings.mean():.3f} (target {TARGET_MEAN} +/- {MEAN_TOLERANCE})",
        "  mean by group  : "
        + " | ".join(
            f"{g} {table.loc[g, 'mean']:.3f} (n={int(table.loc[g, 'rows'])})" for g in table.index if g != "all"
        ),
        f"  share of {RATING_MAX:.1f}  : {ten_share:.3%} ({tens} of {total} rows; limit < {MAX_SHARE_OF_TENS:.1%})",
        f"  min / max      : {ratings.min():.1f} / {ratings.max():.1f}",
    ]
    if abs(ratings.mean() - TARGET_MEAN) > MEAN_TOLERANCE:
        lines.append(f"  WARNING: mean rating {ratings.mean():.3f} is outside {TARGET_MEAN} +/- {MEAN_TOLERANCE}")
    for g in table.index:
        if g != "all" and abs(table.loc[g, "mean"] - TARGET_MEAN) > MEAN_TOLERANCE:
            lines.append(f"  WARNING: {g} mean rating {table.loc[g, 'mean']:.3f} is outside {TARGET_MEAN} +/- {MEAN_TOLERANCE}")
    if ten_share >= MAX_SHARE_OF_TENS:
        lines.append(f"  WARNING: {ten_share:.3%} of ratings are {RATING_MAX:.1f}; the limit is < {MAX_SHARE_OF_TENS:.1%}")
    return lines
