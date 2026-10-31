"""reports/scale.md quotes tables that are generated from the code (curve anchors, curve
statistics, the rating histogram and group means on the SYNTHETIC season, a tie stress
test). These tests regenerate every table and require it to appear verbatim in the report,
so the report cannot drift from the code. If one fails, regenerate the table (the failure
message prints it) and paste it into reports/scale.md."""
import numpy as np
import pytest

from src.scale.scaler import PercentileScaler
from src.scale.summary import format_anchor_table, format_curve_statistics, format_group_table, format_histogram
from tests.conftest import ROOT
from tests.test_scale_scaler import make_frame

REPORT = ROOT / "reports" / "scale.md"


def tie_stress_table() -> str:
    """Markdown table: mean rating of a fitted group when one raw score is shared by a block of its rows.

    SYNTHETIC normal raw scores, n = 2000, block placed at the bottom, middle or top of the group."""
    n = 2000
    base = np.sort(np.random.default_rng(5).normal(size=n))
    lines = [
        "| rows sharing one raw score | block at the bottom | block in the middle | block at the top |",
        "|---------------------------:|--------------------:|--------------------:|-----------------:|",
    ]
    for share in (0.02, 0.05, 0.10, 0.20, 0.30):
        k = int(round(share * n))
        cells = []
        for start in (0, (n - k) // 2, n - k):
            raw = base.copy()
            raw[start : start + k] = raw[start + k // 2]
            cells.append(f"{PercentileScaler().fit_transform(make_frame(raw))['rating'].mean():.3f}")
        lines.append(f"| {share:.0%} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def report_text() -> str:
    assert REPORT.is_file(), "reports/scale.md is missing"
    return REPORT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def synthetic_rated(fake_season):
    return PercentileScaler().fit(fake_season).transform(fake_season)


def _require_block(text: str, block: str, what: str) -> None:
    assert block in text, f"reports/scale.md is out of date: {what} should read\n{block}"


def test_report_quotes_the_current_anchor_table(report_text):
    _require_block(report_text, format_anchor_table(), "the anchor table")


def test_report_quotes_the_current_curve_statistics(report_text):
    _require_block(report_text, format_curve_statistics(), "the curve statistics")


def test_report_quotes_the_synthetic_season_histogram_and_group_table(report_text, synthetic_rated):
    _require_block(report_text, format_histogram(synthetic_rated["rating"]), "the histogram")
    _require_block(report_text, format_group_table(synthetic_rated), "the group table")


def test_report_quotes_the_current_tie_stress_table(report_text):
    _require_block(report_text, tie_stress_table(), "the tie stress table")


def test_report_states_the_synthetic_season_size(report_text, fake_season, synthetic_rated):
    assert f"{len(synthetic_rated):,}" in report_text and f"{len(fake_season):,}" in report_text


@pytest.mark.parametrize(
    "keyword",
    ["SYNTHETIC", "mid-rank", "row-wise", "isolation", "clip", "one decimal", "position group", "10 minutes", "Known limitations"],
)
def test_report_covers_the_requested_topics(report_text, keyword):
    assert keyword.lower() in report_text.lower(), f"report never mentions {keyword!r}"


def test_report_quotes_the_cli_summary_of_a_run_on_the_synthetic_season(report_text, fake_season, tmp_path, capsys):
    from src.scale.apply import main

    fake_season.to_parquet(tmp_path / "raw_test.parquet", index=False)
    assert main(["--method", "test", "--processed", str(tmp_path)]) == 0
    printed = capsys.readouterr().out.splitlines()
    summary = "\n".join(line for line in printed if not line.startswith("  wrote "))
    _require_block(report_text, summary, "the CLI summary")
