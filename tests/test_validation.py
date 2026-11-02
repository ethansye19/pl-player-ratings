"""The independent verifier (src/validation): it must run on a fresh checkout, where the frozen
pre-decision snapshot data/processed/before/ does not exist."""
from __future__ import annotations

import re

import pytest

from src.validation import before_after, validate


@pytest.mark.data
@pytest.mark.slow
def test_report_builds_without_the_before_snapshot(tmp_path, monkeypatch):
    """A fresh checkout has no data/processed/before/: the report is still written, every before value
    reads n/a instead of crashing, and the report lists every comparison that was skipped."""
    monkeypatch.setattr(before_after, "BEFORE", tmp_path / "no-before-snapshot")
    report = validate.build_report(skip_tests=True)

    for heading in ("## Headline numbers", "## 6. Findings and recommendations", "## 7. Before / after the Step 4 decisions"):
        assert heading in report
    assert "Before -> after the Step 4 decisions: **skipped**" in report
    assert "Before -> after the Step 4 decisions (details in section 7)" not in report, "no before/after table without the snapshot"
    for item in validate.BEFORE_AFTER_SKIPPED:
        assert report.count(f"- {item}") == 2, f"skipped comparison not listed in 6 (ii) and section 7: {item}"
    findings = report[report.index("## 6. Findings"):report.index("## 7. Before / after")]
    assert "n/a" in findings
    assert "GK, 3+ conceded" not in findings  # the event row is used, not leaked as a key
    assert "**H / short substitutes (builders' flag ii).** Before/after comparison skipped" in findings
    assert not re.search(r"\bnan\b", report), "a missing value was formatted as nan instead of n/a"


def test_missing_values_are_printed_as_n_a():
    assert validate._v(None) == "n/a"
    assert validate._v(float("nan"), "+.3f") == "n/a"
    assert validate._v(0.397, ".0f", pct=True) == "40%"
    assert validate._v(-0.1123, "+.3f") == "-0.112"
    assert validate._item({}, "own goal", 2) is None
    assert validate._item({"own goal": (5, 5.17, 4.41)}, "own goal", 2) == 4.41


def _write_json(path, obj):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_web_payload_check_skips_current_season_entries(tmp_path):
    """SYNTHETIC: web/data/index.json holds a 2015/16 match and a current-season FPL fixture (merged by the
    gameweek pipeline). The FPL fixture is counted and skipped, not looked up in the StatsBomb lineups."""
    import pandas as pd

    from src.validation.checks import web_payload_check
    from src.validation.data import Data

    lineups = pd.DataFrame({"match_id": [101, 101], "player_id": [1, 2], "position_group": ["MID", "FWD"], "minutes": [90.0, 5.0]})
    ratings = pd.DataFrame({"match_id": [101], "player_id": [1], "rating": [7.1]})
    data = Data(matches=pd.DataFrame(), lineups=lineups, ratings={"baseline": ratings, "vaep": ratings.assign(rating=6.4)})
    players = [{"player_id": 1, "position_group": "MID", "minutes": 90.0, "ratings": {"baseline": 7.1, "vaep": 6.4}},
               {"player_id": 2, "position_group": "FWD", "minutes": 5.0, "ratings": {"baseline": None, "vaep": None}}]
    _write_json(tmp_path / "2015-16" / "101.json", {"match_id": 101, "players": players})
    _write_json(tmp_path / "current" / "2026-27" / "77.json", {"match_id": 77, "players": [{"player_id": 305, "position_group": "DEF", "minutes": 90.0, "ratings": {"baseline": 6.0}}]})
    _write_json(tmp_path / "index.json", [{"season": "2015/16", "match_id": 101, "path": "2015-16/101.json"},
                                          {"season": "2026/27", "match_id": 77, "path": "current/2026-27/77.json"}])
    name, ok, detail = web_payload_check(data, tmp_path)
    assert ok, detail
    assert "1 matches, 2 player rows, 0 mismatches" in detail and "1 index entries from other seasons" in detail

    players[0]["ratings"]["vaep"] = 6.5  # a payload that disagrees with the parquet file is a mismatch
    _write_json(tmp_path / "2015-16" / "101.json", {"match_id": 101, "players": players})
    name, ok, detail = web_payload_check(data, tmp_path)
    assert not ok and "1 mismatches" in detail


def _git(*args) -> str:
    import subprocess

    from src.validation.data import ROOT
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def test_pre_decision_weights_come_from_main_history():
    """The 'before' team-context shares are computed with weights read from git: the commit must exist in
    this branch's history, its weights.py must be the one the Step 4 decision commit (216671e) replaced,
    and it must hold the old values."""
    import subprocess

    from src.baseline import weights as current
    from src.validation.data import ROOT

    commit = before_after.PRE_DECISION_WEIGHTS_COMMIT
    assert subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode == 0, f"{commit} is not in HEAD's history"
    assert _git("rev-parse", f"{commit}:src/baseline/weights.py") == _git("rev-parse", "216671e^:src/baseline/weights.py")
    changes = _git("log", "--format=%h", "--reverse", f"{commit}..HEAD", "--", "src/baseline/weights.py").split()
    assert changes and changes[0].startswith("216671e"), "the next change to weights.py after the commit must be the decision commit"

    old = before_after._old_weights()
    assert old is not None, f"could not read src/baseline/weights.py at {commit}"
    assert "fouls_won" in old.STATS and "penalties_scored" not in old.STATS
    assert old.WEIGHTS["DEF"]["goals_conceded"].weight == -0.80 and current.WEIGHTS["DEF"]["goals_conceded"].weight == -0.40
    assert old.WEIGHTS["MID"]["clean_sheet"].weight == 0.50 and current.WEIGHTS["MID"]["clean_sheet"].weight == 0.25


SYNTHETIC_PYTEST_OUTPUT = """\
........F.....E
=========================== short test summary info ===========================
FAILED tests/test_explain.py::test_every_number_in_the_explanation_appears_in_the_input[vaep] - AssertionError
FAILED tests/test_pipeline.py::test_nothing_is_written_when_checks_fail - assert False
ERROR tests/test_scale.py::test_ratings_bounded_3_to_10 - FileNotFoundError
2 failed, 400 passed, 1 error in 12.00s
"""


def test_failures_in_step_5_and_6_tests_are_real_failures(monkeypatch):
    """SYNTHETIC pytest output: test_explain.py and test_pipeline.py failures are no longer 'expected'."""
    import subprocess

    from src.validation import checks

    assert not hasattr(checks, "EXPECTED_FAILING_FILES")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, SYNTHETIC_PYTEST_OUTPUT, ""))
    res = checks.run_test_suite()
    assert res["failed"] == [
        "tests/test_explain.py::test_every_number_in_the_explanation_appears_in_the_input[vaep]",
        "tests/test_pipeline.py::test_nothing_is_written_when_checks_fail",
        "tests/test_scale.py::test_ratings_bounded_3_to_10",
    ]
    assert "expected_failures" not in res and "unexpected_failures" not in res
    assert res["summary"] == "2 failed, 400 passed, 1 error in 12.00s"
    text = "\n".join(validate.section_tests(res))
    assert "Failures (3)" in text and "test_explain.py::" in text and "test_pipeline.py::" in text
    assert "xpected" not in text, "no 'expected failures' wording"


def test_a_broken_pytest_run_is_never_reported_as_clean():
    from src.validation.checks import parse_failures

    assert parse_failures("no tests ran in 0.01s", returncode=4) == ["pytest exited with code 4 and reported no individual test"]
    assert parse_failures("10 passed in 1.00s", returncode=0) == []
    text = "\n".join(validate.section_tests({"command": "-m pytest -q", "seconds": 1.0, "summary": "10 passed in 1.00s", "failed": []}))
    assert "Failures: **none**" in text
