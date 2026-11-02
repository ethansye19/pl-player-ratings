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
