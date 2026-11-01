"""Scheduler safeguards of the FPL pipeline: do nothing when nothing new has finished, and
never replace published output with worse data. Uses the synthetic FPL client."""
import json

from src.pipeline.gameweek import existing_state, run
from tests.test_pipeline import FakeFplClient


def _published_rated_rows(out_dir) -> int:
    index = json.loads((out_dir / "index.json").read_text())
    return sum(
        sum(1 for p in json.loads((out_dir / e["path"]).read_text())["players"] if p["ratings"]["baseline"] is not None)
        for e in index
    )


def test_second_run_with_only_if_new_does_nothing(tmp_path):
    first = run(FakeFplClient(), out_dir=tmp_path)
    assert first.ok and not first.skipped
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.json")}
    second = run(FakeFplClient(), out_dir=tmp_path, only_if_new=True)
    assert second.skipped is True and second.ok is True and second.written == []
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.json")} == stamps, "no file may be touched"
    assert existing_state(tmp_path)["gameweek"] == 2


def test_a_newer_gameweek_is_published_with_only_if_new(tmp_path):
    run(FakeFplClient(gameweeks=(1,)), out_dir=tmp_path)
    assert existing_state(tmp_path)["gameweek"] == 1
    result = run(FakeFplClient(gameweeks=(1, 2)), out_dir=tmp_path, only_if_new=True)
    assert result.ok and not result.skipped and result.gameweek == 2
    assert existing_state(tmp_path)["gameweek"] == 2


def test_published_output_is_not_replaced_by_fewer_rated_rows(tmp_path):
    full = run(FakeFplClient(), out_dir=tmp_path)
    assert full.ok
    before = _published_rated_rows(tmp_path)

    poorer = FakeFplClient()
    live = poorer._live[2]["elements"]
    dropped = 0
    for element in list(live):
        if dropped >= 15:
            break
        if 10 <= element["stats"]["minutes"] < 60:  # substitutes who would have been rated
            live.remove(element)
            dropped += 1
    assert dropped == 15
    result = run(poorer, out_dir=tmp_path)
    assert result.ok is False
    assert any("rated rows" in f for f in result.failures), result.failures
    assert _published_rated_rows(tmp_path) == before, "published files must be untouched"

    forced = run(poorer, out_dir=tmp_path, allow_regression=True)
    assert forced.ok is True
    assert _published_rated_rows(tmp_path) == before - dropped


def test_an_older_gameweek_never_replaces_a_newer_one(tmp_path):
    run(FakeFplClient(), out_dir=tmp_path)
    result = run(FakeFplClient(), out_dir=tmp_path, gw=1)
    assert result.ok is False
    assert any("older gameweek" in f for f in result.failures), result.failures
    assert existing_state(tmp_path)["gameweek"] == 2
