"""The public GitHub Pages site (docs/), built by src/pipeline/export_site.py.

The unit tests run on the SYNTHETIC processed directory from tests/test_web_export.py and a
copy of the page files. The last two tests check the committed docs/ folder itself, because
that folder is what GitHub publishes: it must hold the 2015/16 StatsBomb payloads and nothing
from the current season, and its page files must match web/.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from src.pipeline.export_site import MARKER, PAGE_FILES, build_site, main, verify_site
from src.pipeline.export_web import SEASON, SOURCE_STATSBOMB
from tests.conftest import ROOT
from tests.test_web_export import build_processed

WEB = ROOT / "web"
DOCS = ROOT / "docs"


@pytest.fixture
def web_with_current(tmp_path):
    """The page files plus a fake current-season export, the way web/ looks after the FPL pipeline ran."""
    web = tmp_path / "web"
    web.mkdir()
    for name in PAGE_FILES:
        shutil.copyfile(WEB / name, web / name)
    shutil.copytree(WEB / "img", web / "img")
    (web / "sample").mkdir()
    (web / "sample" / "index.json").write_text("[]", encoding="utf-8")
    current = web / "data" / "current" / "2026-27"
    current.mkdir(parents=True)
    (current / "1.json").write_text(json.dumps({"source": "Fantasy Premier League API", "season": "2026/27"}), encoding="utf-8")
    (web / "data" / "current" / "state.json").write_text("{}", encoding="utf-8")
    (web / "data" / "index.json").write_text(
        json.dumps([{"season": "2026/27", "match_id": 1, "path": "current/2026-27/1.json"}]), encoding="utf-8")
    return web


@pytest.fixture
def processed(tmp_path) -> str:
    return build_processed(tmp_path / "processed")


def test_build_site_publishes_the_page_and_only_the_2015_16_payloads(tmp_path, web_with_current, processed):
    out = tmp_path / "site"
    summary = build_site(out, processed, web_with_current)

    index = json.loads((out / "data" / "index.json").read_text(encoding="utf-8"))
    assert summary.matches == len(index) == 2
    assert {e["season"] for e in index} == {SEASON}
    assert all(e["path"].startswith("2015-16/") for e in index)
    for entry in index:
        payload = json.loads((out / "data" / entry["path"]).read_text(encoding="utf-8"))
        assert payload["season"] == SEASON and payload["source"] == SOURCE_STATSBOMB

    json_files = {p.relative_to(out).as_posix() for p in out.rglob("*.json")}
    assert json_files == {"data/index.json", "data/2015-16/101.json", "data/2015-16/102.json"}
    assert not [p for p in out.rglob("*") if p.name == "current"], "the FPL output folder must never be copied"
    assert not (out / "sample").exists(), "synthetic sample data is not published"
    for name in PAGE_FILES:
        assert (out / name).read_bytes() == (WEB / name).read_bytes()
    assert (out / "img" / "statsbomb-logo.png").is_file()
    assert (out / MARKER).is_file()
    assert summary.files == len([p for p in out.rglob("*") if p.is_file()]) and summary.bytes > 0


def test_a_rebuild_replaces_an_earlier_build_in_place(tmp_path, web_with_current, processed):
    out = tmp_path / "site"
    build_site(out, processed, web_with_current)
    (out / "data" / "2015-16" / "stale.json").write_text("{}", encoding="utf-8")
    build_site(out, processed, web_with_current)
    assert not (out / "data" / "2015-16" / "stale.json").exists()
    assert len(verify_site(out)) == 2


def test_build_site_refuses_to_wipe_a_folder_it_did_not_build(tmp_path, web_with_current, processed):
    out = tmp_path / "other"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="not built by export_site"):
        build_site(out, processed, web_with_current)
    assert (out / "keep.txt").read_text(encoding="utf-8") == "mine"
    with pytest.raises(ValueError, match="page source folder"):
        build_site(web_with_current, processed, web_with_current)
    assert (web_with_current / "index.html").is_file()


@pytest.mark.parametrize("name, plant", [
    ("a current-season payload", lambda out: _write(out / "data" / "current" / "2026-27" / "1.json", "{}")),
    ("a stray JSON file", lambda out: _write(out / "data" / "2015-16" / "999.json", "{}")),
    ("a non-JSON file under data", lambda out: _write(out / "data" / "scaler_baseline.parquet", "x")),
    ("an index entry from another season", lambda out: _edit_index(out, lambda e: e.__setitem__("season", "2026/27"))),
    ("an index path outside 2015-16", lambda out: _edit_index(out, lambda e: e.__setitem__("path", "current/" + e["path"]))),
    ("a payload with another source", lambda out: _edit_payload(out, lambda p: p.__setitem__("source", "Fantasy Premier League API"))),
    ("an invalid payload", lambda out: _edit_payload(out, lambda p: p.__setitem__("players", []))),
    ("a missing marker", lambda out: (out / MARKER).unlink()),
    ("a missing page file", lambda out: (out / "app.js").unlink()),
])
def test_verify_site_rejects(tmp_path, web_with_current, processed, name, plant):
    out = tmp_path / "site"
    build_site(out, processed, web_with_current)
    plant(out)
    with pytest.raises(ValueError):
        verify_site(out)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _edit_index(out, change):
    index_path = out / "data" / "index.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    change(entries[0])
    index_path.write_text(json.dumps(entries), encoding="utf-8")


def _edit_payload(out, change):
    entry = json.loads((out / "data" / "index.json").read_text(encoding="utf-8"))[0]
    path = out / "data" / entry["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_builds_then_verifies(tmp_path, web_with_current, processed, capsys):
    out = tmp_path / "site"
    assert main(["--out", str(out), "--processed", processed, "--web", str(web_with_current)]) == 0
    assert "2 matches" in capsys.readouterr().out
    assert main(["--out", str(out), "--verify-only"]) == 0
    assert "no current-season files" in capsys.readouterr().out
    _write(out / "data" / "current" / "x.json", "{}")
    assert main(["--out", str(out), "--verify-only"]) == 1
    assert "current" in capsys.readouterr().err


# ------------------------------------------------------------------ the committed site (what GitHub publishes)
def test_committed_docs_folder_holds_the_2015_16_season_and_nothing_current():
    assert DOCS.is_dir(), "docs/ (the GitHub Pages site) is missing: run python -m src.pipeline.export_site"
    entries = verify_site(DOCS)
    assert len(entries) == 380
    assert {e["season"] for e in entries} == {SEASON}
    names = {p.name for p in DOCS.rglob("*")}
    assert "current" not in names and "state.json" not in names and "scaler_baseline.json" not in names
    assert not any(p.suffix in (".parquet", ".pkl", ".log") for p in DOCS.rglob("*"))


def test_committed_docs_page_files_match_web():
    for name in PAGE_FILES:
        assert (DOCS / name).read_bytes() == (WEB / name).read_bytes(), f"docs/{name} is stale: rebuild the site"
    for source in (WEB / "img").iterdir():
        assert (DOCS / "img" / source.name).read_bytes() == source.read_bytes(), f"docs/img/{source.name} is stale"


def test_committed_docs_data_is_not_gitignored():
    """A bare `data/` ignore rule would silently drop docs/data from the commit and publish an empty page."""
    for path in ("docs/data/index.json", "docs/index.html", "docs/img/statsbomb-logo.png", "web/img/statsbomb-logo.png"):
        result = subprocess.run(["git", "check-ignore", "-v", path], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 1, f"{path} must be tracked, but git ignores it: {result.stdout.strip() or result.stderr.strip()}"
    for path in ("data/processed/matches.parquet", "web/data/index.json", "web/data/current/state.json", "logs/pipeline.log"):
        ignored = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True, text=True)
        assert ignored.returncode == 0, f"{path} must stay ignored (data/, web/data/ and logs/ are never committed)"
