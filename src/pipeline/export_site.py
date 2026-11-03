"""Build the public GitHub Pages site under ``docs/``: the match-ratings page plus the
2015/16 StatsBomb payloads, and nothing else.

``python -m src.pipeline.export_site [--out docs] [--processed data/processed] [--web web]``

What goes in, and what never does:

* ``index.html``, ``app.js``, ``styles.css`` and ``img/`` are copied from ``web/``.
  ``web/sample/`` (synthetic) and ``web/data/`` are not copied.
* ``data/index.json`` and ``data/2015-16/<match_id>.json`` are written by
  :func:`src.pipeline.export_web.export_season`, which only knows the 2015/16 StatsBomb
  season. The current-season FPL output (``web/data/current/``) is never read, so it cannot
  leak into the site.
* ``.nojekyll`` so GitHub Pages serves the files as they are. It also marks the folder as one
  this module built: :func:`build_site` refuses to wipe a folder without it.

:func:`verify_site` re-reads everything under ``<out>/data`` and fails unless every index
entry and payload is season 2015/16 from StatsBomb Open Data, no ``current`` folder exists,
every JSON file is referenced by the index and every payload validates. The CLI runs it after
each build; ``tests/test_site.py`` runs it on the committed ``docs/`` folder, which is what
GitHub publishes.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import sys
from dataclasses import dataclass

from src.console import use_utf8_output
from src.pipeline.export_web import SEASON, SOURCE_STATSBOMB, export_season, season_dir, validate_payload

log = logging.getLogger(__name__)

DEFAULT_OUT = "docs"
PAGE_FILES = ("index.html", "app.js", "styles.css")
PAGE_DIRS = ("img",)
MARKER = ".nojekyll"
PUBLIC_SEASON = SEASON  # "2015/16"
PUBLIC_SOURCE = SOURCE_STATSBOMB  # "StatsBomb Open Data"
FORBIDDEN_DIR_NAMES = ("current",)  # the FPL pipeline's output folder name


@dataclass(frozen=True)
class SiteSummary:
    out_dir: pathlib.Path
    matches: int
    files: int
    bytes: int


def _check_inputs(web: pathlib.Path, out: pathlib.Path) -> None:
    for name in PAGE_FILES:
        if not (web / name).is_file():
            raise FileNotFoundError(f"{web / name} is missing")
    if out.exists():
        if out.resolve() == web.resolve():
            raise ValueError(f"refusing to build the site over the page source folder {web}")
        if any(out.iterdir()) and not (out / MARKER).is_file():
            raise ValueError(f"refusing to replace {out}: it is not empty and was not built by export_site "
                             f"(no {MARKER} marker). Choose another --out or empty the folder yourself.")


def build_site(out_dir: str | pathlib.Path = DEFAULT_OUT, processed_dir: str | pathlib.Path = "data/processed",
               web_dir: str | pathlib.Path = "web") -> SiteSummary:
    """Rebuild ``out_dir`` from scratch (an earlier build there is deleted first) and verify it."""
    out, web = pathlib.Path(out_dir), pathlib.Path(web_dir)
    _check_inputs(web, out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in PAGE_FILES:
        shutil.copyfile(web / name, out / name)
    for name in PAGE_DIRS:
        if (web / name).is_dir():
            shutil.copytree(web / name, out / name)
    entries = export_season(out / "data", processed_dir)
    (out / MARKER).write_text("", encoding="utf-8")
    verify_site(out)
    files = [p for p in out.rglob("*") if p.is_file()]
    summary = SiteSummary(out, len(entries), len(files), sum(p.stat().st_size for p in files))
    log.info("built %s: %d matches, %d files, %.2f MB", out, summary.matches, summary.files, summary.bytes / 1e6)
    return summary


def verify_site(out_dir: str | pathlib.Path) -> list:
    """Raise ``ValueError`` unless ``out_dir`` holds the page and ONLY 2015/16 StatsBomb payloads.

    Returns the index entries. Problems are collected and reported together (first 15 shown).
    """
    out = pathlib.Path(out_dir)
    problems: list[str] = []
    for name in PAGE_FILES:
        if not (out / name).is_file():
            problems.append(f"{name} is missing")
    if not (out / MARKER).is_file():
        problems.append(f"{MARKER} is missing")

    data = out / "data"
    index_path = data / "index.json"
    if not index_path.is_file():
        raise ValueError(f"{index_path} is missing")
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{index_path} must be a non-empty list")

    public_dir = season_dir(PUBLIC_SEASON)
    referenced = {index_path.resolve()}
    for entry in entries:
        label = f"index entry for match {entry.get('match_id')!r}"
        if entry.get("season") != PUBLIC_SEASON:
            problems.append(f"{label}: season {entry.get('season')!r} is not {PUBLIC_SEASON!r}")
        path = str(entry.get("path", ""))
        if not path.startswith(public_dir + "/") or any(part in FORBIDDEN_DIR_NAMES for part in path.split("/")):
            problems.append(f"{label}: path {path!r} is outside {public_dir}/")
        target = data / path
        referenced.add(target.resolve())
        if not target.is_file():
            problems.append(f"{label}: {path} is missing")
            continue
        payload = json.loads(target.read_text(encoding="utf-8"))
        try:
            validate_payload(payload)
        except ValueError as exc:
            problems.append(f"{path}: {exc}")
            continue
        if payload["season"] != PUBLIC_SEASON:
            problems.append(f"{path}: payload season {payload['season']!r} is not {PUBLIC_SEASON!r}")
        if payload["source"] != PUBLIC_SOURCE:
            problems.append(f"{path}: payload source {payload['source']!r} is not {PUBLIC_SOURCE!r}")

    for item in out.rglob("*"):
        rel = item.relative_to(out).as_posix()
        if item.is_dir() and item.name in FORBIDDEN_DIR_NAMES:
            problems.append(f"{rel}/: this folder must never be published")
        elif item.is_file() and item.suffix == ".json" and item.resolve() not in referenced:
            problems.append(f"{rel}: JSON file not referenced by data/index.json")
        elif item.is_file() and data in item.parents and item.suffix != ".json":
            problems.append(f"{rel}: only JSON payloads belong under data/")

    if problems:
        shown = "\n  ".join(problems[:15])
        raise ValueError(f"{len(problems)} problem(s) in {out}:\n  {shown}")
    return entries


def main(argv: list | None = None) -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.export_site",
                                     description="Build (or verify) the public GitHub Pages site: the page plus the "
                                                 "2015/16 StatsBomb payloads only.")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"site folder (default: {DEFAULT_OUT})")
    parser.add_argument("--processed", default="data/processed", help="folder with the processed parquet files")
    parser.add_argument("--web", default="web", help="folder with index.html, app.js, styles.css and img/")
    parser.add_argument("--verify-only", action="store_true", help="only verify an existing site folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.verify_only:
            entries = verify_site(args.out)
            print(f"{args.out}: {len(entries)} matches, all {PUBLIC_SEASON} from {PUBLIC_SOURCE}; no current-season files")
            return 0
        summary = build_site(args.out, args.processed, args.web)
    except (ValueError, FileNotFoundError) as exc:
        print(f"site build failed: {exc}", file=sys.stderr)
        return 1
    print(f"built {summary.out_dir}: {summary.matches} matches, {summary.files} files, {summary.bytes / 1e6:.2f} MB; "
          f"verified: only {PUBLIC_SEASON} from {PUBLIC_SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
