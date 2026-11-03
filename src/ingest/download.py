"""Download StatsBomb Open Data for Premier League 2015/16 (competition 2, season 27).

Files are mirrored under ``data/raw/statsbomb/`` with the same layout as the open-data
repository (``matches/2/27.json``, ``events/<match_id>.json``, ``lineups/<match_id>.json``)
so that socceraction's local StatsBombLoader can read them. Nothing is transformed here.

Run:  .venv/Scripts/python -m src.ingest.download
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.console import use_utf8_output

log = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
COMMIT_URL = "https://api.github.com/repos/statsbomb/open-data/commits/master"
COMPETITION_ID = 2
SEASON_ID = 27
RAW_ROOT = Path("data") / "raw" / "statsbomb"


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=6, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "rating-system-ingest/0.1 (StatsBomb open data mirror)"
    return session


def fetch_json_file(session: requests.Session, relative: str, root: Path, force: bool = False) -> tuple[str, int, bool]:
    """Download one file to root/relative. Returns (relative, bytes, downloaded?)."""
    dest = root / relative
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return relative, dest.stat().st_size, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(BASE_URL + relative, timeout=120)
    response.raise_for_status()
    json.loads(response.content)  # refuse to keep anything that is not valid JSON
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(response.content)
    part.replace(dest)
    return relative, len(response.content), True


def upstream_commit(session: requests.Session) -> str | None:
    try:
        response = session.get(COMMIT_URL, timeout=30)
        if response.ok:
            return response.json().get("sha")
    except requests.RequestException:
        pass
    return None


def download_season(root: Path = RAW_ROOT, workers: int = 8, force: bool = False) -> dict:
    session = make_session()
    started = time.time()
    fetch_json_file(session, "competitions.json", root, force)
    matches_rel = f"matches/{COMPETITION_ID}/{SEASON_ID}.json"
    fetch_json_file(session, matches_rel, root, force)
    matches = json.loads((root / matches_rel).read_text(encoding="utf-8"))
    match_ids = sorted(int(m["match_id"]) for m in matches)
    log.info("%d matches listed for competition %d season %d", len(match_ids), COMPETITION_ID, SEASON_ID)

    wanted = [f"events/{mid}.json" for mid in match_ids] + [f"lineups/{mid}.json" for mid in match_ids]
    total_bytes, downloaded, failures = 0, 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_json_file, session, rel, root, force): rel for rel in wanted}
        for i, future in enumerate(as_completed(futures), 1):
            rel = futures[future]
            try:
                _, nbytes, was_downloaded = future.result()
                total_bytes += nbytes
                downloaded += int(was_downloaded)
            except Exception as exc:  # noqa: BLE001 - reported in the summary, nothing is substituted
                failures.append((rel, repr(exc)))
                log.error("failed %s: %r", rel, exc)
            if i % 50 == 0 or i == len(wanted):
                log.info("%d/%d files, %.0f MB so far, %.0fs", i, len(wanted), total_bytes / 1e6, time.time() - started)

    manifest = {
        "source": "StatsBomb Open Data (https://github.com/statsbomb/open-data)",
        "competition_id": COMPETITION_ID,
        "season_id": SEASON_ID,
        "upstream_commit": upstream_commit(session),
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_matches": len(match_ids),
        "n_files": len(wanted) + 2,
        "n_downloaded_this_run": downloaded,
        "total_bytes": total_bytes,
        "failures": failures,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=RAW_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = download_season(args.root, args.workers, args.force)
    print(
        f"matches: {manifest['n_matches']}  files: {manifest['n_files']}  "
        f"downloaded now: {manifest['n_downloaded_this_run']}  total: {manifest['total_bytes'] / 1e6:.0f} MB  "
        f"failures: {len(manifest['failures'])}  upstream commit: {manifest['upstream_commit']}"
    )
    if manifest["failures"]:
        for rel, err in manifest["failures"]:
            print(f"  FAILED {rel}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
