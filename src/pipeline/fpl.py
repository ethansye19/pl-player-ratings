"""Fantasy Premier League API client (the only data source for the current season).

Endpoints used (all public, no key):
- GET https://fantasy.premierleague.com/api/bootstrap-static/   gameweeks (events), teams, players (elements)
- GET https://fantasy.premierleague.com/api/fixtures/?event=<gw> fixtures of a gameweek with scores
- GET https://fantasy.premierleague.com/api/event/<gw>/live/     per-player stats for the gameweek, with an
                                                                 ``explain`` block that names the fixture(s)
- GET https://fantasy.premierleague.com/api/element-summary/<id>/ one player's season so far, one ``history`` row
                                                                 per fixture (starts, minutes, saves, goals
                                                                 conceded, ..., ``opponent_team``, ``was_home``);
                                                                 fetched only for players whose per-fixture numbers
                                                                 or club the gameweek feed cannot give

``FplClient`` is the protocol the pipeline depends on; tests use a fake with the same shapes.
``HttpFplClient`` keeps its requests polite: at most one every ``MIN_REQUEST_INTERVAL`` seconds, a
generic User-Agent, three attempts with back-off, and ``Retry-After`` honoured on HTTP 429.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api/"
USER_AGENT = "rating-system/0.2 (football player ratings)"  # generic: no personal contact details
MIN_REQUEST_INTERVAL = 1.0  # seconds between two requests to the API
MAX_RETRY_AFTER = 120.0  # longest Retry-After (seconds) the client will wait for


class FplClient(Protocol):
    def bootstrap(self) -> dict[str, Any]: ...

    def fixtures(self, gw: int) -> list[dict[str, Any]]: ...

    def live(self, gw: int) -> dict[str, Any]: ...

    def element_summary(self, element_id: int) -> dict[str, Any]: ...


class HttpFplClient:
    """Live client. Optionally mirrors every response under ``cache_dir`` for reproducibility."""

    def __init__(self, cache_dir: Path | str | None = None, timeout: float = 60.0, min_interval: float = MIN_REQUEST_INTERVAL) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.min_interval = min_interval
        self.requests_made = 0
        self._last_request: float | None = None

    def _wait_turn(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()
        self.requests_made += 1

    def _get(self, path: str, cache_name: str) -> Any:
        url = BASE_URL + path
        for attempt in range(3):
            self._wait_turn()
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 429 and attempt < 2:  # rate limited: wait as asked, then retry
                    try:
                        delay = float(response.headers.get("Retry-After", 10))
                    except ValueError:
                        delay = 10.0
                    log.warning("FPL API rate limit on %s; waiting %.0f s", url, min(delay, MAX_RETRY_AFTER))
                    time.sleep(min(delay, MAX_RETRY_AFTER))
                    continue
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"FPL API request failed: {url}: {exc}") from exc
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"FPL API request failed: {url}: still rate limited after 3 attempts")
        if self.cache_dir:
            target = self.cache_dir / cache_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def bootstrap(self) -> dict[str, Any]:
        return self._get("bootstrap-static/", "bootstrap-static.json")

    def fixtures(self, gw: int) -> list[dict[str, Any]]:
        return self._get(f"fixtures/?event={int(gw)}", f"fixtures_gw{int(gw)}.json")

    def live(self, gw: int) -> dict[str, Any]:
        return self._get(f"event/{int(gw)}/live/", f"live_gw{int(gw)}.json")

    def element_summary(self, element_id: int) -> dict[str, Any]:
        """One player's ``history`` (a row per fixture played this season); mirrored as ``element-summary/<id>.json``."""
        return self._get(f"element-summary/{int(element_id)}/", f"element-summary/{int(element_id)}.json")


def finished_gameweeks(bootstrap: dict[str, Any]) -> list[int]:
    """Gameweek ids that are finished and whose data has been checked, ascending."""
    return sorted(int(e["id"]) for e in bootstrap["events"] if e.get("finished") and e.get("data_checked"))


def latest_finished_gameweek(bootstrap: dict[str, Any]) -> int:
    done = finished_gameweeks(bootstrap)
    if not done:
        raise RuntimeError("no finished, data-checked gameweek in bootstrap-static")
    return done[-1]


def season_label(fixtures: list[dict[str, Any]]) -> str:
    """'2026/27' from the earliest kickoff year of the given fixtures (August start)."""
    years = sorted(int(f["kickoff_time"][:4]) for f in fixtures if f.get("kickoff_time"))
    if not years:
        raise RuntimeError("fixtures carry no kickoff_time; cannot derive the season label")
    months = sorted((f["kickoff_time"][:7] for f in fixtures if f.get("kickoff_time")))
    start = years[0] if int(months[0][5:7]) >= 7 else years[0] - 1
    return f"{start}/{str(start + 1)[-2:]}"
