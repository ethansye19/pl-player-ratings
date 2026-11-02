"""Current-season pipeline: rate the latest finished FPL gameweek with the baseline model.

    python -m src.pipeline.gameweek [--gw N] [--out web/data/current]

Steps: pull bootstrap-static, pick the latest finished and data-checked gameweek (or
``--gw``), build a contract-format frame for every finished gameweek of the season so
far (one row per player-fixture with minutes > 0), fit the percentile scaler on all of
them (season-to-date reference), rate the target gameweek, run the validity checks and,
only if every check passes, write one web payload per fixture plus ``index.json``.
If a check fails nothing is written and the failures are printed (exit code 1).

Double gameweeks: a player's ``explain`` block lists each fixture with the stats FPL
itemises per fixture (minutes, goals, assists, clean sheets, goals conceded, cards,
saves, penalties, own goals); stats FPL only reports per gameweek (xG, xGC, tackles,
CBI, recoveries) are split across the player's fixtures in proportion to minutes.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.contract import (
    CONTRACT_COLUMNS,
    MEAN_TOLERANCE,
    MIN_MINUTES,
    RATING_DECIMALS,
    RATING_MAX,
    RATING_MIN,
    TARGET_MEAN,
    validate_contract,
)
from src.pipeline.export_web import validate_payload
from src.pipeline.fpl import FplClient, HttpFplClient, finished_gameweeks, latest_finished_gameweek, season_label
from src.pipeline.fpl_mapping import (
    FPL_DECIMAL_FIELDS,
    FPL_MINUTES_FIELD,
    FPL_POSITION_GROUPS,
    FPL_STARTS_FIELD,
    FPL_STAT_MAP,
    FPL_WEIGHTS,
)
from src.scale.scaler import PercentileScaler

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("web") / "data" / "current"
SOURCE = "Fantasy Premier League API"
CREDITS = "Data: Fantasy Premier League API (https://fantasy.premierleague.com)"
GAMEWEEK_MEAN_TOLERANCE = 0.3  # single gameweek; the +/- 0.1 rule is for a full season
PER_FIXTURE_IDENTIFIERS = {
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals", "penalties_saved",
    "penalties_missed", "yellow_cards", "red_cards", "saves",
}


@dataclass
class RunResult:
    gameweek: int
    ok: bool
    failures: list[str] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)
    ratings: pd.DataFrame | None = None
    skipped: bool = False  # True when only_if_new found nothing newer than the published output


STATE_FILE = "state.json"


def existing_state(out_dir: Path | str) -> dict[str, Any] | None:
    """What is currently published in out_dir: {'gameweek', 'fixtures', 'rated_rows', ...} or None."""
    out_dir = Path(out_dir)
    state_path = out_dir / STATE_FILE
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    index_path = out_dir / "index.json"
    if not index_path.exists():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        rated = 0
        for entry in index:
            payload = json.loads((out_dir / entry["path"]).read_text(encoding="utf-8"))
            rated += sum(1 for p in payload["players"] if any(v is not None for v in p["ratings"].values()))
        return {"gameweek": max(int(e["matchweek"]) for e in index), "fixtures": len(index), "rated_rows": rated}
    except (ValueError, OSError, KeyError):
        return None


def _num(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _player_fixtures(element: dict[str, Any], fixture_ids: set[int]) -> list[tuple[int, int]]:
    """[(fixture_id, minutes)] from the explain block, falling back to the gameweek total."""
    out = []
    for block in element.get("explain") or []:
        fid = int(block["fixture"])
        if fid not in fixture_ids:
            continue
        minutes = next((int(s["value"]) for s in block.get("stats", []) if s.get("identifier") == "minutes"), None)
        out.append((fid, minutes if minutes is not None else int(element["stats"].get(FPL_MINUTES_FIELD, 0))))
    return out


def gameweek_frame(gw: int, client: FplClient, bootstrap: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Contract frame (+ team_id, is_starter, side, stat columns) for one gameweek, and its fixtures."""
    fixtures = [f for f in client.fixtures(gw) if f.get("finished")]
    fixture_by_id = {int(f["id"]): f for f in fixtures}
    teams = {int(t["id"]): t for t in bootstrap["teams"]}
    elements = {int(e["id"]): e for e in bootstrap["elements"]}
    live = client.live(gw)

    rows = []
    for item in live["elements"]:
        stats = item["stats"]
        total_minutes = int(stats.get(FPL_MINUTES_FIELD, 0) or 0)
        if total_minutes <= 0:
            continue
        element = elements.get(int(item["id"]))
        if element is None:
            continue
        played = _player_fixtures(item, set(fixture_by_id))
        if not played:
            continue
        per_fixture_itemised = {
            int(b["fixture"]): {s["identifier"]: _num(s.get("value")) for s in b.get("stats", [])}
            for b in item.get("explain") or []
        }
        team_id = int(element["team"])
        group = FPL_POSITION_GROUPS[int(element["element_type"])]
        for fid, minutes in played:
            if minutes <= 0:
                continue
            fixture = fixture_by_id[fid]
            share = minutes / total_minutes if total_minutes else 1.0
            itemised = per_fixture_itemised.get(fid, {})
            row: dict[str, Any] = {
                "match_id": fid,
                "player_id": int(element["id"]),
                "player_name": element.get("known_name") or element.get("web_name") or f"{element.get('first_name', '')} {element.get('second_name', '')}".strip(),
                "team": teams[team_id]["name"],
                "team_id": team_id,
                "position_group": group,
                "minutes": float(minutes),
                "is_starter": bool(int(stats.get(FPL_STARTS_FIELD, 0) or 0) >= 1) if len(played) == 1 else bool(itemised.get("starts", stats.get(FPL_STARTS_FIELD, 0))),
                "side": "home" if int(fixture["team_h"]) == team_id else "away",
                "position": element.get("element_type"),
                "jersey_number": element.get("squad_number"),
            }
            for stat, fpl_field in FPL_STAT_MAP.items():
                if len(played) == 1:
                    value = _num(stats.get(fpl_field))
                elif fpl_field in PER_FIXTURE_IDENTIFIERS:
                    value = itemised.get(fpl_field, 0.0)
                else:
                    value = _num(stats.get(fpl_field)) * share
                row[stat] = value
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"gameweek {gw}: no player appearances in the live feed")
    weights = {g: {s: w.weight for s, w in table.items()} for g, table in FPL_WEIGHTS.items()}
    frame["raw_score"] = [
        sum(weights[r.position_group].get(stat, 0.0) * float(getattr(r, stat)) for stat in FPL_STAT_MAP) for r in frame.itertuples(index=False)
    ]
    frame["match_id"] = frame["match_id"].astype("int64")
    frame["player_id"] = frame["player_id"].astype("int64")
    frame = frame.sort_values(["match_id", "team", "player_id"]).reset_index(drop=True)
    validate_contract(frame)
    return frame, fixtures


def run_checks(
    ratings: pd.DataFrame,
    fixtures: list[dict[str, Any]],
    teams: list[dict[str, Any]],
    appearances: pd.DataFrame | None = None,
) -> list[str]:
    """Validity checks on a rated gameweek frame; an empty list means pass.

    ``ratings`` holds the rated rows (minutes >= 10); ``appearances`` holds every player-fixture
    with minutes > 0, i.e. the frame before the scaler dropped the short appearances (both need
    ``match_id, player_id, player_name, team_id, minutes, is_starter``). Without ``appearances``
    the rated rows are taken to be every appearance.

    Only teams with a fixture in the gameweek are checked, so a blank gameweek (fewer than 20
    teams playing) can publish. For every fixture, each side must have rated players and exactly
    11 starters in the FPL data, and every starter with 10+ minutes must be rated. These mirror the
    11-starters rule of ``validate_payload``, so a missing starter is reported here as a failure
    instead of raising later while the payloads are built.
    """
    failures: list[str] = []
    names = {int(t["id"]): t["name"] for t in teams}
    if ratings.empty:
        return ["no rated players"]
    apps = ratings if appearances is None else appearances
    present = set(int(t) for t in ratings["team_id"].unique())
    playing = {int(f["team_h"]) for f in fixtures} | {int(f["team_a"]) for f in fixtures}
    for team_id in sorted(playing - present):
        failures.append(f"team {names.get(team_id, team_id)} has a fixture but no rated players")

    rating = ratings["rating"].to_numpy(dtype="float64")
    if not np.isfinite(rating).all():
        failures.append("ratings contain NaN or inf")
    else:
        if rating.min() < RATING_MIN or rating.max() > RATING_MAX:
            failures.append(f"ratings outside [{RATING_MIN}, {RATING_MAX}]: min {rating.min():.1f}, max {rating.max():.1f}")
        scaled = rating * 10 ** RATING_DECIMALS
        if not np.allclose(scaled, np.round(scaled), atol=1e-6):
            failures.append("ratings are not rounded to one decimal")
        mean = float(rating.mean())
        if abs(mean - TARGET_MEAN) > GAMEWEEK_MEAN_TOLERANCE:
            failures.append(f"gameweek mean rating {mean:.2f} is outside {TARGET_MEAN} +/- {GAMEWEEK_MEAN_TOLERANCE}")
    if (ratings["minutes"] < MIN_MINUTES).any():
        failures.append("rated rows with fewer than 10 minutes")
    rated_keys = set(zip(ratings["match_id"].astype("int64"), ratings["player_id"].astype("int64")))
    for fixture in fixtures:
        fid = int(fixture["id"])
        for team_id in (int(fixture["team_h"]), int(fixture["team_a"])):
            team = names.get(team_id, team_id)
            part = ratings[(ratings["match_id"] == fid) & (ratings["team_id"] == team_id)]
            if part.empty:
                failures.append(f"fixture {fid}: {team} has no rated players")
                continue
            starters = apps[(apps["match_id"] == fid) & (apps["team_id"] == team_id) & apps["is_starter"].astype(bool)]
            if len(starters) != 11:
                failures.append(f"fixture {fid}: {team} has {len(starters)} starters in the FPL data (expected 11)")
            for r in starters.itertuples(index=False):
                if r.minutes >= MIN_MINUTES and (int(r.match_id), int(r.player_id)) not in rated_keys:
                    failures.append(f"fixture {fid}: {team} starter {r.player_name} ({r.minutes:.0f} min) is not rated")
    return failures


def _payload(fixture: dict[str, Any], rated: pd.DataFrame, unrated: pd.DataFrame, teams: dict[int, dict], season: str, gw: int) -> dict[str, Any]:
    home_id, away_id = int(fixture["team_h"]), int(fixture["team_a"])
    players = []
    for frame, has_rating in ((rated, True), (unrated, False)):
        for r in frame.itertuples(index=False):
            players.append(
                {
                    "player_id": int(r.player_id),
                    "player_name": str(r.player_name),
                    "team_id": int(r.team_id),
                    "team": str(r.team),
                    "side": str(r.side),
                    "position_id": None,
                    "position": {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}.get(int(r.position), None) if r.position is not None else None,
                    "position_group": str(r.position_group),
                    "is_starter": bool(r.is_starter),
                    "jersey_number": None if r.jersey_number is None or (isinstance(r.jersey_number, float) and np.isnan(r.jersey_number)) else int(r.jersey_number),
                    "minutes": float(r.minutes),
                    "ratings": {"baseline": float(r.rating) if has_rating else None},
                    "raw_scores": {"baseline": float(r.raw_score)},
                }
            )
    players.sort(key=lambda p: (p["side"] != "home", not p["is_starter"], -p["minutes"], p["player_name"]))
    return {
        "schema_version": 1,
        "source": SOURCE,
        "season": season,
        "competition": "Premier League",
        "match_id": int(fixture["id"]),
        "match_date": str(fixture.get("kickoff_time", ""))[:10],
        "matchweek": int(gw),
        "home": {"team_id": home_id, "team": teams[home_id]["name"], "score": fixture.get("team_h_score"), "formation": None},
        "away": {"team_id": away_id, "team": teams[away_id]["name"], "score": fixture.get("team_a_score"), "formation": None},
        "methods": ["baseline"],
        "players": players,
        "credits": CREDITS,
    }


def run(
    client: FplClient,
    out_dir: Path | str = DEFAULT_OUT,
    gw: int | None = None,
    only_if_new: bool = False,
    allow_regression: bool = False,
) -> RunResult:
    """Rate a gameweek and publish it to out_dir.

    only_if_new: do nothing (RunResult.skipped) unless the target gameweek is newer than
    the one already published in out_dir.
    allow_regression: by default a run refuses to replace published output with an older
    gameweek, fewer fixtures or fewer rated rows for the same gameweek.
    """
    out_dir = Path(out_dir)
    bootstrap = client.bootstrap()
    target = int(gw) if gw is not None else latest_finished_gameweek(bootstrap)
    published = existing_state(out_dir)
    if only_if_new and published is not None and int(published["gameweek"]) >= target:
        log.info("gameweek %d is already published (latest finished: %d); nothing to do", published["gameweek"], target)
        return RunResult(target, True, [], [], None, skipped=True)
    done = [g for g in finished_gameweeks(bootstrap) if g <= target]
    if target not in done:
        return RunResult(target, False, [f"gameweek {target} is not finished and data-checked"])
    teams = {int(t["id"]): t for t in bootstrap["teams"]}

    frames, fixtures_by_gw = [], {}
    for g in done:
        frame, fixtures = gameweek_frame(g, client, bootstrap)
        frame["gameweek"] = g
        frames.append(frame)
        fixtures_by_gw[g] = fixtures
        log.info("gameweek %d: %d fixtures, %d appearances", g, len(fixtures), len(frame))
    season_rows = pd.concat(frames, ignore_index=True)
    scaler = PercentileScaler().fit(season_rows[CONTRACT_COLUMNS])
    target_rows = season_rows[season_rows["gameweek"] == target].reset_index(drop=True)
    rated = scaler.transform(target_rows)
    unrated = target_rows[target_rows["minutes"] < MIN_MINUTES]
    fixtures = fixtures_by_gw[target]

    failures = run_checks(rated, fixtures, list(teams.values()), appearances=target_rows)
    if published is not None and not allow_regression:
        if target < int(published["gameweek"]):
            failures.append(f"refusing to replace published gameweek {published['gameweek']} with older gameweek {target}")
        elif target == int(published["gameweek"]):
            if len(fixtures) < int(published.get("fixtures", 0)):
                failures.append(f"refusing to overwrite gameweek {target}: {len(fixtures)} fixtures now vs {published['fixtures']} published")
            if len(rated) < int(published.get("rated_rows", 0)):
                failures.append(f"refusing to overwrite gameweek {target}: {len(rated)} rated rows now vs {published['rated_rows']} published")
    if failures:
        for f in failures:
            log.error("check failed: %s", f)
        return RunResult(target, False, failures, [], rated)

    season = season_label(fixtures)
    folder = season.replace("/", "-")
    payloads = []
    for fixture in fixtures:
        fid = int(fixture["id"])
        payload = _payload(fixture, rated[rated["match_id"] == fid], unrated[unrated["match_id"] == fid], teams, season, target)
        validate_payload(payload)
        payloads.append(payload)

    # everything is validated; now write each file to a temp name and swap it into place
    written: list[Path] = []
    (out_dir / folder).mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        path = out_dir / folder / f"{payload['match_id']}.json"
        _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=1))
        written.append(path)
    index = [
        {
            "season": p["season"], "match_id": p["match_id"], "match_date": p["match_date"], "matchweek": p["matchweek"],
            "home": p["home"]["team"], "away": p["away"]["team"], "home_score": p["home"]["score"], "away_score": p["away"]["score"],
            "methods": p["methods"], "path": f"{folder}/{p['match_id']}.json",
        }
        for p in payloads
    ]
    index_path = out_dir / "index.json"
    _atomic_write(index_path, json.dumps(index, ensure_ascii=False, indent=1))
    written.append(index_path)
    scaler.save(out_dir / "scaler_baseline.json")
    written.append(out_dir / "scaler_baseline.json")
    state = {
        "gameweek": target, "season": season, "fixtures": len(payloads), "rated_rows": int(len(rated)),
        "appearances": int(len(target_rows)), "fit_gameweeks": done, "fit_rows": int(len(season_rows)),
        "written_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write(out_dir / STATE_FILE, json.dumps(state, indent=1))
    written.append(out_dir / STATE_FILE)
    return RunResult(target, True, [], written, rated)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def merge_site_index(out_dir: Path | str = DEFAULT_OUT, site_dir: Path | str = Path("web") / "data") -> int:
    """Merge the gameweek index into the page's main index (web/data/index.json).

    Entries of the same season are replaced; paths are rewritten relative to ``site_dir``
    so the page can load current-season payloads next to the 2015/16 ones.
    """
    out_dir, site_dir = Path(out_dir), Path(site_dir)
    gw_index_path = out_dir / "index.json"
    if not gw_index_path.exists():
        return 0
    new_entries = json.loads(gw_index_path.read_text(encoding="utf-8"))
    prefix = out_dir.resolve().relative_to(site_dir.resolve()).as_posix()
    for entry in new_entries:
        entry["path"] = f"{prefix}/{entry['path']}"
    seasons = {e["season"] for e in new_entries}
    site_index_path = site_dir / "index.json"
    existing = json.loads(site_index_path.read_text(encoding="utf-8")) if site_index_path.exists() else []
    kept = [e for e in existing if e.get("season") not in seasons or not str(e.get("path", "")).startswith(prefix)]
    kept = [e for e in kept if not (e.get("season") in seasons and e.get("matchweek") in {n["matchweek"] for n in new_entries})]
    merged = kept + new_entries
    merged.sort(key=lambda e: (str(e.get("season")), int(e.get("matchweek") or 0), str(e.get("match_date")), int(e["match_id"])))
    site_index_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(new_entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gw", type=int, default=None, help="gameweek to rate (default: latest finished)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=Path("data") / "raw" / "fpl", help="mirror API responses here")
    parser.add_argument("--if-new", action="store_true", help="do nothing unless a newer gameweek than the published one has finished")
    parser.add_argument("--allow-regression", action="store_true", help="allow replacing published output with fewer fixtures/rows or an older gameweek")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run(HttpFplClient(cache_dir=args.cache), out_dir=args.out, gw=args.gw, only_if_new=args.if_new, allow_regression=args.allow_regression)
    if result.skipped:
        print(f"nothing to do: latest finished gameweek {result.gameweek} is already published in {args.out}")
        return 0
    if not result.ok:
        print(f"gameweek {result.gameweek}: checks FAILED, nothing written")
        for f in result.failures:
            print(f"  - {f}")
        return 1
    r = result.ratings
    print(f"gameweek {result.gameweek}: {len(r)} rated player-fixtures in {r['match_id'].nunique()} fixtures, "
          f"mean {r['rating'].mean():.2f}, by group "
          + ", ".join(f"{g} {v:.2f}" for g, v in r.groupby('position_group')['rating'].mean().items())
          + f"; wrote {len(result.written)} files to {args.out}")
    try:
        merged = merge_site_index(args.out)
        if merged:
            print(f"merged {merged} fixtures into web/data/index.json for the page")
    except ValueError:
        pass  # out_dir is not under web/data; nothing to merge
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
