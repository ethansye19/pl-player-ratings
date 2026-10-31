"""Build the processed parquet files for Premier League 2015/16 from the raw StatsBomb
JSON mirrored by ``src.ingest.download``.

Outputs (data/processed/): matches.parquet, teams.parquet, players.parquet,
lineups.parquet, events.parquet, actions.parquet, plus reports/ingest.md.

Run:  .venv/Scripts/python -m src.ingest.build [--limit N] [--only lineups]
      --only lineups rebuilds matches/teams/players/lineups and the report without
      redoing events.parquet and actions.parquet (which take ~8 minutes).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from socceraction import spadl
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.spadl.statsbomb import convert_to_actions

from src.contract import POSITION_GROUPS
from src.ingest.download import COMPETITION_ID, RAW_ROOT, SEASON_ID
from src.ingest.events import TypeRegistry, flatten_match_events, records_to_table
from src.ingest.lineups import build_match_lineups
from src.ingest.positions import POSITION_NAMES

log = logging.getLogger(__name__)

PROCESSED = Path("data") / "processed"
REPORTS = Path("reports")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def match_rows(matches_json: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in matches_json:
        rows.append(
            {
                "match_id": int(m["match_id"]),
                "match_date": m["match_date"],
                "kick_off": m["kick_off"],
                "match_week": int(m["match_week"]),
                "home_team_id": int(m["home_team"]["home_team_id"]),
                "home_team": m["home_team"]["home_team_name"],
                "away_team_id": int(m["away_team"]["away_team_id"]),
                "away_team": m["away_team"]["away_team_name"],
                "home_score": int(m["home_score"]),
                "away_score": int(m["away_score"]),
                "stadium": (m.get("stadium") or {}).get("name"),
                "referee": (m.get("referee") or {}).get("name"),
                "competition": m["competition"]["competition_name"],
                "season": m["season"]["season_name"],
            }
        )
    return pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)


def formations(events_json: list[dict[str, Any]]) -> dict[int, int]:
    """team_id -> starting formation, from the Starting XI events."""
    out = {}
    for event in events_json:
        if event["type"]["name"] == "Starting XI":
            out[int(event["team"]["id"])] = int(event["tactics"]["formation"])
    return out


def build(
    raw_root: Path = RAW_ROOT,
    out_dir: Path = PROCESSED,
    limit: int | None = None,
    only: str = "all",
) -> dict[str, Any]:
    started = time.time()
    heavy = only == "all"  # events.parquet + actions.parquet
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = match_rows(load_json(raw_root / "matches" / str(COMPETITION_ID) / f"{SEASON_ID}.json"))
    if limit:
        matches = matches.head(limit).copy()
    loader = StatsBombLoader(getter="local", root=str(raw_root))

    registry = TypeRegistry()
    lineup_rows: list[dict[str, Any]] = []
    action_frames: list[pd.DataFrame] = []
    socceraction_minutes: list[dict[str, Any]] = []
    anomalies: list[str] = []
    notes: list[str] = []
    position_names_seen: dict[int, set[str]] = {}
    home_formation, away_formation = {}, {}
    n_events = 0

    # ---- pass 1: lineups, formations, (SPADL actions, event schema discovery)
    for i, m in enumerate(matches.itertuples(index=False), 1):
        mid = int(m.match_id)
        events_json = load_json(raw_root / "events" / f"{mid}.json")
        lineups_json = load_json(raw_root / "lineups" / f"{mid}.json")

        rows, problems, fixed = build_match_lineups(mid, lineups_json, events_json)
        lineup_rows.extend(rows)
        anomalies.extend(problems)
        notes.extend(fixed)
        for team in lineups_json:
            for player in team["lineup"]:
                for q in player.get("positions") or []:
                    position_names_seen.setdefault(int(q["position_id"]), set()).add(q["position"])

        forms = formations(events_json)
        home_formation[mid] = forms.get(int(m.home_team_id))
        away_formation[mid] = forms.get(int(m.away_team_id))

        if heavy:
            records = flatten_match_events(events_json, mid)
            registry.update(records)
            n_events += len(records)
            actions = spadl.add_names(convert_to_actions(loader.events(mid), int(m.home_team_id)))
            action_frames.append(actions)

        players = loader.players(mid)
        for p in players.itertuples(index=False):
            socceraction_minutes.append(
                {"match_id": mid, "player_id": int(p.player_id), "sa_minutes": float(p.minutes_played), "sa_is_starter": bool(p.is_starter)}
            )

        if i % 20 == 0 or i == len(matches):
            log.info("pass 1: %d/%d matches, %d events, %d lineup rows, %.0fs", i, len(matches), n_events, len(lineup_rows), time.time() - started)

    matches["home_formation"] = matches["match_id"].map(home_formation)
    matches["away_formation"] = matches["match_id"].map(away_formation)

    lineups = pd.DataFrame(lineup_rows).sort_values(["match_id", "team", "player_id"]).reset_index(drop=True)
    lineups["minutes"] = lineups["minutes"].astype("float64")

    teams = (
        pd.concat(
            [
                matches[["home_team_id", "home_team"]].rename(columns={"home_team_id": "team_id", "home_team": "team_name"}),
                matches[["away_team_id", "away_team"]].rename(columns={"away_team_id": "team_id", "away_team": "team_name"}),
            ]
        )
        .drop_duplicates()
        .sort_values("team_id")
        .reset_index(drop=True)
    )
    players = (
        lineups.groupby("player_id")
        .agg(player_name=("player_name", "first"), player_full_name=("player_full_name", "first"), appearances=("match_id", "size"))
        .reset_index()
        .sort_values("player_id")
        .reset_index(drop=True)
    )

    matches.to_parquet(out_dir / "matches.parquet", index=False)
    teams.to_parquet(out_dir / "teams.parquet", index=False)
    players.to_parquet(out_dir / "players.parquet", index=False)
    lineups.to_parquet(out_dir / "lineups.parquet", index=False)

    if heavy:
        actions_df = pd.concat(action_frames, ignore_index=True).sort_values(["game_id", "action_id"]).reset_index(drop=True)
        actions_df.to_parquet(out_dir / "actions.parquet", index=False)
        n_actions = len(actions_df)

        # ---- pass 2: write events.parquet with one schema, one row group per match
        schema = registry.schema()
        with pq.ParquetWriter(out_dir / "events.parquet", schema, compression="zstd") as writer:
            for i, mid in enumerate(matches["match_id"], 1):
                records = flatten_match_events(load_json(raw_root / "events" / f"{mid}.json"), int(mid))
                writer.write_table(records_to_table(records, schema))
                if i % 50 == 0 or i == len(matches):
                    log.info("pass 2: %d/%d matches written to events.parquet, %.0fs", i, len(matches), time.time() - started)
        n_event_columns = len(schema.names)
    else:
        events_meta = pq.ParquetFile(out_dir / "events.parquet")
        n_events, n_event_columns = events_meta.metadata.num_rows, len(events_meta.schema_arrow.names)
        n_actions = pq.ParquetFile(out_dir / "actions.parquet").metadata.num_rows

    # ---- cross-checks
    sa = pd.DataFrame(socceraction_minutes)
    check = lineups.merge(sa, on=["match_id", "player_id"], how="outer", indicator=True)
    both = check[check["_merge"] == "both"]
    diff = (both["minutes"] - both["sa_minutes"]).abs()
    starters_per_team_match = lineups[lineups["is_starter"]].groupby(["match_id", "team"]).size()
    position_name_mismatches = {
        pid: sorted(names) for pid, names in position_names_seen.items() if names != {POSITION_NAMES.get(pid)}
    }

    summary = {
        "matches": int(len(matches)),
        "teams": int(teams["team_id"].nunique()),
        "events": int(n_events),
        "event_columns": int(n_event_columns),
        "players": int(players["player_id"].nunique()),
        "lineup_rows": int(len(lineups)),
        "actions": int(n_actions),
        "team_matches_with_11_starters": f"{int((starters_per_team_match == 11).sum())}/{2 * len(matches)}",
        "position_groups": lineups["position_group"].value_counts().reindex(list(POSITION_GROUPS)).fillna(0).astype(int).to_dict(),
        "minutes_vs_socceraction": {
            "players_compared": int(len(both)),
            "only_in_ours": int((check["_merge"] == "left_only").sum()),
            "only_in_socceraction": int((check["_merge"] == "right_only").sum()),
            "mean_abs_diff_min": round(float(diff.mean()), 2) if len(both) else None,
            "median_abs_diff_min": round(float(diff.median()), 2) if len(both) else None,
            "max_abs_diff_min": round(float(diff.max()), 2) if len(both) else None,
            "n_diff_over_2_min": int((diff > 2).sum()),
            "n_diff_over_10_min": int((diff > 10).sum()),
            "starter_flag_disagreements": int((both["is_starter"] != both["sa_is_starter"]).sum()),
        },
        "position_name_mismatches": position_name_mismatches,
        "anomalies": anomalies,
        "lineup_interval_corrections": notes,
        "seconds": round(time.time() - started),
    }
    write_report(summary, out_dir)
    return summary


def write_report(summary: dict[str, Any], out_dir: Path) -> None:
    REPORTS.mkdir(exist_ok=True)
    mv = summary["minutes_vs_socceraction"]
    lines = [
        "# Ingestion report: StatsBomb Open Data, Premier League 2015/16",
        "",
        f"Built from `{RAW_ROOT}` (competition {COMPETITION_ID}, season {SEASON_ID}) into `{out_dir}`.",
        "Data: StatsBomb Open Data (https://github.com/statsbomb/open-data).",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| matches | {summary['matches']} |",
        f"| teams | {summary['teams']} |",
        f"| events | {summary['events']:,} ({summary['event_columns']} columns) |",
        f"| SPADL actions | {summary['actions']:,} |",
        f"| players who appeared | {summary['players']} |",
        f"| player-match appearances | {summary['lineup_rows']:,} |",
        f"| team-matches with 11 starters | {summary['team_matches_with_11_starters']} |",
        f"| appearances by position group | {summary['position_groups']} |",
        "",
        "## Minutes cross-check against socceraction's `minutes_played`",
        "",
        "Ours: on-pitch spans from the Starting XI / Substitution / Player Off / Player On / red",
        "card events, period lengths from the Half End timestamps (stoppage time counts, to the",
        "second; treatment time off the pitch does not).",
        "socceraction: whole minutes from the Starting XI / Substitution / Half End events, which",
        "misreads substitutions at exactly 45:00 of the second half and ignores Player Off gaps.",
        "",
        f"- players compared: {mv['players_compared']} (only in ours: {mv['only_in_ours']}, only in socceraction: {mv['only_in_socceraction']})",
        f"- |difference|: mean {mv['mean_abs_diff_min']} min, median {mv['median_abs_diff_min']} min, max {mv['max_abs_diff_min']} min; "
        f"more than 2 min apart: {mv['n_diff_over_2_min']}, more than 10 min apart: {mv['n_diff_over_10_min']}",
        f"- starter flag disagreements: {mv['starter_flag_disagreements']}",
        "",
        "## Position names seen in the data vs `src/ingest/positions.py`",
        "",
        "none" if not summary["position_name_mismatches"] else "\n".join(f"- {k}: {v}" for k, v in summary["position_name_mismatches"].items()),
        "",
        "## Anomalies",
        "",
        "none" if not summary["anomalies"] else "\n".join(f"- {a}" for a in summary["anomalies"]),
        "",
        "## Lineup-interval inconsistencies corrected from the events",
        "",
        "Minutes come from the events; these players' lineup position intervals disagreed by",
        "more than a minute (intervals that ignore a dismissal or a half-time substitution).",
        "",
        "none"
        if not summary["lineup_interval_corrections"]
        else "\n".join(f"- {n}" for n in summary["lineup_interval_corrections"]),
        "",
    ]
    (REPORTS / "ingest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_ROOT)
    parser.add_argument("--out", type=Path, default=PROCESSED)
    parser.add_argument("--limit", type=int, default=None, help="only the first N matches (smoke test)")
    parser.add_argument("--only", choices=["all", "lineups"], default="all", help="'lineups' skips events/actions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = build(args.raw, args.out, args.limit, args.only)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
