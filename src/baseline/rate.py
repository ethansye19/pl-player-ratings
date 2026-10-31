"""Baseline raw scores: ``raw_score = sum(WEIGHTS[position_group][stat] * stat)``.

``rate_match`` is a per-match rating function under the isolation rule (it only ever
looks at the requested match). ``rate_season`` drives it over all 380 matches, reading
events in small chunks; the CLI ``python -m src.baseline.rate`` writes
``data/processed/baseline_stats.parquet`` and ``data/processed/raw_baseline.parquet``.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import time

import numpy as np
import pandas as pd

from src.baseline.stats import EVENT_COLUMNS, KEY_COLUMNS, player_match_stats, select_match
from src.baseline.weights import STATS, WEIGHTS, weight_of
from src.contract import CONTRACT_COLUMNS, POSITION_GROUPS, validate_contract

log = logging.getLogger(__name__)

PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
STATS_FILE = "baseline_stats.parquet"
RAW_FILE = "raw_baseline.parquet"
SORT_KEYS = ["match_id", "team", "player_id"]


def weight_matrix() -> pd.DataFrame:
    """Position groups x STATS matrix of weights (0.0 where a stat is not in a profile)."""
    return pd.DataFrame(
        {stat: [weight_of(group, stat) for group in POSITION_GROUPS] for stat in STATS},
        index=list(POSITION_GROUPS),
        dtype="float64",
    )


_WEIGHT_MATRIX = weight_matrix()


def raw_scores(stats: pd.DataFrame) -> np.ndarray:
    """Weighted sum of the STATS columns of ``stats`` using each row's position_group."""
    unknown = set(stats["position_group"].unique()) - set(POSITION_GROUPS)
    if unknown:
        raise ValueError(f"unknown position_group values: {sorted(map(str, unknown))}")
    weights = _WEIGHT_MATRIX.reindex(stats["position_group"].to_numpy()).to_numpy(dtype="float64")
    values = stats[STATS].to_numpy(dtype="float64")
    return (values * weights).sum(axis=1)


def score_stats(stats: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Contract frame for one match from its stats table plus names/teams from lineups."""
    match_id = int(stats["match_id"].iloc[0])
    names = select_match(lineups, match_id, "lineups")[["player_id", "player_name", "team"]]
    out = stats[KEY_COLUMNS].merge(names, on="player_id", how="left", validate="one_to_one")
    out["raw_score"] = raw_scores(stats)
    out = out[CONTRACT_COLUMNS]
    return validate_contract(out)


def rate_match(match_id: int, events: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Contract-format raw scores for every player who appeared in ``match_id``.

    Only that match's rows of ``events`` and ``lineups`` are used; ``ValueError`` if the
    match is absent from either.
    """
    match_id = int(match_id)
    stats = player_match_stats(match_id, events, lineups)
    return score_stats(stats, lineups)


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def season_tables(processed_dir: pathlib.Path | str = PROCESSED_DIR, chunk_size: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stats, raw) for every match in ``matches.parquet``, rated one match at a time.

    Events are read ``chunk_size`` matches at a time (pyarrow filter; the file has one
    row group per match) so memory stays small. Both frames are sorted by
    ``match_id, team, player_id`` and aligned row for row.
    """
    processed_dir = pathlib.Path(processed_dir)
    matches = pd.read_parquet(processed_dir / "matches.parquet", columns=["match_id"])
    lineups = pd.read_parquet(processed_dir / "lineups.parquet")
    match_ids = sorted(int(m) for m in matches["match_id"].unique())
    log.info("rating %d matches from %s", len(match_ids), processed_dir)

    stats_parts: list[pd.DataFrame] = []
    raw_parts: list[pd.DataFrame] = []
    started = time.time()
    done = 0
    for chunk in _chunks(match_ids, max(1, int(chunk_size))):
        events = pd.read_parquet(
            processed_dir / "events.parquet", columns=EVENT_COLUMNS, filters=[("match_id", "in", chunk)]
        )
        for match_id in chunk:
            match_lineups = lineups.loc[lineups["match_id"] == match_id]
            stats = player_match_stats(match_id, events, match_lineups)
            raw = score_stats(stats, match_lineups)
            stats_parts.append(stats)
            raw_parts.append(raw)
        done += len(chunk)
        log.info("rated %d/%d matches (%.1fs)", done, len(match_ids), time.time() - started)

    stats_all = pd.concat(stats_parts, ignore_index=True)
    raw_all = pd.concat(raw_parts, ignore_index=True)
    order = np.lexsort((raw_all["player_id"].to_numpy(), raw_all["team"].to_numpy(), raw_all["match_id"].to_numpy()))
    stats_all = stats_all.iloc[order].reset_index(drop=True)
    raw_all = raw_all.iloc[order].reset_index(drop=True)
    validate_contract(raw_all)
    return stats_all, raw_all


def rate_season(processed_dir: pathlib.Path | str = PROCESSED_DIR, chunk_size: int = 20) -> pd.DataFrame:
    """Contract-format raw scores for the whole season (see ``season_tables``)."""
    return season_tables(processed_dir, chunk_size)[1]


def summary(raw: pd.DataFrame, stats: pd.DataFrame) -> str:
    lines = [
        f"matches: {raw['match_id'].nunique()}  appearances: {len(raw)}  stats: {len(STATS)}",
        f"{'group':5s} {'n':>6s} {'mean':>8s} {'sd':>8s} {'min':>8s} {'max':>8s} {'zeros':>7s}",
    ]
    for group in POSITION_GROUPS:
        part = raw.loc[raw["position_group"] == group, "raw_score"]
        zeros = float((part == 0).mean()) if len(part) else float("nan")
        lines.append(
            f"{group:5s} {len(part):6d} {part.mean():8.3f} {part.std():8.3f} {part.min():8.3f} {part.max():8.3f} {zeros:7.2%}"
        )
    top = raw.sort_values("raw_score", ascending=False).head(5)
    lines.append("top 5 raw scores: " + "; ".join(
        f"{r.player_name} ({r.position_group}, {r.team}, match {r.match_id}) {r.raw_score:.2f}" for r in top.itertuples()
    ))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute baseline per-match stats and raw scores for 2015/16.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="folder with the processed parquet files")
    parser.add_argument("--chunk-size", type=int, default=20, help="matches per events read")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    processed_dir = pathlib.Path(args.processed_dir)
    started = time.time()
    stats, raw = season_tables(processed_dir, args.chunk_size)
    stats.to_parquet(processed_dir / STATS_FILE, index=False)
    raw.to_parquet(processed_dir / RAW_FILE, index=False)
    print(summary(raw, stats))
    print(f"wrote {processed_dir / STATS_FILE} ({len(stats)} rows x {stats.shape[1]} cols)")
    print(f"wrote {processed_dir / RAW_FILE} ({len(raw)} rows x {raw.shape[1]} cols)")
    print(f"done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
