"""CLI: fit the percentile scaler on a method's raw scores, then write the frozen scaler and the ratings.

    python -m src.scale.apply --method baseline|vaep [--processed data/processed]

Reads   <processed>/raw_<method>.parquet     interface contract, from the rating step
Writes  <processed>/scaler_<method>.json     the frozen PercentileScaler (see scaler.py)
        <processed>/ratings_<method>.parquet contract (plus any extra input columns) + rating,
                                             rated rows only (minutes >= 10), sorted by
                                             match_id, team, player_id

and prints rows rated, the mean rating (overall and per position group), the share of
10.0s, and the minimum and maximum rating. This is calibration: the scaler is fitted on the
whole file, so run it once per rating method, not once per match. Per-match rating then
goes through ``PercentileScaler.load(...).transform(...)``.

Exit status: 0 on success; 1 if the raw file is missing or breaks the interface contract
(nothing is written in that case).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.console import use_utf8_output
from src.contract import validate_contract
from src.scale.scaler import PercentileScaler
from src.scale.summary import format_cli_summary

DEFAULT_PROCESSED = Path("data") / "processed"
SORT_KEYS = ["match_id", "team", "player_id"]
_METHOD_NAME = re.compile(r"[A-Za-z0-9_]+")


def _method_name(text: str) -> str:
    if not _METHOD_NAME.fullmatch(text):
        raise argparse.ArgumentTypeError(
            f"invalid method name {text!r}: use letters, digits and underscores (for example baseline or vaep)"
        )
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.scale.apply",
        description="Fit the percentile scaler on raw_<method>.parquet and write scaler_<method>.json "
        "and ratings_<method>.parquet.",
    )
    parser.add_argument("--method", required=True, type=_method_name, help="rating method, for example baseline or vaep")
    parser.add_argument(
        "--processed",
        default=str(DEFAULT_PROCESSED),
        help="folder holding raw_<method>.parquet and receiving the outputs (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    use_utf8_output()
    args = build_parser().parse_args(argv)
    processed = Path(args.processed)
    raw_path = processed / f"raw_{args.method}.parquet"
    if not raw_path.is_file():
        print(f"error: {raw_path} not found; run the {args.method} rating step first", file=sys.stderr)
        return 1

    raw = pd.read_parquet(raw_path)
    try:
        scaler = PercentileScaler().fit(raw)
        rated = scaler.transform(raw)
    except ValueError as exc:          # ContractError is a ValueError
        print(f"error: cannot scale {raw_path}: {exc}", file=sys.stderr)
        return 1
    rated = rated.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)
    validate_contract(rated, with_rating=True)

    ratings_path = processed / f"ratings_{args.method}.parquet"
    tmp_path = ratings_path.with_name(ratings_path.name + ".tmp")
    try:
        rated.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, ratings_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    scaler_path = scaler.save(processed / f"scaler_{args.method}.json")

    for line in format_cli_summary(args.method, len(raw), rated):
        print(line)
    print(f"  wrote {scaler_path}")
    print(f"  wrote {ratings_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
