"""Re-score the 2015/16 season with the saved VAEP models: ``python -m src.vaep.score``.

Runs steps 5 and 6 of ``src.vaep.train`` only: cross-fitted action values
(``action_values.parquet``), raw scores (``raw_vaep.parquet``) and the holdout raw scores
(``raw_vaep_holdout.parquet``), using the fold and holdout models, the splits in
``vaep_splits.json`` and the StatsBomb xG already on disk. Nothing is refitted, and the models,
splits, features and labels are not touched. Use it after changing how probabilities become
action values (``src.vaep.value``), then run ``python -m src.scale.apply --method vaep``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from src.console import use_utf8_output
from src.vaep.config import ACTIONS_PATH, HOLDOUT_MODEL_PATH, LINEUPS_PATH, MATCHES_PATH, SPLITS_PATH, fold_model_path
from src.vaep.labels import shot_xg_lookup
from src.vaep.model import load_model
from src.vaep.splits import Split
from src.vaep.train import score_and_write
from src.vaep.value import VALUE_RULE

log = logging.getLogger("src.vaep.score")


def recorded_splits(payload: dict) -> tuple[Split, list[Split]]:
    """The holdout split and the folds (ordered by ``fold_id``) recorded in ``vaep_splits.json``."""
    holdout = Split("holdout", payload["holdout"]["train"], payload["holdout"]["score"])
    folds = sorted(payload["folds"], key=lambda f: int(f["fold_id"]))
    if [int(f["fold_id"]) for f in folds] != list(range(len(folds))):
        raise ValueError("vaep_splits.json: fold ids must be 0..n-1")
    return holdout, [Split(f"fold{int(f['fold_id'])}", f["train"], f["score"]) for f in folds]


def main(argv=None) -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.perf_counter()

    matches = pd.read_parquet(MATCHES_PATH)
    lineups = pd.read_parquet(LINEUPS_PATH)
    actions = pd.read_parquet(ACTIONS_PATH)
    holdout, folds = recorded_splits(json.loads(SPLITS_PATH.read_text(encoding="utf-8")))
    fold_models = [load_model(fold_model_path(k)) for k in range(len(folds))]
    holdout_model = load_model(HOLDOUT_MODEL_PATH)
    log.info("scoring %d actions with %d saved fold models and the holdout model (value rule %s)", len(actions), len(folds), VALUE_RULE)
    values, raw, raw_holdout = score_and_write(matches, lineups, actions, folds, fold_models, holdout, holdout_model, shot_xg_lookup())

    print(f"VAEP re-scored with the saved models in {time.perf_counter() - t0:.0f} s (value rule: {VALUE_RULE}; nothing refitted)")
    for name, mask in (
        ("open-play goal", (values["type_name"] == "shot") & (values["result_name"] == "success")),
        ("open-play miss", (values["type_name"] == "shot") & (values["result_name"] != "success")),
        ("penalty scored", (values["type_name"] == "shot_penalty") & (values["result_name"] == "success")),
        ("penalty missed", (values["type_name"] == "shot_penalty") & (values["result_name"] != "success")),
        ("keeper save", values["type_name"] == "keeper_save"),
    ):
        print(f"  mean VAEP value of a {name}: {values.loc[mask, 'vaep_value'].mean():+.4f} (n={int(mask.sum())})")
    print(f"  action_values: {len(values):,} actions; raw_vaep: {len(raw):,} player-matches; raw_vaep_holdout: {len(raw_holdout):,}")
    print("  next: python -m src.scale.apply --method vaep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
