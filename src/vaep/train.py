"""Train the VAEP models and score the 2015/16 season. CLI: ``python -m src.vaep.train``.

Steps
-----
1. Build socceraction's ``games`` frame from ``matches.parquet``.
2. Compute VAEP features (``nb_prev_actions=3``) and labels per game once and cache
   them (``vaep_features.parquet``, ``vaep_labels.parquet``); reruns reuse the cache.
3. Fit one lightgbm VAEP model per split -- the matchweek 1-19 holdout and the 5
   cross-fitting folds -- with fixed seeds, and pickle them under ``models/``.
4. Record the splits in ``vaep_splits.json``.
5. Score every action of every match with the fold model that did not train on it
   (``action_values.parquet``, cross-fitted) and sum per player per match through
   ``src.vaep.rate.rate_match`` (``raw_vaep.parquet``).
6. Score the holdout matches (weeks 20-38) with the holdout model (``raw_vaep_holdout.parquet``).

Model quality on each split's never-seen games is written to ``models/vaep_metrics.json``;
``python -m src.vaep.report`` turns it into ``reports/vaep_splits.md``.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from src.contract import SEED, validate_contract
from src.vaep.config import (
    ACTION_VALUE_COLUMNS,
    ACTION_VALUES_PATH,
    ACTIONS_PATH,
    EARLY_STOPPING_ROUNDS,
    HOLDOUT_MODEL_PATH,
    LABELS,
    LINEUPS_PATH,
    MATCHES_PATH,
    METRICS_PATH,
    MODELS_DIR,
    N_FOLDS,
    NB_PREV_ACTIONS,
    RAW_VAEP_HOLDOUT_PATH,
    RAW_VAEP_PATH,
    SPLITS_PATH,
    TREE_PARAMS,
    VALIDATION_SHARE,
    fold_model_path,
)
from src.vaep.features import feature_columns, games_frame, load_or_build_features
from src.vaep.model import fit_model, new_model, save_model, validation_games
from src.vaep.rate import action_values_for_match, rate_match
from src.vaep.splits import Split, make_cv_splits, make_holdout_split, splits_payload

log = logging.getLogger("src.vaep.train")


def fit_split(split: Split, X: pd.DataFrame, y: pd.DataFrame, cols: list[str], seed: int = SEED) -> tuple:
    """Fit the model of one split and measure it on the split's score games.

    The training games are divided into fit games and a seeded 15 % of validation games
    (early stopping); the score games take no part in either and provide the quality
    numbers (AUC, Brier) reported in ``vaep_metrics.json``.
    """
    val_ids = validation_games(split.train, seed=seed)
    fit_ids = sorted(set(split.train) - set(val_ids))
    game_id = X["game_id"].to_numpy()
    is_fit = np.isin(game_id, fit_ids)
    is_val = np.isin(game_id, val_ids)
    is_score = np.isin(game_id, split.score)
    labels = list(LABELS)

    log.info(
        "%s: fitting on %d games / %d game states, validating on %d games, scoring %d games",
        split.name, len(fit_ids), int(is_fit.sum()), len(val_ids), len(split.score),
    )
    t0 = time.perf_counter()
    X_fit = X.loc[is_fit, cols].astype("float32")
    X_val = X.loc[is_val, cols].astype("float32")
    y_fit = y.loc[is_fit, labels]
    y_val = y.loc[is_val, labels]
    model, per_label = fit_model(X_fit, y_fit, X_val, y_val, seed=seed)
    fit_rate = {label: float(y_fit[label].mean()) for label in labels}
    del X_fit, X_val
    gc.collect()

    y_score = y.loc[is_score, labels]
    quality = model.score(X.loc[is_score, cols], y_score)
    metrics = {}
    for label in labels:
        truth = y_score[label].to_numpy(dtype="float64")
        metrics[label] = {
            "auroc": float(quality[label]["auroc"]),
            "brier": float(quality[label]["brier"]),
            "brier_base_rate": float(np.mean((truth - fit_rate[label]) ** 2)),  # predicting the fit-set rate
            "score_positive_rate": float(truth.mean()),
        }
    info = {
        "name": split.name,
        "n_train_games": len(split.train),
        "n_fit_games": len(fit_ids),
        "n_validation_games": len(val_ids),
        "n_score_games": len(split.score),
        "n_fit_rows": int(is_fit.sum()),
        "n_validation_rows": int(is_val.sum()),
        "n_score_rows": int(is_score.sum()),
        "fit_positive_rate": fit_rate,
        "labels": per_label,
        "metrics": metrics,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log.info(
        "%s: AUC scores %.3f / concedes %.3f, Brier scores %.5f / concedes %.5f (%.0fs)",
        split.name, metrics["scores"]["auroc"], metrics["concedes"]["auroc"],
        metrics["scores"]["brier"], metrics["concedes"]["brier"], info["seconds"],
    )
    return model, info


def score_games(game_ids, model_for_game, actions_by_game, lineups_by_game, home_of_game, fold_of_game=None, log_every=20):
    """Action values (optional) and contract raw scores for ``game_ids`` via ``rate_match``."""
    value_parts, raw_parts = [], []
    t0 = time.perf_counter()
    game_ids = sorted(int(g) for g in game_ids)
    for i, gid in enumerate(game_ids, 1):
        model = model_for_game(gid)
        if fold_of_game is not None:
            values = action_values_for_match(gid, actions_by_game[gid], model, home_team_id=home_of_game[gid])
            values["fold_id"] = np.int64(fold_of_game[gid])
            value_parts.append(values)
        raw_parts.append(rate_match(gid, actions_by_game[gid], lineups_by_game[gid], model, home_team_id=home_of_game[gid]))
        if i % log_every == 0 or i == len(game_ids):
            log.info("scored %d/%d games (%.0fs)", i, len(game_ids), time.perf_counter() - t0)
    values = pd.concat(value_parts, ignore_index=True) if value_parts else None
    raw = pd.concat(raw_parts, ignore_index=True)
    raw = raw.sort_values(["match_id", "team", "player_id"], kind="mergesort").reset_index(drop=True)
    return values, raw


def check_raw_against_lineups(raw: pd.DataFrame, lineups: pd.DataFrame, match_ids) -> None:
    validate_contract(raw)
    expected = lineups.loc[lineups["match_id"].isin(match_ids), ["match_id", "player_id"]]
    got = set(map(tuple, raw[["match_id", "player_id"]].to_numpy()))
    want = set(map(tuple, expected.to_numpy()))
    if got != want:
        raise RuntimeError(f"raw scores do not cover the lineups exactly: {len(got - want)} extra, {len(want - got)} missing")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild-features", action="store_true", help="ignore the cached features/labels")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t_start = time.perf_counter()

    matches = pd.read_parquet(MATCHES_PATH)
    lineups = pd.read_parquet(LINEUPS_PATH)
    actions = pd.read_parquet(ACTIONS_PATH)
    games = games_frame(matches)
    log.info("loaded %d matches, %d lineup rows, %d actions", len(matches), len(lineups), len(actions))

    # 1-2. features and labels (cached)
    template = new_model()
    cols = feature_columns(template)
    t0 = time.perf_counter()
    X, y = load_or_build_features(template, games, actions, force=args.rebuild_features)
    features_seconds = round(time.perf_counter() - t0, 1)
    if len(X) != len(actions):
        raise RuntimeError(f"feature cache has {len(X)} rows for {len(actions)} actions")

    # 4. splits (recorded before training so that the models and the file always agree)
    holdout = make_holdout_split(matches)
    folds = make_cv_splits(matches["match_id"].tolist(), n_folds=N_FOLDS, seed=SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(splits_payload(holdout, folds)), encoding="utf-8")
    log.info("splits: holdout %d/%d games, %d folds of %s score games",
             len(holdout.train), len(holdout.score), len(folds), [len(f.score) for f in folds])

    # 3. one model per split
    holdout_model, holdout_info = fit_split(holdout, X, y, cols)
    save_model(holdout_model, HOLDOUT_MODEL_PATH)
    fold_models, fold_infos = [], []
    for k, fold in enumerate(folds):
        model, info = fit_split(fold, X, y, cols)
        info["fold_id"] = k
        save_model(model, fold_model_path(k))
        fold_models.append(model)
        fold_infos.append(info)
    del X, y
    gc.collect()

    # 5. cross-fitted action values and raw scores for the whole season
    fold_of_game = {gid: k for k, fold in enumerate(folds) for gid in fold.score}
    home_of_game = {int(m): int(h) for m, h in zip(matches["match_id"], matches["home_team_id"])}
    actions_by_game = {int(gid): df for gid, df in actions.groupby("game_id", sort=True)}
    lineups_by_game = {int(mid): df for mid, df in lineups.groupby("match_id", sort=True)}
    all_ids = sorted(int(m) for m in matches["match_id"])
    log.info("scoring the season with the cross-fitting fold models")
    values, raw = score_games(
        all_ids, lambda gid: fold_models[fold_of_game[gid]], actions_by_game, lineups_by_game, home_of_game, fold_of_game
    )
    values = values[ACTION_VALUE_COLUMNS].sort_values(["game_id", "action_id"], kind="mergesort").reset_index(drop=True)
    for col in ("game_id", "action_id", "player_id", "team_id", "period_id", "fold_id"):
        values[col] = values[col].astype("int64")
    if not np.isfinite(values[["offensive_value", "defensive_value", "vaep_value"]].to_numpy()).all():
        raise RuntimeError("non-finite action values")
    check_raw_against_lineups(raw, lineups, all_ids)
    sums = values.groupby(["game_id", "player_id"])["vaep_value"].sum()
    keyed = raw.set_index(["match_id", "player_id"])["raw_score"]
    diff = (keyed - sums.reindex(keyed.index).fillna(0.0)).abs().max()
    if diff > 1e-9:
        raise RuntimeError(f"raw_score differs from the summed action values by up to {diff}")
    values.to_parquet(ACTION_VALUES_PATH, index=False)
    raw.to_parquet(RAW_VAEP_PATH, index=False)

    # 6. holdout matches scored by the holdout model
    log.info("scoring the holdout matches with the holdout model")
    _, raw_holdout = score_games(holdout.score, lambda gid: holdout_model, actions_by_game, lineups_by_game, home_of_game)
    check_raw_against_lineups(raw_holdout, lineups, holdout.score)
    raw_holdout.to_parquet(RAW_VAEP_HOLDOUT_PATH, index=False)

    runtime = round(time.perf_counter() - t_start, 1)
    metrics = {
        "seed": SEED,
        "versions": {name: version(name) for name in ("socceraction", "lightgbm", "pandas", "numpy")},
        "nb_prev_actions": NB_PREV_ACTIONS,
        "labels": list(LABELS),
        "n_feature_columns": len(cols),
        "n_games": len(all_ids),
        "n_actions": int(len(actions)),
        "tree_params": TREE_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "validation_share": VALIDATION_SHARE,
        "features_seconds": features_seconds,
        "runtime_seconds": runtime,
        "holdout": holdout_info,
        "folds": fold_infos,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"VAEP training done in {runtime / 60:.1f} min")
    print(f"  games {len(all_ids)}, actions {len(actions):,}, features {len(cols)} (cache step {features_seconds}s)")
    print(f"  holdout: train {len(holdout.train)} games (weeks 1-19), score {len(holdout.score)} games")
    print(f"  folds: {N_FOLDS} x {[len(f.score) for f in folds]} score games, seed {SEED}")
    print("  model quality on never-seen games (AUC / Brier):")
    for info in [holdout_info] + fold_infos:
        m = info["metrics"]
        rounds = {label: info["labels"][label]["best_iteration"] for label in LABELS}
        print(
            f"    {info['name']:<8} scores {m['scores']['auroc']:.3f} / {m['scores']['brier']:.5f}"
            f"   concedes {m['concedes']['auroc']:.3f} / {m['concedes']['brier']:.5f}   rounds {rounds}"
        )
    v = values["vaep_value"]
    print(f"  vaep_value: mean {v.mean():.5f}, sd {v.std():.4f}, min {v.min():.3f}, max {v.max():.3f}, "
          f"exactly zero {100 * (v == 0).mean():.1f}%")
    print(f"  raw_vaep: {len(raw):,} player-matches; raw_vaep_holdout: {len(raw_holdout):,}")
    print(f"  wrote {SPLITS_PATH.name}, {ACTION_VALUES_PATH.name}, {RAW_VAEP_PATH.name}, "
          f"{RAW_VAEP_HOLDOUT_PATH.name}, models/vaep_*.pkl, {METRICS_PATH.relative_to(METRICS_PATH.parents[1]).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
