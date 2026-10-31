"""Write ``reports/vaep_splits.md`` from the VAEP outputs. CLI: ``python -m src.vaep.report``.

The report documents the holdout and fold definitions, the model settings, model
quality on never-seen games (AUC and Brier per label), the distribution of the action
values and of ``raw_score`` per position group, and an eye test of the best and worst
player-matches. Every number is computed from ``data/processed`` at run time.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from src.contract import POSITION_GROUPS
from src.vaep.config import (
    ACTION_VALUES_PATH,
    LABELS,
    MATCHES_PATH,
    METRICS_PATH,
    RAW_VAEP_HOLDOUT_PATH,
    RAW_VAEP_PATH,
    REPORT_PATH,
    SPLITS_PATH,
)
from src.vaep.splits import HOLDOUT_LAST_TRAIN_WEEK

log = logging.getLogger("src.vaep.report")

QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def md_table(df: pd.DataFrame, decimals: dict | int = 3) -> str:
    """A GitHub-flavoured Markdown table; numeric columns are right-aligned."""
    cols = list(df.columns)

    def fmt(col, value):
        nd = decimals if isinstance(decimals, int) else decimals.get(col, decimals.get("default", 3))
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.{nd}f}"
        return str(value)

    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in cols]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(c, v) for c, v in zip(cols, row)) + " |")
    return "\n".join(lines)


def distribution(series: pd.Series) -> dict:
    s = series.astype("float64")
    out = {"n": int(len(s)), "mean": float(s.mean()), "sd": float(s.std()), "min": float(s.min())}
    for q in QUANTILES:
        out[f"p{int(q * 100)}"] = float(s.quantile(q))
    out["max"] = float(s.max())
    out["share zero"] = float((s == 0).mean())
    return out


def split_table(splits: dict, matches: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    week = matches.set_index("match_id")["match_week"]
    rows = []
    entries = [("holdout", splits["holdout"], metrics["holdout"])] + [
        (f"fold{f['fold_id']}", f, info) for f, info in zip(splits["folds"], metrics["folds"])
    ]
    for name, split, info in entries:
        train_weeks, score_weeks = week[split["train"]], week[split["score"]]
        rows.append(
            {
                "split": name,
                "train games": len(split["train"]),
                "of which fit": info["n_fit_games"],
                "early-stop validation": info["n_validation_games"],
                "score games": len(split["score"]),
                "train matchweeks": f"{train_weeks.min()}-{train_weeks.max()}",
                "score matchweeks": f"{score_weeks.min()}-{score_weeks.max()}",
                "score games per week (min-max)": f"{score_weeks.value_counts().min()}-{score_weeks.value_counts().max()}",
                "model": "models/vaep_holdout.pkl" if name == "holdout" else f"models/vaep_{name}.pkl",
            }
        )
    return pd.DataFrame(rows)


def quality_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for info in [metrics["holdout"]] + metrics["folds"]:
        for label in LABELS:
            m, fit = info["metrics"][label], info["labels"][label]
            rows.append(
                {
                    "split": info["name"],
                    "label": label,
                    "fit rows": info["n_fit_rows"],
                    "fit positive rate": info["fit_positive_rate"][label],
                    "rounds": fit["best_iteration"],
                    "val logloss": fit["val_logloss"],
                    "held-out rows": info["n_score_rows"],
                    "held-out positive rate": m["score_positive_rate"],
                    "AUC": m["auroc"],
                    "Brier": m["brier"],
                    "Brier (base rate)": m["brier_base_rate"],
                }
            )
    table = pd.DataFrame(rows)
    folds = table[table["split"] != "holdout"]
    means = folds.groupby("label", sort=False)[["AUC", "Brier", "Brier (base rate)"]].mean().reset_index()
    means.insert(0, "split", "fold mean")
    return pd.concat([table, means], ignore_index=True)


def eye_test(raw: pd.DataFrame, matches: pd.DataFrame, n: int, lowest: bool) -> pd.DataFrame:
    frame = raw.merge(
        matches[["match_id", "match_week", "home_team", "away_team", "home_score", "away_score"]], on="match_id", how="left"
    )
    is_home = frame["team"] == frame["home_team"]
    frame["opponent"] = np.where(is_home, frame["away_team"], frame["home_team"])
    frame["venue"] = np.where(is_home, "H", "A")
    frame["result"] = frame["home_score"].astype(str) + "-" + frame["away_score"].astype(str)
    frame = frame.sort_values(["raw_score", "match_id", "player_id"], ascending=[lowest, True, True]).head(n)
    out = frame[["player_name", "team", "position_group", "match_id", "match_week", "opponent", "venue", "result", "minutes", "raw_score"]]
    out = out.rename(columns={"player_name": "player", "position_group": "pos", "match_week": "week"}).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def group_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in POSITION_GROUPS:
        row = {"position_group": group}
        row.update(distribution(raw.loc[raw["position_group"] == group, "raw_score"]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_report() -> str:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    matches = pd.read_parquet(MATCHES_PATH)
    values = pd.read_parquet(ACTION_VALUES_PATH)
    raw = pd.read_parquet(RAW_VAEP_PATH)
    raw_holdout = pd.read_parquet(RAW_VAEP_HOLDOUT_PATH)
    week = matches.set_index("match_id")["match_week"]

    parts: list[str] = []
    add = parts.append
    add("# VAEP: train/score splits, model settings and model quality\n")
    add(
        f"Generated by `python -m src.vaep.report` on {dt.date.today().isoformat()} from the outputs of "
        f"`python -m src.vaep.train` (StatsBomb Open Data, Premier League 2015/16: {metrics['n_games']} matches, "
        f"{metrics['n_actions']:,} SPADL actions). Training run time: {metrics['runtime_seconds'] / 60:.1f} min "
        f"(feature step {metrics['features_seconds']:.0f} s).\n"
    )

    # ------------------------------------------------------------------ splits
    add("## Splits\n")
    add(
        "Two kinds of split, both recorded with their full match-id lists in `data/processed/vaep_splits.json` "
        "(`{\"holdout\": {\"train\", \"score\"}, \"folds\": [{\"fold_id\", \"train\", \"score\"}, ...]}`):\n"
    )
    hold = splits["holdout"]
    add(
        f"- **Holdout** (chronological): train on matchweeks 1-{HOLDOUT_LAST_TRAIN_WEEK} "
        f"({len(hold['train'])} matches), score matchweeks {HOLDOUT_LAST_TRAIN_WEEK + 1}-{week.max()} "
        f"({len(hold['score'])} matches). Output: `raw_vaep_holdout.parquet`.\n"
        f"- **{len(splits['folds'])}-fold cross-fitting**: a seeded shuffle (seed {metrics['seed']}) of all "
        f"{metrics['n_games']} matches into {len(splits['folds'])} folds of {sorted({len(f['score']) for f in splits['folds']})} "
        "matches. The model of fold *k* is trained on the other folds and scores only fold *k*, so every match "
        "is scored exactly once by a model that never saw it. Outputs: `action_values.parquet` (with `fold_id`) "
        "and `raw_vaep.parquet`.\n"
        f"- Inside every split, a seeded {metrics['validation_share']:.0%} of the *training* games is set aside "
        "for early stopping (whole games, never rows, because consecutive game states share labels); the rest "
        "fits the trees. Score games are never used for fitting or round selection.\n"
    )
    add(md_table(split_table(splits, matches, metrics)) + "\n")

    # ---------------------------------------------------------- model settings
    v = metrics["versions"]
    add("## Model settings\n")
    add(
        f"- `socceraction {v['socceraction']}` `VAEP(nb_prev_actions={metrics['nb_prev_actions']})` with the default "
        f"feature transformers (`xfns_default`: action type, result, type x result and body-part one-hots, time, "
        f"start/end location, polar coordinates, movement, possession-team flags, time and space deltas, score line) "
        f"-> {metrics['n_feature_columns']} feature columns per game state (the action and its "
        f"{metrics['nb_prev_actions'] - 1} predecessors), pitch flipped so the acting team attacks left to right.\n"
        f"- Labels: `scores` / `concedes` = the acting team scores / concedes within the next 10 actions "
        "(socceraction's defaults). Values: `offensive_value` = change in P(scores), `defensive_value` = "
        "-change in P(concedes), `vaep_value` = their sum (`socceraction.vaep.formula`).\n"
        f"- Learner: `lightgbm {v['lightgbm']}` `LGBMClassifier`, one per label, fitted through `VAEP.fit(learner=\"lightgbm\")` "
        f"with early stopping on binary log loss (patience {metrics['early_stopping_rounds']} rounds) on the "
        "validation games. socceraction's own `val_size` option is not used: it samples validation rows "
        "(not games) and relies on the `early_stopping_rounds` fit argument that lightgbm 4 removed.\n"
        f"- Determinism: `SEED = {metrics['seed']}` seeds the fold shuffle, the validation-game draw, socceraction's "
        "row shuffle (`np.random.seed`) and lightgbm (`random_state`, `deterministic=True`, `force_row_wise=True`), "
        "so a rerun reproduces every model and value bit for bit.\n"
        f"- Features are cast to float32 for fitting; scoring uses the float64 features computed on the fly from "
        "the match's actions, exactly as `src.vaep.rate.rate_match` does.\n"
    )
    params = pd.DataFrame({"parameter": list(metrics["tree_params"]), "value": [str(x) for x in metrics["tree_params"].values()]})
    add(md_table(params) + "\n")

    # ------------------------------------------------------------ model quality
    add("## Model quality on never-seen games\n")
    add(
        "AUC and Brier score of each split's model on the games it scores (held out from fitting and early "
        "stopping). `Brier (base rate)` is the Brier score of always predicting the fit set's positive rate; a "
        "useful model is below it. `rounds` is the boosting round early stopping selected.\n"
    )
    quality = quality_table(metrics)
    add(md_table(quality, {"default": 3, "fit positive rate": 4, "held-out positive rate": 4, "val logloss": 5, "Brier": 5, "Brier (base rate)": 5}) + "\n")
    fold_rows = quality[~quality["split"].isin(["holdout", "fold mean"])]
    worst_auc = fold_rows.groupby("label")["AUC"].min()
    beats = (fold_rows["Brier"] < fold_rows["Brier (base rate)"]).all()
    capped = [f"{i['name']}/{label}" for i in [metrics["holdout"]] + metrics["folds"] for label in LABELS if i["labels"][label]["hit_round_cap"]]
    add(
        f"Lowest fold AUC: scores {worst_auc['scores']:.3f}, concedes {worst_auc['concedes']:.3f}. "
        f"Every fold model {'beats' if beats else 'does NOT beat'} the base-rate Brier score on both labels. "
        + ("No fit reached the round cap." if not capped else f"Fits that reached the round cap: {', '.join(capped)}.")
        + "\n"
    )

    # ------------------------------------------------- action value distribution
    add("## Distribution of `vaep_value` (cross-fitted, all matches)\n")
    dist = pd.DataFrame([distribution(values["vaep_value"])])
    add(md_table(dist, {"default": 4, "n": 0}) + "\n")
    by_type = (
        values.groupby("type_name")["vaep_value"]
        .agg(n="size", mean="mean", sd="std", **{"share zero": lambda s: float((s == 0).mean())}, **{"share positive": lambda s: float((s > 0).mean())})
        .sort_values("n", ascending=False)
        .reset_index()
    )
    add("Per action type:\n")
    add(md_table(by_type, {"default": 4, "n": 0}) + "\n")

    # --------------------------------------------------------- raw_score groups
    add("## `raw_score` per position group\n")
    add(
        f"`raw_score` = sum of a player's `vaep_value` over the match; every player who appeared gets a row "
        f"({len(raw):,} player-matches in `raw_vaep.parquet`, {len(raw_holdout):,} in `raw_vaep_holdout.parquet`). "
        "`share zero` is the share of players with no valued action (typically very short substitute appearances).\n"
    )
    add("Full season, cross-fitted (`raw_vaep.parquet`):\n")
    add(md_table(group_table(raw), {"default": 3, "n": 0, "share zero": 4}) + "\n")
    add(f"Holdout matches, holdout model (`raw_vaep_holdout.parquet`, matchweeks {HOLDOUT_LAST_TRAIN_WEEK + 1}-{week.max()}):\n")
    add(md_table(group_table(raw_holdout), {"default": 3, "n": 0, "share zero": 4}) + "\n")

    # ------------------------------------------------- holdout vs cross-fitted
    both = raw_holdout.merge(raw[["match_id", "player_id", "raw_score"]], on=["match_id", "player_id"], suffixes=("_holdout", "_cv"))
    pearson = both["raw_score_holdout"].corr(both["raw_score_cv"])
    spearman = both["raw_score_holdout"].corr(both["raw_score_cv"], method="spearman")
    absdiff = (both["raw_score_holdout"] - both["raw_score_cv"]).abs()
    add("## Holdout model vs cross-fitted values on the holdout matches\n")
    add(
        f"The same {len(both):,} player-matches scored by the holdout model (trained on weeks 1-{HOLDOUT_LAST_TRAIN_WEEK}) "
        f"and by their cross-fitting fold models: Pearson r = {pearson:.3f}, Spearman rho = {spearman:.3f}, "
        f"mean |difference| = {absdiff.mean():.4f}, max |difference| = {absdiff.max():.4f} "
        f"(raw_score sd in this set: {both['raw_score_cv'].std():.4f}).\n"
    )

    # ----------------------------------------------------------------- eye test
    add("## Eye test\n")
    add("Top 15 player-matches by `raw_score` (cross-fitted, full season):\n")
    add(md_table(eye_test(raw, matches, 15, lowest=False), {"default": 3, "minutes": 1}) + "\n")
    add("The 10 lowest:\n")
    add(md_table(eye_test(raw, matches, 10, lowest=True), {"default": 3, "minutes": 1}) + "\n")
    add("Data: StatsBomb Open Data (https://github.com/statsbomb/open-data).\n")
    return "\n".join(parts)


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    text = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT_PATH} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
