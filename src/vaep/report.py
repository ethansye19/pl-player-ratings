"""Write ``reports/vaep_splits.md`` from the VAEP outputs. CLI: ``python -m src.vaep.report``.

The report documents the holdout and fold definitions, the model settings, the xG-label
variant (what changed, the label formula, the join to StatsBomb xG, a before/after table
of what a shot is worth against the frozen snapshot in ``data/processed/before``), model
quality on never-seen games, the distribution of the action values and of ``raw_score``
per position group, and an eye test of the best and worst player-matches. Every number
is computed from ``data/processed`` at run time.
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
    BEFORE_DIR,
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
SHOT_CATEGORIES = [
    ("open-play goal", "shot", "success"),
    ("open-play shot not scored", "shot", "fail"),
    ("converted penalty", "shot_penalty", "success"),
    ("missed penalty", "shot_penalty", "fail"),
    ("free-kick goal", "shot_freekick", "success"),
    ("free-kick shot not scored", "shot_freekick", "fail"),
]


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


def quality_table(metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    soft = "brier_soft" in metrics["holdout"]["metrics"]["scores"]
    rows = []
    for info in [metrics["holdout"]] + metrics["folds"]:
        for label in LABELS:
            m, fit = info["metrics"][label], info["labels"][label]
            row = {
                "split": info["name"],
                "label": label,
                "fit rows": info["n_fit_rows"],
                "fit goal rate": info["fit_positive_rate"][label],
                "rounds": fit["best_iteration"],
                "val loss": fit["val_logloss"],
                "held-out rows": info["n_score_rows"],
                "held-out goal rate": m["score_positive_rate"],
                "AUC (goals)": m["auroc"],
                "Brier (goals)": m["brier"],
                "Brier (goals, base rate)": m["brier_base_rate"],
            }
            if soft:
                row["Brier (soft)"] = m["brier_soft"]
                row["Brier (soft, base rate)"] = m["brier_soft_base_rate"]
            rows.append(row)
    table = pd.DataFrame(rows)
    folds = table[table["split"] != "holdout"]
    stat_cols = ["rounds", "AUC (goals)", "Brier (goals)"] + (["Brier (soft)"] if soft else [])
    means = folds.groupby("label", sort=False)[stat_cols].agg(["mean", "min", "max"])
    means.columns = [f"{metric} {stat}" for metric, stat in means.columns]
    return table, means.reset_index()


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


def shot_value_table(after: pd.DataFrame, before: pd.DataFrame | None) -> pd.DataFrame:
    """Mean offensive / VAEP value of a shot by category, before (frozen snapshot) and after."""
    rows = []
    for name, type_name, result in SHOT_CATEGORIES:
        sel_a = after[(after["type_name"] == type_name) & (after["result_name"] == result)]
        row = {"shot": name, "n": len(sel_a)}
        if before is not None:
            sel_b = before[(before["type_name"] == type_name) & (before["result_name"] == result)]
            row["before: mean offensive"] = float(sel_b["offensive_value"].mean())
            row["before: mean vaep"] = float(sel_b["vaep_value"].mean())
        row["after: mean offensive"] = float(sel_a["offensive_value"].mean())
        row["after: mean vaep"] = float(sel_a["vaep_value"].mean())
        row["after: sd vaep"] = float(sel_a["vaep_value"].std())
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(raw: pd.DataFrame, before_raw: pd.DataFrame) -> pd.DataFrame:
    both = raw.merge(before_raw[["match_id", "player_id", "raw_score"]], on=["match_id", "player_id"], suffixes=("", "_before"))
    rows = []
    for group in ("all",) + tuple(POSITION_GROUPS):
        sel = both if group == "all" else both[both["position_group"] == group]
        rated = sel[sel["minutes"] >= 10]
        rows.append(
            {
                "position_group": group,
                "player-matches": len(sel),
                "Pearson r": float(sel["raw_score"].corr(sel["raw_score_before"])),
                "Spearman rho": float(sel["raw_score"].corr(sel["raw_score_before"], method="spearman")),
                "Spearman rho (10+ min)": float(rated["raw_score"].corr(rated["raw_score_before"], method="spearman")),
                "mean before": float(sel["raw_score_before"].mean()),
                "mean after": float(sel["raw_score"].mean()),
                "sd before": float(sel["raw_score_before"].std()),
                "sd after": float(sel["raw_score"].std()),
            }
        )
    return pd.DataFrame(rows)


def build_report() -> str:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    matches = pd.read_parquet(MATCHES_PATH)
    values = pd.read_parquet(ACTION_VALUES_PATH)
    raw = pd.read_parquet(RAW_VAEP_PATH)
    raw_holdout = pd.read_parquet(RAW_VAEP_HOLDOUT_PATH)
    week = matches.set_index("match_id")["match_week"]
    label_mode = metrics.get("label_mode", "outcome")
    before_values_path, before_raw_path = BEFORE_DIR / "action_values.parquet", BEFORE_DIR / "raw_vaep.parquet"
    before_values = pd.read_parquet(before_values_path) if before_values_path.exists() else None
    before_raw = pd.read_parquet(before_raw_path) if before_raw_path.exists() else None

    parts: list[str] = []
    add = parts.append
    add("# VAEP: train/score splits, model settings and model quality\n")
    add(
        f"Generated by `python -m src.vaep.report` on {dt.date.today().isoformat()} from the outputs of "
        f"`python -m src.vaep.train --labels {label_mode}` (StatsBomb Open Data, Premier League 2015/16: {metrics['n_games']} matches, "
        f"{metrics['n_actions']:,} SPADL actions). Training run time: {metrics['runtime_seconds'] / 60:.1f} min "
        f"(feature step {metrics['features_seconds']:.0f} s, label step {metrics.get('labels_seconds', 0):.0f} s).\n"
    )
    add(
        "Framework: VAEP (Decroos, Bransen, Van Haaren and Davis, \"Actions Speak Louder than Goals: Valuing Player "
        "Actions in Soccer\", KDD 2019), as implemented in `socceraction`. "
        + (
            "Shots are valued by chance quality: the labels are built from StatsBomb expected goals instead of the "
            "shot outcome (section *Shots valued by expected goals* below).\n"
            if label_mode == "xg"
            else "This run uses socceraction's binary outcome labels (`--labels outcome`).\n"
        )
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
        f"{metrics['n_games']} matches into {len(splits['folds'])} folds of "
        f"{' / '.join(str(n) for n in sorted({len(f['score']) for f in splits['folds']}))} matches. "
        "The model of fold *k* is trained on the other folds and scores only fold *k*, so every match "
        "is scored exactly once by a model that never saw it. Outputs: `action_values.parquet` (with `fold_id`) "
        "and `raw_vaep.parquet`.\n"
        f"- Inside every split, a seeded {metrics['validation_share']:.0%} of the *training* games is set aside "
        "for early stopping (whole games, never rows, because consecutive game states share labels); the rest "
        "fits the trees. Score games are never used for fitting or round selection.\n"
    )
    if before_values is not None and (BEFORE_DIR / "vaep_splits.json").exists():
        same = json.loads((BEFORE_DIR / "vaep_splits.json").read_text(encoding="utf-8")) == splits
        add(f"The splits are {'identical to' if same else 'DIFFERENT from'} the frozen pre-change `before/vaep_splits.json`.\n")
    add(md_table(split_table(splits, matches, metrics)) + "\n")

    # ---------------------------------------------------------- model settings
    v = metrics["versions"]
    add("## Model settings\n")
    add(
        f"- `socceraction {v['socceraction']}` `VAEP(nb_prev_actions={metrics['nb_prev_actions']})` with the default "
        f"feature transformers (`xfns_default`: action type, result, type x result and body-part one-hots, time, "
        f"start/end location, polar coordinates, movement, possession-team flags, time and space deltas, score line) "
        f"-> {metrics['n_feature_columns']} feature columns per game state (the action and its "
        f"{metrics['nb_prev_actions'] - 1} predecessors), pitch flipped so the acting team attacks left to right. "
        "The features and the game-state construction are unchanged from the outcome-label run.\n"
        f"- Labels (`src/vaep/labels.py`): `scores` / `concedes` for the window of the action and the next "
        f"{metrics.get('nr_label_actions', 10) - 1} actions -- "
        + (
            "soft xG-based labels (probability that the acting team scores / concedes from the chances in the window, "
            "see the next section). "
            if label_mode == "xg"
            else "socceraction's binary outcome labels. "
        )
        + "Values: `offensive_value` = change in P(scores), `defensive_value` = -change in P(concedes), "
        "`vaep_value` = their sum (`socceraction.vaep.formula.value`, unchanged).\n"
        f"- Learner: `lightgbm {v['lightgbm']}` `{metrics.get('learner', 'LGBMClassifier')}`, one per label, through "
        "`src.vaep.model.SoftLabelVAEP` (a `socceraction.vaep.VAEP` subclass that replaces only fit/predict: "
        "socceraction's `VAEP.fit` builds `LGBMClassifier`s, which reject targets strictly between 0 and 1). "
        f"Early stopping on the validation games' cross-entropy (patience {metrics['early_stopping_rounds']} rounds); "
        "the tree parameters below are those of the outcome-label run.\n"
        f"- Determinism: `SEED = {metrics['seed']}` seeds the fold shuffle, the validation-game draw and lightgbm "
        "(`random_state`, `deterministic=True`, `force_row_wise=True`); the row order is the action order, so "
        "a rerun reproduces every value.\n"
        "- Features are cast to float32 for fitting; scoring uses the float64 features computed on the fly from "
        "the match's actions, exactly as `src.vaep.rate.rate_match` does (train.py builds `raw_vaep*.parquet` "
        "through that same function).\n"
    )
    params = pd.DataFrame({"parameter": list(metrics["tree_params"]), "value": [str(x) for x in metrics["tree_params"].values()]})
    add(md_table(params) + "\n")

    # ------------------------------------------------------------- xG variant
    if label_mode == "xg":
        cov = metrics.get("xg_coverage", {})
        add("## Shots valued by expected goals\n")
        add(
            "**Why.** With outcome labels (validation finding B) VAEP is a goals-and-misses detector for attackers: "
            "a goal is worth about +0.8, every unconverted shot is charged the whole chance it consumed, a converted "
            "penalty is worth little to the taker and a missed one costs about -0.75. The standard remedy in the VAEP "
            "literature for this outcome sensitivity is to replace the binary goal labels by expected-goals-based "
            "labels, so that a shot's offensive value reflects the quality of the chance rather than whether it went in.\n"
        )
        add(
            "**Label formula.** For game state *i* and the window *W(i)* of the action and the next 9 actions "
            "(socceraction's window and team logic, unchanged):\n\n"
            "    scores(i)   = 1 - prod_{j in W(i)} (1 - g_j)   over shots j by the acting team; an own goal by the opponent counts g = 1\n"
            "    concedes(i) = 1 - prod_{j in W(i)} (1 - g_j)   over shots j by the opponent;    an own goal by the acting team counts g = 1\n\n"
            "where `g_j` is the StatsBomb xG of shot *j*, i.e. the probability that at least one chance in the window "
            "is converted when shots are independent. With `g_j` replaced by the outcome (1 for a goal, 0 otherwise) the "
            "formula is exactly socceraction's OR of goal flags, and a unit test checks that equivalence. Positions past "
            "the last action of the game contribute nothing (socceraction repeats the last action there, which is "
            "harmless for an OR but would multiply a final chance into the product several times). The models are "
            "then regressed on these soft labels with lightgbm's `cross_entropy` objective, and the values come from "
            "`socceraction.vaep.formula.value` applied to the predicted probabilities, as before.\n"
        )
        add(
            f"**Join to StatsBomb xG and coverage.** SPADL shot actions (`shot`, `shot_penalty`, `shot_freekick`) are joined "
            f"to the StatsBomb event through `original_event_id` = events `id`, taking `shot_statsbomb_xg`: "
            f"{cov.get('shots', 0):,} shot actions, {cov.get('shots_without_xg', 0)} without an xG value (those would fall back to "
            f"the binary outcome). Every penalty carries StatsBomb's constant penalty xG (0.7835) whatever happened to it. "
            f"{cov.get('own_goals', 0)} own goals have no xG and count as a goal (1.0) for the team that benefits and as a goal "
            "conceded for the other. Note: in the SPADL conversion an own goal is a `bad_touch` with result `owngoal`, and "
            "socceraction's outcome labels only recognise own goals on shot-type actions, so the previous run did not see "
            "them; the xG labels do (the only place where the two label definitions differ beyond the xG substitution).\n"
        )
        add("**What a shot is worth (mean value credited to the shooter's action, cross-fitted, all 380 matches):**\n")
        add(
            md_table(shot_value_table(values, before_values), {"default": 4, "n": 0})
            + ("\n" if before_values is not None else "\n(no `data/processed/before/action_values.parquet` snapshot found, so only the new values are shown)\n")
        )
        if before_raw is not None:
            add(
                "**Correlation of the new raw scores with the previous ones** (same player-matches, cross-fitted "
                "`raw_vaep.parquet` vs the frozen `before/raw_vaep.parquet`):\n"
            )
            add(md_table(correlation_table(raw, before_raw), {"default": 3, "player-matches": 0}) + "\n")

    # ------------------------------------------------------------ model quality
    add("## Model quality on never-seen games\n")
    add(
        "Metrics of each split's model on the games it scores (held out from fitting and early stopping). "
        "`AUC (goals)` and `Brier (goals)` are against the actual binary goal labels, so they stay comparable with "
        "the outcome-label run"
        + ("; `Brier (soft)` is against the xG-based soft labels the models were fitted on. " if label_mode == "xg" else ". ")
        + "A `base rate` Brier is that of always predicting the fit set's mean label; a useful model is below it. "
        "`rounds` is the boosting round early stopping selected.\n"
    )
    quality, fold_means = quality_table(metrics)
    add(md_table(quality, {"default": 3, "fit goal rate": 4, "held-out goal rate": 4, "val loss": 5, "Brier (goals)": 5,
                           "Brier (goals, base rate)": 5, "Brier (soft)": 5, "Brier (soft, base rate)": 5}) + "\n")
    add("Across the 5 folds:\n")
    add(md_table(fold_means, {"default": 5, "rounds mean": 1, "rounds min": 0, "rounds max": 0,
                              "AUC (goals) mean": 3, "AUC (goals) min": 3, "AUC (goals) max": 3}) + "\n")
    fold_rows = quality[quality["split"] != "holdout"]
    worst_auc = fold_rows.groupby("label")["AUC (goals)"].min()
    beats = (fold_rows["Brier (goals)"] < fold_rows["Brier (goals, base rate)"]).all()
    capped = [f"{i['name']}/{label}" for i in [metrics["holdout"]] + metrics["folds"] for label in LABELS if i["labels"][label]["hit_round_cap"]]
    text = (
        f"Lowest fold AUC against goals: scores {worst_auc['scores']:.3f}, concedes {worst_auc['concedes']:.3f}. "
        f"Every fold model {'beats' if beats else 'does NOT beat'} the base-rate Brier score against goals on both labels. "
    )
    if "Brier (soft)" in fold_rows:
        beats_soft = (fold_rows["Brier (soft)"] < fold_rows["Brier (soft, base rate)"]).all()
        text += f"Against the soft labels every fold model {'beats' if beats_soft else 'does NOT beat'} the base rate. "
    text += "No fit reached the round cap." if not capped else f"Fits that reached the round cap: {', '.join(capped)}."
    add(text + "\n")
    before_metrics_path = BEFORE_DIR / "models" / "vaep_metrics.json"
    if before_metrics_path.exists():
        bm = json.loads(before_metrics_path.read_text(encoding="utf-8"))
        rows = []
        for label in LABELS:
            b_auc = np.mean([f["metrics"][label]["auroc"] for f in bm["folds"]])
            b_brier = np.mean([f["metrics"][label]["brier"] for f in bm["folds"]])
            a_auc = np.mean([f["metrics"][label]["auroc"] for f in metrics["folds"]])
            a_brier = np.mean([f["metrics"][label]["brier"] for f in metrics["folds"]])
            rows.append({"label": label, "fold-mean AUC before": b_auc, "fold-mean AUC after": a_auc,
                         "fold-mean Brier (goals) before": b_brier, "fold-mean Brier (goals) after": a_brier})
        add("Against the frozen outcome-label run (`before/models/vaep_metrics.json`), both measured against the binary goal labels:\n")
        add(md_table(pd.DataFrame(rows), {"default": 3, "fold-mean Brier (goals) before": 5, "fold-mean Brier (goals) after": 5}) + "\n")

    # ------------------------------------------------- action value distribution
    add("## Distribution of `vaep_value` (cross-fitted, all matches)\n")
    dist = pd.DataFrame([distribution(values["vaep_value"])])
    if before_values is not None:
        dist = pd.concat([dist, pd.DataFrame([distribution(before_values["vaep_value"])])], ignore_index=True)
        dist.insert(0, "run", ["after (xG labels)", "before (outcome labels)"])
    add(md_table(dist, {"default": 4, "n": 0}) + "\n")
    by_type = (
        values.groupby("type_name")["vaep_value"]
        .agg(n="size", mean="mean", sd="std", **{"share zero": lambda s: float((s == 0).mean())}, **{"share positive": lambda s: float((s > 0).mean())})
        .sort_values("n", ascending=False)
        .reset_index()
    )
    if before_values is not None:
        by_type = by_type.merge(before_values.groupby("type_name")["vaep_value"].mean().rename("mean before").reset_index(), on="type_name", how="left")
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
    if before_raw is not None:
        add("The same for the frozen pre-change `before/raw_vaep.parquet`:\n")
        add(md_table(group_table(before_raw), {"default": 3, "n": 0, "share zero": 4}) + "\n")
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
