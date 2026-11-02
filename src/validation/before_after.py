"""Section 7: compare the frozen pre-decision outputs (``data/processed/before/``) with the
current outputs. Read-only. ``python -m src.validation.before_after`` prints the headline numbers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.contract import POSITION_GROUPS
from src.validation import analysis
from src.validation.data import PROCESSED, Data, md_table

BEFORE = PROCESSED / "before"
TEAM_CONTEXT = analysis.TEAM_CONTEXT
SHOT_TYPES = analysis.SHOT_TYPES


def load_before(data: Data) -> Data | None:
    """A Data object whose rating files, stats and action values come from before/."""
    if not (BEFORE / "ratings_baseline.parquet").exists():
        return None
    old = Data(matches=data.matches, lineups=data.lineups)
    for method in ("baseline", "vaep"):
        old.raw[method] = pd.read_parquet(BEFORE / f"raw_{method}.parquet")
        old.ratings[method] = pd.read_parquet(BEFORE / f"ratings_{method}.parquet")
    old.stats = pd.read_parquet(BEFORE / "baseline_stats.parquet")
    old.actions_values = pd.read_parquet(BEFORE / "action_values.parquet", columns=[c for c in data.actions_values.columns])
    old.splits = json.loads((BEFORE / "vaep_splits.json").read_text(encoding="utf-8"))
    rb, rv = old.ratings["baseline"], old.ratings["vaep"]
    both = rb.merge(rv[["match_id", "player_id", "raw_score", "rating"]], on=["match_id", "player_id"], suffixes=("_b", "_v"))
    both = both.merge(data.lineups[["match_id", "player_id", "is_starter", "position_id", "position", "minute_on", "red_card_minute"]], on=["match_id", "player_id"], how="left")
    both["week"] = both["match_id"].map(data.matches.set_index("match_id")["match_week"])
    old.both = both
    return old


# The last commit on main with the pre-decision baseline weights: the first parent of the merge
# (4c4d690) that brought in 216671e ("apply Step 4 weight decisions"). src/baseline/weights.py is
# identical there and at 216671e^ (9763e4d); its weights reproduce data/processed/before/raw_baseline.parquet
# from before/baseline_stats.parquet exactly.
PRE_DECISION_WEIGHTS_COMMIT = "4d7cbc4"


def _old_weights():
    """The pre-decision baseline weights, read from git history (for the 'before' contributions)."""
    import subprocess
    import types
    from src.validation.data import ROOT
    try:
        src = subprocess.run(["git", "show", f"{PRE_DECISION_WEIGHTS_COMMIT}:src/baseline/weights.py"], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", check=True).stdout
    except Exception:
        return None
    import sys
    mod = types.ModuleType("src.validation._old_weights")
    sys.modules[mod.__name__] = mod
    exec(compile(src, "old_weights.py", "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------ notable matches
def match_side_by_side(data: Data, old: Data, match_id: int) -> tuple[str, list[list]]:
    m = data.match_index.loc[match_id]
    new = data.both[data.both["match_id"] == match_id]
    prev = old.both[old.both["match_id"] == match_id].set_index("player_id")
    new = new.assign(side=np.where(new["team"] == m["home_team"], "H", "A")).sort_values(["side", "is_starter", "rating_b"], ascending=[False, False, False])
    rows = []
    for r in new.itertuples():
        s = analysis.stat_row(data, match_id, r.player_id)
        marks = []
        if s["goals"]:
            marks.append(f"{int(s['goals'])}G")
        if "penalties_scored" in s.index and s["penalties_scored"]:
            marks.append(f"{int(s['penalties_scored'])}pen")
        if s["assists"]:
            marks.append(f"{int(s['assists'])}A")
        if s["own_goals"]:
            marks.append("OG")
        if s["red_cards"]:
            marks.append("RED")
        if s["errors"]:
            marks.append("err")
        ob, ov = prev.loc[r.player_id, "rating_b"], prev.loc[r.player_id, "rating_v"]
        rows.append([r.side, r.player_name, r.position_group, "S" if r.is_starter else "sub", f"{r.minutes:.0f}", " ".join(marks),
                     f"{ob:.1f}", f"{r.rating_b:.1f}", f"{r.rating_b - ob:+.1f}", f"{ov:.1f}", f"{r.rating_v:.1f}", f"{r.rating_v - ov:+.1f}"])
    header = f"{m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']} (match_id {match_id})"
    return header, rows


# ------------------------------------------------------------------------ top 20
def top20_before_after(data: Data, old: Data, method: str) -> list[list]:
    def top(d: Data) -> pd.DataFrame:
        r = d.ratings[method].sort_values(["rating", "raw_score"], ascending=False).head(20)
        return r.reset_index(drop=True)
    tn, to = top(data), top(old)
    keys_new = set(map(tuple, tn[["match_id", "player_id"]].to_numpy()))
    keys_old = set(map(tuple, to[["match_id", "player_id"]].to_numpy()))
    other_new = data.ratings[method].set_index(["match_id", "player_id"])["rating"]
    other_old = old.ratings[method].set_index(["match_id", "player_id"])["rating"]
    rows = []
    for i in range(20):
        a, b = to.iloc[i], tn.iloc[i]
        oa, sa, _, _ = data.context(a["match_id"], a["team"])
        ob, sb, _, _ = data.context(b["match_id"], b["team"])
        ka, kb = (a["match_id"], a["player_id"]), (b["match_id"], b["player_id"])
        rows.append([i + 1, f"{a['player_name']} ({a['position_group']}) v {oa} {sa}", f"{a['rating']:.1f}", f"{float(other_new.get(ka, np.nan)):.1f}", "left" if ka not in keys_new else "",
                     f"{b['player_name']} ({b['position_group']}) v {ob} {sb}", f"{float(other_old.get(kb, np.nan)):.1f}", f"{b['rating']:.1f}", "**new**" if kb not in keys_old else ""])
    return rows


# ------------------------------------------------------------------- disagreements
def disagreements(d: Data, n: int = 10) -> list[list]:
    b = d.both.assign(diff=d.both["rating_v"] - d.both["rating_b"])
    b = b.reindex(b["diff"].abs().sort_values(ascending=False).index).head(n)
    rows = []
    for r in b.itertuples():
        opp, score, _, _ = d.context(r.match_id, r.team)
        rows.append([f"{r.player_name} v {opp} {score}", r.position_group, f"{r.minutes:.0f}", f"{r.rating_b:.1f}", f"{r.rating_v:.1f}", f"{r.diff:+.1f}"])
    return rows


# -------------------------------------------------------------------- correlations
def agreement(d: Data) -> dict:
    both = d.both
    out = {"all": float(spearmanr(both["rating_b"], both["rating_v"])[0])}
    for g, part in both.groupby("position_group"):
        out[g] = float(spearmanr(part["rating_b"], part["rating_v"])[0])
    sea = both.groupby("player_id").agg(n=("rating_b", "size"), b=("rating_b", "mean"), v=("rating_v", "mean"))
    sea = sea[sea["n"] >= 10]
    out["season"] = float(spearmanr(sea["b"], sea["v"])[0])
    out["season_n"] = len(sea)
    return out


def old_vs_new(data: Data, old: Data) -> list[list]:
    rows = []
    for method in ("baseline", "vaep"):
        m = data.ratings[method].merge(old.ratings[method][["match_id", "player_id", "rating", "raw_score"]], on=["match_id", "player_id"], suffixes=("", "_old"))
        row = [method, len(m), round(float(spearmanr(m["rating"], m["rating_old"])[0]), 3)]
        for g in POSITION_GROUPS:
            part = m[m["position_group"] == g]
            row.append(round(float(spearmanr(part["rating"], part["rating_old"])[0]), 3))
        row.append(round(float((m["rating"] - m["rating_old"]).abs().mean()), 3))
        row.append(int(((m["rating"] - m["rating_old"]).abs() >= 1.0).sum()))
        rows.append(row)
    return rows


def team_context_share(d: Data, weights_module) -> dict:
    st = d.stats.merge(d.ratings["baseline"][["match_id", "player_id", "raw_score"]], on=["match_id", "player_id"])
    out = {}
    for g, part in st.groupby("position_group"):
        w = weights_module.WEIGHTS[g]
        team = sum(part[s] * w[s].weight for s in TEAM_CONTEXT if s in w and s in part.columns)
        out[g] = float(team.var() / part["raw_score"].var())
    return out


def shot_values(d: Data) -> list[list]:
    av = d.actions_values
    rows = []
    for label, mask in (("open-play goal", (av["type_name"] == "shot") & (av["result_name"] == "success")),
                        ("open-play shot not scored", (av["type_name"] == "shot") & (av["result_name"] == "fail")),
                        ("converted penalty", (av["type_name"] == "shot_penalty") & (av["result_name"] == "success")),
                        ("missed penalty", (av["type_name"] == "shot_penalty") & (av["result_name"] == "fail")),
                        ("free-kick goal", (av["type_name"] == "shot_freekick") & (av["result_name"] == "success")),
                        ("free-kick shot not scored", (av["type_name"] == "shot_freekick") & (av["result_name"] == "fail")),
                        ("keeper save (all)", av["type_name"] == "keeper_save"),
                        ("keeper save, success", (av["type_name"] == "keeper_save") & (av["result_name"] == "success")),
                        ("keeper save, into danger", (av["type_name"] == "keeper_save") & (av["result_name"] == "fail")),
                        ("keeper punch", av["type_name"] == "keeper_punch"),
                        ("foul conceding a penalty (next action a penalty)", None)):
        if mask is None:
            nxt = av.sort_values(["game_id", "action_id"])
            is_pen_next = nxt["type_name"].shift(-1).eq("shot_penalty") & nxt["game_id"].eq(nxt["game_id"].shift(-1))
            part = nxt[is_pen_next & (nxt["type_name"] == "foul")]
        else:
            part = av[mask]
        rows.append([label, len(part), float(part["offensive_value"].mean()), float(part["defensive_value"].mean()), float(part["vaep_value"].mean())])
    return rows


def penalty_states(data: Data) -> dict:
    """The model's predicted P(scores) for penalty states vs their label, from the cached features/labels and the fold models."""
    out = {}
    try:
        import pickle
        from src.vaep.config import LABELS_PATH, FEATURES_PATH, fold_model_path
        labels = pd.read_parquet(LABELS_PATH)
        av = data.actions_values[data.actions_values["type_name"] == "shot_penalty"][["game_id", "action_id", "result_name", "fold_id"]]
        keyed = labels.merge(av, on=["game_id", "action_id"])
        out["n"] = len(keyed)
        out["label_mean"] = float(keyed["scores"].mean())
        # predictions: load fold models and the feature rows of the penalty states
        import pyarrow.parquet as pq
        feats = pd.read_parquet(FEATURES_PATH, filters=[("game_id", "in", sorted(set(int(g) for g in av["game_id"])))])
        feats = feats.merge(av, on=["game_id", "action_id"])
        preds = []
        for fold_id, part in feats.groupby("fold_id"):
            with open(fold_model_path(int(fold_id)), "rb") as fh:
                model = pickle.load(fh)
            p = model.predict_probabilities(part)["scores"]
            preds.append(pd.DataFrame({"result_name": part["result_name"].to_numpy(), "pred": np.asarray(p, dtype=float)}))
        preds = pd.concat(preds)
        out["pred_mean"] = float(preds["pred"].mean())
        out["pred_scored"] = float(preds.loc[preds["result_name"] == "success", "pred"].mean())
        out["pred_missed"] = float(preds.loc[preds["result_name"] == "fail", "pred"].mean())
    except Exception as exc:  # the cached features may be absent; the report then says so
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def minutes_buckets(d: Data) -> dict:
    return analysis.minutes_buckets(d)


def stability(d: Data) -> dict:
    return analysis.stability(d)


def event_means(d: Data) -> dict:
    """Mean rating by event for both methods (penalty scorers/missers, goal scorers, own goals, red cards, errors, clean sheets, heavy defeats)."""
    stt = d.stats.merge(d.both[["match_id", "player_id", "rating_b", "rating_v"]], on=["match_id", "player_id"])
    av = d.actions_values
    pen = av[av["type_name"] == "shot_penalty"][["game_id", "player_id", "result_name"]].rename(columns={"game_id": "match_id"}).drop_duplicates()
    out = {}
    for label, part in (("converted a penalty", pen[pen["result_name"] == "success"].merge(d.both, on=["match_id", "player_id"])),
                        ("missed a penalty", pen[pen["result_name"] == "fail"].merge(d.both, on=["match_id", "player_id"])),
                        ("scored exactly 1 goal (any kind)", stt[(stt["goals"] + (stt["penalties_scored"] if "penalties_scored" in stt.columns else 0)) == 1]),
                        ("own goal", stt[stt["own_goals"] >= 1]), ("red card", stt[stt["red_cards"] >= 1]), ("error leading to a shot", stt[stt["errors"] >= 1]),
                        ("clean sheet (DEF)", stt[(stt["clean_sheet"] == 1) & (stt["position_group"] == "DEF")]),
                        ("4+ conceded, 60+ min (DEF)", stt[(stt["goals_conceded"] >= 4) & (stt["minutes"] >= 60) & (stt["position_group"] == "DEF")]),
                        ("GK, 3+ conceded", stt[(stt["goals_conceded"] >= 3) & (stt["position_group"] == "GK")]),
                        ("GK, 2+ errors", stt[(stt["errors"] >= 2) & (stt["position_group"] == "GK")])):
        out[label] = (len(part), float(part["rating_b"].mean()) if len(part) else float("nan"), float(part["rating_v"].mean()) if len(part) else float("nan"))
    return out


def result_dependence(d: Data) -> dict:
    m = d.matches.set_index("match_id")
    r = d.both.copy()
    home = r["team"] == r["match_id"].map(m["home_team"])
    hs, as_ = r["match_id"].map(m["home_score"]), r["match_id"].map(m["away_score"])
    r["gd"] = np.where(home, hs - as_, as_ - hs)
    out = {"W_b": float(r.loc[r["gd"] > 0, "rating_b"].mean()), "L_b": float(r.loc[r["gd"] < 0, "rating_b"].mean()),
           "W_v": float(r.loc[r["gd"] > 0, "rating_v"].mean()), "L_v": float(r.loc[r["gd"] < 0, "rating_v"].mean()),
           "rho_b": float(spearmanr(r["rating_b"], r["gd"])[0]), "rho_v": float(spearmanr(r["rating_v"], r["gd"])[0])}
    for g, part in r.groupby("position_group"):
        out[f"rho_b_{g}"] = float(spearmanr(part["rating_b"], part["gd"])[0])
        out[f"rho_v_{g}"] = float(spearmanr(part["rating_v"], part["gd"])[0])
    return out


def role_means(d: Data) -> pd.DataFrame:
    from src.validation.checks import ROLE_OF_POSITION
    full = d.both[d.both["minutes"] >= 60].copy()
    full["role"] = full["position_id"].map(ROLE_OF_POSITION)
    return full.groupby(["position_group", "role"]).agg(n=("rating_b", "size"), baseline=("rating_b", "mean"), vaep=("rating_v", "mean"))


def sub_stats(d: Data) -> dict:
    """Short substitutes in one set of outputs: 10-20 minute baseline cameos and where their raw score sits in the group."""
    r = d.ratings["baseline"]
    cameo = r[(r["minutes"] >= 10) & (r["minutes"] < 20)]
    res = {"n": len(cameo), "mean": float(cameo["rating"].mean()), "below6": float((cameo["rating"] < 6).mean()), "median_raw": float(cameo["raw_score"].median())}
    for g in POSITION_GROUPS:
        part = r[r["position_group"] == g]
        c = cameo[cameo["position_group"] == g]
        med = float(c["raw_score"].median()) if len(c) else float("nan")
        res[f"cameo_n_{g}"] = len(c)
        res[f"cameo_mean_{g}"] = float(c["rating"].mean()) if len(c) else float("nan")
        res[f"cameo_median_raw_{g}"] = med
        res[f"cameo_pct_{g}"] = float((part["raw_score"] < med).mean() + 0.5 * (part["raw_score"] == med).mean())
        res[f"share_neg_{g}"] = float((part["raw_score"] < 0).mean())
        res[f"share_neg_starters_{g}"] = float((part.loc[part["minutes"] >= 60, "raw_score"] < 0).mean())
        res[f"p25_{g}"] = float(part["raw_score"].quantile(0.25))
        res[f"p30_{g}"] = float(part["raw_score"].quantile(0.30))
    return res


def sub_analysis(data: Data, old: Data) -> dict:
    """Why short substitutes moved: raw-score percentile of a zero raw score per group, before and after."""
    return {"before": sub_stats(old), "after": sub_stats(data)}


def vaep_sign(d: Data) -> dict:
    """Share of rated appearances with a non-positive VAEP sum, per group."""
    r = d.ratings["vaep"]
    return {g: float((r.loc[r["position_group"] == g, "raw_score"] <= 0).mean()) for g in POSITION_GROUPS}


def compute_current(data: Data) -> dict:
    """The quantities sections 6 and 7 need from the CURRENT outputs only. Needs no before/ snapshot."""
    from src.baseline import weights as new_w
    return {
        "shots": shot_values(data),
        "events": event_means(data),
        "result": result_dependence(data),
        "roles": role_means(data),
        "subs": sub_stats(data),
        "vaep_sign": vaep_sign(data),
        "team_context": team_context_share(data, new_w),
        "penalty_states": penalty_states(data),
    }


def compute(data: Data, current: dict | None = None) -> dict | None:
    """Before/after comparison, or None when ``data/processed/before/`` is absent.

    ``current`` is :func:`compute_current` of ``data``; it is computed here when not given.
    """
    old = load_before(data)
    if old is None:
        return None
    cur = current if current is not None else compute_current(data)
    old_w = _old_weights()
    out = {"old": old}
    out["matches"] = {mid: match_side_by_side(data, old, mid) for mid in (3754013, 3754290, 3754275, 3754079, 3754092)}
    out["top20"] = {m: top20_before_after(data, old, m) for m in ("baseline", "vaep")}
    out["disagree"] = {"before": disagreements(old), "after": disagreements(data)}
    out["agree"] = {"before": agreement(old), "after": agreement(data)}
    out["old_vs_new"] = old_vs_new(data, old)
    out["team_context"] = {"before": team_context_share(old, old_w) if old_w else None, "after": cur["team_context"]}
    out["shots"] = {"before": shot_values(old), "after": cur["shots"]}
    out["penalty_states"] = cur["penalty_states"]
    out["buckets"] = {"before": minutes_buckets(old), "after": minutes_buckets(data)}
    out["stability"] = {"before": stability(old), "after": stability(data)}
    out["events"] = {"before": event_means(old), "after": cur["events"]}
    out["result"] = {"before": result_dependence(old), "after": cur["result"]}
    out["roles"] = {"before": role_means(old), "after": cur["roles"]}
    out["subs"] = {"before": sub_stats(old), "after": cur["subs"]}
    out["vaep_sign"] = {"before": vaep_sign(old), "after": cur["vaep_sign"]}
    return out


if __name__ == "__main__":
    from src.validation.data import load
    pd.set_option("display.width", 250)
    data = load()
    res = compute(data)
    if res is None:
        raise SystemExit(f"{BEFORE} is not present: nothing to compare")
    print("agree", res["agree"])
    print("old_vs_new", res["old_vs_new"])
    print("team_context", res["team_context"])
    print("shots before"); print(pd.DataFrame(res["shots"]["before"]))
    print("shots after"); print(pd.DataFrame(res["shots"]["after"]))
    print("penalty_states", res["penalty_states"])
    for lab in ("before", "after"):
        s = res["stability"][lab]
        print(lab, "stability", {m: (round(s["halves"][m]["pearson"], 3), round(s["halves"][m]["spearman"], 3), s["halves"][m]["by_group"]) for m in ("baseline", "vaep")})
        print(lab, "buckets baseline", res["buckets"][lab]["baseline"]["bucket"][["size", "mean", "share_below_6"]].round(3).to_dict())
        print(lab, "buckets vaep", res["buckets"][lab]["vaep"]["bucket"][["size", "mean", "share_below_6"]].round(3).to_dict())
        print(lab, "events", {k: (n, round(b, 2), round(v, 2)) for k, (n, b, v) in res["events"][lab].items()})
        print(lab, "result", {k: round(v, 3) for k, v in res["result"][lab].items()})
        print(lab, "roles"); print(res["roles"][lab].round(3))
        print(lab, "vaep_sign", res["vaep_sign"][lab])
        print(lab, "subs", {k: round(v, 3) for k, v in res["subs"][lab].items()})
        print(lab, "disagree"); print(pd.DataFrame(res["disagree"][lab]))
    for m in ("baseline", "vaep"):
        print(m, "top20"); print(pd.DataFrame(res["top20"][m]).to_string())
    for mid, (h, rows) in res["matches"].items():
        print(h); print(pd.DataFrame(rows).to_string())
