"""Section 8 of the validation report: the xG-label VAEP (frozen in ``data/processed/xg_labels/``)
against the current option-3 VAEP. Read-only; ``old`` and ``new`` are ``Data`` objects that share
the baseline files."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.contract import POSITION_GROUPS, RATING_MAX
from src.validation import analysis, before_after
from src.validation.data import Data, fmt, md_table

SHOTS = ("shot", "shot_penalty", "shot_freekick")
AGUERO = (3754079, 3237)  # Manchester City 6-1 Newcastle United, five goals


def _actions(d: Data) -> pd.DataFrame:
    av = d.actions_values.sort_values(["game_id", "action_id"], kind="mergesort").reset_index(drop=True)
    prev = av.groupby("game_id")[["type_name", "result_name", "team_id", "time_seconds"]].shift(1)
    av["after_unscored_shot"] = (prev["type_name"].isin(SHOTS) & (prev["result_name"] != "success") & (prev["team_id"] != av["team_id"])
                                 & ((av["time_seconds"] - prev["time_seconds"]).abs() <= 10))
    return av


def _penalty_takers(d: Data) -> pd.DataFrame:
    av = d.actions_values
    pen = av[av["type_name"] == "shot_penalty"][["game_id", "player_id", "result_name"]].rename(columns={"game_id": "match_id"})
    pen["player_id"] = pen["player_id"].astype("int64")
    scored = pen.groupby(["match_id", "player_id"])["result_name"].agg(lambda s: int((s == "success").sum())).rename("scored")
    missed = pen.groupby(["match_id", "player_id"])["result_name"].agg(lambda s: int((s != "success").sum())).rename("missed")
    return pd.concat([scored, missed], axis=1).reset_index().merge(d.both, on=["match_id", "player_id"])


def metrics(d: Data) -> dict:
    av = _actions(d)
    both = d.both
    stats = d.stats.assign(g=d.stats["goals"] + d.stats["penalties_scored"])[["match_id", "player_id", "g", "assists"]]
    rated = both.merge(stats, on=["match_id", "player_id"])
    scorers = rated[rated["g"] >= 1]
    takers = _penalty_takers(d)
    halves = analysis.stability(d)["halves"]["vaep"]
    nxt = av[av["after_unscored_shot"] & (av["type_name"] != "keeper_save")]
    shots = av[av["type_name"].isin(SHOTS)]
    top = d.ratings["vaep"].sort_values(["rating", "raw_score"], ascending=False).head(20).merge(stats, on=["match_id", "player_id"])
    aguero = both[(both["match_id"] == AGUERO[0]) & (both["player_id"] == AGUERO[1])]
    return {
        "pen_value_scored": float(shots.loc[(shots["type_name"] == "shot_penalty") & (shots["result_name"] == "success"), "vaep_value"].mean()),
        "pen_value_missed": float(shots.loc[(shots["type_name"] == "shot_penalty") & (shots["result_name"] != "success"), "vaep_value"].mean()),
        "pen_scorers": (len(takers[takers["scored"] >= 1]), float(takers.loc[takers["scored"] >= 1, "rating_v"].mean()), float(takers.loc[takers["scored"] >= 1, "rating_b"].mean())),
        "pen_missers": (len(takers[takers["missed"] >= 1]), float(takers.loc[takers["missed"] >= 1, "rating_v"].mean()), float(takers.loc[takers["missed"] >= 1, "rating_b"].mean())),
        "goal_value": float(shots.loc[(shots["type_name"] == "shot") & (shots["result_name"] == "success"), "vaep_value"].mean()),
        "miss_value": float(shots.loc[(shots["type_name"] == "shot") & (shots["result_name"] != "success"), "vaep_value"].mean()),
        "save_value": float(av.loc[av["type_name"] == "keeper_save", "vaep_value"].mean()),
        "next_mean": float(nxt["vaep_value"].mean()), "next_p99": float(nxt["vaep_value"].quantile(0.99)), "next_max": float(nxt["vaep_value"].max()), "next_n": len(nxt),
        "clearance_max": float(av.loc[av["type_name"] == "clearance", "vaep_value"].max()),
        "tackle_max": float(av.loc[av["type_name"] == "tackle", "vaep_value"].max()),
        "big_non_shot": int(((av["vaep_value"] > 0.3) & ~av["type_name"].isin(SHOTS)).sum()),
        "big_after_shot_non_save": int(((av["vaep_value"] > 0.3) & av["after_unscored_shot"] & (av["type_name"] != "keeper_save")).sum()),
        "scorers_n": len(scorers), "scorers_below6": int((scorers["rating_v"] < 6.0).sum()), "scorers_below55": int((scorers["rating_v"] < 5.5).sum()),
        "scorers_below6_baseline": int((scorers["rating_b"] < 6.0).sum()),
        "one_goal_mean": float(rated.loc[rated["g"] == 1, "rating_v"].mean()),
        "aguero": float(aguero["rating_v"].iloc[0]) if len(aguero) else float("nan"),
        "halves": halves["pearson"], "halves_n": halves["n"], "halves_by_group": {x[0]: (x[1], x[2]) for x in halves["by_group"]},
        "spearman_all": float(spearmanr(both["rating_b"], both["rating_v"])[0]),
        "spearman_def": float(spearmanr(both.loc[both["position_group"] == "DEF", "rating_b"], both.loc[both["position_group"] == "DEF", "rating_v"])[0]),
        "top20_no_goal_or_assist": int(((top["g"] == 0) & (top["assists"] == 0)).sum()),
        "tens": int((d.ratings["vaep"]["rating"] == RATING_MAX).sum()),
        "takers": takers,
    }


def compute(new: Data, old: Data | None) -> dict | None:
    if old is None:
        return None
    m_old, m_new = metrics(old), metrics(new)
    baseline_halves = analysis.stability(new)["halves"]["baseline"]
    moved = new.both[["match_id", "player_id", "player_name", "team", "position_group", "minutes", "rating_b", "rating_v"]].merge(
        old.both[["match_id", "player_id", "rating_v"]], on=["match_id", "player_id"], suffixes=("", "_old"))
    moved["d"] = moved["rating_v"] - moved["rating_v_old"]
    av_old, av_new = old.actions_values, new.actions_values
    delta = av_new[["game_id", "action_id", "player_id", "type_name", "result_name", "time_seconds", "period_id", "vaep_value"]].merge(
        av_old[["game_id", "action_id", "vaep_value"]], on=["game_id", "action_id"], suffixes=("", "_old"))
    delta["dv"] = delta["vaep_value"] - delta["vaep_value_old"]
    rows = []
    for r in moved.reindex(moved["d"].abs().sort_values(ascending=False).index).head(15).itertuples():
        opp, score, _, _ = new.context(r.match_id, r.team)
        mine = delta[(delta["game_id"] == r.match_id) & (delta["player_id"] == r.player_id)]
        top = mine.reindex(mine["dv"].abs().sort_values(ascending=False).index).iloc[0] if len(mine) else None
        act = (f"{top.type_name}/{top.result_name} {'H1' if top.period_id == 1 else 'H2'} {top.time_seconds / 60:.0f}': {top.vaep_value_old:+.3f} -> {top.vaep_value:+.3f}"
               if top is not None and abs(top.dv) > 1e-9 else "")
        rows.append([f"{r.player_name} v {opp} {score}", r.position_group, f"{r.minutes:.0f}", f"{r.rating_v_old:.1f}", f"{r.rating_v:.1f}", f"{r.d:+.1f}", f"{r.rating_b:.1f}", act])
    return {"old": m_old, "new": m_new, "baseline_halves": baseline_halves, "top20": before_after.top20_before_after(new, old, "vaep"),
            "movers": rows, "moved_05": int((moved["d"].abs() >= 0.5).sum()), "moved_10": int((moved["d"].abs() >= 1.0).sum()), "rows": len(moved)}


def summary_rows(o3: dict) -> list[list]:
    """The short table for the headline numbers."""
    a, b = o3["old"], o3["new"]
    return [
        ["VAEP value of a converted / missed penalty to the taker", f"{a['pen_value_scored']:+.3f} / {a['pen_value_missed']:+.3f}", f"{b['pen_value_scored']:+.3f} / {b['pen_value_missed']:+.3f}"],
        ["penalty scorers' / missers' mean VAEP rating", f"{a['pen_scorers'][1]:.2f} / {a['pen_missers'][1]:.2f}", f"{b['pen_scorers'][1]:.2f} / {b['pen_missers'][1]:.2f}"],
        [f"appearances with a goal rated below 6.0 under VAEP (of {b['scorers_n']})", str(a["scorers_below6"]), str(b["scorers_below6"])],
        ["first non-save action after an opponent's unscored shot: mean / 99th percentile / max", f"{a['next_mean']:+.3f} / {a['next_p99']:+.3f} / {a['next_max']:+.3f}", f"{b['next_mean']:+.3f} / {b['next_p99']:+.3f} / {b['next_max']:+.3f}"],
        ["VAEP split-half Pearson, all / DEF", f"{a['halves']:.3f} / {a['halves_by_group']['DEF'][1]:.3f}", f"{b['halves']:.3f} / {b['halves_by_group']['DEF'][1]:.3f}"],
    ]


def section(o3: dict | None) -> list[str]:
    out = ["## 8. xG-label VAEP vs option 3 (decision 2026-11-02)", ""]
    if o3 is None:
        return out + ["`data/processed/xg_labels/` (the frozen xG-label VAEP outputs) is not present, so the comparison with option 3 was skipped.", ""]
    a, b = o3["old"], o3["new"]
    bh = o3["baseline_halves"]
    bh_group = {x[0]: x[2] for x in bh["by_group"]}
    out += ["Before = the xG-label VAEP as it stood after the Step 4 decisions (frozen in `data/processed/xg_labels/`); after = the current VAEP, which adds the two "
            "option-3 scoring rules in `src/vaep/value.py`: every penalty state's P(scores) is that penalty's StatsBomb xG, and the action after an unscored shot is "
            "measured against the shot state's P(scores) with the shot's own chance removed, except a keeper save, which keeps the full drop. The fold models, "
            "features, labels and splits are unchanged and nothing was retrained; the baseline files are the same files in both columns.", ""]
    out += ["### 8.1 Summary", "",
            md_table(["quantity", "xG labels", "option 3", "baseline"], [
                ["VAEP value of a converted / missed penalty to the taker", f"{a['pen_value_scored']:+.3f} / {a['pen_value_missed']:+.3f}", f"{b['pen_value_scored']:+.3f} / {b['pen_value_missed']:+.3f}", ""],
                [f"penalty scorers' mean rating ({b['pen_scorers'][0]} appearances)", fmt(a["pen_scorers"][1], 2), fmt(b["pen_scorers"][1], 2), fmt(b["pen_scorers"][2], 2)],
                [f"penalty missers' mean rating ({b['pen_missers'][0]} appearances)", fmt(a["pen_missers"][1], 2), fmt(b["pen_missers"][1], 2), fmt(b["pen_missers"][2], 2)],
                ["VAEP value of an open-play goal / miss / keeper save", f"{a['goal_value']:+.3f} / {a['miss_value']:+.3f} / {a['save_value']:+.3f}", f"{b['goal_value']:+.3f} / {b['miss_value']:+.3f} / {b['save_value']:+.3f}", ""],
                [f"appearances with a goal rated below 6.0 / below 5.5 (of {b['scorers_n']})", f"{a['scorers_below6']} / {a['scorers_below55']}", f"{b['scorers_below6']} / {b['scorers_below55']}", str(b["scorers_below6_baseline"]) + " / -"],
                ["one-goal scorers' mean rating", fmt(a["one_goal_mean"], 2), fmt(b["one_goal_mean"], 2), ""],
                ["Aguero's five goals v Newcastle", f"{a['aguero']:.1f}", f"{b['aguero']:.1f}", "10.0"],
                [f"first non-save action within 10 s of an opponent's unscored shot ({b['next_n']}): mean / p99 / max", f"{a['next_mean']:+.3f} / {a['next_p99']:+.3f} / {a['next_max']:+.3f}", f"{b['next_mean']:+.3f} / {b['next_p99']:+.3f} / {b['next_max']:+.3f}", ""],
                ["most valuable clearance / tackle", f"{a['clearance_max']:+.3f} / {a['tackle_max']:+.3f}", f"{b['clearance_max']:+.3f} / {b['tackle_max']:+.3f}", ""],
                ["non-shot actions worth more than +0.30 (of which non-save actions right after an unscored shot)", f"{a['big_non_shot']} ({a['big_after_shot_non_save']})", f"{b['big_non_shot']} ({b['big_after_shot_non_save']})", ""],
                [f"split-half Pearson, all ({b['halves_n']} players)", fmt(a["halves"]), fmt(b["halves"]), fmt(bh["pearson"])],
                *[[f"... {g} ({b['halves_by_group'][g][0]} players)", fmt(a["halves_by_group"][g][1]), fmt(b["halves_by_group"][g][1]), fmt(bh_group[g])] for g in POSITION_GROUPS],
                ["Spearman with the baseline, all / DEF", f"{a['spearman_all']:.3f} / {a['spearman_def']:.3f}", f"{b['spearman_all']:.3f} / {b['spearman_def']:.3f}", ""],
                ["VAEP top 20 rows with neither a goal nor an assist", str(a["top20_no_goal_or_assist"]), str(b["top20_no_goal_or_assist"]), ""],
                ["ratings of 10.0", str(a["tens"]), str(b["tens"]), ""],
            ]), "",
            f"{o3['moved_05']} of {o3['rows']} rated appearances moved by 0.5 or more and {o3['moved_10']} by 1.0 or more.", ""]
    nb = sum(1 for r in o3["top20"] if r[8])
    out += ["### 8.2 Top 20 VAEP single-match ratings, xG labels vs option 3", "",
            md_table(["#", "xG labels", "rating", "same row under option 3", "", "option 3", "same row under xG labels", "rating", ""], o3["top20"]), "",
            f"{nb} of the 20 option-3 rows are new.", ""]
    out += ["### 8.3 The fifteen largest rating changes", "",
            "With the action whose value changed most (type/result, half, minute, value under xG labels -> option 3).", "",
            md_table(["player-match", "group", "min", "xG labels", "option 3", "change", "baseline", "action that changed most"], o3["movers"]), ""]
    t = b["takers"].sort_values(["missed", "rating_v"], ascending=[False, False])
    ta = a["takers"].set_index(["match_id", "player_id"])["rating_v"]
    rows = []
    for r in t[t["missed"] >= 1].itertuples():
        rows.append([r.player_name, r.position_group, int(r.scored), int(r.missed), f"{ta.get((r.match_id, r.player_id), np.nan):.1f}", f"{r.rating_v:.1f}", f"{r.rating_b:.1f}"])
    out += ["### 8.4 Players who missed a penalty", "",
            md_table(["player", "group", "penalties scored", "missed", "VAEP, xG labels", "VAEP, option 3", "baseline"], rows), ""]
    out += [f"*Judgement (verifier, 2026-11-02).* Option 3 does what it was designed to do and nothing more. Penalties: the kick is now worth {b['pen_value_scored']:+.3f} to a taker who "
            f"scores and {b['pen_value_missed']:+.3f} to one who misses, so penalty scorers rise from {a['pen_scorers'][1]:.2f} to {b['pen_scorers'][1]:.2f}; but the outcome no longer matters, "
            f"and players who missed a penalty average {b['pen_missers'][1]:.2f}, above the scorers (baseline {b['pen_scorers'][2]:.2f} and {b['pen_missers'][2]:.2f}). Defending: the first "
            f"non-save action after an opponent's missed shot averages {b['next_mean']:+.3f} instead of {a['next_mean']:+.3f} and is worth at most {b['next_max']:+.3f} instead of "
            f"{a['next_max']:+.3f}, no non-save action straight after a shot is worth more than +0.30 any more, and defenders' split-half stability rose from "
            f"{a['halves_by_group']['DEF'][1]:.3f} to {b['halves_by_group']['DEF'][1]:.3f}, still far below the baseline's {bh_group['DEF']:.3f}. The large moves in 8.3 are of two kinds: "
            "defenders whose match rested on one clearance straight after a missed chance fall by two to three points (Charlie Daniels 9.0 -> 5.4, Paul Dummett 8.1 -> 5.0, Kouyate 9.5 -> 6.7), "
            "and players who had been charged with their own or a teammate's miss rise (Berahino 4.0 -> 6.4 after missing two penalties, James Morrison 4.8 -> 7.1, Cazorla 4.3 -> 7.1). "
            f"Not fixed: goals barely count (an open-play goal {b['goal_value']:+.3f} against a miss {b['miss_value']:+.3f}; Aguero's five goals {b['aguero']:.1f}; {b['scorers_below6']} of "
            f"{b['scorers_n']} goal-scoring appearances below 6.0, against {b['scorers_below6_baseline']} under the baseline); results and clean sheets are invisible, so Daniels' clean sheet "
            "at Leicester and Dummett's in a 1-0 win now rate 5.4 and 5.0; and one big action can still decide a rating (van Aanholt 9.6 in a 0-2 home defeat, for a tackle that ended a "
            "dribble). Verdict: option 3 is an improvement worth keeping for the experimental VAEP view; the baseline stays the headline rating.", ""]
    return out
