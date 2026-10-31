"""Tasks 3-5 of the verification brief: eye test, method comparison, stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.baseline.weights import LABELS, STATS, WEIGHTS
from src.contract import POSITION_GROUPS
from src.validation.data import Data

TEAM_CONTEXT = ("goals_conceded", "clean_sheet", "xg_faced")
SHOT_TYPES = ("shot", "shot_penalty", "shot_freekick")
MINUTE_BINS = [10, 20, 45, 70, 85, 1000]
MINUTE_LABELS = ["10-20", "20-45", "45-70", "70-85", "85+"]


# ------------------------------------------------------------------------- helpers
def stat_row(data: Data, match_id: int, player_id: int) -> pd.Series:
    s = data.stats
    return s[(s["match_id"] == match_id) & (s["player_id"] == player_id)].iloc[0]


def contributions(data: Data, match_id: int, player_id: int, group: str, k: int = 3) -> list[tuple[str, float, float, float]]:
    """[(stat, count, weight, value)] sorted by |value|, largest first."""
    s = stat_row(data, match_id, player_id)
    items = [(stat, float(s[stat]), w.weight, float(s[stat]) * w.weight) for stat, w in WEIGHTS[group].items() if s[stat] != 0]
    items.sort(key=lambda x: -abs(x[3]))
    return items[:k]


def fmt_contribs(items) -> str:
    return "; ".join(f"{LABELS[st]} {cnt:g} x {w:g} = {v:+.2f}" for st, cnt, w, v in items)


def player_actions(data: Data, match_id: int, player_id: int) -> pd.DataFrame:
    av = data.actions_values
    return av[(av["game_id"] == match_id) & (av["player_id"] == player_id)]


def fmt_actions(a: pd.DataFrame, k: int = 3, best: bool = True) -> str:
    part = a.sort_values("vaep_value", ascending=not best).head(k)
    return "; ".join(f"{r.type_name}/{r.result_name} {'H1' if r.period_id == 1 else 'H2'} {r.time_seconds / 60:.0f}' {r.vaep_value:+.3f}" for r in part.itertuples())


# --------------------------------------------------------------------- 3a. top 20
def top20(data: Data, method: str) -> list[dict]:
    r = data.ratings[method]
    other = data.ratings["vaep" if method == "baseline" else "baseline"].set_index(["match_id", "player_id"])["rating"]
    top = r.sort_values(["rating", "raw_score"], ascending=False).head(20)
    rows = []
    for row in top.itertuples():
        opp, score, week, result = data.context(row.match_id, row.team)
        s = stat_row(data, row.match_id, row.player_id)
        flags = []
        if result == "L":
            flags.append("team lost")
        if row.position_group == "GK":
            if s["goals_conceded"] >= 2:
                flags.append(f"conceded {int(s['goals_conceded'])}")
        elif s["goals"] + s["assists"] == 0:
            flags.append("no goal or assist")
        if method == "baseline":
            detail = fmt_contribs(contributions(data, row.match_id, row.player_id, row.position_group))
        else:
            a = player_actions(data, row.match_id, row.player_id)
            detail = fmt_actions(a)
            best = a.sort_values("vaep_value", ascending=False).iloc[0]
            if best["type_name"] not in SHOT_TYPES and best["defensive_value"] > 0.5 * best["vaep_value"]:
                flags.append(f"top action is defensive credit on a {best['type_name']}")
        rows.append({"player": row.player_name, "team": row.team, "opponent": opp, "score": score, "week": week, "group": row.position_group,
                     "minutes": row.minutes, "raw": row.raw_score, "rating": row.rating, "other": float(other.get((row.match_id, row.player_id), np.nan)),
                     "detail": detail, "flags": ", ".join(flags), "goals": int(s["goals"]), "assists": int(s["assists"]), "match_id": row.match_id})
    return rows


# ------------------------------------------------------------------ 3b. notable matches
def match_table(data: Data, match_id: int) -> tuple[dict, list[list], list[str], str]:
    m = data.match_index.loc[match_id]
    both = data.both[data.both["match_id"] == match_id].copy()
    both["side"] = np.where(both["team"] == m["home_team"], "H", "A")
    both = both.sort_values(["side", "is_starter", "rating_b"], ascending=[False, False, False])
    rows = []
    for row in both.itertuples():
        s = stat_row(data, match_id, row.player_id)
        marks = []
        if s["goals"]:
            marks.append(f"{int(s['goals'])}G")
        if s["assists"]:
            marks.append(f"{int(s['assists'])}A")
        if s["own_goals"]:
            marks.append("OG")
        if s["red_cards"]:
            marks.append("RED")
        if s["penalties_saved"]:
            marks.append("pen saved")
        if s["penalties_missed"]:
            marks.append("pen missed")
        if s["errors"]:
            marks.append(f"{int(s['errors'])} error")
        a = player_actions(data, match_id, row.player_id)
        rows.append([row.side, row.team, row.player_name, row.position_group, row.position, "starter" if row.is_starter else f"sub {row.minute_on:.0f}'", f"{row.minutes:.1f}",
                     f"{row.raw_score_b:.2f}", f"{row.rating_b:.1f}", f"{row.raw_score_v:.3f}", f"{row.rating_v:.1f}", " ".join(marks),
                     fmt_contribs(contributions(data, match_id, row.player_id, row.position_group, 2)), fmt_actions(a, 1)])
    lu = data.lineups
    unrated = [f"{r.team}: {r.player_name} ({r.minutes:.1f} min, on {r.minute_on:.0f}')" for r in lu[(lu["match_id"] == match_id) & (lu["minutes"] < 10)].itertuples()]
    header = f"{m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']} ({m['match_date']}, matchweek {m['match_week']}, match_id {match_id})"
    return m, rows, unrated, header


# ------------------------------------------------------------------- 3c. minutes buckets
def minutes_buckets(data: Data) -> dict:
    out = {}
    for method, col in (("baseline", "rating_b"), ("vaep", "rating_v")):
        b = data.both.copy()
        b["bucket"] = pd.cut(b["minutes"], bins=MINUTE_BINS, labels=MINUTE_LABELS, right=False)
        tab = b.groupby("bucket")[col].agg(["size", "mean", "median", "std"])
        tab["share_7plus"] = b.groupby("bucket")[col].apply(lambda s: (s >= 7).mean())
        tab["share_below_6"] = b.groupby("bucket")[col].apply(lambda s: (s < 6).mean())
        by_group = b.groupby(["bucket", "position_group"])[col].mean().unstack()
        starter = b.groupby("is_starter")[col].agg(["size", "mean", "median", "std"])
        out[method] = {"bucket": tab, "by_group": by_group, "starter": starter,
                       "rho_minutes_starters": float(spearmanr(b.loc[b["is_starter"], col], b.loc[b["is_starter"], "minutes"])[0]),
                       "rho_minutes_subs": float(spearmanr(b.loc[~b["is_starter"], col], b.loc[~b["is_starter"], "minutes"])[0])}
    return out


# --------------------------------------------------------------- 4. method comparison
def hypothesis(data: Data, row, s: pd.Series, a: pd.DataFrame) -> str:
    """A data-grounded explanation of why the two methods disagree on one player-match."""
    group = row.position_group
    base_hi = row.rating_b > row.rating_v
    reasons = []
    shots = a[a["type_name"].isin(SHOT_TYPES)]
    neg_shots = shots[shots["vaep_value"] < 0]
    best = a.sort_values("vaep_value", ascending=False).iloc[0] if len(a) else None
    team_part = sum(float(s[st]) * WEIGHTS[group][st].weight for st in TEAM_CONTEXT if st in WEIGHTS[group])
    if group == "GK":
        if base_hi:
            reasons.append(f"baseline credits saves ({int(s['saves'])} x 0.5), xG faced ({s['xg_faced']:.2f} x 0.8)"
                           + (", clean sheet (+1.5)" if s["clean_sheet"] else "") + (f", {int(s['penalties_saved'])} penalty save(s) (+2 each)" if s["penalties_saved"] else "")
                           + "; VAEP values a keeper's saves at about +0.01 each (a parry into danger -0.07), gives no credit for penalty saves"
                           + (f" and charges the foul that conceded the penalty ({a['vaep_value'].min():+.2f})" if s["penalties_conceded"] else "")
                           + f"; the keeper's VAEP sum is {row.raw_score_v:+.3f} (offensive {a['offensive_value'].sum():+.3f}, defensive {a['defensive_value'].sum():+.3f})")
        else:
            reasons.append(f"VAEP does not charge the {int(s['goals_conceded'])} goals conceded" + (f" or the {int(s['errors'])} error(s)" if s["errors"] else "")
                           + f" (baseline: -1 per goal, -2 per error) and credits the keeper for carries/passes made after the opponent's attack broke down"
                           + (f" (best action {best['type_name']} {best['vaep_value']:+.3f}, {100 * best['defensive_value'] / best['vaep_value']:.0f}% defensive value)" if best is not None and best["vaep_value"] > 0 else ""))
        return "; ".join(reasons)
    if base_hi:
        if len(neg_shots) >= 2 and neg_shots["vaep_value"].sum() < -0.2:
            reasons.append(f"VAEP charges each unconverted shot ({len(neg_shots)} shots, {neg_shots['vaep_value'].sum():+.2f} in total) while the baseline rewards xG ({s['xg']:.2f} x {WEIGHTS[group]['xg'].weight:g}) and shots on target ({int(s['shots_on_target'])} x {WEIGHTS[group]['shots_on_target'].weight:g}) whatever the outcome")
        pens = shots[(shots["type_name"] == "shot_penalty") & (shots["result_name"] == "success")]
        if len(pens):
            reasons.append(f"his converted penalty is worth {pens['vaep_value'].sum():+.2f} in VAEP (season mean +0.06: the pre-shot state already carries P(goal) = 0.79) but a full goal ({WEIGHTS[group]['goals'].weight:g}) in the baseline")
        if group == "FWD":
            if s["aerials_won"] >= 8:
                reasons.append(f"baseline credits {int(s['aerials_won'])} aerial duels won ({s['aerials_won'] * WEIGHTS[group]['aerials_won'].weight:+.2f}) as hold-up play; VAEP values them at about zero")
        elif team_part > 0.5 or s["clean_sheet"]:
            reasons.append(f"baseline team context ({team_part:+.2f}: clean sheet / low xG faced) and defensive volume (clearances {int(s['clearances'])}, aerials won {int(s['aerials_won'])}, interceptions {int(s['interceptions'])}) that VAEP values near zero (defensive value sum {a['defensive_value'].sum():+.3f})")
        elif s["aerials_won"] + s["clearances"] + s["blocks"] >= 10:
            reasons.append(f"baseline rewards defensive volume (aerials won {int(s['aerials_won'])}, clearances {int(s['clearances'])}, blocks {int(s['blocks'])}) that VAEP values near zero (defensive value sum {a['defensive_value'].sum():+.3f})")
        if s["assists"] or s["key_passes"] >= 3:
            reasons.append(f"assists/key passes ({int(s['assists'])}/{int(s['key_passes'])}) earn {s['assists'] * WEIGHTS[group]['assists'].weight + s['key_passes'] * WEIGHTS[group]['key_passes'].weight:+.2f} in the baseline; in VAEP the pass is only worth its change in scoring probability")
        if not reasons:
            reasons.append(f"baseline top contributions: {fmt_contribs(contributions(data, row.match_id, row.player_id, group, 3))}; VAEP bottom actions: {fmt_actions(a, 3, best=False)}")
    else:
        charged = []
        if s["goals_conceded"] >= 2:
            charged.append(f"{int(s['goals_conceded'])} goals conceded")
        if s["errors"]:
            charged.append(f"{int(s['errors'])} error(s)")
        if s["own_goals"]:
            charged.append("an own goal (VAEP values the own-goal touch at about -0.003)")
        if s["red_cards"]:
            charged.append("a red card (VAEP: the foul is worth about -0.06)")
        if s["penalties_conceded"]:
            charged.append("a conceded penalty")
        if charged:
            reasons.append(f"baseline charges {', '.join(charged)} (team context {team_part:+.2f}); VAEP ignores results and cards")
        if best is not None and best["defensive_value"] > 0.15 and best["type_name"] not in SHOT_TYPES:
            reasons.append(f"VAEP's credit assignment gives the first defensive-team action after a dangerous attack the whole drop in P(concede): {best['type_name']} {best['vaep_value']:+.3f}")
        off = a[(a["offensive_value"] > 0.05) & (~a["type_name"].isin(SHOT_TYPES))]
        if len(off) >= 2:
            reasons.append(f"VAEP rewards ball progression into dangerous zones ({len(off)} passes/carries worth {off['vaep_value'].sum():+.2f}) that the baseline weights at {WEIGHTS[group].get('progressive_passes').weight if 'progressive_passes' in WEIGHTS[group] else 0:g} per progressive pass")
        if not reasons:
            reasons.append(f"VAEP top actions: {fmt_actions(a, 3)}; baseline top contributions: {fmt_contribs(contributions(data, row.match_id, row.player_id, group, 3))}")
    return "; ".join(reasons)


def method_comparison(data: Data) -> dict:
    both = data.both
    out: dict = {"n": len(both)}
    out["spearman_rating"] = float(spearmanr(both["rating_b"], both["rating_v"])[0])
    out["spearman_raw"] = float(spearmanr(both["raw_score_b"], both["raw_score_v"])[0])
    out["pearson_rating"] = float(pearsonr(both["rating_b"], both["rating_v"])[0])
    out["by_group"] = [[g, len(d), float(spearmanr(d["rating_b"], d["rating_v"])[0]), float(pearsonr(d["rating_b"], d["rating_v"])[0])] for g, d in both.groupby("position_group")]
    sixty = both[both["minutes"] >= 60]
    out["spearman_60"] = (float(spearmanr(sixty["rating_b"], sixty["rating_v"])[0]), len(sixty))
    # season averages
    main_group = both.groupby("player_id")["position_group"].agg(lambda s: s.value_counts().index[0])
    sea = both.groupby("player_id").agg(n=("rating_b", "size"), player=("player_name", "first"), baseline=("rating_b", "mean"), vaep=("rating_v", "mean"))
    sea = sea[sea["n"] >= 10].join(main_group.rename("group"))
    out["season"] = {"n_players": len(sea), "spearman": float(spearmanr(sea["baseline"], sea["vaep"])[0]), "pearson": float(pearsonr(sea["baseline"], sea["vaep"])[0]),
                     "by_group": [[g, len(d), float(spearmanr(d["baseline"], d["vaep"])[0])] for g, d in sea.groupby("group")],
                     "top_baseline": sea.sort_values("baseline", ascending=False).head(12), "top_vaep": sea.sort_values("vaep", ascending=False).head(12)}
    sea = sea.assign(diff=sea["vaep"] - sea["baseline"])
    out["season_disagree"] = pd.concat([sea.sort_values("diff").head(8), sea.sort_values("diff").tail(8)])
    # player-match disagreements
    b = both.assign(diff=both["rating_v"] - both["rating_b"])
    b = b.reindex(b["diff"].abs().sort_values(ascending=False).index)
    dis = []
    for row in b.head(10).itertuples():
        s = stat_row(data, row.match_id, row.player_id)
        a = player_actions(data, row.match_id, row.player_id)
        opp, score, week, result = data.context(row.match_id, row.team)
        dis.append({"player": row.player_name, "team": row.team, "opponent": opp, "score": score, "group": row.position_group, "minutes": row.minutes,
                    "raw_b": row.raw_score_b, "rating_b": row.rating_b, "raw_v": row.raw_score_v, "rating_v": row.rating_v, "diff": row.diff,
                    "hypothesis": hypothesis(data, row, s, a)})
    out["disagreements"] = dis
    outfield = b[b["position_group"] != "GK"].head(5)
    dis_out = []
    for row in outfield.itertuples():
        s = stat_row(data, row.match_id, row.player_id)
        a = player_actions(data, row.match_id, row.player_id)
        opp, score, week, result = data.context(row.match_id, row.team)
        dis_out.append({"player": row.player_name, "team": row.team, "opponent": opp, "score": score, "group": row.position_group, "minutes": row.minutes,
                        "raw_b": row.raw_score_b, "rating_b": row.rating_b, "raw_v": row.raw_score_v, "rating_v": row.rating_v, "diff": row.diff,
                        "hypothesis": hypothesis(data, row, s, a)})
    out["disagreements_outfield"] = dis_out
    out["gk_share_top20"] = int((b.head(20)["position_group"] == "GK").sum())
    out["gk_share_top10"] = int((b.head(10)["position_group"] == "GK").sum())
    # what each method rewards: rank correlation of each stat with each raw score, within group
    st = data.stats.merge(both[["match_id", "player_id", "raw_score_b", "raw_score_v"]], on=["match_id", "player_id"])
    corr = {}
    for g, d in st.groupby("position_group"):
        rows = []
        for stat in STATS:
            if d[stat].nunique() < 2:
                continue
            rows.append([stat, float(d[stat].mean()), WEIGHTS[g][stat].weight if stat in WEIGHTS[g] else None,
                         float(spearmanr(d[stat], d["raw_score_b"])[0]), float(spearmanr(d[stat], d["raw_score_v"])[0])])
        corr[g] = sorted(rows, key=lambda r: -abs(r[4]))
    out["stat_corr"] = corr
    # linear fit of the VAEP raw score on the baseline stats (how much of VAEP the stats can explain)
    r2 = {}
    for g, d in st.groupby("position_group"):
        cols = [s_ for s_ in STATS if d[s_].std() > 0]
        X = d[cols].to_numpy(dtype=float)
        X = (X - X.mean(0)) / X.std(0)
        X = np.column_stack([np.ones(len(X)), X])
        y = d["raw_score_v"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        r2[g] = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out["vaep_r2_on_stats"] = r2
    # team context share of the baseline raw score
    tc = []
    for g, d in st.groupby("position_group"):
        team = sum(d[s_] * WEIGHTS[g][s_].weight for s_ in TEAM_CONTEXT if s_ in WEIGHTS[g])
        indiv = d["raw_score_b"] - team
        tc.append([g, float(team.var() / d["raw_score_b"].var()) if d["raw_score_b"].var() else 0.0, float(team.min()), float(team.max()), float(indiv.std())])
    out["team_context"] = tc
    # offensive vs defensive VAEP and the shot share
    av = data.actions_values
    pm = av.groupby(["game_id", "player_id"])[["offensive_value", "defensive_value"]].sum().reset_index().rename(columns={"game_id": "match_id"})
    noshot = av[~av["type_name"].isin(SHOT_TYPES)].groupby(["game_id", "player_id"])["vaep_value"].sum().rename("v_noshot").reset_index().rename(columns={"game_id": "match_id"})
    f = both.merge(pm, on=["match_id", "player_id"], how="left").merge(noshot, on=["match_id", "player_id"], how="left").fillna({"offensive_value": 0, "defensive_value": 0, "v_noshot": 0})
    vo = []
    for g, d in f.groupby("position_group"):
        vo.append([g, float(d["offensive_value"].mean()), float(d["offensive_value"].std()), float(d["defensive_value"].mean()), float(d["defensive_value"].std()),
                   float(spearmanr(d["raw_score_v"], d["offensive_value"])[0]), float(spearmanr(d["raw_score_v"], d["defensive_value"])[0]),
                   float(1 - d["v_noshot"].abs().sum() / d["raw_score_v"].abs().sum()), float((d["raw_score_v"] <= 0).mean())])
    out["vaep_parts"] = vo
    out["keeper_actions"] = av[av["type_name"].isin(["keeper_save", "keeper_punch", "keeper_claim", "shot_penalty", "shot", "shot_freekick"])].groupby(["type_name", "result_name"])["vaep_value"].agg(["size", "mean", "min", "max"]).reset_index()
    gk_ids = data.lineups.loc[data.lineups["position_group"] == "GK", ["match_id", "player_id"]].rename(columns={"match_id": "game_id"})
    gk_av = av.merge(gk_ids, on=["game_id", "player_id"])
    out["gk_value_by_type"] = gk_av.groupby("type_name")["vaep_value"].agg(["size", "sum", "mean"]).sort_values("sum", ascending=False).reset_index()
    out["special_results"] = av[av["result_name"].isin(["owngoal", "red_card", "yellow_card"])].groupby(["type_name", "result_name"])["vaep_value"].agg(["size", "mean", "min"]).reset_index()
    # result / team dependence
    m = data.matches.set_index("match_id")
    r = both.copy()
    r["home"] = r["team"] == r["match_id"].map(m["home_team"])
    hs, as_ = r["match_id"].map(m["home_score"]), r["match_id"].map(m["away_score"])
    r["gd"] = np.where(r["home"], hs - as_, as_ - hs)
    r["result"] = np.select([r["gd"] > 0, r["gd"] == 0], ["W", "D"], "L")
    out["by_result"] = r.groupby("result")[["rating_b", "rating_v"]].agg(["mean", "size"])
    out["by_result_group"] = r.groupby(["position_group", "result"])[["rating_b", "rating_v"]].mean().unstack()
    out["rho_gd"] = {"all": (float(spearmanr(r["rating_b"], r["gd"])[0]), float(spearmanr(r["rating_v"], r["gd"])[0]))}
    for g, d in r.groupby("position_group"):
        out["rho_gd"][g] = (float(spearmanr(d["rating_b"], d["gd"])[0]), float(spearmanr(d["rating_v"], d["gd"])[0]))
    out["home_away"] = r.groupby("home")[["rating_b", "rating_v"]].mean()
    pts = []
    for row in m.itertuples():
        hp = 3 if row.home_score > row.away_score else (1 if row.home_score == row.away_score else 0)
        ap = 3 if row.away_score > row.home_score else (1 if row.home_score == row.away_score else 0)
        pts += [(row.home_team, hp), (row.away_team, ap)]
    pts = pd.DataFrame(pts, columns=["team", "pts"]).groupby("team")["pts"].sum()
    tm = r.groupby("team")[["rating_b", "rating_v"]].mean().join(pts).sort_values("pts", ascending=False)
    out["team_table"] = tm
    out["rho_team_points"] = (float(spearmanr(tm["rating_b"], tm["pts"])[0]), float(spearmanr(tm["rating_v"], tm["pts"])[0]))
    # events with a fixed meaning: goals, penalties, own goals, red cards, errors
    stt = data.stats.merge(both[["match_id", "player_id", "rating_b", "rating_v"]], on=["match_id", "player_id"])
    ev = []
    one = stt[stt["goals"] == 1]
    ev.append(["scored exactly 1 goal", len(one), float(one["rating_b"].mean()), float(one["rating_v"].mean()), int((one["rating_b"] < 6).sum()), int((one["rating_v"] < 6).sum())])
    for g in POSITION_GROUPS[1:]:
        d = one[one["position_group"] == g]
        ev.append([f"scored exactly 1 goal ({g})", len(d), float(d["rating_b"].mean()), float(d["rating_v"].mean()), int((d["rating_b"] < 6).sum()), int((d["rating_v"] < 6).sum())])
    two = stt[stt["goals"] >= 2]
    ev.append(["scored 2+ goals", len(two), float(two["rating_b"].mean()), float(two["rating_v"].mean()), int((two["rating_b"] < 6).sum()), int((two["rating_v"] < 6).sum())])
    for label, mask in (("penalty missed", stt["penalties_missed"] >= 1), ("own goal", stt["own_goals"] >= 1), ("red card", stt["red_cards"] >= 1),
                        ("error leading to a shot", stt["errors"] >= 1), ("penalty conceded", stt["penalties_conceded"] >= 1), ("penalty saved (GK)", stt["penalties_saved"] >= 1),
                        ("clean sheet (DEF)", (stt["clean_sheet"] == 1) & (stt["position_group"] == "DEF")), ("4+ conceded, 60+ min (DEF)", (stt["goals_conceded"] >= 4) & (stt["minutes"] >= 60) & (stt["position_group"] == "DEF"))):
        d = stt[mask]
        ev.append([label, len(d), float(d["rating_b"].mean()) if len(d) else float("nan"), float(d["rating_v"].mean()) if len(d) else float("nan"), int((d["rating_b"] < 6).sum()), int((d["rating_v"] < 6).sum())])
    pen = av[av["type_name"] == "shot_penalty"]
    pen_scorers = pen[pen["result_name"] == "success"][["game_id", "player_id"]].drop_duplicates().rename(columns={"game_id": "match_id"}).merge(both, on=["match_id", "player_id"])
    ev.append(["converted a penalty", len(pen_scorers), float(pen_scorers["rating_b"].mean()), float(pen_scorers["rating_v"].mean()), int((pen_scorers["rating_b"] < 6).sum()), int((pen_scorers["rating_v"] < 6).sum())])
    out["events"] = ev
    out["penalty_values"] = pen.groupby("result_name")["vaep_value"].agg(["size", "mean"])
    low_scorers = stt[(stt["goals"] >= 1) & (stt["rating_v"] < 5.5)].sort_values("rating_v")
    out["low_scorers_vaep"] = low_scorers.merge(both[["match_id", "player_id", "player_name", "team"]], on=["match_id", "player_id"]).head(8)
    return out


# ----------------------------------------------------------------------- 5. stability
def _split(frame: pd.DataFrame, col: str, mask1, mask2, min_n: int = 5, group: pd.Series | None = None) -> dict:
    h1 = frame[mask1].groupby("player_id")[col].agg(["size", "mean"])
    h2 = frame[mask2].groupby("player_id")[col].agg(["size", "mean"])
    j = h1.join(h2, lsuffix="_1", rsuffix="_2", how="inner")
    j = j[(j["size_1"] >= min_n) & (j["size_2"] >= min_n)]
    res = {"n": len(j), "pearson": float(pearsonr(j["mean_1"], j["mean_2"])[0]), "spearman": float(spearmanr(j["mean_1"], j["mean_2"])[0]), "by_group": []}
    if group is not None:
        j = j.join(group.rename("group"))
        for g, d in j.groupby("group"):
            if len(d) >= 5:
                res["by_group"].append([g, len(d), float(pearsonr(d["mean_1"], d["mean_2"])[0]), float(spearmanr(d["mean_1"], d["mean_2"])[0])])
    return res


def stability(data: Data) -> dict:
    both = data.both
    main_group = both.groupby("player_id")["position_group"].agg(lambda s: s.value_counts().index[0])
    first, second = both["week"] <= 19, both["week"] >= 20
    out: dict = {"halves": {}, "raw": {}, "benchmarks": {}, "sixty": {}, "odd_even": {}}
    for method, col in (("baseline", "rating_b"), ("vaep", "rating_v")):
        out["halves"][method] = _split(both, col, first, second, group=main_group)
        out["raw"][method] = _split(both, "raw_score_b" if method == "baseline" else "raw_score_v", first, second, group=main_group)
        sixty = both[both["minutes"] >= 60]
        out["sixty"][method] = _split(sixty, col, sixty["week"] <= 19, sixty["week"] >= 20)
    parity = both.assign(parity=both.groupby("player_id").cumcount() % 2)
    for method, col in (("baseline", "rating_b"), ("vaep", "rating_v")):
        out["odd_even"][method] = _split(parity, col, parity["parity"] == 0, parity["parity"] == 1)
    # simple per-90 benchmarks from the same appearances
    st = data.stats.merge(both[["match_id", "player_id", "week"]], on=["match_id", "player_id"])
    for label, num in (("goals + assists per 90", lambda d: d["goals"] + d["assists"]), ("non-penalty xG + 0.1 x key passes per 90", lambda d: d["xg"] + 0.1 * d["key_passes"]),
                       ("passes completed per 90", lambda d: d["passes_completed"]), ("tackles + interceptions per 90", lambda d: d["tackles_won"] + d["interceptions"])):
        tmp = st.assign(v=num(st))
        def per90(d: pd.DataFrame) -> pd.Series:
            return pd.Series({"size": len(d), "mean": 90 * d["v"].sum() / d["minutes"].sum()})
        h1 = tmp[tmp["week"] <= 19].groupby("player_id").apply(per90)
        h2 = tmp[tmp["week"] >= 20].groupby("player_id").apply(per90)
        j = h1.join(h2, lsuffix="_1", rsuffix="_2", how="inner")
        j = j[(j["size_1"] >= 5) & (j["size_2"] >= 5)]
        out["benchmarks"][label] = {"n": len(j), "pearson": float(pearsonr(j["mean_1"], j["mean_2"])[0]), "spearman": float(spearmanr(j["mean_1"], j["mean_2"])[0])}
    return out
