"""Regenerate ``reports/validation.md``: ``python -m src.validation.validate [--skip-tests] [--out PATH]``.

Independent verification of the baseline and VAEP ratings (PROMPT.md Step 4). Read-only on
``data/processed``; the only file written is the report. Every table is recomputed from the
processed files, the saved scalers and (unless ``--skip-tests``) a fresh run of the test suite,
so the report can be re-run after weight or curve changes. When ``data/processed/before/``
holds the frozen pre-decision outputs, section 7 compares them with the current files. The
judgement paragraphs are the verifier's reading of the run they are dated with: the numbers
next to them regenerate, the prose does not.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import sys
import time

import pandas as pd

from src.contract import MIN_MINUTES, POSITION_GROUPS
from src.validation import analysis, before_after, checks, option3
from src.validation.data import METHODS, REPORT_PATH, Data, fmt, load, load_xg_labels, md_table

log = logging.getLogger("src.validation.validate")

NOTABLE_MATCHES = [3754013, 3754290, 3754275, 3754079, 3754092]
ISOLATION_MATCHES = [3754013, 3754290, 3754079]
ISOLATION_OTHER = 3754275  # shares Leicester players with two of the matches above

MATCH_NOTES = {
    3754013: "What happened: Vardy (34') and Mahrez (48') put Leicester 2-0 up, Remy pulled one back; Mourinho's last match in charge of Chelsea "
             "(the brief lists this as 3-1; the data and the record say 2-1). Baseline (after the decisions): Mahrez 8.5 top, Vardy 7.1, Albrighton 7.7 and Kante 7.2 read right; "
             "Ivanovic 8.0 (was 7.6) as Chelsea's best in a defeat is the one questionable row and got slightly worse because the halved team context no longer pulls a losing defender down. "
             "VAEP (option 3): Ivanovic 9.1 (was 7.7) now tops the match in a defeat, mostly for a rebound he shot at 62' that did not go in: under the after-shot rule a rebound is measured "
             "against the chance left after the first miss, so it is worth +0.23 instead of +0.01. Mahrez 8.7 is next; Vardy's goal is worth so little that he rates 5.5 (was 7.4); "
             "Kante is back to 7.2 because his dribble straight after a Chelsea miss no longer collects that miss.",
    3754290: "What happened: Huth headed in at 3' and 60', Mahrez scored at 48'; Aguero's 87' goal was a consolation. Baseline: Huth 10.0 (was 9.8), Mahrez 8.2, Fuchs 8.3 (assist, six tackles) - right; "
             "Demichelis 7.6 (was 7.0) as City's best player in a 1-3 home defeat is the price of halving the team context (his seven aerials and twelve progressive passes are no longer offset); "
             "Ulloa's 13-minute cameo at 5.0 is still harsh. VAEP: Huth's two headers are now worth 7.1 (was 9.9) because a header from a corner adds little to a state that already carried the corner's xG, "
             "while Fernandinho (8.1, was 5.9) and Vardy (8.8, no goal) rise on chances created; Fuchs' assist leaves him at 5.1. For this match the outcome-based VAEP matched the eye better.",
    3754275: "What happened: Vardy's 60' volley from Mahrez's long ball, then a second at 71'. Baseline: Fuchs 8.5 still edges Vardy 8.3 (a 7.0 DEF raw remains rarer than an 8.1 FWD raw); Firmino 5.1 (13 possessions lost). "
             "VAEP: Vardy 8.4 (was 9.0) still Leicester's best, Emre Can 7.7 (was 6.1) and Lucas 8.0 rise; Firmino 3.5.",
    3754079: "What happened: Mitrovic opened the scoring; Aguero scored five between 42' and 62'; De Bruyne (goal, two assists) and Silva (two assists) ran the game. Baseline: Aguero 10.0, De Bruyne 9.4, Silva 8.0 - right; "
             "Newcastle's beaten defenders rise from 4.5-4.7 to 5.1-5.3 and Anita from 4.9 to 5.8 after the team-context halving (still the bottom of the match, which is fair in a 1-6). VAEP: Aguero's five goals are worth +0.49 between them "
             "(+0.82 for his whole match) and make him an 8.9 (was 10.0); Kolarov 8.6, Silva 8.2 and Mitrovic 8.2 rank close to him; Fernando 7.8 (was 4.1) is no longer punished for 1.29 xG of misses.",
    3754092: "What happened: the 'Battle of the Bridge' - Kane 35', Son 44', Cahill 58', Hazard 83' ended Tottenham's title challenge; twelve yellow cards. Baseline: Cahill 8.2, Hazard 7.7, Kane 7.7, Son 7.5 - the four scorers lead, as before. "
             "VAEP (option 3): Son 8.1 (was 7.7) leads, then Azpilicueta 7.9; Ivanovic is 7.3 (was 6.5; 8.3 with xG labels alone), because his dribble straight after a Tottenham miss no longer collects that miss; "
             "Cahill falls to 7.5 (a header from a cross: little added xG), Hazard 6.8, Kane 7.0; Lloris 6.6 (was 8.0) is the goalkeeper correction working.",
}


# ----------------------------------------------------------------------------- sections
def section_tests(res: dict | None) -> list[str]:
    out = ["## 1. Test suite", ""]
    if res is None:
        out += ["Skipped (`--skip-tests`).", ""]
        return out
    out += [f"Command: `python {res['command']}` (repo root, {res['seconds']} s).", "", f"**Result: `{res['summary']}`**", ""]
    if res["failed"]:
        out += [f"**Failures ({len(res['failed'])}) - real findings, listed again in section 6 (i):**", ""] + [f"- `{f}`" for f in res["failed"]] + [""]
    else:
        out += ["Failures: **none**. Every test in the suite passes, including the explanation (Step 5) and FPL pipeline (Step 6) tests.", ""]
    return out


def section_rules(rules: dict, iso_rows: list[list], cons: list) -> list[str]:
    out = ["### 1.1 Distribution rules re-derived from `ratings_<method>.parquet`", "", "Computed here from the parquet files, independently of the tests.", ""]
    for method in METHODS:
        r = rules[method]
        out += [f"**{method}** ({r['n']} rated rows)", "",
                md_table(["rule", "pass", "detail"], [[name, "yes" if ok else "**NO**", detail] for name, ok, detail in r["checks"]], "lll"), "",
                md_table(["group", "rated rows", "mean", "sd", "10.0s", "min", "max", ">= 9.0", ">= 8.0", "< 5.0", "distinct raw", "raw <= 0"], r["groups"]), ""]
    ss = rules["baseline"]["short_starters"]
    out += [f"Starters with fewer than {MIN_MINUTES:g} minutes (the only reason a team-match has fewer than 11 rated starters): {len(ss)}, none of them a red card.", "",
            md_table(["match_id", "team", "player", "minutes", "reason"], [[r.match_id, r.team, r.player_name, f"{r.minutes:.1f}", r.reason] for r in ss.itertuples()]), ""]
    out += ["### 1.2 Isolation rule, re-checked through the public functions", "",
            "Each match is rated alone with `src.baseline.rate.rate_match(match_id, events, lineups)` / `src.vaep.rate.rate_match(match_id, actions, lineups, load_model_for_match(match_id))`, "
            f"then scaled with the saved `PercentileScaler` and compared with the cached season files; it is also rated with match {ISOLATION_OTHER}'s rows mixed into the inputs, "
            "and with its own rows absent (must raise `ValueError`).", "",
            md_table(["method", "match_id", "rows", "max abs diff raw vs cache", "same with other match mixed in", "rated rows", "max abs diff rating vs cache", "absent match raises"], iso_rows), ""]
    out += ["### 1.3 File consistency", "", md_table(["check", "pass", "detail"], [[n, "yes" if ok else "**NO**", d] for n, ok, d in cons], "lll"), ""]
    return out


def section_positions(pos: dict) -> list[str]:
    out = ["## 2. Position-group handling", "",
           "### 2.1 How a player's group is assigned and used (code trace)", "",
           "- **Assignment** (`src/ingest/positions.py`, `src/ingest/lineups.py::build_match_lineups`): the StatsBomb lineup file gives each player a list of position intervals; each interval is "
           "clipped to the player's on-pitch spans (derived from the events) and the position with the most on-pitch seconds wins (ties -> the earlier interval). Its group comes from the single "
           "table `POSITION_GROUP_BY_ID` (GK = 1; DEF = 2-8; MID = 9-16 and 18-20; FWD = 17, 21-25). The result is one `position_group` per `(match_id, player_id)` row of `lineups.parquet`; "
           "`starting_position_id` keeps the kick-off slot separately. A player therefore has one group per match, not per season.",
           "- **Baseline** (`src/baseline/stats.py`, `rate.py`): `player_match_stats` copies `position_group` from that match's lineup rows; `raw_scores` picks the weight vector `WEIGHTS[group]` row by row. "
           "Stats are counted from the player's events regardless of the position he was in when he made them, so a player who switched role at half-time is scored entirely with his majority-role weights.",
           "- **VAEP** (`src/vaep/rate.py`): the sum of `vaep_value` ignores groups entirely; `position_group` is copied from the lineups onto the output so that the scaler can split by it.",
           "- **Scaler** (`src/scale/scaler.py`): `fit` builds one reference table per group from the rated rows (`minutes >= 10`); `transform` looks up a row's own group's table and raises for a group "
           "absent from the fit. No row is ever ranked against a row of another group.",
           "- **Web export** (`src/pipeline/export_web.py`): `position_group`, `position_id` and `position` come from the lineups; ratings are joined on `(match_id, player_id)` only. The page (`web/app.js`) "
           "uses `position_id` for the pitch layout and `position_group` as a label and, since the Step 4 decisions, to show goalkeepers their baseline rating in VAEP mode (payloads unchanged).",
           "- **Where raw scores are compared across groups**: nowhere in a rating function. The only cross-group raw comparisons are in reports (`reports/vaep_splits.md` top tables, "
           "`src/baseline/rate.py::summary` top-5 print), which are descriptive.", ""]
    out += ["### 2.2 Edge cases in the real data", ""]
    out += [f"- Position ids seen in lineups: {pos['ids_seen']} (25 Secondary Striker never occurs). `position_group == group(position_id)` on every lineup row: {pos['mapping_consistent']}.",
            f"- Starters whose most-minutes group differs from their kick-off slot: **{len(pos['moved'])} of {pos['n_starters']}** ({100 * len(pos['moved']) / pos['n_starters']:.2f}%). All are MID <-> FWD/DEF reshuffles:", ""]
    ct = pos["moved_crosstab"]
    out += [md_table(["kick-off group \\ most-minutes group"] + list(ct.columns), [[idx] + [int(v) for v in row] for idx, row in ct.iterrows()]), ""]
    out += [md_table(["match_id", "team", "player", "kick-off slot", "most-minutes position", "minutes"],
                     [[r.match_id, r.team, r.player_name, r.starting_position_id, r.position, f"{r.minutes:.1f}"] for r in pos["moved"].head(12).itertuples()]), ""]
    gs = pos["gk_subs"]
    out += [f"- Goalkeepers: {pos['gk_rows']} GK rows, {pos['gk_not_position_1']} with a position id other than 1, {pos['team_matches_no_gk']} team-matches without a GK row, "
            f"{pos['team_matches_two_gk']} team-matches with two (a goalkeeper substitution). The {len(gs)} substitute goalkeepers are rated in the GK group:", "",
            md_table(["match_id", "team", "player", "on at", "minutes"], [[r.match_id, r.team, r.player_name, f"{r.minute_on:.0f}'", f"{r.minutes:.1f}"] for r in gs.itertuples()]), ""]
    og = pos["outfield_in_goal"]
    out += [f"- Outfield players in goal: {len(og)} player-match(es) with events in the Goalkeeper position or of type Goal Keeper while assigned to another group "
            "(an outfield player who took the gloves after a late dismissal; he keeps his majority group, which is correct under the most-minutes rule):", "",
            md_table(["match_id", "player", "team", "group", "events", "first minute", "last minute"], [[r.match_id, r.player, r.team, r.position_group, r.events, r.first_minute, r.last_minute] for r in og.itertuples()]) if len(og) else "none", ""]
    eo = pos["events_other_group"]
    out += [f"- Players in two or more groups over the season: **{pos['players_multi']} of {pos['n_players']}** ({100 * pos['players_multi'] / pos['n_players']:.0f}%), {pos['players_three']} in three; "
            f"players who are both a goalkeeper and an outfield player: {pos['players_gk_and_outfield']}.",
            f"- Events made in a position of another group than the one assigned for the match: {eo['over_30']} of {eo['player_matches']} player-matches have 30% or more of their events there, {eo['over_50']} more than half (half-time role swaps)."]
    iv = pos["intervals"]
    if iv is not None:
        out += [f"- Lineup intervals (raw StatsBomb JSON) spanning two or more groups: **{len(iv)} player-matches** ({100 * len(iv) / 10469:.1f}%); in {int((iv['share_main'] < 0.6).sum())} of them the chosen group covers "
                f"less than 60% of the minutes and in {int((iv['share_main'] < 0.75).sum())} less than 75%. The closest calls:", "",
                md_table(["match_id", "team", "player", "group chosen", "share of minutes", "minutes", "minutes by group"],
                         [[r.match_id, r.team, r.player, r.main, f"{r.share_main:.3f}", f"{r.minutes:.1f}", r.minutes_by_group] for r in iv.sort_values("share_main").head(10).itertuples()]), ""]
    else:
        out += ["- Raw StatsBomb JSON not available: the interval-share analysis was skipped.", ""]
    out += ["**Do multi-group players rate consistently?** For every player with at least 3 rated appearances in each of two groups, the mean rating in the group he played less often "
            "minus the mean in his main group (a within-player comparison, so team and ability are held roughly constant; the matches differ, so this is indicative only):", ""]
    rows = []
    for method in METHODS:
        t = pos["paired"][method]
        rows.append([method, len(t), fmt(t["diff"].mean()), fmt(t["diff"].std()), int((t["diff"].abs() > 0.5).sum())])
    out += [md_table(["method", "players", "mean (other - main)", "sd", "players with |diff| > 0.5"], rows), ""]
    rows = []
    pairs = sorted(set(pos["paired"]["baseline"]["pair"]) | set(pos["paired"]["vaep"]["pair"]))
    for p in pairs:
        tb = pos["paired"]["baseline"][pos["paired"]["baseline"]["pair"] == p]
        tv = pos["paired"]["vaep"][pos["paired"]["vaep"]["pair"] == p]
        rows.append([p, len(tb), fmt(tb["diff"].mean()) if len(tb) else "", fmt(tv["diff"].mean()) if len(tv) else ""])
    out += [md_table(["main -> other group", "players", "baseline mean diff", "vaep mean diff"], rows), ""]
    t = pos["paired"]["baseline"].sort_values("diff")
    out += ["Largest baseline shifts (negative = rated lower when playing out of his usual group):", "",
            md_table(["player", "main", "n", "mean", "other", "n", "mean", "diff"],
                     [[r.player, r.main, r.n_main, fmt(r.mean_main, 2), r.other, r.n_other, fmt(r.mean_other, 2), fmt(r.diff, 2)] for r in pd.concat([t.head(5), t.tail(5)]).itertuples()]), ""]
    out += ["### 2.3 Roles pooled inside a group", "",
            "Mean rating of 60+ minute appearances by the role actually played (from `position_id`). The groups are anchored at 6.5 each, so a spread between roles inside a group means "
            "the group pools roles whose stat profiles the method treats differently:", ""]
    role = pos["role"]
    out += [md_table(["group", "role", "n", "baseline mean", "vaep mean", "baseline raw mean", "vaep raw mean"],
                     [[r.position_group, f"{r.role} ({checks.ROLE_LABELS.get(r.role, '')})", r.n, fmt(r.baseline, 3), fmt(r.vaep, 3), fmt(r.raw_b, 2), fmt(r.raw_v, 3)] for r in role.itertuples()]), ""]
    w = pos["wide"]
    out += [f"The MID/FWD boundary is formation-dependent: Right/Left Midfield (ids 12/16, MID) account for {w['wide_mid']} appearances and Right/Left Wing (17/21, FWD) for {w['wingers']}; "
            f"{w['players_both']} players appear as both over the season, which is the main reason {pos['players_multi']} players have two groups.", ""]
    return out


def section_eye(data: Data, tops: dict, matches: list, buckets: dict) -> list[str]:
    out = ["## 3. Eye test", ""]
    for method in METHODS:
        rows = tops[method]
        out += [f"### 3.1 Top 20 single-match ratings: {method}" if method == "baseline" else f"### 3.2 Top 20 single-match ratings: {method}", ""]
        detail_header = "three biggest stat contributions (count x weight = value)" if method == "baseline" else "three most valuable actions (type/result, half, minute, value)"
        out += [md_table(["#", "player", "team", "opponent", "score", "wk", "group", "min", "raw", "rating", "other method", detail_header, "flags"],
                         [[i + 1, r["player"], r["team"], r["opponent"], r["score"], r["week"], r["group"], f"{r['minutes']:.0f}", fmt(r["raw"], 3 if method == "vaep" else 2), f"{r['rating']:.1f}",
                           f"{r['other']:.1f}", r["detail"], r["flags"]] for i, r in enumerate(rows)]), ""]
        n_goal = sum(1 for r in rows if r["goals"] >= 1)
        n_flag = sum(1 for r in rows if r["flags"])
        n_lost = sum(1 for r in rows if "team lost" in r["flags"])
        n_none = sum(1 for r in rows if r["goals"] + r["assists"] == 0 and r["group"] != "GK")
        out += [f"{n_goal} of the 20 scored at least once; {n_none} outfield rows have neither a goal nor an assist; {n_flag} rows carry a flag ({n_lost} in a defeat). Automatic flags: team lost; no goal or assist (outfield); 2+ conceded (GK); "
                "top VAEP action is defensive credit on a non-shot action.", ""]
    out += ["**Verifier's judgement (re-validation run after the Step 4 decisions; the pre-decision reading is in `reports/validation_before.md`).** Baseline: the list is still goal-driven and legitimate "
            "(eight hat-tricks or better, two-goal defenders, Forster's eleven-save 0-0 at Arsenal); the entries that moved in - Mane's hat-trick, De Bruyne's goal and two assists, Ramsey, Boruc's clean sheet with a penalty save - "
            "are as defensible as those they replaced (Aguero at Chelsea, Barkley, Silva, Koscielny), and the five-goal Aguero game remains the clear number one. "
            "VAEP (xG labels with the option-3 rules; the xG-label list is in section 8.2): Bony (two goals and an assist), Ramsey (a goal and an assist) and Lamine Kone (two goals) top it, Steve Cook and Kouyate are gone "
            "(their clearances straight after missed penalties were worth +0.50 and +0.54 and are now worth about -0.06), and Walcott, Jonny Evans, Mane's hat-trick and Coleman came in. What still reads wrong: eight of the twenty have "
            "neither a goal nor an assist, and five come from defeats without one - Baines 9.8 in a 0-1 at Old Trafford, van Aanholt 9.6 in a 0-2 home defeat for one tackle worth +0.37 that ended a dribble, Payet 9.5 in a 1-4 home "
            "defeat, Mignolet 9.4 conceding two, Coleman 9.4 in a 0-1 home defeat. Aguero's five goals are worth +0.49 between them and an 8.9, outside the top 20. Option 3 removed the clearance-after-a-miss lottery "
            "but not the underlying problem: the list still rewards being on the ball around big chances more than scoring them, which is why the baseline is the headline rating.", ""]
    out += ["### 3.3 Five notable matches", "",
            "B = baseline, V = VAEP. 'marks' are derived from the match events (G goals incl. penalties, A assists, OG, RED, errors, penalties). Both teams, every rated player; unrated (< 10 minutes) players are listed below each table. "
            "Section 7.1 repeats these matches with the pre-decision ratings side by side.", ""]
    for m, rows, unrated, header in matches:
        out += [f"#### {header}", "",
                md_table(["side", "team", "player", "group", "position", "role", "min", "B raw", "B", "V raw", "V", "marks", "biggest baseline contributions", "best VAEP action"], rows), ""]
        if unrated:
            out += ["Unrated (< 10 minutes): " + "; ".join(unrated), ""]
        note = MATCH_NOTES.get(int(m.name) if hasattr(m, "name") else 0)
        if note:
            out += [f"*Judgement (verifier, re-validation run):* {note}", ""]
    out += ["### 3.4 Ratings by minutes played and by starter / substitute", ""]
    for method in METHODS:
        b = buckets[method]
        out += [f"**{method}**", "",
                md_table(["minutes", "n", "mean", "median", "sd", "share >= 7.0", "share < 6.0"] + [f"{g} mean" for g in POSITION_GROUPS],
                         [[idx, int(row["size"]), fmt(row["mean"]), f"{row['median']:.1f}", fmt(row["std"]), fmt(row["share_7plus"]), fmt(row["share_below_6"])]
                          + [fmt(b["by_group"].loc[idx, g], 2) if g in b["by_group"].columns and not pd.isna(b["by_group"].loc[idx, g]) else "" for g in POSITION_GROUPS]
                          for idx, row in b["bucket"].iterrows()]), "",
                md_table(["starter", "n", "mean", "median", "sd"], [[("starter" if idx else "substitute"), int(row["size"]), fmt(row["mean"]), f"{row['median']:.1f}", fmt(row["std"])] for idx, row in b["starter"].iterrows()]), "",
                f"Spearman(rating, minutes): starters {fmt(b['rho_minutes_starters'])}, substitutes {fmt(b['rho_minutes_subs'])}.", ""]
    bb, bv = buckets["baseline"]["bucket"], buckets["vaep"]["bucket"]
    out += [f"**Judgement.** A 10-20 minute cameo averages {bb.loc['10-20', 'mean']:.2f} (baseline) / {bv.loc['10-20', 'mean']:.2f} (VAEP) against 6.5 for the season, "
            f"{100 * bb.loc['10-20', 'share_below_6']:.0f}% / {100 * bv.loc['10-20', 'share_below_6']:.0f}% of them rate below 6.0 and only {100 * bb.loc['10-20', 'share_7plus']:.1f}% / {100 * bv.loc['10-20', 'share_7plus']:.1f}% reach 7.0. "
            "Both raw scores are sums, so a quiet cameo scores near zero; what has changed since the pre-decision run is where zero sits inside each group (section 7.7): with the team context halved, far fewer "
            "starters in beaten teams go negative, so the same quiet cameo now ranks lower. See finding (ii) 'subs' in section 6.", ""]
    return out


def section_compare(comp: dict) -> list[str]:
    out = ["## 4. Comparing the methods", "", "### 4.1 Rank correlation between baseline and VAEP", "",
           f"All {comp['n']} rated player-matches: Spearman **{comp['spearman_rating']:.3f}** on ratings ({comp['spearman_raw']:.3f} on raw scores, Pearson {comp['pearson_rating']:.3f}); "
           f"60+ minute appearances only: {comp['spearman_60'][0]:.3f} (n = {comp['spearman_60'][1]}).", "",
           md_table(["group", "n", "Spearman", "Pearson"], [[g, n, fmt(s), fmt(p)] for g, n, s, p in comp["by_group"]]), "",
           f"Season averages per player (at least 10 rated appearances, {comp['season']['n_players']} players): Spearman **{comp['season']['spearman']:.3f}**, Pearson {comp['season']['pearson']:.3f}; "
           "by the player's main group: " + ", ".join(f"{g} {fmt(s)} (n = {n})" for g, n, s in comp["season"]["by_group"]) + ".", ""]
    sb, sv = comp["season"]["top_baseline"], comp["season"]["top_vaep"]
    out += ["Season-average leaders (mean rating, 10+ appearances):", "",
            md_table(["#", "baseline leader", "group", "n", "baseline", "vaep", "", "vaep leader", "group", "n", "baseline", "vaep"],
                     [[i + 1, a.player, a.group, a.n, fmt(a.baseline, 2), fmt(a.vaep, 2), "", b.player, b.group, b.n, fmt(b.baseline, 2), fmt(b.vaep, 2)]
                      for i, (a, b) in enumerate(zip(sb.itertuples(), sv.itertuples()))]), ""]
    sd = comp["season_disagree"]
    out += ["Players whose season averages disagree most (vaep - baseline):", "",
            md_table(["player", "group", "n", "baseline", "vaep", "diff"], [[r.player, r.group, r.n, fmt(r.baseline, 2), fmt(r.vaep, 2), f"{r.diff:+.2f}"] for r in sd.itertuples()]), ""]
    out += ["### 4.2 The ten player-matches where the methods disagree most", "",
            f"Unit: player-match (the largest |vaep rating - baseline rating| over all rated rows). {comp['gk_share_top20']} of the 20 largest disagreements are goalkeepers; the five largest outfield "
            "disagreements follow separately. Hypotheses are generated from the player's own stats and action values (the generator was written for the outcome-label VAEP; under xG labels the "
            "'unconverted shot' clause rarely fires and the defensive-credit clause is the usual explanation).", ""]
    for i, d in enumerate(comp["disagreements"], 1):
        out += [f"{i}. **{d['player']}** ({d['team']} v {d['opponent']}, {d['score']}, {d['group']}, {d['minutes']:.0f} min): baseline {d['raw_b']:.2f} -> **{d['rating_b']:.1f}**, VAEP {d['raw_v']:+.3f} -> **{d['rating_v']:.1f}** "
                f"(diff {d['diff']:+.1f}). Hypothesis: {d['hypothesis']}."]
    out += ["", "Largest outfield disagreements:", ""]
    for i, d in enumerate(comp["disagreements_outfield"], 1):
        out += [f"{i}. **{d['player']}** ({d['team']} v {d['opponent']}, {d['score']}, {d['group']}, {d['minutes']:.0f} min): baseline {d['raw_b']:.2f} -> **{d['rating_b']:.1f}**, VAEP {d['raw_v']:+.3f} -> **{d['rating_v']:.1f}** "
                f"(diff {d['diff']:+.1f}). Hypothesis: {d['hypothesis']}."]
    out += ["", "### 4.3 What the two methods actually reward", "",
            "Spearman rank correlation of each baseline stat with the VAEP raw score and with the baseline raw score, within group (rated rows; the twelve stats most related to VAEP shown, "
            "plus every stat whose two correlations differ markedly):", ""]
    for g in POSITION_GROUPS:
        rows = comp["stat_corr"][g]
        shown = rows[:12] + [r for r in rows[12:] if (r[3] > 0.2 and r[4] < 0) or (r[3] < -0.2 and r[4] > 0) or abs(r[3] - r[4]) > 0.4]
        out += [f"**{g}** (linear fit of the VAEP raw score on all 35 stats: R² = {comp['vaep_r2_on_stats'][g]:.2f})", "",
                md_table(["stat", "mean per match", "baseline weight", "rho with baseline raw", "rho with VAEP raw"],
                         [[s, fmt(m, 2), ("" if w is None else f"{w:g}"), fmt(rb), fmt(rv)] for s, m, w, rb, rv in shown]), ""]
    out += ["Team context in the baseline (`goals_conceded`, `clean_sheet`, `xg_faced` are identical for every teammate on the pitch):", "",
            md_table(["group", "share of raw-score variance from team context", "team-context min", "team-context max", "sd of the individual part"],
                     [[g, fmt(s), fmt(lo, 2), fmt(hi, 2), fmt(sd, 2)] for g, s, lo, hi, sd in comp["team_context"]]), ""]
    out += ["VAEP decomposed (rated rows): offensive vs defensive value, how much the raw score tracks each, the share of |value| that comes from shots, and the share of appearances with a non-positive sum:", "",
            md_table(["group", "offensive mean", "offensive sd", "defensive mean", "defensive sd", "rho(raw, offensive)", "rho(raw, defensive)", "share of |value| from shots", "share raw <= 0"],
                     [[g, fmt(om), fmt(osd), fmt(dm), fmt(dsd), fmt(ro), fmt(rd), fmt(sh), fmt(z)] for g, om, osd, dm, dsd, ro, rd, sh, z in comp["vaep_parts"]]), ""]
    ka = comp["keeper_actions"]
    out += ["Value of shot and keeper actions in VAEP (all actions, cross-fitted):", "",
            md_table(["action", "result", "n", "mean value", "min", "max"], [[r.type_name, r.result_name, int(r["size"]), fmt(r["mean"], 4), fmt(r["min"]), fmt(r["max"])] for _, r in ka.iterrows()]), ""]
    gv = comp["gk_value_by_type"]
    out += [f"Where goalkeepers' VAEP comes from (all GK-group rows; total {gv['sum'].sum():.1f}):", "",
            md_table(["action type", "n", "total value", "mean value"], [[r.type_name, int(r["size"]), fmt(r["sum"], 2), fmt(r["mean"], 4)] for _, r in gv.iterrows()]), ""]
    sr = comp["special_results"]
    out += ["Own goals and cards in VAEP (SPADL result of the action):", "",
            md_table(["action", "result", "n", "mean value", "min"], [[r.type_name, r.result_name, int(r["size"]), fmt(r["mean"], 3), fmt(r["min"], 3)] for _, r in sr.iterrows()]), ""]
    br = comp["by_result"]
    out += ["Dependence on the match result:", "",
            md_table(["result", "n", "baseline mean", "vaep mean"], [[idx, int(br.loc[idx, ("rating_b", "size")]), fmt(br.loc[idx, ("rating_b", "mean")]), fmt(br.loc[idx, ("rating_v", "mean")])] for idx in ["W", "D", "L"]]), ""]
    brg = comp["by_result_group"]
    out += [md_table(["group", "baseline W", "baseline D", "baseline L", "vaep W", "vaep D", "vaep L"],
                     [[g] + [fmt(brg.loc[g, ("rating_b", r)], 2) for r in "WDL"] + [fmt(brg.loc[g, ("rating_v", r)], 2) for r in "WDL"] for g in POSITION_GROUPS]), ""]
    rg = comp["rho_gd"]
    out += ["Spearman(rating, own team's goal difference in the match): " + ", ".join(f"{k} baseline {fmt(v[0])} / vaep {fmt(v[1])}" for k, v in rg.items()) + ".",
            f"Home / away mean: baseline {comp['home_away'].loc[True, 'rating_b']:.3f} / {comp['home_away'].loc[False, 'rating_b']:.3f}, vaep {comp['home_away'].loc[True, 'rating_v']:.3f} / {comp['home_away'].loc[False, 'rating_v']:.3f}.", ""]
    tm = comp["team_table"]
    out += [f"Team mean rating against final league points (Spearman: baseline {comp['rho_team_points'][0]:.3f}, vaep {comp['rho_team_points'][1]:.3f}):", "",
            md_table(["team", "points", "baseline mean", "vaep mean"], [[idx, int(r.pts), fmt(r.rating_b), fmt(r.rating_v)] for idx, r in tm.iterrows()]), ""]
    out += ["Events with a fixed football meaning, and the mean rating each method gives the player (goals include penalties):", "",
            md_table(["event", "n", "baseline mean", "vaep mean", "baseline < 6.0", "vaep < 6.0"], [[e, n, fmt(b, 2), fmt(v, 2), lb, lv] for e, n, b, v, lb, lv in comp["events"]]), ""]
    pv = comp["penalty_values"]
    out += [f"VAEP value of the penalty kick to the taker: scored {pv.loc['success', 'mean']:+.3f} (n = {int(pv.loc['success', 'size'])}), missed {pv.loc['fail', 'mean']:+.3f} (n = {int(pv.loc['fail', 'size'])}); "
            "socceraction fixes the pre-penalty scoring probability at 0.792.", ""]
    ls = comp["low_scorers_vaep"]
    if len(ls):
        out += ["Goal-scorers rated below 5.5 by VAEP:", "",
                md_table(["match_id", "player", "team", "group", "min", "goals", "xG", "shots on target", "baseline", "vaep"],
                         [[r.match_id, r.player_name, r.team, r.position_group, f"{r.minutes:.0f}", int(r.goals), fmt(r.xg, 2), int(r.shots_on_target), f"{r.rating_b:.1f}", f"{r.rating_v:.1f}"] for r in ls.itertuples()]), ""]
    return out


def section_stability(stab: dict) -> list[str]:
    out = ["## 5. Stability: first half of the season vs second half", "",
           "Players with at least 5 rated appearances in matchweeks 1-19 and at least 5 in 20-38; the correlation of their mean rating in the two halves (higher = the rating carries more persistent player signal).", ""]
    rows = []
    for method in METHODS:
        h = stab["halves"][method]
        rows.append([f"{method} rating", h["n"], fmt(h["pearson"]), fmt(h["spearman"])])
    for method in METHODS:
        h = stab["raw"][method]
        rows.append([f"{method} raw score", h["n"], fmt(h["pearson"]), fmt(h["spearman"])])
    for method in METHODS:
        h = stab["sixty"][method]
        rows.append([f"{method} rating, 60+ minute appearances only", h["n"], fmt(h["pearson"]), fmt(h["spearman"])])
    for method in METHODS:
        h = stab["odd_even"][method]
        rows.append([f"{method} rating, odd vs even appearances (controls for form and transfers)", h["n"], fmt(h["pearson"]), fmt(h["spearman"])])
    for label, h in stab["benchmarks"].items():
        rows.append([f"benchmark: {label}", h["n"], fmt(h["pearson"]), fmt(h["spearman"])])
    out += [md_table(["signal", "players", "Pearson", "Spearman"], rows), ""]
    rows = []
    for g in POSITION_GROUPS:
        row = [g]
        for method in METHODS:
            hit = [x for x in stab["halves"][method]["by_group"] if x[0] == g]
            row += [hit[0][1], fmt(hit[0][2]), fmt(hit[0][3])] if hit else ["", "", ""]
        rows.append(row)
    out += ["Per position group (rating, halves):", "", md_table(["group", "players", "baseline Pearson", "baseline Spearman", "players", "vaep Pearson", "vaep Spearman"], rows), ""]
    hb, hv = stab["halves"]["baseline"], stab["halves"]["vaep"]
    better = "baseline" if hb["pearson"] > hv["pearson"] else "vaep"
    out += [f"**The {better} rating is the more reliable signal** (Pearson {hb['pearson']:.3f} vs {hv['pearson']:.3f}, Spearman {hb['spearman']:.3f} vs {hv['spearman']:.3f} over {hb['n']} players; "
            f"odd/even split {stab['odd_even']['baseline']['pearson']:.3f} vs {stab['odd_even']['vaep']['pearson']:.3f}). "
            "Both are far below the pure volume benchmark (completed passes per 90, which measures role and team style more than quality). Goalkeepers are the least stable group under both methods (19 players).", ""]
    return out


# ------------------------------------------------------------------ 7. before / after
def section_before_after(ba: dict | None, data: Data) -> list[str]:
    out = ["## 7. Before / after the Step 4 decisions", ""]
    if ba is None:
        out += skipped_comparisons_note()
        return out
    out += ["Before = the frozen pre-decision outputs in `data/processed/before/` (outcome-label VAEP, original baseline weights); after = the state right after the decisions: "
            "the current baseline files and the xG-label VAEP frozen in `data/processed/xg_labels/` (the current VAEP adds option 3, section 8). "
            "Decisions applied: D (DEF/MID team-context weights halved), E (FWD aerials/possession weights eased), F (`penalties_scored` at 1.25, `goals` non-penalty only, `key_passes` without assists), "
            "B (VAEP shots valued through xG soft labels), A (goalkeepers shown with their baseline rating in VAEP mode on the page). G, H and the 12/16 reclassification were skipped.", ""]
    # 7.1 matches
    out += ["### 7.1 The five notable matches, before and after", "", "B = baseline, V = VAEP; 'd' = after minus before. Marks from the match events (G = goals incl. penalties).", ""]
    for mid, (header, rows) in ba["matches"].items():
        out += [f"#### {header}", "", md_table(["side", "player", "group", "role", "min", "marks", "B before", "B after", "d", "V before", "V after", "d"], rows), ""]
    out += ["*Judgement.* Baseline: the changes are small (median |d| about 0.1) and mostly in the intended direction - losing defenders and midfielders rise (in the 1-6 Newcastle's defenders go from 4.5-4.7 to 5.1-5.3 and Anita from 4.9 to 5.8; "
            "Lovren/Sakho +0.4/+0.3 at Leicester), clean-sheet defenders ease slightly, penalty scorers lose about 0.4; the only moves that read worse are losing centre-backs who were partly carried by "
            "volume (Demichelis 7.0 -> 7.6 in the 1-3, Ivanovic 7.6 -> 8.0 in the 2-1). VAEP: large moves in both directions. Improvements: Fernando 4.1 -> 7.7 (1.29 xG of misses no longer charged), "
            "Lloris 8.0 -> 6.6 and Krul 4.7 -> 6.5 (keepers judged on chances faced), Costa 5.3 -> 6.7, Kante 7.3 -> 8.2. Regressions: Aguero's five goals 10.0 -> 8.9, Huth's two headers 9.9 -> 7.1, "
            "Vardy's goal v Chelsea 7.4 -> 5.6, Cahill's header 8.8 -> 7.4, Fuchs' assist 6.9 -> 5.3: the decisive actions of these matches are now worth less than the build-up and the defending around them.", ""]
    # 7.2 top 20
    for method in METHODS:
        out += [f"### 7.2 Top 20 single-match ratings before and after: {method}", "",
                md_table(["#", "before", "rating", "same row's rating after", "", "after", "same row's rating before", "rating", ""], ba["top20"][method]), ""]
    nb = sum(1 for r in ba["top20"]["baseline"] if r[8]); nv = sum(1 for r in ba["top20"]["vaep"] if r[8])
    out += [f"{nb} of the 20 baseline rows are new; {nv} of the 20 VAEP rows are new (the VAEP list was replaced entirely; see 3.2 for what the new rows contain).", ""]
    # 7.3 disagreements
    out += ["### 7.3 Ten largest baseline-vs-VAEP disagreements, before and after", "",
            "Before:", "", md_table(["player-match", "group", "min", "baseline", "vaep", "vaep - baseline"], ba["disagree"]["before"]), "",
            "After:", "", md_table(["player-match", "group", "min", "baseline", "vaep", "vaep - baseline"], ba["disagree"]["after"]), "",
            f"Goalkeepers in the before list: {sum(1 for r in ba['disagree']['before'] if r[1] == 'GK')}; in the after list: {sum(1 for r in ba['disagree']['after'] if r[1] == 'GK')}. "
            "The after list is dominated by defenders whom VAEP credits with one big action (van Aanholt's tackle at the end of a dribble, Steve Cook's clearance after a missed penalty, Delaney's missed shot worth +0.34 after a "
            "fumbled save) and goal-scorers whose goals add little beyond the state that preceded them (Lukaku's two goals are worth +0.03 and -0.01, Antonio's +0.15, and the centre-back Ashley Williams' goal -0.02).", ""]
    # 7.4 agreement + old vs new
    ag_b, ag_a = ba["agree"]["before"], ba["agree"]["after"]
    rows = [[k, fmt(ag_b[k]), fmt(ag_a[k]), f"{ag_a[k] - ag_b[k]:+.3f}"] for k in ("all", "GK", "DEF", "MID", "FWD", "season")]
    out += ["### 7.4 Agreement between the methods, before and after", "",
            md_table(["Spearman(baseline, vaep)", "before", "after", "change"], rows), "",
            f"(season = per-player season averages, {ag_a['season_n']} players with 10+ appearances.)", "",
            "Spearman between each method's own old and new ratings (same player-matches):", "",
            md_table(["method", "rows", "all", "GK", "DEF", "MID", "FWD", "mean |change|", "rows moved >= 1.0"], ba["old_vs_new"]), ""]
    # 7.5 stability
    rows = []
    for method in METHODS:
        hb, ha = ba["stability"]["before"]["halves"][method], ba["stability"]["after"]["halves"][method]
        rows.append([f"{method} - all", hb["n"], fmt(hb["pearson"]), fmt(hb["spearman"]), fmt(ha["pearson"]), fmt(ha["spearman"])])
        for g in POSITION_GROUPS:
            b = [x for x in hb["by_group"] if x[0] == g][0]
            a = [x for x in ha["by_group"] if x[0] == g][0]
            rows.append([f"{method} - {g}", b[1], fmt(b[2]), fmt(b[3]), fmt(a[2]), fmt(a[3])])
        ob, oa = ba["stability"]["before"]["odd_even"][method], ba["stability"]["after"]["odd_even"][method]
        rows.append([f"{method} - odd/even split", ob["n"], fmt(ob["pearson"]), fmt(ob["spearman"]), fmt(oa["pearson"]), fmt(oa["spearman"])])
    out += ["### 7.5 Split-half stability before and after (matchweeks 1-19 vs 20-38, 5+ appearances in each half)", "",
            md_table(["signal", "players", "Pearson before", "Spearman before", "Pearson after", "Spearman after"], rows), ""]
    # 7.6 team context
    tcb, tca = ba["team_context"]["before"], ba["team_context"]["after"]
    rows = [[g, fmt(tcb[g]) if tcb else "n/a", fmt(tca[g])] for g in ("GK", "DEF", "MID", "FWD")]
    rb, ra = ba["result"]["before"], ba["result"]["after"]
    out += ["### 7.6 Team context in the baseline, before and after", "",
            md_table(["group", "share of raw variance from team context, before", "after"], rows), "",
            f"Dependence on the result: winners/losers average {rb['W_b']:.2f}/{rb['L_b']:.2f} before and {ra['W_b']:.2f}/{ra['L_b']:.2f} after (baseline); Spearman(rating, goal difference) "
            f"baseline {rb['rho_b']:.3f} -> {ra['rho_b']:.3f} (DEF {rb['rho_b_DEF']:.3f} -> {ra['rho_b_DEF']:.3f}, MID {rb['rho_b_MID']:.3f} -> {ra['rho_b_MID']:.3f}, GK {rb['rho_b_GK']:.3f} -> {ra['rho_b_GK']:.3f}); "
            f"VAEP {rb['rho_v']:.3f} -> {ra['rho_v']:.3f} (DEF {rb['rho_v_DEF']:.3f} -> {ra['rho_v_DEF']:.3f}, GK {rb['rho_v_GK']:.3f} -> {ra['rho_v_GK']:.3f}).", ""]
    # events
    eb, ea = ba["events"]["before"], ba["events"]["after"]
    out += ["Mean rating by event, before -> after:", "",
            md_table(["event", "n", "baseline before", "baseline after", "vaep before", "vaep after"],
                     [[k, eb[k][0], fmt(eb[k][1], 2), fmt(ea[k][1], 2), fmt(eb[k][2], 2), fmt(ea[k][2], 2)] for k in eb]), ""]
    rb_, ra_ = ba["roles"]["before"], ba["roles"]["after"]
    out += ["Roles inside a group (60+ minute appearances), before -> after:", "",
            md_table(["group", "role", "n", "baseline before", "baseline after", "vaep before", "vaep after"],
                     [[g, r, int(rb_.loc[(g, r), "n"]), fmt(rb_.loc[(g, r), "baseline"], 2), fmt(ra_.loc[(g, r), "baseline"], 2), fmt(rb_.loc[(g, r), "vaep"], 2), fmt(ra_.loc[(g, r), "vaep"], 2)] for g, r in rb_.index]), ""]
    # 7.7 minutes buckets
    out += ["### 7.7 Ratings by minutes bucket, before and after", ""]
    for method in METHODS:
        bb, ba_ = ba["buckets"]["before"][method]["bucket"], ba["buckets"]["after"][method]["bucket"]
        out += [f"**{method}**", "",
                md_table(["minutes", "n", "mean before", "mean after", "share < 6.0 before", "share < 6.0 after", "share >= 7.0 before", "share >= 7.0 after"],
                         [[idx, int(bb.loc[idx, "size"]), fmt(bb.loc[idx, "mean"]), fmt(ba_.loc[idx, "mean"]), fmt(bb.loc[idx, "share_below_6"]), fmt(ba_.loc[idx, "share_below_6"]), fmt(bb.loc[idx, "share_7plus"]), fmt(ba_.loc[idx, "share_7plus"])] for idx in bb.index]), ""]
    sb, sa = ba["subs"]["before"], ba["subs"]["after"]
    rows = []
    for g in ("DEF", "MID", "FWD"):
        rows.append([g, sb[f"cameo_n_{g}"], fmt(sb[f"cameo_mean_{g}"], 2), fmt(sa[f"cameo_mean_{g}"], 2), fmt(sb[f"cameo_median_raw_{g}"], 2), fmt(sa[f"cameo_median_raw_{g}"], 2),
                     fmt(sb[f"cameo_pct_{g}"]), fmt(sa[f"cameo_pct_{g}"]), fmt(sb[f"share_neg_starters_{g}"]), fmt(sa[f"share_neg_starters_{g}"]), fmt(sb[f"p25_{g}"], 2), fmt(sa[f"p25_{g}"], 2),
                     f"{sa[f'p30_{g}'] - sa[f'cameo_median_raw_{g}']:+.2f}"])
    out += ["Why the 10-20 minute baseline cameos moved (outfield groups):", "",
            md_table(["group", "cameos", "mean rating before", "after", "median cameo raw before", "after", "its percentile in the group before", "after", "share of 60+ min appearances with raw < 0 before", "after",
                      "group raw p25 before", "after", "raw offset that would put the median cameo at p30 now"], rows), ""]
    # 7.8 shot values
    out += ["### 7.8 What VAEP pays for shots and saves, before and after (mean value credited to the acting player)", "",
            md_table(["action", "n", "offensive before", "offensive after", "defensive before", "defensive after", "vaep before", "vaep after"],
                     [[b[0], b[1], fmt(b[2], 3), fmt(a[2], 3), fmt(b[3], 3), fmt(a[3], 3), fmt(b[4], 3), fmt(a[4], 3)] for b, a in zip(ba["shots"]["before"], ba["shots"]["after"])]), ""]
    ps = ba["penalty_states"]
    if "error" not in ps:
        out += [f"Penalty states (the {ps['n']} `shot_penalty` game states, cached features scored with their own fold models): label mean {ps['label_mean']:.3f} (the kick's xG 0.7835 plus any rebound in the window), "
                f"model prediction mean {ps['pred_mean']:.3f} - {ps['pred_scored']:.3f} for scored kicks and {ps['pred_missed']:.3f} for missed ones, against socceraction's fixed prior of 0.792 for the state before the kick. "
                "The split by outcome shows the mechanism: the shot's result is a one-hot feature of the state, so a missed penalty state is pulled towards the (low) prediction of failed open-play shots even though its label is the same 0.78.", ""]
    else:
        out += [f"Penalty-state predictions could not be recomputed: {ps['error']}.", ""]
    vs_b, vs_a = ba["vaep_sign"]["before"], ba["vaep_sign"]["after"]
    out += ["Share of rated appearances with a non-positive VAEP sum, before -> after: " + ", ".join(f"{g} {100 * vs_b[g]:.0f}% -> {100 * vs_a[g]:.0f}%" for g in POSITION_GROUPS) + ".", ""]
    return out


# ------------------------------------------------------------------------ 6. findings
BEFORE_AFTER_SKIPPED = [
    "the before -> after table in the headline numbers",
    "the before values in section 6 (ii), concerns A-H (printed as n/a)",
    "the short-substitute before/after analysis (6 (ii), H)",
    "7.1 the five notable matches, before and after",
    "7.2 top 20 single-match ratings, before and after",
    "7.3 the largest baseline-vs-VAEP disagreements, before",
    "7.4 agreement between the methods before, and each method's old vs new ratings",
    "7.5 split-half stability before",
    "7.6 team-context share, result dependence, event means and role means before",
    "7.7 ratings by minutes bucket before, and why the cameos moved",
    "7.8 what VAEP paid for shots and saves before",
]


def skipped_comparisons_note() -> list[str]:
    """The list of comparisons that need ``data/processed/before/`` (printed when it is absent)."""
    return ["`data/processed/before/` (the frozen pre-decision outputs) is not present, so these comparisons were skipped:", ""] + \
        [f"- {item}" for item in BEFORE_AFTER_SKIPPED] + [""]


def _v(x, spec: str = ".2f", pct: bool = False) -> str:
    """Format a number, or 'n/a' when it is missing (a before value without the before/ snapshot)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:{spec}}%" if pct else f"{x:{spec}}"


def _item(d: dict | None, key, i: int):
    """d[key][i], or None when d or the key is missing."""
    if not d or key not in d:
        return None
    return d[key][i]


def section_findings(rules: dict, cons: list, iso_rows: list, tests: dict | None, pos: dict, comp: dict, stab: dict, buckets: dict,
                     ba: dict | None, cur: dict, step4_frozen: bool = True) -> list[str]:
    """Section 6. (i) uses the current outputs; (ii) uses the state after the Step 4 decisions (``pos``, ``comp``, ``stab``
    and ``cur`` computed on it), with ``ba`` None without the before/ snapshot."""
    all_rules_pass = all(ok for m in METHODS for _, ok, _ in rules[m]["checks"]) and all(ok for _, ok, _ in cons) and all(r[4] == "yes" and r[7] == "yes" for r in iso_rows)
    failed_tests = tests["failed"] if tests else []
    tc = {g: s for g, s, *_ in comp["team_context"]}
    role = pos["role"].set_index(["position_group", "role"])
    ev = {e[0]: e for e in comp["events"]}
    gk_rho = [s for g, n, s, p in comp["by_group"] if g == "GK"][0]
    hb, hv = stab["halves"]["baseline"], stab["halves"]["vaep"]
    pv = comp["penalty_values"]

    out = ["## 6. Findings and recommendations", "", "### (i) Bugs or rule violations on the current outputs - must fix", ""]
    if all_rules_pass and not failed_tests:
        out += ["**None found.** All distribution rules hold for both methods on the regenerated files (1.1), rating one match alone through the public functions and the refit scalers reproduces the "
                "season files exactly (1.2), the VAEP splits are unchanged and leak-free, `raw_vaep` equals the summed action values, the web payloads equal the parquet files (1.3), and "
                + ("the whole test suite passes (1)." if tests else "the test suite was not run (`--skip-tests`).") + " No code outside `src/validation` was changed by the verifier.", ""]
    else:
        out += ["The following checks failed and need investigation:", ""]
        out += [f"- test: `{f}`" for f in failed_tests]
        out += [f"- {m}: {name} ({detail})" for m in METHODS for name, ok, detail in rules[m]["checks"] if not ok]
        out += [f"- consistency: {name} ({detail})" for name, ok, detail in cons if not ok]
        out += [f"- isolation: {r}" for r in iso_rows if not (r[4] == "yes" and r[7] == "yes")]
        out += [""]
    out += ["Still open from the first report (not code): the brief's 'Leicester City 3-1 Chelsea' was 2-1. Resolved on 2026-11-02: `test_each_position_group_is_anchored_separately` now uses "
            "`MEAN_TOLERANCE` (+/- 0.1, as CLAUDE.md says) instead of +/- 0.15, and the data passes.", ""]

    out += ["### (ii) What the decisions achieved against the original concerns A-H, and what they introduced", ""]
    if step4_frozen:
        out += ["The 'after' numbers here are the state right after the Step 4 decisions: the current baseline and the xG-label VAEP frozen in `data/processed/xg_labels/`. "
                "Option 3 later changed the VAEP side of B(2), C and the penalties; section 8 has what it changed.", ""]
    else:
        out += ["(No `xg_labels/` snapshot: the VAEP numbers below are computed on the current option-3 outputs, not on the xG-label state this section describes; "
                "section 8 was skipped.)", ""]
    have = ba is not None
    if not have:
        out += ["(No `before/` snapshot: every computed before value below is printed as n/a. The judgement prose is static text from the dated run, so the before values it "
                "quotes in words come from `reports/validation_before.md`.)", ""]
        out += skipped_comparisons_note()
    ag_b = ba["agree"]["before"] if have else {}
    st_b = {m: (ba["stability"]["before"]["halves"][m]["pearson"], ba["stability"]["before"]["halves"][m]["spearman"]) if have else (None, None) for m in METHODS}
    tcb = (ba["team_context"]["before"] if have else None) or {}
    eb = ba["events"]["before"] if have else {}
    shots_b = {r[0]: r for r in ba["shots"]["before"]} if have else {}
    vs_b = ba["vaep_sign"]["before"] if have else {}
    res_b = ba["result"]["before"] if have else {}
    roles_b = ba["roles"]["before"] if have else None
    sb = ba["subs"]["before"] if have else None
    sa = cur["subs"]
    ps = cur["penalty_states"]
    ag_a = {"all": comp["spearman_rating"], "season": comp["season"]["spearman"], **{g: s for g, n, s, p in comp["by_group"]}}
    shots_a = {r[0]: r for r in cur["shots"]}
    ea = {k: (v[1], v[2], v[3]) for k, v in ev.items()}
    ea.update({k: v for k, v in cur["events"].items() if k not in ea})  # adds the GK and 'missed a penalty' rows

    def group_stab(halves: dict | None, method: str, group: str):
        if halves is None:
            return None
        return [x for x in halves[method]["by_group"] if x[0] == group][0][2]

    halves_b = ba["stability"]["before"]["halves"] if have else None
    gk_stab_b, gk_stab_a = group_stab(halves_b, "vaep", "GK"), group_stab(stab["halves"], "vaep", "GK")
    def_stab = {m: (group_stab(halves_b, m, "DEF"), group_stab(stab["halves"], m, "DEF")) for m in METHODS}
    fwd_stab = {m: (group_stab(halves_b, m, "FWD"), group_stab(stab["halves"], m, "FWD")) for m in METHODS}
    gv = comp["gk_value_by_type"].set_index("type_name")
    keeper_save_after = _item(shots_a, "keeper save (all)", 4)
    keeper_save_before = _item(shots_b, "keeper save (all)", 4)

    def role_b(group: str, r: str):
        return None if roles_b is None else roles_b.loc[(group, r), "baseline"]

    out += [f"**A. VAEP goalkeepers (page shows the baseline).** Achieved: the page no longer shows the VAEP keeper rating, and the xG labels incidentally made the VAEP keeper score more meaningful: "
            f"a save is now worth {_v(keeper_save_after, '+.3f')} on average (before {_v(keeper_save_before, '+.3f')}); keeper saves contribute {gv.loc['keeper_save', 'sum']:.0f} of the GK total of {gv['sum'].sum():.0f} (before: carries and passes dominated, saves were negative); "
            f"Spearman(baseline, VAEP) for GK rose from {_v(ag_b.get('GK'))} to {gk_rho:.2f}; keepers with 3+ conceded average {ea['GK, 3+ conceded'][2]:.2f} under VAEP (baseline {ea['GK, 3+ conceded'][1]:.2f}). Not achieved: the VAEP keeper score still ignores goals conceded "
            f"(Spearman with goal difference {comp['rho_gd']['GK'][1]:+.2f}) - a keeper who faces many big chances and concedes two can rate 9.4 (Mignolet, 0-2 at West Ham) - and its split-half stability is {gk_stab_a:.2f} (before {_v(gk_stab_b)}). "
            "Verdict: the display rule is right and should stay; the VAEP keeper number remains unfit to show on its own.", ""]
    shot_goal_a = _item(shots_a, "open-play goal", 4)
    shot_miss_a = _item(shots_a, "open-play shot not scored", 4)
    out += [f"**B. VAEP shots by xG.** Achieved the stated aim: an open-play goal is now worth {_v(shot_goal_a, '+.3f')} to the shooter and a miss {_v(shot_miss_a, '+.3f')} (before +0.838 / -0.053; the builders' figures are confirmed "
            "to three decimals in 7.8), so VAEP no longer punishes a striker for shooting: the share of forward appearances with a non-positive VAEP sum fell from "
            f"{_v(vs_b.get('FWD'), '.0f', pct=True)} to {_v(cur['vaep_sign']['FWD'], '.0f', pct=True)}, Fernando's 1.29-xG-no-goal match went from 4.1 to 7.7, agreement with the baseline rose from "
            f"{_v(ag_b.get('all'))} to {ag_a['all']:.2f} (season averages {_v(ag_b.get('season'))} -> {ag_a['season']:.2f}), and the VAEP rating became more stable overall ({_v(st_b['vaep'][0])} -> {hv['pearson']:.2f} Pearson; "
            f"FWD {_v(fwd_stab['vaep'][0])} -> {_v(fwd_stab['vaep'][1])}, MID likewise up). **Introduced three problems.** (1) *Goals barely count*: scoring is worth about 0.10 more than missing, a five-goal Aguero is an 8.9 "
            f"(was 10.0) and outside the top 20, Huth's two headers v City are a 7.1 (was 9.9), one-goal scorers average {ea['scored exactly 1 goal'][2]:.2f} under VAEP (before {_v(_item(eb, 'scored exactly 1 goal (any kind)', 2))}; baseline {ea['scored exactly 1 goal'][1]:.2f}), "
            f"and {comp['low_scorers_vaep'].shape[0]} goal-scorers rate below 5.5 - a match rating that does not reward the goal is hard to present to fans. (2) *The outcome lottery moved to the defence*: because a state that contains a "
            "0.5-xG shot keeps P(concede) at 0.5 whether or not the shot went in, the defending team's first action within 10 seconds of a missed shot absorbs the whole drop. In SPADL that action is a keeper save or a "
            "clearance, never a tackle: single clearances straight after missed penalties worth +0.50 and +0.54 (Steve Cook 9.9, Kouyate 9.5) top the DEF list. A separate effect looks the same on the page: one tackle that ends "
            "a dangerous dribble can be worth +0.37 (van Aanholt 9.8 in a 0-2 home defeat). Nine of the VAEP top 20 have neither goal nor assist, and DEF split-half stability fell from "
            f"{_v(def_stab['vaep'][0])} to {_v(def_stab['vaep'][1])} (every other group rose). (3) *Penalties got worse*, see below. Verdict: the direction is right but the implementation over-corrects; see the options below.", ""]
    pen_a = ea["converted a penalty"]
    out += [f"**Penalties under xG labels (builders' flag i).** Confirmed: the kick is worth {pv.loc['success', 'mean']:+.3f} to a taker who scores and {pv.loc['fail', 'mean']:+.3f} to one who misses "
            f"(before +0.063 / -0.752). Mechanism, re-derived from the cached features and the fold models: the {ps.get('n', 91)} penalty states carry a label of {_v(ps.get('label_mean'), '.3f')} but are predicted at "
            f"{_v(ps.get('pred_mean'), '.3f')} on average - {_v(ps.get('pred_scored'), '.3f')} when scored and {_v(ps.get('pred_missed'), '.3f')} when missed - against the fixed 0.792 prior for the state before the kick. "
            "The outcome one-hot in the state features is doing the damage (a missed penalty state looks like a failed open-play shot to the trees; 91 states cannot overrule 8,500), not just the small sample. "
            f"Effect on ratings: penalty scorers average {pen_a[2]:.2f} under VAEP (before {_v(_item(eb, 'converted a penalty', 2))}; baseline {pen_a[1]:.2f}), i.e. a scored penalty now rates its taker *below* the 6.5 average, and misses average "
            f"{ea['missed a penalty'][2]:.2f}. **Not acceptable** as a published number (74 scored penalties a season, every one visibly wrong on the page). Options, cheapest first: (a) override the predicted P(scores) of a "
            "`shot_penalty` state with its label (the constant 0.7835 plus any rebound), which makes the kick worth about -0.01 whatever the outcome - a two-line change in `rate` before `formula.value`; "
            "(b) more generally use the shot's own xG as the post-shot state value for every shot type (the state after a shot is the chance that was taken), which also fixes problem B(2) at the source because "
            "the next defensive action then starts from the xG-state rather than a model guess; (c) drop the result one-hot from shot-type states or add a penalty indicator feature and retrain - slower and less certain.", ""]
    out += [f"**C. VAEP for defenders.** Partly achieved: own goals are now charged (own-goal scorers {ea['own goal'][2]:.2f} under VAEP, before {_v(_item(eb, 'own goal', 2))}), and a defender's goal no longer "
            f"dominates his group (one-goal defenders {ea['scored exactly 1 goal (DEF)'][2]:.2f}, before 8.52). Not achieved and partly worse: results and cards are still invisible (DEF Spearman with goal difference "
            f"{comp['rho_gd']['DEF'][1]:+.2f}; red cards {ea['red card'][2]:.2f}), full-backs still out-rate centre-backs ({role.loc[('DEF', 'FB'), 'vaep']:.2f} vs {role.loc[('DEF', 'CB'), 'vaep']:.2f}), and the clearance/tackle "
            f"credit lottery (B(2)) cut DEF stability to {_v(def_stab['vaep'][1])} and put defenders from beaten teams at the top of the list. Verdict: keep the README caveat; fix B(2) before trusting VAEP defender ratings.", ""]
    ovn = ba["old_vs_new"][0] if have else None
    out += [f"**D. Baseline team context halved.** Achieved: the team-context share of raw variance fell from {_v(tcb.get('DEF'), '.0f', pct=True)} to {100 * tc['DEF']:.0f}% for DEF and from {_v(tcb.get('MID'), '.0f', pct=True)} to {100 * tc['MID']:.0f}% for MID "
            f"(GK unchanged at {100 * tc['GK']:.0f}%, as intended); Spearman(rating, goal difference) for defenders {_v(res_b.get('rho_b_DEF'))} -> {cur['result']['rho_b_DEF']:.2f}; defenders through a 4+ goal "
            f"concession {_v(_item(eb, '4+ conceded, 60+ min (DEF)', 1))} -> {ea['4+ conceded, 60+ min (DEF)'][1]:.2f}, clean-sheet defenders {_v(_item(eb, 'clean sheet (DEF)', 1))} -> {ea['clean sheet (DEF)'][1]:.2f}; "
            f"split-half stability of the baseline rose ({_v(st_b['baseline'][0])} -> {hb['pearson']:.2f}; DEF {_v(def_stab['baseline'][0])} -> {_v(def_stab['baseline'][1])}) and agreement with VAEP for DEF rose "
            f"({_v(ag_b.get('DEF'))} -> {ag_a['DEF']:.2f}). Baseline ratings moved little (Spearman old vs new {_v(ovn[2] if ovn else None, '.3f')}; mean |change| {_v(ovn[7] if ovn else None)}). Cost: busy defenders in heavy defeats rate higher "
            "(Demichelis 7.6 in the 1-3), and the knock-on effect on short substitutes below. Verdict: a clear improvement; keep.", ""]
    gap_b = None if roles_b is None else role_b("FWD", "W") - role_b("FWD", "CF")
    out += [f"**E. FWD weights eased, roles pooled.** Achieved in part: centre-forwards rose from {_v(role_b('FWD', 'CF'))} to {role.loc[('FWD', 'CF'), 'baseline']:.2f} against wingers "
            f"{_v(role_b('FWD', 'W'))} -> {role.loc[('FWD', 'W'), 'baseline']:.2f} (gap {_v(gap_b)} -> "
            f"{role.loc[('FWD', 'W'), 'baseline'] - role.loc[('FWD', 'CF'), 'baseline']:.2f}). The MID spread is unchanged (attacking midfielders {role.loc[('MID', 'AM'), 'baseline']:.2f} vs central {role.loc[('MID', 'CM'), 'baseline']:.2f}) "
            "because nothing was changed there, and the 12/16 reclassification was skipped by decision. Verdict: adequate; the remaining spreads are within 0.15 and can wait.", ""]
    out += [f"**F. Penalties in the baseline.** Achieved: penalty scorers average {pen_a[1]:.2f} (before {_v(_item(eb, 'converted a penalty', 1))}), one-goal scorers {ea['scored exactly 1 goal'][1]:.2f}; `key_passes` no longer double-counts assists. "
            "Verdict: done. (The VAEP side of penalties is the open problem above.)", ""]
    out += ["**G. Baseline keeper save volume** - skipped by decision; unchanged (Forster's eleven saves remain the top GK rating; GK baseline split-half still about 0.10 with 19 keepers). Acceptable to leave.", ""]
    bucket_a = buckets["baseline"]["bucket"]
    if sb is not None:
        bucket_b = ba["buckets"]["before"]["baseline"]["bucket"]
        off = ", ".join(f"{g} {sa[f'p30_{g}'] - sa[f'cameo_median_raw_{g}']:+.2f}" for g in ("DEF", "MID", "FWD"))
        out += [f"**H / short substitutes (builders' flag ii).** Confirmed: 10-20 minute baseline cameos fell from {sb['mean']:.2f} to {sa['mean']:.2f}, with {100 * sa['below6']:.0f}% now below 6.0 (before {100 * sb['below6']:.0f}%); "
                f"20-45 minute appearances fell from {bucket_b.loc['20-45', 'mean']:.2f} to {bucket_a.loc['20-45', 'mean']:.2f}. The cameos themselves did not change "
                f"(median raw {sb['cameo_median_raw_MID']:.2f} -> {sa['cameo_median_raw_MID']:.2f} for MID): the starters around them did. Halving the team context lifted the bottom of every group - the 25th percentile of DEF raw scores "
                f"moved from {sb['p25_DEF']:.2f} to {sa['p25_DEF']:.2f}, the share of 60+ minute defenders with a negative raw score from {100 * sb['share_neg_starters_DEF']:.0f}% to {100 * sa['share_neg_starters_DEF']:.0f}% - so a quiet cameo "
                f"that used to sit at the {100 * sb['cameo_pct_DEF']:.0f}th percentile of DEF now sits at the {100 * sa['cameo_pct_DEF']:.0f}th (MID {100 * sb['cameo_pct_MID']:.0f}th -> {100 * sa['cameo_pct_MID']:.0f}th, FWD {100 * sb['cameo_pct_FWD']:.0f}th -> {100 * sa['cameo_pct_FWD']:.0f}th). "
                "Is it acceptable? The size is small (-0.16 on the mean) but the visible effect is not: two thirds of cameos now show red on the page, which reads as 'substitutes play badly' rather than 'substitutes did little'. "
                f"Options: (a) a minutes-aware offset added to the raw score inside `rate_match` (a function of the row's own minutes, so the isolation rule holds), e.g. `offset = c_group * max(0, 60 - minutes) / 60`; to return the median "
                f"cameo to about the 30th percentile of its group the full offset at 0 minutes would need to be roughly {off} raw points, i.e. about one quarter of a quiet starter's score - document it in `weights.py` as the "
                "'low activity' rule; (b) re-tune the per-action negatives that make up most of a cameo's score (`passes_incomplete` -0.05/-0.06 and `possession_lost` -0.25/-0.15: a cameo averages 1.4 incomplete passes and 0.8 possessions "
                "lost), which would move the median cameo by only about 0.1 raw and the percentile by 2-3 points - not enough on its own; (c) accept and relabel the colour bands for substitutes. (a) is the one that addresses the cause.", ""]
    else:
        out += [f"**H / short substitutes (builders' flag ii).** Before/after comparison skipped (no `before/` snapshot). Current outputs: 10-20 minute baseline cameos average {sa['mean']:.2f}, "
                f"{100 * sa['below6']:.0f}% of them below 6.0; 20-45 minute appearances average {bucket_a.loc['20-45', 'mean']:.2f}; the median 10-20 minute cameo sits at the "
                f"{100 * sa['cameo_pct_DEF']:.0f}th percentile of DEF raw scores (MID {100 * sa['cameo_pct_MID']:.0f}th, FWD {100 * sa['cameo_pct_FWD']:.0f}th).", ""]
    out += ["### (iii) Things that look right after the changes", "",
            f"- Every distribution rule, for both methods, on the regenerated files: means 6.500 overall and per group, {rules['baseline']['tens']} and {rules['vaep']['tens']} tens, min 3.4, max 10.0, one decimal, 10-minute rule, "
            "all 380 matches, 11 rated starters except the 6 early injuries; isolation exact; splits identical to the frozen ones and leak-free; web payloads equal the parquet files.",
            f"- The baseline after D/E/F: stability up ({_v(st_b['baseline'][0])} -> {hb['pearson']:.2f}), agreement with VAEP up ({_v(ag_b.get('all'))} -> {ag_a['all']:.2f} overall), team context no longer dominates defenders, "
            f"penalties no longer count as full goals, red cards ({ea['red card'][1]:.2f}), own goals ({ea['own goal'][1]:.2f}) and errors ({ea['error leading to a shot'][1]:.2f}) are still penalised, and the top 20 is still a list of great performances.",
            "- The xG-label VAEP for creative midfielders and forwards: Cazorla, Ramsey, Payet, Eriksen and Mahrez now lead the single-match list on chances created, Fernando-type misses are no longer punished, MID and FWD stability rose sharply.",
            "- The goalkeeper display rule on the page and the README caveat.", ""]
    return out


# --------------------------------------------------------------------------------- main
def build_report(skip_tests: bool = False) -> str:
    t0 = time.time()
    log.info("loading processed data")
    data = load()
    tests = None if skip_tests else checks.run_test_suite()
    if tests:
        log.info("tests: %s", tests["summary"])
    rules = checks.distribution_rules(data)
    cons = checks.consistency_checks(data)
    iso = checks.isolation_check(data, ISOLATION_MATCHES, ISOLATION_OTHER)
    log.info("rules, consistency and isolation done (%.0fs)", time.time() - t0)
    pos = checks.position_group_analysis(data)
    log.info("position groups done (%.0fs)", time.time() - t0)
    tops = {m: analysis.top20(data, m) for m in METHODS}
    matches = [analysis.match_table(data, mid) for mid in NOTABLE_MATCHES]
    buckets = analysis.minutes_buckets(data)
    comp = analysis.method_comparison(data)
    stab = analysis.stability(data)
    log.info("analysis done (%.0fs)", time.time() - t0)
    # Sections 6 (ii) and 7 are the history of the Step 4 decisions: their 'after' is the current baseline with the
    # xG-label VAEP frozen in data/processed/xg_labels/ (the current VAEP adds option 3, which section 8 compares).
    step4 = load_xg_labels(data)
    hist = step4 if step4 is not None else data
    if step4 is not None:
        comp_h, stab_h, pos_h = analysis.method_comparison(step4), analysis.stability(step4), {"role": checks.role_table(step4)}
    else:
        comp_h, stab_h, pos_h = comp, stab, pos
    cur = before_after.compute_current(hist)
    ba = before_after.compute(hist, current=cur)
    o3 = option3.compute(data, step4)
    log.info("before/after done (%.0fs)%s%s", time.time() - t0, "" if ba is not None else "; no before/ snapshot, comparisons skipped",
             "" if o3 is not None else "; no xg_labels/ snapshot, option-3 comparison skipped")

    lines = [f"# Validation report: Premier League 2015/16 player ratings (Step 4, re-validated after the weight and label decisions and option 3)", "",
             f"Generated by `python -m src.validation.validate` on {dt.date.today().isoformat()} from `data/processed` (StatsBomb Open Data, Premier League 2015/16, 380 matches; "
             "https://github.com/statsbomb/open-data). Written by an independent verifier who did not build the rating code; the builders' reports in `reports/` were treated as claims and re-checked. "
             "Sections 1-5 and 6 (i) describe the CURRENT outputs, whose VAEP uses the option-3 scoring rules (decision 2026-11-02). Section 6 (ii) and section 7 are the history of the Step 4 decisions: "
             "they compare the frozen pre-decision outputs in `data/processed/before/` with the state right after those decisions, i.e. the current baseline and the xG-label VAEP frozen in "
             "`data/processed/xg_labels/`. Section 8 compares that xG-label VAEP with option 3. The pre-decision report is `reports/validation_before.md`. "
             "All tables are recomputed on every run; paragraphs marked as the verifier's judgement describe the run they are dated with.", "",
             "## Headline numbers", ""]
    hb, hv = stab["halves"]["baseline"], stab["halves"]["vaep"]
    lines += [(f"- Test suite: `{tests['summary']}`; " + (f"**{len(tests['failed'])} failures** (section 1)." if tests["failed"] else "no failures.")) if tests else "- Test suite: skipped.",
              f"- Baseline: {rules['baseline']['n']} rated rows, mean {rules['baseline']['mean']:.3f}, {rules['baseline']['tens']} tens ({100 * rules['baseline']['tens'] / rules['baseline']['n']:.3f}%), min {rules['baseline']['groups'][-1][5]}, max {rules['baseline']['groups'][-1][6]}.",
              f"- VAEP: {rules['vaep']['n']} rated rows, mean {rules['vaep']['mean']:.3f}, {rules['vaep']['tens']} tens ({100 * rules['vaep']['tens'] / rules['vaep']['n']:.3f}%), min {rules['vaep']['groups'][-1][5]}, max {rules['vaep']['groups'][-1][6]}.",
              f"- Spearman between the methods: {comp['spearman_rating']:.3f} per player-match (GK {[s for g, n, s, p in comp['by_group'] if g == 'GK'][0]:.3f}, DEF {[s for g, n, s, p in comp['by_group'] if g == 'DEF'][0]:.3f}, "
              f"MID {[s for g, n, s, p in comp['by_group'] if g == 'MID'][0]:.3f}, FWD {[s for g, n, s, p in comp['by_group'] if g == 'FWD'][0]:.3f}); {comp['season']['spearman']:.3f} on season averages ({comp['season']['n_players']} players).",
              f"- Split-half stability (Pearson / Spearman, {hb['n']} players): baseline {hb['pearson']:.3f} / {hb['spearman']:.3f}, VAEP {hv['pearson']:.3f} / {hv['spearman']:.3f}.",
              "- Rule violations or bugs found: " + ("none." if all(ok for m in METHODS for _, ok, _ in rules[m]["checks"]) and all(ok for _, ok, _ in cons) and not (tests and tests["failed"]) else "**see section 6 (i)**.")]
    if ba is not None:
        agb, aga = ba["agree"]["before"], ba["agree"]["after"]
        sb_, sa_ = ba["stability"]["before"]["halves"], ba["stability"]["after"]["halves"]
        tcb, tca = ba["team_context"]["before"], ba["team_context"]["after"]
        lines += ["", "Before -> after the Step 4 decisions (details in section 7):", "",
                  md_table(["quantity", "before", "after"], [
                      ["Spearman(baseline, VAEP), all player-matches", fmt(agb["all"]), fmt(aga["all"])],
                      ["... GK / DEF / MID / FWD", " / ".join(fmt(agb[g], 2) for g in POSITION_GROUPS), " / ".join(fmt(aga[g], 2) for g in POSITION_GROUPS)],
                      ["... season averages", fmt(agb["season"]), fmt(aga["season"])],
                      ["split-half Pearson, baseline / VAEP", f"{sb_['baseline']['pearson']:.3f} / {sb_['vaep']['pearson']:.3f}", f"{sa_['baseline']['pearson']:.3f} / {sa_['vaep']['pearson']:.3f}"],
                      ["team-context share of baseline raw variance, DEF / MID / GK", " / ".join(f"{100 * tcb[g]:.0f}%" for g in ("DEF", "MID", "GK")) if tcb else "n/a", " / ".join(f"{100 * tca[g]:.0f}%" for g in ("DEF", "MID", "GK"))],
                      ["VAEP value of a converted / missed penalty to the taker", f"{ba['shots']['before'][2][4]:+.3f} / {ba['shots']['before'][3][4]:+.3f}", f"{ba['shots']['after'][2][4]:+.3f} / {ba['shots']['after'][3][4]:+.3f}"],
                      ["VAEP value of an open-play goal / miss / keeper save", f"{ba['shots']['before'][0][4]:+.3f} / {ba['shots']['before'][1][4]:+.3f} / {ba['shots']['before'][6][4]:+.3f}", f"{ba['shots']['after'][0][4]:+.3f} / {ba['shots']['after'][1][4]:+.3f} / {ba['shots']['after'][6][4]:+.3f}"],
                      ["baseline 10-20 minute cameos: mean / share below 6.0", f"{ba['subs']['before']['mean']:.2f} / {100 * ba['subs']['before']['below6']:.0f}%", f"{ba['subs']['after']['mean']:.2f} / {100 * ba['subs']['after']['below6']:.0f}%"],
                      ["penalty scorers' mean rating, baseline / VAEP", f"{ba['events']['before']['converted a penalty'][1]:.2f} / {ba['events']['before']['converted a penalty'][2]:.2f}", f"{ba['events']['after']['converted a penalty'][1]:.2f} / {ba['events']['after']['converted a penalty'][2]:.2f}"],
                  ]), ""]
    else:
        lines += ["- Before -> after the Step 4 decisions: **skipped**, `data/processed/before/` is not present (section 7 lists every comparison that was skipped).", ""]
    if o3 is not None:
        lines += ["xG-label VAEP -> option 3 (details in section 8):", "", md_table(["quantity", "xG labels", "option 3"], option3.summary_rows(o3)), ""]
    else:
        lines += ["- xG-label VAEP -> option 3: **skipped**, `data/processed/xg_labels/` is not present (section 8).", ""]
    lines += section_tests(tests)
    lines += section_rules(rules, iso, cons)
    lines += section_positions(pos)
    lines += section_eye(data, tops, matches, buckets)
    lines += section_compare(comp)
    lines += section_stability(stab)
    lines += section_findings(rules, cons, iso, tests, pos_h, comp_h, stab_h, buckets, ba, cur, step4_frozen=step4 is not None)
    lines += section_before_after(ba, hist)
    lines += option3.section(o3)
    lines += ["---", "", f"Report generated in {time.time() - t0:.0f} s. Data: StatsBomb Open Data (https://github.com/statsbomb/open-data).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate reports/validation.md (read-only on data/processed).")
    parser.add_argument("--skip-tests", action="store_true", help="do not run the pytest suite")
    parser.add_argument("--out", default=str(REPORT_PATH), help="output path (default reports/validation.md)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = build_report(skip_tests=args.skip_tests)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)
    print(f"wrote {args.out} ({len(report.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
