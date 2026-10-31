"""Task 1 (test suite, distribution rules, isolation, file consistency) and task 2
(position-group handling) of the verification brief. Everything here is read-only."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.contract import CONTRACT_COLUMNS, MAX_SHARE_OF_TENS, MEAN_TOLERANCE, MIN_MINUTES, POSITION_GROUPS, RATING_DECIMALS, RATING_MAX, RATING_MIN, TARGET_MEAN
from src.ingest.positions import POSITION_GROUP_BY_ID, POSITION_NAMES
from src.validation.data import METHODS, PROCESSED, RAW, ROOT, WEB_DATA, Data

EXPECTED_FAILING_FILES = ("tests/test_explain.py", "tests/test_pipeline.py")  # steps not built yet
ROLE_OF_POSITION = {
    1: "GK", 2: "FB", 6: "FB", 7: "WB", 8: "WB", 3: "CB", 4: "CB", 5: "CB",
    9: "DM", 10: "DM", 11: "DM", 12: "WM", 16: "WM", 13: "CM", 14: "CM", 15: "CM",
    18: "AM", 19: "AM", 20: "AM", 17: "W", 21: "W", 22: "CF", 23: "CF", 24: "CF", 25: "SS",
}
ROLE_LABELS = {"GK": "goalkeeper", "FB": "full-back", "WB": "wing-back", "CB": "centre-back", "DM": "defensive midfielder",
               "CM": "central midfielder", "WM": "wide midfielder (RM/LM)", "AM": "attacking midfielder",
               "W": "winger (RW/LW)", "CF": "centre-forward", "SS": "second striker"}


# ----------------------------------------------------------------------------- 1. tests
def run_test_suite(timeout: int = 1800) -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning"]
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    summary = ""
    for line in reversed(out.splitlines()):
        if re.search(r"\d+ (passed|failed)", line):
            summary = line.strip("= ").strip()
            break
    failed = sorted({m.group(1) for m in re.finditer(r"^(?:FAILED|ERROR) (tests/[^\s:]+::[^\s]+)", out, flags=re.M)})
    expected = [f for f in failed if f.split("::")[0] in EXPECTED_FAILING_FILES]
    unexpected = [f for f in failed if f.split("::")[0] not in EXPECTED_FAILING_FILES]
    return {"command": " ".join(cmd[1:]), "summary": summary, "returncode": proc.returncode, "seconds": round(time.time() - started, 1),
            "failed": failed, "expected_failures": expected, "unexpected_failures": unexpected}


# ------------------------------------------------------------------ 1. distribution rules
def distribution_rules(data: Data) -> dict:
    """Re-derive every PROMPT.md rule from ratings_<method>.parquet (not from the tests)."""
    out = {}
    lineups = data.lineups
    eligible = set(map(tuple, lineups.loc[lineups["minutes"] >= MIN_MINUTES, ["match_id", "player_id"]].to_numpy()))
    starters = lineups[lineups["is_starter"]][["match_id", "team", "player_id", "minutes", "red_card_minute", "player_name"]]
    short_starters = starters[starters["minutes"] < MIN_MINUTES]
    for method in METHODS:
        r = data.ratings[method]
        raw = data.raw[method]
        checks = []
        mean = float(r["rating"].mean())
        checks.append(("season mean 6.5 +/- 0.1", abs(mean - TARGET_MEAN) <= MEAN_TOLERANCE, f"{mean:.4f}"))
        gm = r.groupby("position_group")["rating"].mean()
        checks.append(("every position group mean 6.5 +/- 0.1", bool((gm - TARGET_MEAN).abs().max() <= MEAN_TOLERANCE),
                       ", ".join(f"{g} {gm[g]:.4f}" for g in POSITION_GROUPS)))
        tens = int((r["rating"] == RATING_MAX).sum())
        checks.append(("fewer than 0.1% of ratings are 10.0", tens / len(r) < MAX_SHARE_OF_TENS, f"{tens} of {len(r)} = {100 * tens / len(r):.3f}%"))
        checks.append(("bounded 3.0-10.0", bool(r["rating"].min() >= RATING_MIN and r["rating"].max() <= RATING_MAX), f"min {r['rating'].min():.1f}, max {r['rating'].max():.1f}"))
        scaled = r["rating"].to_numpy() * 10 ** RATING_DECIMALS
        checks.append(("one decimal place", bool(np.allclose(scaled, np.round(scaled), atol=1e-9)), ""))
        checks.append(("only 10+ minute appearances rated", bool((r["minutes"] >= MIN_MINUTES).all()), f"min minutes rated {r['minutes'].min():.1f}"))
        got = set(map(tuple, r[["match_id", "player_id"]].to_numpy()))
        checks.append(("every 10+ minute appearance rated (set equality with lineups)", got == eligible, f"{len(got)} rated, {len(eligible)} eligible"))
        checks.append(("all 380 matches rated", set(r["match_id"]) == set(data.matches["match_id"]), f"{r['match_id'].nunique()} matches"))
        exp = starters[starters["minutes"] >= MIN_MINUTES].groupby(["match_id", "team"]).size()
        rated_starters = r.merge(starters[["match_id", "player_id"]], on=["match_id", "player_id"]).groupby(["match_id", "team"]).size()
        table = pd.concat([exp.rename("expected"), rated_starters.rename("got")], axis=1).fillna(0).astype(int)
        checks.append(("11 rated starters per team-match unless a starter had < 10 minutes",
                       bool(len(table) == 760 and (table["expected"] == table["got"]).all() and table["got"].max() == 11),
                       f"{int((table['got'] == 11).sum())} of {len(table)} team-matches have 11; {int((table['got'] < 11).sum())} have fewer, "
                       f"all explained by {len(short_starters)} starters under 10 minutes"))
        mono = all((np.diff(part.sort_values("raw_score", kind="mergesort")["rating"].to_numpy()) >= 0).all() for _, part in r.groupby("position_group"))
        checks.append(("rating monotone in raw_score within each group", mono, ""))
        m = r.merge(lineups[["match_id", "player_id", "position_group", "minutes", "team"]], on=["match_id", "player_id"], suffixes=("", "_lu"))
        checks.append(("position_group, minutes and team equal lineups.parquet", bool((m["position_group"] == m["position_group_lu"]).all() and (m["minutes"] == m["minutes_lu"]).all() and (m["team"] == m["team_lu"]).all()), ""))
        m2 = r.merge(raw[["match_id", "player_id", "raw_score"]], on=["match_id", "player_id"], suffixes=("", "_raw"))
        checks.append(("raw_score equals raw_<method>.parquet", bool((m2["raw_score"] - m2["raw_score_raw"]).abs().max() == 0), ""))
        checks.append(("raw file has one row per appearance (= lineups)", set(map(tuple, raw[["match_id", "player_id"]].to_numpy())) == set(map(tuple, lineups[["match_id", "player_id"]].to_numpy())), f"{len(raw)} rows"))
        scaler = json.loads((PROCESSED / f"scaler_{method}.json").read_text(encoding="utf-8"))
        sizes = {g: v["n"] for g, v in scaler["groups"].items()}
        checks.append(("saved scaler reference sizes equal rated rows per group", sizes == r.groupby("position_group").size().to_dict(), str(sizes)))
        group_rows = []
        for g in POSITION_GROUPS + ("all",):
            part = r if g == "all" else r[r["position_group"] == g]
            group_rows.append([g, len(part), round(float(part["rating"].mean()), 4), round(float(part["rating"].std()), 3), int((part["rating"] == RATING_MAX).sum()),
                               f"{part['rating'].min():.1f}", f"{part['rating'].max():.1f}", int((part["rating"] >= 9).sum()), int((part["rating"] >= 8).sum()),
                               int((part["rating"] < 5).sum()), int(part["raw_score"].nunique()), int((part["raw_score"] <= 0).sum())])
        out[method] = {"checks": checks, "groups": group_rows, "n": len(r), "mean": mean, "tens": tens,
                       "short_starters": short_starters.assign(reason=np.where(short_starters["red_card_minute"].notna(), "red card", "injury/early substitution"))}
    return out


# ------------------------------------------------------------------------- 1. consistency
def consistency_checks(data: Data) -> list[tuple[str, bool, str]]:
    checks = []
    sp = data.splits
    weeks = data.matches.set_index("match_id")["match_week"]
    all_ids = sorted(int(m) for m in data.matches["match_id"])
    scored = sorted(x for f in sp["folds"] for x in f["score"])
    checks.append(("VAEP folds: every match scored in exactly one fold", scored == all_ids, f"{len(sp['folds'])} folds"))
    checks.append(("VAEP folds: no match in its own training set; train + score = 380",
                   all(not set(f["train"]) & set(f["score"]) and len(f["train"]) + len(f["score"]) == 380 for f in sp["folds"]), ""))
    checks.append(("VAEP holdout: trains on weeks <= 19, scores weeks >= 20",
                   int(weeks.loc[sp["holdout"]["train"]].max()) == 19 and int(weeks.loc[sp["holdout"]["score"]].min()) == 20,
                   f"{len(sp['holdout']['train'])} train / {len(sp['holdout']['score'])} score"))
    av = data.actions_values
    fold_of = {g: f["fold_id"] for f in sp["folds"] for g in f["score"]}
    checks.append(("action_values.fold_id agrees with vaep_splits.json", bool((av["game_id"].map(fold_of) == av["fold_id"]).all()), f"{len(av):,} actions"))
    sums = av.groupby(["game_id", "player_id"])["vaep_value"].sum()
    keyed = data.raw["vaep"].set_index(["match_id", "player_id"])["raw_score"]
    diff = float((keyed - sums.reindex(keyed.index).fillna(0.0)).abs().max())
    checks.append(("raw_vaep = sum of the player's action values", diff < 1e-9, f"max |diff| {diff:.1e}; (game, player) pairs with actions but no lineup row: {len(sums.index.difference(keyed.index))}"))
    hold = pd.read_parquet(PROCESSED / "raw_vaep_holdout.parquet")
    checks.append(("raw_vaep_holdout covers exactly the holdout score matches", set(hold["match_id"]) == set(sp["holdout"]["score"]), f"{len(hold)} rows"))
    # web export: every exported rating equals the parquet rating; position_group and minutes equal the lineups
    index_path = WEB_DATA / "index.json"
    if index_path.exists():
        entries = json.loads(index_path.read_text(encoding="utf-8"))
        lu = data.lineups.set_index(["match_id", "player_id"])
        rated = {m: data.ratings[m].set_index(["match_id", "player_id"])["rating"] for m in METHODS}
        n = bad = 0
        for e in entries:
            payload = json.loads((WEB_DATA / e["path"]).read_text(encoding="utf-8"))
            mid = payload["match_id"]
            for p in payload["players"]:
                n += 1
                key = (mid, p["player_id"])
                ok = p["position_group"] == lu.loc[key, "position_group"] and abs(p["minutes"] - lu.loc[key, "minutes"]) < 1e-9
                for m in METHODS:
                    if p["minutes"] >= MIN_MINUTES:
                        ok = ok and p["ratings"][m] == float(rated[m].loc[key])
                    else:
                        ok = ok and p["ratings"][m] is None
                bad += int(not ok)
        checks.append(("web/data payloads: rating null iff < 10 minutes, ratings and position_group equal the parquet files", bad == 0, f"{len(entries)} matches, {n} player rows, {bad} mismatches"))
    else:
        checks.append(("web/data payloads", False, "web/data/index.json not found (run python -m src.pipeline.export_web)"))
    return checks


# ---------------------------------------------------------------------------- 1. isolation
def isolation_check(data: Data, match_ids: list[int], other_match: int) -> list[list]:
    """Rate each match alone through the public rate_match functions and the saved scaler
    and compare with the cached season files; also mix in another match's rows."""
    from src.baseline.rate import rate_match as base_rate
    from src.baseline.stats import EVENT_COLUMNS
    from src.scale.scaler import PercentileScaler
    from src.vaep.rate import load_model_for_match, rate_match as vaep_rate

    rows = []
    lu = data.lineups
    key = ["team", "player_id"]
    for method in METHODS:
        scaler = PercentileScaler.load(PROCESSED / f"scaler_{method}.json")
        raw_cache, rat_cache = data.raw[method], data.ratings[method]
        for mid in match_ids:
            pair = [mid, other_match]
            if method == "baseline":
                ev = pd.read_parquet(PROCESSED / "events.parquet", columns=EVENT_COLUMNS, filters=[("match_id", "in", pair)])
                alone = base_rate(mid, ev[ev["match_id"] == mid], lu[lu["match_id"] == mid])
                mixed = base_rate(mid, ev, lu[lu["match_id"].isin(pair)])
                absent = lambda: base_rate(mid, ev[ev["match_id"] == other_match], lu[lu["match_id"] == other_match])  # noqa: E731
            else:
                ac = pd.read_parquet(PROCESSED / "actions.parquet", filters=[("game_id", "in", pair)])
                model = load_model_for_match(mid)
                alone = vaep_rate(mid, ac[ac["game_id"] == mid], lu[lu["match_id"] == mid], model)
                mixed = vaep_rate(mid, ac, lu[lu["match_id"].isin(pair)], model)
                absent = lambda: vaep_rate(mid, ac[ac["game_id"] == other_match], lu[lu["match_id"] == other_match], model)  # noqa: E731
            alone = alone[CONTRACT_COLUMNS].sort_values(key).reset_index(drop=True)
            mixed = mixed[CONTRACT_COLUMNS].sort_values(key).reset_index(drop=True)
            cached = raw_cache[raw_cache["match_id"] == mid][CONTRACT_COLUMNS].sort_values(key).reset_index(drop=True)
            try:
                assert_frame_equal(alone, mixed)
                same_mixed = True
            except AssertionError:
                same_mixed = False
            raw_diff = float((alone["raw_score"].to_numpy() - cached["raw_score"].to_numpy()).__abs__().max()) if len(alone) == len(cached) else float("nan")
            rated = scaler.transform(alone).sort_values(key).reset_index(drop=True)
            cr = rat_cache[rat_cache["match_id"] == mid].sort_values(key).reset_index(drop=True)
            same_keys = len(rated) == len(cr) and bool((rated["player_id"].to_numpy() == cr["player_id"].to_numpy()).all())
            rating_diff = float(np.abs(rated["rating"].to_numpy() - cr["rating"].to_numpy()).max()) if same_keys else float("nan")
            try:
                absent()
                raises = False
            except ValueError:
                raises = True
            rows.append([method, mid, len(alone), f"{raw_diff:.1e}", "yes" if same_mixed else "NO", len(rated), f"{rating_diff:.1e}" if same_keys else "row mismatch", "yes" if raises else "NO"])
    return rows


# ----------------------------------------------------------------- 2. position groups
def position_group_analysis(data: Data) -> dict:
    out: dict = {}
    lu = data.lineups.copy()
    out["ids_seen"] = sorted(int(x) for x in lu["position_id"].unique())
    out["mapping_consistent"] = bool((lu["position_id"].map(POSITION_GROUP_BY_ID) == lu["position_group"]).all())
    lu["start_group"] = lu["starting_position_id"].map(POSITION_GROUP_BY_ID)
    starters = lu[lu["is_starter"]]
    moved = starters[starters["start_group"] != starters["position_group"]]
    out["n_starters"] = len(starters)
    out["moved"] = moved
    out["moved_crosstab"] = pd.crosstab(moved["start_group"], moved["position_group"])
    gk = lu[lu["position_group"] == "GK"]
    out["gk_subs"] = gk[~gk["is_starter"]]
    out["gk_rows"] = len(gk)
    out["gk_not_position_1"] = int((gk["position_id"] != 1).sum())
    per_team_gk = gk.groupby(["match_id", "team"]).size()
    out["team_matches_two_gk"] = int((per_team_gk >= 2).sum())
    out["team_matches_no_gk"] = 2 * len(data.matches) - len(per_team_gk)
    groups_per_player = lu.groupby("player_id")["position_group"].agg(lambda s: tuple(sorted(set(s))))
    out["n_players"] = int(lu["player_id"].nunique())
    out["players_multi"] = int((groups_per_player.map(len) >= 2).sum())
    out["players_three"] = int((groups_per_player.map(len) >= 3).sum())
    out["players_gk_and_outfield"] = int(groups_per_player.map(lambda g: "GK" in g and len(g) > 1).sum())
    # events: who did goalkeeper things without being in the GK group (outfield players in goal)
    ev = pd.read_parquet(PROCESSED / "events.parquet", columns=["match_id", "player_id", "player", "team", "position_id", "type", "minute"])
    gk_events = ev[((ev["position_id"] == 1) | (ev["type"] == "Goal Keeper")) & ev["player_id"].notna()]
    m = gk_events.merge(lu[["match_id", "player_id", "position_group"]], on=["match_id", "player_id"], how="left")
    odd = m[m["position_group"] != "GK"]
    out["outfield_in_goal"] = odd.groupby(["match_id", "player", "team", "position_group"]).agg(events=("type", "size"), first_minute=("minute", "min"), last_minute=("minute", "max")).reset_index()
    # share of a player's events outside his assigned group
    pe = ev[ev["player_id"].notna()].copy()
    pe["player_id"] = pe["player_id"].astype("int64")
    pe["ev_group"] = pe["position_id"].map(POSITION_GROUP_BY_ID)
    pe = pe.merge(lu[["match_id", "player_id", "position_group"]], on=["match_id", "player_id"], how="inner")
    pe["other"] = (pe["ev_group"] != pe["position_group"]).astype(int)
    share = pe.groupby(["match_id", "player_id"]).agg(n=("other", "size"), other=("other", "sum"))
    share["share_other"] = share["other"] / share["n"]
    out["events_other_group"] = {"player_matches": len(share), "over_30": int((share["share_other"] >= 0.3).sum()), "over_50": int((share["share_other"] > 0.5).sum())}
    # lineup intervals from the raw JSON: minutes per group when a player's intervals span 2+ groups
    out["intervals"] = None
    if RAW.exists():
        from src.ingest.lineups import _overlap, absolute_seconds, on_pitch_spans, period_lengths
        rows = []
        for mid in data.matches["match_id"]:
            ev_json = json.loads((RAW / "events" / f"{mid}.json").read_text(encoding="utf-8"))
            lu_json = json.loads((RAW / "lineups" / f"{mid}.json").read_text(encoding="utf-8"))
            lengths = period_lengths(ev_json)
            spans, _ = on_pitch_spans(ev_json, lengths)
            total = sum(lengths.values())
            for team in lu_json:
                for p in team["lineup"]:
                    sp = spans.get(int(p["player_id"]), [])
                    if not sp:
                        continue
                    by_group: dict[str, float] = {}
                    for q in p.get("positions") or []:
                        s = absolute_seconds(q["from"], q["from_period"], lengths)
                        e = total if q["to"] is None else absolute_seconds(q["to"], q["to_period"], lengths)
                        g = POSITION_GROUP_BY_ID[int(q["position_id"])]
                        by_group[g] = by_group.get(g, 0.0) + sum(_overlap((s, e), x) for x in sp)
                    secs = sum(e - s for s, e in sp)
                    if len(by_group) >= 2 and secs > 0:
                        main = max(by_group, key=by_group.get)
                        rows.append([int(mid), team["team_name"], p.get("player_nickname") or p["player_name"], main, by_group[main] / secs, secs / 60,
                                     ", ".join(f"{k} {v / 60:.0f}'" for k, v in sorted(by_group.items(), key=lambda kv: -kv[1]))])
        out["intervals"] = pd.DataFrame(rows, columns=["match_id", "team", "player", "main", "share_main", "minutes", "minutes_by_group"])
    # multi-group players: paired comparison of their mean rating in the main group vs the other group
    both = data.both
    multi_ids = groups_per_player[groups_per_player.map(len) >= 2].index
    paired = {}
    for method, col in (("baseline", "rating_b"), ("vaep", "rating_v")):
        rows = []
        for pid, d in both[both["player_id"].isin(multi_ids)].groupby("player_id"):
            g = d.groupby("position_group")[col].agg(["size", "mean"])
            g = g[g["size"] >= 3].sort_values("size", ascending=False)
            if len(g) >= 2:
                rows.append([pid, d["player_name"].iloc[0], g.index[0], int(g["size"].iloc[0]), float(g["mean"].iloc[0]), g.index[1], int(g["size"].iloc[1]), float(g["mean"].iloc[1])])
        t = pd.DataFrame(rows, columns=["player_id", "player", "main", "n_main", "mean_main", "other", "n_other", "mean_other"])
        t["diff"] = t["mean_other"] - t["mean_main"]
        t["pair"] = t["main"] + "->" + t["other"]
        paired[method] = t
    out["paired"] = paired
    # role within group (60+ minutes)
    full = both[both["minutes"] >= 60].copy()
    full["role"] = full["position_id"].map(ROLE_OF_POSITION)
    role = full.groupby(["position_group", "role"]).agg(n=("rating_b", "size"), baseline=("rating_b", "mean"), vaep=("rating_v", "mean"),
                                                        raw_b=("raw_score_b", "mean"), raw_v=("raw_score_v", "mean")).reset_index()
    out["role"] = role
    out["position_counts"] = lu["position_id"].map(POSITION_NAMES).value_counts()
    wm, w = lu[lu["position_id"].isin([12, 16])], lu[lu["position_id"].isin([17, 21])]
    out["wide"] = {"wide_mid": len(wm), "wingers": len(w), "players_both": len(set(wm["player_id"]) & set(w["player_id"]))}
    return out
