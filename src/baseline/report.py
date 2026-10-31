"""Write ``reports/baseline.md`` from the cached baseline tables.

``python -m src.baseline.report`` reads ``data/processed/baseline_stats.parquet``,
``raw_baseline.parquet``, ``matches.parquet`` and ``lineups.parquet`` and writes the stat
table (definition, weight per group, justification), the raw-score distribution per
position group (including where low-activity players land) and the top 15 / bottom 10
player-matches per group with their biggest stat contributions.
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np
import pandas as pd

from src.baseline.rate import PROCESSED_DIR, RAW_FILE, STATS_FILE
import src.baseline.weights as weights_module
from src.baseline.weights import LABELS, STATS, WEIGHTS, stat_definitions, weight_of
from src.contract import MIN_MINUTES, POSITION_GROUPS

log = logging.getLogger(__name__)

REPORT_PATH = pathlib.Path(__file__).resolve().parents[2] / "reports" / "baseline.md"
MINUTE_BUCKETS = [0, 10, 20, 30, 45, 60, 80, 200]
BUCKET_LABELS = ["0-10 (unrated)", "10-20", "20-30", "30-45", "45-60", "60-80", "80+"]


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def _fmt_count(x: float) -> str:
    return f"{x:g}" if float(x).is_integer() else f"{x:.2f}"


def load_tables(processed_dir: pathlib.Path) -> pd.DataFrame:
    stats = pd.read_parquet(processed_dir / STATS_FILE)
    raw = pd.read_parquet(processed_dir / RAW_FILE)
    matches = pd.read_parquet(processed_dir / "matches.parquet")
    df = stats.merge(raw[["match_id", "player_id", "player_name", "team", "raw_score"]], on=["match_id", "player_id"])
    df = df.merge(matches[["match_id", "home_team", "away_team", "home_score", "away_score"]], on="match_id")
    is_home = df["team"] == df["home_team"]
    df["opponent"] = np.where(is_home, df["away_team"], df["home_team"])
    own = np.where(is_home, df["home_score"], df["away_score"])
    other = np.where(is_home, df["away_score"], df["home_score"])
    df["score"] = [f"{a}-{b}" for a, b in zip(own, other)]
    return df


def contributions(row: pd.Series, group: str, k: int = 3) -> str:
    items = []
    for stat in STATS:
        value = float(row[stat]) * weight_of(group, stat)
        if value != 0.0:
            items.append((stat, float(row[stat]), weight_of(group, stat), value))
    items.sort(key=lambda t: -abs(t[3]))
    return "; ".join(f"{LABELS[s]} {_fmt_count(c)} x {w:g} = {v:+.2f}" for s, c, w, v in items[:k])


def docstring_section(title: str) -> list[str]:
    """Body of a titled section of the weights docstring (up to the next underlined title)."""
    doc = weights_module.__doc__ or ""
    marker = title + "\n" + "-" * len(title) + "\n"
    if marker not in doc:
        return []
    body = doc.split(marker, 1)[1]
    lines = []
    for line in body.splitlines():
        if line and set(line) == {"-"}:
            lines.pop()  # the next section's title line
            break
        lines.append(line)
    return [" ".join(l.strip() for l in lines).strip()]


def stat_table() -> list[str]:
    definitions = stat_definitions()
    lines = [
        "| stat | definition | GK | DEF | MID | FWD | justification (group with the largest weight) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for stat in STATS:
        weights = {g: weight_of(g, stat) for g in POSITION_GROUPS}
        # quote the outfield group that weights the stat most (ties: DEF, MID, FWD order); GK only for GK-only stats
        candidates = [g for g in ("DEF", "MID", "FWD") if stat in WEIGHTS[g]] or ["GK"]
        primary = max(candidates, key=lambda g: abs(weights[g]))
        why = WEIGHTS[primary][stat].why if stat in WEIGHTS[primary] else ""
        cells = [f"{weights[g]:g}" if stat in WEIGHTS[g] else "-" for g in POSITION_GROUPS]
        lines.append(f"| `{stat}` | {definitions[stat]} | {' | '.join(cells)} | {primary}: {why} |")
    return lines


def justification_tables() -> list[str]:
    lines = []
    for group in POSITION_GROUPS:
        lines += [f"### {group} ({len(WEIGHTS[group])} stats)", "", "| stat | weight | why |", "|---|---:|---|"]
        for stat, w in sorted(WEIGHTS[group].items(), key=lambda kv: -abs(kv[1].weight)):
            lines.append(f"| `{stat}` | {w.weight:g} | {w.why} |")
        lines.append("")
    return lines


def distribution_section(df: pd.DataFrame) -> list[str]:
    lines = [
        "| group | n | mean | sd | min | p1 | p5 | p25 | median | p75 | p95 | p99 | max | exact zeros | share <= 0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in POSITION_GROUPS:
        s = df.loc[df["position_group"] == group, "raw_score"]
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        lines.append(
            f"| {group} | {len(s)} | {_fmt(s.mean())} | {_fmt(s.std())} | {_fmt(s.min())} | {_fmt(q[0.01])} | {_fmt(q[0.05])} | "
            f"{_fmt(q[0.25])} | {_fmt(q[0.5])} | {_fmt(q[0.75])} | {_fmt(q[0.95])} | {_fmt(q[0.99])} | {_fmt(s.max())} | "
            f"{100 * (s == 0).mean():.2f}% | {100 * (s <= 0).mean():.1f}% |"
        )
    lines += ["", "Share of each group's raw-score variance explained by the team-context stats (variance of the summed",
              "goals_conceded + xg_faced + clean_sheet contributions over the variance of raw_score):", ""]
    lines += ["| group | team-context variance share |", "|---|---:|"]
    for group in POSITION_GROUPS:
        part = df[df["position_group"] == group]
        context = sum(part[s] * weight_of(group, s) for s in ("goals_conceded", "xg_faced", "clean_sheet"))
        share = context.var() / part["raw_score"].var() if part["raw_score"].var() > 0 else float("nan")
        lines.append(f"| {group} | {100 * share:.1f}% |")
    before = PROCESSED_DIR / "before" / "raw_baseline.parquet"
    if before.exists():
        old = pd.read_parquet(before)[["match_id", "player_id", "position_group", "raw_score"]].rename(columns={"raw_score": "before"})
        both = df[["match_id", "player_id", "position_group", "raw_score"]].merge(old, on=["match_id", "player_id", "position_group"])
        lines += ["", f"Before vs after the Step 4 weight decisions (`before` = frozen snapshot in `{before.as_posix().split('processed/')[-1]}`):", "",
                  "| group | n | mean before | mean after | sd before | sd after | min before | min after | max before | max after | zeros before | zeros after | Spearman before/after |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in POSITION_GROUPS:
            b = both[both["position_group"] == group]
            rho = b["before"].corr(b["raw_score"], method="spearman")
            lines.append(
                f"| {group} | {len(b)} | {_fmt(b['before'].mean())} | {_fmt(b['raw_score'].mean())} | {_fmt(b['before'].std())} | {_fmt(b['raw_score'].std())} | "
                f"{_fmt(b['before'].min())} | {_fmt(b['raw_score'].min())} | {_fmt(b['before'].max())} | {_fmt(b['raw_score'].max())} | "
                f"{100 * (b['before'] == 0).mean():.2f}% | {100 * (b['raw_score'] == 0).mean():.2f}% | {rho:.3f} |"
            )
    lines += ["", "Where low-activity players land (median raw score of each minutes bucket and its percentile rank within the group):", ""]
    header = "| group | " + " | ".join(BUCKET_LABELS) + " | quiet 85+ min starters* | corr(raw, minutes) |"
    lines += [header, "|---|" + "---:|" * (len(BUCKET_LABELS) + 2)]
    notable = ["goals", "assists", "yellow_cards", "red_cards", "errors", "own_goals", "penalties_conceded"]
    for group in POSITION_GROUPS:
        part = df[df["position_group"] == group]
        ranks = part["raw_score"].rank(pct=True)
        cells = []
        bins = pd.cut(part["minutes"], MINUTE_BUCKETS, right=False, labels=BUCKET_LABELS)
        for label in BUCKET_LABELS:
            idx = part.index[bins == label]
            if len(idx) == 0:
                cells.append("n/a")
            else:
                cells.append(f"{_fmt(part.loc[idx, 'raw_score'].median())} (p{100 * ranks[idx].median():.0f}, n={len(idx)})")
        quiet_idx = part.index[(part["minutes"] >= 85) & (part[notable].sum(axis=1) == 0)]
        quiet = f"{_fmt(part.loc[quiet_idx, 'raw_score'].median())} (p{100 * ranks[quiet_idx].median():.0f}, n={len(quiet_idx)})"
        corr = part[["raw_score", "minutes"]].corr().iloc[0, 1]
        lines.append(f"| {group} | " + " | ".join(cells) + f" | {quiet} | {corr:.2f} |")
    lines += ["", "*quiet = no goals, assists, cards, errors, own goals or penalties conceded."]
    return lines


def ranking_section(df: pd.DataFrame, group: str, n: int, ascending: bool) -> list[str]:
    part = df[df["position_group"] == group].sort_values(["raw_score", "match_id", "player_id"], ascending=[ascending, True, True]).head(n)
    lines = [
        "| # | raw | player | team | match_id | opponent | score | min | biggest contributions (count x weight = value) |",
        "|---:|---:|---|---|---|---|---|---:|---|",
    ]
    for i, (_, r) in enumerate(part.iterrows(), start=1):
        lines.append(
            f"| {i} | {r['raw_score']:.2f} | {r['player_name']} | {r['team']} | {r['match_id']} | {r['opponent']} | {r['score']} | "
            f"{r['minutes']:.1f} | {contributions(r, group)} |"
        )
    return lines


def season_leaders(df: pd.DataFrame, group: str, n: int = 10, min_apps: int = 15) -> list[str]:
    full = df[df["minutes"] >= 60]
    agg = full[full["position_group"] == group].groupby(["player_id", "player_name"])["raw_score"].agg(["count", "mean"]).reset_index()
    agg = agg[agg["count"] >= min_apps].sort_values("mean", ascending=False)
    lines = ["| # | player | 60+ min apps | mean raw |", "|---:|---|---:|---:|"]
    for i, r in enumerate(agg.head(n).itertuples(), start=1):
        lines.append(f"| {i} | {r.player_name} | {r.count} | {r.mean:.2f} |")
    return lines


def build_report(processed_dir: pathlib.Path = PROCESSED_DIR) -> str:
    df = load_tables(processed_dir)
    n_groups = {g: int((df["position_group"] == g).sum()) for g in POSITION_GROUPS}
    out: list[str] = [
        "# Baseline method: hand-weighted per-match stats",
        "",
        "Data: StatsBomb Open Data, Premier League 2015/16 (https://github.com/statsbomb/open-data).",
        "Generated by `python -m src.baseline.report` from `data/processed/baseline_stats.parquet` and",
        "`data/processed/raw_baseline.parquet` (written by `python -m src.baseline.rate`).",
        "",
        f"- matches: {df['match_id'].nunique()}; player-match appearances: {len(df)} "
        f"(GK {n_groups['GK']}, DEF {n_groups['DEF']}, MID {n_groups['MID']}, FWD {n_groups['FWD']})",
        f"- stats: {len(STATS)}; weights per group: " + ", ".join(f"{g} {len(WEIGHTS[g])}" for g in POSITION_GROUPS),
        f"- raw_score = sum over the group's profile of weight x stat; appearances under {MIN_MINUTES:g} minutes are scored but dropped by the scaler",
        "- every stat is counted from that match's events only; goals conceded / xG faced / clean sheet are counted while the player was on the pitch",
        "",
        "## 1. Stats and weights",
        "",
        "Definitions are in terms of `events.parquet` columns (statsbombpy naming); `-` means the stat is not in that group's profile (weight 0).",
        "",
        *stat_table(),
        "",
        "### Retired stat",
        "",
        *docstring_section("Retired stat"),
        "",
        "### How low activity is treated",
        "",
        "The raw score has no minutes term: a player with no notable events scores exactly 0. Routine volume (passes,",
        "pressures, recoveries, carries) earns little while every misplaced pass, lost duel, lost possession and goal",
        "conceded while on the pitch costs something, so a 15-minute substitute with nothing to show sits at ~0, below",
        "the quiet full-match starters (who land a little above the middle) but above everyone with a clearly negative",
        "game. The bottom of each group is reserved for red cards, own goals, errors, penalties conceded and heavy",
        "defeats (see section 2 for the measured positions, including before/after the Step 4 decisions, and section 4",
        "for the lowest scores).",
        "",
        "## 2. Raw-score distribution per position group",
        "",
        *distribution_section(df),
        "",
        "## 3. Top 15 raw scores of the season per group",
        "",
    ]
    for group in POSITION_GROUPS:
        out += [f"### {group}", "", *ranking_section(df, group, 15, ascending=False), ""]
    out += ["## 4. Lowest 10 raw scores per group", ""]
    for group in POSITION_GROUPS:
        out += [f"### {group}", "", *ranking_section(df, group, 10, ascending=True), ""]
    out += ["## 5. Season leaders (mean raw score over 60+ minute appearances, at least 15 of them)", ""]
    for group in POSITION_GROUPS:
        out += [f"### {group}", "", *season_leaders(df, group), ""]
    out += [
        "## 6. Design notes and known limitations",
        "",
        "- Team context is counted while the player was on the pitch: event times come from `period` + `timestamp` with period",
        "  lengths from the match's Half End events, compared with the lineups' `minute_on` / `minute_off` (the same clock the",
        "  ingest step used), so a substitute is only charged for goals conceded after coming on and a dismissed player is not",
        "  charged for goals after the red card.",
        "- `clean_sheet` follows the FPL rule (60+ minutes and nothing conceded while on the pitch) so the current-season pipeline",
        "  can map it one to one; `goals_conceded` includes own goals, as FPL does.",
        "- `xg_faced` is positive for goalkeepers (shot-stopping is judged against the chances faced) and negative for defenders",
        "  and midfielders (chance suppression); forwards carry only a token share of `goals_conceded` and no `xg_faced`.",
        "- `xg` excludes penalties (a penalty is credited through `goals` or `penalties_missed`); failed crosses are not counted",
        "  as incomplete passes because a cross is an attacking attempt, not a giveaway in build-up.",
        "- Event totals reconcile with the match sheet: per-team goals (shots + opponents' own goals) equal `matches.parquet` in",
        "  all 760 team-matches and the season total is 1,026 goals (`tests/test_baseline_stats.py`).",
        "- Limitations: team-context stats are identical for every teammate on the pitch; aerials and clearances reward target",
        "  men and deep-sitting defences by volume; set-piece takers collect key passes cheaply; there is no 'big chance missed'",
        "  stat, no opponent-strength or home/away adjustment, and a 15-minute cameo can only move a few tenths from zero unless",
        "  it contains a goal, an assist or a card. Weights are hand-set and open to revision at the Step 4 checkpoint.",
        "",
        "## Appendix: every weight with its justification",
        "",
        *justification_tables(),
    ]
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write reports/baseline.md from the cached baseline tables.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    text = build_report(pathlib.Path(args.processed_dir))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
