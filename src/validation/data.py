"""Read-only loading of the processed data for the validation report."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "statsbomb"
WEB_DATA = ROOT / "web" / "data"
REPORT_PATH = ROOT / "reports" / "validation.md"
METHODS = ("baseline", "vaep")

ACTION_VALUE_COLUMNS = [
    "game_id", "action_id", "player_id", "team_id", "period_id", "time_seconds",
    "type_name", "result_name", "offensive_value", "defensive_value", "vaep_value", "fold_id",
]


@dataclass
class Data:
    matches: pd.DataFrame
    lineups: pd.DataFrame
    raw: dict = field(default_factory=dict)       # method -> contract frame (all appearances)
    ratings: dict = field(default_factory=dict)   # method -> contract + rating (rated rows)
    stats: pd.DataFrame | None = None             # baseline_stats
    actions_values: pd.DataFrame | None = None    # action_values
    splits: dict | None = None
    both: pd.DataFrame | None = None              # rated rows of both methods joined

    @property
    def match_index(self) -> pd.DataFrame:
        return self.matches.set_index("match_id")

    def context(self, match_id: int, team: str) -> tuple[str, str, int, str]:
        """(opponent, score as 'h-a (H|A)', matchweek, result W/D/L) for a team in a match."""
        m = self.match_index.loc[int(match_id)]
        home = team == m["home_team"]
        opp = m["away_team"] if home else m["home_team"]
        gf, ga = (m["home_score"], m["away_score"]) if home else (m["away_score"], m["home_score"])
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        return str(opp), f"{m['home_score']}-{m['away_score']} ({'H' if home else 'A'})", int(m["match_week"]), result


def _read(name: str, **kwargs) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the earlier steps first (see CLAUDE.md)")
    return pd.read_parquet(path, **kwargs)


def load() -> Data:
    data = Data(matches=_read("matches.parquet"), lineups=_read("lineups.parquet"))
    for method in METHODS:
        data.raw[method] = _read(f"raw_{method}.parquet")
        data.ratings[method] = _read(f"ratings_{method}.parquet")
    data.stats = _read("baseline_stats.parquet")
    data.actions_values = _read("action_values.parquet", columns=ACTION_VALUE_COLUMNS)
    data.splits = json.loads((PROCESSED / "vaep_splits.json").read_text(encoding="utf-8"))
    rb, rv = data.ratings["baseline"], data.ratings["vaep"]
    both = rb.merge(rv[["match_id", "player_id", "raw_score", "rating"]], on=["match_id", "player_id"], suffixes=("_b", "_v"))
    both = both.merge(
        data.lineups[["match_id", "player_id", "is_starter", "position_id", "position", "minute_on", "red_card_minute"]],
        on=["match_id", "player_id"], how="left",
    )
    weeks = data.matches.set_index("match_id")["match_week"]
    both["week"] = both["match_id"].map(weeks)
    data.both = both
    return data


def md_table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    """Markdown table. ``align`` is a string of 'l'/'r' per column (default: numbers right)."""
    def cell(value) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    if align is None:
        align = "".join("r" if rows and isinstance(rows[0][i], (int, float)) and not isinstance(rows[0][i], bool) else "l" for i in range(len(headers)))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def fmt(value, nd: int = 3) -> str:
    return f"{value:.{nd}f}"
