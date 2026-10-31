"""Regenerates tests/fixtures/sample_match.json: a SYNTHETIC web payload in the schema
from CLAUDE.md, for frontend development and tests. Deterministic; not real data."""
from __future__ import annotations

import json
import pathlib
import random

OUT = pathlib.Path(__file__).with_name("sample_match.json")

HOME_XI = [(1, "Goalkeeper", "GK"), (2, "Right Back", "DEF"), (3, "Right Center Back", "DEF"),
           (5, "Left Center Back", "DEF"), (6, "Left Back", "DEF"), (13, "Right Center Midfield", "MID"),
           (10, "Center Defensive Midfield", "MID"), (15, "Left Center Midfield", "MID"),
           (17, "Right Wing", "FWD"), (23, "Striker", "FWD"), (21, "Left Wing", "FWD")]
AWAY_XI = [(1, "Goalkeeper", "GK"), (2, "Right Back", "DEF"), (3, "Right Center Back", "DEF"),
           (5, "Left Center Back", "DEF"), (6, "Left Back", "DEF"), (9, "Right Defensive Midfield", "MID"),
           (11, "Left Defensive Midfield", "MID"), (18, "Right Attacking Midfield", "MID"),
           (19, "Center Attacking Midfield", "MID"), (20, "Left Attacking Midfield", "MID"), (23, "Striker", "FWD")]
SUBS = [(14, "Center Midfield", "MID", 25.0), (23, "Striker", "FWD", 12.0), (4, "Center Back", "DEF", 6.0)]


def build() -> dict:
    rng = random.Random(2016)
    players = []
    pid = 1
    for side, team_id, team, xi, formation in (("home", 1, "Synthetic Home FC", HOME_XI, "433"),
                                               ("away", 2, "Synthetic Away FC", AWAY_XI, "4231")):
        subbed_off = {5, 9}
        for i, (position_id, position, group) in enumerate(xi):
            minutes = 65.0 if i in subbed_off else 94.0
            players.append(_player(rng, pid, side, team_id, team, position_id, position, group, True, i + 1, minutes))
            pid += 1
        for j, (position_id, position, group, minutes) in enumerate(SUBS):
            players.append(_player(rng, pid, side, team_id, team, position_id, position, group, False, 12 + j, minutes))
            pid += 1
    return {
        "schema_version": 1,
        "note": "SYNTHETIC sample payload for frontend development and tests. Not real data.",
        "source": "StatsBomb Open Data",
        "season": "2015/16",
        "competition": "Premier League",
        "match_id": 1,
        "match_date": "2015-08-08",
        "matchweek": 1,
        "home": {"team_id": 1, "team": "Synthetic Home FC", "score": 2, "formation": "433"},
        "away": {"team_id": 2, "team": "Synthetic Away FC", "score": 1, "formation": "4231"},
        "methods": ["baseline", "vaep"],
        "players": players,
        "credits": "Data: StatsBomb Open Data (https://github.com/statsbomb/open-data)",
    }


def _player(rng, pid, side, team_id, team, position_id, position, group, is_starter, jersey, minutes):
    rated = minutes >= 10.0
    base = round(min(10.0, max(3.0, rng.gauss(6.6, 0.8))), 1) if rated else None
    vaep = round(min(10.0, max(3.0, rng.gauss(6.6, 0.8))), 1) if rated else None
    return {
        "player_id": pid,
        "player_name": f"Synthetic {side.title()} {jersey}",
        "team_id": team_id,
        "team": team,
        "side": side,
        "position_id": position_id,
        "position": position,
        "position_group": group,
        "is_starter": is_starter,
        "jersey_number": jersey,
        "minutes": minutes,
        "ratings": {"baseline": base, "vaep": vaep},
        "raw_scores": {"baseline": round(rng.gauss(0, 2), 2), "vaep": round(rng.gauss(0, 0.5), 3)},
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
