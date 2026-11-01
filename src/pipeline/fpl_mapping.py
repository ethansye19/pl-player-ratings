"""How the baseline model maps onto the Fantasy Premier League API (current season).

FPL publishes per-player, per-gameweek totals (``/api/event/<gw>/live/``) and no event
data, so only a subset of the 35 baseline stats exists. ``FPL_STAT_MAP`` names the FPL
field for each baseline stat that has one; ``UNAVAILABLE_STATS`` explains every stat that
does not. ``FPL_WEIGHTS`` is the baseline weight table restricted to the mapped stats,
with three documented adjustments (see ``reports/fpl_mapping.md``):

1. ``goals`` <- ``goals_scored`` includes penalties, and FPL has no penalties-scored field,
   so the goal weight is blended towards the 1.25 penalty weight using the 2015/16 share
   of goals that were penalties (DEF 4%, MID 9%, FWD 8%): DEF 3.5 -> 3.4, MID 3.0 -> 2.85,
   FWD 3.0 -> 2.85, GK unchanged (keepers' goals are not penalties).
2. ``clearances`` <- ``clearances_blocks_interceptions`` combines three baseline stats, so
   it carries the count-weighted mean of their three weights measured on 2015/16:
   GK 0.06, DEF 0.15, MID 0.15, FWD 0.09; ``blocks`` and ``interceptions`` are then
   marked unavailable (they are inside that field, not lost).
3. Everything else keeps its baseline weight. Stats that FPL lacks simply drop out; the
   percentile scaler re-anchors each group at 6.5, so the loss is resolution (fewer
   distinct raw scores), not bias.

Positions come from FPL ``element_type`` (1 GK, 2 DEF, 3 MID, 4 FWD). FPL's MID/FWD
split follows the game's classification, not the formation, so wingers are often MID.
"""
from __future__ import annotations

from src.baseline.weights import STATS, WEIGHTS, Weight

FPL_POSITION_GROUPS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# baseline stat -> FPL live-stats field
FPL_STAT_MAP: dict[str, str] = {
    "goals": "goals_scored",
    "assists": "assists",
    "xg": "expected_goals",
    "penalties_missed": "penalties_missed",
    "tackles_won": "tackles",
    "clearances": "clearances_blocks_interceptions",
    "ball_recoveries": "recoveries",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "own_goals": "own_goals",
    "goals_conceded": "goals_conceded",
    "clean_sheet": "clean_sheets",
    "xg_faced": "expected_goals_conceded",
    "saves": "saves",
    "penalties_saved": "penalties_saved",
}

UNAVAILABLE_STATS: dict[str, str] = {
    "penalties_scored": "FPL reports goals_scored with penalties included and has no penalties-scored field; the goal weight is blended instead",
    "shots_on_target": "FPL publishes no shot counts (threat is an index, not a count)",
    "key_passes": "FPL publishes no key-pass count; expected_assists is a different quantity and not a baseline stat",
    "dribbles_completed": "FPL publishes no dribble or take-on counts",
    "progressive_carries": "FPL publishes no carry data",
    "penalties_won": "FPL does not record who won a penalty",
    "passes_completed": "FPL publishes no passing counts",
    "passes_incomplete": "FPL publishes no passing counts",
    "progressive_passes": "FPL publishes no passing counts or locations",
    "possession_lost": "FPL publishes no dispossession or miscontrol counts",
    "dribbled_past": "FPL publishes no defensive-duel outcomes",
    "interceptions": "counted inside clearances_blocks_interceptions, which is mapped through 'clearances' with a blended weight",
    "blocks": "counted inside clearances_blocks_interceptions, which is mapped through 'clearances' with a blended weight",
    "aerials_won": "FPL publishes no aerial-duel counts",
    "aerials_lost": "FPL publishes no aerial-duel counts",
    "pressures": "FPL publishes no pressure events",
    "fouls_committed": "FPL publishes no foul counts (only cards)",
    "errors": "FPL publishes no error-leading-to-shot events",
    "penalties_conceded": "FPL does not record who conceded a penalty",
    "keeper_claims": "FPL publishes no claims, punches or sweeps for goalkeepers",
}

assert set(FPL_STAT_MAP) | set(UNAVAILABLE_STATS) == set(STATS), "every baseline stat must be mapped or explained"
assert not set(FPL_STAT_MAP) & set(UNAVAILABLE_STATS)

# FPL field used for minutes and starts (not baseline stats, but needed for the contract)
FPL_MINUTES_FIELD = "minutes"
FPL_STARTS_FIELD = "starts"
# fields reported as strings by the API
FPL_DECIMAL_FIELDS = {"expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
                      "influence", "creativity", "threat", "ict_index"}

_GOAL_WEIGHT = {"GK": 4.00, "DEF": 3.40, "MID": 2.85, "FWD": 2.85}
_CBI_WEIGHT = {"GK": 0.06, "DEF": 0.15, "MID": 0.15, "FWD": 0.09}


def _build_weights() -> dict[str, dict[str, Weight]]:
    out: dict[str, dict[str, Weight]] = {}
    for group, table in WEIGHTS.items():
        rows: dict[str, Weight] = {}
        for stat in FPL_STAT_MAP:
            if stat == "goals":
                rows[stat] = Weight(_GOAL_WEIGHT[group], f"baseline goal weight {table['goals'].weight} blended towards the 1.25 penalty weight because FPL goals include penalties")
            elif stat == "clearances":
                rows[stat] = Weight(_CBI_WEIGHT[group], "count-weighted mean of the baseline clearances, blocks and interceptions weights (2015/16 mix) for FPL's combined CBI field")
            elif stat in table:
                rows[stat] = Weight(table[stat].weight, f"baseline weight kept: {table[stat].why}")
        out[group] = rows
    return out


FPL_WEIGHTS: dict[str, dict[str, Weight]] = _build_weights()
