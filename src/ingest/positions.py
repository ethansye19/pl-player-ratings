"""StatsBomb position id -> position group. This is the only place the mapping lives.

Wingers (17, 21) count as forwards; attacking midfielders (18-20) count as midfielders.
"""
from __future__ import annotations

POSITION_NAMES = {
    1: "Goalkeeper",
    2: "Right Back",
    3: "Right Center Back",
    4: "Center Back",
    5: "Left Center Back",
    6: "Left Back",
    7: "Right Wing Back",
    8: "Left Wing Back",
    9: "Right Defensive Midfield",
    10: "Center Defensive Midfield",
    11: "Left Defensive Midfield",
    12: "Right Midfield",
    13: "Right Center Midfield",
    14: "Center Midfield",
    15: "Left Center Midfield",
    16: "Left Midfield",
    17: "Right Wing",
    18: "Right Attacking Midfield",
    19: "Center Attacking Midfield",
    20: "Left Attacking Midfield",
    21: "Left Wing",
    22: "Right Center Forward",
    23: "Center Forward",
    24: "Left Center Forward",
    25: "Secondary Striker",
}

POSITION_GROUP_BY_ID = {
    1: "GK",
    **{pid: "DEF" for pid in (2, 3, 4, 5, 6, 7, 8)},
    **{pid: "MID" for pid in (9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20)},
    **{pid: "FWD" for pid in (17, 21, 22, 23, 24, 25)},
}


def position_group(position_id: int) -> str:
    """Return GK / DEF / MID / FWD for a StatsBomb position id."""
    try:
        return POSITION_GROUP_BY_ID[int(position_id)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unknown StatsBomb position id: {position_id!r}") from None
