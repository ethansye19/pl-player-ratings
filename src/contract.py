"""Shared interface contract for the rating system.

Every rating method (baseline, VAEP, the FPL current-season pipeline) must output a
DataFrame with exactly the columns in ``CONTRACT_COLUMNS``. The scale module turns
``raw_score`` into ``rating``. All constants that encode a rule live here so that
source modules and tests agree on them. Do not redefine them elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

CONTRACT_COLUMNS = [
    "match_id",        # int64  StatsBomb match_id (FPL fixture id in the current-season pipeline)
    "player_id",       # int64  StatsBomb player_id (FPL element id in the current-season pipeline)
    "player_name",     # str
    "team",            # str    team name
    "position_group",  # str    one of POSITION_GROUPS
    "minutes",         # float  minutes on the pitch including stoppage time, 1 decimal
    "raw_score",       # float  finite; higher is better; only comparable within position_group
]
RATING_COLUMN = "rating"
POSITION_GROUPS = ("GK", "DEF", "MID", "FWD")

MIN_MINUTES = 10.0          # only players with 10+ minutes get a rating
RATING_MIN = 3.0
RATING_MAX = 10.0
RATING_DECIMALS = 1
TARGET_MEAN = 6.5           # full-season mean rating
MEAN_TOLERANCE = 0.1        # ... within +/- this
MAX_SHARE_OF_TENS = 0.001   # fewer than 0.1% of ratings may equal 10.0

SEED = 20152016             # single seed for anything stochastic (VAEP folds, model fitting)


class ContractError(ValueError):
    """Raised when a DataFrame does not satisfy the interface contract."""


def validate_contract(
    df: pd.DataFrame,
    *,
    with_rating: bool = False,
    require_appearance: bool = True,
) -> pd.DataFrame:
    """Validate ``df`` against the interface contract and return it unchanged.

    Parameters
    ----------
    with_rating:
        Also require the ``rating`` column produced by the scale module, and check the
        rating rules that hold row by row (bounds, one decimal, 10+ minutes).
    require_appearance:
        Require ``minutes > 0`` (rating methods only output players who appeared).
    """
    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"missing contract columns: {missing}")
    if with_rating and RATING_COLUMN not in df.columns:
        raise ContractError(f"missing '{RATING_COLUMN}' column")

    for col in ("match_id", "player_id"):
        if not is_integer_dtype(df[col]):
            raise ContractError(f"'{col}' must be an integer dtype, got {df[col].dtype}")
    for col in ("minutes", "raw_score"):
        if not is_numeric_dtype(df[col]):
            raise ContractError(f"'{col}' must be numeric, got {df[col].dtype}")
        if not np.isfinite(df[col].to_numpy(dtype="float64")).all():
            raise ContractError(f"'{col}' contains NaN or inf")
    for col in ("player_name", "team"):
        if df[col].isna().any() or (df[col].astype(str).str.len() == 0).any():
            raise ContractError(f"'{col}' has empty values")

    bad_groups = set(df["position_group"].unique()) - set(POSITION_GROUPS)
    if bad_groups:
        raise ContractError(f"unknown position_group values: {sorted(map(str, bad_groups))}")

    if (df["minutes"] < 0).any():
        raise ContractError("negative minutes")
    if require_appearance and (df["minutes"] <= 0).any():
        raise ContractError("rows with minutes <= 0 (only players who appeared belong in the output)")

    dupes = df.duplicated(subset=["match_id", "player_id"])
    if dupes.any():
        raise ContractError(f"{int(dupes.sum())} duplicate (match_id, player_id) rows")

    if with_rating:
        rating = df[RATING_COLUMN]
        if not is_numeric_dtype(rating) or not np.isfinite(rating.to_numpy(dtype="float64")).all():
            raise ContractError("'rating' must be numeric and finite")
        if (rating < RATING_MIN).any() or (rating > RATING_MAX).any():
            raise ContractError(f"ratings outside [{RATING_MIN}, {RATING_MAX}]")
        scaled = rating.to_numpy(dtype="float64") * 10 ** RATING_DECIMALS
        if not np.allclose(scaled, np.round(scaled), atol=1e-6):
            raise ContractError(f"ratings must have {RATING_DECIMALS} decimal place")
        if (df["minutes"] < MIN_MINUTES).any():
            raise ContractError(f"rated rows with fewer than {MIN_MINUTES} minutes")
    return df
