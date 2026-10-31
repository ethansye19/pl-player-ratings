"""The interface contract itself: validate_contract() must accept conforming frames and
reject the ways a rating method could silently break the rules."""
import numpy as np
import pandas as pd
import pytest

from src.contract import (
    CONTRACT_COLUMNS,
    MIN_MINUTES,
    POSITION_GROUPS,
    ContractError,
    validate_contract,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": np.array([1, 1, 1, 2], dtype="int64"),
            "player_id": np.array([10, 11, 12, 10], dtype="int64"),
            "player_name": ["Synthetic A", "Synthetic B", "Synthetic C", "Synthetic A"],
            "team": ["Synthetic FC", "Synthetic FC", "Synthetic United", "Synthetic FC"],
            "position_group": ["GK", "DEF", "FWD", "GK"],
            "minutes": [90.0, 12.5, 3.0, 95.2],
            "raw_score": [0.4, -1.2, 0.0, 2.25],
        }
    )


def test_validate_contract_accepts_a_conforming_frame():
    df = _valid_frame()
    assert validate_contract(df) is df
    assert list(df.columns) == CONTRACT_COLUMNS


def test_validate_contract_rejects_a_missing_column():
    with pytest.raises(ContractError, match="missing contract columns"):
        validate_contract(_valid_frame().drop(columns=["position_group"]))


def test_validate_contract_rejects_an_unknown_position_group():
    df = _valid_frame()
    df.loc[0, "position_group"] = "WING"
    with pytest.raises(ContractError, match="position_group"):
        validate_contract(df)
    assert set(POSITION_GROUPS) == {"GK", "DEF", "MID", "FWD"}


def test_validate_contract_rejects_duplicate_player_match_rows():
    df = pd.concat([_valid_frame(), _valid_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate"):
        validate_contract(df)


def test_validate_contract_rejects_non_finite_raw_scores():
    df = _valid_frame()
    df.loc[1, "raw_score"] = np.nan
    with pytest.raises(ContractError, match="raw_score"):
        validate_contract(df)


def test_validate_contract_with_rating_enforces_row_level_rules():
    df = _valid_frame()
    df = df[df["minutes"] >= MIN_MINUTES].copy()
    df["rating"] = [6.5, 5.9, 8.1]
    validate_contract(df, with_rating=True)

    bad = df.copy()
    bad.loc[bad.index[0], "rating"] = 6.55  # two decimals
    with pytest.raises(ContractError, match="decimal"):
        validate_contract(bad, with_rating=True)

    bad = df.copy()
    bad.loc[bad.index[0], "rating"] = 10.1  # out of bounds
    with pytest.raises(ContractError, match="outside"):
        validate_contract(bad, with_rating=True)

    bad = df.copy()
    bad.loc[bad.index[0], "minutes"] = 9.9  # rated with under 10 minutes
    with pytest.raises(ContractError, match="minutes"):
        validate_contract(bad, with_rating=True)
