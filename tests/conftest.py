"""Shared fixtures for the rating-system test suite.

Data-backed fixtures FAIL (they do not skip) when the processed data or cached season
ratings are missing: the rules they encode must be demonstrably satisfied, and a silent
skip would hide that. Run the ingestion / rating steps first (see CLAUDE.md).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from src.contract import CONTRACT_COLUMNS

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIXTURES = pathlib.Path(__file__).parent / "fixtures"
METHODS = ("baseline", "vaep")

INGEST_HINT = "Run `.venv/Scripts/python -m src.ingest.build` (Step 2) first."


def rating_hint(method: str) -> str:
    return f"Run the {method} rating step and `python -m src.scale.apply --method {method}` (Step 3) first."


def require_file(path: pathlib.Path, hint: str) -> pathlib.Path:
    if not path.exists():
        pytest.fail(f"{path.relative_to(ROOT)} is missing. {hint}")
    return path


def load_parquet(name: str, hint: str, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(require_file(PROCESSED / name, hint), **kwargs)


def events_for(match_ids) -> pd.DataFrame:
    """Flattened events for the given matches only (keeps memory small)."""
    ids = [int(m) for m in match_ids]
    return load_parquet("events.parquet", INGEST_HINT, filters=[("match_id", "in", ids)])


def actions_for(match_ids) -> pd.DataFrame:
    """SPADL actions for the given matches only."""
    ids = [int(m) for m in match_ids]
    return load_parquet("actions.parquet", INGEST_HINT, filters=[("game_id", "in", ids)])


def import_or_fail(module: str, *names: str):
    """Import names from a module, failing the test with a clear message if absent."""
    try:
        mod = __import__(module, fromlist=list(names) or ["__name__"])
    except ImportError as exc:
        pytest.fail(f"{module} is not implemented yet: {exc}")
    out = []
    for name in names:
        if not hasattr(mod, name):
            pytest.fail(f"{module}.{name} is not implemented yet")
        out.append(getattr(mod, name))
    return out[0] if len(out) == 1 else tuple(out)


# --------------------------------------------------------------------------- real data
@pytest.fixture(scope="session")
def matches() -> pd.DataFrame:
    return load_parquet("matches.parquet", INGEST_HINT)


@pytest.fixture(scope="session")
def lineups() -> pd.DataFrame:
    return load_parquet("lineups.parquet", INGEST_HINT)


@pytest.fixture(scope="session", params=METHODS, ids=METHODS)
def method(request) -> str:
    """Parametrises a test over both rating methods."""
    return request.param


@pytest.fixture(scope="session")
def season_raw(method) -> pd.DataFrame:
    """Contract-format raw scores for the full 2015/16 season."""
    return load_parquet(f"raw_{method}.parquet", rating_hint(method))


@pytest.fixture(scope="session")
def season_ratings(method) -> pd.DataFrame:
    """Contract + rating for the full 2015/16 season (rated rows only)."""
    return load_parquet(f"ratings_{method}.parquet", rating_hint(method))


# ------------------------------------------------------------------ synthetic season
def make_fake_season(seed: int = 7, n_matches: int = 380) -> pd.DataFrame:
    """SYNTHETIC contract-format season for scale tests. Not real data.

    Each position group gets a deliberately different raw_score distribution so that
    any cross-group comparison shows up as a per-group mean far from 6.5. About 3% of
    raw scores are exact ties at 0.0, as stat-based scores produce in practice.
    """
    rng = np.random.default_rng(seed)
    dists = {
        "GK": lambda n: rng.normal(0.0, 1.0, n),
        "DEF": lambda n: rng.normal(50.0, 20.0, n),
        "MID": lambda n: rng.lognormal(0.0, 0.7, n),
        "FWD": lambda n: rng.exponential(2.0, n) - 1.0,
    }
    starters = ["GK"] + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 2
    rows = []
    pid = 1
    for match_id in range(1, n_matches + 1):
        for team in ("Synthetic Home FC", "Synthetic Away FC"):
            subbed_off = set(rng.choice(np.arange(1, 11), size=2, replace=False))  # never the GK
            for i, group in enumerate(starters):
                minutes = rng.uniform(55, 88) if i in subbed_off else rng.uniform(90, 97)
                rows.append((match_id, pid, f"Synthetic Player {pid}", team, group, minutes))
                pid += 1
            for group in rng.choice(["DEF", "MID", "FWD"], size=3):
                rows.append((match_id, pid, f"Synthetic Player {pid}", team, str(group), rng.uniform(1, 40)))
                pid += 1
    df = pd.DataFrame(rows, columns=["match_id", "player_id", "player_name", "team", "position_group", "minutes"])
    df["raw_score"] = 0.0
    for group, draw in dists.items():
        mask = (df["position_group"] == group).to_numpy()
        df.loc[mask, "raw_score"] = draw(int(mask.sum()))
    ties = rng.choice(df.index.to_numpy(), size=int(0.03 * len(df)), replace=False)
    df.loc[ties, "raw_score"] = 0.0
    df["minutes"] = df["minutes"].round(1)
    df["match_id"] = df["match_id"].astype("int64")
    df["player_id"] = df["player_id"].astype("int64")
    return df[CONTRACT_COLUMNS]


@pytest.fixture(scope="session")
def fake_season() -> pd.DataFrame:
    return make_fake_season()
