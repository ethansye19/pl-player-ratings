"""End-to-end test of `python -m src.scale.apply` on a SYNTHETIC raw file.

The raw file is written by make_fake_season() into pytest's tmp_path as raw_test.parquet and
run with `--method test`; the real data/ folder is never read or written."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.contract import (
    CONTRACT_COLUMNS,
    MAX_SHARE_OF_TENS,
    MEAN_TOLERANCE,
    MIN_MINUTES,
    POSITION_GROUPS,
    RATING_MAX,
    TARGET_MEAN,
    validate_contract,
)
from src.scale.apply import SORT_KEYS, main
from src.scale.scaler import PercentileScaler
from tests.conftest import ROOT

KEYS = ["match_id", "player_id"]


@pytest.fixture()
def processed(tmp_path, fake_season):
    """A temporary 'processed' folder holding the synthetic raw_test.parquet."""
    fake_season.to_parquet(tmp_path / "raw_test.parquet", index=False)
    return tmp_path


def run_cli(folder, method="test"):
    return main(["--method", method, "--processed", str(folder)])


def test_cli_writes_the_frozen_scaler_and_the_ratings(processed, fake_season, capsys):
    assert run_cli(processed) == 0
    assert {p.name for p in processed.iterdir()} == {"raw_test.parquet", "ratings_test.parquet", "scaler_test.json"}

    ratings = pd.read_parquet(processed / "ratings_test.parquet")
    validate_contract(ratings, with_rating=True)
    eligible = fake_season[fake_season["minutes"] >= MIN_MINUTES]
    assert len(ratings) == len(eligible) < len(fake_season)
    assert set(map(tuple, ratings[KEYS].to_numpy())) == set(map(tuple, eligible[KEYS].to_numpy()))
    assert list(ratings.columns) == CONTRACT_COLUMNS + ["rating"]
    assert ratings["rating"].dtype == np.float64

    # sorted by match_id, team, player_id
    assert_frame_equal(ratings, ratings.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True))
    assert SORT_KEYS == ["match_id", "team", "player_id"]

    # identical to fitting and transforming in memory
    expected = PercentileScaler().fit(fake_season).transform(fake_season)
    expected = expected.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)
    assert_frame_equal(ratings, expected)

    # the saved scaler reproduces the ratings, one match at a time (the isolation rule)
    scaler = PercentileScaler.load(processed / "scaler_test.json")
    for match_id in (1, 190, 380):
        one = scaler.transform(fake_season[fake_season["match_id"] == match_id])
        one = one.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)
        full = ratings[ratings["match_id"] == match_id].reset_index(drop=True)
        assert_frame_equal(one, full)


def test_cli_prints_the_documented_summary(processed, fake_season, capsys):
    run_cli(processed)
    out = capsys.readouterr().out
    rated = int((fake_season["minutes"] >= MIN_MINUTES).sum())
    assert f"rows rated     : {rated} of {len(fake_season)} raw rows" in out
    assert "mean rating" in out and "mean by group" in out
    for group in POSITION_GROUPS:
        assert f"{group} 6." in out, f"per-group mean for {group} missing"
    assert f"share of {RATING_MAX:.1f}" in out and "min / max" in out
    assert "WARNING" not in out
    assert str(processed / "scaler_test.json") in out and str(processed / "ratings_test.parquet") in out


def test_cli_ratings_satisfy_the_distribution_rules(processed):
    run_cli(processed)
    ratings = pd.read_parquet(processed / "ratings_test.parquet")
    assert abs(ratings["rating"].mean() - TARGET_MEAN) <= MEAN_TOLERANCE
    for group, mean in ratings.groupby("position_group")["rating"].mean().items():
        assert abs(mean - TARGET_MEAN) <= MEAN_TOLERANCE, group
    assert (ratings["rating"] == RATING_MAX).mean() < MAX_SHARE_OF_TENS


def test_cli_keeps_extra_input_columns_so_transform_of_raw_matches_the_ratings_file(tmp_path, fake_season):
    raw = fake_season.assign(team_id=5, is_starter=fake_season["minutes"] > 80)
    raw.to_parquet(tmp_path / "raw_test.parquet", index=False)
    assert run_cli(tmp_path) == 0
    ratings = pd.read_parquet(tmp_path / "ratings_test.parquet")
    assert list(ratings.columns) == list(raw.columns) + ["rating"]
    one = PercentileScaler.load(tmp_path / "scaler_test.json").transform(raw[raw["match_id"] == 7])
    full = ratings[ratings["match_id"] == 7][one.columns]
    assert_frame_equal(one.sort_values(KEYS).reset_index(drop=True), full.sort_values(KEYS).reset_index(drop=True))


def test_cli_with_a_missing_raw_file_fails_cleanly(tmp_path, capsys):
    assert run_cli(tmp_path, method="vaep") == 1
    err = capsys.readouterr().err
    assert "raw_vaep.parquet" in err and "not found" in err
    assert list(tmp_path.iterdir()) == []


def test_cli_with_a_raw_file_that_breaks_the_contract_writes_nothing(tmp_path, fake_season, capsys):
    fake_season.drop(columns=["raw_score"]).to_parquet(tmp_path / "raw_test.parquet", index=False)
    assert run_cli(tmp_path) == 1
    assert "raw_score" in capsys.readouterr().err
    assert [p.name for p in tmp_path.iterdir()] == ["raw_test.parquet"]


@pytest.mark.parametrize("bad", ["../escape", "a b", "x/y", "", "ba.seline"])
def test_cli_rejects_method_names_that_are_not_plain_identifiers(tmp_path, bad):
    with pytest.raises(SystemExit) as exc:
        main(["--method", bad, "--processed", str(tmp_path)])
    assert exc.value.code == 2


def test_cli_requires_a_method():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_cli_runs_as_a_module_from_the_repo_root(processed):
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "src.scale.apply", "--method", "test", "--processed", str(processed)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "rows rated" in result.stdout
    assert (processed / "ratings_test.parquet").is_file() and (processed / "scaler_test.json").is_file()
