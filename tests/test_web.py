"""Web page contract: colour bands, the per-match JSON payload the page loads, and the
StatsBomb credit."""
import json

import pytest

from src.contract import MIN_MINUTES, RATING_MAX, RATING_MIN
from tests.conftest import FIXTURES, ROOT, import_or_fail


@pytest.fixture(scope="module")
def sample_payload() -> dict:
    return json.loads((FIXTURES / "sample_match.json").read_text(encoding="utf-8"))


def test_rating_band_thresholds():
    rating_band = import_or_fail("src.scale.bands", "rating_band")
    assert rating_band(3.0) == "red"
    assert rating_band(5.9) == "red"
    assert rating_band(6.0) == "orange"
    assert rating_band(6.9) == "orange"
    assert rating_band(7.0) == "green"
    assert rating_band(7.9) == "green"
    assert rating_band(8.0) == "blue"
    assert rating_band(10.0) == "blue"
    assert rating_band(None) == "unrated"


def test_sample_payload_is_valid(sample_payload):
    validate_payload = import_or_fail("src.pipeline.export_web", "validate_payload")
    validate_payload(sample_payload)


def test_sample_payload_has_both_methods_and_full_lineups(sample_payload):
    assert sample_payload["methods"] == ["baseline", "vaep"]
    for side in ("home", "away"):
        starters = [p for p in sample_payload["players"] if p["side"] == side and p["is_starter"]]
        assert len(starters) == 11
    ids = [p["player_id"] for p in sample_payload["players"]]
    assert len(ids) == len(set(ids))
    assert "StatsBomb" in sample_payload["credits"]


def test_payload_ratings_follow_the_contract_rules(sample_payload):
    for p in sample_payload["players"]:
        for m in sample_payload["methods"]:
            r = p["ratings"][m]
            if p["minutes"] < MIN_MINUTES:
                assert r is None, f"{p['player_name']} played under 10 minutes and must be unrated"
            else:
                assert RATING_MIN <= r <= RATING_MAX
                assert round(r, 1) == r


def test_validate_payload_rejects_a_rated_sub_10_minute_player(sample_payload):
    validate_payload = import_or_fail("src.pipeline.export_web", "validate_payload")
    bad = json.loads(json.dumps(sample_payload))
    short = next(p for p in bad["players"] if p["minutes"] < MIN_MINUTES)
    short["ratings"] = {m: 6.5 for m in bad["methods"]}
    with pytest.raises(ValueError):
        validate_payload(bad)


def test_web_page_credits_statsbomb_and_loads_the_app():
    index = ROOT / "web" / "index.html"
    assert index.exists(), "web/index.html is missing"
    html = index.read_text(encoding="utf-8")
    assert "StatsBomb" in html
    assert "app.js" in html


@pytest.mark.data
def test_exported_real_match_payload_is_valid(matches, lineups):
    match_payload, validate_payload = import_or_fail("src.pipeline.export_web", "match_payload", "validate_payload")
    match_id = int(matches["match_id"].iloc[0])
    payload = match_payload(match_id)
    validate_payload(payload)
    assert payload["methods"] == ["baseline", "vaep"]
    assert payload["source"] == "StatsBomb Open Data"
    assert len([p for p in payload["players"] if p["is_starter"]]) == 22
    assert {p["player_id"] for p in payload["players"]} == set(lineups.loc[lineups["match_id"] == match_id, "player_id"])
