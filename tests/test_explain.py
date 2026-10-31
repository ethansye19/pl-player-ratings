"""Grounded explanations: built only from that player's top positive/negative items,
rendered by a pluggable provider, and every number in the text must appear in the input."""
import pytest

from tests.conftest import import_or_fail, load_parquet


def test_verify_numbers_accepts_only_numbers_present_in_the_input():
    numbers_in_text, verify_numbers = import_or_fail("src.explain.verify", "numbers_in_text", "verify_numbers")
    data = {"rating": 7.4, "minutes": 90.0, "top_positive": [{"value": 0.42, "minute": 57, "label": "shot"}]}
    assert numbers_in_text("3 shots, 0.42 xG, 90 min, rated 7.4") == [3.0, 0.42, 90.0, 7.4]
    assert verify_numbers("Rated 7.4 after a 0.42 action in minute 57 of 90.", data)
    assert not verify_numbers("Rated 7.4 after a 0.5 action.", data)
    assert not verify_numbers("He made 3 key passes.", data)
    assert verify_numbers("No numbers at all.", data)


def test_provider_is_selected_by_environment_variable(monkeypatch):
    get_provider, TemplateProvider = import_or_fail("src.explain.llm", "get_provider", "TemplateProvider")
    monkeypatch.delenv("RATING_LLM_PROVIDER", raising=False)
    assert isinstance(get_provider(), TemplateProvider), "template is the offline default"
    monkeypatch.setenv("RATING_LLM_PROVIDER", "template")
    assert isinstance(get_provider(), TemplateProvider)
    monkeypatch.setenv("RATING_LLM_PROVIDER", "nonsense")
    with pytest.raises(ValueError):
        get_provider()


def test_anthropic_provider_requires_an_explicit_api_key(monkeypatch):
    get_provider = import_or_fail("src.explain.llm", "get_provider")
    monkeypatch.setenv("RATING_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(Exception) as excinfo:
        get_provider()
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def _sample_players(season_ratings):
    """Top-rated, lowest-rated and a median player-match: three very different inputs."""
    ordered = season_ratings.sort_values(["rating", "raw_score"], kind="mergesort")
    picks = [ordered.iloc[-1], ordered.iloc[0], ordered.iloc[len(ordered) // 2]]
    return [(int(p["match_id"]), int(p["player_id"])) for p in picks]


@pytest.mark.data
def test_input_contains_only_that_players_items(method, season_ratings):
    explanation_input = import_or_fail("src.explain.inputs", "explanation_input")
    if method == "vaep":
        values = load_parquet(
            "action_values.parquet", "Run the VAEP step first.", columns=["game_id", "action_id", "player_id", "vaep_value"]
        )
    else:
        STATS, WEIGHTS = import_or_fail("src.baseline.weights", "STATS", "WEIGHTS")
        stats = load_parquet("baseline_stats.parquet", "Run the baseline step first.").set_index(["match_id", "player_id"])
    for match_id, player_id in _sample_players(season_ratings):
        data = explanation_input(match_id, player_id, method, k=5)
        assert data["match_id"] == match_id and data["player_id"] == player_id and data["method"] == method
        assert len(data["top_positive"]) <= 5 and len(data["top_negative"]) <= 5
        assert all(i["value"] > 0 for i in data["top_positive"])
        assert all(i["value"] < 0 for i in data["top_negative"])
        for item in data["top_positive"] + data["top_negative"]:
            if method == "vaep":
                own = values[(values["game_id"] == match_id) & (values["player_id"] == player_id)]
                hit = own[own["action_id"] == item["action_id"]]
                assert len(hit) == 1, f"action {item['action_id']} is not this player's in this match"
                assert abs(float(hit["vaep_value"].iloc[0]) - item["value"]) < 1e-6
            else:
                assert item["stat"] in STATS
                row = stats.loc[(match_id, player_id)]
                weight = float(WEIGHTS[row["position_group"]][item["stat"]].weight)
                assert abs(item["count"] - float(row[item["stat"]])) < 1e-9
                assert abs(item["value"] - weight * float(row[item["stat"]])) < 1e-9


@pytest.mark.data
def test_every_number_in_the_explanation_appears_in_the_input(method, season_ratings, monkeypatch):
    explain = import_or_fail("src.explain.explain", "explain")
    numbers_in_text, verify_numbers = import_or_fail("src.explain.verify", "numbers_in_text", "verify_numbers")
    monkeypatch.setenv("RATING_LLM_PROVIDER", "template")
    for match_id, player_id in _sample_players(season_ratings):
        result = explain(match_id, player_id, method)
        assert result.text.strip(), "an explanation must not be empty"
        assert numbers_in_text(result.text), "an explanation states at least the rating"
        assert verify_numbers(result.text, result.input), result.text
        assert result.verified is True
