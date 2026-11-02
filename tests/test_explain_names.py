"""Surnames in the template explanations follow the page's shortName() rules (web/app.js)."""
import re

import pytest

from src.explain.llm import NAME_PARTICLES, NAME_SUFFIXES, TemplateProvider, short_name
from tests.conftest import ROOT


@pytest.mark.parametrize("full, short", [
    ("Kevin De Bruyne", "De Bruyne"),
    ("Virgil van Dijk", "van Dijk"),
    ("David de Gea", "de Gea"),
    ("Dionatan do Nascimento Teixeira", "Teixeira"),
    ("Siem Stefan de Jong", "de Jong"),
    ("Sergio Agüero", "Agüero"),
    ("Kanté", "Kanté"),
    ("Van Persie", "Persie"),  # the first word is never treated as a particle, as on the page
    ("John Smith Jr.", "Smith"),  # suffixes are dropped, as on the page
    ("Synthetic Home 4", "Home 4"),
    ("", "?"),
])
def test_short_name_matches_the_page(full, short):
    assert short_name(full) == short


def _js_list(name: str) -> tuple[str, ...]:
    source = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    body = re.search(rf"const {name} = \[(.*?)\];", source, flags=re.S).group(1)
    return tuple(re.findall(r"'([^']*)'", body))


def test_particles_and_suffixes_are_the_page_lists():
    assert NAME_PARTICLES == _js_list("NAME_PARTICLES")
    assert NAME_SUFFIXES == _js_list("NAME_SUFFIXES")


def _synthetic_input(player_name: str, method: str) -> dict:
    """SYNTHETIC explanation input in the shape of src.explain.inputs.explanation_input."""
    if method == "baseline":
        pos = [{"stat": "assists", "label": "assists", "count": 2, "weight": 2.0, "value": 4.0}]
        neg = [{"stat": "possession_lost", "label": "possessions lost", "count": 3, "weight": -0.25, "value": -0.75}]
    else:
        pos = [{"action_id": 10, "label": "pass (success)", "minute": 31, "value": 0.21}]
        neg = [{"action_id": 11, "label": "bad touch (fail)", "minute": 64, "value": -0.12}]
    return {"method": method, "match_id": 1, "player_id": 2, "player_name": player_name, "team": "Synthetic FC",
            "opponent": "Synthetic United", "home_team": "Synthetic FC", "away_team": "Synthetic United", "home_score": 2,
            "away_score": 1, "position_group": "MID", "minutes": 90.0, "raw_score": 3.25, "rating": 8.1,
            "top_positive": pos, "top_negative": neg}


@pytest.mark.parametrize("method", ["baseline", "vaep"])
@pytest.mark.parametrize("full, possessive", [
    ("Kevin De Bruyne", "De Bruyne's 8.1"),
    ("Virgil van Dijk", "Van Dijk's 8.1"),  # capitalised because it opens the sentence
    ("David de Gea", "De Gea's 8.1"),
])
def test_template_uses_the_full_surname(method, full, possessive):
    text = TemplateProvider().explain(_synthetic_input(full, method))
    assert possessive in text
    for broken in ("Bruyne's", "Dijk's", "Gea's"):
        assert f" {broken}" not in text.replace(possessive, "")
