"""Explanation providers, selected by the RATING_LLM_PROVIDER environment variable.

- ``template`` (default): deterministic sentence templates filled from the input; no
  network, no key. Every number it writes is a value from the input, formatted so the
  grounding check in ``src.explain.verify`` passes.
- ``anthropic``: Claude through the official SDK. Requires ``ANTHROPIC_API_KEY`` (never
  use a key without the owner's approval) and the ``anthropic`` package. Model from
  ``RATING_LLM_MODEL`` (default ``claude-opus-5-5``). The prompt forbids anything that is
  not in the input, and ``src.explain.explain`` still verifies the output.
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol

from src.baseline.weights import FLOAT_STATS

DEFAULT_MODEL = "claude-opus-5-5"
METHOD_LABEL = {"baseline": "baseline", "vaep": "VAEP"}

SYSTEM_PROMPT = """You explain a football player's match rating to a fan in two to four plain sentences.
You receive a JSON object with the player, the match, the rating and the player's top positive and
negative items (stat contributions for the baseline method, individual actions for VAEP).
Rules, without exception:
- Mention only facts, numbers and events that appear in the JSON. Do not add context, opinions about
  the match, or anything you know about the player or the teams from elsewhere.
- Every number you write must be copied from the JSON (you may round to two decimals). Do not compute
  new numbers, totals, averages or percentages.
- Name the rating and the method, then the main positives, then the main negatives, in prose.
- If the rating is null, say the player was not rated because he played under 10 minutes.
Write plain text, no lists, no markdown."""


class Provider(Protocol):
    name: str

    def explain(self, data: dict[str, Any]) -> str: ...


# stat -> (singular, plural) wording; stats missing here fall back to the LABELS text
STAT_WORDS = {
    "goals": ("non-penalty goal", "non-penalty goals"),
    "penalties_scored": ("penalty scored", "penalties scored"),
    "assists": ("assist", "assists"),
    "shots_on_target": ("shot on target", "shots on target"),
    "key_passes": ("key pass", "key passes"),
    "dribbles_completed": ("completed dribble", "completed dribbles"),
    "progressive_carries": ("progressive carry", "progressive carries"),
    "penalties_won": ("penalty won", "penalties won"),
    "penalties_missed": ("penalty missed", "penalties missed"),
    "passes_completed": ("completed pass", "completed passes"),
    "passes_incomplete": ("incomplete pass", "incomplete passes"),
    "progressive_passes": ("progressive pass", "progressive passes"),
    "possession_lost": ("possession lost", "possessions lost"),
    "tackles_won": ("tackle won", "tackles won"),
    "dribbled_past": ("time dribbled past", "times dribbled past"),
    "interceptions": ("interception", "interceptions"),
    "clearances": ("clearance", "clearances"),
    "blocks": ("block", "blocks"),
    "ball_recoveries": ("ball recovery", "ball recoveries"),
    "aerials_won": ("aerial duel won", "aerial duels won"),
    "aerials_lost": ("aerial duel lost", "aerial duels lost"),
    "pressures": ("pressure", "pressures"),
    "fouls_committed": ("foul committed", "fouls committed"),
    "yellow_cards": ("yellow card", "yellow cards"),
    "red_cards": ("red card", "red cards"),
    "errors": ("error leading to a shot", "errors leading to a shot"),
    "own_goals": ("own goal", "own goals"),
    "penalties_conceded": ("penalty conceded", "penalties conceded"),
    "goals_conceded": ("goal conceded while on the pitch", "goals conceded while on the pitch"),
    "clean_sheet": ("clean sheet", "clean sheets"),
    "saves": ("save", "saves"),
    "penalties_saved": ("penalty saved", "penalties saved"),
    "keeper_claims": ("claim, punch or sweep", "claims, punches and sweeps"),
}
ACTION_WORDS = {
    "pass": "pass", "cross": "cross", "throw_in": "throw-in", "freekick_crossed": "crossed free kick",
    "freekick_short": "short free kick", "corner_crossed": "crossed corner", "corner_short": "short corner",
    "take_on": "take-on", "foul": "foul", "tackle": "tackle", "interception": "interception", "shot": "shot",
    "shot_penalty": "penalty", "shot_freekick": "free-kick shot", "keeper_save": "save", "keeper_claim": "claim",
    "keeper_punch": "punch", "clearance": "clearance", "bad_touch": "bad touch", "dribble": "carry",
    "goalkick": "goal kick",
}
RESULT_WORDS = {"success": "successful", "fail": "unsuccessful", "offside": "offside", "owngoal": "own-goal",
                "yellow_card": "yellow-card", "red_card": "red-card"}


def _count_phrase(item: dict[str, Any]) -> str:
    if item["stat"] in FLOAT_STATS:
        return f"{item['label']} of {item['count']:.2f}"
    count = item["count"]
    singular, plural = STAT_WORDS.get(item["stat"], (item["label"], item["label"]))
    return f"{count} {singular if count == 1 else plural}"


def _action_phrase(item: dict[str, Any]) -> str:
    """'a successful pass in minute 48' from an action label like 'pass (success)'."""
    label = item["label"]
    kind, _, result = label.partition(" (")
    result = result.rstrip(")")
    noun = ACTION_WORDS.get(kind.replace(" ", "_"), kind)
    if kind.replace(" ", "_") == "shot" and result == "success":
        noun, result = "goal", ""
    adjective = RESULT_WORDS.get(result, "") if kind.replace(" ", "_") not in ("bad_touch", "foul") else ""
    words = f"{adjective} {noun}".strip()
    article = "an" if words[0].lower() in "aeiou" else "a"
    return f"{article} {words} in minute {item['minute']}"


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


class TemplateProvider:
    name = "template"

    def explain(self, data: dict[str, Any]) -> str:
        method = METHOD_LABEL.get(data["method"], data["method"])
        score = f"{data['home_team']} {data['home_score']}-{data['away_score']} {data['away_team']}"
        who = f"{data['player_name']} ({data['team']}, {data['position_group']})"
        if data["rating"] is None:
            opening = (
                f"{who} was not rated with the {method} method in {score}: he played "
                f"{data['minutes']:.1f} minutes, under the 10-minute threshold."
            )
        else:
            opening = (
                f"{who} rated {data['rating']:.1f} with the {method} method in {score}, "
                f"playing {data['minutes']:.1f} minutes."
            )
        surname = data["player_name"].split()[-1]
        top_pos, top_neg = data["top_positive"][:3], data["top_negative"][:2]
        if data["method"] == "baseline":
            pos = [f"{_count_phrase(i)} ({i['value']:+.2f})" for i in top_pos]
            neg = [f"{_count_phrase(i)} ({i['value']:+.2f})" for i in top_neg]
            if data["rating"] is not None and pos:
                positives = f"{surname}'s {data['rating']:.1f} came mostly from {_join(pos)}."
            elif pos:
                positives = f"His best contributions were {_join(pos)}."
            else:
                positives = "No stat counted in his favour."
            negatives = f"Against that, {_join(neg)} pulled the score down." if neg else "Nothing counted against him."
        else:
            pos = [f"{_action_phrase(i)} ({i['value']:+.2f})" for i in top_pos]
            neg = [f"{_action_phrase(i)} ({i['value']:+.2f})" for i in top_neg]
            if data["rating"] is not None and pos:
                positives = f"{surname}'s {data['rating']:.1f} rests on {_join(pos)}."
            elif pos:
                positives = f"His most valuable actions were {_join(pos)}."
            else:
                positives = "None of his actions added value."
            negatives = f"The costliest moments were {_join(neg)}." if neg else "No action cost value."
        closing = "" if data["raw_score"] is None else f" Raw score {data['raw_score']:+.2f}."
        return f"{opening} {positives} {negatives}{closing}"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set: the anthropic provider needs an API key "
                "(set RATING_LLM_PROVIDER=template to run without one)"
            )
        self.model = model or os.environ.get("RATING_LLM_MODEL") or DEFAULT_MODEL

    def explain(self, data: dict[str, Any]) -> str:
        try:
            import anthropic
        except ImportError as exc:  # the package is not part of the pinned environment
            raise RuntimeError("the 'anthropic' package is not installed (pip install anthropic)") from exc
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined to answer (stop_reason=refusal)")
        return "".join(block.text for block in response.content if block.type == "text").strip()


def get_provider() -> Provider:
    name = os.environ.get("RATING_LLM_PROVIDER", "template").strip().lower()
    if name == "template":
        return TemplateProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"unknown RATING_LLM_PROVIDER {name!r}; expected 'template' or 'anthropic'")
