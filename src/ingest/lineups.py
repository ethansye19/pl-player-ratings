"""Per-match lineups with minutes played and position groups, from the raw StatsBomb
lineup and event JSON of ONE match.

Time on the pitch is derived from the EVENTS, which are internally consistent:
``Starting XI`` (on at kick-off), ``Substitution`` (one player off, the replacement on),
``Player Off`` / ``Player On`` (treatment breaks; ``permanent`` means no return) and a
Red Card / Second Yellow (off for good). Period lengths come from the ``Half End``
timestamps, so stoppage time counts to the second.

The lineup file's ``positions`` intervals are used only to decide WHICH position a
player spent the most time in, after clipping them to the on-pitch spans. They are not
reliable for time: they do not stop at a dismissal, and when a treatment break or a
tactical shift happens close to a half-time substitution the last interval runs to the
final whistle even though the player went off (e.g. matches 3754339, 3754217, 3754336).
Every such disagreement of more than one minute is reported as a note.

Clock conventions (verified on the raw data): lineup intervals use the match clock
(second half starts at 45:00); event timestamps restart at 00:00:00 every period.

Minutes are rounded to one decimal; anyone who stepped on the pitch gets at least 0.1.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from src.ingest.positions import position_group

log = logging.getLogger(__name__)

PERIOD_CLOCK_BASE = {1: 0.0, 2: 45 * 60.0, 3: 90 * 60.0, 4: 105 * 60.0}
RED_CARDS = {"Red Card", "Second Yellow"}
Span = tuple[float, float]


def parse_clock(mmss: str) -> float:
    """'67:54' -> seconds on the match clock."""
    minutes, seconds = mmss.split(":")
    return int(minutes) * 60 + float(seconds)


def parse_timestamp(hhmmss: str) -> float:
    """'00:46:07.805' -> seconds since the start of the period."""
    hours, minutes, seconds = hhmmss.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def period_lengths(events: list[dict[str, Any]]) -> dict[int, float]:
    """Real length in seconds of every regular period, from the Half End events."""
    lengths: dict[int, float] = {}
    for event in events:
        if event["type"]["name"] == "Half End" and event["period"] in PERIOD_CLOCK_BASE:
            period = int(event["period"])
            lengths[period] = max(lengths.get(period, 0.0), parse_timestamp(event["timestamp"]))
    if not lengths:
        raise ValueError("no Half End events: cannot compute period lengths")
    return lengths


def period_offsets(lengths: dict[int, float]) -> dict[int, float]:
    return {p: sum(length for q, length in lengths.items() if q < p) for p in lengths}


def starting_xi(events: list[dict[str, Any]]) -> dict[int, dict[int, int]]:
    """team_id -> {player_id: position_id} from the Starting XI events."""
    out: dict[int, dict[int, int]] = {}
    for event in events:
        if event["type"]["name"] == "Starting XI":
            out[int(event["team"]["id"])] = {
                int(p["player"]["id"]): int(p["position"]["id"]) for p in event["tactics"]["lineup"]
            }
    return out


def absolute_seconds(clock: str, period: int, lengths: dict[int, float]) -> float:
    """Convert a lineup match-clock time within a period to seconds since kick-off."""
    period = int(period)
    if period not in lengths:
        raise ValueError(f"period {period} has no Half End event")
    within = parse_clock(clock) - PERIOD_CLOCK_BASE[period]
    within = min(max(within, 0.0), lengths[period])  # clamp to the period's real length
    return period_offsets(lengths)[period] + within


def _red_card(event: dict[str, Any]) -> bool:
    kind = event["type"]["name"]
    if kind == "Foul Committed":
        return (event.get("foul_committed") or {}).get("card", {}).get("name") in RED_CARDS
    if kind == "Bad Behaviour":
        return (event.get("bad_behaviour") or {}).get("card", {}).get("name") in RED_CARDS
    return False


def on_pitch_spans(events: list[dict[str, Any]], lengths: dict[int, float]) -> tuple[dict[int, list[Span]], dict[int, float]]:
    """player_id -> on-pitch spans in seconds since kick-off; player_id -> dismissal time."""
    offsets = period_offsets(lengths)
    total = sum(lengths.values())

    def when(event: dict[str, Any]) -> float:
        period = int(event["period"])
        return offsets[period] + min(parse_timestamp(event["timestamp"]), lengths[period])

    ons: dict[int, list[float]] = defaultdict(list)
    offs: dict[int, list[float]] = defaultdict(list)
    dismissed: dict[int, float] = {}
    for team_xi in starting_xi(events).values():
        for player_id in team_xi:
            ons[player_id].append(0.0)
    for event in sorted(events, key=lambda e: e["index"]):
        kind = event["type"]["name"]
        if int(event["period"]) not in lengths:
            continue
        if kind == "Substitution":
            offs[int(event["player"]["id"])].append(when(event))
            ons[int(event["substitution"]["replacement"]["id"])].append(when(event))
        elif kind == "Player Off":
            offs[int(event["player"]["id"])].append(when(event))
        elif kind == "Player On":
            ons[int(event["player"]["id"])].append(when(event))
        elif _red_card(event) and event.get("player"):
            player_id = int(event["player"]["id"])
            dismissed[player_id] = min(dismissed.get(player_id, when(event)), when(event))

    spans: dict[int, list[Span]] = {}
    for player_id in set(ons) | set(offs):
        marks = sorted([(t, 0) for t in ons[player_id]] + [(t, 1) for t in offs[player_id]])
        current: float | None = None
        out: list[Span] = []
        for time, kind in marks:
            if kind == 0 and current is None:
                current = time
            elif kind == 1 and current is not None:
                out.append((current, time))
                current = None
        if current is not None:
            out.append((current, total))
        if player_id in dismissed:
            cap = dismissed[player_id]
            out = [(start, min(end, cap)) for start, end in out if start < cap]
        spans[player_id] = out
    return spans, dismissed


def _overlap(a: Span, b: Span) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def build_match_lineups(
    match_id: int, lineups_json: list[dict[str, Any]], events_json: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return (rows, anomalies, notes) for every player who appeared in the match.

    anomalies: things that should never happen (fail the build's expectations);
    notes: lineup-interval inconsistencies that were corrected from the events.
    """
    lengths = period_lengths(events_json)
    xi_by_team = starting_xi(events_json)
    spans_by_player, dismissed = on_pitch_spans(events_json, lengths)
    rows: list[dict[str, Any]] = []
    anomalies: list[str] = []
    notes: list[str] = []
    seen: set[int] = set()

    for team in lineups_json:
        team_id = int(team["team_id"])
        team_xi = xi_by_team.get(team_id, {})
        if len(team_xi) != 11:
            anomalies.append(f"match {match_id}: {team['team_name']} Starting XI event lists {len(team_xi)} players")
        starters_seen = 0
        for player in team["lineup"]:
            player_id = int(player["player_id"])
            name = player.get("player_nickname") or player["player_name"]
            spans = spans_by_player.get(player_id, [])
            intervals = player.get("positions") or []
            seen.add(player_id)
            if not spans:
                if intervals:
                    anomalies.append(f"match {match_id}: {name} ({team['team_name']}) has lineup positions but never enters play in the events")
                continue  # unused substitute
            seconds = sum(end - start for start, end in spans)
            if seconds <= 0:
                anomalies.append(f"match {match_id}: {name} ({team['team_name']}) entered play for zero seconds; dropped")
                continue

            # position with the most on-pitch time, from lineup intervals clipped to the spans
            time_by_position: dict[int, float] = {}
            names_by_position: dict[int, str] = {}
            order: list[int] = []
            lineup_seconds = 0.0
            for q in intervals:
                start = absolute_seconds(q["from"], q["from_period"], lengths)
                end = sum(lengths.values()) if q["to"] is None else absolute_seconds(q["to"], q["to_period"], lengths)
                lineup_seconds += max(0.0, end - start)
                pid = int(q["position_id"])
                names_by_position.setdefault(pid, q["position"])
                if pid not in order:
                    order.append(pid)
                time_by_position[pid] = time_by_position.get(pid, 0.0) + sum(_overlap((start, end), s) for s in spans)
            if order and max(time_by_position.values()) > 0:
                main_pid = max(order, key=lambda pid: (time_by_position[pid], -order.index(pid)))
            elif order:
                main_pid = order[0]
            else:
                anomalies.append(f"match {match_id}: {name} ({team['team_name']}) played but has no lineup position intervals")
                continue
            if abs(lineup_seconds - seconds) > 60:
                notes.append(
                    f"match {match_id}: {name} ({team['team_name']}) lineup intervals say {lineup_seconds / 60:.1f} min, "
                    f"events say {seconds / 60:.1f} min; events used"
                )

            is_starter = player_id in team_xi
            starters_seen += int(is_starter)
            rows.append(
                {
                    "match_id": int(match_id),
                    "team_id": team_id,
                    "team": team["team_name"],
                    "player_id": player_id,
                    "player_name": name,
                    "player_full_name": player["player_name"],
                    "jersey_number": int(player["jersey_number"]),
                    "is_starter": bool(is_starter),
                    "starting_position_id": int(team_xi.get(player_id, main_pid)),
                    "position_id": main_pid,
                    "position": names_by_position[main_pid],
                    "position_group": position_group(main_pid),
                    "minutes": max(0.1, round(seconds / 60.0, 1)),
                    "minute_on": round(spans[0][0] / 60.0, 1),
                    "minute_off": round(spans[-1][1] / 60.0, 1),
                    "red_card_minute": round(dismissed[player_id] / 60.0, 1) if player_id in dismissed else None,
                }
            )
        if starters_seen != 11:
            anomalies.append(f"match {match_id}: {team['team_name']} has {starters_seen} starters who entered play")

    for player_id, spans in spans_by_player.items():
        if spans and player_id not in seen:
            anomalies.append(f"match {match_id}: player {player_id} appears in the events but not in the lineup file")
    return rows, anomalies, notes
