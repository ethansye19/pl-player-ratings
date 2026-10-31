"""Baseline method: the 35 per-match stats and their position-specific weights.

``raw_score = sum(WEIGHTS[position_group][stat].weight * stat)`` for the stats listed in
that group's profile (a stat absent from a group's profile has weight 0 there). Scores are
later percentile-scaled WITHIN each position group, so only the ranking inside a group
matters; the absolute scale differs between groups on purpose.

Stat definitions (columns of ``data/processed/events.parquet``, statsbombpy naming; one
row per player who appeared, counted over that player's events in that match only)
-----------------------------------------------------------------------------------
Attacking
  goals               ``type == "Shot"`` and ``shot_outcome == "Goal"`` (penalties included; own goals are not shots).
  assists             ``type == "Pass"`` and ``pass_goal_assist``.
  xg                  sum of ``shot_statsbomb_xg`` over the player's shots with ``shot_type != "Penalty"`` (non-penalty xG).
  shots_on_target     ``type == "Shot"`` and ``shot_outcome in ("Goal", "Saved", "Saved to Post")``.
  key_passes          ``type == "Pass"`` and (``pass_shot_assist`` or ``pass_goal_assist``): passes that led directly to a shot, assists included.
  dribbles_completed  ``type == "Dribble"`` and ``dribble_outcome == "Complete"``.
  progressive_carries ``type == "Carry"`` starting outside the own defensive 40% (``location_x >= 48``) and moving the ball
                      at least 10 yards towards the opponent's goal (``carry_end_x - location_x >= 10``).
  penalties_won       ``type == "Foul Won"`` and ``foul_won_penalty``.
  penalties_missed    ``type == "Shot"``, ``shot_type == "Penalty"`` and ``shot_outcome != "Goal"``.
Possession
  passes_completed    ``type == "Pass"`` and ``pass_outcome`` null (StatsBomb leaves the outcome empty for a completed pass).
  passes_incomplete   ``type == "Pass"`` and ``pass_outcome in ("Incomplete", "Out", "Unknown", "Pass Offside")`` and not
                      ``pass_cross`` (a failed cross is an attacking attempt, not a giveaway; ``"Injury Clearance"`` is
                      neither completed nor incomplete).
  progressive_passes  completed pass (``pass_outcome`` null) with ``pass_type`` not in ("Throw-in", "Goal Kick", "Kick Off",
                      "Corner"), ``location_x >= 48`` and ``pass_end_x - location_x >= 10``.
  possession_lost     ``type in ("Dispossessed", "Miscontrol")``.
Defending
  tackles_won         ``type == "Duel"``, ``duel_type == "Tackle"`` and ``duel_outcome in ("Won", "Success In Play", "Success Out")``.
  dribbled_past       ``type == "Dribbled Past"`` (the defender beaten by a completed dribble).
  interceptions       ``type == "Interception"`` and ``interception_outcome in ("Won", "Success In Play", "Success Out")``.
  clearances          ``type == "Clearance"``.
  blocks              ``type == "Block"`` (shots and passes blocked).
  ball_recoveries     ``type == "Ball Recovery"`` and not ``ball_recovery_recovery_failure``.
  aerials_won         any of ``clearance_aerial_won``, ``pass_aerial_won``, ``shot_aerial_won``, ``miscontrol_aerial_won``
                      (StatsBomb marks the winner of an aerial duel on the event that follows it).
  aerials_lost        ``type == "Duel"`` and ``duel_type == "Aerial Lost"``.
  pressures           ``type == "Pressure"``.
Discipline and errors
  fouls_committed     ``type == "Foul Committed"``.
  fouls_won           ``type == "Foul Won"``.
  yellow_cards        ``foul_committed_card == "Yellow Card"`` or ``bad_behaviour_card == "Yellow Card"``.
  red_cards           ``foul_committed_card`` or ``bad_behaviour_card`` in ("Red Card", "Second Yellow").
  errors              ``type == "Error"`` (StatsBomb: a mistake that leads directly to a shot).
  own_goals           ``type == "Own Goal Against"`` (attributed to the player who scored it).
  penalties_conceded  ``type == "Foul Committed"`` and ``foul_committed_penalty``.
Team context while on the pitch (same match only; the on-pitch span is ``lineups.minute_on .. minute_off``,
event times come from ``period`` + ``timestamp`` with period lengths taken from that match's "Half End" events)
  goals_conceded      opponent goals while the player was on the pitch: opponent ``Shot`` with ``shot_outcome == "Goal"``
                      plus the own team's ``"Own Goal Against"`` events.
  clean_sheet         1 if ``goals_conceded == 0`` and the player played at least 60 minutes, else 0 (the FPL rule).
  xg_faced            sum of ``shot_statsbomb_xg`` over opponent shots taken while the player was on the pitch.
Goalkeeping (``type == "Goal Keeper"``)
  saves               ``goalkeeper_type in ("Shot Saved", "Shot Saved to Post", "Save", "Penalty Saved", "Penalty Saved to Post")``.
  penalties_saved     ``goalkeeper_type in ("Penalty Saved", "Penalty Saved to Post")`` (also counted in ``saves``).
  keeper_claims       successful claims, punches, sweeps and smothers: (``goalkeeper_type``, ``goalkeeper_outcome``) in
                      Collected/Success, Collected/Collected Twice, Punch/Punched out, Punch/In Play Safe,
                      Keeper Sweeper/Claim, Keeper Sweeper/Clear, Smother/Success In Play, Smother/Won.

How low activity is treated
---------------------------
The raw score is a pure weighted sum with no minutes term, so a player with no notable
events scores exactly 0. The weights are set so that routine volume (passes, pressures,
recoveries, carries) earns only a little, while every misplaced pass, lost duel, lost
possession and goal conceded while on the pitch costs something. Measured on 2015/16
(``reports/baseline.md``, section 2): a 10-20 minute substitute's median raw score sits at
the 27th-30th percentile of the outfield groups (lower-middle, not the bottom), a quiet
full-match starter (no goals, assists, cards, errors, own goals or penalties conceded) at
the 55th-61st, and 16-26% of each group's appearances score at or below 0. Below the
quiet players come those charged with goals conceded, errors, cards, own goals or conceded
penalties, which are the main ways to go clearly negative. Players under 10 minutes still
get a raw score; the scale module drops them.

Design of the weights
---------------------
* Match-deciding events dominate: a goal is worth roughly a full match of routine work in
  every group; assists, penalties won/saved, errors, own goals, red cards and conceded
  penalties are in the same league.
* Goalkeepers are judged on shot-stopping relative to the chances faced (``saves`` and
  ``xg_faced`` positive, ``goals_conceded`` negative, so conceding 1 from 2.5 xG beats
  conceding 1 from 0.3 xG), on claims/punches, distribution and the clean sheet.
* Defenders: defensive actions, aerials, clearances, clean sheet and goals conceded /
  xG faced (team context, negative). Midfielders: progression, key passes, recoveries,
  pressures, duels, with a smaller share of the team's defensive record. Forwards: goals,
  xG, shots on target, assists, key passes, dribbles; team defence barely matters to them.
* Negative events pull a score down explicitly rather than being modelled as "missing
  positives", so the explanation feature can name them.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.contract import POSITION_GROUPS

STATS: list[str] = [
    # attacking
    "goals",
    "assists",
    "xg",
    "shots_on_target",
    "key_passes",
    "dribbles_completed",
    "progressive_carries",
    "penalties_won",
    "penalties_missed",
    # possession
    "passes_completed",
    "passes_incomplete",
    "progressive_passes",
    "possession_lost",
    # defending
    "tackles_won",
    "dribbled_past",
    "interceptions",
    "clearances",
    "blocks",
    "ball_recoveries",
    "aerials_won",
    "aerials_lost",
    "pressures",
    # discipline and errors
    "fouls_committed",
    "fouls_won",
    "yellow_cards",
    "red_cards",
    "errors",
    "own_goals",
    "penalties_conceded",
    # team context while on the pitch
    "goals_conceded",
    "clean_sheet",
    "xg_faced",
    # goalkeeping
    "saves",
    "penalties_saved",
    "keeper_claims",
]

# Stats that are real-valued (everything else is an integer count or a 0/1 flag).
FLOAT_STATS: tuple[str, ...] = ("xg", "xg_faced")

# Short fan-facing labels for the explanation feature and the reports.
LABELS: dict[str, str] = {
    "goals": "goals",
    "assists": "assists",
    "xg": "non-penalty xG",
    "shots_on_target": "shots on target",
    "key_passes": "key passes",
    "dribbles_completed": "dribbles completed",
    "progressive_carries": "progressive carries",
    "penalties_won": "penalties won",
    "penalties_missed": "penalties missed",
    "passes_completed": "passes completed",
    "passes_incomplete": "passes incomplete",
    "progressive_passes": "progressive passes",
    "possession_lost": "possession lost",
    "tackles_won": "tackles won",
    "dribbled_past": "dribbled past",
    "interceptions": "interceptions",
    "clearances": "clearances",
    "blocks": "blocks",
    "ball_recoveries": "ball recoveries",
    "aerials_won": "aerial duels won",
    "aerials_lost": "aerial duels lost",
    "pressures": "pressures",
    "fouls_committed": "fouls committed",
    "fouls_won": "fouls won",
    "yellow_cards": "yellow cards",
    "red_cards": "red cards",
    "errors": "errors leading to a shot",
    "own_goals": "own goals",
    "penalties_conceded": "penalties conceded",
    "goals_conceded": "goals conceded while on the pitch",
    "clean_sheet": "clean sheet",
    "xg_faced": "xG faced while on the pitch",
    "saves": "saves",
    "penalties_saved": "penalties saved",
    "keeper_claims": "claims, punches and sweeps",
}


@dataclass(frozen=True)
class Weight:
    """A weight and its one-line justification."""

    weight: float
    why: str


def _w(weight: float, why: str) -> Weight:
    return Weight(float(weight), why)


# ----------------------------------------------------------------------------- GK
_GK: dict[str, Weight] = {
    "saves": _w(0.50, "Shot-stopping is the keeper's core job; each save roughly cancels the typical goal threat it stopped"),
    "penalties_saved": _w(2.00, "A penalty save is a match-swinging moment worth a whole match of routine keeping"),
    "keeper_claims": _w(0.25, "Successful claims, punches, sweeps and smothers defuse crosses and through balls"),
    "goals_conceded": _w(-1.00, "Every goal conceded counts against the keeper, scaled so xg_faced can offset it"),
    "xg_faced": _w(0.80, "Credit for the quality of chances faced: conceding 1 from 2.5 xG beats conceding 1 from 0.3 xG"),
    "clean_sheet": _w(1.50, "A clean sheet (60+ minutes, nothing conceded) is the headline keeper achievement"),
    "passes_completed": _w(0.03, "Distribution: completed passes build play from the back"),
    "passes_incomplete": _w(-0.03, "Misplaced kicks hand possession straight back"),
    "progressive_passes": _w(0.06, "Accurate long distribution that moves the team up the pitch"),
    "possession_lost": _w(-0.50, "A keeper dispossessed or miscontrolling is usually a chance for the opponent"),
    "dribbled_past": _w(-0.50, "A keeper rounded by an attacker almost always concedes"),
    "errors": _w(-2.00, "A keeper error leading to a shot is the most visible mistake in the game"),
    "own_goals": _w(-3.00, "An own goal is a goal for the opponent credited to the player"),
    "penalties_conceded": _w(-2.00, "Conceding a penalty gifts the opponent a 0.78 xG chance"),
    "yellow_cards": _w(-0.50, "A booking is a discipline lapse and risks a sending-off"),
    "red_cards": _w(-3.00, "A sending-off leaves the team a man down; one of the worst individual outcomes"),
    "fouls_committed": _w(-0.20, "Keeper fouls are rare and dangerous (usually in or near the box)"),
    "tackles_won": _w(0.20, "A keeper winning the ball in a challenge is a sweeping intervention"),
    "interceptions": _w(0.20, "Reading a through ball counts as a sweeping intervention"),
    "clearances": _w(0.10, "Clearing danger from the box"),
    "aerials_won": _w(0.20, "Winning aerial duels under crosses is part of commanding the box"),
    "aerials_lost": _w(-0.30, "A keeper losing an aerial duel usually leaves an open goal"),
    "goals": _w(4.00, "A goalkeeper goal is extraordinary and match-defining"),
    "assists": _w(2.00, "A keeper assist (usually a long kick) directly creates a goal"),
    "key_passes": _w(0.30, "A pass that leads straight to a shot"),
    "shots_on_target": _w(0.30, "A keeper shot on target is a rare attacking contribution"),
    "xg": _w(1.00, "Chance quality of any shots taken"),
}

# ----------------------------------------------------------------------------- DEF
_DEF: dict[str, Weight] = {
    "goals": _w(3.50, "A goal from a defender is rare and usually decides the match"),
    "assists": _w(2.00, "An assist directly creates a goal"),
    "xg": _w(1.00, "Getting into scoring positions (set pieces) is rewarded even without a goal"),
    "shots_on_target": _w(0.30, "Testing the keeper from a set piece or a late run"),
    "key_passes": _w(0.40, "Creating a shot, typically a cross or a cut-back from a full-back"),
    "dribbles_completed": _w(0.20, "Beating a man to progress play"),
    "progressive_carries": _w(0.12, "Carrying the ball out of defence moves the team up the pitch"),
    "penalties_won": _w(1.50, "Winning a penalty creates a 0.78 xG chance"),
    "penalties_missed": _w(-2.00, "Missing a penalty wastes the best chance in football"),
    "passes_completed": _w(0.015, "Safe passing from the back earns a little; volume alone should not make a rating"),
    "passes_incomplete": _w(-0.05, "Giveaways from a defender concede territory and counters"),
    "progressive_passes": _w(0.10, "Forward passing that breaks lines from defence"),
    "possession_lost": _w(-0.25, "Being dispossessed or miscontrolling in defensive areas is dangerous"),
    "tackles_won": _w(0.30, "Winning tackles is a core defensive action"),
    "dribbled_past": _w(-0.25, "Being beaten by a dribble is a defender's most visible failure"),
    "interceptions": _w(0.30, "Reading the game and cutting out passes"),
    "clearances": _w(0.10, "Clearing danger from the box; high volume so a small weight"),
    "blocks": _w(0.20, "Blocking shots and passes with the body"),
    "ball_recoveries": _w(0.06, "Picking up loose balls restores possession"),
    "aerials_won": _w(0.20, "Dominance in the air is central to defending"),
    "aerials_lost": _w(-0.15, "Losing aerial duels lets attackers win first contact"),
    "pressures": _w(0.015, "Pressing work rate; small weight because pressures are frequent"),
    "fouls_committed": _w(-0.15, "Fouls give away free kicks in dangerous areas"),
    "fouls_won": _w(0.10, "Drawing fouls relieves pressure"),
    "yellow_cards": _w(-0.50, "A booking is a discipline lapse and risks a sending-off"),
    "red_cards": _w(-3.00, "A sending-off leaves the team a man down; one of the worst individual outcomes"),
    "errors": _w(-1.50, "An error leading to a shot is a defender's clearest mistake"),
    "own_goals": _w(-3.00, "An own goal is a goal for the opponent credited to the player"),
    "penalties_conceded": _w(-2.00, "Conceding a penalty gifts the opponent a 0.78 xG chance"),
    "goals_conceded": _w(-0.80, "Defenders share responsibility for every goal conceded while on the pitch"),
    "clean_sheet": _w(1.00, "Keeping a clean sheet (60+ minutes) is the defender's headline achievement"),
    "xg_faced": _w(-0.40, "Allowing many good chances is poor defending even when the keeper bails you out"),
}

# ----------------------------------------------------------------------------- MID
_MID: dict[str, Weight] = {
    "goals": _w(3.00, "Goals decide matches; a midfield goal is a major contribution"),
    "assists": _w(2.00, "An assist directly creates a goal"),
    "xg": _w(1.50, "Getting into scoring positions is rewarded even without a goal"),
    "shots_on_target": _w(0.40, "Shots that test the keeper"),
    "key_passes": _w(0.50, "Chance creation is a midfielder's main attacking output"),
    "dribbles_completed": _w(0.25, "Beating a man in midfield opens up the pitch"),
    "progressive_carries": _w(0.12, "Carrying the ball forward through the lines"),
    "penalties_won": _w(1.50, "Winning a penalty creates a 0.78 xG chance"),
    "penalties_missed": _w(-2.00, "Missing a penalty wastes the best chance in football"),
    "passes_completed": _w(0.01, "Accurate passing keeps the team ticking; kept small so volume alone does not make a rating"),
    "passes_incomplete": _w(-0.06, "Giveaways in midfield turn possession over"),
    "progressive_passes": _w(0.10, "Line-breaking passes are the core of midfield progression"),
    "possession_lost": _w(-0.25, "Being dispossessed or miscontrolling in midfield starts counters"),
    "tackles_won": _w(0.30, "Winning the ball back in midfield duels"),
    "dribbled_past": _w(-0.20, "Being beaten by a dribble lets the opponent through the middle"),
    "interceptions": _w(0.25, "Reading the game and cutting out passes"),
    "clearances": _w(0.10, "Clearing danger when defending set pieces"),
    "blocks": _w(0.15, "Blocking shots and passes with the body"),
    "ball_recoveries": _w(0.06, "Winning loose balls is the engine-room contribution"),
    "aerials_won": _w(0.15, "Winning second balls in the air"),
    "aerials_lost": _w(-0.10, "Losing aerial duels concedes second balls"),
    "pressures": _w(0.015, "Pressing work rate; frequent, so a small weight"),
    "fouls_committed": _w(-0.15, "Fouls give away free kicks and risk cards"),
    "fouls_won": _w(0.10, "Drawing fouls wins territory and relieves pressure"),
    "yellow_cards": _w(-0.50, "A booking is a discipline lapse and risks a sending-off"),
    "red_cards": _w(-3.00, "A sending-off leaves the team a man down; one of the worst individual outcomes"),
    "errors": _w(-1.50, "An error leading to a shot is a clear individual mistake"),
    "own_goals": _w(-3.00, "An own goal is a goal for the opponent credited to the player"),
    "penalties_conceded": _w(-2.00, "Conceding a penalty gifts the opponent a 0.78 xG chance"),
    "goals_conceded": _w(-0.60, "Midfielders share responsibility for goals conceded while on the pitch, less than defenders"),
    "clean_sheet": _w(0.50, "A clean sheet reflects the midfield's defensive shield"),
    "xg_faced": _w(-0.20, "A share of responsibility for the chances the team allows"),
}

# ----------------------------------------------------------------------------- FWD
_FWD: dict[str, Weight] = {
    "goals": _w(3.00, "Scoring is the forward's job and decides matches"),
    "assists": _w(1.80, "An assist directly creates a goal"),
    "xg": _w(1.50, "Getting into high-quality shooting positions is rewarded even without a goal"),
    "shots_on_target": _w(0.40, "Shots that test the keeper show finishing quality"),
    "key_passes": _w(0.40, "Creating chances for teammates"),
    "dribbles_completed": _w(0.30, "Beating defenders one-on-one is a forward's trademark"),
    "progressive_carries": _w(0.12, "Carrying the ball at the defence"),
    "penalties_won": _w(1.50, "Winning a penalty creates a 0.78 xG chance"),
    "penalties_missed": _w(-2.00, "Missing a penalty wastes the best chance in football"),
    "passes_completed": _w(0.01, "Link-up passing earns a little; forwards are not rated on pass volume"),
    "passes_incomplete": _w(-0.05, "Giveaways in the final third are cheap but still turnovers"),
    "progressive_passes": _w(0.08, "Forward passing that moves the attack on"),
    "possession_lost": _w(-0.20, "Being dispossessed or miscontrolling up front is common but still a turnover"),
    "tackles_won": _w(0.20, "Winning the ball high up the pitch"),
    "dribbled_past": _w(-0.10, "A forward beaten by a dribble is a minor defensive lapse"),
    "interceptions": _w(0.15, "Cutting out passes high up the pitch"),
    "clearances": _w(0.05, "Helping out at defensive set pieces"),
    "blocks": _w(0.10, "Blocking passes and shots while pressing"),
    "ball_recoveries": _w(0.05, "Winning loose balls to sustain attacks"),
    "aerials_won": _w(0.15, "Winning headers holds the ball up and threatens goal"),
    "aerials_lost": _w(-0.10, "Losing aerial duels up front gives the ball straight back"),
    "pressures": _w(0.015, "Pressing from the front; frequent, so a small weight"),
    "fouls_committed": _w(-0.15, "Fouls break up the team's own attacks and risk cards"),
    "fouls_won": _w(0.10, "Drawing fouls earns free kicks in dangerous areas"),
    "yellow_cards": _w(-0.50, "A booking is a discipline lapse and risks a sending-off"),
    "red_cards": _w(-3.00, "A sending-off leaves the team a man down; one of the worst individual outcomes"),
    "errors": _w(-1.50, "An error leading to a shot is a clear individual mistake"),
    "own_goals": _w(-3.00, "An own goal is a goal for the opponent credited to the player"),
    "penalties_conceded": _w(-2.00, "Conceding a penalty gifts the opponent a 0.78 xG chance"),
    "goals_conceded": _w(-0.20, "Forwards carry only a token share of goals conceded while on the pitch"),
}

WEIGHTS: dict[str, dict[str, Weight]] = {"GK": _GK, "DEF": _DEF, "MID": _MID, "FWD": _FWD}

# --- consistency checks at import time (cheap, and they keep the tables honest) ---------
assert len(STATS) == len(set(STATS)), "STATS must be unique"
assert set(WEIGHTS) == set(POSITION_GROUPS), "WEIGHTS must cover exactly the contract position groups"
assert set(LABELS) == set(STATS), "LABELS must cover exactly STATS"
for _group, _profile in WEIGHTS.items():
    _unknown = set(_profile) - set(STATS)
    assert not _unknown, f"{_group}: unknown stats {_unknown}"


def weight_of(group: str, stat: str) -> float:
    """Weight of ``stat`` for ``group`` (0.0 when the stat is not in that group's profile)."""
    entry = WEIGHTS[group].get(stat)
    return 0.0 if entry is None else float(entry.weight)


def stat_definitions() -> dict[str, str]:
    """Parse the stat definitions out of this module's docstring: {stat: one-line definition}.

    The docstring is the single source of truth; this keeps reports in sync with it.
    """
    block = __doc__.split("Stat definitions", 1)[1].split("How low activity is treated", 1)[0]
    out: dict[str, str] = {}
    current = None
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or not line.startswith("  "):
            continue
        head = stripped.split(" ", 1)[0]
        if head in STATS and line.startswith("  ") and not line.startswith("    "):
            current = head
            out[current] = stripped[len(head):].strip()
        elif current is not None and line.startswith("    "):
            out[current] = out[current] + " " + stripped
    missing = [stat for stat in STATS if stat not in out]
    assert not missing, f"docstring lacks definitions for {missing}"
    return out
