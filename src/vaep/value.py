"""From predicted probabilities to action values: socceraction's VAEP formula plus two scoring rules.

Decision 2026-11-02 ("option 3", ``reports/validation.md`` section 8): the models, features and
xG labels are unchanged; only the step from the fold models' probabilities to the action values is.

socceraction's formula (``socceraction.vaep.formula.value``) values action *i* by the change from
the previous game state, seen from the acting team::

    offensive = P_scores(S_i)   - P_scores(S_{i-1})
    defensive = -(P_concedes(S_i) - P_concedes(S_{i-1}))
    vaep      = offensive + defensive

where the previous state's two probabilities swap when the ball changed team, both are 0 after a
goal or a gap of more than 10 seconds, and the previous P_scores is fixed at 0.792453 before a
penalty and 0.0465 before a corner. :func:`formula_value` reproduces it exactly.

Rule 1, penalties. The P_scores of every ``shot_penalty`` state is set to that penalty's StatsBomb
xG (0.7835, the value StatsBomb gives every penalty, when the xG is missing). With xG labels a
penalty state's target is its xG, but the models predicted 0.68 for scored and 0.50 for missed
penalties (91 states among 9,900 shots), so a converted penalty was worth -0.112 to the taker and
a miss -0.298. After the override the kick is worth xG - 0.792453, about -0.01, whatever happens.

Rule 2, after an unscored shot. The action that follows an unscored shot is measured against the
shot state's P_scores with that shot's own chance taken out,
``r = 1 - (1 - p) / (1 - xG)``, clipped to ``[0, p]``: the probability that the rest of the label
window still produces a goal. Without the rule the first defending action after a missed shot,
usually a clearance, inherited the whole chance (a clearance after a missed penalty was worth
+0.50). A ``keeper_save`` keeps the full drop, because the save is what stopped the chance. Shots
without an xG value are left as they are. Rule 2 uses the P_scores after rule 1.

Both rules read only the game's own actions, probabilities and shot xG, so the per-match rating
function stays isolated. With no shots in a game, :func:`value` equals socceraction's formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHOT_TYPES = ("shot", "shot_penalty", "shot_freekick")
PENALTY_PRIOR = 0.792453  # socceraction's fixed P_scores of the state before a penalty
CORNER_PRIOR = 0.046500  # socceraction's fixed P_scores of the state before a corner
SAMEPHASE_SECONDS = 10  # socceraction.vaep.formula._samephase_nb
PENALTY_XG_FALLBACK = 0.7835  # StatsBomb's xG for every penalty, used when a penalty has no xG value
VALUE_RULE = "option3"  # recorded in the scoring summary and the reports


def _prev(x: pd.Series, first: pd.Series | None = None) -> pd.Series:
    """The previous row's value; the first row is its own predecessor (as in socceraction).

    ``first`` supplies that first row's own value when ``x`` is a modified copy (the after-shot
    rule changes what the *next* action is measured against, not the first action itself).
    """
    prev_x = x.shift(1)
    prev_x.iloc[:1] = (x if first is None else first).iloc[:1].to_numpy()
    return prev_x


def formula_value(
    actions: pd.DataFrame,
    scores,
    concedes,
    prev_scores=None,
    prev_concedes=None,
) -> pd.DataFrame:
    """socceraction's VAEP formula, with optional separate series for the previous state's probabilities.

    ``actions`` is one game in action order with ``team_id``, ``time_seconds``, ``type_name`` and
    ``result_name``. ``scores`` / ``concedes`` are P_scores / P_concedes of each state (positional).
    ``prev_scores`` / ``prev_concedes`` (default: the same series) are what the *next* action is
    measured against.
    """
    idx = actions.index
    p_s = pd.Series(np.asarray(scores, dtype="float64"), index=idx)
    p_c = pd.Series(np.asarray(concedes, dtype="float64"), index=idx)
    src_s = p_s if prev_scores is None else pd.Series(np.asarray(prev_scores, dtype="float64"), index=idx)
    src_c = p_c if prev_concedes is None else pd.Series(np.asarray(prev_concedes, dtype="float64"), index=idx)
    team, seconds = actions["team_id"], actions["time_seconds"]
    types, results = actions["type_name"], actions["result_name"]

    sameteam = _prev(team) == team
    toolong = (seconds - _prev(seconds)).abs() > SAMEPHASE_SECONDS
    prevgoal = _prev(types).isin(SHOT_TYPES) & (_prev(results) == "success")

    prev_s = _prev(src_s, p_s) * sameteam + _prev(src_c, p_c) * (~sameteam)
    prev_s[toolong] = 0.0
    prev_s[prevgoal] = 0.0
    prev_s[types == "shot_penalty"] = PENALTY_PRIOR
    prev_s[types.isin(["corner_crossed", "corner_short"])] = CORNER_PRIOR

    prev_c = _prev(src_c, p_c) * sameteam + _prev(src_s, p_s) * (~sameteam)
    prev_c[toolong] = 0.0
    prev_c[prevgoal] = 0.0

    out = pd.DataFrame({"offensive_value": p_s - prev_s, "defensive_value": -(p_c - prev_c)}, index=idx)
    out["vaep_value"] = out["offensive_value"] + out["defensive_value"]
    return out


def penalty_scores(actions: pd.DataFrame, scores, shot_xg=None) -> pd.Series:
    """Rule 1: P_scores with every ``shot_penalty`` state replaced by that penalty's xG."""
    idx = actions.index
    p_s = pd.Series(np.asarray(scores, dtype="float64"), index=idx)
    xg = _xg_series(idx, shot_xg)
    is_pen = actions["type_name"] == "shot_penalty"
    return p_s.where(~is_pen, xg.fillna(PENALTY_XG_FALLBACK))


def after_shot_scores(actions: pd.DataFrame, scores, shot_xg=None) -> pd.Series:
    """Rule 2: the P_scores the next action is measured against, with each unscored shot's own chance removed.

    A shot followed by a ``keeper_save`` keeps its full P_scores, and so does a shot without xG.
    """
    idx = actions.index
    p_s = pd.Series(np.asarray(scores, dtype="float64"), index=idx)
    xg = _xg_series(idx, shot_xg)
    types = actions["type_name"]
    unscored = types.isin(SHOT_TYPES) & (actions["result_name"] != "success") & xg.notna()
    next_is_save = types.shift(-1).eq("keeper_save")
    residual = (1.0 - (1.0 - p_s) / (1.0 - xg.clip(upper=0.999))).clip(lower=0.0)
    residual = np.minimum(residual, p_s)
    return p_s.where(~(unscored & ~next_is_save), residual)


def value(actions: pd.DataFrame, scores, concedes, shot_xg=None) -> pd.DataFrame:
    """Offensive, defensive and VAEP value of each action of one game under option 3 (rules 1 and 2).

    ``shot_xg`` holds the StatsBomb xG of each shot action and NaN elsewhere (positional); None
    means no xG is known, so rule 1 uses 0.7835 and rule 2 is skipped.
    """
    p_s = penalty_scores(actions, scores, shot_xg)
    prev_s = after_shot_scores(actions, p_s, shot_xg)
    return formula_value(actions, p_s, concedes, prev_scores=prev_s)


def _xg_series(idx: pd.Index, shot_xg) -> pd.Series:
    if shot_xg is None:
        return pd.Series(np.nan, index=idx, dtype="float64")
    return pd.Series(np.asarray(shot_xg, dtype="float64"), index=idx)
