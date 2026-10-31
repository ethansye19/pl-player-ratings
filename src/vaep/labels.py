"""VAEP labels: socceraction's binary outcome labels and the xG-based soft labels.

socceraction's labels (``socceraction.vaep.labels.scores`` / ``concedes``) are binary:
did the acting team score / concede within the window made of the current action and
the next ``NR_LABEL_ACTIONS - 1`` actions? A model trained on them values a shot by what
happened to it, so a goal is worth about +0.8 and a miss costs the whole chance.

The xG variant keeps the window and the team logic exactly and replaces each shot's
outcome with its StatsBomb expected goals ``g``::

    scores(i)   = 1 - prod_{j in window(i)} (1 - g_j)   over shots by the acting team
                                                        (an own goal by the opponent counts 1.0)
    concedes(i) = 1 - prod_{j in window(i)} (1 - g_j)   over shots by the opponent
                                                        (an own goal by the acting team counts 1.0)

i.e. the probability that at least one of the window's chances is converted, assuming
independent shots. With ``g`` restricted to the outcome (1 for a goal, 0 otherwise) the
formula reduces to socceraction's OR of goal flags, which is what ``outcome_labels``
computes through socceraction itself. Shots are joined to the StatsBomb event
(``shot_statsbomb_xg``) through ``original_event_id``; a shot without an xG value falls
back to its outcome and is counted in the coverage report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from socceraction.vaep import labels as sa_labels

from src.vaep.config import EVENTS_PATH, LABELS, NR_LABEL_ACTIONS

SHOT_TYPES = ("shot", "shot_penalty", "shot_freekick")


def shot_xg_lookup(events_path: Path = EVENTS_PATH) -> pd.Series:
    """StatsBomb xG of every shot event, indexed by event id (joins on SPADL ``original_event_id``)."""
    events = pd.read_parquet(events_path, columns=["id", "type", "shot_statsbomb_xg"])
    shots = events.loc[events["type"] == "Shot"]
    return shots.set_index("id")["shot_statsbomb_xg"].astype("float64")


def outcome_labels(actions: pd.DataFrame, nr_actions: int = NR_LABEL_ACTIONS) -> pd.DataFrame:
    """socceraction's binary labels (``scores``, ``concedes``) for one game's actions."""
    return pd.concat([sa_labels.scores(actions, nr_actions), sa_labels.concedes(actions, nr_actions)], axis=1)


def shot_goal_probability(actions: pd.DataFrame, xg_by_event: pd.Series) -> tuple[pd.Series, dict]:
    """Goal probability credited to each action's own shot (0 for non-shots) and the xG coverage counts.

    Shot actions take their StatsBomb xG; shots whose event has no xG fall back to the
    outcome (1 for a goal, 0 otherwise). Own goals are not shots for the acting team and
    are handled separately by the label window.
    """
    is_shot = actions["type_name"].isin(SHOT_TYPES) & (actions["result_name"] != "owngoal")
    prob = pd.Series(0.0, index=actions.index)
    if "original_event_id" in actions.columns and len(xg_by_event):
        xg = actions.loc[is_shot, "original_event_id"].map(xg_by_event).astype("float64")
    else:
        xg = pd.Series(np.nan, index=actions.index[is_shot], dtype="float64")
    missing = xg.isna()
    xg[missing] = (actions.loc[xg.index[missing], "result_name"] == "success").astype("float64")
    prob[is_shot] = xg
    coverage = {"shots": int(is_shot.sum()), "shots_without_xg": int(missing.sum()),
                "own_goals": int((actions["result_name"] == "owngoal").sum())}
    return prob, coverage


def xg_labels(actions: pd.DataFrame, xg_by_event: pd.Series, nr_actions: int = NR_LABEL_ACTIONS) -> tuple[pd.DataFrame, dict]:
    """Soft labels ``scores`` / ``concedes`` in [0, 1] for one game's actions (see the module docstring).

    ``actions`` must be one game in action order (any index). Mirrors socceraction's
    window exactly: action *i* and the next ``nr_actions - 1`` actions. Positions past
    the last action contribute nothing (socceraction pads them by repeating the last
    action, which is harmless for its OR of goal flags but would multiply a final
    chance into the product several times).
    """
    prob, coverage = shot_goal_probability(actions, xg_by_event)
    n = len(actions)
    team = actions["team_id"].to_numpy()
    goal = prob.to_numpy(dtype="float64")
    owngoal = (actions["result_name"] == "owngoal").to_numpy(dtype="float64")
    pad = np.zeros(nr_actions - 1)
    goal_p, own_p = np.concatenate([goal, pad]), np.concatenate([owngoal, pad])
    team_p = np.concatenate([team, np.repeat(team[-1:], nr_actions - 1)])
    idx = np.arange(n)
    not_scores = 1.0 - goal_p[idx]  # i = 0: only the acting team's own shot counts for `scores`
    not_concedes = 1.0 - own_p[idx]  # i = 0: an own goal by the acting team counts for `concedes`
    for i in range(1, nr_actions):
        same_team = team_p[idx + i] == team
        not_scores = not_scores * (1.0 - np.where(same_team, goal_p[idx + i], own_p[idx + i]))
        not_concedes = not_concedes * (1.0 - np.where(same_team, own_p[idx + i], goal_p[idx + i]))
    out = pd.DataFrame({LABELS[0]: 1.0 - not_scores, LABELS[1]: 1.0 - not_concedes}, index=actions.index)
    return out, coverage
