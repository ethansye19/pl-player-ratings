"""Train/score splits for the VAEP models.

Two kinds of split are used, both recorded in ``data/processed/vaep_splits.json``:

* **holdout** -- train on matchweeks 1-19, score matchweeks 20-38: a strict
  chronological split (the classic "train on part of the season, score the rest").
* **5-fold cross-fitting** -- a seeded shuffle of all matches into 5 roughly equal
  folds; the model of fold *k* is trained on the other folds and scores only fold *k*,
  so every match of the season is scored exactly once and never by a model that was
  trained on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from src.contract import SEED

HOLDOUT_LAST_TRAIN_WEEK = 19  # holdout model trains on match_week <= 19 and scores the rest


@dataclass
class Split:
    """Match ids a model is trained on (``train``) and the match ids it scores (``score``)."""

    name: str
    train: list[int] = field(default_factory=list)
    score: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.train = [int(m) for m in self.train]
        self.score = [int(m) for m in self.score]
        overlap = set(self.train) & set(self.score)
        if overlap:
            raise ValueError(f"split {self.name!r}: {len(overlap)} matches are in both train and score")
        if not self.train or not self.score:
            raise ValueError(f"split {self.name!r}: train and score must both be non-empty")


def make_cv_splits(match_ids: Iterable[int], n_folds: int = 5, seed: int = SEED) -> list[Split]:
    """Seeded shuffle of ``match_ids`` into ``n_folds`` roughly equal folds.

    Fold *k* scores its own matches and trains on all the others, so every match is
    scored in exactly one fold and never appears in that fold's training set. The
    same ids and seed always give the same folds.
    """
    ids = [int(m) for m in match_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("match_ids contains duplicates")
    if n_folds < 2 or n_folds > len(ids):
        raise ValueError(f"n_folds must be between 2 and the number of matches ({len(ids)}), got {n_folds}")
    ids = sorted(ids)
    order = np.random.default_rng(seed).permutation(len(ids))
    splits: list[Split] = []
    for k, chunk in enumerate(np.array_split(order, n_folds)):
        score = sorted(ids[i] for i in chunk)
        held = set(score)
        splits.append(Split(name=f"fold{k}", train=[m for m in ids if m not in held], score=score))
    return splits


def make_holdout_split(matches: pd.DataFrame, last_train_week: int = HOLDOUT_LAST_TRAIN_WEEK) -> Split:
    """Chronological split: train on ``match_week <= last_train_week``, score the rest."""
    for col in ("match_id", "match_week"):
        if col not in matches.columns:
            raise ValueError(f"matches is missing the {col!r} column")
    weeks = matches[["match_id", "match_week"]].drop_duplicates("match_id")
    early = weeks["match_week"] <= last_train_week
    train = sorted(int(m) for m in weeks.loc[early, "match_id"])
    score = sorted(int(m) for m in weeks.loc[~early, "match_id"])
    if not train or not score:
        raise ValueError(f"no matches on one side of match_week {last_train_week}; cannot build the holdout split")
    return Split(name="holdout", train=train, score=score)


def splits_payload(holdout: Split, folds: list[Split]) -> dict:
    """The exact ``vaep_splits.json`` layout from CLAUDE.md, with plain int match ids."""
    return {
        "holdout": {"train": list(holdout.train), "score": list(holdout.score)},
        "folds": [{"fold_id": k, "train": list(f.train), "score": list(f.score)} for k, f in enumerate(folds)],
    }


def fold_for_match(payload: dict, match_id: int) -> dict:
    """The recorded fold whose model scores ``match_id`` (and therefore did not train on it)."""
    match_id = int(match_id)
    hits = [f for f in payload["folds"] if match_id in f["score"]]
    if len(hits) != 1:
        raise ValueError(f"match {match_id} is scored by {len(hits)} folds in vaep_splits.json; expected exactly 1")
    fold = hits[0]
    if match_id in fold["train"]:
        raise ValueError(f"match {match_id} is in both the train and score set of fold {fold['fold_id']}")
    return fold
