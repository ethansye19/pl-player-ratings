"""Percentile scaler: ``raw_score`` -> ``rating`` (3.0-10.0), separately per position group.

How a rating is made
--------------------
``fit`` (calibration, may read the whole reference season)
    Validates the contract, keeps the rows with ``minutes >= MIN_MINUTES`` and stores, for
    every position group present, the *reference table*: the distinct raw scores in
    increasing order and how many reference rows share each one. Nothing else is learned.

``transform`` (per-row, reads only the rows it is given plus the frozen table)
    1. Drops rows with ``minutes < MIN_MINUTES``.
    2. Converts each remaining row's raw score to a percentile ``p`` within ITS group's
       table, with the mid-rank convention ``p = (count_below + 0.5 * count_equal) / n``.
       Equal raw scores therefore get equal percentiles, and the reference rows' own
       percentiles are spread evenly over (0, 1) on average, which is what makes the
       reference season's mean rating equal the curve's mean (6.5). A raw score strictly
       between two reference scores gets a linearly interpolated ``p``. Below the lowest
       reference score ``p = 0`` (rating exactly 3.0), above the highest ``p = 1`` (10.0).
    3. ``rating = round(curve(p), RATING_DECIMALS)``, clipped to [RATING_MIN, RATING_MAX].
       ``curve`` is the fixed target curve in ``src/scale/curve.py``.

Why this respects the isolation rule
------------------------------------
A row's rating is a function of three things only: its own ``raw_score``, its own
``position_group``, and the frozen artefacts (reference table and fixed curve). It never
looks at the other rows of the frame it is given, so rating one match alone, or a match
mixed with other matches, gives identical numbers. The table is fitted once on a reference
set (the whole season, or the season to date for the FPL pipeline) and then never changes;
a saved scaler (``save`` / ``load``) reproduces ratings bit for bit. Rows are never ranked
against each other at transform time: that would make a rating depend on who else played.

Why per group: raw scores are only comparable within a position group (a goalkeeper's
raw score is on a different scale from a forward's), so each group has its own table and
every group is anchored to the same mean separately.

Saved format
------------
JSON (no pickle), written by ``to_dict`` / ``save``: ``{"format", "version", "min_minutes",
"rating_decimals", "curve": {"percentiles", "ratings"}, "groups": {group: {"n", "values",
"counts"}}}``. Floats are written with ``repr`` precision, so a reload is exact. The curve
anchors the scaler was fitted with are embedded and checked on load: a scaler saved under
a different curve is refused (re-run ``python -m src.scale.apply``) rather than silently
rating differently from the ratings file next to it.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.contract import (
    CONTRACT_COLUMNS,
    MIN_MINUTES,
    POSITION_GROUPS,
    RATING_COLUMN,
    RATING_DECIMALS,
    RATING_MAX,
    RATING_MIN,
    validate_contract,
)
from src.scale.curve import ANCHORS, curve

logger = logging.getLogger(__name__)

FORMAT = "PercentileScaler"
FORMAT_VERSION = 1
SMALL_GROUP_ROWS = 50   # below this a group's percentiles are too coarse to trust (warn only)


@dataclass(frozen=True, eq=False)
class _GroupTable:
    """Frozen reference for one position group."""

    values: np.ndarray       # distinct reference raw scores, strictly increasing, float64
    counts: np.ndarray       # reference rows holding each value, int64 >= 1
    percentiles: np.ndarray  # mid-rank percentile of each value, float64, strictly increasing
    n: int                   # reference rows in this group

    @classmethod
    def from_counts(cls, values: Any, counts: Any) -> "_GroupTable":
        try:
            vals = np.array(values, dtype="float64")
            cnts = np.array(counts)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reference table is not numeric: {exc}") from exc
        if vals.ndim != 1 or cnts.ndim != 1 or vals.size == 0 or vals.size != cnts.size:
            raise ValueError("reference table needs non-empty 1-D values and counts of equal length")
        if cnts.dtype.kind not in "iu":
            raise ValueError("reference counts must be integers")
        cnts = cnts.astype("int64")
        if not np.isfinite(vals).all():
            raise ValueError("reference values must be finite")
        if not (np.diff(vals) > 0).all():
            raise ValueError("reference values must be distinct and strictly increasing")
        if (cnts < 1).any():
            raise ValueError("reference counts must be at least 1")
        n = int(cnts.sum())
        below = np.cumsum(cnts) - cnts
        pct = (below + 0.5 * cnts) / n
        for arr in (vals, cnts, pct):
            arr.setflags(write=False)
        return cls(values=vals, counts=cnts, percentiles=pct, n=n)

    @classmethod
    def from_raw(cls, raw: np.ndarray) -> "_GroupTable":
        values, counts = np.unique(np.asarray(raw, dtype="float64"), return_counts=True)
        return cls.from_counts(values, counts)

    def percentile(self, x: np.ndarray) -> np.ndarray:
        """Mid-rank percentile of ``x`` against this table (0 below it, 1 above it)."""
        return np.interp(x, self.values, self.percentiles, left=0.0, right=1.0)


def _float_column(df: pd.DataFrame, name: str) -> np.ndarray:
    col = df[name]
    if not is_numeric_dtype(col):
        raise ValueError(f"'{name}' must be numeric, got {col.dtype}")
    try:
        return col.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' contains missing values") from exc


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-finite number {name} in scaler JSON")


class PercentileScaler:
    """Fit once on a reference set of raw scores, then map raw scores to ratings row by row.

    Typical use (``reference`` is a contract-format frame; ``one_match`` any rows to rate)::

        scaler = PercentileScaler().fit(reference)    # calibration, whole season
        ratings = scaler.transform(one_match)         # per-match, frozen artefacts only
    """

    def __init__(self) -> None:
        self._tables: dict[str, _GroupTable] = {}

    # ------------------------------------------------------------------ state
    @property
    def is_fitted(self) -> bool:
        return bool(self._tables)

    @property
    def groups(self) -> tuple[str, ...]:
        """Position groups that have a reference table, in ``POSITION_GROUPS`` order."""
        return tuple(g for g in POSITION_GROUPS if g in self._tables)

    def reference_sizes(self) -> dict[str, int]:
        """Reference rows per group (rows with ``minutes >= MIN_MINUTES`` seen by ``fit``)."""
        return {g: self._tables[g].n for g in self.groups}

    def _require_fitted(self) -> None:
        if not self._tables:
            raise RuntimeError("PercentileScaler is not fitted: call fit() or PercentileScaler.load() first")

    def __repr__(self) -> str:
        if not self._tables:
            return "PercentileScaler(unfitted)"
        sizes = ", ".join(f"{g}={n}" for g, n in self.reference_sizes().items())
        return f"PercentileScaler({sizes})"

    # -------------------------------------------------------------- calibration
    def fit(self, df: pd.DataFrame) -> "PercentileScaler":
        """Store the frozen reference table of every position group present in ``df``.

        ``df`` must satisfy the interface contract (``ContractError``, a ``ValueError``,
        otherwise). Only rows with ``minutes >= MIN_MINUTES`` enter the tables: cameo
        appearances have too little activity for their raw scores to be comparable.
        """
        validate_contract(df)
        rated = df.loc[df["minutes"].to_numpy(dtype="float64") >= MIN_MINUTES]
        if rated.empty:
            raise ValueError(f"no rows with minutes >= {MIN_MINUTES:g}: nothing to fit the scaler on")
        tables: dict[str, _GroupTable] = {}
        for group in POSITION_GROUPS:
            raw = rated.loc[rated["position_group"] == group, "raw_score"].to_numpy(dtype="float64")
            if raw.size:
                tables[group] = _GroupTable.from_raw(raw)
        self._tables = tables
        logger.info("fitted %r", self)
        for group, table in tables.items():
            if table.n < SMALL_GROUP_ROWS:
                logger.warning(
                    "position group %s has only %d reference rows; its percentiles are coarse", group, table.n
                )
        return self

    # ---------------------------------------------------------------- rating
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rate every row with ``minutes >= MIN_MINUTES``; each rating depends only on its own row.

        Returns the input columns plus ``rating`` (float64, one decimal, 3.0-10.0), in the
        input's row order with a fresh 0..n-1 index. Rows under ``MIN_MINUTES`` are dropped
        (and may carry anything in their other columns). Raises ``ValueError`` for missing
        contract columns, non-numeric or non-finite ``minutes`` / ``raw_score`` on rated
        rows, and a ``position_group`` that is unknown or was absent from the fit.
        """
        self._require_fitted()
        missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"missing contract columns: {missing}")
        minutes = _float_column(df, "minutes")
        if np.isnan(minutes).any():
            raise ValueError("'minutes' contains NaN")

        out = df.loc[minutes >= MIN_MINUTES].reset_index(drop=True)
        if RATING_COLUMN in out.columns:
            out = out.drop(columns=RATING_COLUMN)
        raw = _float_column(out, "raw_score")
        if not np.isfinite(raw).all():
            raise ValueError(f"'raw_score' contains NaN or inf on rows with minutes >= {MIN_MINUTES:g}")
        groups = out["position_group"].to_numpy(dtype=object)
        self._check_groups(groups)

        rating = np.empty(len(out), dtype="float64")
        for group, table in self._tables.items():
            mask = groups == group
            if mask.any():
                rating[mask] = curve(table.percentile(raw[mask]))
        out[RATING_COLUMN] = np.clip(np.round(rating, RATING_DECIMALS), RATING_MIN, RATING_MAX)
        return out

    def _check_groups(self, groups: np.ndarray) -> None:
        unknown = sorted({str(g) for g in groups.tolist() if g not in self._tables})
        if not unknown:
            return
        invalid = [g for g in unknown if g not in POSITION_GROUPS]
        if invalid:
            raise ValueError(f"unknown position_group values {invalid}; expected one of {list(POSITION_GROUPS)}")
        raise ValueError(
            f"no reference table for position_group {unknown}; this scaler was fitted on {list(self.groups)}"
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """``fit(df)`` then ``transform(df)``: rates the reference set itself."""
        return self.fit(df).transform(df)

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable description of the frozen scaler (see the module docstring)."""
        self._require_fitted()
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "min_minutes": MIN_MINUTES,
            "rating_decimals": RATING_DECIMALS,
            "curve": {
                "percentiles": [p for p, _ in ANCHORS],
                "ratings": [r for _, r in ANCHORS],
            },
            "groups": {
                g: {"n": t.n, "values": t.values.tolist(), "counts": t.counts.tolist()}
                for g, t in self._tables.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PercentileScaler":
        """Inverse of ``to_dict``. Raises ``ValueError`` for anything malformed or stale."""
        if not isinstance(data, Mapping):
            raise ValueError("scaler data must be a mapping")
        if data.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} file (format={data.get('format')!r})")
        if data.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported scaler version {data.get('version')!r}; expected {FORMAT_VERSION}")
        if data.get("min_minutes") != MIN_MINUTES or data.get("rating_decimals") != RATING_DECIMALS:
            raise ValueError(
                "scaler was saved with different MIN_MINUTES / RATING_DECIMALS than src.contract defines; refit it"
            )
        expected_curve = {
            "percentiles": [p for p, _ in ANCHORS],
            "ratings": [r for _, r in ANCHORS],
        }
        if data.get("curve") != expected_curve:
            raise ValueError(
                "scaler was fitted with a different rating curve than src.scale.curve defines; "
                "re-run `python -m src.scale.apply` to refit it"
            )
        groups = data.get("groups")
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError("scaler data has no position groups")
        tables: dict[str, _GroupTable] = {}
        for group, entry in groups.items():
            if group not in POSITION_GROUPS:
                raise ValueError(f"unknown position_group {group!r} in scaler data")
            try:
                table = _GroupTable.from_counts(entry["values"], entry["counts"])
                stated_n = entry["n"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"malformed reference table for {group}: {exc!r}") from exc
            if stated_n != table.n:
                raise ValueError(f"{group}: n={stated_n!r} does not match the counts (sum {table.n})")
            tables[group] = table
        scaler = cls()
        scaler._tables = {g: tables[g] for g in POSITION_GROUPS if g in tables}
        return scaler

    def save(self, path: str | os.PathLike) -> Path:
        """Write the scaler as JSON (atomically: a temporary file is renamed into place)."""
        payload = json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(payload + "\n")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "PercentileScaler":
        """Read a scaler written by ``save``; the loaded scaler rates exactly like the saved one."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text, parse_constant=_reject_constant))
