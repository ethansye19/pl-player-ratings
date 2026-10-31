"""Flatten raw StatsBomb event JSON into parquet-friendly records.

Naming follows ``statsbombpy.sb.events()``: the type-specific sub-object is expanded to
``<type>_<key>`` columns and every ``{id, name}`` object is replaced by its name, with
``*_id`` kept for team, player, possession_team, pass_recipient and substitution
fields (that is exactly what ``statsbombpy.helpers.flatten_event`` does, and we call it).

On top of that, so that everything is a plain parquet column:
- coordinate lists are split into ``*_x``, ``*_y`` (and ``*_z``) float columns;
- ``tactics`` keeps only ``tactics_formation`` (lineups live in lineups.parquet);
- ``shot_freeze_frame`` is dropped; ``related_events`` stays a list of event ids;
- any other nested object (e.g. ``50_50``) is flattened to ``<key>_<subkey>``;
- ``match_id``, ``type_id`` and ``position_id`` are added.

Column types are decided from everything seen across the season (``TypeRegistry``) so
that all 380 matches share one schema: booleans that StatsBomb only emits when true
become ``bool`` with ``False`` for missing, ints stay nullable int64, mixed numbers
become float64, text becomes string.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pyarrow as pa
from statsbombpy.helpers import flatten_event

COORD_COLUMNS = {
    "location": ("location_x", "location_y"),
    "pass_end_location": ("pass_end_x", "pass_end_y"),
    "carry_end_location": ("carry_end_x", "carry_end_y"),
    "shot_end_location": ("shot_end_x", "shot_end_y", "shot_end_z"),
    "goalkeeper_end_location": ("goalkeeper_end_x", "goalkeeper_end_y"),
}
DROP_COLUMNS = {"shot_freeze_frame"}
LEADING_COLUMNS = [
    "match_id", "id", "index", "period", "timestamp", "minute", "second", "type", "type_id",
    "possession", "possession_team", "possession_team_id", "play_pattern", "team", "team_id",
    "player", "player_id", "position", "position_id", "location_x", "location_y", "duration",
    "under_pressure", "counterpress", "off_camera", "out", "related_events",
]


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)


def flatten_match_events(events_json: list[dict[str, Any]], match_id: int) -> list[dict[str, Any]]:
    """Flatten one match's raw events (list of dicts) into flat records."""
    records: list[dict[str, Any]] = []
    for raw in events_json:
        event = flatten_event(dict(raw), True)
        rec: dict[str, Any] = {"match_id": int(match_id), "type_id": int(raw["type"]["id"])}
        if isinstance(raw.get("position"), dict):
            rec["position_id"] = int(raw["position"]["id"])
        for key, value in event.items():
            if key in DROP_COLUMNS or value is None:
                continue
            if key == "tactics":
                if isinstance(value, dict) and value.get("formation") is not None:
                    rec["tactics_formation"] = int(value["formation"])
                continue
            if key in COORD_COLUMNS or _is_number_list(value):
                names = COORD_COLUMNS.get(key) or tuple(f"{key}_{axis}" for axis in "xyz"[: len(value)])
                for name, coord in zip(names, value):
                    rec[name] = float(coord)
                continue
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "name" in subvalue:
                        subvalue = subvalue["name"]
                    if subvalue is None or isinstance(subvalue, (dict, list)):
                        continue
                    rec[f"{key}_{subkey}"] = subvalue
                continue
            if isinstance(value, list):
                if all(isinstance(x, str) for x in value):
                    rec[key] = list(value)
                continue  # lists of objects other than the freeze frame are not kept
            rec[key] = value
        records.append(rec)
    return records


class TypeRegistry:
    """Collects the Python types observed per column across all matches."""

    def __init__(self) -> None:
        self.kinds: dict[str, set[str]] = defaultdict(set)

    def update(self, records: list[dict[str, Any]]) -> None:
        for rec in records:
            for key, value in rec.items():
                self.kinds[key].add(_kind(value))

    def schema(self) -> pa.Schema:
        names = [c for c in LEADING_COLUMNS if c in self.kinds] + sorted(c for c in self.kinds if c not in LEADING_COLUMNS)
        return pa.schema([pa.field(name, _arrow_type(self.kinds[name])) for name in names])


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    raise TypeError(f"unexpected value type {type(value).__name__}: {value!r}")


def _arrow_type(kinds: set[str]) -> pa.DataType:
    if kinds == {"bool"}:
        return pa.bool_()
    if kinds <= {"int"}:
        return pa.int64()
    if kinds <= {"int", "float"}:
        return pa.float64()
    if kinds == {"list"}:
        return pa.list_(pa.string())
    return pa.string()


def records_to_table(records: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build an Arrow table with exactly ``schema`` from flat records."""
    columns: dict[str, list[Any]] = {name: [] for name in schema.names}
    for rec in records:
        for name in schema.names:
            columns[name].append(rec.get(name))
    arrays = []
    for field in schema:
        values = columns[field.name]
        if pa.types.is_boolean(field.type):
            values = [bool(v) if v is not None else False for v in values]
        elif pa.types.is_floating(field.type):
            values = [float(v) if v is not None else None for v in values]
        elif pa.types.is_string(field.type):
            values = [str(v) if v is not None else None for v in values]
        arrays.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)
