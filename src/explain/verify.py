"""Grounding check for explanations: every number in the text must come from the input.

``numbers_in_text`` extracts every number in order of appearance (``"3 shots, 0.42 xG"``
-> ``[3.0, 0.42]``; ``"6-1"`` yields 6 and 1; ``"48th"`` yields 48).
``verify_numbers`` accepts a number if some numeric leaf of the input data equals it
after rounding to 0, 1, 2 or 3 decimals (signs are ignored, so ``-0.60`` matches a
contribution of ``-0.6`` and the ``1`` in ``6-1`` matches an away score of 1).
Text without any number passes.
"""
from __future__ import annotations

import re
from typing import Any, Iterator

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers_in_text(text: str) -> list[float]:
    return [float(token) for token in _NUMBER.findall(text)]


def numeric_leaves(data: Any) -> Iterator[float]:
    """Yield every int/float inside nested dicts/lists/tuples (bools excluded)."""
    if isinstance(data, bool) or data is None:
        return
    if isinstance(data, (int, float)):
        yield float(data)
    elif isinstance(data, dict):
        for value in data.values():
            yield from numeric_leaves(value)
    elif isinstance(data, (list, tuple, set)):
        for value in data:
            yield from numeric_leaves(value)


def allowed_numbers(data: Any, decimals: tuple[int, ...] = (0, 1, 2, 3)) -> set[float]:
    allowed: set[float] = set()
    for leaf in numeric_leaves(data):
        magnitude = abs(leaf)
        for d in decimals:
            allowed.add(round(magnitude, d))
    return allowed


def verify_numbers(text: str, data: Any) -> bool:
    """True iff every number in ``text`` appears (after rounding) somewhere in ``data``."""
    allowed = allowed_numbers(data)
    return all(any(abs(n - a) < 1e-9 for a in allowed) for n in numbers_in_text(text))


def unsupported_numbers(text: str, data: Any) -> list[float]:
    """The numbers in ``text`` that ``verify_numbers`` would reject (for diagnostics)."""
    allowed = allowed_numbers(data)
    return [n for n in numbers_in_text(text) if not any(abs(n - a) < 1e-9 for a in allowed)]
