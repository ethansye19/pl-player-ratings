"""Colour bands for ratings, shared by the scale module, the web export and the page.

red < 6.0, orange 6.0-6.9, green 7.0-7.9, blue >= 8.0; None/NaN (under 10 minutes) -> unrated.
"""
from __future__ import annotations

import math


def rating_band(rating) -> str:
    if rating is None:
        return "unrated"
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return "unrated"
    if math.isnan(value):
        return "unrated"
    if value < 6.0:
        return "red"
    if value < 7.0:
        return "orange"
    if value < 8.0:
        return "green"
    return "blue"
