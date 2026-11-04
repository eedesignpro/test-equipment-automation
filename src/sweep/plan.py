"""Frequency lists and drive amplitude against frequency."""

from __future__ import annotations

import math
from typing import Callable, Sequence, Union

import numpy as np

from sweep.capture import Rule, rule_value


def frequencies(
    start: float,
    stop: float,
    *,
    points_per_decade: float | None = None,
    points: int | None = None,
    spacing: str = "log",
    extra: Sequence[float] = (),
) -> list[float]:
    """A sorted frequency list. Log spacing by points per decade or total points.

    ``extra`` adds named frequencies of interest, such as grid harmonics or a
    product's operating tones, inside or outside the range.
    """
    if not (start > 0 and stop > start):
        raise ValueError("need 0 < start < stop")
    if spacing == "log":
        if points is None:
            ppd = points_per_decade or 10
            points = max(2, int(round(ppd * math.log10(stop / start))) + 1)
        base = np.logspace(math.log10(start), math.log10(stop), int(points))
    elif spacing == "linear":
        base = np.linspace(start, stop, int(points or 50))
    else:
        raise ValueError("spacing must be 'log' or 'linear'")
    out = sorted(set(float(f) for f in base) | set(float(f) for f in extra))
    return out


Amplitude = Union[float, Callable[[float], float], Sequence[tuple[float, float]]]


def amplitude_at(amplitude: Amplitude, freq: float) -> float:
    """Evaluate a drive amplitude given as a constant, a function or a table."""
    value = rule_value(amplitude, freq)
    if value is None:
        raise ValueError("amplitude must not be None")
    return value


def first_order(
    level: float, corner: float, *, rising: bool = True, low: float | None = None, high=None
) -> Callable[[float], float]:
    """A drive that rises or falls 20 dB per decade above ``corner``, then clamps."""

    def fn(f: float) -> float:
        m = math.sqrt(1.0 + (f / corner) ** 2)
        v = level * m if rising else level / m
        if low is not None:
            v = max(low, v)
        if high is not None:
            v = min(high, v)
        return v

    return fn


def clamp(amplitude: Amplitude, low: float | None = None, high: float | None = None):
    """Wrap any amplitude so it stays between ``low`` and ``high``."""

    def fn(f: float) -> float:
        v = amplitude_at(amplitude, f)
        if low is not None:
            v = max(low, v)
        if high is not None:
            v = min(high, v)
        return v

    return fn


__all__ = ["Amplitude", "Rule", "amplitude_at", "clamp", "first_order", "frequencies"]
