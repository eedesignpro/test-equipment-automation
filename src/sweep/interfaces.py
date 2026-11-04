"""The two shapes the sweep loop needs: a source and a measurement.

A source sets a frequency and, if it has one, an amplitude. A measurement
returns either a :class:`Capture` (sampled records that the toolkit analyzes)
or :class:`Readings` (values the instrument already computed). Plain
functions work for both, so a user with other equipment writes a few lines in
their own script and passes them in::

    def my_source(freq, amplitude):
        box.write(f"FREQ {freq}")          # whatever the box needs
        return {"freq": freq}

    def my_meter(freq, plan):
        return {"V_out": meter.read_ac()}  # one number per signal

    run_sweep(freqs, my_source, [my_meter], ...)

A class with ``set``/``measure`` methods works the same way and may also have
``output(state)``, ``open()``, ``close()`` and ``describe()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from sweep.capture import CapturePlan


@dataclass
class Capture:
    """Sampled records from one acquisition. All signals share one time axis.

    Give ``t`` (seconds) or ``dt``. ``units`` maps signal name to unit text.
    ``group`` names the capture: phase between signals is meaningful only
    inside one group. ``full_scale`` maps a signal to its (low, high) limit
    for clipping checks, and ``clipped`` to an instrument-reported clip flag.
    ``settings`` holds what the instrument actually used.
    """

    signals: dict[str, np.ndarray]
    t: np.ndarray | None = None
    dt: float | None = None
    units: dict[str, str] = field(default_factory=dict)
    group: str = "capture"
    full_scale: dict[str, tuple[float, float]] | None = None
    clipped: dict[str, bool] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, dict] | None = None
    """Optional lossless form, saved instead of the scaled samples: per signal
    ``{"codes": int array, "ymult": .., "yoff": .., "yzero": ..}``, plus the
    key ``"_time"`` holding ``{"xincr": .., "xzero": .., "pt_off": ..}``."""


@dataclass
class Readings:
    """Values measured directly: a number (magnitude) or a complex phasor per name.

    With ``group=None`` every name is its own group, so no phase is formed
    between them.
    """

    values: dict[str, complex | float]
    units: dict[str, str] = field(default_factory=dict)
    group: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)


class FunctionSource:
    """A source made from a function ``fn(freq, amplitude) -> dict | float | None``."""

    def __init__(self, fn: Callable, name: str | None = None, *, on=None, off=None):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "source")
        self._on, self._off = on, off

    def set(self, freq: float, amplitude: float | None) -> dict:
        r = self._fn(freq, amplitude)
        if isinstance(r, dict):
            return {"freq": freq, "amplitude": amplitude, **r}
        if isinstance(r, (int, float)):
            return {"freq": float(r), "amplitude": amplitude}
        return {"freq": freq, "amplitude": amplitude}

    def output(self, state: bool) -> None:
        fn = self._on if state else self._off
        if fn is not None:
            fn()

    def describe(self) -> dict:
        return {"kind": "function", "name": self.name}


class FunctionMeasurement:
    """A measurement made from ``fn(freq, plan)`` returning a Capture, Readings,
    a dict of values, or one number."""

    def __init__(self, fn: Callable, name: str | None = None, *, unit: str = ""):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "measurement")
        self.unit = unit

    def measure(self, freq: float, plan: CapturePlan):
        r = self._fn(freq, plan)
        if isinstance(r, (Capture, Readings)):
            return r
        if isinstance(r, dict):
            return Readings(dict(r), units={k: self.unit for k in r})
        return Readings({self.name: r}, units={self.name: self.unit})

    def describe(self) -> dict:
        return {"kind": "function", "name": self.name}


def as_source(obj) -> Any:
    """Accept a source object or a plain function."""
    if hasattr(obj, "set"):
        return obj
    if callable(obj):
        return FunctionSource(obj)
    raise TypeError("a source needs a set(freq, amplitude) method or must be callable")


def as_measurement(obj) -> Any:
    """Accept a measurement object or a plain function."""
    if hasattr(obj, "measure"):
        return obj
    if callable(obj):
        return FunctionMeasurement(obj)
    raise TypeError("a measurement needs a measure(freq, plan) method or must be callable")


def describe(obj) -> dict:
    fn = getattr(obj, "describe", None)
    try:
        return dict(fn()) if fn else {"name": getattr(obj, "name", type(obj).__name__)}
    except Exception as exc:  # a description must never stop a sweep
        return {"name": getattr(obj, "name", type(obj).__name__), "describe_error": str(exc)}
