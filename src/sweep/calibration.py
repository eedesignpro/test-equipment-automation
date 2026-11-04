"""Calibration: a complex correction against frequency, built once and applied later.

A calibration is the ratio ``expected / measured`` at each calibration
frequency. Applying it multiplies a later response by that ratio,
interpolated onto the response frequencies with :func:`sweep.interp.interpolate`.
Typical uses::

    cal = Calibration.from_measurement(f, h_through, expected=1.0, name="cables")
    cal.save(run / "cal_cables.csv")
    h_fixed, outside = Calibration.load(run / "cal_cables.csv").apply(f2, h2)

    # Two receiver channels probing one node: make ch_b read like ch_a.
    cal = Calibration.from_two_channel_through(f, v_ch1, v_ch2)

A measured point that is NaN gives a NaN correction, which :meth:`apply`
skips. A measured zero is refused, because it means nothing was connected.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from sweep.interp import interpolate

Expected = Union[complex, float, np.ndarray, Callable[[float], complex]]

COLUMNS = ("freq_hz", "corr_real", "corr_imag", "corr_mag_db", "corr_phase_deg")

_FORMAT = "sweep-calibration 1"


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"meta value of type {type(obj).__name__} is not JSON serializable")


@dataclass
class Calibration:
    """Complex correction ``correction`` at ``freqs`` (Hz), sorted by frequency."""

    freqs: np.ndarray
    correction: np.ndarray
    name: str = "calibration"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        f = np.asarray(self.freqs, dtype=float)
        c = np.asarray(self.correction, dtype=complex)
        if f.ndim != 1 or c.ndim != 1 or f.size != c.size or f.size == 0:
            raise ValueError("freqs and correction must be 1-D, non-empty and the same length")
        if np.any(~(f > 0)):
            raise ValueError("calibration frequencies must be positive")
        order = np.argsort(f, kind="stable")
        self.freqs = f[order]
        self.correction = c[order]
        self.meta = dict(self.meta)

    @classmethod
    def from_measurement(
        cls,
        freqs,
        measured,
        expected: Expected = 1.0,
        *,
        name: str = "calibration",
        meta: dict | None = None,
    ) -> Calibration:
        """Correction ``expected / measured``.

        ``expected`` is a constant, an array the length of ``freqs``, or a
        function of one frequency in Hz that returns the complex value the
        measurement should have read.
        """
        f = np.asarray(freqs, dtype=float)
        m = np.asarray(measured, dtype=complex)
        if m.shape != f.shape:
            raise ValueError("freqs and measured must be the same length")
        if callable(expected):
            e = np.array([expected(float(x)) for x in f], dtype=complex)
        else:
            e = np.broadcast_to(np.asarray(expected, dtype=complex), f.shape)
        if np.any(m == 0):
            raise ValueError("measured response is zero at some frequency; nothing to correct")
        with np.errstate(invalid="ignore"):
            corr = e / m
        return cls(f, corr, name=name, meta=dict(meta or {}))

    @classmethod
    def from_two_channel_through(
        cls, freqs, ch_a, ch_b, *, name: str = "through", meta: dict | None = None
    ) -> Calibration:
        """Correction that makes ``ch_b`` read like ``ch_a`` when both probe one node.

        Apply it to later ``ch_b`` data. It removes the gain and phase
        difference between the two receiver paths, including probe skew.
        """
        return cls.from_measurement(freqs, ch_b, expected=np.asarray(ch_a), name=name, meta=meta)

    def apply(self, freqs, values, *, extrapolate: str = "hold"):
        """Corrected values at ``freqs``: (values * correction, outside mask).

        The mask is True where ``freqs`` lies outside the calibrated range;
        those points use the ``extrapolate`` rule of :func:`interpolate`.
        """
        v = np.asarray(values)
        corr, outside = interpolate(self.freqs, self.correction, freqs, extrapolate=extrapolate)
        if corr.shape != v.shape:
            raise ValueError("freqs and values must be the same length")
        return v * corr, outside

    def save(self, path) -> Path:
        """Write the calibration as CSV with ``#`` header lines. Returns the path."""
        p = Path(path)
        c = self.correction
        with np.errstate(divide="ignore", invalid="ignore"):
            mag_db = 20.0 * np.log10(np.abs(c))
        phase = np.full(c.size, np.nan)
        ok = np.isfinite(c)
        phase[ok] = np.degrees(np.unwrap(np.angle(c[ok])))
        with p.open("w", newline="", encoding="utf-8") as fh:
            fh.write(f"# format: {_FORMAT}\n")
            fh.write(f"# name: {json.dumps(self.name)}\n")
            fh.write(f"# meta: {json.dumps(self.meta, default=_json_default)}\n")
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            for row in zip(self.freqs, c.real, c.imag, mag_db, phase):
                w.writerow([repr(float(x)) for x in row])
        return p

    @classmethod
    def load(cls, path) -> Calibration:
        """Read a calibration written by :meth:`save`."""
        p = Path(path)
        name, meta = "calibration", {}
        rows = []
        with p.open("r", newline="", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        body = []
        for line in lines:
            if line.startswith("#"):
                key, _, val = line[1:].partition(":")
                key, val = key.strip(), val.strip()
                if key == "name":
                    name = json.loads(val)
                elif key == "meta":
                    meta = json.loads(val)
            elif line.strip():
                body.append(line)
        reader = csv.DictReader(body)
        missing = {"freq_hz", "corr_real", "corr_imag"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{p} is missing columns {sorted(missing)}")
        for r in reader:
            rows.append(
                (float(r["freq_hz"]), complex(float(r["corr_real"]), float(r["corr_imag"])))
            )
        if not rows:
            raise ValueError(f"{p} holds no calibration points")
        f = np.array([r[0] for r in rows])
        c = np.array([r[1] for r in rows])
        return cls(f, c, name=name, meta=meta)
