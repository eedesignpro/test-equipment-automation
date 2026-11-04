"""Interpolation of frequency responses onto new frequencies.

Responses are interpolated on a log10 frequency axis. A complex response is
split into magnitude in dB and unwrapped phase, and each is interpolated
linearly, so a straight line on a Bode plot stays straight. A real response
is interpolated linearly in value, which keeps its sign and passes through
zero. Points whose value or frequency is not finite are skipped.

:func:`common_grid` picks the frequencies two or more runs share, for a
comparison.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

EXTRAPOLATE = ("hold", "nan", "linear")

# Decades of slack before a frequency counts as outside the measured range.
_EDGE_TOL = 1e-9

_TINY = 1e-300


def _interp1(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, mode: str) -> np.ndarray:
    y = np.interp(x, xp, fp)
    if mode == "linear" and xp.size >= 2:
        lo = x < xp[0]
        hi = x > xp[-1]
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def _clean(freqs, values) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(freqs, dtype=float)
    v = np.asarray(values)
    if f.ndim != 1 or v.ndim != 1 or f.size != v.size:
        raise ValueError("freqs and values must be 1-D and the same length")
    keep = np.isfinite(f) & np.isfinite(v)
    f, v = f[keep], v[keep]
    if f.size == 0:
        raise ValueError("no finite points to interpolate")
    if np.any(f <= 0):
        raise ValueError("frequencies must be positive for a log axis")
    order = np.argsort(f, kind="stable")
    f, v = f[order], v[order]
    if np.any(np.diff(f) == 0):
        raise ValueError("frequencies must not repeat")
    return f, v


def interpolate(freqs, values, new_freqs, *, extrapolate: str = "hold"):
    """Values of a response at ``new_freqs``: (interpolated, outside mask).

    ``extrapolate`` sets what happens outside the measured range: ``"hold"``
    repeats the edge value, ``"nan"`` returns NaN, ``"linear"`` extends the
    end segment in log frequency. The mask is True where a new frequency lies
    outside the measured range, whatever the mode. Real input returns a real
    array, complex input a complex array.
    """
    if extrapolate not in EXTRAPOLATE:
        raise ValueError(f"extrapolate {extrapolate!r} is not one of {EXTRAPOLATE}")
    f, v = _clean(freqs, values)
    nf = np.asarray(new_freqs, dtype=float)
    shape = nf.shape
    nf = nf.ravel()
    if np.any(~(nf > 0)):
        raise ValueError("new frequencies must be positive for a log axis")
    xp = np.log10(f)
    x = np.log10(nf)
    outside = (x < xp[0] - _EDGE_TOL) | (x > xp[-1] + _EDGE_TOL)
    x = np.where(outside, x, np.clip(x, xp[0], xp[-1]))
    if np.iscomplexobj(v):
        db = 20.0 * np.log10(np.maximum(np.abs(v), _TINY))
        ph = np.unwrap(np.angle(v))
        out = 10.0 ** (_interp1(x, xp, db, extrapolate) / 20.0) * np.exp(
            1j * _interp1(x, xp, ph, extrapolate)
        )
        nan = complex(np.nan, np.nan)
    else:
        out = _interp1(x, xp, v.astype(float), extrapolate)
        nan = np.nan
    if extrapolate == "nan":
        out[outside] = nan
    return out.reshape(shape), outside.reshape(shape)


def common_grid(
    freq_arrays: Sequence, *, mode: str = "overlap", points: int | None = None
) -> np.ndarray:
    """Frequencies shared by several runs, for a comparison.

    ``mode="overlap"`` spans only the range every run covers. ``"union"``
    spans the range any run covers. With ``points`` the grid is that many
    log-spaced points over the span; without it the grid is every measured
    frequency inside the span, sorted, with repeats removed.
    """
    if mode not in ("overlap", "union"):
        raise ValueError(f"mode {mode!r} is not 'overlap' or 'union'")
    arrays = []
    for a in freq_arrays:
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            arrays.append(a)
    if not arrays:
        raise ValueError("no positive finite frequencies given")
    if mode == "overlap":
        lo = max(a.min() for a in arrays)
        hi = min(a.max() for a in arrays)
        if lo > hi:
            raise ValueError("the runs share no frequency range")
    else:
        lo = min(a.min() for a in arrays)
        hi = max(a.max() for a in arrays)
    if points is not None:
        if points < 1:
            raise ValueError("points must be at least 1")
        return np.geomspace(lo, hi, int(points))
    # Runs that each refine their own received frequency report the same
    # planned point a few parts per million apart: merge those, and keep a
    # point that sits just past the shared span by the same tolerance.
    tol = 1e-5
    allf = np.unique(np.concatenate(arrays))
    allf = allf[(allf >= lo * (1 - tol)) & (allf <= hi * (1 + tol))]
    merged = [allf[0]] if allf.size else []
    for f in allf[1:]:
        if f > merged[-1] * (1 + tol):
            merged.append(f)
    out = np.clip(np.asarray(merged, dtype=float), lo, hi)
    return np.unique(out)
