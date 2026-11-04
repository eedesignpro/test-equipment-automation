"""Other ways to read a tone, and to find one, besides the windowed DFT.

Each method has a best use and a blind spot; ``docs/frequency-sweep.md``
lists both. All work on a uniformly sampled record with first time ``t0``
and step ``step``, and return RMS phasors referenced to time zero, the same
convention as :func:`sweep.analysis.tone`, so results compare directly.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import scipy.fft as _fft
from scipy.signal import zoom_fft

from sweep.analysis import _cexp, make_window


def czt_search(
    x,
    t0: float,
    step: float,
    guess: float,
    half_width: float,
    *,
    window: str = "blackman",
    points: int = 512,
) -> float:
    """The strongest frequency within ``guess`` +- ``half_width``, by chirp-Z zoom.

    Evaluates the windowed DFT on ``points`` frequencies spread evenly over
    the band, far finer than the FFT bins, at about the cost of one FFT. Use it
    for a wide tolerance or when several peaks may sit inside it. It returns
    the peak of the band; refine it with exact DFT values afterwards.
    """
    x = np.asarray(x, dtype=float)
    fs = 1.0 / step
    lo = max(guess - half_width, 0.0)
    hi = min(guess + half_width, fs / 2)
    if hi <= lo:
        return float(guess)
    w = make_window(x.size, window)
    y = zoom_fft((x - x.mean()) * w, [lo, hi], m=points, fs=fs, endpoint=True)
    f = np.linspace(lo, hi, points)
    return float(f[int(np.argmax(np.abs(y)))])


def zoom_spectrum(
    x,
    step: float,
    center: float,
    half_width: float,
    *,
    window: str = "blackman",
    points: int = 1001,
) -> tuple[np.ndarray, np.ndarray]:
    """RMS amplitude spectrum on ``points`` frequencies across ``center`` +- ``half_width``.

    The same scale as :func:`sweep.analysis.spectrum`, on a grid as fine as
    asked, by chirp-Z zoom. A finer grid shows the shape of a peak and its
    close neighbours; it does not narrow the resolution, which stays the
    window's main lobe over the record length. Use it on a saved waveform
    (``Run.waveform``) to look around a tone after a run.
    """
    x = np.asarray(x, dtype=float)
    fs = 1.0 / step
    lo = max(center - half_width, 0.0)
    hi = min(center + half_width, fs / 2)
    w = make_window(x.size, window)
    y = zoom_fft((x - x.mean()) * w, [lo, hi], m=points, fs=fs, endpoint=True)
    return np.linspace(lo, hi, points), np.abs(y) * (2.0 / np.sum(w) / math.sqrt(2.0))


# ---------- coherent FFT ----------

_RECT: OrderedDict = OrderedDict()


def _rect_fft(x: np.ndarray) -> np.ndarray:
    key = (id(x), x.size)
    hit = _RECT.get(key)
    if hit is not None and hit[0] is x:
        return hit[1]
    y = _fft.rfft(x - x.mean(), workers=-1)
    _RECT[key] = (x, y)
    if len(_RECT) > 8:
        _RECT.popitem(last=False)
    return y


def coherent_phasor(x, t0: float, step: float, freq: float) -> tuple[complex, float, float, float]:
    """Read ``freq`` straight from one FFT bin of a coherent record, no window.

    Returns (phasor, distance of ``freq`` from the nearest bin in bins, level
    of the stronger neighbouring bin against this one in dB, level of that
    neighbour above the local noise floor in dB). On a coherent record the
    distance is 0 and the neighbours hold only noise. A distance above about
    1e-3 bin, or neighbours well above the noise floor, means the record is
    not coherent and the value leaks: use the DFT instead. The noise floor is
    from the bins 2 to 32 either side (median, scaled to RMS), so a few other
    tones in that span do not raise it.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    y = _rect_fft(x)
    kf = freq * n * step
    k = int(round(kf))
    if k < 1 or k >= y.size - 1:
        return complex(np.nan, np.nan), float("nan"), float("nan"), float("nan")
    p = y[k] * (2.0 / n) / math.sqrt(2.0) * np.exp(-2j * np.pi * freq * t0)
    side = max(abs(y[k - 1]), abs(y[k + 1]))
    leak = 20 * math.log10(max(side, 1e-300) / max(abs(y[k]), 1e-300))
    near = np.r_[y[max(1, k - 32) : max(1, k - 1)], y[k + 2 : k + 33]]
    # The median of a noise-only bin magnitude is 0.83 of its RMS.
    floor = float(np.median(np.abs(near))) / 0.8326 if near.size else 0.0
    excess = 20 * math.log10(max(side, 1e-300) / max(floor, 1e-300))
    return complex(p), float(kf - k), float(leak), float(excess)


# ---------- least-squares sine fits ----------

_FITS: OrderedDict = OrderedDict()


def sine_fit(
    x, t0: float, step: float, freqs, *, fit_freq: bool = False, iterations: int = 6
) -> dict:
    """Least-squares fit of a sum of sines at ``freqs`` plus a constant.

    Returns ``{"phasors": {freq: complex}, "residual_rms": float, "freq":
    fitted first frequency}``. With one frequency this is the IEEE 1057
    3-parameter fit; ``fit_freq`` also fits that frequency (4-parameter, by
    Gauss-Newton from the given start). With several frequencies it is a joint
    multi-tone fit: each tone is separated from the others by the fit, not by
    a window, so tones closer than a window's main lobe still resolve. The
    residual is the record minus every fitted tone: noise, distortion and
    every tone not in the list.
    """
    x = np.asarray(x, dtype=float)
    fl = [float(f) for f in freqs]
    key = (id(x), x.size, tuple(fl), fit_freq)
    hit = _FITS.get(key)
    if hit is not None and hit[0] is x:
        return hit[1]
    n = x.size
    xm = x - x.mean()
    f0 = fl[0]
    for _ in range(iterations if fit_freq else 1):
        cols = []
        for f in fl:
            c, s = _cexp(f, n, t0, step)
            cols += [c, s]
        cols.append(np.ones(n))
        if fit_freq:
            c, s = _cexp(f0, n, t0, step)
            a0, b0 = _solve(cols, xm)[:2]
            tt = t0 + step * np.arange(n)
            # d/df of a cos(wt) - b sin(wt) with the basis cos(-wt), sin(-wt)
            cols.append(2 * np.pi * tt * (a0 * s - b0 * c))
        coef = _solve(cols, xm)
        if not fit_freq:
            break
        df = coef[-1]
        f0 = f0 + df
        fl[0] = f0
        if abs(df) * step * n < 1e-7:
            break
    if fit_freq:
        cols = []
        for f in fl:
            c, s = _cexp(f, n, t0, step)
            cols += [c, s]
        cols.append(np.ones(n))
        coef = _solve(cols, xm)
    model = sum(coef[i] * cols[i] for i in range(len(cols)))
    resid = xm - model
    phasors = {}
    for i, f in enumerate(fl):
        a, b = coef[2 * i], coef[2 * i + 1]
        # The basis is cos(-wt) + j sin(-wt) = exp(-jwt), so a*cos(wt) - b*sin(wt)
        # equals Re{(a + jb) exp(jwt)}: the peak phasor is a + jb.
        phasors[f] = complex(a, b) / math.sqrt(2.0)
    out = {"phasors": phasors, "residual_rms": float(np.sqrt(np.mean(resid**2))), "freq": f0}
    _FITS[key] = (x, out)
    if len(_FITS) > 16:
        _FITS.popitem(last=False)
    return out


def _solve(cols, y) -> np.ndarray:
    """Least squares by the normal equations: m dot products of the record."""
    m = len(cols)
    g = np.empty((m, m))
    r = np.empty(m)
    for i in range(m):
        r[i] = cols[i] @ y
        for j in range(i, m):
            g[i, j] = g[j, i] = cols[i] @ cols[j]
    return np.linalg.lstsq(g, r, rcond=None)[0]
