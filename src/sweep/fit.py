"""Fits of measured responses: pure delay, rational transfer functions, auto order.

:func:`fit_delay` fits a straight line to unwrapped phase against linear
frequency. The slope is a time delay, which is how probe or cable skew shows.

:func:`fit_rational` fits ``H(s) = N(s) / D(s)`` with chosen numerator and
denominator orders by Sanathanan-Koerner iteration: repeated weighted linear
least squares on ``N - H D``, each pass weighted by the previous ``1 / |D|``
so the error approaches the true ``N / D - H``. The polynomials use a
normalized ``s = j 2 pi f / w0``, with ``w0`` the geometric mean angular
frequency, to keep the equations well conditioned over wide spans. A short
nonlinear polish on the log error (dB and phase) follows, which also refines
the optional delay jointly with the polynomials.

:func:`auto_fit` tries a range of orders and keeps the simplest one that fits.
Every fit returns a :class:`TransferFit` with poles and zeros in Hz, so a
pole at ``-1000`` is a 1 kHz corner, and :meth:`TransferFit.describe` puts it
in words.

Unwrapping needs the phase to change by less than 180 degrees between
neighbouring points. A sparse sweep over a long delay cannot satisfy that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

_NEPER_DB = 20.0 / math.log(10.0)
_REAL_TOL = 1e-6
_DC_FRACTION = 1e-6


def _prepare(freqs, values, weights=None):
    f = np.asarray(freqs, dtype=float)
    h = np.asarray(values, dtype=complex)
    if f.ndim != 1 or h.shape != f.shape:
        raise ValueError("freqs and values must be 1-D and the same length")
    w = np.ones(f.size) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != f.shape:
        raise ValueError("weights must be the same length as freqs")
    keep = np.isfinite(f) & np.isfinite(h) & (h != 0) & np.isfinite(w) & (w > 0)
    f, h, w = f[keep], h[keep], w[keep]
    if np.any(f <= 0):
        raise ValueError("frequencies must be positive")
    order = np.argsort(f, kind="stable")
    return f[order], h[order], w[order]


def _wrap_deg(x: float) -> float:
    return float(-((-x + 180.0) % 360.0) + 180.0)


def fit_delay(freqs, values) -> tuple[float, float]:
    """Pure delay fit: (delay in s, phase at zero frequency in degrees).

    Least squares of ``unwrapped phase = phase0 - 360 f delay``. A positive
    delay means the response lags. ``phase0`` is wrapped to (-180, 180]; near
    180 means an inversion.
    """
    f, h, _ = _prepare(freqs, values)
    if f.size < 2:
        raise ValueError("a delay fit needs at least 2 points")
    ph = np.degrees(np.unwrap(np.angle(h)))
    a = np.column_stack([np.ones(f.size), -360.0 * f])
    (phase0, delay), *_ = np.linalg.lstsq(a, ph, rcond=None)
    return float(delay), _wrap_deg(float(phase0))


def _roots(coeffs: np.ndarray, scale: float) -> np.ndarray:
    c = np.trim_zeros(np.asarray(coeffs, dtype=float), "b")
    if c.size <= 1:
        return np.zeros(0, dtype=complex)
    return np.roots(c[::-1]).astype(complex) * scale


def _eng(x: float, unit: str = "Hz") -> str:
    if not math.isfinite(x):
        return f"{x} {unit}"
    if x == 0:
        return f"0 {unit}"
    k = int(math.floor(math.log10(abs(x)) / 3))
    k = max(-4, min(4, k))
    prefix = {-4: "p", -3: "n", -2: "u", -1: "m", 0: "", 1: "k", 2: "M", 3: "G", 4: "T"}[k]
    return f"{x / 10 ** (3 * k):.4g} {prefix}{unit}"


def _group_roots(roots: np.ndarray) -> list[tuple[str, complex]]:
    """Real roots and one member (positive imaginary part) of each conjugate pair."""
    out = []
    for r in sorted(roots, key=lambda z: (abs(z), z.imag)):
        if abs(r.imag) <= _REAL_TOL * max(abs(r), 1e-300):
            out.append(("real", complex(r.real, 0.0)))
        elif r.imag > 0:
            out.append(("pair", r))
    return out


@dataclass
class TransferFit:
    """A fitted transfer function in Hz.

    ``H(f) = gain * prod(j f - zeros) / prod(j f - poles) * exp(-j 2 pi f delay_s)``.
    ``zeros`` and ``poles`` are s-plane roots divided by 2 pi, so they read
    in Hz. Errors are the RMS over the fitted points of the magnitude ratio in
    dB and the phase difference in degrees.
    """

    zeros: np.ndarray
    poles: np.ndarray
    gain: float
    delay_s: float = 0.0
    rms_error_db: float = float("nan")
    rms_error_deg: float = float("nan")
    freq_range: tuple[float, float] | None = None

    @property
    def n_zeros(self) -> int:
        return int(np.size(self.zeros))

    @property
    def n_poles(self) -> int:
        return int(np.size(self.poles))

    @property
    def dc_tolerance(self) -> float:
        """Roots smaller than this (Hz) count as at DC: 1e-6 of the lowest fitted frequency."""
        return _DC_FRACTION * self.freq_range[0] if self.freq_range else 0.0

    @property
    def dc_gain(self) -> float | None:
        """Gain at 0 Hz, or None when a pole sits at DC."""
        p = np.asarray(self.poles, dtype=complex)
        z = np.asarray(self.zeros, dtype=complex)
        tol = self.dc_tolerance
        if np.any(np.abs(p) <= tol):
            return None
        if np.any(np.abs(z) <= tol):
            return 0.0
        return float(np.real(self.gain * np.prod(-z) / np.prod(-p)))

    def evaluate(self, freqs) -> np.ndarray:
        """Complex response at ``freqs`` (Hz)."""
        f = np.asarray(freqs, dtype=float)
        q = 1j * f.ravel()
        z = np.asarray(self.zeros, dtype=complex)
        p = np.asarray(self.poles, dtype=complex)
        num = np.prod(q[:, None] - z[None, :], axis=1)
        den = np.prod(q[:, None] - p[None, :], axis=1)
        h = self.gain * num / den * np.exp(-2j * np.pi * f.ravel() * self.delay_s)
        return h.reshape(f.shape)

    def describe(self) -> str:
        """Short plain-text summary: corners, Q of pairs, DC gain, delay, fit error."""
        lines = [f"Transfer fit: {self.n_zeros} zero(s), {self.n_poles} pole(s)"]
        if self.freq_range:
            lines.append(f"Fitted over {_eng(self.freq_range[0])} to {_eng(self.freq_range[1])}")
        dc = self.dc_gain
        if dc is None:
            lines.append("DC gain: infinite (pole at DC)")
        elif dc == 0:
            lines.append("DC gain: 0 (zero at DC)")
        else:
            lines.append(f"DC gain: {dc:.4g} ({20 * math.log10(abs(dc)) + 0.0:.2f} dB)")
        slope = 20 * (self.n_zeros - self.n_poles)
        lines.append(f"High-frequency slope: {slope:+d} dB/decade")
        if self.delay_s:
            lines.append(f"Delay: {_eng(self.delay_s, 's')}")
        for label, roots in (("Poles", self.poles), ("Zeros", self.zeros)):
            items = _group_roots(np.asarray(roots, dtype=complex))
            if not items:
                continue
            lines.append(f"{label}:")
            for kind, r in items:
                at_dc = abs(r) <= self.dc_tolerance
                lines.append("  " + _describe_root(kind, r, label == "Poles", at_dc))
        lines.append(f"Fit error: {self.rms_error_db:.3g} dB rms, {self.rms_error_deg:.3g} deg rms")
        return "\n".join(lines)


def _describe_root(kind: str, r: complex, is_pole: bool, at_dc: bool = False) -> str:
    what = "pole" if is_pole else "zero"
    if r == 0 or at_dc:
        return f"{what} at DC"
    side = "left half plane"
    if r.real > 0:
        side = "right half plane, unstable" if is_pole else "right half plane, non-minimum phase"
    if kind == "real":
        return f"real {what} at {_eng(abs(r.real))} ({side})"
    f0 = abs(r)
    q = f0 / (2 * abs(r.real)) if r.real != 0 else float("inf")
    return f"{what} pair f0 = {_eng(f0)}, Q = {q:.4g} ({side})"


def _sk(sn, h, wt, nz, np_, iterations):
    vn = sn[:, None] ** np.arange(nz + 1)
    vd = sn[:, None] ** np.arange(np_ + 1)
    d_prev = np.ones(sn.size)
    x = np.zeros(nz + np_ + 2)
    for _ in range(max(1, iterations)):
        w = wt / np.maximum(np.abs(d_prev), 1e-300)
        a = np.hstack([vn, -h[:, None] * vd]) * w[:, None]
        ar = np.vstack([a.real, a.imag])
        scale = np.linalg.norm(ar, axis=0)
        scale[scale == 0] = 1.0
        _, _, vt = np.linalg.svd(ar / scale, full_matrices=False)
        x = vt[-1] / scale
        d_new = np.abs(vd @ x[nz + 1 :])
        if np.max(np.abs(d_new / np.maximum(d_prev, 1e-300) - 1.0)) < 1e-12:
            break
        d_prev = d_new
    return x[: nz + 1], x[nz + 1 :]


def fit_rational(
    freqs,
    values,
    n_zeros: int,
    n_poles: int,
    *,
    iterations: int = 20,
    weights=None,
    delay: bool = False,
) -> TransferFit:
    """Fit ``n_zeros`` zeros, ``n_poles`` poles and a gain to a measured response.

    ``weights`` scales each point's share of the log error (dB and phase);
    by default every point counts the same, so a stopband 60 dB down is fitted
    as carefully as the passband. ``delay=True`` adds a pure delay, started
    from :func:`fit_delay` and refined with the rest.
    """
    if n_zeros < 0 or n_poles < 0:
        raise ValueError("orders must be zero or more")
    f, h, w_user = _prepare(freqs, values, weights)
    n_par = n_zeros + n_poles + 1 + (1 if delay else 0)
    if 2 * f.size <= n_par:
        raise ValueError(f"{f.size} points cannot fit {n_par} parameters")
    om = 2 * np.pi * f
    w0 = float(np.exp(np.mean(np.log(om))))
    sn = 1j * om / w0

    tau = fit_delay(f, h)[0] if delay else 0.0
    h0 = h * np.exp(1j * om * tau)
    mag = np.abs(h0)
    wt = w_user / np.maximum(mag, 1e-12 * mag.max())
    b, a = _sk(sn, h0, wt, n_zeros, n_poles, iterations)

    k = int(np.argmax(np.abs(a)))
    b, a = b / a[k], a / a[k]
    vn = sn[:, None] ** np.arange(n_zeros + 1)
    vd = sn[:, None] ** np.arange(n_poles + 1)
    on = om / w0

    def unpack(x):
        bb = x[: n_zeros + 1]
        aa = np.insert(x[n_zeros + 1 : n_zeros + 1 + n_poles], k, 1.0)
        tn = x[-1] if delay else 0.0
        return bb, aa, tn

    def resid(x):
        bb, aa, tn = unpack(x)
        with np.errstate(all="ignore"):
            r = np.log((vn @ bb) / (vd @ aa) * np.exp(-1j * on * tn) / h)
        out = np.concatenate([w_user * r.real, w_user * r.imag])
        out[~np.isfinite(out)] = 1e6
        return out

    x0 = np.concatenate([b, np.delete(a, k)] + ([[tau * w0]] if delay else []))
    best = x0
    try:
        res = least_squares(resid, x0, x_scale="jac", max_nfev=200 * x0.size)
        if np.sum(res.fun**2) < np.sum(resid(x0) ** 2):
            best = res.x
    except (ValueError, np.linalg.LinAlgError):
        pass
    b, a, tn = unpack(best)
    tau = tn / w0 if delay else 0.0

    c = w0 / (2 * np.pi)
    zeros = _roots(b, c)
    poles = _roots(a, c)
    bt = np.trim_zeros(b, "b")
    at = np.trim_zeros(a, "b")
    if bt.size == 0:
        gain = 0.0
    else:
        gain = float(bt[-1] / at[-1] * c ** ((at.size - 1) - (bt.size - 1)))
    fit = TransferFit(
        zeros, poles, gain, delay_s=float(tau), freq_range=(float(f[0]), float(f[-1]))
    )
    with np.errstate(all="ignore"):
        ratio = fit.evaluate(f) / h
    fit.rms_error_db = float(np.sqrt(np.mean((20 * np.log10(np.abs(ratio))) ** 2)))
    fit.rms_error_deg = float(np.sqrt(np.mean(np.degrees(np.angle(ratio)) ** 2)))
    return fit


def _score(fit: TransferFit, penalty: float) -> float:
    err = math.hypot(fit.rms_error_db / _NEPER_DB, math.radians(fit.rms_error_deg))
    if not math.isfinite(err):
        return float("inf")
    return math.log(max(err, 1e-6)) + penalty * (fit.n_zeros + fit.n_poles)


def auto_fit(
    freqs,
    values,
    max_order: int = 4,
    *,
    penalty: float = 0.1,
    weights=None,
    delay: bool = False,
    iterations: int = 20,
) -> TransferFit:
    """Best :class:`TransferFit` over pole counts 0 to ``max_order``.

    Zero counts run from 0 to one more than the pole count, capped at
    ``max_order``. The score is the log of the combined RMS log error plus
    ``penalty`` per pole and zero, so each added root must cut the error by
    about ``penalty`` (10 % by default) to be kept. Errors below 1e-6 (about
    1e-5 dB) count as exact, so a clean synthetic response picks its true order.
    """
    best, best_score = None, float("inf")
    for n_p in range(max_order + 1):
        for n_z in range(min(n_p + 1, max_order) + 1):
            try:
                fit = fit_rational(
                    freqs, values, n_z, n_p, iterations=iterations, weights=weights, delay=delay
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            s = _score(fit, penalty)
            if s < best_score:
                best, best_score = fit, s
    if best is None:
        raise ValueError("no order could be fitted to these points")
    return best


@dataclass
class ModelFit:
    """Parameters of a user circuit model fitted to a measured response."""

    params: dict[str, float]
    rms_error_db: float
    rms_error_deg: float
    model: object

    def evaluate(self, freqs) -> np.ndarray:
        return np.asarray(self.model(np.asarray(freqs, dtype=float), **self.params), complex)

    def describe(self) -> str:
        parts = ", ".join(f"{k} = {v:.4g}" for k, v in self.params.items())
        return (
            f"Model fit: {parts}. Error {self.rms_error_db:.3f} dB rms, "
            f"{self.rms_error_deg:.3f} deg rms"
        )


def fit_model(
    freqs, values, model, initial: dict[str, float], *, bounds=None, weights=None
) -> ModelFit:
    """Fit named parameters of ``model(freqs, **params) -> complex`` to a response.

    Use it when the circuit is known, for example a resistor with lead
    inductance and probe skew: ``lambda f, R, L, tau: (R + 2j*pi*f*L) *
    exp(2j*pi*f*tau)``. The error is the log of the ratio, so magnitude (dB)
    and phase count equally at every frequency. ``bounds`` maps a name to
    (low, high).
    """
    f = np.asarray(freqs, dtype=float)
    h = np.asarray(values, dtype=complex)
    ok = np.isfinite(h) & (np.abs(h) > 0)
    f, h = f[ok], h[ok]
    w = np.ones(f.size) if weights is None else np.asarray(weights, dtype=float)[ok]
    names = list(initial)
    x0 = np.array([float(initial[k]) for k in names])
    scale = np.where(x0 != 0, np.abs(x0), 1.0)
    lo = np.full(len(names), -np.inf)
    hi = np.full(len(names), np.inf)
    for i, k in enumerate(names):
        if bounds and k in bounds:
            lo[i], hi[i] = bounds[k]

    def resid(x):
        p = dict(zip(names, x * scale))
        r = np.log(np.asarray(model(f, **p), complex) / h)
        r = r.real + 1j * ((r.imag + np.pi) % (2 * np.pi) - np.pi)
        return np.concatenate([r.real * w, r.imag * w])

    sol = least_squares(resid, x0 / scale, bounds=(lo / scale, hi / scale), x_scale="jac")
    params = dict(zip(names, (sol.x * scale).tolist()))
    r = resid(sol.x)
    n = f.size
    return ModelFit(
        params=params,
        rms_error_db=float(np.sqrt(np.mean((r[:n] / w) ** 2)) * _NEPER_DB),
        rms_error_deg=float(np.degrees(np.sqrt(np.mean((r[n:] / w) ** 2)))),
        model=model,
    )
