"""Single-frequency analysis of sampled records.

Everything here works on a uniformly sampled record and needs no instrument.
The core is :func:`tone`, a windowed single-frequency DFT that keeps whole
cycles and corrects for the window gain, so a pure sine of peak ``A`` returns
a phasor of magnitude ``A / sqrt(2)`` (RMS) and its phase. Around it:
frequency refinement for unlocked clocks, harmonics and THD, noise floor and
SNR, spurs, alias prediction, clipping and low-signal checks, and a segment
analysis that shows drift, jitter and amplitude modulation.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import scipy.fft as _fft
from scipy.signal import find_peaks
from scipy.signal import windows as _sw

WINDOWS = ("blackman", "blackmanharris", "kaiser", "flattop", "hann", "rect")
ESTIMATORS = ("dft", "coherent", "fit", "fit4", "multifit")
SEARCHES = ("fft", "czt")

# Main-lobe half width in bins, used to blank a tone out of a spectrum and to
# keep neighbouring tones apart. Kaiser is beta 14.
_MAINLOBE_BINS = {
    "rect": 1,
    "hann": 2,
    "blackman": 3,
    "blackmanharris": 4,
    "kaiser": 5,
    "flattop": 5,
}


@lru_cache(maxsize=32)
def _window_cached(n: int, name: str) -> np.ndarray:
    if name == "blackman":
        w = _sw.blackman(n, sym=True)
    elif name == "flattop":
        w = _sw.flattop(n, sym=True)
    elif name == "hann":
        w = _sw.hann(n, sym=True)
    elif name == "blackmanharris":
        w = _sw.blackmanharris(n, sym=True)
    elif name == "kaiser":
        w = _sw.kaiser(n, 14.0, sym=True)
    elif name == "rect":
        w = np.ones(n)
    else:
        raise ValueError(f"window {name!r} is not one of {WINDOWS}")
    w.setflags(write=False)
    return w


def make_window(n: int, name: str = "blackman") -> np.ndarray:
    """Return a symmetric window of length ``n`` (cached, read-only)."""
    return _window_cached(int(n), name)


def enbw_bins(name: str) -> float:
    """Equivalent noise bandwidth of a window, in bins."""
    w = make_window(4096, name)
    return float(len(w) * np.sum(w**2) / np.sum(w) ** 2)


def _time(x: np.ndarray, t=None, dt=None) -> np.ndarray:
    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.shape != x.shape:
            raise ValueError("t and x must be the same length")
        return t
    if dt is None or not dt > 0:
        raise ValueError("give a time axis t or a sample interval dt > 0")
    return np.arange(x.size, dtype=float) * float(dt)


def whole_cycle_length(n: int, dt: float, freq: float) -> int:
    """Samples that span the largest whole number of cycles of ``freq``, at least one."""
    total = freq * dt * (n - 1)
    k = max(1, int(math.floor(total)))
    need = int(math.ceil(k / (freq * dt) - 1e-9)) + 1
    return min(n, need)


def _axis(x: np.ndarray, t=None, dt=None) -> tuple[float, float]:
    """(first time, sample step) of a uniformly sampled record, without building arrays."""
    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.shape != x.shape:
            raise ValueError("t and x must be the same length")
        step = float(t[-1] - t[0]) / (t.size - 1) if t.size > 1 else 1.0
        return float(t[0]), step
    if dt is None or not dt > 0:
        raise ValueError("give a time axis t or a sample interval dt > 0")
    return 0.0, float(dt)


# cos and sin of the full record for one frequency, kept so every window, slice
# and channel of one capture reuses them: the trigonometry is the costly part.
_CEXP: OrderedDict = OrderedDict()
_CEXP_MAX = 24
_BASIS: OrderedDict = OrderedDict()
_BASIS_MAX = 64


def _cexp(freq: float, n: int, t0: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    key = (freq, n, t0, step)
    hit = _CEXP.get(key)
    if hit is not None:
        _CEXP.move_to_end(key)
        return hit
    ph = (-2.0 * np.pi * freq) * (t0 + step * np.arange(n, dtype=np.float64))
    out = (np.cos(ph), np.sin(ph))
    _CEXP[key] = out
    if len(_CEXP) > _CEXP_MAX:
        _CEXP.popitem(last=False)
    return out


def _basis(freq: float, n: int, t0: float, step: float, window: str, whole: bool):
    key = (freq, n, t0, step, window, whole)
    hit = _BASIS.get(key)
    if hit is not None:
        _BASIS.move_to_end(key)
        return hit
    n_used = whole_cycle_length(n, step, freq) if whole else n
    c, sn = _cexp(freq, n, t0, step)
    w = make_window(n_used, window)
    out = (n_used, w * c[:n_used], w * sn[:n_used], 2.0 / (n_used * w.mean()) / math.sqrt(2.0))
    _BASIS[key] = out
    if len(_BASIS) > _BASIS_MAX:
        _BASIS.popitem(last=False)
    return out


def tone(
    x,
    freq: float,
    *,
    t=None,
    dt=None,
    window: str = "blackman",
    whole_cycles: bool = True,
    detrend: bool = True,
) -> complex:
    """Complex RMS phasor of ``x`` at ``freq``.

    Phase is referenced to time zero of the time axis, so channels of one
    capture share a reference. ``whole_cycles`` cuts the record to the largest
    whole number of cycles first, as the old tool did. The windowed exponential
    is computed once per frequency and record and shared, and the sums run as
    two real dot products.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("x must be a 1-D record of at least 4 samples")
    if not freq > 0:
        raise ValueError("freq must be positive")
    t0, step = _axis(x, t, dt)
    n_used, br, bi, scale = _basis(float(freq), x.size, t0, step, window, whole_cycles)
    xs = x[:n_used]
    if detrend:
        xs = xs - xs.mean()
    return complex(float(xs @ br), float(xs @ bi)) * scale


def refine_frequency(
    x,
    freq_guess: float,
    *,
    t=None,
    dt=None,
    window: str = "blackman",
    search_bins: float = 2.0,
    tolerance_hz: float | None = None,
) -> float:
    """The frequency near ``freq_guess`` where the windowed DFT magnitude peaks.

    Use it when the source and receiver clocks are not locked. The search
    spans ``search_bins`` resolution bins either side of the guess, or
    ``tolerance_hz`` either side when that is wider. It takes the strongest
    bin of one windowed FFT (shared with the rest of the analysis), fits a
    parabola to the log magnitude of that bin and its neighbours, then fits
    once more on three exact DFT magnitudes 1/20 bin apart.
    """
    x = np.asarray(x, dtype=float)
    t0, step = _axis(x, t, dt)
    n = x.size
    rbw = 1.0 / (step * (n - 1))
    half = max(search_bins * rbw, tolerance_hz or 0.0)
    f, amp = spectrum(x, step, window=window)
    df = f[1] - f[0]
    lo = max(freq_guess - half, freq_guess * 0.5, df)
    hi = freq_guess + half
    i0 = max(1, int(math.floor(lo / df)))
    i1 = min(f.size - 2, int(math.ceil(hi / df)))
    if i1 < i0:
        return float(freq_guess)
    k = i0 + int(np.argmax(amp[i0 : i1 + 1]))
    la, lb, lc = np.log(np.maximum(amp[k - 1 : k + 2], 1e-300))
    den = la - 2 * lb + lc
    d = 0.5 * (la - lc) / den if den < 0 else 0.0
    fe = f[k] + max(-0.5, min(0.5, d)) * df
    h = 0.05 * df
    x0 = x - x.mean()
    m = []
    for fq in (fe - h, fe, fe + h):
        _, br, bi, _ = _basis(float(fq), n, t0, step, window, False)
        m.append(math.log(max(math.hypot(float(x0 @ br), float(x0 @ bi)), 1e-300)))
    den = m[0] - 2 * m[1] + m[2]
    if den < 0:
        fe = fe + max(-1.0, min(1.0, 0.5 * (m[0] - m[2]) / den)) * h
    return float(min(max(fe, lo), hi))


def alias_frequency(freq: float, sample_rate: float) -> float:
    """Where a tone at ``freq`` lands in 0 to sample_rate/2 after sampling."""
    if not sample_rate > 0:
        raise ValueError("sample_rate must be positive")
    a = abs(freq) % sample_rate
    return min(a, sample_rate - a)


_SPECTRA: OrderedDict = OrderedDict()
_SPECTRA_MAX = 8


def spectrum(x, dt: float, *, window: str = "blackman"):
    """Single-sided RMS amplitude spectrum: (freqs, rms per bin).

    Computed once per record, window and step (scipy.fft on every core) and
    kept, so the frequency search, the noise estimate, the strongest-tone
    search and the saved spectrum of one capture share one FFT.
    """
    key = (id(x), getattr(x, "size", None), float(dt), window)
    hit = _SPECTRA.get(key)
    if hit is not None and hit[0] is x:
        _SPECTRA.move_to_end(key)
        return hit[1], hit[2]
    arr = np.asarray(x, dtype=float)
    w = make_window(arr.size, window)
    y = _fft.rfft((arr - arr.mean()) * w, workers=-1)
    f = _fft.rfftfreq(arr.size, dt)
    amp = np.abs(y) * (2.0 / np.sum(w) / math.sqrt(2.0))
    amp[0] /= 2.0
    _SPECTRA[key] = (x, f, amp)
    if len(_SPECTRA) > _SPECTRA_MAX:
        _SPECTRA.popitem(last=False)
    return f, amp


def reduced_spectrum(x, dt: float, *, bins: int = 512, window: str = "blackman"):
    """A log-spaced peak-hold spectrum: (centre freqs, RMS per band).

    Small enough to keep for every point of a sweep, so a waterfall of the
    whole sweep can be drawn later without the raw records. Each band holds
    the largest bin inside it, so a tone keeps its level.
    """
    f, amp = spectrum(x, dt, window=window)
    f, amp = f[1:], amp[1:]
    if f.size == 0:
        return np.array([]), np.array([])
    edges = np.geomspace(f[0], f[-1] * 1.000001, bins + 1)
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, bins - 1)
    # f rises, so idx rises: each band is one run of samples.
    starts = np.flatnonzero(np.r_[True, np.diff(idx) > 0])
    peaks = np.maximum.reduceat(amp, starts)
    centres = np.sqrt(edges[idx[starts]] * edges[idx[starts] + 1])
    return centres, peaks


@dataclass
class Spur:
    freq: float
    rms: float
    dbc: float


@dataclass
class SignalQuality:
    noise_rms: float
    noise_floor_rms_per_bin: float
    snr_db: float
    thd_pct: float
    sinad_db: float
    harmonics: list[complex]
    spurs: list[Spur] = field(default_factory=list)
    estimate_noise_rms: float = float("nan")
    estimate_snr_db: float = float("nan")
    phase_uncertainty_deg: float = float("nan")


def quality(
    x,
    dt: float,
    freq: float,
    fundamental: complex,
    *,
    window: str = "blackman",
    harmonics: int = 5,
    other_tones=(),
    n_spurs: int = 5,
    t=None,
    estimate=None,
) -> SignalQuality:
    """Harmonics, THD, noise, SNR, SINAD and the largest spurs of one record.

    Noise is everything left in the spectrum after the fundamental, its
    harmonics, the other known tones and DC are blanked. Harmonics above
    Nyquist are folded to where they alias, because that is where their
    energy lands.
    """
    x = np.asarray(x, dtype=float)
    fs = 1.0 / dt
    hs = []
    for k in range(2, harmonics + 2):
        fk = k * freq
        if fk < fs / 2:
            if estimate is not None:
                hs.append(estimate(fk))
            else:
                hs.append(tone(x, fk, t=t, dt=dt, window=window, whole_cycles=window == "rect"))
        else:
            hs.append(complex(np.nan, np.nan))
    f, amp = spectrum(x, dt, window=window)
    df = f[1] - f[0]
    half = _MAINLOBE_BINS.get(window, 3) + 1
    keep = np.ones(f.size, dtype=bool)
    keep[: half + 1] = False
    blank = [freq] + [alias_frequency(k * freq, fs) for k in range(2, harmonics + 2)]
    blank += [alias_frequency(ft, fs) for ft in other_tones]
    for fb in blank:
        i = int(round(fb / df))
        keep[max(0, i - half) : i + half + 1] = False
    enbw = enbw_bins(window)
    noise_bins = amp[keep]
    noise_rms = float(np.sqrt(np.sum(noise_bins**2) / enbw)) if noise_bins.size else float("nan")
    floor = float(np.median(noise_bins)) if noise_bins.size else float("nan")
    # The single-frequency estimate sees only the noise in its own bin, so its
    # uncertainty is the local noise per bin, not the wideband noise.
    i0 = int(round(freq / df))
    near = np.zeros(f.size, dtype=bool)
    near[max(0, i0 - 60) : i0 + 61] = True
    local = amp[keep & near]
    est_noise = float(np.sqrt(np.mean(local**2))) if local.size else floor
    fund = abs(fundamental)
    hmag = np.array([abs(h) for h in hs if np.isfinite(h)])
    harm_rms = float(np.sqrt(np.sum(hmag**2))) if hmag.size else 0.0
    thd = 100.0 * harm_rms / fund if fund > 0 else float("nan")
    snr = 20 * math.log10(fund / noise_rms) if noise_rms > 0 and fund > 0 else float("inf")
    nad = math.hypot(noise_rms, harm_rms)
    sinad = 20 * math.log10(fund / nad) if nad > 0 and fund > 0 else float("inf")
    spurs = []
    if noise_bins.size and n_spurs > 0:
        masked = np.where(keep, amp, 0.0)
        idx, _ = find_peaks(masked, height=max(floor * 10.0, 1e-30))
        for i in idx[np.argsort(masked[idx])[::-1]][:n_spurs]:
            r = float(masked[i])
            spurs.append(Spur(float(f[i]), r, 20 * math.log10(r / fund) if fund > 0 else 0.0))
    est_snr = 20 * math.log10(fund / est_noise) if est_noise > 0 and fund > 0 else float("inf")
    ph_unc = math.degrees(est_noise / fund) if fund > 0 else float("nan")
    return SignalQuality(noise_rms, floor, snr, thd, sinad, hs, spurs, est_noise, est_snr, ph_unc)


@dataclass
class Stability:
    """What happens to the tone across the record, from equal segments."""

    freq_offset_hz: float
    drift_hz_per_s: float
    phase_jitter_deg: float
    amplitude_variation_pct: float
    segments: int


def _segments_clear(segments: int, freq: float, tones, n: int, step: float) -> int:
    """Fewest-cut segment count whose Hann segments keep the other tones out.

    A segment is 1/segments of the record, so its bins are segments times
    wider. A neighbour tone inside 4 segment bins leaks into every segment and
    reads as drift, jitter and amplitude change that are not there. Returns 0
    (no stability result) when even 3 segments cannot keep it out.
    """
    if not tones:
        return segments
    nearest = min(abs(float(ft) - freq) for ft in tones)
    fit = int(nearest * n * step / 4.0)
    s = min(segments, fit)
    return s if s >= 3 else 0


def stability(x, dt: float, freq: float, *, segments: int = 8, t=None) -> Stability:
    """Split the record into segments and track the tone's phase and amplitude.

    A straight phase slope is a frequency offset. Curvature is drift. The
    scatter left after a quadratic fit is phase jitter. The spread of segment
    amplitudes is amplitude variation, a sign of modulation or a changing
    device.
    """
    x = np.asarray(x, dtype=float)
    if segments < 3:
        return Stability(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    t0, step = _axis(x, t, dt)
    n = x.size // segments
    if n < 8 or freq * n * step < 1.0:
        return Stability(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    c, sn = _cexp(float(freq), x.size, t0, step)
    w = make_window(n, "hann")
    m = segments * n
    xs = x[:m].reshape(segments, n)
    xs = xs - xs.mean(axis=1, keepdims=True)
    re = np.einsum("ij,ij->i", xs * w, c[:m].reshape(segments, n))
    im = np.einsum("ij,ij->i", xs * w, sn[:m].reshape(segments, n))
    p = (re + 1j * im) * (2.0 / (n * w.mean()) / math.sqrt(2.0))
    mids = t0 + step * (np.arange(segments) * n + (n - 1) / 2.0)
    mids = mids - mids.mean()
    phs = np.unwrap(np.angle(p))
    amps = np.abs(p)
    c2, c1, c0 = np.polyfit(mids, phs, 2)
    resid = phs - np.polyval([c2, c1, c0], mids)
    return Stability(
        freq_offset_hz=float(c1 / (2 * np.pi)),
        drift_hz_per_s=float(2 * c2 / (2 * np.pi)),
        phase_jitter_deg=float(np.degrees(np.std(resid))),
        amplitude_variation_pct=float(100 * np.std(amps) / np.mean(amps)) if amps.mean() else 0.0,
        segments=segments,
    )


def clip_fraction(x, *, full_scale: tuple[float, float] | None = None) -> float:
    """Fraction of samples that sit on the record's extremes or on a full-scale limit.

    Without ``full_scale`` it counts samples equal to the record minimum or
    maximum: a clipped record piles many samples onto one code.
    """
    x = np.asarray(x, dtype=float)
    if full_scale is not None:
        lo, hi = full_scale
        return float(np.mean((x <= lo) | (x >= hi)))
    span = x.max() - x.min()
    if span <= 0:
        return 1.0
    tol = span * 1e-6
    return float(np.mean((x >= x.max() - tol) | (x <= x.min() + tol)))


# Guidelines for the flags. A user can pass different values to analyze().
DEFAULT_FLAG_LIMITS = {
    "clip_fraction": 0.01,
    "low_snr_db": 30.0,
    "thd_pct": 5.0,
    "amplitude_variation_pct": 2.0,
    "phase_jitter_deg": 2.0,
    "sideband_dbc": -40.0,
    "sideband_fraction": 0.2,
    "repeat_spread_pct": 1.0,
    "repeat_spread_deg": 1.0,
    "settle_pct": 0.2,
    "peak_off_target_bins": 0.5,
    "coherent_off_bins": 1e-3,
    "coherent_leak_db": -40.0,
    "coherent_leak_excess_db": 10.0,
    "strongest_elsewhere_db": 0.0,
    "settle_deg": 0.2,
}


@dataclass
class ToneResult:
    """Everything measured about one signal at one tone of one capture."""

    signal: str
    freq: float
    phasor: complex
    order: int = 1
    quality: SignalQuality | None = None
    stability: Stability | None = None
    clip: float = 0.0
    flags: list[str] = field(default_factory=list)
    offset_bins: float = float("nan")
    extra: dict = field(default_factory=dict)
    strongest_freq: float = float("nan")
    strongest_dbc: float = float("nan")

    @property
    def rms(self) -> float:
        return abs(self.phasor)

    @property
    def phase_deg(self) -> float:
        return math.degrees(np.angle(self.phasor))


def _flags(r: ToneResult, freq_cmd: float, rbw: float, limits: dict) -> list[str]:
    out = []
    if r.clip > limits["clip_fraction"]:
        out.append("clip")
    q = r.quality
    if q is not None:
        if q.estimate_snr_db < limits["low_snr_db"]:
            out.append("low_snr")
        if np.isfinite(q.thd_pct) and q.thd_pct > limits["thd_pct"]:
            out.append("nonlinear")
        for s in q.spurs:
            near = abs(s.freq - r.freq) < limits["sideband_fraction"] * r.freq
            if near and s.dbc > limits["sideband_dbc"]:
                out.append("sideband")
                break
    st = r.stability
    if st is not None and st.segments:
        if st.amplitude_variation_pct > limits["amplitude_variation_pct"]:
            out.append("amplitude_varies")
        if st.phase_jitter_deg > limits["phase_jitter_deg"]:
            out.append("phase_jitter")
    if rbw > 0 and abs(r.offset_bins) > limits["peak_off_target_bins"]:
        out.append("peak_off_target")
    if np.isfinite(r.strongest_dbc) and r.strongest_dbc > limits["strongest_elsewhere_db"]:
        out.append("strongest_elsewhere")
    return out


def analyze(
    signals: dict,
    freq: float,
    *,
    t=None,
    dt=None,
    window: str = "blackman",
    tones=(),
    harmonics: int = 5,
    refine: bool | str = True,
    reference: str | None = None,
    full_scale: dict | None = None,
    segments: int = 8,
    limits: dict | None = None,
    tolerance: float | None = None,
    tolerance_hz: float | None = None,
    estimator: str = "dft",
    search: str = "fft",
) -> tuple[float, dict[str, list[ToneResult]]]:
    """Analyze every signal of one capture at ``freq`` and at the extra ``tones``.

    ``refine`` finds the frequency actually received, once per capture, from
    the ``reference`` signal (default: the strongest), and uses it for every
    signal so their phases stay comparable. Pass ``refine=False`` when the
    source and receiver are locked. Returns (frequency used, results by
    signal name, fundamental first).

    ``tolerance`` (a fraction, 1e-3 is 0.1 %) or ``tolerance_hz`` widens the
    search for the received tone; without either the search spans 2 bins.
    Either way each fundamental reports how many bins the peak sat from the
    expected frequency and where the strongest peak of the whole record is.
    Flags: ``peak_off_target`` (the expected bin is not the peak),
    ``peak_at_search_edge`` (the true peak may lie outside the search) and
    ``strongest_elsewhere`` (a stronger tone sits outside the search).

    ``estimator`` picks how each value is read: ``"dft"`` (windowed DFT at
    the exact frequency), ``"coherent"`` (one FFT bin of a coherent record, no
    window), ``"fit"`` (3-parameter sine fit), ``"fit4"`` (4-parameter fit,
    frequency included) or ``"multifit"`` (one joint fit of every tone and
    harmonic of the capture). ``search`` picks how an unlocked tone is found:
    ``"fft"`` (FFT peak and interpolation) or ``"czt"`` (chirp-Z zoom over the
    tolerance window). Both end with a fit on exact DFT values. The guide
    lists the best use and the blind spot of each.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator {estimator!r} is not one of {ESTIMATORS}")
    if search not in SEARCHES:
        raise ValueError(f"search {search!r} is not one of {SEARCHES}")
    from sweep import estimators as _est

    limits = {**DEFAULT_FLAG_LIMITS, **(limits or {})}
    # A tapered window over hundreds of cycles needs no whole-cycle cut, and one
    # window length then serves every tone of the capture.
    whole = window == "rect" and estimator == "dft"
    names = list(signals)
    arrays = {k: np.asarray(v, dtype=float) for k, v in signals.items()}
    first = arrays[names[0]]
    tt = _time(first, t, dt)
    step = float(np.median(np.diff(tt)))
    rbw = 1.0 / (tt[-1] - tt[0])
    f_used = freq
    window_hz = max(2.0 * rbw, (tolerance or 0.0) * freq, tolerance_hz or 0.0)
    ref = reference
    t0, _ = _axis(first, tt, None)
    if refine and estimator != "coherent":
        if ref is None:
            ref = max(
                names,
                key=lambda k: abs(tone(arrays[k], freq, t=tt, window=window, whole_cycles=whole)),
            )
        guess = freq
        if search == "czt":
            guess = _est.czt_search(arrays[ref], t0, step, freq, window_hz, window=window)
        f_used = refine_frequency(
            arrays[ref],
            guess,
            t=tt,
            window=window,
            tolerance_hz=window_hz if search == "fft" else 2.0 * rbw,
        )
        f_used = min(max(f_used, freq - window_hz), freq + window_hz)
        if estimator == "fit4":
            f_used = _est.sine_fit(arrays[ref], t0, step, [f_used], fit_freq=True)["freq"]
    at_edge = bool(refine) and estimator != "coherent" and abs(f_used - freq) >= 0.98 * window_hz
    nyq = 0.5 / step
    harm_freqs = [k * f_used for k in range(2, harmonics + 2) if k * f_used < nyq]

    def reader(x):
        """The chosen estimator as a function of frequency, for one signal."""
        extra = {}
        if estimator == "dft":
            return lambda fq: tone(x, fq, t=tt, window=window, whole_cycles=whole), extra
        if estimator == "coherent":

            def rd(fq):
                p, off, leak, excess = _est.coherent_phasor(x, t0, step, fq)
                if fq == f_used:
                    extra.update(off_bins=off, leak_db=leak, leak_excess_db=excess)
                return p

            return rd, extra
        if estimator in ("fit", "fit4"):
            return (lambda fq: _est.sine_fit(x, t0, step, [fq])["phasors"][float(fq)]), extra
        joint = _est.sine_fit(x, t0, step, [f_used, *tones, *harm_freqs])
        extra["residual_rms"] = joint["residual_rms"]

        def rd(fq):
            ph = joint["phasors"]
            key = min(ph, key=lambda k: abs(k - fq))
            if abs(key - fq) <= 1e-9 * max(1.0, fq):
                return ph[key]
            return _est.sine_fit(x, t0, step, [fq])["phasors"][float(fq)]

        return rd, extra

    out: dict[str, list[ToneResult]] = {}
    for name in names:
        x = arrays[name]
        rd, extra = reader(x)
        fund = rd(f_used)
        q = quality(
            x,
            step,
            f_used,
            fund,
            window=window,
            harmonics=harmonics,
            other_tones=tones,
            t=tt,
            estimate=rd if estimator != "dft" else None,
        )
        st = stability(
            x, step, f_used, segments=_segments_clear(segments, f_used, tones, x.size, step), t=tt
        )
        fs_pair = (full_scale or {}).get(name)
        r = ToneResult(name, f_used, fund, 1, q, st, clip_fraction(x, full_scale=fs_pair))
        r.offset_bins = (f_used - freq) / rbw
        sf, sa = spectrum(x, step, window=window)
        lo_bin = _MAINLOBE_BINS.get(window, 3) + 1
        outside = np.abs(sf - f_used) > window_hz
        outside[: lo_bin + 1] = False
        for ft in tones:  # known tones of the same capture are not "elsewhere"
            outside &= np.abs(sf - ft) > window_hz
        if outside.any() and abs(fund) > 0:
            k = int(np.argmax(np.where(outside, sa, 0.0)))
            r.strongest_freq = float(sf[k])
            r.strongest_dbc = float(20 * math.log10(max(sa[k], 1e-300) / abs(fund)))
        r.flags = _flags(r, freq, rbw, limits)
        if at_edge:
            r.flags.append("peak_at_search_edge")
        if estimator == "coherent":
            if abs(extra.get("off_bins", 0.0)) > limits["coherent_off_bins"]:
                r.flags.append("not_coherent")
            if (
                extra.get("leak_db", -999.0) > limits["coherent_leak_db"]
                and extra.get("leak_excess_db", -999.0) > limits["coherent_leak_excess_db"]
            ):
                r.flags.append("leakage")
        r.extra = extra
        results = [r]
        for ft in tones:
            results.append(ToneResult(name, float(ft), rd(ft), 0))
        for k, h in enumerate(q.harmonics, start=2):
            results.append(ToneResult(name, k * f_used, h, k))
        out[name] = results
    return f_used, out
