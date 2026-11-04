"""Plots made from a run folder, in one house style.

Every plot reads only saved data, so the same figure comes out during a run,
after it, and months later. Nothing here talks to an instrument.

House style: frequency on a shared log axis rounded out to whole decades,
major and minor grids, engineering unit labels, dot markers, unwrapped phase,
flagged points circled in red, and a title that names the run and its status.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, LogLocator

from sweep.store import Run, Series, safe_name

_DB_UNITS = {"", "V/V", "A/A", "W/W", "1", "ratio"}
_FLAG_STYLE = dict(marker="o", ms=8, mfc="none", mec="red", mew=1.2, ls="none")


def _as_run(run) -> Run:
    return run if isinstance(run, Run) else Run(run)


def _style_x(ax, fmin: float, fmax: float, log: bool) -> None:
    ax.grid(which="major", linestyle="-", linewidth=0.8)
    ax.grid(which="minor", linestyle="--", linewidth=0.5)
    if log and fmin > 0:
        ax.set_xscale("log")
        ax.set_xlim(
            10 ** math.floor(math.log10(fmin) + 1e-3), 10 ** math.ceil(math.log10(fmax) - 1e-3)
        )
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=50))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=50))
    else:
        pad = 0.02 * (fmax - fmin or 1.0)
        ax.set_xlim(fmin - pad, fmax + pad)
        ax.minorticks_on()


def _min_span(ax, what: str) -> None:
    """Keep a flat response from filling the axis with noise.

    The y span is at least 1 % of the level for a linear magnitude, 1 dB for
    dB, and 2 degrees for phase, centred on the data.
    """
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        return
    mid = (lo + hi) / 2
    label = ax.get_ylabel()
    if what == "phase":
        need = 2.0
    elif "(dB)" in label:
        need = 1.0
    else:
        need = 0.01 * abs(mid)
    if hi - lo < need:
        ax.set_ylim(mid - need / 2, mid + need / 2)


def _rel_uncertainty(s: Series):
    """Relative 1-sigma uncertainty of a magnitude from its estimate SNR, or None."""
    est = s.estimate_snr_db
    if est is None or not np.isfinite(est).any():
        return None
    return np.where(np.isfinite(est), 10 ** (-est / 20), 0.0)


def is_ratio(s: Series) -> bool:
    return s.kind == "expr" and s.unit in _DB_UNITS


def magnitude(s: Series) -> tuple[np.ndarray, str]:
    """The magnitude to plot and its axis label: dB for unitless ratios."""
    if is_ratio(s):
        with np.errstate(divide="ignore"):
            return 20 * np.log10(s.mag), "dB"
    return s.mag, s.unit or "rms"


def _flag_mask(s: Series, ignore=()) -> np.ndarray:
    return np.array([any(f not in ignore for f in fl) for fl in s.flags], dtype=bool)


def _default_names(run: Run) -> tuple[list[str], list[str]]:
    exprs = run.names(kinds=("expr",))
    raw = run.names(kinds=("signal", "scalar"))
    return exprs, raw


def plot_run(
    run,
    *,
    names: list[str] | None = None,
    signals: bool = True,
    quality: bool = True,
    save: bool = True,
    dpi: int = 200,
    log_x: bool | None = None,
) -> Figure:
    """The standard figure for one run.

    Rows, top to bottom: each expression's magnitude (ratios in dB, others in
    their unit), one phase row for every expression with a valid phase, each
    raw signal's amplitude, then SNR and THD of the raw signals. Saved as
    plot.png and plot.svg in the run folder.
    """
    run = _as_run(run)
    exprs, raw = _default_names(run)
    if names is not None:
        exprs = [n for n in names if n in exprs]
        raw = [n for n in names if n in raw]
    if not exprs and not raw:
        raise ValueError(f"{run.path.name} has no data to plot")
    ser = {n: run.series(n) for n in exprs + raw}
    by_level = run.meta.get("x_axis") == "amplitude"
    if by_level:
        ser = {n: replace(s, freqs=s.amplitudes) for n, s in ser.items()}
    allf = np.concatenate([s.freqs for s in ser.values()])
    fmin, fmax = float(np.nanmin(allf)), float(np.nanmax(allf))
    if log_x is None:
        log_x = fmin > 0 and fmax / fmin > 20

    mag_rows = [[n] for n in exprs]
    phase_names = [n for n in exprs if ser[n].phase_valid]
    raw_rows = [[n] for n in raw] if signals or not exprs else []
    q_names = [n for n in raw if ser[n].kind == "signal"] if quality else []
    has_thd = q_names and any(np.isfinite(ser[n].thd_pct).any() for n in q_names)
    layout = (
        [("mag", r) for r in mag_rows]
        + ([("phase", phase_names)] if phase_names else [])
        + [("amp", r) for r in raw_rows]
        + ([("snr", q_names)] if q_names else [])
        + ([("thd", q_names)] if has_thd else [])
    )
    fig = Figure(figsize=(10, 2.4 * len(layout) + 0.8), constrained_layout=True)
    axes = fig.subplots(len(layout), 1, sharex=True, squeeze=False)[:, 0]
    colors = [c["color"] for c in __import__("matplotlib").rcParams["axes.prop_cycle"]]
    color = {n: colors[i % len(colors)] for i, n in enumerate(exprs + raw)}

    for ax, (what, group) in zip(axes, layout):
        for n in group:
            s = ser[n]
            if what in ("mag", "amp"):
                y, unit = magnitude(s)
                ax.plot(s.freqs, y, ".-", ms=4, lw=1, color=color[n], label=n)
                rel = _rel_uncertainty(s)
                if rel is not None:
                    if unit == "dB":
                        lo = 20 * np.log10(np.clip(1 - 3 * rel, 1e-3, None))
                        hi = 20 * np.log10(1 + 3 * rel)
                        ax.fill_between(
                            s.freqs,
                            y + lo,
                            y + hi,
                            color=color[n],
                            alpha=0.2,
                            lw=0,
                            label="3 sigma",
                        )
                    else:
                        ax.fill_between(
                            s.freqs,
                            y * np.clip(1 - 3 * rel, 0, None),
                            y * (1 + 3 * rel),
                            color=color[n],
                            alpha=0.2,
                            lw=0,
                            label="3 sigma",
                        )
                fm = _flag_mask(s)
                if fm.any():
                    ax.plot(s.freqs[fm], y[fm], label="flagged", **_FLAG_STYLE)
                if unit == "dB":
                    ax.set_ylabel(f"|{n}| (dB)")
                else:
                    if (
                        np.all(y[np.isfinite(y)] > 0)
                        and np.nanmax(y) / max(np.nanmin(y), 1e-300) > 30
                    ):
                        ax.set_yscale("log")
                    ax.yaxis.set_major_formatter(EngFormatter(unit=unit))
                    ax.set_ylabel(f"|{n}|")
                ax.set_title(("Magnitude " if what == "mag" else "Amplitude ") + n, fontsize=9)
            elif what == "phase":
                ax.plot(s.freqs, s.phase_deg, ".-", ms=4, lw=1, color=color[n], label=n)
                pu = s.phase_uncertainty_deg
                if pu is not None and np.isfinite(pu).any():
                    ax.fill_between(
                        s.freqs,
                        s.phase_deg - 3 * pu,
                        s.phase_deg + 3 * pu,
                        color=color[n],
                        alpha=0.2,
                        lw=0,
                        label="3 sigma",
                    )
                ax.set_ylabel("Phase (deg)")
                ax.set_title("Phase", fontsize=9)
            elif what == "snr":
                est = s.estimate_snr_db
                if est is not None and np.isfinite(est).any():
                    ax.plot(s.freqs, est, ".-", ms=4, lw=1, color=color[n], label=f"{n} estimate")
                ax.plot(
                    s.freqs,
                    s.snr_db,
                    ".--",
                    ms=3,
                    lw=0.8,
                    color=color[n],
                    alpha=0.6,
                    label=f"{n} wideband",
                )
                ax.set_ylabel("SNR (dB)")
                ax.set_title(
                    "Signal to noise: in the estimate (solid), whole record (dashed)", fontsize=9
                )
            elif what == "thd":
                ax.plot(s.freqs, s.thd_pct, ".-", ms=4, lw=1, color=color[n], label=n)
                ax.set_ylabel("THD (%)")
                pos = s.thd_pct[np.isfinite(s.thd_pct) & (s.thd_pct > 0)]
                if pos.size and pos.max() / pos.min() > 30:
                    ax.set_yscale("log")
                ax.set_title("Total harmonic distortion", fontsize=9)
        if what in ("mag", "amp", "phase"):
            _min_span(ax, what)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        if len(uniq) > 1:
            ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")
        _style_x(ax, fmin, fmax, log_x)
    if by_level:
        axes[-1].set_xlabel(f"Drive amplitude at {run.meta.get('level_freq', 0):.6g} Hz")
        axes[-1].xaxis.set_major_formatter(EngFormatter(unit="Vpp"))
    else:
        axes[-1].set_xlabel("Frequency")
        axes[-1].xaxis.set_major_formatter(EngFormatter(unit="Hz"))
    status = run.meta.get("status", "")
    fig.suptitle(f"{run.title}   {run.meta.get('started', '')}   [{status}]", fontsize=10)
    if save:
        fig.savefig(run.path / "plot.png", dpi=dpi)
        fig.savefig(run.path / "plot.svg")
    return fig


def overlay_runs(
    runs,
    name: str,
    *,
    labels: list[str] | None = None,
    save_to=None,
    dpi: int = 200,
) -> Figure:
    """Overlay one named quantity from several runs: magnitude and, if valid, phase."""
    runs = [_as_run(r) for r in runs]
    labels = labels or [r.title for r in runs]
    ser = [r.series(name) for r in runs]
    show_phase = all(s.phase_valid for s in ser)
    fig = Figure(figsize=(10, 6 if show_phase else 3.4), constrained_layout=True)
    axes = fig.subplots(2 if show_phase else 1, 1, sharex=True, squeeze=False)[:, 0]
    allf = np.concatenate([s.freqs for s in ser])
    fmin, fmax = float(allf.min()), float(allf.max())
    for s, lab in zip(ser, labels):
        y, unit = magnitude(s)
        axes[0].plot(s.freqs, y, ".-", ms=4, lw=1, label=lab)
        if show_phase:
            axes[1].plot(s.freqs, s.phase_deg, ".-", ms=4, lw=1, label=lab)
    y_unit = magnitude(ser[0])[1]
    if y_unit == "dB":
        axes[0].set_ylabel(f"|{name}| (dB)")
    else:
        axes[0].set_yscale("log")
        axes[0].yaxis.set_major_formatter(EngFormatter(unit=y_unit))
        axes[0].set_ylabel(f"|{name}|")
    axes[0].legend(fontsize=8)
    if show_phase:
        axes[1].set_ylabel("Phase (deg)")
    for ax in axes:
        _style_x(ax, fmin, fmax, fmax / fmin > 20)
    axes[-1].set_xlabel("Frequency")
    axes[-1].xaxis.set_major_formatter(EngFormatter(unit="Hz"))
    fig.suptitle(f"{name}: {len(runs)} runs", fontsize=10)
    if save_to:
        p = Path(save_to)
        fig.savefig(p.with_suffix(".png"), dpi=dpi)
        fig.savefig(p.with_suffix(".svg"))
    return fig


def plot_harmonics(run, name: str, *, max_order: int = 6, save: bool = True) -> Figure:
    """Each harmonic of one signal in dB relative to its fundamental."""
    run = _as_run(run)
    fund = run.series(name, kind="signal")
    hs = run.harmonics(name, max_order)
    fig = Figure(figsize=(10, 3.4), constrained_layout=True)
    ax = fig.subplots()
    for k, h in hs.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(fund.freqs, 20 * np.log10(h.mag / fund.mag), ".-", ms=4, lw=1, label=f"H{k}")
    ax.set_ylabel("dBc")
    ax.legend(fontsize=8, ncol=len(hs) or 1)
    _style_x(ax, fund.freqs.min(), fund.freqs.max(), fund.freqs.max() / fund.freqs.min() > 20)
    ax.set_xlabel("Fundamental frequency")
    ax.xaxis.set_major_formatter(EngFormatter(unit="Hz"))
    fig.suptitle(f"{run.title}: harmonics of {name}", fontsize=10)
    if save:
        fig.savefig(run.path / f"harmonics_{safe_name(name)}.png", dpi=200)
    return fig


def plot_waveform(run, point: int, *, save: bool = True) -> Figure:
    """The raw record of one point, when waveforms were saved."""
    run = _as_run(run)
    w = run.waveform(point)
    fig = Figure(figsize=(10, 3.4), constrained_layout=True)
    ax = fig.subplots()
    for n, y in w["signals"].items():
        ax.plot(w["t"], y, lw=0.8, label=f"{n} ({w['units'].get(n, '')})")
    ax.xaxis.set_major_formatter(EngFormatter(unit="s"))
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.set_title(f"{run.title}: point {point} at {w['freq']:.6g} Hz", fontsize=10)
    if save:
        fig.savefig(run.path / f"waveform_{point:04d}.png", dpi=150)
    return fig


def _meas_groups(run: Run) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in run.instrument_measurements():
        name = r.get("label") or f"{r.get('type', '')} {r.get('source', '')}".strip()
        key = f"{r.get('measurement', '')}: MEAS{r.get('slot')} {name}".strip()
        groups.setdefault(key, []).append(r)
    return groups


def plot_instrument_measurements(run, *, save: bool = True, dpi: int = 200) -> Figure | None:
    """Every instrument measurement saved per point against frequency, one row each.

    The line is the current-acquisition value and the band its min to max
    within that acquisition. Saved as instrument_measurements.png.
    """
    run = _as_run(run)
    groups = _meas_groups(run)
    if not groups:
        return None
    fig = Figure(figsize=(10, 1.9 * len(groups) + 0.8), constrained_layout=True)
    axes = fig.subplots(len(groups), 1, sharex=True, squeeze=False)[:, 0]
    allf = [r["freq_cmd"] for rows in groups.values() for r in rows]
    fmin, fmax = float(min(allf)), float(max(allf))
    for ax, (key, rows) in zip(axes, groups.items()):
        rows = sorted(rows, key=lambda r: r["freq_cmd"])
        f = np.array([r["freq_cmd"] for r in rows], float)
        y = np.array([r["value"] for r in rows], float)
        lo = np.array([r.get("curr_min", np.nan) for r in rows], float)
        hi = np.array([r.get("curr_max", np.nan) for r in rows], float)
        ax.plot(f, y, ".-", ms=4, lw=1)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(f, lo, hi, alpha=0.2, lw=0)
        unit = rows[0].get("unit") or ""
        ax.yaxis.set_major_formatter(EngFormatter(unit=unit))
        ax.set_title(key, fontsize=9)
        _style_x(ax, fmin, fmax, fmax / fmin > 20)
    axes[-1].set_xlabel("Frequency")
    axes[-1].xaxis.set_major_formatter(EngFormatter(unit="Hz"))
    fig.suptitle(f"{run.title}: instrument measurements at each point", fontsize=10)
    if save:
        fig.savefig(run.path / "instrument_measurements.png", dpi=dpi)
    return fig


def plot_waterfall(
    run,
    signal: str,
    *,
    bins: int = 600,
    floor_db: float | None = None,
    save: bool = True,
    dpi: int = 200,
) -> Figure | None:
    """Waterfall of one signal across the sweep: every point's spectrum as a row.

    Left: absolute frequency, so fixed tones (pickup, switchers, aliases) are
    vertical lines and the drive is the diagonal. Right: frequency over the
    drive frequency, so the drive and its harmonics are vertical lines at 1,
    2, 3 and fixed tones are diagonals. Colour is dB relative to 1 unit RMS.
    Saved as waterfall_<signal>.png.
    """
    run = _as_run(run)
    rows = run.spectra(signal)
    if len(rows) < 2:
        return None
    drive = np.array([r[0] for r in rows])
    fmin = min(float(r[1][0]) for r in rows)
    fmax = max(float(r[1][-1]) for r in rows)
    gx = np.geomspace(fmin, fmax, bins)
    gn = np.geomspace(0.05, 50.0, bins)
    img = np.full((len(rows), bins), np.nan)
    imn = np.full((len(rows), bins), np.nan)
    for i, (fd, f, a, _) in enumerate(rows):
        lf, la = np.log10(f), 20 * np.log10(np.clip(a, 1e-15, None))
        for grid, out, x in ((gx, img, np.log10(gx)), (gn, imn, np.log10(gn * fd))):
            v = np.interp(x, lf, la, left=np.nan, right=np.nan)
            out[i] = v
    finite = img[np.isfinite(img)]
    vmax = float(np.nanmax(finite)) if finite.size else 0.0
    vmin = (
        floor_db
        if floor_db is not None
        else (float(np.nanpercentile(finite, 5)) if finite.size else vmax - 100)
    )
    unit = rows[0][3] or "V"
    fig = Figure(figsize=(12, 5.5), constrained_layout=True)
    ax1, ax2 = fig.subplots(1, 2, sharey=True)
    for ax, x, im, label in (
        (ax1, gx, img, "Spectrum frequency"),
        (ax2, gn, imn, "Frequency / drive frequency"),
    ):
        mesh = ax.pcolormesh(x, drive, im, shading="nearest", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
    ax1.plot(drive, drive, color="white", lw=0.6, ls="--", alpha=0.6)
    for k in (1, 2, 3):
        ax2.axvline(k, color="white", lw=0.5, ls="--", alpha=0.5)
    ax1.xaxis.set_major_formatter(EngFormatter(unit="Hz"))
    ax1.yaxis.set_major_formatter(EngFormatter(unit="Hz"))
    ax1.set_ylabel("Drive frequency")
    fig.colorbar(mesh, ax=[ax1, ax2], label=f"dB re 1 {unit} rms")
    fig.suptitle(f"{run.title}: waterfall of {signal}", fontsize=10)
    if save:
        fig.savefig(run.path / f"waterfall_{safe_name(signal)}.png", dpi=dpi)
    return fig


IMPEDANCE_UNITS = {"ohm", "Ohm", "ohms", "Ω"}


def impedance_parts(s: Series) -> dict[str, np.ndarray]:
    """Series-equivalent parts of an impedance: |Z|, phase, R, X, L, C and Q = |X| / R.

    L = X / w where the reactance is positive (inductive), C = -1 / (w X) where
    it is negative (capacitive); each is NaN on the other side, and both are NaN
    where |X| is under 3 times its noise, so noise never reads as a part value.
    """
    z = s.values
    w = 2 * np.pi * s.freqs
    x = z.imag
    # X is resolved only where it stands above its own noise: |X| > 3 |Z| sigma_phase.
    pu = s.phase_uncertainty_deg
    sig = np.abs(z) * np.radians(pu) if pu is not None else np.full(z.shape, np.nan)
    sig = np.where(np.isfinite(sig), sig, 1e-3 * np.abs(z))
    resolved = np.abs(x) > 3 * sig
    with np.errstate(divide="ignore", invalid="ignore"):
        ind = np.where((x > 0) & resolved, x / w, np.nan)
        cap = np.where((x < 0) & resolved, -1.0 / (w * x), np.nan)
        q = np.abs(x) / np.where(z.real != 0, np.abs(z.real), np.nan)
    return {
        "mag": np.abs(z),
        "phase_deg": s.phase_deg,
        "R": z.real,
        "X": x,
        "L": ind,
        "C": cap,
        "Q": q,
    }


def plot_impedance(run, name: str = "Z", *, save: bool = True, dpi: int = 200) -> Figure:
    """The standard impedance figure: |Z|, phase, R, series L and series C against frequency.

    Saved as impedance_<name>.png and .svg. The phase row is left out when the
    phase is not valid for this quantity.
    """
    run = _as_run(run)
    s = run.series(name)
    parts = impedance_parts(s)
    rows = [
        ("mag", f"|{name}|", "ohm", True),
        ("phase_deg", "Phase", "deg", False),
        ("R", "R (series)", "ohm", False),
        ("X", "X (series reactance)", "ohm", False),
        ("L", "L (series, where X > 0)", "H", True),
        ("C", "C (series, where X < 0)", "F", True),
        ("Q", "Q = |X| / R", "", True),
    ]
    if not s.phase_valid:
        rows = rows[:1]
    fig = Figure(figsize=(10, 2.3 * len(rows) + 0.8), constrained_layout=True)
    axes = fig.subplots(len(rows), 1, sharex=True, squeeze=False)[:, 0]
    fmin, fmax = float(np.nanmin(s.freqs)), float(np.nanmax(s.freqs))
    flagged = _flag_mask(s)
    for ax, (key, title, unit, logy) in zip(axes, rows):
        y = parts[key]
        ok = np.isfinite(y)
        ax.plot(s.freqs[ok], y[ok], ".-", ms=4, lw=1)
        if flagged.any():
            fm = flagged & ok
            ax.plot(s.freqs[fm], y[fm], label="flagged", **_FLAG_STYLE)
        if logy and ok.any() and np.all(y[ok] > 0) and np.nanmax(y) / np.nanmin(y[ok]) > 30:
            ax.set_yscale("log")
        if key == "phase_deg":
            ax.set_ylabel("deg")
        elif key == "Q":
            ax.set_ylabel("Q")
        else:
            ax.yaxis.set_major_formatter(EngFormatter(unit="Ω" if unit == "ohm" else unit))
        if not ok.any():
            ax.text(
                0.5,
                0.5,
                "none resolved: |X| never stands 3 sigma above noise with this sign",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
        elif key in ("mag", "R"):
            _min_span(ax, "mag")
        ax.set_title(title, fontsize=9)
        _style_x(ax, fmin, fmax, fmax / fmin > 20)
    axes[-1].set_xlabel("Frequency")
    axes[-1].xaxis.set_major_formatter(EngFormatter(unit="Hz"))
    fig.suptitle(f"{run.title}: impedance {name}   {run.meta.get('started', '')}", fontsize=10)
    if save:
        fig.savefig(run.path / f"impedance_{safe_name(name)}.png", dpi=dpi)
        fig.savefig(run.path / f"impedance_{safe_name(name)}.svg")
    return fig
