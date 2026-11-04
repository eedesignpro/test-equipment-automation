"""AI-friendly views of a run: a condensed table and markdown review tables.

A full sweep can hold thousands of rows. These views give a person or an
agent the shape of the result in a few thousand tokens: about 100 points chosen
evenly in log frequency plus the extremes of each response, and a markdown
review with a summary per quantity, corner and peak estimates, quality, flags,
and a short table.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from sweep.plot import is_ratio, magnitude
from sweep.store import Run, Series, unwrap_finite


def _as_run(run) -> Run:
    return run if isinstance(run, Run) else Run(run)


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    mag = abs(v)
    if 1e-3 <= mag < 1e5:
        return f"{v:.{digits}g}"
    return f"{v:.{digits - 1}e}"


def _eng_hz(f: float) -> str:
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, "")):
        if abs(f) >= scale:
            return f"{f / scale:.4g} {suffix}Hz"
    return f"{f:.4g} Hz"


def pick_points(freqs: np.ndarray, n: int, keep: list[int] = (), *, log: bool = True) -> list[int]:
    """Indices of about ``n`` points evenly spaced in log (or linear) x, plus ``keep``."""
    m = len(freqs)
    if m <= n:
        return list(range(m))
    lf = np.log10(np.clip(freqs, 1e-300, None)) if log else np.asarray(freqs, dtype=float)
    targets = np.linspace(lf.min(), lf.max(), n)
    idx = {int(np.argmin(np.abs(lf - t))) for t in targets}
    idx |= {int(k) for k in keep}
    return sorted(idx)


def _quantities(run: Run) -> list[Series]:
    names = run.names(kinds=("expr",)) + run.names(kinds=("signal", "scalar"))
    return [run.series(n) for n in names]


def condensed(run, n: int = 100) -> tuple[list[str], list[list[str]]]:
    """Header and rows of the condensed table: about ``n`` frequencies."""
    run = _as_run(run)
    qs = _quantities(run)
    if not qs:
        return ["freq_hz"], []
    base = qs[0]
    keep = []
    for s in qs:
        y = s.mag
        if np.isfinite(y).any():
            keep += [int(np.nanargmax(y)), int(np.nanargmin(y))]
    by_level = run.meta.get("x_axis") == "amplitude"
    x = base.amplitudes if by_level and base.amplitudes is not None else base.freqs
    idx = pick_points(x, n, keep, log=not by_level)
    header = ["drive_vpp" if by_level else "freq_hz"]
    for s in qs:
        unit = "dB" if is_ratio(s) else (s.unit or "rms")
        header.append(f"{s.name}_mag_{unit}")
        if s.phase_valid:
            header.append(f"{s.name}_phase_deg")
    header.append("flags")
    rows = []
    for i in idx:
        row = [_fmt(x[i], 6)]
        flags = set()
        for s in qs:
            if i >= len(s.freqs):
                row += [""] * (2 if s.phase_valid else 1)
                continue
            y, _ = magnitude(s)
            row.append(_fmt(float(y[i])))
            if s.phase_valid:
                row.append(_fmt(float(s.phase_deg[i])))
            flags |= set(s.flags[i])
        row.append(";".join(sorted(flags)))
        rows.append(row)
    return header, rows


def _corners(s: Series) -> list[str]:
    """-3 dB crossings relative to the low-frequency value, and peaks."""
    y, unit = magnitude(s)
    db = y if unit == "dB" else 20 * np.log10(np.clip(y, 1e-300, None))
    f = s.freqs
    ok = np.isfinite(db)
    if ok.sum() < 4:
        return []
    db, f = db[ok], f[ok]
    ref = float(np.median(db[: max(2, len(db) // 20)]))
    notes = []
    for i in range(1, len(db)):
        for level, word in ((ref - 3, "falls"), (ref + 3, "rises")):
            a, b = db[i - 1] - level, db[i] - level
            if a * b < 0 and ((word == "falls" and b < 0) or (word == "rises" and b > 0)):
                x = f[i - 1] * (f[i] / f[i - 1]) ** (a / (a - b))
                notes.append(f"{word} 3 dB below/above the low-frequency value near {_eng_hz(x)}")
    for i in range(1, len(db) - 1):
        lo = min(db[max(0, i - 3) : i].min(), db[i + 1 : i + 4].min())
        if db[i] >= db[i - 1] and db[i] >= db[i + 1] and db[i] - lo > 3:
            notes.append(f"peak {db[i] - ref:+.1f} dB at {_eng_hz(f[i])}")
        if db[i] <= db[i - 1] and db[i] <= db[i + 1]:
            hi = max(db[max(0, i - 3) : i].max(), db[i + 1 : i + 4].max())
            if hi - db[i] > 3:
                notes.append(f"notch {db[i] - ref:+.1f} dB at {_eng_hz(f[i])}")
    return notes[:6]


def _delay_note(s: Series) -> str:
    if not s.phase_valid or len(s.freqs) < 4:
        return ""
    try:
        from sweep.fit import fit_delay
    except ImportError:
        return ""
    try:
        d, p0 = fit_delay(s.freqs, s.values)
    except Exception:
        return ""
    return f"phase fits a delay of {d * 1e9:.3g} ns with {p0:.2f} deg offset"


def group_delay(s: Series) -> tuple[np.ndarray, np.ndarray]:
    """Group delay in seconds, -d(phase)/d(omega), at the midpoints between points."""
    if not s.phase_valid or len(s.freqs) < 2:
        return np.array([]), np.array([])
    ph = unwrap_finite(np.angle(s.values))
    w = 2 * np.pi * s.freqs
    return np.sqrt(s.freqs[1:] * s.freqs[:-1]), -np.diff(ph) / np.diff(w)


def _group_delay_note(s: Series) -> str:
    f, gd = group_delay(s)
    ok = np.isfinite(gd)
    if ok.sum() < 3:
        return ""
    gd = gd[ok]
    lo, med, hi = np.percentile(gd, [10, 50, 90])
    return (
        f"group delay median {med * 1e9:.4g} ns, 10 to 90 % of points within "
        f"{lo * 1e9:.4g} to {hi * 1e9:.4g} ns (noisy where points are close)"
    )


def _fit_note(s: Series) -> str:
    if not s.phase_valid or len(s.freqs) < 6:
        return ""
    try:
        from sweep.fit import auto_fit

        fit = auto_fit(s.freqs, s.values, max_order=3, delay=True)
    except Exception:
        return ""
    text = " ".join(line.strip() for line in fit.describe().splitlines()[1:] if line.strip())
    return text


def wide_table(run) -> tuple[list[str], list[list[str]]]:
    """Every point as one row: frequency, then magnitude, phase and flags per quantity."""
    run = _as_run(run)
    qs = _quantities(run)
    by_point: dict[int, dict] = {}
    for r in run.rows:
        if r["kind"] in ("signal", "scalar", "expr"):
            by_point.setdefault(r["point"], {})[r["signal"]] = r
    header = ["point", "freq_cmd_hz", "freq_used_hz", "amplitude_cmd", "amplitude_set"]
    for s in qs:
        header += [
            f"{s.name}_re",
            f"{s.name}_im",
            f"{s.name}_mag_{s.unit or 'rms'}",
            f"{s.name}_phase_deg",
            f"{s.name}_flags",
        ]
    rows = []
    for pt in sorted(by_point, key=lambda k: (next(iter(by_point[k].values()))["freq_cmd"], k)):
        d = by_point[pt]
        any_row = next(iter(d.values()))
        row = [
            str(pt),
            _fmt(any_row["freq_cmd"], 10),
            _fmt(any_row["freq_used"], 10),
            _fmt(any_row["amplitude_cmd"], 8),
            _fmt(any_row["amplitude_set"], 8),
        ]
        for s in qs:
            r = d.get(s.name)
            if r is None:
                row += [""] * 5
                continue
            ph = r["phase_deg"] if r["phase_valid"] == 1 else float("nan")
            row += [
                _fmt(r["re"], 10),
                _fmt(r["im"], 10),
                _fmt(r["mag"], 10),
                _fmt(ph, 8),
                r["flags"],
            ]
        rows.append(row)
    return header, rows


def review(run, n_rows: int = 25) -> str:
    """Markdown review of one run, sized for an agent to read in full."""
    run = _as_run(run)
    m = run.meta
    qs = _quantities(run)
    out = [f"# {run.title}", ""]
    f_all = qs[0].freqs if qs else np.array([])
    out += [
        "| Item | Value |",
        "| --- | --- |",
        f"| Status | {m.get('status', '')} {m.get('reason', '') or ''} |",
        f"| Started | {m.get('started', '')} |",
        f"| Points | {len(f_all)} |",
        f"| Range | {_eng_hz(f_all.min()) if len(f_all) else ''} to "
        f"{_eng_hz(f_all.max()) if len(f_all) else ''} |",
        f"| Source | {m.get('source', {}).get('class', m.get('source', {}).get('name', ''))} |",
        f"| Measurements | {', '.join(str(x.get('class', x.get('name', ''))) for x in m.get('measurements', []))} |",
        f"| Folder | {run.path.name} |",
    ]
    if m.get("notes"):
        out.append(f"| Notes | {m['notes']} |")
    out += [
        "",
        "## Quantities",
        "",
        "| Name | Unit | Min | at | Max | at | Phase range (deg) | Worst estimate SNR (dB) | Max THD (%) | Flagged points |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in qs:
        y, unit = magnitude(s)
        if not np.isfinite(y).any():
            continue
        i0, i1 = int(np.nanargmin(y)), int(np.nanargmax(y))
        ph = s.phase_deg
        pr = (
            f"{np.nanmin(ph):.1f} to {np.nanmax(ph):.1f}"
            if s.phase_valid and np.isfinite(ph).any()
            else ""
        )
        est = s.estimate_snr_db if s.estimate_snr_db is not None else s.snr_db
        snr = np.nanmin(est) if np.isfinite(est).any() else float("nan")
        thd = np.nanmax(s.thd_pct) if np.isfinite(s.thd_pct).any() else float("nan")
        nflag = sum(1 for fl in s.flags if fl)
        out.append(
            f"| {s.name} | {unit} | {_fmt(y[i0])} | {_eng_hz(s.freqs[i0])} | "
            f"{_fmt(y[i1])} | {_eng_hz(s.freqs[i1])} | {pr} | {_fmt(snr, 3)} | "
            f"{_fmt(thd, 3)} | {nflag} |"
        )
    notes = []
    by_level = m.get("x_axis") == "amplitude"
    for s in [] if by_level else qs:
        for c in _corners(s):
            notes.append(f"- {s.name}: {c}")
        d = _delay_note(s)
        if d:
            notes.append(f"- {s.name}: {d}")
        g = _group_delay_note(s)
        if g:
            notes.append(f"- {s.name}: {g}")
    if notes:
        out += ["", "## Features", ""] + notes
    fits = (
        []
        if by_level
        else [f"- {s.name}: {_fit_note(s)}" for s in qs if s.kind == "expr" and _fit_note(s)]
    )
    if by_level:
        out += [
            "",
            f"## Against drive level at {m.get('level_freq', 0):.6g} Hz",
            "",
            "| Drive (Vpp) | "
            + " | ".join(f"{s.name} mag" for s in qs)
            + " | "
            + " | ".join(f"{s.name} THD %" for s in qs if s.kind == "signal")
            + " |",
            "|" + " --- |" * (1 + len(qs) + sum(s.kind == "signal" for s in qs)),
        ]
        for k in range(len(qs[0].freqs)):
            cells = [_fmt(qs[0].amplitudes[k])] + [_fmt(float(s.mag[k])) for s in qs]
            cells += [_fmt(float(s.thd_pct[k]), 3) for s in qs if s.kind == "signal"]
            out.append("| " + " | ".join(cells) + " |")
    if fits:
        out += ["", "## Fits (automatic, lowest order that fits)", ""] + fits
    inst = run.instrument_measurements()
    if inst:
        from sweep.plot import _meas_groups

        out += [
            "",
            "## Instrument measurements at each point",
            "",
            "| Measurement | Unit | Min | at | Max | at |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for key, rows in _meas_groups(run).items():
            ok = [
                r for r in rows if isinstance(r["value"], (int, float)) and np.isfinite(r["value"])
            ]
            if not ok:
                out.append(f"| {key} | | | | | |")
                continue
            lo = min(ok, key=lambda r: r["value"])
            hi = max(ok, key=lambda r: r["value"])
            out.append(
                f"| {key} | {rows[0].get('unit', '')} | {_fmt(lo['value'])} | "
                f"{_eng_hz(lo['freq_cmd'])} | {_fmt(hi['value'])} | {_eng_hz(hi['freq_cmd'])} |"
            )
    counts = Counter()
    where: dict[str, list[float]] = {}
    for s in qs:
        for f, fl in zip(s.freqs, s.flags):
            for x in fl:
                counts[x] += 1
                where.setdefault(x, []).append(f)
    if counts:
        out += ["", "## Flags", "", "| Flag | Count | Frequencies |", "| --- | --- | --- |"]
        for x, c in counts.most_common():
            fs = where[x]
            out.append(f"| {x} | {c} | {_eng_hz(min(fs))} to {_eng_hz(max(fs))} |")
    header, rows = condensed(run, n_rows)
    if rows:
        out += [
            "",
            f"## Condensed table ({len(rows)} points)",
            "",
            "| " + " | ".join(header) + " |",
            "|" + " --- |" * len(header),
        ]
        out += ["| " + " | ".join(r) + " |" for r in rows]
    files = sorted(p.name for p in run.path.iterdir() if p.is_file())
    out += ["", "## Files", "", ", ".join(files)]
    return "\n".join(out) + "\n"


def write_views(run, n: int = 100, n_rows: int = 25) -> dict[str, Path]:
    """Write condensed.csv and review.md into the run folder."""
    run = _as_run(run)
    header, rows = condensed(run, n)
    p_csv = run.path / "condensed.csv"
    with open(p_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    p_wide = run.path / "wide.csv"
    wh, wr = wide_table(run)
    with open(p_wide, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(wh)
        w.writerows(wr)
    p_md = run.path / "review.md"
    p_md.write_text(review(run, n_rows), encoding="utf-8")
    return {"condensed": p_csv, "wide": p_wide, "review": p_md}


def compare_review(runs, name: str, *, labels=None, points: int = 25) -> str:
    """Markdown table of one quantity across runs on a common frequency grid."""
    from sweep.interp import common_grid, interpolate

    runs = [_as_run(r) for r in runs]
    labels = labels or [r.title for r in runs]
    ser = [r.series(name) for r in runs]
    grid = common_grid([s.freqs for s in ser], mode="overlap", points=points)
    cols = []
    for s in ser:
        v, outside = interpolate(s.freqs, s.values, grid)
        tmp = Series(
            s.name,
            grid,
            v,
            s.unit,
            s.kind,
            s.phase_valid,
            [[] for _ in grid],
            grid * math.nan,
            grid * math.nan,
        )
        cols.append(magnitude(tmp)[0])
    unit = magnitude(ser[0])[1]
    out = [
        f"## {name} across runs ({unit})",
        "",
        "| freq_hz | " + " | ".join(labels) + " |",
        "|" + " --- |" * (len(labels) + 1),
    ]
    for i, f in enumerate(grid):
        out.append(f"| {_fmt(f, 6)} | " + " | ".join(_fmt(float(c[i])) for c in cols) + " |")
    return "\n".join(out) + "\n"
