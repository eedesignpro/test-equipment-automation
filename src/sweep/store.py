"""The run folder: the normalized record of one sweep.

Layout::

    <root>/<YYYY-MM-DD_HH-MM-SS>_<title>/
        meta.json        setup, plan, instruments, status (written first, updated last)
        points.csv       long table, one row per point, signal and tone, appended live
        spurs.csv        largest spurs per point and signal
        waveforms/       optional raw records, one .npz per point
        plot.png/.svg    made from this folder by sweep.plot
        condensed.csv    about 100 points, made by sweep.views
        review.md        markdown review tables, made by sweep.views

``points.csv`` is the normalized form. Every value is a complex phasor (re,
im) with its magnitude, phase, unit and quality, so any later tool reads one
format. ``kind`` is ``signal`` (fundamental), ``harmonic``, ``tone`` (an extra
tone), ``scalar`` (a direct reading) or ``expr`` (a user expression).
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

COLUMNS = [
    "point",
    "freq_cmd",
    "freq_planned",
    "freq_set",
    "freq_used",
    "amplitude_cmd",
    "amplitude_set",
    "measurement",
    "group",
    "signal",
    "kind",
    "order",
    "tone_freq",
    "re",
    "im",
    "mag",
    "phase_deg",
    "unit",
    "phase_valid",
    "offset_bins",
    "strongest_freq_hz",
    "strongest_dbc",
    "fit_residual_rms",
    "coherent_off_bins",
    "coherent_leak_db",
    "snr_db",
    "estimate_snr_db",
    "phase_uncertainty_deg",
    "thd_pct",
    "sinad_db",
    "noise_rms",
    "clip",
    "freq_offset_hz",
    "drift_hz_per_s",
    "phase_jitter_deg",
    "amplitude_variation_pct",
    "repeats",
    "repeat_mag_spread_pct",
    "repeat_phase_spread_deg",
    "sample_rate",
    "record_length",
    "cycles",
    "rbw",
    "window",
    "flags",
    "settings",
    "time",
]

_TEXT = {"measurement", "group", "signal", "kind", "unit", "window", "flags", "settings", "time"}


def safe_name(text: str) -> str:
    """A file-name-safe version of a title."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._")
    return s[:80] or "sweep"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else str(f)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def unwrap_finite(angles: np.ndarray) -> np.ndarray:
    """Unwrap phase over the finite points only, leaving NaN where it was.

    ``np.unwrap`` alone turns every point after one NaN into NaN.
    """
    a = np.asarray(angles, dtype=float)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    if ok.any():
        out[ok] = np.unwrap(a[ok])
    return out


class RunWriter:
    """Create a run folder and write it as the sweep goes."""

    def __init__(self, root, title: str, meta: dict, *, stamp: datetime | None = None):
        stamp = stamp or datetime.now()
        base = Path(root) / f"{stamp:%Y-%m-%d_%H-%M-%S}_{safe_name(title)}"
        self.path = base
        k = 2
        while self.path.exists():
            self.path = base.with_name(f"{base.name}_{k}")
            k += 1
        self.path.mkdir(parents=True)
        self.meta = {"title": title, "started": stamp.isoformat(timespec="seconds"), **meta}
        self.meta.setdefault("status", "running")
        self.write_meta()
        self._points = open(self.path / "points.csv", "w", newline="", encoding="utf-8")
        self._pw = csv.DictWriter(self._points, fieldnames=COLUMNS, extrasaction="ignore")
        self._pw.writeheader()
        self._points.flush()
        self._spurs = open(self.path / "spurs.csv", "w", newline="", encoding="utf-8")
        self._sw = csv.writer(self._spurs)
        self._sw.writerow(["point", "freq_used", "signal", "spur_freq", "rms", "dbc"])

    def write_meta(self) -> None:
        tmp = self.path / "meta.json.tmp"
        tmp.write_text(json.dumps(_jsonable(self.meta), indent=2), encoding="utf-8")
        # A reader holding meta.json open on Windows blocks the replace; wait for it.
        for k in range(200):
            try:
                tmp.replace(self.path / "meta.json")
                return
            except PermissionError:
                if k == 199:
                    raise
                time.sleep(0.01)

    def add_rows(self, rows: list[dict]) -> None:
        for r in rows:
            out = {}
            for k in COLUMNS:
                v = r.get(k, "")
                if isinstance(v, float):
                    v = "" if math.isnan(v) else f"{v:.12g}"
                elif isinstance(v, (list, tuple)):
                    v = ";".join(str(x) for x in v)
                out[k] = v
            self._pw.writerow(out)
        self._points.flush()

    INSTRUMENT_COLUMNS = [
        "point",
        "freq_cmd",
        "measurement",
        "slot",
        "label",
        "type",
        "source",
        "unit",
        "value",
        "curr_min",
        "curr_max",
        "curr_std",
    ]

    def add_instrument_rows(self, rows: list[dict]) -> None:
        """The instrument's own measurements at one point, one row per measurement."""
        if not rows:
            return
        if getattr(self, "_inst", None) is None:
            self._inst = open(
                self.path / "instrument_measurements.csv", "w", newline="", encoding="utf-8"
            )
            self._iw = csv.DictWriter(
                self._inst, fieldnames=self.INSTRUMENT_COLUMNS, extrasaction="ignore"
            )
            self._iw.writeheader()
        for r in rows:
            self._iw.writerow(
                {k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            )
        self._inst.flush()

    TIMING_COLUMNS = [
        "point",
        "freq_cmd",
        "set_s",
        "settle_s",
        "measure_s",
        "horizontal_setup_s",
        "acquire_s",
        "transfer_s",
        "readback_s",
        "process_s",
        "save_s",
        "point_s",
        "process_pending",
    ]

    def add_timing(self, row: dict) -> None:
        """Where the time of one point went, one row per point."""
        if getattr(self, "_tim", None) is None:
            self._tim = open(self.path / "timing.csv", "w", newline="", encoding="utf-8")
            self._tw = csv.DictWriter(
                self._tim, fieldnames=self.TIMING_COLUMNS, extrasaction="ignore"
            )
            self._tw.writeheader()
        self._tw.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
        self._tim.flush()

    def add_spectra(self, point: int, freq: float, spectra: dict, units: dict) -> None:
        """A reduced spectrum per signal for one point, for the waterfall plot."""
        d = self.path / "spectra"
        d.mkdir(exist_ok=True)
        payload = {"freq_cmd": freq, "names": np.array(list(spectra))}
        for k, (f, a) in spectra.items():
            payload[f"f_{k}"] = np.asarray(f, dtype=np.float32)
            payload[f"a_{k}"] = np.asarray(a, dtype=np.float32)
            payload[f"u_{k}"] = units.get(k, "")
        np.savez_compressed(d / f"point_{point:04d}.npz", **payload)

    def add_spurs(self, point: int, freq: float, signal: str, spurs) -> None:
        for s in spurs:
            self._sw.writerow(
                [point, f"{freq:.12g}", signal, f"{s.freq:.12g}", f"{s.rms:.6g}", f"{s.dbc:.2f}"]
            )
        self._spurs.flush()

    def save_waveforms(self, point: int, capture, freq: float, measurement: str) -> Path:
        """Save one capture's records.

        With ``capture.raw`` (an instrument's own integer codes and scale
        factors, as from the MSO) the codes are saved unchanged, so the file is
        lossless and about half the size of float32. Otherwise the scaled
        samples are saved as float32 with the time axis as start and step.
        """
        d = self.path / "waveforms"
        d.mkdir(exist_ok=True)
        p = d / f"point_{point:04d}_{safe_name(measurement)}.npz"
        names = list(capture.signals)
        payload = {
            "freq": freq,
            "names": np.array(names),
            "units": np.array([capture.units.get(n, "") for n in names]),
        }
        raw = capture.raw or {}
        if raw and all(n in raw for n in names):
            payload["format"] = "codes"
            for k, n in enumerate(names):
                payload[f"codes_{k}"] = np.asarray(raw[n]["codes"])
                payload[f"scale_{k}"] = np.array(
                    [raw[n]["ymult"], raw[n]["yoff"], raw[n]["yzero"]], dtype=np.float64
                )
            tm = raw.get("_time")
            if tm:
                payload["time"] = np.array([tm["xzero"], tm["xincr"], tm["pt_off"]])
        else:
            payload["format"] = "float32"
            for k, n in enumerate(names):
                payload[f"data_{k}"] = np.asarray(capture.signals[n], dtype=np.float32)
        if "time" not in payload:
            if capture.t is not None:
                t = np.asarray(capture.t, dtype=np.float64)
                payload["time"] = np.array([t[0], float(np.median(np.diff(t))), 0.0])
            else:
                payload["time"] = np.array([0.0, float(capture.dt), 0.0])
        np.savez_compressed(p, **payload)
        return p

    def finish(self, status: str, **extra) -> None:
        self.meta["status"] = status
        self.meta["finished"] = datetime.now().isoformat(timespec="seconds")
        self.meta.update(extra)
        self.write_meta()
        for fh in (
            self._points,
            self._spurs,
            getattr(self, "_inst", None),
            getattr(self, "_tim", None),
        ):
            if fh is not None and not fh.closed:
                fh.close()


@dataclass
class Series:
    """One named quantity across the sweep."""

    name: str
    freqs: np.ndarray
    values: np.ndarray
    unit: str
    kind: str
    phase_valid: bool
    flags: list[list[str]]
    snr_db: np.ndarray
    thd_pct: np.ndarray
    estimate_snr_db: np.ndarray | None = None
    phase_uncertainty_deg: np.ndarray | None = None
    amplitudes: np.ndarray | None = None

    @property
    def mag(self) -> np.ndarray:
        return np.abs(self.values)

    @property
    def phase_deg(self) -> np.ndarray:
        if not self.phase_valid:
            return np.full(self.values.shape, np.nan)
        return np.degrees(unwrap_finite(np.angle(self.values)))


class Run:
    """A saved run folder, read back."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.rows = []
        pts = self.path / "points.csv"
        if pts.exists():
            with open(pts, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.rows.append({k: (v if k in _TEXT else _num(v)) for k, v in r.items()})

    @property
    def title(self) -> str:
        return self.meta.get("title", self.path.name)

    def names(self, kinds=("signal", "scalar", "expr")) -> list[str]:
        seen = []
        for r in self.rows:
            if r["kind"] in kinds and r["signal"] not in seen:
                seen.append(r["signal"])
        return seen

    def series(self, name: str, *, kind: str | None = None, order: int | None = None) -> Series:
        rows = [r for r in self.rows if r["signal"] == name]
        if kind:
            rows = [r for r in rows if r["kind"] == kind]
        elif order is None:
            rows = [r for r in rows if r["kind"] in ("signal", "scalar", "expr")]
        if order is not None:
            rows = [r for r in rows if r["order"] == order]
        if not rows:
            raise KeyError(f"no data for {name!r} in {self.path.name}")
        rows = sorted(rows, key=lambda r: (r["freq_cmd"], r["point"]))
        f = np.array(
            [r["freq_used"] if not math.isnan(r["freq_used"]) else r["freq_cmd"] for r in rows]
        )
        v = np.array([complex(r["re"], 0.0 if math.isnan(r["im"]) else r["im"]) for r in rows])
        return Series(
            name=name,
            freqs=f,
            values=v,
            unit=rows[0]["unit"],
            kind=rows[0]["kind"],
            phase_valid=all(r["phase_valid"] == 1 for r in rows),
            flags=[[x for x in r["flags"].split(";") if x] for r in rows],
            snr_db=np.array([r["snr_db"] for r in rows]),
            thd_pct=np.array([r["thd_pct"] for r in rows]),
            estimate_snr_db=np.array([r.get("estimate_snr_db", np.nan) for r in rows], float),
            amplitudes=np.array([r.get("amplitude_cmd", np.nan) for r in rows], float),
            phase_uncertainty_deg=np.array(
                [r.get("phase_uncertainty_deg", np.nan) for r in rows], float
            ),
        )

    def instrument_measurements(self) -> list[dict]:
        """Rows of instrument_measurements.csv, numbers parsed; empty when none."""
        p = self.path / "instrument_measurements.csv"
        if not p.exists():
            return []
        text = {"measurement", "label", "type", "source", "unit"}
        with open(p, newline="", encoding="utf-8") as fh:
            return [
                {k: (v if k in text else _num(v)) for k, v in r.items()} for r in csv.DictReader(fh)
            ]

    def spectra(self, signal: str) -> list[tuple[float, np.ndarray, np.ndarray, str]]:
        """(drive frequency, spectrum freqs, RMS per band, unit) per point, by drive frequency."""
        d = self.path / "spectra"
        out = []
        for p in sorted(d.glob("point_*.npz")) if d.exists() else []:
            z = np.load(p, allow_pickle=False)
            if f"f_{signal}" in z:
                out.append(
                    (
                        float(z["freq_cmd"]),
                        z[f"f_{signal}"],
                        z[f"a_{signal}"],
                        str(z[f"u_{signal}"]),
                    )
                )
        return sorted(out, key=lambda r: r[0])

    def harmonics(self, name: str, max_order: int = 10) -> dict[int, Series]:
        out = {}
        for k in range(2, max_order + 1):
            try:
                out[k] = self.series(name, kind="harmonic", order=k)
            except KeyError:
                break
        return out

    def waveform(self, point: int, measurement: str | None = None) -> dict:
        """One saved capture, scaled to physical units: freq, t, signals, units.

        Also returns ``codes`` (the instrument's integer codes) when the file
        holds them.
        """
        d = self.path / "waveforms"
        hits = sorted(d.glob(f"point_{point:04d}_*.npz"))
        if not hits:
            # A shared capture (multi-tone) is saved under its group's first point.
            for r in self.rows:
                if r["point"] == point and r["kind"] == "signal" and r.get("settings"):
                    wp = json.loads(r["settings"]).get("waveform_point")
                    if wp is not None:
                        hits = sorted(d.glob(f"point_{int(wp):04d}_*.npz"))
                    break
        if measurement:
            hits = [h for h in hits if h.stem.endswith(safe_name(measurement))]
        if not hits:
            raise FileNotFoundError(f"no waveform for point {point}")
        z = np.load(hits[0], allow_pickle=False)
        names = [str(n) for n in z["names"]]
        fmt = str(z["format"]) if "format" in z else "float64"
        signals, codes = {}, {}
        for k, n in enumerate(names):
            if fmt == "codes":
                c = z[f"codes_{k}"]
                ymult, yoff, yzero = z[f"scale_{k}"]
                codes[n] = c
                signals[n] = yzero + ymult * (c.astype(np.float64) - yoff)
            elif fmt == "float32":
                signals[n] = z[f"data_{k}"].astype(np.float64)
            else:
                signals[n] = z["data"][k]
        if "time" in z:
            x0, dx, off = z["time"]
            n0 = len(next(iter(signals.values())))
            t = x0 + dx * (np.arange(n0, dtype=np.float64) - off)
        else:
            t = z["t"]
        out = {
            "freq": float(z["freq"]),
            "t": t,
            "signals": signals,
            "units": {str(n): str(u) for n, u in zip(z["names"], z["units"])},
            "format": fmt,
        }
        if codes:
            out["codes"] = codes
        return out


def _num(v: str):
    if v == "":
        return float("nan")
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f


def load_run(path) -> Run:
    return Run(path)
