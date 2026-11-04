"""The sweep loop: set the source, measure, analyze, save, repeat.

:func:`run_sweep` is the one entry point. It takes any source and any list
of measurements (objects or plain functions, see :mod:`sweep.interfaces`),
writes the run folder point by point, and at the end makes the plot and the
AI-friendly views from that folder. The last line it prints is
``RUN_FOLDER <path>``.
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from common.errors import InstrumentUnresponsiveError
from sweep import analysis
from sweep.capture import CapturePlan, CaptureRules, plan_capture, validate_capture
from sweep.interfaces import Capture, Readings, as_measurement, as_source, describe
from sweep.plan import Amplitude, amplitude_at
from sweep.store import Run, RunWriter, git_commit


class EStop(Exception):
    """An emergency stop: end now, output off first, then teardown and restore."""


class StopSweep(Exception):
    """Raise from a ``stop_if`` check, or return a reason string, to end the sweep early."""


@dataclass
class SweepResult:
    path: Path
    status: str
    points: int

    def load(self) -> Run:
        return Run(self.path)


class _Tracked(dict):
    """A dict of complex values that remembers which keys an expression read."""

    def __init__(self, data, groups):
        super().__init__(data)
        self.groups = groups
        self.used: set[str] = set()

    def __getitem__(self, key):
        self.used.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            self.used.add(key)
        return super().get(key, default)


def _expr_spec(spec):
    if isinstance(spec, tuple):
        fn, unit = spec
        return fn, unit
    return spec, ""


def _tones_at(tones, freq) -> list[float]:
    if tones is None:
        return []
    if callable(tones):
        return [float(t) for t in tones(freq)]
    return [float(t) for t in tones]


def _calibrate(cal, name, freq, value):
    c = (cal or {}).get(name)
    if c is None:
        return value, False
    corrected, outside = c.apply(np.array([freq]), np.array([value]))
    return complex(corrected[0]), bool(np.asarray(outside)[0])


def _plan_summary(freqs, amplitude, plan_fn) -> list[dict]:
    rows = []
    for f in freqs:
        p = plan_fn(f)
        rows.append(
            {
                "freq": f,
                "amplitude": None if amplitude is None else amplitude_at(amplitude, f),
                **{k: v for k, v in p.as_dict().items() if k != "freq"},
            }
        )
    return rows


def run_sweep(
    freqs,
    source,
    measurements,
    *,
    amplitude: Amplitude | None = None,
    capture: CaptureRules | Callable[[float], CapturePlan] | None = None,
    expressions: dict | None = None,
    tones=None,
    harmonics: int = 5,
    refine: bool = True,
    reference: str | None = None,
    settle_s: float | Callable[[float], float] = 0.0,
    stop_if: Callable[[dict], str | None] | None = None,
    calibration: dict | None = None,
    limits: dict | None = None,
    title: str = "sweep",
    folder=None,
    save_waveforms: bool = True,
    notes: str = "",
    plot: bool = True,
    review: bool = True,
    live_plot: bool | float = 30.0,
    teardown=None,
    dry_run: bool = False,
    segments: int = 8,
    on_invalid: str = "retry",
    drive_off: bool = False,
    leave_output_off: bool = True,
    repeats: int = 1,
    settle_captures: int = 0,
    save_spectra: bool = True,
    pipeline: bool = True,
    estimator: str | None = None,
    search: str = "fft",
    coherent: bool = False,
    freq_tolerance: float | None = None,
    freq_tolerance_hz: float | None = None,
    refine_on: str | None = None,
    refine_db: float = 1.0,
    refine_deg: float = 10.0,
    refine_rounds: int = 2,
    refine_max_points: int = 40,
    quiet: bool = False,
    _extra_meta: dict | None = None,
) -> SweepResult:
    """Run a frequency sweep and save everything to a new run folder.

    ``freqs``: the frequency list (see :func:`sweep.plan.frequencies`).
    ``source``: any object with ``set(freq, amplitude)`` or a function.
    ``measurements``: a list of measurement objects or functions.
    ``amplitude``: drive as a constant, a function of frequency, or a table.
    ``capture``: :class:`CaptureRules`, or a function ``freq -> CapturePlan``.
    ``expressions``: ``{name: fn}`` or ``{name: (fn, unit)}``; ``fn`` gets a
    dict of fundamental phasors by signal name, for example
    ``{"Z": (lambda v: v["V_R"] / v["I_R"], "ohm")}``.
    ``tones``: extra tones to measure in every capture, a list or a function
    of the drive frequency. ``harmonics``: how many harmonics to measure.
    ``refine``: find the frequency actually received (unlocked clocks).
    ``stop_if``: a function of this point's values that returns a reason to
    stop, checked after each point. ``calibration``: ``{name: Calibration}``.
    ``dry_run``: write the plan and touch no instrument.

    Every capture is checked against its plan from the instrument's own
    read-back (sample rate, record length, and for a scope the time scale,
    divisions and horizontal mode), and every point checks the frequency the
    source reports against the commanded one. ``on_invalid`` decides what an
    invalid capture does: ``"retry"`` measures once more and then flags the
    point ``capture_invalid``; ``"flag"`` only flags it; ``"stop"`` ends the
    sweep.

    ``drive_off=True`` sets every frequency but never turns the output on, so
    the run measures the noise floor and pickup the receivers see at each
    point. Compare it with a driven run of the same plan.

    ``freq_tolerance`` (a fraction, 1e-3 is 0.1 %) or ``freq_tolerance_hz``
    says how far the received tone may sit from the commanded frequency, for
    source and receiver clocks that are not locked. It is optional: without it
    the search spans 2 resolution bins. Each point records how far the peak
    sat from the expected bin and where the strongest tone of the record is.

    ``estimator`` and ``search`` choose how values are read and how an
    unlocked tone is found (see :func:`sweep.analysis.analyze` and the guide).
    ``coherent=True`` needs locked clocks and a receiver with
    ``prepare_horizontal``: at each point the receiver's horizontal setting is
    made first, then the drive frequency moves to the nearest bin of the
    record it will take, so the tone and its harmonics fall exactly on FFT
    bins; the estimator becomes ``"coherent"`` unless another is given.

    ``pipeline`` (default True) analyzes and saves each point in a worker
    thread while the main thread sets the source and captures the next one,
    so processing overlaps the instrument time. ``timing.csv`` records, per
    point, the set, settle, horizontal setup, acquire, transfer, read-back,
    process and save times.

    ``refine_on`` names a quantity, for example ``"Z"``. After the planned
    points, the sweep adds points at the log midpoint of every step where
    that quantity changes by more than ``refine_db`` or ``refine_deg``, for up
    to ``refine_rounds`` rounds and ``refine_max_points`` extra points. This
    finds resonances and corners without a dense plan everywhere.

    ``settle_captures`` keeps capturing, up to that many extra times, until
    two captures in a row agree within ``settle_pct`` and ``settle_deg``
    (in ``limits``); only the settled capture is kept. Use it when the device
    or a filter needs time after a frequency step.

    ``repeats`` captures each point that many times, averages them, and
    records the spread; a spread above the limits flags the point
    ``unstable``, a sign of settling, drift or a changing device.

    Stopping: ``python -m sweep stop <run>`` ends the sweep after the current
    point; ``python -m sweep estop <run>`` raises an emergency stop at the next
    step (settle waits are checked every 50 ms). Either way the source output
    goes off first, then every ``teardown`` function runs (one callable or a
    list: put the safe-off of your own equipment here), then each instrument's
    found setup is restored. A terminate or console-close signal does the same.
    ``progress.json`` in the run folder is rewritten after every point and
    ``plot.png`` at most every ``live_plot`` seconds, so a background run can be
    checked with ``python -m sweep status <run>``.

    ``leave_output_off`` (default True) turns the source output off once more
    after the found settings are restored, so a sweep never ends energized even
    when the output was on before it. A plain-function source must enforce its
    own amplitude limit and its own off.
    """
    freqs = [float(f) for f in freqs]
    if estimator is None:
        estimator = "coherent" if coherent else "dft"
    rules = capture if capture is not None else CaptureRules()
    plan_fn = (
        rules
        if callable(rules) and not isinstance(rules, CaptureRules)
        else (lambda f: plan_capture(f, rules))
    )
    src = as_source(source)
    meas = [_EstopGuard(as_measurement(m)) for m in measurements]
    exprs = {k: _expr_spec(v) for k, v in (expressions or {}).items()}
    root = Path(folder) if folder else Path.cwd() / "sweeps"
    root.mkdir(parents=True, exist_ok=True)

    meta = {
        "notes": notes,
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "frequencies": {"count": len(freqs), "min": min(freqs), "max": max(freqs)},
        "amplitude": amplitude if isinstance(amplitude, (int, float)) else repr(amplitude),
        "capture_rules": asdict(rules) if isinstance(rules, CaptureRules) else repr(rules),
        "analysis": {
            "harmonics": harmonics,
            "refine": refine,
            "reference": reference,
            "segments": segments,
            "limits": {**analysis.DEFAULT_FLAG_LIMITS, **(limits or {})},
        },
        "expressions": {k: u for k, (_, u) in exprs.items()},
        "calibration": {k: getattr(c, "name", repr(c)) for k, c in (calibration or {}).items()},
        "save_waveforms": save_waveforms,
        "drive_off": drive_off,
        **(_extra_meta or {}),
        "repeats": repeats,
        "estimator": estimator,
        "search": search,
        "coherent": coherent,
        "freq_tolerance": freq_tolerance,
        "freq_tolerance_hz": freq_tolerance_hz,
        "refine": {
            "on": refine_on,
            "db": refine_db,
            "deg": refine_deg,
            "rounds": refine_rounds,
            "max_points": refine_max_points,
        },
    }
    if dry_run:
        meta["status"] = "planned"
        meta["plan"] = _plan_summary(freqs, amplitude, plan_fn)
        meta["source"] = {"name": getattr(src, "name", "source")}
        meta["measurements"] = [{"name": getattr(m, "name", "measurement")} for m in meas]
        w = RunWriter(root, title + "_dry_run", meta)
        w.finish("planned")
        if not quiet:
            amps = [r["amplitude"] for r in meta["plan"] if r["amplitude"] is not None]
            total = sum(r["duration"] for r in meta["plan"])
            print(
                f"dry run: {len(freqs)} points, {min(freqs):.6g} to {max(freqs):.6g} Hz, "
                f"highest drive {max(amps) if amps else 'n/a'}, record time {total:.3g} s (setup and transfer time not included)"
            )
            print(f"RUN_FOLDER {w.path}")
        return SweepResult(w.path, "planned", 0)

    opened = []
    pool = None
    state = {"stop": None, "done": 0}
    status, reason, n_done = "error", "", 0
    restore_signals = _trap_stop_signals()
    writer = None
    try:
        for obj in [src, *meas]:
            opened.append(obj)
            if hasattr(obj, "open"):
                obj.open()
        meta["source"] = describe(src)
        meta["measurements"] = [describe(m) for m in meas]
        writer = RunWriter(root, title, meta)
        writer.save_spectra = save_spectra
        stop_file = writer.path / "STOP"
        estop_file = writer.path / "ESTOP"
        writer.meta["stop_file"] = str(stop_file)
        writer.meta["estop_file"] = str(estop_file)
        writer.write_meta()
        progress = _Progress(writer.path, len(freqs))
        if not quiet:
            print(f"RUN_FOLDER {writer.path}", flush=True)
            print(f"stop after this point: python -m sweep stop {writer.path}", flush=True)
            print(f"emergency stop:        python -m sweep estop {writer.path}", flush=True)

        def check_estop():
            if estop_file.exists():
                raise EStop("ESTOP file")

        for m in meas:
            m.check = check_estop

        last_plot = time.monotonic()
        state = {"stop": None, "done": 0, "last_plot": last_plot, "live": None}
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweep") if pipeline else None
        pending: deque = deque()
        # A stop_if limit is a safety check: the next point must not be set
        # until this point's values are known, so the worker runs in step.
        max_pending = 0 if stop_if is not None else 2

        def drain():
            while pending:
                pending.popleft().result()

        def process_point(job):
            i, f, base, collected, timing, total = job
            t0 = time.perf_counter()
            values, groups, units, point_flags = {}, {}, {}, []
            rel_noise: dict[str, float] = {}
            rows = []
            save_s = 0.0
            for mname, plan, res in collected:
                if isinstance(res, Readings):
                    rows += _readings_rows(res, f, base, mname, calibration, values, groups, units)
                    continue
                rows += _capture_rows(
                    res,
                    f,
                    plan,
                    base,
                    mname,
                    tones,
                    harmonics,
                    refine,
                    reference,
                    limits,
                    segments,
                    calibration,
                    values,
                    groups,
                    units,
                    writer,
                    i,
                    rel_noise,
                    (freq_tolerance, freq_tolerance_hz, estimator, search),
                )
                first = res[0]
                st = first.settings.get("time_s") or {}
                for k in ("horizontal_setup", "acquire", "transfer", "readback"):
                    if k in st:
                        timing[f"{k}_s"] = timing.get(f"{k}_s", 0.0) + float(st[k])
                # A capture shared by several tones (multi-tone) is saved once,
                # under the first tone's point; the others point to it.
                if save_waveforms and not getattr(first, "_waveform_saved", False):
                    ts = time.perf_counter()
                    writer.save_waveforms(i, first, f, mname)
                    first._waveform_saved = True
                    save_s += time.perf_counter() - ts
            rows += _expr_rows(
                exprs, values, groups, rel_noise, calibration, base, rows, f, point_flags
            )
            writer.add_rows(rows)
            state["done"] = i + 1
            if refine_on and refine_on in values:
                seen[f] = values[refine_on]
            if not quiet:
                print(_status_line(i, total, f, rows), end="\r", flush=True)
            progress.update(i + 1, total, f, rows)
            every = 0.0 if live_plot is True else float(live_plot or 0)
            if live_plot is not False and time.monotonic() - state["last_plot"] >= every:
                state["live"] = _live_update(state["live"], writer.path)
                state["last_plot"] = time.monotonic()
            if stop_if is not None and not state["stop"]:
                why = stop_if({k: values[k] for k in values})
                if why:
                    state["stop"] = str(why)
            timing["process_s"] = time.perf_counter() - t0 - save_s
            timing["save_s"] = save_s
            timing["point_s"] = timing["set_s"] + timing["settle_s"] + timing["measure_s"]
            timing["process_pending"] = len(pending)
            writer.add_timing(timing)

        output_on = False
        prep = next((m for m in meas if hasattr(m, "prepare_horizontal")), None)
        if coherent and prep is None and not quiet:
            print(
                "coherent=True: no receiver can prepare its horizontal setting; "
                "frequencies are not snapped",
                flush=True,
            )
        todo = list(freqs)
        seen: dict[float, complex] = {}
        rounds = 0
        i = -1
        while i + 1 < len(todo):
            i += 1
            f = todo[i]
            check_estop()
            if stop_file.exists():
                status, reason = "stopped", "STOP file"
                break
            t_point = time.perf_counter()
            amp = None if amplitude is None else amplitude_at(amplitude, f)
            f_planned = f
            point_plan = plan_fn(f)
            if coherent and prep is not None:
                sr_act, rl_act = prep.prepare_horizontal(point_plan)
                if hasattr(src, "coherent_freq"):
                    f = src.coherent_freq(f, sr_act, rl_act)
                else:
                    bin_hz = sr_act / rl_act
                    f = max(1, round(f / bin_hz)) * bin_hz
            got = src.set(f, amp)
            if not output_on and not drive_off and hasattr(src, "output"):
                level = src.output(True)
                output_on = True
                if isinstance(level, (int, float)):
                    got = {**got, "amplitude": float(level)}
            wait = settle_s(f) if callable(settle_s) else settle_s
            if wait:
                _sleep_checked(wait, check_estop)
            check_estop()
            base = {
                "point": i,
                "freq_cmd": f,
                "freq_planned": f_planned,
                "freq_set": float(got.get("freq", f)),
                "amplitude_cmd": amp if amp is not None else math.nan,
                "amplitude_set": math.nan
                if got.get("amplitude") is None
                else float(got.get("amplitude")),
                "time": datetime.now().isoformat(timespec="milliseconds"),
            }
            t_set_done = time.perf_counter()
            freq_flags = _source_freq_flags(f, base["freq_set"])
            collected = []
            for m in meas:
                check_estop()
                plan = point_plan
                res = m.measure(f, plan)
                mname = getattr(m, "name", "measurement")
                if isinstance(res, Capture):
                    res = _checked_capture(m, f, plan, res, harmonics, on_invalid, freq_flags)
                    if settle_captures:
                        res = _settled(
                            m,
                            f,
                            plan,
                            res,
                            settle_captures,
                            limits,
                            lambda r: _checked_capture(
                                m, f, plan, r, harmonics, on_invalid, freq_flags
                            ),
                        )
                    more = [
                        _checked_capture(
                            m, f, plan, m.measure(f, plan), harmonics, on_invalid, freq_flags
                        )
                        for _ in range(max(1, int(repeats)) - 1)
                    ]
                    collected.append((mname, plan, [res, *more]))
                elif isinstance(res, Readings):
                    collected.append((mname, None, res))
                else:
                    raise TypeError(
                        f"{mname} returned {type(res).__name__}, not Capture or Readings"
                    )
                # Screenshots and instrument measurements read the same capture,
                # so they run here, before the next point changes the instrument.
                hook = getattr(m, "after_point", None)
                if hook is not None:
                    extra_rows = hook(i, f, writer.path) or []
                    for r in extra_rows:
                        r.setdefault("measurement", mname)
                    writer.add_instrument_rows(
                        [{"point": i, "freq_cmd": f, **r} for r in extra_rows]
                    )
            timing = {
                "point": i,
                "freq_cmd": f,
                "set_s": t_set_done - t_point,
                "settle_s": float(wait or 0.0),
                "measure_s": time.perf_counter() - t_set_done,
            }
            job = (i, f, base, collected, timing, len(todo))
            if pool is None:
                process_point(job)
            else:
                pending.append(pool.submit(process_point, job))
                while len(pending) > max_pending:
                    pending.popleft().result()
            if state["stop"]:
                reason = state["stop"]
                status = "stopped"
                break
            if i + 1 == len(todo) and refine_on and rounds < refine_rounds:
                drain()
                room = refine_max_points - (len(todo) - len(freqs))
                new = refine_points(seen, db_step=refine_db, deg_step=refine_deg, limit=room)
                if new:
                    todo.extend(new)
                    rounds += 1
        else:
            status = "complete"
        drain()
        n_done = state["done"]
        if state["stop"] and status == "complete":
            status, reason = "stopped", state["stop"]
    except StopSweep as exc:
        status, reason = "stopped", str(exc) or "StopSweep"
    except EStop as exc:
        status, reason = "estopped", str(exc)
    except KeyboardInterrupt:
        status, reason = "aborted", "KeyboardInterrupt"
    except Exception as exc:
        status, reason = "error", f"{type(exc).__name__}: {exc}"
        if isinstance(exc, InstrumentUnresponsiveError):
            # The driver's rule: stop that instrument, no retry, a person power cycles it.
            reason += "; the instrument needs a power cycle"
            print(f"WARNING: {exc}", file=sys.stderr)
        if writer is not None:
            writer.meta["traceback"] = traceback.format_exc()
        else:
            raise
    finally:
        restore_signals.hold()
        off_ok = True
        restore_ok = True
        if hasattr(src, "output"):
            try:
                src.output(False)
            except BaseException as exc:
                off_ok = False
                reason = (reason + f"; output off failed: {exc}").strip("; ")
        for fn in [teardown] if callable(teardown) else list(teardown or []):
            try:
                fn()
            except BaseException as exc:
                reason = (reason + f"; teardown failed: {exc}").strip("; ")
        for obj in reversed(opened):
            if not hasattr(obj, "close"):
                continue
            try:
                obj.close()
            except BaseException as exc:
                restore_ok = False
                reason = (reason + f"; close failed: {exc}").strip("; ")
        # The close restores the settings found before the sweep, which can
        # include an output that was on. The sweep leaves the output off.
        if leave_output_off and hasattr(src, "output"):
            try:
                src.output(False)
            except BaseException as exc:
                off_ok = False
                reason = (reason + f"; final output off failed: {exc}").strip("; ")
        restore_signals()
        if not restore_ok and status == "complete":
            status = "complete_restore_failed"
        if not off_ok:
            status = "error"
            print(f"WARNING: the source output may still be on: {reason}", file=sys.stderr)
        if writer is not None:
            if pool is not None:
                pool.shutdown(wait=True)
                n_done = max(n_done, state["done"])
            try:
                progress.finish(status, reason)
            except Exception:
                pass
            writer.finish(
                status,
                reason=reason,
                points_done=n_done,
                output_off_ok=off_ok,
                restore_ok=restore_ok,
            )
    if writer is None:
        # Stopped before the run folder existed, for example Ctrl+C during open.
        if not quiet:
            print()
            print(f"status {status}{': ' + reason if reason else ''}")
        if status == "aborted":
            raise KeyboardInterrupt(reason)
        raise StopSweep(reason or status)
    if not quiet:
        print()
    if plot and n_done:
        from sweep.plot import plot_run

        try:
            plot_run(writer.path)
            _auto_harmonic_plots(writer.path)
            from sweep.plot import plot_instrument_measurements

            plot_instrument_measurements(writer.path)
            from sweep.plot import plot_waterfall

            for name in Run(writer.path).names(kinds=("signal",)):
                plot_waterfall(writer.path, name)
            from sweep.plot import IMPEDANCE_UNITS, plot_impedance

            done_run = Run(writer.path)
            for name in done_run.names(kinds=("expr",)):
                if done_run.series(name).unit in IMPEDANCE_UNITS:
                    plot_impedance(done_run, name)
        except Exception as exc:
            print(f"plot failed: {exc}")
    if review and n_done:
        from sweep.views import write_views

        try:
            write_views(writer.path)
        except Exception as exc:
            print(f"views failed: {exc}")
    if not quiet:
        print(f"status {status}{': ' + reason if reason else ''}")
        print(f"RUN_FOLDER {writer.path}")
    return SweepResult(writer.path, status, n_done)


def _auto_harmonic_plots(path) -> None:
    """Draw the harmonics of every signal that was flagged nonlinear at any point."""
    from sweep.plot import plot_harmonics

    run = Run(path)
    for name in run.names(kinds=("signal",)):
        s = run.series(name, kind="signal")
        if any("nonlinear" in fl for fl in s.flags):
            plot_harmonics(run, name)


def _ratios(capture: Capture, freq: float) -> dict[str, tuple[complex, float]]:
    """Each signal relative to the first, with its relative noise from the estimate SNR."""
    _, res = analysis.analyze(
        capture.signals, freq, t=capture.t, dt=capture.dt, harmonics=0, refine=False, segments=0
    )
    names = list(res)
    p0 = res[names[0]][0]
    noise0 = 10 ** (-p0.quality.estimate_snr_db / 20)
    out = {"_ref": (complex(abs(p0.phasor)), noise0)}
    for n in names:
        r = res[n][0]
        rel = r.phasor / p0.phasor if abs(p0.phasor) > 0 else complex(math.nan)
        out[n] = (rel, math.hypot(noise0, 10 ** (-r.quality.estimate_snr_db / 20)))
    return out


def _settled(m, f, plan, res, extra, limits, check) -> Capture:
    """Capture again until two captures in a row agree; keep the last one.

    Agreement means a change below the larger of the limit and three times
    the noise expected from the two estimates, so noise alone never reads as
    settling.
    """
    lim = {**analysis.DEFAULT_FLAG_LIMITS, **(limits or {})}
    prev = _ratios(res, f)
    for k in range(1, int(extra) + 1):
        nxt = check(m.measure(f, plan))
        cur = _ratios(nxt, f)
        ok = True
        for n, (v, noise) in cur.items():
            a, b = complex(prev[n][0]), complex(v)
            if not (np.isfinite(a) and np.isfinite(b)) or abs(a) == 0:
                ok = False
                break
            band = 3 * math.hypot(noise, prev[n][1])
            if abs(abs(b) / abs(a) - 1) * 100 > max(lim["settle_pct"], 100 * band):
                ok = False
            ddeg = abs(math.degrees(np.angle(b / a)))
            if n != "_ref" and ddeg > max(lim["settle_deg"], math.degrees(band)):
                ok = False
        res, prev = nxt, cur
        if ok:
            res.settings["settle_captures"] = k
            return res
    res.settings["settle_captures"] = int(extra)
    res.settings.setdefault("check_flags", []).append("not_settled")
    return res


def refine_points(
    values: dict[float, complex], *, db_step: float = 1.0, deg_step: float = 10.0, limit: int = 40
) -> list[float]:
    """Log midpoints of the steps where a response changes fast, largest steps first."""
    fs = sorted(values)
    cand = []
    for a, b in zip(fs, fs[1:]):
        va, vb = values[a], values[b]
        if not (np.isfinite(va) and np.isfinite(vb)) or abs(va) == 0 or abs(vb) == 0:
            continue
        ddb = abs(20 * math.log10(abs(vb) / abs(va)))
        ddeg = abs(math.degrees(np.angle(vb / va)))
        score = max(ddb / db_step, ddeg / deg_step)
        if score > 1 and b / a > 1.0001:
            cand.append((score, math.sqrt(a * b)))
    cand.sort(reverse=True)
    return sorted(f for _, f in cand[: max(0, limit)])


def _combine_repeats(analyzed, reference, limits) -> dict:
    """Average repeated captures of one point and measure their spread.

    Each capture starts at its own moment, so every signal's phasor is first
    turned by the reference signal's phase in that capture (the reference
    defaults to the first signal). The fundamental phasors are then averaged
    into the first capture's results, and the spread of magnitude (percent)
    and of phase relative to the reference (degrees) is returned per signal.
    """
    lim = {**analysis.DEFAULT_FLAG_LIMITS, **(limits or {})}
    first = analyzed[0][1]
    ref = reference if reference in first else next(iter(first))
    rel = {n: [] for n in first}
    for _, res in analyzed:
        r0 = res[ref][0].phasor
        turn = np.conj(r0) / abs(r0) if abs(r0) > 0 else 1.0
        for n in first:
            rel[n].append(res[n][0].phasor * turn)
    r_first = first[ref][0].phasor
    back = r_first / abs(r_first) if abs(r_first) > 0 else 1.0
    out = {}
    for n, vals in rel.items():
        v = np.asarray(vals)
        mean = complex(v.mean())
        first[n][0].phasor = mean * back
        mag_pct = float(100 * np.std(np.abs(v)) / np.mean(np.abs(v))) if np.mean(np.abs(v)) else 0.0
        ph_deg = float(np.degrees(np.std(np.angle(v / mean)))) if n != ref else 0.0
        fl = []
        if mag_pct > lim["repeat_spread_pct"] or ph_deg > lim["repeat_spread_deg"]:
            fl.append("unstable")
        out[n] = {
            "repeats": len(vals),
            "repeat_mag_spread_pct": mag_pct,
            "repeat_phase_spread_deg": ph_deg,
            "_flags": fl,
        }
    return out


class _Progress:
    """progress.json in the run folder, rewritten after every point.

    Small enough for an agent to read at every check-in: points done, the
    current frequency, elapsed and remaining time, the latest values, and the
    flags seen so far.
    """

    def __init__(self, path, total: int):
        self.path = path / "progress.json"
        self.t0 = time.monotonic()
        self.total = total
        self.flags: dict[str, int] = {}
        self._write({"status": "running", "points_done": 0, "points_planned": total})

    def _write(self, d: dict) -> None:
        """Replace progress.json; a reader holding it open must never stop the sweep.

        Windows refuses to replace a file another process has open, for
        example a check-in reading it, so the write retries for up to 0.2 s and
        then skips this update. Progress is not measurement data.
        """
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
        except OSError:
            return
        for _ in range(20):
            try:
                tmp.replace(self.path)
                return
            except PermissionError:
                time.sleep(0.01)
            except OSError:
                return

    def update(self, done: int, total: int, freq: float, rows: list[dict]) -> None:
        for r in rows:
            for fl in r.get("flags", []) or []:
                self.flags[fl] = self.flags.get(fl, 0) + 1
        el = time.monotonic() - self.t0
        latest = {
            r["signal"]: {
                "mag": r.get("mag"),
                "unit": r.get("unit"),
                "phase_deg": r.get("phase_deg") if r.get("phase_valid") else None,
            }
            for r in rows
            if r.get("kind") in ("signal", "scalar", "expr")
        }
        self._write(
            {
                "status": "running",
                "points_done": done,
                "points_planned": total,
                "freq_hz": freq,
                "elapsed_s": round(el, 1),
                "remaining_s": round(el / done * (total - done), 1) if done else None,
                "latest": latest,
                "flags_so_far": self.flags,
                "updated": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def finish(self, status: str, reason: str) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        d.update(
            status=status,
            reason=reason,
            elapsed_s=round(time.monotonic() - self.t0, 1),
            updated=datetime.now().isoformat(timespec="seconds"),
        )
        self._write(d)


def _sleep_checked(seconds: float, check) -> None:
    """Sleep in short steps so an emergency stop is seen within 50 ms."""
    end = time.monotonic() + seconds
    while True:
        check()
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.05, left))


class _EstopGuard:
    """A measurement whose every capture first checks the emergency stop.

    The sweep sets ``check`` once the run folder exists. Repeats, settle
    captures and the retry of an invalid capture all go through ``measure``,
    so none of them runs after an ESTOP file appears.
    """

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "measurement")
        self.check = None

    def measure(self, freq, plan):
        if self.check is not None:
            self.check()
        return self.inner.measure(freq, plan)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _trap_stop_signals():
    """Turn a terminate or console-close signal into KeyboardInterrupt, so the
    sweep ends through its normal clean-up: output off, setups restored.

    Returns a function that puts the previous handlers back. Its ``hold()``
    makes the handlers ignore a signal, so a second signal during the off
    sequence cannot cut it short. A hard kill of the process cannot be caught,
    and on Windows neither can a terminate from another process; for those,
    create the run folder's STOP or ESTOP file instead.
    """
    import signal
    import threading

    held = [False]

    def nothing():
        return None

    nothing.hold = nothing
    if threading.current_thread() is not threading.main_thread():
        return nothing

    def handler(signum, frame):
        if held[0]:
            return
        raise KeyboardInterrupt(f"signal {signum}")

    saved = {}
    for name in ("SIGTERM", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            saved[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):
            pass

    def restore():
        for sig, old in saved.items():
            try:
                signal.signal(sig, old)
            except (ValueError, OSError):
                pass

    def hold():
        held[0] = True

    restore.hold = hold
    return restore


def _expr_rows(exprs, values, groups, rel_noise, calibration, base, rows, f, point_flags):
    out = []
    for name, (fn, unit) in exprs.items():
        tv = _Tracked(values, groups)
        try:
            val = complex(fn(tv))
        except (ZeroDivisionError, KeyError, ValueError) as exc:
            val = complex(math.nan, math.nan)
            point_flags.append(f"expr_error:{name}:{type(exc).__name__}")
        gs = {groups.get(k) for k in tv.used}
        unc = _propagate(fn, values, tv.used, rel_noise, val)
        if math.isfinite(unc):
            rel_noise[name] = unc
        valid = len(gs) == 1 and None not in gs
        val, outside = _calibrate(calibration, name, f, val)
        fl = ["cal_extrapolated"] if outside else []
        freq_used = next((r["freq_used"] for r in rows if r.get("kind") == "signal"), f)
        out.append(
            {
                **base,
                "freq_used": freq_used,
                "measurement": "expr",
                "group": next(iter(gs)) if valid else "",
                "signal": name,
                "kind": "expr",
                "order": 1,
                "tone_freq": freq_used,
                **_complex_fields(val),
                "unit": unit,
                "phase_valid": int(valid),
                "estimate_snr_db": -20 * math.log10(unc) if unc > 0 else math.nan,
                "phase_uncertainty_deg": math.degrees(unc) if unc == unc else math.nan,
                "flags": fl,
            }
        )
        values[name] = val
        groups[name] = next(iter(gs)) if valid else None
    return out


def _propagate(fn, values: dict, used, rel_noise: dict, val: complex) -> float:
    """Relative 1-sigma noise of an expression, by perturbing each input.

    Each input's noise is circular with RMS ``rel_noise * |value|``. Moving the
    input by that much along the real and the imaginary axis, one input at a
    time, gives the expression's response; the squares add. A difference of
    two nearly equal values therefore shows its true, larger noise.
    """
    if not (val == val) or abs(val) == 0:
        return math.nan
    total, seen = 0.0, False
    for k in used:
        r = rel_noise.get(k)
        if r is None or not math.isfinite(r) or k not in values:
            continue
        d = r * abs(values[k]) / math.sqrt(2.0)
        for step in (d, 1j * d):
            moved = dict(values)
            moved[k] = values[k] + step
            try:
                total += abs(complex(fn(moved)) - val) ** 2
            except (ZeroDivisionError, KeyError, ValueError):
                return math.nan
        seen = True
    return math.sqrt(total) / abs(val) if seen else math.nan


def _source_freq_flags(cmd: float, got: float) -> list[str]:
    """Sanity check of the frequency the source reports against the command."""
    if not (got and math.isfinite(got)):
        return ["source_freq_missing"]
    rel = abs(got - cmd) / cmd
    if rel > 1e-3:
        return ["source_freq_wrong"]
    if rel > 1e-6:
        return ["source_freq_snapped"]
    return []


def _with_readback(res: Capture) -> Capture:
    """Fill sample rate and record length from the time axis when not reported."""
    st = res.settings
    first = next(iter(res.signals.values()))
    if not st.get("sample_rate"):
        if res.t is not None and len(res.t) > 1:
            st["sample_rate"] = 1.0 / float(np.median(np.diff(res.t)))
        elif res.dt:
            st["sample_rate"] = 1.0 / res.dt
        st["sample_rate_source"] = "time_axis"
    if not st.get("record_length"):
        st["record_length"] = len(first)
        st["record_length_source"] = "samples"
    return res


def _capture_dt(res: Capture) -> float | None:
    if res.t is not None and len(res.t) > 1:
        return float(np.median(np.diff(res.t)))
    return res.dt


def _checked_capture(m, f, plan, res, harmonics, on_invalid, freq_flags) -> Capture:
    """Validate a capture's read-back against its plan; retry, flag or stop."""
    res = _with_readback(res)
    flags, invalid = validate_capture(plan, res.settings, harmonics=harmonics, dt=_capture_dt(res))
    if invalid and on_invalid == "retry":
        again = _with_readback(m.measure(f, plan))
        flags2, invalid2 = validate_capture(
            plan, again.settings, harmonics=harmonics, dt=_capture_dt(again)
        )
        res, flags, invalid = again, flags2 + ["retried"], invalid2
    if invalid:
        flags.append("capture_invalid")
        if on_invalid == "stop":
            raise StopSweep(f"invalid capture at {f:.6g} Hz: {', '.join(flags)}")
    res.settings["check_flags"] = flags + list(freq_flags)
    return res


def run_level_sweep(freq: float, levels, source, measurements, **kw) -> SweepResult:
    """Sweep the drive amplitude at one frequency, for distortion against level.

    Takes the same arguments as :func:`run_sweep` except ``amplitude`` and
    ``refine_on``. Plots and reviews use the drive level as the x axis, so THD,
    harmonics and gain compression read against drive. The levels run in the
    order given; put them low to high so the device is never overdriven first.
    """
    levels = [float(a) for a in levels]
    order = iter(levels)
    kw.pop("refine_on", None)
    kw.setdefault("title", "level_sweep")
    extra_meta = {"x_axis": "amplitude", "level_freq": float(freq), "levels": levels}
    return run_sweep(
        [float(freq)] * len(levels),
        source,
        measurements,
        amplitude=lambda f: next(order),
        _extra_meta=extra_meta,
        **kw,
    )


def _complex_fields(v: complex) -> dict:
    return {
        "re": v.real,
        "im": v.imag,
        "mag": abs(v),
        "phase_deg": math.degrees(math.atan2(v.imag, v.real)),
    }


def _capture_rows(
    captures,
    f,
    plan,
    base,
    mname,
    tones,
    harmonics,
    refine,
    reference,
    limits,
    segments,
    calibration,
    values,
    groups,
    units,
    writer,
    point,
    rel_noise=None,
    tol=(None, None, "dft", "fft"),
):
    extra = _tones_at(tones, f)
    res = captures[0]
    analyzed = [
        analysis.analyze(
            c.signals,
            f,
            t=c.t,
            dt=c.dt,
            window=plan.window,
            tones=extra,
            harmonics=harmonics,
            refine=refine,
            reference=reference,
            full_scale=c.full_scale,
            segments=segments,
            limits=limits,
            tolerance=tol[0],
            tolerance_hz=tol[1],
            estimator=tol[2],
            search=tol[3],
        )
        for c in captures
    ]
    f_used, results = analyzed[0]
    res.settings.setdefault("waveform_point", point)
    if (
        writer is not None
        and getattr(writer, "save_spectra", True)
        and not getattr(res, "_spectra_saved", False)
    ):
        res._spectra_saved = True
        dt = _capture_dt(res)
        writer.add_spectra(
            point,
            f,
            {
                k: analysis.reduced_spectrum(v, dt, window=plan.window)
                for k, v in res.signals.items()
            },
            res.units,
        )
    spread = _combine_repeats(analyzed, reference, limits) if len(analyzed) > 1 else {}
    for c in captures[1:]:
        for k in c.settings.get("check_flags", []):
            if k not in res.settings.setdefault("check_flags", []):
                res.settings["check_flags"].append(k)
    sr = res.settings.get("sample_rate")
    rl = res.settings.get("record_length")
    extra_settings = json.dumps(
        {
            k: v
            for k, v in res.settings.items()
            if k not in ("sample_rate", "record_length", "check_flags")
        },
        default=str,
        separators=(",", ":"),
    )
    common = {
        **base,
        "freq_used": f_used,
        "measurement": mname,
        "group": res.group,
        "sample_rate": sr,
        "record_length": rl,
        "cycles": f_used * rl / sr if sr else "",
        "rbw": sr / rl if rl else "",
        "window": plan.window,
    }
    rows = []
    for name, rs in results.items():
        unit = res.units.get(name, "V")
        units[name] = unit
        for r in rs:
            kind = "signal" if r.order == 1 else ("tone" if r.order == 0 else "harmonic")
            flags = list(r.flags)
            if r.order == 1:
                if res.clipped.get(name):
                    flags.append("clip")
                flags += [f"plan_{x}" for x in plan.limited_by]
                flags += res.settings.get("check_flags", [])
                v, outside = _calibrate(calibration, name, f_used, r.phasor)
                if outside:
                    flags.append("cal_extrapolated")
                values[name] = v
                groups[name] = res.group
                if rel_noise is not None and r.quality is not None:
                    rel_noise[name] = 10 ** (-r.quality.estimate_snr_db / 20)
            else:
                # The same correction at the harmonic's or tone's own frequency,
                # so ratios to the corrected fundamental stay right.
                v, outside = _calibrate(calibration, name, r.freq, r.phasor)
                if outside:
                    flags.append("cal_extrapolated")
            row = {
                **common,
                "signal": name,
                "kind": kind,
                "order": r.order,
                "tone_freq": r.freq,
                **_complex_fields(v),
                "unit": unit,
                "phase_valid": 0,
                "flags": sorted(set(flags)),
            }
            if r.order == 1:
                q, st = r.quality, r.stability
                if name in spread:
                    row.update(spread[name])
                    row["flags"] = sorted(set(row["flags"]) | set(spread[name]["_flags"]))
                row.update(
                    offset_bins=r.offset_bins,
                    strongest_freq_hz=r.strongest_freq,
                    strongest_dbc=r.strongest_dbc,
                    fit_residual_rms=r.extra.get("residual_rms", math.nan),
                    coherent_off_bins=r.extra.get("off_bins", math.nan),
                    coherent_leak_db=r.extra.get("leak_db", math.nan),
                    snr_db=q.snr_db,
                    estimate_snr_db=q.estimate_snr_db,
                    phase_uncertainty_deg=q.phase_uncertainty_deg,
                    settings=extra_settings,
                    thd_pct=q.thd_pct,
                    sinad_db=q.sinad_db,
                    noise_rms=q.noise_rms,
                    clip=r.clip,
                )
                if st and st.segments:
                    row.update(
                        freq_offset_hz=st.freq_offset_hz,
                        drift_hz_per_s=st.drift_hz_per_s,
                        phase_jitter_deg=st.phase_jitter_deg,
                        amplitude_variation_pct=st.amplitude_variation_pct,
                    )
                writer.add_spurs(point, f_used, name, q.spurs)
            rows.append(row)
    return rows


def _readings_rows(res, f, base, mname, calibration, values, groups, units):
    rows = []
    for name, raw in res.values.items():
        v = complex(raw)
        is_phasor = isinstance(raw, complex)
        v, outside = _calibrate(calibration, name, f, v)
        group = res.group if (res.group and is_phasor) else None
        values[name] = v
        groups[name] = group
        units[name] = res.units.get(name, "")
        rows.append(
            {
                **base,
                "freq_used": f,
                "measurement": mname,
                "group": group or "",
                "signal": name,
                "kind": "scalar",
                "order": 1,
                "tone_freq": f,
                **_complex_fields(v),
                "unit": units[name],
                "phase_valid": int(bool(group)),
                "flags": ["cal_extrapolated"] if outside else [],
            }
        )
    return rows


def _status_line(i, n, f, rows) -> str:
    sig = next((r for r in rows if r.get("kind") in ("expr", "signal", "scalar")), None)
    txt = f"{sig['signal']}={sig['mag']:.4g} {sig['unit']}" if sig else ""
    return f"point {i + 1}/{n}  {f:12.6g} Hz  {txt}        "


def _live_update(state, path):
    """Redraw plot.png after each point, so a person or an agent can open it mid-run."""
    from sweep.plot import plot_run

    try:
        plot_run(path)
    except Exception:
        pass
    return True
