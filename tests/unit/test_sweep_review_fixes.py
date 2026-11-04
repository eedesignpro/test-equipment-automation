"""Checks for the defects found in the 2026-11-03 code review of src/sweep."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from sweep.capture import CaptureRules, plan_capture
from sweep.interfaces import Capture
from sweep.interp import common_grid
from sweep.run import StopSweep, run_sweep
from sweep.simulate import SimulatedBench
from sweep.store import Run, unwrap_finite


def _scope(sig):
    def rx(freq, plan):
        t = np.arange(plan.record_length) / plan.sample_rate
        return Capture({k: fn(freq, t) for k, fn in sig.items()}, t=t)

    return rx


def test_channel_is_looked_up_at_read_time_not_at_build_time():
    from sweep.instruments.scalar import keysight_b2900_current

    class Smu:
        connected = False

        def channel(self, n):
            if n == 2 and not self.connected:
                raise ValueError("channel must be one of [1]")
            return SimpleNamespace(meas_current=lambda: 0.25)

    smu = Smu()
    m = keysight_b2900_current(smu, 2)
    smu.connected = True
    assert m.measure(1.0, None).values["I_smu2"] == 0.25


def test_one_nan_point_does_not_erase_later_phase(tmp_path):
    def rx(freq, plan):
        t = np.arange(plan.record_length) / plan.sample_rate
        b = 0.0 if freq == 1000.0 else 0.5
        return Capture(
            {"a": np.cos(2 * np.pi * freq * t), "b": b * np.cos(2 * np.pi * freq * t - 0.3)}, t=t
        )

    res = run_sweep(
        [100.0, 300.0, 1000.0, 3000.0, 10000.0],
        lambda f, a: None,
        [rx],
        expressions={"H": lambda v: v["b"] / v["a"]},
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    ph = Run(res.path).series("H").phase_deg
    assert np.isfinite(ph[3]) and np.isfinite(ph[4])
    assert np.allclose(
        unwrap_finite(np.array([0.0, np.nan, 1.0])), [0.0, np.nan, 1.0], equal_nan=True
    )


def test_common_grid_merges_refined_duplicates():
    a = np.array([100.0, 1000.0, 2511.8864, 10000.0])
    b = np.array([100.0001, 1000.0003, 2511.8874, 10000.002])
    g = common_grid([a, b])
    assert len(g) == 4 and g[0] >= 100.0001 * (1 - 1e-9)


def test_plan_keeps_sample_rate_limits_under_the_record_cap():
    p = plan_capture(1e6, CaptureRules(max_sample_rate=10e6, max_record_length=500))
    assert p.sample_rate <= 10e6
    p = plan_capture(
        1e3, CaptureRules(min_samples_per_cycle=50, max_record_length=2000, min_record_length=None)
    )
    assert p.sample_rate >= 50e3 or "min_samples_per_cycle" in p.limited_by


def test_dry_run_uses_a_capture_function(tmp_path):
    from sweep.capture import CapturePlan

    def fn(f):
        return CapturePlan(f, 1e6, 1234, 1234e-6, f * 1234e-6, 1e6 / 1234, "hann")

    res = run_sweep(
        [1000.0],
        lambda f, a: None,
        [_scope({})],
        capture=fn,
        folder=tmp_path,
        dry_run=True,
        quiet=True,
    )
    assert Run(res.path).meta["plan"][0]["record_length"] == 1234


def test_restore_failure_and_stop_before_the_folder(tmp_path):
    class Rx:
        name = "rx"

        def measure(self, f, plan):
            t = np.arange(plan.record_length) / plan.sample_rate
            return Capture({"y": np.cos(2 * np.pi * f * t)}, t=t)

        def close(self):
            raise OSError("restore failed")

    res = run_sweep([1000.0], lambda f, a: None, [Rx()], folder=tmp_path, quiet=True, plot=False)
    assert res.status == "complete_restore_failed"

    class Opener:
        name = "o"

        def open(self):
            raise KeyboardInterrupt

        def measure(self, f, plan):  # pragma: no cover
            return {}

    with pytest.raises(KeyboardInterrupt):
        run_sweep([1000.0], lambda f, a: None, [Opener()], folder=tmp_path, quiet=True)


def test_expression_using_get_keeps_phase_valid(tmp_path):
    rx = _scope(
        {
            "a": lambda f, t: np.cos(2 * np.pi * f * t),
            "b": lambda f, t: 0.5 * np.cos(2 * np.pi * f * t),
        }
    )
    res = run_sweep(
        [1000.0],
        lambda f, a: None,
        [rx],
        expressions={"H": lambda v: v.get("b") / v.get("a")},
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    h = Run(res.path).series("H")
    assert h.phase_valid and np.isfinite(h.phase_uncertainty_deg[0])


def test_level_sweep_condensed_keeps_every_level(tmp_path):
    from sweep.run import run_level_sweep
    from sweep.views import condensed

    state = {"a": 0.0}

    def src(f, a):
        state["a"] = a
        return {"freq": f, "amplitude": a}

    rx = _scope({"y": lambda f, t: state["a"] * np.cos(2 * np.pi * f * t)})
    res = run_level_sweep(
        1000.0, np.linspace(0.1, 3.0, 30), src, [rx], folder=tmp_path, quiet=True, plot=False
    )
    header, rows = condensed(res.path, 25)
    assert header[0] == "drive_vpp" and len(rows) >= 25


def test_harmonics_are_calibrated_with_the_fundamental(tmp_path):
    from sweep.calibration import Calibration

    b = SimulatedBench({"v": lambda f: 1.0}, harmonic_levels={2: 0.1}, noise_rms=1e-6)
    fr = [1000.0, 2000.0, 4000.0]
    cal = Calibration(np.array([100.0, 1e5]), np.array([10.0 + 0j, 10.0 + 0j]))
    res = run_sweep(
        fr,
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        folder=tmp_path,
        quiet=True,
        plot=False,
        calibration={"v": cal},
        capture=CaptureRules(max_record_length=20_000),
    )
    run = Run(res.path)
    fund = run.series("v", kind="signal").mag
    h2 = run.series("v", kind="harmonic", order=2).mag
    assert np.allclose(h2 / fund, 0.1, rtol=0.02)


def test_two_parts_on_one_instrument_share_one_saved_setup():
    import contextlib

    from sweep.instruments.base import Preserved

    log = []

    class Inst:
        @contextlib.contextmanager
        def preserved_setup(self):
            log.append("save")
            yield
            log.append("recall")

    inst = Inst()
    a, b = Preserved(inst), Preserved(inst)
    a.open()
    b.open()
    a.close()
    assert log == ["save"]
    b.close()
    assert log == ["save", "recall"]


def test_zero_amplitude_read_back_is_kept(tmp_path):
    res = run_sweep(
        [1000.0],
        lambda f, a: {"freq": f, "amplitude": 0.0},
        [lambda f, p: {"x": 1.0}],
        amplitude=0.0,
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    row = Run(res.path).rows[0]
    assert row["amplitude_set"] == 0 and not math.isnan(row["amplitude_set"])
    assert StopSweep  # imported for the public name


def test_stop_file_ends_the_sweep_cleanly(tmp_path):
    b = SimulatedBench({"x": lambda f: 1.0})
    calls = {"n": 0}
    src = b.source()

    class Stopper:
        name = "rx"

        def measure(self, f, plan):
            calls["n"] += 1
            if calls["n"] == 2:
                (next(tmp_path.glob("*_stopper")) / "STOP").write_text("")
            return b.receiver().measure(f, plan)

    res = run_sweep(
        [100.0, 200.0, 300.0, 400.0],
        src,
        [Stopper()],
        amplitude=1.0,
        folder=tmp_path,
        title="stopper",
        quiet=True,
        plot=False,
    )
    assert res.status == "stopped" and res.points == 2 and not b.output_on
    assert Run(res.path).meta["reason"] == "STOP file"


def test_estop_ends_now_runs_teardown_and_writes_progress(tmp_path):
    b = SimulatedBench({"x": lambda f: 1.0})
    order = []

    class Src:
        name = "src"

        def set(self, f, a):
            return {"freq": f}

        def output(self, state):
            order.append(("output", state))

    class Rx:
        name = "rx"
        n = 0

        def measure(self, f, plan):
            Rx.n += 1
            if Rx.n == 2:
                (next(tmp_path.glob("*_estop")) / "ESTOP").write_text("")
            return b.receiver().measure(f, plan)

    res = run_sweep(
        [100.0, 200.0, 300.0, 400.0],
        Src(),
        [Rx()],
        amplitude=1.0,
        folder=tmp_path,
        title="estop",
        quiet=True,
        plot=False,
        settle_s=0.2,
        teardown=lambda: order.append("teardown"),
    )
    assert res.status == "estopped" and res.points == 2
    assert order[-3:] == [("output", False), "teardown", ("output", False)]
    import json

    prog = json.loads((res.path / "progress.json").read_text())
    assert prog["status"] == "estopped" and prog["points_done"] == 2


def test_cli_stop_and_status(tmp_path, capsys):
    from sweep.__main__ import main

    b = SimulatedBench({"x": lambda f: 1.0})
    res = run_sweep(
        [100.0, 200.0],
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    main(["status", str(tmp_path)])
    out = capsys.readouterr().out
    assert "points_done=2" in out and "latest x" in out
    main(["estop", str(res.path)])
    assert (res.path / "ESTOP").exists()


def test_a_reader_holding_progress_open_does_not_stop_the_sweep(tmp_path):
    b = SimulatedBench({"x": lambda f: 1.0})
    held = []

    class Rx:
        name = "rx"

        def measure(self, f, plan):
            prog = next(tmp_path.glob("*_held")) / "progress.json"
            if prog.exists() and not held:
                held.append(open(prog, encoding="utf-8"))  # a check-in reading it
            return b.receiver().measure(f, plan)

    try:
        res = run_sweep(
            [100.0, 200.0, 300.0, 400.0],
            b.source(),
            [Rx()],
            amplitude=1.0,
            folder=tmp_path,
            title="held",
            quiet=True,
            plot=False,
        )
    finally:
        for fh in held:
            fh.close()
    assert res.status == "complete" and res.points == 4
