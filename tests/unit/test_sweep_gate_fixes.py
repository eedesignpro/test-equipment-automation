"""The MSO receiver and 33500 source adapters, and the stop paths found by the iteration 11 gates.

The adapters run against small fakes that hold the driver calls they use and
log them. A pass says the adapters call the drivers in the right order and
carry the read-back through; it says nothing new about the instruments.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sweep.__main__ import _running
from sweep.capture import CaptureRules, plan_capture, validate_capture
from sweep.instruments.keysight_33500 import Keysight33500Source
from sweep.instruments.tektronix_mso import TektronixMsoReceiver
from sweep.interfaces import Capture
from sweep.multitone import run_multitone_sweep
from sweep.plan import frequencies
from sweep.run import run_level_sweep, run_sweep
from sweep.simulate import SimulatedBench, rc_lowpass
from sweep.store import Run

# ---------- fakes ----------


class _Ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeMso:
    """The slice of ``Tektronix_MSO`` the receiver uses, with a 1 kHz tone on each channel."""

    def __init__(self, *, extra_point_on=None, clip_first=False):
        self.log: list = []
        self.sr, self.rl = 1e6, 1000
        self.extra_point_on = extra_point_on
        self.clip_first = clip_first
        self.captures = 0
        self.scales = {"CH1": 0.1, "CH2": 0.1}
        self.trig = _Ns(
            set_mode=lambda m: self.log.append(("trig_mode", m)),
            set_edge=lambda s, lv: self.log.append(("trig_edge", s)),
        )
        self.acq = _Ns(cmd_single=self._single, get_state=lambda: 0)
        self.horz = _Ns(
            set_manual=self._set_manual,
            get_samplerate=lambda: self.sr,
            get_record_length=lambda: self.rl,
            get_scale=lambda: self.rl / self.sr / 10,
            get_div_count=lambda: 10,
            get_mode=lambda: "MANUAL",
        )
        self.data = _Ns(get_waveform_codes=self._codes)
        self.ch1 = self._channel("CH1")
        self.ch2 = self._channel("CH2")

    def _channel(self, name):
        def set_scale(v):
            self.log.append(("scale", name, v))
            self.scales[name] = v
            return v

        return _Ns(
            set_selected=lambda on: None,
            probe=_Ns(get_units=lambda: "V"),
            get_clipping=lambda: self.clip_first and self.captures == 1,
            get_scale=lambda: self.scales[name],
            get_offset=lambda: 0.0,
            set_scale=set_scale,
        )

    def _set_manual(self, sr, rl):
        self.log.append(("set_manual", sr, rl))
        self.sr, self.rl = float(sr), int(rl)
        return self.sr, self.rl

    def _single(self):
        self.captures += 1
        self.log.append(("single",))

    def _codes(self, src, width=2):
        n = self.rl + (1 if src == self.extra_point_on else 0)
        t = np.arange(n) / self.sr
        amp = 0.25 if src == "CH1" else 0.1
        codes = np.round(amp * np.sin(2 * np.pi * 1e3 * t) / 1e-5).astype(np.int16)
        pre = {
            "ymult": 1e-5,
            "yoff": 0.0,
            "yzero": 0.0,
            "xincr": 1 / self.sr,
            "xzero": 0.0,
            "pt_off": 0.0,
        }
        return codes, pre


class FakeAwgChannel:
    def __init__(self, log):
        self.log = log

    def set_output_state(self, on):
        self.log.append(("output", bool(on)))

    def set_amplitude_unit(self, unit):
        self.log.append(("unit", unit))

    def set_load(self, ohms):
        self.log.append(("load", ohms))

    def set_waveform_sine(self, freq, ampl, offset=0.0):
        self.log.append(("sine", freq, ampl))
        return _Ns(frequency=freq, amplitude=ampl)

    def set_frequency(self, freq):
        self.log.append(("freq", freq))
        return freq

    def set_amplitude(self, ampl):
        self.log.append(("ampl", ampl))
        return ampl

    def set_waveform_arb(self, samples, *, sample_rate, ampl, offset, name):
        self.log.append(("arb", len(samples), sample_rate, ampl, name))
        return _Ns(amplitude=ampl)

    def get_load(self):
        return 5.0


class FakeAwg:
    def __init__(self):
        self.log: list = []
        self.ch1 = FakeAwgChannel(self.log)
        self.sync = _Ns(
            set_source=lambda n: self.log.append(("sync_source", n)),
            set_state=lambda on: self.log.append(("sync", on)),
        )


def _plan():
    return plan_capture(1e3, CaptureRules(min_cycles=1, max_record_length=1000))


# ---------- MSO receiver ----------


def test_mso_capture_carries_codes_scales_and_a_valid_read_back():
    scope = FakeMso(extra_point_on="CH2")
    rx = TektronixMsoReceiver(scope, {"a": "CH1", "b": "CH2"}, preserve=False)
    rx.open()
    plan = _plan()
    cap = rx.measure(1e3, plan)
    assert isinstance(cap, Capture)
    n = len(cap.t)
    assert all(len(v) == n for v in cap.signals.values())
    for name in ("a", "b"):
        raw = cap.raw[name]
        assert raw["codes"].dtype == np.int16 and raw["codes"].size == n
        back = raw["yzero"] + raw["ymult"] * (raw["codes"].astype(float) - raw["yoff"])
        assert np.array_equal(back, cap.signals[name])
    flags, invalid = validate_capture(plan, cap.settings)
    assert not invalid, flags
    assert set(cap.settings["time_s"]) == {"horizontal_setup", "acquire", "transfer", "readback"}


def test_mso_prepared_horizontal_is_used_once_then_set_again():
    scope = FakeMso()
    rx = TektronixMsoReceiver(scope, {"a": "CH1"}, preserve=False)
    rx.open()
    plan = _plan()
    rx.prepare_horizontal(plan)
    rx.measure(1e3, plan)
    rx.measure(1e3, plan)
    sets = [e for e in scope.log if e[0] == "set_manual"]
    assert len(sets) == 2  # one from prepare, one from the second measure
    assert scope.log.index(sets[0]) < scope.log.index(("single",))


def test_mso_autoscale_widens_a_clipped_channel_and_captures_again():
    scope = FakeMso(clip_first=True)
    rx = TektronixMsoReceiver(scope, {"a": "CH1"}, preserve=False, autoscale=True)
    rx.open()
    cap = rx.measure(1e3, _plan())
    assert cap.settings["tries"] >= 2
    assert ("scale", "CH1", pytest.approx(0.25)) in scope.log


def test_mso_waveforms_reload_exactly_from_a_run(tmp_path):
    scope = FakeMso()
    rx = TektronixMsoReceiver(scope, {"a": "CH1", "b": "CH2"}, preserve=False)
    src = _Ns(set=lambda f, a: {"freq": f, "amplitude": a})
    res = run_sweep(
        [1e3],
        src,
        [rx],
        capture=lambda f: _plan(),
        folder=tmp_path,
        quiet=True,
        plot=False,
        refine=False,
        harmonics=0,
    )
    w = Run(res.path).waveform(0)
    assert np.array_equal(w["codes"]["a"], scope._codes("CH1")[0])
    assert np.allclose(w["signals"]["a"], 1e-5 * scope._codes("CH1")[0], rtol=0, atol=0)


# ---------- 33500 source ----------


def test_33500_open_set_output_close_order():
    awg = FakeAwg()
    src = Keysight33500Source(awg, 1, max_vpp=1.0, load=5.0, preserve=False)
    src.open()
    src.set(1e3, 0.5)
    src.output(True)
    src.set(2e3, 0.4)
    src.close()
    assert awg.log == [
        ("output", False),
        ("unit", "VPP"),
        ("load", 5.0),
        ("sync_source", 1),
        ("sync", True),
        ("sine", 1e3, 0.5),
        ("output", True),
        ("freq", 2e3),
        ("ampl", 0.4),
        ("output", False),
    ]


def test_33500_refuses_above_the_limit_before_any_write():
    awg = FakeAwg()
    src = Keysight33500Source(awg, 1, max_vpp=1.0, preserve=False)
    src.open()
    n = len(awg.log)
    with pytest.raises(ValueError):
        src.set(1e3, 1.5)
    with pytest.raises(ValueError):
        src.set_multitone([1e3, 2e3], 1.5, 1e3)
    assert len(awg.log) == n


def test_33500_multitone_is_one_base_period_at_ten_samples_per_top_cycle():
    awg = FakeAwg()
    src = Keysight33500Source(awg, 1, max_vpp=1.0, preserve=False)
    src.open()
    got = src.set_multitone([200.0, 300.0, 500.0], 0.6, 100.0)
    arb = [e for e in awg.log if e[0] == "arb"][0]
    assert arb[1] == 50 and arb[2] == pytest.approx(5000.0) and arb[4] == "SWEEPMT"
    assert got["points"] == 50


# ---------- stop paths ----------


def _bench():
    return SimulatedBench(
        {"vin": lambda f: 1.0, "vout": rc_lowpass(1e3, 159.155e-9)}, noise_rms=1e-6
    )


def test_stop_if_ends_a_pipelined_level_sweep_before_the_next_level(tmp_path):
    b = _bench()
    res = run_level_sweep(
        1e3,
        [0.1, 0.2, 0.3, 0.4, 0.5],
        b.source(),
        [b.receiver()],
        stop_if=lambda v: "vin above 0.1 V" if abs(v["vin"]) > 0.1 else None,
        folder=tmp_path,
        quiet=True,
        plot=False,
        capture=CaptureRules(max_record_length=20_000),
    )
    levels = [e[2] for e in b.log if e[0] == "set"]
    assert levels == [0.1, 0.2, 0.3]
    assert res.status == "stopped"


def test_estop_is_seen_between_repeat_captures(tmp_path):
    b = _bench()
    rx = b.receiver()
    calls = []
    inner = rx.measure

    def measure(freq, plan):
        calls.append(freq)
        if len(calls) == 1:
            run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
            (run_dir / "ESTOP").write_text("test", encoding="utf-8")
        return inner(freq, plan)

    rx.measure = measure
    res = run_sweep(
        [1e3, 2e3],
        b.source(),
        [rx],
        amplitude=1.0,
        repeats=4,
        folder=tmp_path,
        quiet=True,
        plot=False,
        capture=CaptureRules(max_record_length=20_000),
    )
    assert len(calls) == 1
    assert res.status != "complete"
    assert b.output_on is False


def test_stop_command_writes_into_the_running_run_only(tmp_path):
    old, live = tmp_path / "2026-01-01_00-00-00_old", tmp_path / "2026-01-02_00-00-00_live"
    for d, status in ((old, "complete"), (live, "running")):
        d.mkdir()
        (d / "meta.json").write_text("{}", encoding="utf-8")
        (d / "progress.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    (old / "late_file.png").write_text("x", encoding="utf-8")  # makes the old run newest by time
    assert _running(tmp_path) == live
    (live / "progress.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        _running(tmp_path)


def test_multitone_refuses_repeats_it_cannot_take(tmp_path):
    b = _bench()
    with pytest.raises(ValueError):
        run_multitone_sweep(
            frequencies(1e3, 1e4, points=8),
            b.source(),
            [b.receiver()],
            repeats=3,
            folder=tmp_path,
            quiet=True,
            plot=False,
        )


def test_coherent_with_an_explicit_dft_keeps_the_dft(tmp_path):
    b = _bench()
    b.sample_clock_hz = 6.25e9
    res = run_sweep(
        [1e3],
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        coherent=True,
        estimator="dft",
        folder=tmp_path,
        quiet=True,
        plot=False,
        capture=CaptureRules(max_record_length=20_000),
    )
    assert Run(res.path).meta["estimator"] == "dft"
