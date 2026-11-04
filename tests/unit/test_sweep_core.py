"""Core of the frequency-sweep toolkit, on synthetic signals only."""

import math

import numpy as np
import pytest

from sweep import analysis
from sweep.capture import CapturePlan, CaptureRules, plan_capture, rule_value
from sweep.interfaces import Capture, Readings
from sweep.plan import amplitude_at, clamp, first_order, frequencies
from sweep.run import StopSweep, run_sweep
from sweep.simulate import SimulatedBench, delay, rc_lowpass, resonance
from sweep.store import Run
from sweep.views import condensed, review

FS = 1e6


def _sine(f, n=100_000, amp=1.0, ph=0.0, fs=FS):
    t = np.arange(n) / fs
    return t, amp * np.cos(2 * np.pi * f * t + ph)


# ---------- analysis ----------


@pytest.mark.parametrize("window", analysis.WINDOWS)
def test_tone_gives_rms_and_phase(window):
    t, x = _sine(1000.0, amp=2.0, ph=0.7)
    p = analysis.tone(x, 1000.0, t=t, window=window)
    assert abs(p) == pytest.approx(2.0 / math.sqrt(2), rel=2e-3)
    assert np.angle(p) == pytest.approx(0.7, abs=2e-3)


def test_refine_frequency_recovers_unlocked_clock():
    f_true = 1000.0 * (1 + 50e-6) + 0.31
    t, x = _sine(f_true, n=200_000)
    f = analysis.refine_frequency(x, 1000.0, t=t)
    assert f == pytest.approx(f_true, abs=1e-3)


def test_analyze_ratio_phase_and_freq_mismatch_flag():
    f_true = 1007.0
    t, a = _sine(f_true, amp=1.0, ph=0.2)
    _, b = _sine(f_true, amp=0.5, ph=-0.3)
    f_used, res = analysis.analyze({"a": a, "b": b}, 1000.0, t=t)
    assert f_used == pytest.approx(f_true, abs=1e-3)
    h = res["b"][0].phasor / res["a"][0].phasor
    assert abs(h) == pytest.approx(0.5, rel=1e-4)
    assert np.angle(h) == pytest.approx(-0.5, abs=1e-4)
    assert "peak_off_target" in res["a"][0].flags
    assert res["a"][0].offset_bins == pytest.approx(0.7, abs=0.01)


def test_unrefined_wrong_frequency_loses_amplitude_but_ratio_survives():
    t, a = _sine(1003.0)
    _, b = _sine(1003.0, amp=0.5)
    _, res = analysis.analyze({"a": a, "b": b}, 1000.0, t=t, refine=False)
    assert abs(res["a"][0].phasor) < 0.7
    assert abs(res["b"][0].phasor / res["a"][0].phasor) == pytest.approx(0.5, rel=1e-6)


def test_harmonics_thd_snr():
    t, x = _sine(1000.0)
    x = x + 0.03 * np.cos(2 * np.pi * 2000 * t) + 0.04 * np.cos(2 * np.pi * 3000 * t)
    x = x + np.random.default_rng(0).normal(0, 1e-3, x.size)
    _, res = analysis.analyze({"x": x}, 1000.0, t=t)
    r = res["x"][0]
    assert r.quality.thd_pct == pytest.approx(5.0, rel=0.02)
    assert r.quality.snr_db == pytest.approx(20 * math.log10(0.7071 / 1e-3), abs=1.0)
    h2 = next(h for h in res["x"] if h.order == 2)
    assert abs(h2.phasor) == pytest.approx(0.03 / math.sqrt(2), rel=0.01)
    assert "nonlinear" in r.flags


def test_extra_tone_from_same_capture():
    t, x = _sine(1000.0)
    x = x + 0.2 * np.cos(2 * np.pi * 60 * t)
    _, res = analysis.analyze({"x": x}, 1000.0, t=t, tones=[60.0])
    tone60 = next(r for r in res["x"] if r.order == 0)
    assert abs(tone60.phasor) == pytest.approx(0.2 / math.sqrt(2), rel=0.01)


def test_alias_and_spur_listing():
    assert analysis.alias_frequency(2e6, 1.5e6) == pytest.approx(0.5e6)
    assert analysis.alias_frequency(900e3, 1e6) == pytest.approx(100e3)
    t, x = _sine(1000.0)
    x = x + 0.01 * np.cos(2 * np.pi * 123_456 * t)
    _, res = analysis.analyze({"x": x}, 1000.0, t=t)
    spurs = res["x"][0].quality.spurs
    assert spurs and spurs[0].freq == pytest.approx(123_456, rel=1e-3)
    assert spurs[0].dbc == pytest.approx(-40, abs=1)


def test_clip_low_snr_am_and_jitter_flags():
    t, x = _sine(1000.0)
    _, res = analysis.analyze({"x": np.clip(x, -0.8, 0.8)}, 1000.0, t=t)
    assert "clip" in res["x"][0].flags
    buried = x * 0.01 + np.random.default_rng(1).normal(0, 0.01, x.size)
    _, res = analysis.analyze({"x": buried}, 1000.0, t=t)
    q = res["x"][0].quality
    assert q.snr_db < 0 < q.estimate_snr_db and "low_snr" not in res["x"][0].flags
    assert abs(res["x"][0].phasor) == pytest.approx(0.01 / math.sqrt(2), rel=0.05)
    noisy = x * 0.01 + np.random.default_rng(1).normal(0, 0.3, x.size)
    _, res = analysis.analyze({"x": noisy}, 1000.0, t=t, refine=False)
    assert "low_snr" in res["x"][0].flags
    am = x * (1 + 0.2 * np.sin(2 * np.pi * 7 * t))
    _, res = analysis.analyze({"x": am}, 1000.0, t=t)
    assert "amplitude_varies" in res["x"][0].flags
    jit = np.cos(
        2 * np.pi * 1000 * t + np.repeat(np.random.default_rng(2).normal(0, 0.2, 8), t.size // 8)
    )
    _, res = analysis.analyze({"x": jit}, 1000.0, t=t, refine=False)
    assert "phase_jitter" in res["x"][0].flags


def test_stability_measures_offset_and_drift():
    fs, n = 1e6, 400_000
    t = np.arange(n) / fs
    f0, drift = 1000.0, 5.0
    x = np.cos(2 * np.pi * (f0 * t + 0.5 * drift * t**2))
    st = analysis.stability(x, 1 / fs, f0)
    assert st.drift_hz_per_s == pytest.approx(drift, rel=0.05)
    assert st.freq_offset_hz == pytest.approx(drift * t.mean(), rel=0.05)


# ---------- capture rules and plan ----------


def test_rule_forms():
    assert rule_value(3, 10.0) == 3
    assert rule_value(lambda f: f / 10, 100.0) == 10
    assert rule_value([(10, 1), (1000, 100)], 100.0) == pytest.approx(10)
    assert rule_value([(10, 1), (1000, 100)], 1e6) == pytest.approx(100)


def test_plan_low_and_high_frequency_tradeoffs():
    rules = CaptureRules(min_cycles=10, rbw_fraction=0.01, max_record_length=100_000)
    lo = plan_capture(10.0, rules)
    assert lo.cycles >= 10 and lo.rbw <= 0.1 + 1e-9
    hi = plan_capture(1e6, rules)
    assert hi.record_length <= 100_000
    assert hi.sample_rate >= 2.5 * 1e6 * 6
    tight = plan_capture(
        1.0, CaptureRules(min_cycles=50, max_record_length=1000, min_samples_per_cycle=100)
    )
    assert tight.record_length <= 1000


def test_plan_reports_limits():
    p = plan_capture(1e6, CaptureRules(min_cycles=1e6, max_record_length=10_000))
    assert "max_record_length" in p.limited_by
    p = plan_capture(1e6, CaptureRules(max_sample_rate=5e6, harmonics=0))
    assert p.sample_rate <= 5e6 and "max_sample_rate" in p.limited_by


def test_frequency_list_and_amplitudes():
    fr = frequencies(10, 1000, points_per_decade=10, extra=[60, 50, 5000])
    assert fr[0] == 10 and fr[-1] == 5000 and 60 in fr and fr == sorted(fr)
    assert len(frequencies(1, 1e3, points=4)) == 4
    assert amplitude_at([(10, 1), (1000, 0.01)], 100) == pytest.approx(0.1)
    fo = first_order(1.0, 100.0, rising=False, low=0.2)
    assert fo(1.0) == pytest.approx(1.0, rel=1e-3) and fo(1e6) == 0.2
    assert clamp(5.0, high=2.0)(10) == 2.0


# ---------- the sweep loop ----------


def _bench(**kw):
    return SimulatedBench(
        {"vin": lambda f: 1.0, "vout": rc_lowpass(1e3, 159.155e-9)},
        units={"vin": "V", "vout": "V"},
        **kw,
    )


def test_run_sweep_end_to_end(tmp_path):
    b = _bench(clock_ppm=30, noise_rms=1e-4)
    fr = frequencies(10, 1e5, points_per_decade=5)
    res = run_sweep(
        fr,
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions={"H": lambda v: v["vout"] / v["vin"]},
        capture=CaptureRules(max_record_length=50_000),
        folder=tmp_path,
        title="rc",
        quiet=True,
    )
    assert res.status == "complete" and res.points == len(fr)
    for name in (
        "meta.json",
        "points.csv",
        "plot.png",
        "plot.svg",
        "condensed.csv",
        "review.md",
        "spurs.csv",
    ):
        assert (res.path / name).exists()
    run = Run(res.path)
    h = run.series("H")
    assert h.phase_valid
    expect = 1 / (1 + 1j * h.freqs / 1000.0)
    assert np.allclose(h.values, expect, rtol=2e-3, atol=2e-4)
    assert b.output_on is False
    assert run.meta["status"] == "complete"
    assert "## Quantities" in review(run)
    assert "falls 3 dB" in review(run)


def test_dry_run_touches_nothing(tmp_path):
    b = _bench()
    res = run_sweep(
        frequencies(10, 1e3, points=5),
        b.source(),
        [b.receiver()],
        amplitude=0.5,
        folder=tmp_path,
        dry_run=True,
        quiet=True,
    )
    assert res.status == "planned" and b.log == []
    assert len(Run(res.path).meta["plan"]) == 5


def test_stop_if_and_output_off_on_error(tmp_path):
    b = _bench()
    res = run_sweep(
        frequencies(10, 1e4, points=8),
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        folder=tmp_path,
        quiet=True,
        plot=False,
        stop_if=lambda v: "too low" if abs(v["vout"]) < 0.5 else None,
    )
    assert res.status == "stopped" and res.points < 8 and not b.output_on

    def broken(freq, plan):
        raise RuntimeError("receiver fell over")

    res = run_sweep(
        [100.0, 200.0], b.source(), [broken], amplitude=1.0, folder=tmp_path, quiet=True
    )
    assert res.status == "error" and not b.output_on
    assert "receiver fell over" in Run(res.path).meta["reason"]

    def stopper(freq, plan):
        raise StopSweep("limit reached")

    res = run_sweep([100.0], b.source(), [stopper], amplitude=1.0, folder=tmp_path, quiet=True)
    assert res.status == "stopped"


def test_plain_function_glue_and_scalar_readings(tmp_path):
    state = {}

    def my_source(freq, amplitude):
        state["f"] = freq
        return {"freq": freq}

    def my_meter(freq, plan):
        return {"gain": 2.0 / (1 + freq / 100.0)}

    def my_phasor_meter(freq, plan):
        return Readings({"a": 1 + 0j, "b": 0.5j}, units={"a": "V", "b": "V"}, group="lockin")

    res = run_sweep(
        frequencies(10, 1000, points=6),
        my_source,
        [my_meter, my_phasor_meter],
        expressions={"ba": (lambda v: v["b"] / v["a"], "")},
        folder=tmp_path,
        quiet=True,
    )
    run = Run(res.path)
    g = run.series("gain")
    assert g.kind == "scalar" and not g.phase_valid
    ba = run.series("ba")
    assert ba.phase_valid and ba.phase_deg[0] == pytest.approx(90)


def test_phase_suppressed_across_groups(tmp_path):
    b = _bench()

    def meter(freq, plan):
        return {"dmm": 0.7}

    res = run_sweep(
        [100.0, 1000.0],
        b.source(),
        [b.receiver(), meter],
        amplitude=1.0,
        expressions={"x": lambda v: v["dmm"] / v["vin"]},
        folder=tmp_path,
        quiet=True,
    )
    assert not Run(res.path).series("x").phase_valid


def test_waveforms_saved_and_read_back(tmp_path):
    b = _bench()
    res = run_sweep(
        [1000.0],
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        folder=tmp_path,
        save_waveforms=True,
        quiet=True,
    )
    w = Run(res.path).waveform(0)
    assert set(w["signals"]) == {"vin", "vout"} and len(w["t"]) == len(w["signals"]["vin"])


def test_capture_from_user_function_with_dt(tmp_path):
    def scope(freq, plan: CapturePlan):
        n = plan.record_length
        t = np.arange(n) / plan.sample_rate
        return Capture({"y": 0.3 * np.cos(2 * np.pi * freq * t)}, dt=1 / plan.sample_rate)

    res = run_sweep([500.0], lambda f, a: None, [scope], folder=tmp_path, quiet=True)
    assert Run(res.path).series("y").mag[0] == pytest.approx(0.3 / math.sqrt(2), rel=1e-3)


def test_condensed_view_size_and_extremes(tmp_path):
    b = SimulatedBench({"in": lambda f: 1.0, "out": resonance(3000.0, 8.0)}, noise_rms=1e-5)
    fr = frequencies(100, 1e5, points=300)
    res = run_sweep(
        fr,
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions={"H": lambda v: v["out"] / v["in"]},
        capture=CaptureRules(max_record_length=20_000, harmonics=0),
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    header, rows = condensed(res.path, 100)
    assert 95 <= len(rows) <= 110
    run = Run(res.path)
    peak_f = run.series("H").freqs[int(np.argmax(run.series("H").mag))]
    assert any(float(r[0]) == pytest.approx(peak_f, rel=1e-6) for r in rows)
    assert "peak" in review(run)


def test_delay_shows_as_phase_slope(tmp_path):
    b = SimulatedBench({"a": lambda f: 1.0, "b": delay(1e-6)}, noise_rms=1e-5)
    res = run_sweep(
        frequencies(1e3, 1e5, points=10),
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions={"H": lambda v: v["b"] / v["a"]},
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    ph = Run(res.path).series("H").phase_deg
    f = Run(res.path).series("H").freqs
    assert np.allclose(ph, -360 * f * 1e-6, atol=0.05)


# ---------- read-back validation ----------


def test_validate_capture_cases():
    from sweep.capture import validate_capture

    plan = plan_capture(1000.0)
    ok = {
        "sample_rate": plan.sample_rate,
        "record_length": plan.record_length,
        "time_scale": plan.duration / 10,
        "div_count": 10,
        "horizontal_mode": "MANUAL",
        "points_transferred": plan.record_length,
    }
    assert validate_capture(plan, ok, harmonics=5) == ([], False)
    flags, bad = validate_capture(plan, {**ok, "horizontal_mode": "AUTO"})
    assert bad and "horizontal_not_manual" in flags
    flags, bad = validate_capture(plan, {**ok, "time_scale": ok["time_scale"] * 2})
    assert bad and "horizontal_inconsistent" in flags
    flags, bad = validate_capture(plan, {**ok, "sample_rate": 1500.0, "time_scale": None})
    assert bad and "sample_rate_invalid" in flags
    flags, bad = validate_capture(plan, {"record_length": 100})
    assert bad and "readback_missing_sample_rate" in flags
    flags, bad = validate_capture(plan, {**ok, "points_transferred": 10})
    assert bad and "transfer_short" in flags


def test_invalid_capture_is_retried_then_flagged_or_stops(tmp_path):
    calls = {"n": 0}

    def scope(freq, plan):
        calls["n"] += 1
        n = plan.record_length
        t = np.arange(n) / plan.sample_rate
        return Capture(
            {"y": np.cos(2 * np.pi * freq * t)},
            t=t,
            settings={
                "sample_rate": plan.sample_rate,
                "record_length": n,
                "horizontal_mode": "AUTO",
            },
        )

    res = run_sweep([1000.0], lambda f, a: {"freq": f}, [scope], folder=tmp_path, quiet=True)
    row = next(r for r in Run(res.path).rows if r["kind"] == "signal")
    assert calls["n"] == 2 and "capture_invalid" in row["flags"] and "retried" in row["flags"]
    res = run_sweep(
        [1000.0], lambda f, a: None, [scope], folder=tmp_path, quiet=True, on_invalid="stop"
    )
    assert res.status == "stopped"


def test_source_frequency_sanity_flags(tmp_path):
    def scope(freq, plan):
        t = np.arange(plan.record_length) / plan.sample_rate
        return Capture({"y": np.cos(2 * np.pi * freq * t)}, t=t)

    res = run_sweep([1000.0], lambda f, a: {"freq": f * 1.01}, [scope], folder=tmp_path, quiet=True)
    row = next(r for r in Run(res.path).rows if r["kind"] == "signal")
    assert "source_freq_wrong" in row["flags"]


def test_drive_off_noise_floor_and_auto_harmonic_plot(tmp_path):
    b = _bench(noise_rms=1e-3)
    res = run_sweep(
        [1000.0, 2000.0],
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        folder=tmp_path,
        quiet=True,
        drive_off=True,
    )
    assert ("output", True) not in b.log
    assert Run(res.path).series("vin").mag.max() < 1e-2
    nb = SimulatedBench({"x": lambda f: 1.0}, harmonic_levels={3: 0.2}, noise_rms=1e-5)
    res = run_sweep(
        [1000.0, 2000.0], nb.source(), [nb.receiver()], amplitude=1.0, folder=tmp_path, quiet=True
    )
    assert (res.path / "harmonics_x.png").exists()


def test_output_left_off_even_when_restore_turns_it_back_on(tmp_path):
    events = []

    class Src:
        name = "src"

        def set(self, f, a):
            return {"freq": f}

        def output(self, state):
            events.append(("output", state))

        def close(self):
            events.append(("restore_output_on",))

    def meter(f, plan):
        return {"x": 1.0}

    run_sweep([100.0], Src(), [meter], folder=tmp_path, quiet=True)
    assert events[-1] == ("output", False)


def test_failed_open_still_closes_and_turns_off(tmp_path):
    events = []

    class Rx:
        name = "rx"

        def open(self):
            raise OSError("link lost during open")

        def close(self):
            events.append("rx_closed")

        def measure(self, f, plan):
            return {"x": 1.0}

    b = _bench()
    with pytest.raises(OSError):
        run_sweep([100.0], b.source(), [Rx()], folder=tmp_path, quiet=True)
    assert events == ["rx_closed"] and not b.output_on


def test_amplitude_limit_rejects_bad_values():
    from sweep.instruments.base import check_amplitude

    for bad in (float("nan"), -0.1, 0.6):
        with pytest.raises(ValueError):
            check_amplitude(bad, 0.5, "src")
    with pytest.raises(ValueError):
        check_amplitude(0.1, float("inf"), "src")


def test_repeats_average_and_flag_unstable(tmp_path):
    rng = np.random.default_rng(5)
    state = {"drift": 0.0}

    def scope(freq, plan):
        n = plan.record_length
        t = np.arange(n) / plan.sample_rate
        start = rng.uniform(0, 2 * np.pi)
        g = 1.0 + state["drift"] * rng.normal()
        return Capture(
            {
                "in": np.cos(2 * np.pi * freq * t + start),
                "out": 0.5 * g * np.cos(2 * np.pi * freq * t + start - 0.4),
            },
            t=t,
        )

    kw = dict(
        expressions={"H": lambda v: v["out"] / v["in"]},
        folder=tmp_path,
        quiet=True,
        plot=False,
        repeats=4,
    )
    run = Run(run_sweep([1000.0], lambda f, a: None, [scope], **kw).path)
    row = next(r for r in run.rows if r["signal"] == "out" and r["kind"] == "signal")
    assert row["repeats"] == 4 and row["repeat_phase_spread_deg"] < 0.01
    h = run.series("H").values[0]
    assert abs(h) == pytest.approx(0.5, rel=1e-4) and np.angle(h) == pytest.approx(-0.4, abs=1e-4)
    assert "unstable" not in row["flags"]
    state["drift"] = 0.05
    run = Run(run_sweep([1000.0], lambda f, a: None, [scope], **kw).path)
    row = next(r for r in run.rows if r["signal"] == "out" and r["kind"] == "signal")
    assert "unstable" in row["flags"] and row["repeat_mag_spread_pct"] > 1


def test_adaptive_refinement_finds_a_sharp_resonance(tmp_path):
    b = SimulatedBench({"in": lambda f: 1.0, "out": resonance(3000.0, 30.0)}, noise_rms=1e-6)
    res = run_sweep(
        frequencies(100, 1e5, points_per_decade=4),
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions={"H": lambda v: v["out"] / v["in"]},
        capture=CaptureRules(harmonics=0, max_record_length=20_000),
        refine_on="H",
        refine_rounds=4,
        refine_max_points=30,
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    h = Run(res.path).series("H")
    assert len(h.freqs) > 13 and np.all(np.diff(h.freqs) > 0)
    assert h.mag.max() > 20  # Q = 30 peak found, coarse plan alone tops out far lower


def test_review_has_group_delay_fit_and_wide_table(tmp_path):
    b = SimulatedBench(
        {"a": lambda f: 1.0, "b": delay(2e-6, rc_lowpass(1e3, 159.155e-9))}, noise_rms=1e-6
    )
    res = run_sweep(
        frequencies(100, 1e5, points_per_decade=6),
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions={"H": lambda v: v["b"] / v["a"]},
        capture=CaptureRules(harmonics=0, max_record_length=50_000),
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    md = (res.path / "review.md").read_text()
    assert "group delay" in md and "## Fits" in md and "1 kHz" in md
    from sweep.views import group_delay, wide_table

    f, gd = group_delay(Run(res.path).series("H"))
    assert gd[0] == pytest.approx(2e-6 + 159.155e-9 * 1e3, rel=0.05)
    header, rows = wide_table(res.path)
    assert header[:3] == ["point", "freq_cmd_hz", "freq_used_hz"] and len(rows) == 19
    assert (res.path / "wide.csv").exists()


def test_settle_captures_wait_for_a_device_to_settle(tmp_path):
    state = {"n": 0}

    def scope(freq, plan):
        state["n"] += 1
        g = 0.5 * (1 + 0.5 * 0.3 ** state["n"])  # decays toward 0.5 capture by capture
        t = np.arange(plan.record_length) / plan.sample_rate
        return Capture(
            {"in": np.cos(2 * np.pi * freq * t), "out": g * np.cos(2 * np.pi * freq * t)}, t=t
        )

    res = run_sweep(
        [1000.0],
        lambda f, a: None,
        [scope],
        folder=tmp_path,
        quiet=True,
        plot=False,
        settle_captures=8,
        expressions={"H": lambda v: v["out"] / v["in"]},
    )
    run = Run(res.path)
    assert abs(run.series("H").mag[0] - 0.5) < 0.002
    row = next(r for r in run.rows if r["kind"] == "signal")
    assert '"settle_captures"' in row["settings"] and "not_settled" not in row["flags"]


def test_level_sweep_shows_distortion_rising_with_drive(tmp_path):
    from sweep.run import run_level_sweep

    state = {"a": 0.0}

    def src(freq, amplitude):
        state["a"] = amplitude
        return {"freq": freq, "amplitude": amplitude}

    def scope(freq, plan):
        t = np.arange(plan.record_length) / plan.sample_rate
        x = state["a"] / 2 * np.cos(2 * np.pi * freq * t)
        return Capture({"in": x, "out": np.tanh(x)}, t=t)

    res = run_level_sweep(
        1000.0,
        [0.1, 0.5, 1.0, 2.0, 4.0],
        src,
        [scope],
        expressions={"gain": lambda v: v["out"] / v["in"]},
        folder=tmp_path,
        quiet=True,
    )
    run = Run(res.path)
    thd = run.series("out").thd_pct
    assert np.all(np.diff(thd) > 0) and thd[0] < 0.1 < 5 < thd[-1]
    g = run.series("gain").mag
    assert g[0] == pytest.approx(1.0, abs=0.01) and g[-1] < 0.8
    assert "Against drive level" in (res.path / "review.md").read_text()
    assert (res.path / "plot.png").exists()


def test_expression_carries_uncertainty_from_its_inputs(tmp_path):
    b = _bench(noise_rms={"vin": 1e-3, "vout": 1e-2})
    res = run_sweep(
        [100.0, 1000.0],
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions={"H": lambda v: v["vout"] / v["vin"]},
        folder=tmp_path,
        quiet=True,
    )
    run = Run(res.path)
    h = run.series("H")
    out = run.series("vout")
    assert np.all(np.isfinite(h.phase_uncertainty_deg))
    assert np.all(h.estimate_snr_db <= out.estimate_snr_db + 1e-6)


def test_tolerance_finds_an_offset_tone_and_flags_the_search_edge():
    t, x = _sine(1050.0, n=100_000)  # rbw 10 Hz: 5 bins off the expected 1000 Hz
    f_used, res = analysis.analyze({"x": x}, 1000.0, t=t)
    r = res["x"][0]
    assert "strongest_elsewhere" in r.flags and r.strongest_freq == pytest.approx(1050, rel=1e-4)
    f_used, res = analysis.analyze({"x": x}, 1000.0, t=t, tolerance=0.1)
    r = res["x"][0]
    assert f_used == pytest.approx(1050.0, abs=0.01) and abs(r.phasor) == pytest.approx(
        1 / math.sqrt(2), rel=1e-3
    )
    assert r.offset_bins == pytest.approx(5.0, abs=0.05) and "peak_off_target" in r.flags
    assert "peak_at_search_edge" not in r.flags


def test_strongest_tone_elsewhere_is_flagged():
    t, x = _sine(1000.0, amp=0.1)
    x = x + np.cos(2 * np.pi * 37_000 * t)
    _, res = analysis.analyze({"x": x}, 1000.0, t=t)
    r = res["x"][0]
    assert "strongest_elsewhere" in r.flags and r.strongest_freq == pytest.approx(37_000, rel=1e-3)
    assert r.strongest_dbc == pytest.approx(20, abs=0.5)


def test_spectra_saved_and_waterfall_drawn(tmp_path):
    b = SimulatedBench({"x": lambda f: 1.0}, spur=(25_000.0, 0.01), noise_rms=1e-5)
    res = run_sweep(
        frequencies(200, 2000, points=6),
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        folder=tmp_path,
        quiet=True,
        capture=CaptureRules(max_record_length=20_000),
    )
    rows = Run(res.path).spectra("x")
    assert len(rows) == 6 and rows[0][1].size > 50
    assert (res.path / "waterfall_x.png").exists()


def test_impedance_plot_parts_for_a_series_rlc(tmp_path):
    from sweep.plot import impedance_parts, plot_impedance

    r, ind, cap = 5.0, 1e-6, 1e-6
    zf = lambda f: r + 2j * np.pi * f * ind + 1 / (2j * np.pi * f * cap)  # noqa: E731
    b = SimulatedBench({"v": zf, "i": lambda f: 1.0}, noise_rms=1e-7)
    res = run_sweep(
        frequencies(1e3, 1e6, points=13),
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions={"Z": (lambda v: v["v"] / v["i"], "ohm")},
        folder=tmp_path,
        quiet=True,
        capture=CaptureRules(harmonics=0, max_record_length=50_000),
    )
    s = Run(res.path).series("Z")
    p = impedance_parts(s)
    lo, hi = s.freqs < 3e4, s.freqs > 3e5
    assert np.allclose(p["R"], 5.0, rtol=0.02)
    assert np.allclose(p["C"][lo][np.isfinite(p["C"][lo])], cap, rtol=0.05)
    assert np.nanmedian(p["L"][hi]) == pytest.approx(
        ind * (1 - 1 / ((2 * np.pi * np.median(s.freqs[hi])) ** 2 * ind * cap)), rel=0.2
    )
    assert (res.path / "impedance_Z.png").exists()
    plot_impedance(res.path)


def test_waveforms_saved_as_instrument_codes_round_trip_exactly(tmp_path):
    def scope(freq, plan):
        n = plan.record_length
        codes = np.round(30000 * np.cos(2 * np.pi * freq * np.arange(n) / plan.sample_rate))
        codes = codes.astype(np.int16)
        ymult, yoff, yzero = 1e-5, 2.0, 0.01
        x = yzero + ymult * (codes.astype(float) - yoff)
        t = 1e-3 + np.arange(n) / plan.sample_rate
        raw = {
            "y": {"codes": codes, "ymult": ymult, "yoff": yoff, "yzero": yzero},
            "_time": {"xzero": 1e-3, "xincr": 1 / plan.sample_rate, "pt_off": 0.0},
        }
        return Capture({"y": x}, t=t, raw=raw)

    res = run_sweep([1000.0], lambda f, a: None, [scope], folder=tmp_path, quiet=True, plot=False)
    w = Run(res.path).waveform(0)
    assert w["format"] == "codes" and w["codes"]["y"].dtype == np.int16
    assert np.max(np.abs(w["codes"]["y"])) == 30000
    t = w["t"]
    assert t[0] == pytest.approx(1e-3)
    f = next(res.path.glob("waveforms/*.npz"))
    assert f.stat().st_size < 2 * len(w["t"]) * 2  # int16, zipped, well under float32


@pytest.mark.parametrize("offset_bins", [0.0, 0.25, 0.5, 0.37])
def test_exact_frequency_dft_between_fft_bins(offset_bins):
    # 100 000 samples at 1 MS/s: FFT bins every 10 Hz. The tone sits between bins.
    f = 1000.0 + offset_bins * 10.0
    t, x = _sine(f, amp=1.0, ph=0.3)
    _, b = _sine(f, amp=0.5, ph=-0.2)
    _, res = analysis.analyze({"a": x, "b": b}, f, t=t, refine=False)
    p = res["a"][0].phasor
    assert abs(p) == pytest.approx(1 / math.sqrt(2), rel=1e-5)
    assert np.angle(p) == pytest.approx(0.3, abs=1e-5)
    h = res["b"][0].phasor / p
    assert abs(h) == pytest.approx(0.5, rel=1e-6) and np.angle(h) == pytest.approx(-0.5, abs=1e-6)
    # Unlocked: the search lands on the true frequency, off the bin grid.
    f_used, _ = analysis.analyze({"a": x}, 1000.0 + 0.1, t=t)
    assert f_used == pytest.approx(f, abs=1e-3)
