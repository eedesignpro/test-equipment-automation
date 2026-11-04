"""The other estimators and searches, and the blind spot of each.

Each blind-spot test states what the method misses, so a change that hides
or removes the limit shows up here and in the guide's table.
"""

import math

import numpy as np
import pytest

from sweep import analysis, estimators
from sweep.capture import CaptureRules
from sweep.multitone import run_multitone_sweep
from sweep.plan import frequencies
from sweep.run import run_sweep
from sweep.simulate import SimulatedBench, rc_lowpass
from sweep.store import Run

FS = 1e6
N = 10_000  # bin = 100 Hz


def _record(parts, noise=0.0, seed=3):
    t = np.arange(N) / FS
    x = np.zeros(N)
    for f, a, ph in parts:
        x += a * math.sqrt(2) * np.cos(2 * np.pi * f * t + ph)
    if noise:
        x += np.random.default_rng(seed).normal(0, noise, N)
    return t, x


def _read(x, t, f, **kw):
    kw.setdefault("refine", False)
    kw.setdefault("harmonics", 0)
    _, res = analysis.analyze({"x": x}, f, t=t, **kw)
    return res["x"][0]


@pytest.mark.parametrize("est", analysis.ESTIMATORS)
def test_every_estimator_reads_a_clean_on_bin_tone(est):
    t, x = _record([(5_000.0, 0.5, 0.3)])
    r = _read(x, t, 5_000.0, estimator=est, window="rect" if est == "coherent" else "blackman")
    assert abs(r.phasor) == pytest.approx(0.5, rel=1e-6)
    assert np.angle(r.phasor) == pytest.approx(0.3, abs=1e-6)
    assert "not_coherent" not in r.flags and "leakage" not in r.flags


def test_coherent_flags_an_off_bin_record_instead_of_reading_it():
    # Blind spot: one bin cannot read a tone between bins. The flags are the only warning.
    t, x = _record([(5_037.0, 0.5, 0.3)])
    r = _read(x, t, 5_037.0, estimator="coherent", window="rect")
    assert abs(r.phasor) < 0.45
    assert "not_coherent" in r.flags and "leakage" in r.flags


def test_single_tone_fit_is_biased_by_an_unlisted_tone_and_multifit_is_not():
    # Blind spot of "fit": a strong tone left out of the fit leaks into the value
    # when the record is not a whole number of periods of both.
    f1 = 5_037.0
    t, x = _record([(f1, 0.5, 0.0), (2 * f1, 0.02, 0.4)])
    lone = estimators.sine_fit(x, 0.0, 1 / FS, [2 * f1])["phasors"][2 * f1]
    joint = estimators.sine_fit(x, 0.0, 1 / FS, [f1, 2 * f1])["phasors"][2 * f1]
    true = 0.02 * np.exp(0.4j)
    assert abs(lone - true) > 0.05 * 0.02
    assert abs(joint) == pytest.approx(0.02, rel=1e-6)
    assert np.angle(joint) == pytest.approx(0.4, abs=1e-5)


def test_multifit_resolves_tones_inside_one_main_lobe_only_when_listed():
    # Two tones 1.5 bins apart sit inside one Blackman main lobe.
    t, x = _record([(5_000.0, 0.5, 0.0), (5_150.0, 0.2, 1.0)])
    dft = _read(x, t, 5_000.0)
    assert abs(abs(dft.phasor) - 0.5) > 0.01
    listed = _read(x, t, 5_000.0, estimator="multifit", tones=[5_150.0])
    assert abs(listed.phasor) == pytest.approx(0.5, rel=1e-6)
    # Blind spot: unlisted, the joint fit is a single-tone fit and is biased too.
    unlisted = _read(x, t, 5_000.0, estimator="multifit")
    assert abs(abs(unlisted.phasor) - 0.5) > 0.01


def test_fit4_finds_the_received_frequency_of_an_unlocked_tone():
    t, x = _record([(5_031.7, 0.5, 0.2)], noise=1e-4)
    r = _read(x, t, 5_000.0, estimator="fit4", refine=True, tolerance_hz=200.0)
    assert r.freq == pytest.approx(5_031.7, abs=0.01)
    assert abs(r.phasor) == pytest.approx(0.5, rel=1e-4)


def test_czt_search_finds_a_tone_far_inside_a_wide_tolerance():
    t, x = _record([(12_340.0, 0.5, 0.0)], noise=1e-4)
    r = _read(x, t, 10_000.0, search="czt", refine=True, tolerance_hz=5_000.0)
    assert r.freq == pytest.approx(12_340.0, abs=0.5)
    assert abs(r.phasor) == pytest.approx(0.5, rel=1e-3)


def test_any_search_takes_the_strongest_peak_in_its_window():
    # Blind spot of every search: a stronger neighbour inside the tolerance wins.
    t, x = _record([(10_000.0, 0.1, 0.0), (11_000.0, 0.5, 0.0)], noise=1e-5)
    for search in analysis.SEARCHES:
        r = _read(x, t, 10_000.0, search=search, refine=True, tolerance_hz=2_000.0)
        assert r.freq == pytest.approx(11_000.0, abs=1.0)
        assert "peak_off_target" in r.flags


def test_unknown_estimator_or_search_is_refused():
    t, x = _record([(5_000.0, 0.5, 0.0)])
    with pytest.raises(ValueError):
        _read(x, t, 5_000.0, estimator="magic")
    with pytest.raises(ValueError):
        _read(x, t, 5_000.0, search="magic")


@pytest.mark.parametrize("name", ["blackmanharris", "kaiser"])
def test_new_windows_read_a_tone(name):
    t, x = _record([(5_037.0, 0.5, 0.3)])
    r = _read(x, t, 5_037.0, window=name)
    assert abs(r.phasor) == pytest.approx(0.5, rel=1e-4)


# ---------- coherent capture in the sweep loop ----------


def _bench():
    return SimulatedBench(
        {"vin": lambda f: 1.0, "vout": rc_lowpass(1e3, 159.155e-9)},
        noise_rms=1e-5,
        sample_clock_hz=6.25e9,
    )


def test_coherent_sweep_moves_each_drive_onto_a_bin(tmp_path):
    b = _bench()
    fr = frequencies(100, 1e5, points_per_decade=4)
    res = run_sweep(
        fr,
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        coherent=True,
        harmonics=2,
        expressions={"H": lambda v: v["vout"] / v["vin"]},
        capture=CaptureRules(max_record_length=50_000),
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    run = Run(res.path)
    assert run.meta["estimator"] == "coherent" and run.meta["coherent"] is True
    rows = [r for r in run.rows if r["kind"] == "signal" and r["order"] == 1]
    assert rows
    for r in rows:
        assert "not_coherent" not in r["flags"] and "leakage" not in r["flags"]
        assert abs(r["freq_cmd"] - r["freq_planned"]) <= 0.01 * r["freq_planned"]
    h = run.series("H")
    assert np.allclose(h.values, 1 / (1 + 1j * h.freqs / 1000.0), rtol=1e-3, atol=1e-4)


def test_coherent_multitone_puts_every_tone_on_a_bin(tmp_path):
    b = _bench()
    fr = frequencies(1e3, 1e5, points_per_decade=16)
    res = run_multitone_sweep(
        fr,
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        coherent=True,
        expressions={"H": lambda v: v["vout"] / v["vin"]},
        capture=CaptureRules(max_record_length=200_000),
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    run = Run(res.path)
    rows = [r for r in run.rows if r["kind"] == "signal" and r["order"] == 1]
    assert len({r["freq_cmd"] for r in rows}) == len(fr)
    for r in rows:
        assert "not_coherent" not in r["flags"], r["freq_cmd"]
    h = run.series("H")
    assert np.allclose(h.values, 1 / (1 + 1j * h.freqs / 1000.0), rtol=2e-3, atol=2e-4)


def test_zoom_spectrum_peaks_at_the_tone_with_the_rms_scale():
    t, x = _record([(5_037.0, 0.5, 0.0)])
    f, a = estimators.zoom_spectrum(x, 1 / FS, 5_000.0, 200.0, points=801)
    assert f[np.argmax(a)] == pytest.approx(5_037.0, abs=0.5)
    assert a.max() == pytest.approx(0.5, rel=1e-3)


def test_noise_alone_does_not_set_the_leakage_flag():
    # Measured on the bench: a 36 dB SNR tone has noise bins near -40 dB of it.
    t, x = _record([(5_000.0, 0.5, 0.0)], noise=0.5 * math.sqrt(N / 2) * 10 ** (-36 / 20))
    r = _read(x, t, 5_000.0, estimator="coherent", window="rect")
    assert r.extra["leak_db"] > -45
    assert "leakage" not in r.flags


def test_neighbour_tones_do_not_read_as_drift_or_jitter():
    # Multitone: 1 % spacing is inside a segment's main lobe; the segments are cut down.
    tones = [5_000.0 * 1.01**k for k in range(8)]
    t, x = _record([(f, 0.1, 0.3 * k) for k, f in enumerate(tones)], noise=1e-5)
    r = _read(x, t, tones[3], tones=[f for f in tones if f != tones[3]])
    assert "amplitude_varies" not in r.flags and "phase_jitter" not in r.flags
