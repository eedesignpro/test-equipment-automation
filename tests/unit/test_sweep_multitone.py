"""Several tones at once."""

import numpy as np
import pytest

from sweep.capture import CaptureRules
from sweep.multitone import make_groups, multitone_samples, run_multitone_sweep
from sweep.plan import frequencies
from sweep.run import run_sweep
from sweep.simulate import SimulatedBench, rc_lowpass
from sweep.store import Run


def test_groups_snap_to_one_base_within_a_quarter_percent():
    fr = frequencies(100, 1e5, points_per_decade=8)
    groups = make_groups(fr, 8)
    for base, tones in groups:
        for f in tones:
            assert abs(f / base - round(f / base)) < 1e-9
    snapped = [f for _, t in groups for f in t]
    assert len(snapped) == len(fr)
    assert max(abs(a / b - 1) for a, b in zip(snapped, fr)) < 2.5e-3


def test_multitone_samples_keep_a_low_crest_factor():
    x = multitone_samples(list(range(10, 90, 10)), 4096)
    crest = 1 / np.sqrt(np.mean(x**2))
    assert np.max(np.abs(x)) == pytest.approx(1.0) and crest < 2.5


def test_multitone_sweep_matches_a_tone_by_tone_sweep(tmp_path):
    def bench():
        return SimulatedBench(
            {"in": lambda f: 1.0, "out": rc_lowpass(1e3, 159.155e-9)}, noise_rms=1e-5
        )

    fr = frequencies(100, 1e4, points_per_decade=8)
    rules = CaptureRules(max_record_length=200_000)
    ex = {"H": lambda v: v["out"] / v["in"]}
    b1 = bench()
    multi = run_multitone_sweep(
        fr,
        b1.source(),
        [b1.receiver()],
        amplitude=1.0,
        expressions=ex,
        capture=rules,
        folder=tmp_path,
        quiet=True,
    )
    assert multi.status == "complete"
    h = Run(multi.path).series("H")
    expect = 1 / (1 + 1j * h.freqs / 1000.0)
    assert np.allclose(h.values, expect, rtol=5e-3, atol=1e-3)
    captures = sum(1 for e in b1.log if e[0] == "multitone")
    assert captures == len(make_groups(fr, 8)) < len(fr)
    files = list((multi.path / "waveforms").glob("*.npz"))
    assert len(files) == captures  # one file per capture, not per tone
    assert len(list((multi.path / "spectra").glob("*.npz"))) == captures
    w = Run(multi.path).waveform(3)  # a later tone of the first group finds the shared file
    assert w["freq"] == pytest.approx(h.freqs[0], rel=1e-6)
    b2 = bench()
    single = run_sweep(
        fr,
        b2.source(),
        [b2.receiver()],
        amplitude=1.0,
        expressions=ex,
        capture=rules,
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    h2 = Run(single.path).series("H")
    assert np.allclose(np.abs(h.values), np.abs(1 / (1 + 1j * h2.freqs / 1000.0)), rtol=0.01)
