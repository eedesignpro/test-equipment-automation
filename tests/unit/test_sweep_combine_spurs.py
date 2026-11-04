"""Math across runs and the recurring spur report."""

import numpy as np

from sweep.capture import CaptureRules
from sweep.combine import combine_runs
from sweep.interfaces import Capture
from sweep.plan import frequencies
from sweep.run import run_sweep
from sweep.simulate import SimulatedBench, rc_lowpass
from sweep.spurs import spur_report

RC = rc_lowpass(1e3, 159.155e-9)


def test_combine_runs_coupled_minus_alone(tmp_path):
    a = SimulatedBench({"in": lambda f: 1.0, "out": RC}, noise_rms=1e-6)
    b = SimulatedBench({"in": lambda f: 1.0, "out": lambda f: 0.5 * RC(f)}, noise_rms=1e-6)
    kw = dict(
        amplitude=1.0,
        expressions={"H": lambda v: v["out"] / v["in"]},
        folder=tmp_path,
        quiet=True,
        plot=False,
        capture=CaptureRules(harmonics=0, max_record_length=20_000),
    )
    ra = run_sweep(frequencies(100, 1e4, points=41), a.source(), [a.receiver()], **kw)
    rb = run_sweep(frequencies(150, 2e4, points=45), b.source(), [b.receiver()], **kw)
    c = combine_runs(
        {"full": ra.path, "half": rb.path},
        {"diff": lambda v: v["full.H"] - v["half.H"]},
        folder=tmp_path,
        title="diff",
        points=10,
    )
    d = c.series("diff")
    expect = 0.5 / (1 + 1j * d.freqs / 1000.0)
    assert d.freqs.min() >= 150 * (1 - 1e-6) and d.freqs.max() <= 1e4 * (1 + 1e-6)
    assert np.allclose(d.values, expect, rtol=0.02, atol=2e-3) and d.phase_valid
    assert (c.path / "review.md").exists() and (c.path / "plot.png").exists()


def test_spur_report_names_an_alias(tmp_path):
    class Rx:
        name = "rx"

        def measure(self, freq, plan):
            t = np.arange(plan.record_length) / plan.sample_rate
            x = np.cos(2 * np.pi * freq * t) + 0.01 * np.cos(2 * np.pi * 2e6 * t)
            return Capture({"y": x}, t=t)

    rules = CaptureRules(
        harmonics=0, min_samples_per_cycle=50, max_sample_rate=1.5e6, max_record_length=20_000
    )
    res = run_sweep(
        [10e3, 12e3],
        lambda f, a: None,
        [Rx()],
        capture=rules,
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    md = spur_report(res.path, known={"2 MHz switcher": 2e6})
    assert "2 MHz switcher alias" in md
