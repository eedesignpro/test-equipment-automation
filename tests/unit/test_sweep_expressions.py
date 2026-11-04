"""Ready-made shunt expressions and noise through differences."""

import math

import numpy as np

from sweep.capture import CaptureRules
from sweep.expressions import high_side_shunt, low_side_shunt
from sweep.plan import frequencies
from sweep.run import run_sweep
from sweep.simulate import SimulatedBench
from sweep.store import Run

R_SHUNT = 1.0
ZDUT = lambda f: 10.0 + 2j * math.pi * f * 50e-6  # noqa: E731


def _rules():
    return CaptureRules(harmonics=0, max_record_length=50_000)


def test_low_side_shunt_gives_the_dut_impedance(tmp_path):
    # Drive 1 V at the top; the DUT and the shunt divide it.
    total = lambda f: 1.0  # noqa: E731
    shunt = lambda f: R_SHUNT / (ZDUT(f) + R_SHUNT)  # noqa: E731
    b = SimulatedBench({"V_total": total, "V_shunt": shunt}, noise_rms=1e-6)
    res = run_sweep(
        frequencies(100, 1e5, points=7),
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions=low_side_shunt("V_total", "V_shunt", R_SHUNT),
        capture=_rules(),
        folder=tmp_path,
        quiet=True,
    )
    z = Run(res.path).series("Z")
    assert z.phase_valid and np.allclose(z.values, [ZDUT(f) for f in z.freqs], rtol=2e-3)
    assert (res.path / "impedance_Z.png").exists()


def test_high_side_shunt_gives_the_dut_impedance(tmp_path):
    src = lambda f: 1.0  # noqa: E731
    dut = lambda f: ZDUT(f) / (ZDUT(f) + R_SHUNT)  # noqa: E731
    b = SimulatedBench({"V_src": src, "V_dut": dut}, noise_rms=1e-6)
    res = run_sweep(
        frequencies(100, 1e5, points=7),
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions=high_side_shunt("V_src", "V_dut", R_SHUNT),
        capture=_rules(),
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    z = Run(res.path).series("Z")
    assert np.allclose(z.values, [ZDUT(f) for f in z.freqs], rtol=2e-3)


def test_noise_of_a_small_difference_is_not_understated(tmp_path):
    # V_dut is 1 % of V_total, so its noise is about 100x the channels' relative noise.
    b = SimulatedBench({"a": lambda f: 1.0, "b": lambda f: 0.99}, noise_rms=1e-4)
    res = run_sweep(
        [1000.0],
        b.source(),
        [b.receiver()],
        amplitude=1.0,
        expressions={"d": (lambda v: v["a"] - v["b"], "V")},
        capture=_rules(),
        folder=tmp_path,
        quiet=True,
        plot=False,
    )
    run = Run(res.path)
    d = run.series("d")
    a = run.series("a")
    assert d.estimate_snr_db[0] < a.estimate_snr_db[0] - 30
