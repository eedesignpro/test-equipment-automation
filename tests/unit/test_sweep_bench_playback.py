"""Play back recorded 5 ohm bench captures through the sweep analysis."""

from pathlib import Path

import numpy as np
import pytest

from sweep import analysis
from sweep.fit import fit_model
from sweep.interfaces import Capture
from sweep.run import run_sweep
from sweep.store import Run

DATA = Path(__file__).resolve().parents[1] / "captures" / "sweep_bench_5ohm"
POINTS = sorted(DATA.glob("point_*.npz"))


def _load(p):
    z = np.load(p)
    sig = {"V_R": z["V_R"].astype(float), "I_R": z["I_R"].astype(float)}
    return float(z["freq_cmd"]), z["t"], sig, complex(z["z_re"], z["z_im"])


@pytest.mark.parametrize("path", POINTS, ids=[p.stem for p in POINTS])
def test_playback_reproduces_the_recorded_impedance(path):
    f, t, sig, z_rec = _load(path)
    f_used, res = analysis.analyze(sig, f, t=t)
    z = res["V_R"][0].phasor / res["I_R"][0].phasor
    assert abs(z) == pytest.approx(abs(z_rec), rel=1e-3)
    assert np.angle(z / z_rec) == pytest.approx(0, abs=2e-3)
    assert abs(f_used - f) / f < 1e-4
    assert res["I_R"][0].quality.estimate_snr_db > 50


def test_low_frequency_impedance_is_five_ohm_with_inverted_probe():
    f, t, sig, _ = _load(POINTS[0])
    _, res = analysis.analyze(sig, f, t=t)
    z = res["V_R"][0].phasor / res["I_R"][0].phasor
    assert abs(z) == pytest.approx(5.0, rel=0.01)
    assert abs(np.degrees(np.angle(z))) == pytest.approx(180, abs=1)


def test_playback_sweep_and_circuit_fit(tmp_path):
    caps = [_load(p) for p in POINTS]
    idx = {"i": 0}

    def rx(freq, plan):
        f, t, sig, _ = caps[idx["i"]]
        idx["i"] += 1
        return Capture(dict(sig), t=t, units={"V_R": "V", "I_R": "A"}, group="mso")

    res = run_sweep(
        [c[0] for c in caps],
        lambda f, a: None,
        [rx],
        expressions={"Z": (lambda v: -v["V_R"] / v["I_R"], "ohm")},
        folder=tmp_path,
        quiet=True,
    )
    z = Run(res.path).series("Z")
    fit = fit_model(
        z.freqs,
        z.values,
        lambda f, R, L, tau: (R + 2j * np.pi * f * L) * np.exp(2j * np.pi * f * tau),
        {"R": 5.0, "L": 1e-7, "tau": 1e-9},
        bounds={"R": (0, 100), "L": (0, 1e-5)},
    )
    assert fit.params["R"] == pytest.approx(5.0, rel=0.01)
    assert 3e-8 < fit.params["L"] < 3e-7
