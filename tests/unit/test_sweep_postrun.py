"""Post-run sweep tools on synthetic responses: interpolation, calibration, fits."""

import math

import numpy as np
import pytest

from sweep.calibration import Calibration
from sweep.fit import TransferFit, auto_fit, fit_delay, fit_rational
from sweep.interp import common_grid, interpolate

F = np.geomspace(10.0, 1e6, 121)


def lowpass1(f, fc=1e3, k=2.0):
    return k / (1 + 1j * f / fc)


def resonance(f, f0=1e4, q=5.0):
    s = 1j * f / f0
    return 1 / (s**2 + s / q + 1)


def lead_lag(f, fz=1e3, fp=1e4):
    return (1 + 1j * f / fz) / (1 + 1j * f / fp)


def noisy(h, level, seed=1):
    rng = np.random.default_rng(seed)
    return h * (1 + level * (rng.standard_normal(h.size) + 1j * rng.standard_normal(h.size)))


def loglinear(f):
    """dB and phase both straight lines in log f, so interpolation is exact."""
    lf = np.log10(f)
    return 10 ** ((-20 * lf + 40) / 20) * np.exp(1j * np.radians(400 * lf))


# --- interpolation ---------------------------------------------------------


def test_interpolate_returns_measured_points_exactly():
    h = lowpass1(F)
    out, outside = interpolate(F, h, F)
    assert np.allclose(out, h, rtol=1e-12)
    assert not outside.any()


def test_interpolate_exact_between_points_for_loglinear_response():
    f = np.geomspace(10, 1e5, 41)
    nf = np.geomspace(12, 9e4, 300)
    out, outside = interpolate(f, loglinear(f), nf)
    assert np.allclose(out, loglinear(nf), rtol=1e-9)
    assert not outside.any()


def test_interpolate_handles_phase_wrapping():
    f = np.geomspace(10, 1e5, 41)
    nf = np.sqrt(f[:-1] * f[1:])
    out, _ = interpolate(f, loglinear(f), nf)
    assert np.allclose(np.angle(out / loglinear(nf)), 0, atol=1e-9)


def test_interpolate_dense_round_trip_on_lowpass():
    fine = np.geomspace(10, 1e6, 801)
    coarse = fine[::8]
    out, _ = interpolate(coarse, lowpass1(coarse), fine)
    db_err = 20 * np.log10(np.abs(out / lowpass1(fine)))
    assert np.max(np.abs(db_err)) < 0.05
    assert np.max(np.abs(np.degrees(np.angle(out / lowpass1(fine))))) < 0.5


def test_interpolate_unsorted_input_and_nan_points_skipped():
    f = np.geomspace(10, 1e5, 41)
    h = loglinear(f)
    h_bad = h.copy()
    h_bad[5] = np.nan
    idx = np.random.default_rng(0).permutation(f.size)
    out, _ = interpolate(f[idx], h_bad[idx], f)
    assert np.allclose(out, h, rtol=1e-9)


def test_interpolate_real_input_stays_real_and_linear():
    f = np.array([10.0, 100.0, 1000.0])
    v = np.array([-1.0, 1.0, 3.0])
    out, _ = interpolate(f, v, [math.sqrt(10) * 10, 100.0])
    assert not np.iscomplexobj(out)
    assert np.allclose(out, [0.0, 1.0])


def test_extrapolate_modes():
    f = np.geomspace(100, 1e4, 21)
    h = loglinear(f)
    nf = np.array([10.0, 1e3, 1e5])
    hold, outside = interpolate(f, h, nf, extrapolate="hold")
    assert outside.tolist() == [True, False, True]
    assert np.isclose(hold[0], h[0]) and np.isclose(hold[2], h[-1])
    nan, _ = interpolate(f, h, nf, extrapolate="nan")
    assert np.isnan(nan[0]) and np.isnan(nan[2]) and np.isfinite(nan[1])
    lin, _ = interpolate(f, h, nf, extrapolate="linear")
    assert np.allclose(lin, loglinear(nf), rtol=1e-9)


def test_interpolate_rejects_bad_input():
    with pytest.raises(ValueError):
        interpolate([1, 2], [1, 2], [1.5], extrapolate="clamp")
    with pytest.raises(ValueError):
        interpolate([0, 2], [1, 2], [1.5])
    with pytest.raises(ValueError):
        interpolate([1, 1], [1, 2], [1.5])
    with pytest.raises(ValueError):
        interpolate([1, 2, 3], [1, 2], [1.5])


def test_common_grid_overlap_union_and_points():
    a = np.array([10.0, 100.0, 1000.0])
    b = np.array([50.0, 500.0, 5000.0])
    assert common_grid([a, b]).tolist() == [50.0, 100.0, 500.0, 1000.0]
    assert common_grid([a, b], mode="union").tolist() == [10, 50, 100, 500, 1000, 5000]
    g = common_grid([a, b], points=5)
    assert np.isclose(g[0], 50) and np.isclose(g[-1], 1000) and g.size == 5
    assert np.allclose(np.diff(np.log10(g)), np.log10(20) / 4)
    with pytest.raises(ValueError):
        common_grid([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        common_grid([a], mode="both")


# --- calibration -----------------------------------------------------------


def test_calibration_from_measurement_corrects_to_expected():
    path_error = lowpass1(F, fc=2e5, k=0.9) * np.exp(-2j * np.pi * F * 3e-9)
    cal = Calibration.from_measurement(F, path_error, expected=1.0, name="cables")
    dut = resonance(F)
    fixed, outside = cal.apply(F, dut * path_error)
    assert np.allclose(fixed, dut, rtol=1e-12)
    assert not outside.any()


def test_calibration_expected_function_and_array():
    meas = 0.5 * lead_lag(F)
    cal_fn = Calibration.from_measurement(F, meas, expected=lambda f: lead_lag(f))
    cal_arr = Calibration.from_measurement(F, meas, expected=lead_lag(F))
    assert np.allclose(cal_fn.correction, 2.0)
    assert np.allclose(cal_arr.correction, 2.0)


def test_calibration_interpolates_and_flags_outside():
    path = lowpass1(F, fc=3e4, k=1.0)
    cal = Calibration.from_measurement(F[::4], path[::4])
    nf = np.geomspace(20, 5e5, 60)
    fixed, outside = cal.apply(nf, lowpass1(nf, fc=3e4, k=1.0))
    assert np.max(np.abs(20 * np.log10(np.abs(fixed)))) < 0.1
    assert np.max(np.abs(np.degrees(np.angle(fixed)))) < 1.0
    assert not outside.any()
    _, outside = cal.apply([1.0, 1e3, 1e8], [1.0, 1.0, 1.0])
    assert outside.tolist() == [True, False, True]


def test_calibration_two_channel_through_matches_channels():
    ch_a = 1.0 * np.exp(-2j * np.pi * F * 1e-9)
    ch_b = 1.02 * np.exp(-2j * np.pi * F * 4e-9)
    cal = Calibration.from_two_channel_through(F, ch_a, ch_b)
    fixed, _ = cal.apply(F, ch_b * resonance(F))
    assert np.allclose(fixed, ch_a * resonance(F), rtol=1e-9)


def test_calibration_save_load_round_trip(tmp_path):
    cal = Calibration.from_measurement(
        F, resonance(F), name="probe A, x10", meta={"amplitude": np.float64(0.5), "ch": 2}
    )
    p = cal.save(tmp_path / "cal.csv")
    text = p.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("#") and text[1].startswith("# name:")
    assert "freq_hz,corr_real,corr_imag,corr_mag_db,corr_phase_deg" in text
    back = Calibration.load(p)
    assert back.name == "probe A, x10"
    assert back.meta == {"amplitude": 0.5, "ch": 2}
    assert np.array_equal(back.freqs, cal.freqs)
    assert np.array_equal(back.correction, cal.correction)


def test_calibration_refuses_zero_and_keeps_nan_out_of_apply():
    with pytest.raises(ValueError):
        Calibration.from_measurement([1.0, 2.0], [1.0, 0.0])
    meas = np.ones(F.size, dtype=complex)
    meas[10] = np.nan
    cal = Calibration.from_measurement(F, meas)
    fixed, _ = cal.apply(F, np.ones(F.size))
    assert np.allclose(fixed, 1.0)


# --- delay fit -------------------------------------------------------------


def test_fit_delay_recovers_skew_and_phase0():
    f = np.linspace(1e5, 1e8, 200)
    h = 0.8 * np.exp(1j * np.radians(30)) * np.exp(-2j * np.pi * f * 2.5e-9)
    delay, phase0 = fit_delay(f, h)
    assert delay == pytest.approx(2.5e-9, rel=1e-9)
    assert phase0 == pytest.approx(30.0, abs=1e-6)


def test_fit_delay_with_noise_and_inversion():
    f = np.linspace(1e5, 1e8, 400)
    h = noisy(-np.exp(-2j * np.pi * f * 1.2e-9), 0.01)
    delay, phase0 = fit_delay(f, h)
    assert delay == pytest.approx(1.2e-9, rel=0.01)
    assert abs(abs(phase0) - 180) < 1.0


# --- rational fits ---------------------------------------------------------


def test_fit_first_order_lowpass():
    fit = fit_rational(F, lowpass1(F), 0, 1)
    assert fit.poles[0].real == pytest.approx(-1e3, rel=1e-6)
    assert abs(fit.poles[0].imag) < 1e-6
    assert fit.dc_gain == pytest.approx(2.0, rel=1e-6)
    assert fit.rms_error_db < 1e-6
    assert np.allclose(fit.evaluate(F), lowpass1(F), rtol=1e-6)


def test_fit_second_order_resonance_f0_and_q():
    fit = fit_rational(F, resonance(F), 0, 2)
    p = fit.poles[np.argmax(fit.poles.imag)]
    assert abs(p) == pytest.approx(1e4, rel=1e-6)
    assert abs(p) / (2 * abs(p.real)) == pytest.approx(5.0, rel=1e-6)
    assert fit.dc_gain == pytest.approx(1.0, rel=1e-6)


def test_fit_lead_lag():
    fit = fit_rational(F, lead_lag(F), 1, 1)
    assert fit.zeros[0].real == pytest.approx(-1e3, rel=1e-6)
    assert fit.poles[0].real == pytest.approx(-1e4, rel=1e-6)
    assert fit.dc_gain == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fits_with_noise(seed):
    lp = fit_rational(F, noisy(lowpass1(F), 0.005, seed), 0, 1)
    assert lp.poles[0].real == pytest.approx(-1e3, rel=0.01)
    assert lp.dc_gain == pytest.approx(2.0, rel=0.01)
    rs = fit_rational(F, noisy(resonance(F), 0.005, seed), 0, 2)
    p = rs.poles[np.argmax(rs.poles.imag)]
    assert abs(p) == pytest.approx(1e4, rel=0.005)
    assert abs(p) / (2 * abs(p.real)) == pytest.approx(5.0, rel=0.03)
    ll = fit_rational(F, noisy(lead_lag(F), 0.005, seed), 1, 1)
    assert ll.zeros[0].real == pytest.approx(-1e3, rel=0.02)
    assert ll.poles[0].real == pytest.approx(-1e4, rel=0.02)
    assert ll.rms_error_db < 0.15 and ll.rms_error_deg < 1.0


def test_fit_with_delay():
    f = np.geomspace(100, 1e7, 161)
    h = lowpass1(f, fc=1e5) * np.exp(-2j * np.pi * f * 50e-9)
    fit = fit_rational(f, h, 0, 1, delay=True)
    assert fit.delay_s == pytest.approx(50e-9, rel=1e-4)
    assert fit.poles[0].real == pytest.approx(-1e5, rel=1e-4)
    assert fit.rms_error_deg < 0.01


def test_fit_weights_and_too_few_points():
    fit = fit_rational(F, lowpass1(F), 0, 1, weights=np.linspace(1, 2, F.size))
    assert fit.poles[0].real == pytest.approx(-1e3, rel=1e-6)
    with pytest.raises(ValueError):
        fit_rational(F[:2], lowpass1(F[:2]), 2, 2)
    with pytest.raises(ValueError):
        fit_rational(F, lowpass1(F), 0, 1, weights=np.ones(3))


@pytest.mark.parametrize(
    "func, nz, np_",
    [(lowpass1, 0, 1), (resonance, 0, 2), (lead_lag, 1, 1)],
)
def test_auto_fit_picks_true_order(func, nz, np_):
    fit = auto_fit(F, func(F), max_order=4)
    assert (fit.n_zeros, fit.n_poles) == (nz, np_)
    fit = auto_fit(F, noisy(func(F), 0.005), max_order=4)
    assert (fit.n_zeros, fit.n_poles) == (nz, np_)


def test_transfer_fit_evaluate_and_describe():
    fit = TransferFit(
        zeros=np.array([-1e3 + 0j]), poles=np.array([-1e4 + 0j]), gain=10.0, delay_s=1e-9
    )
    h = fit.evaluate(np.array([0.0, 1e9]))
    assert h[0] == pytest.approx(1.0)
    assert abs(h[1]) == pytest.approx(10.0, rel=1e-4)
    text = fit_rational(F, resonance(F), 0, 2).describe()
    assert "Q = 5" in text and "10 kHz" in text and "DC gain: 1" in text
    assert "-40 dB/decade" in text
    assert "pole at DC" in TransferFit(np.zeros(0), np.array([0j]), 1.0).describe()


def test_fit_integrator_and_highpass_report_dc_roots():
    f = np.geomspace(1, 1e7, 301)
    integ = fit_rational(f, 1 / (1j * f / 1e3) / (1 + 1j * f / 1e5), 0, 2)
    assert integ.dc_gain is None
    assert "pole at DC" in integ.describe()
    assert np.allclose(integ.evaluate(f[::10]), (1 / (1j * f / 1e3) / (1 + 1j * f / 1e5))[::10])
    hp = fit_rational(f, (1j * f / 1e3) / (1 + 1j * f / 1e3), 1, 1)
    assert hp.dc_gain == 0.0
    assert hp.poles[0].real == pytest.approx(-1e3, rel=1e-6)
