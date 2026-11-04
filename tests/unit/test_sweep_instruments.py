"""Hardware-free tests for the ready-made sweep sources, receiver and scalar readings.

The AFG31000, SDG3000X and RTB2000 adapters run against the existing fakes
(``FakeAfg`` in ``test_tektronix_afg31000.py``, ``tests/mocks/fake_sdg3000x.py``
and ``tests/mocks/fake_rtb2000.py``). The 33120A and the scalar helpers have no
fake, so they run against small stubs that log each call. A pass here says
the adapters call the drivers in the right order; it says nothing new about
the instruments.
"""

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import drivers.rohde_schwarz_rtb2000 as rtb_module
import drivers.siglent_sdg3000x as sdg_module
import drivers.tektronix_afg31000 as afg_module
import sweep.instruments.scalar as scalar
from drivers.rohde_schwarz_rtb2000 import RohdeSchwarz_RTB2000
from drivers.siglent_sdg3000x import Siglent_SDG3000X
from drivers.tektronix_afg31000 import Tektronix_AFG31000
from sweep.capture import CaptureRules, plan_capture
from sweep.instruments.keysight_33120a import Keysight33120ASource
from sweep.instruments.rohde_schwarz_rtb2000 import (
    RohdeSchwarzRtb2000AfgSource,
    RohdeSchwarzRtb2000Receiver,
    nearest_record_length,
)
from sweep.instruments.scalar import ScalarMeasurement
from sweep.instruments.siglent_sdg3000x import SiglentSdg3000xSource
from sweep.instruments.tektronix_afg31000 import TektronixAfg31000Source
from sweep.interfaces import Capture, Readings
from tests.mocks.fake_rtb2000 import FakeRTB
from tests.mocks.fake_sdg3000x import FakeSDG
from tests.unit.test_tektronix_afg31000 import FakeAfg

# ---------------------------------------------------------------- stubs


def _preserved(log: list):
    @contextlib.contextmanager
    def preserved_setup():
        log.append("preserve_enter")
        try:
            yield
        finally:
            log.append("preserve_exit")

    return preserved_setup


class StubChannel:
    """A generator channel that logs each call and reads back a snapped value."""

    def __init__(self, log: list, load: float = math.inf):
        self.log = log
        self.load = load
        self.fail_on: set[str] = set()

    def _call(self, name, *args):
        self.log.append((name, *args))
        if name in self.fail_on:
            raise RuntimeError(f"stub refused {name}")

    def set_output_state(self, state):
        self._call("output", bool(state))
        return bool(state)

    def set_waveform_sine(self, freq, ampl, **kw):
        self._call("sine", freq, ampl)
        lim = kw.get("v_limit")
        if lim is not None:
            self.limits = (-abs(lim), abs(lim)) if isinstance(lim, (int, float)) else lim
        return SimpleNamespace(frequency=round(freq, 1), amplitude=round(ampl, 2))

    def set_modulation_off(self):
        self._call("modulation_off")

    def set_burst_state(self, state):
        self._call("burst", bool(state))
        return bool(state)

    def get_voltage_limits(self):
        lo, hi = getattr(self, "limits", (-5.0, 5.0))
        return SimpleNamespace(low=lo, high=hi, enabled=True)

    def set_frequency(self, hz):
        self._call("freq", hz)
        return round(hz, 1)

    def set_amplitude(self, vpp):
        self._call("ampl", vpp)
        return 0.05 if vpp == "MIN" else round(vpp, 2)

    def set_load(self, load):
        self._call("load", load)
        self.load = math.inf if load == "HIGHZ" else float(load)
        return self.load

    def get_load(self):
        return self.load


class StubChannelDriver:
    """AFG31000 or SDG3000X shape: ``channel(n)`` returns the node."""

    def __init__(self, log: list, load: float = math.inf):
        self.ch1 = StubChannel(log, load)
        self.preserved_setup = _preserved(log)

    def channel(self, n):
        return self.ch1


class StubScope:
    """RTB2000 shape for the generator source: ``.afg`` is the node."""

    def __init__(self, log: list):
        self.afg = StubChannel(log)
        self.preserved_setup = _preserved(log)


class Stub33120A(StubChannel):
    """The 33120A shape: every method on the instrument itself, no relay."""

    def __init__(self, log: list):
        super().__init__(log)
        self.preserved_setup = _preserved(log)

    def set_amplitude_unit(self, unit):
        self._call("unit", unit)
        return unit

    def set_sync_state(self, state):
        self._call("sync", state)
        return state

    def set_load(self, load):
        self._call("load", load)
        self.load = load
        return load

    def get_amplitude(self, limit=None):
        return 0.05

    def set_output_state(self, state):  # pragma: no cover - must never be called
        raise AssertionError("the 33120A has no output relay")


def _stub_sources():
    """(source, log, channel stub) for each source class, on stubs."""
    out = []
    log: list = []
    drv = StubChannelDriver(log)
    out.append((TektronixAfg31000Source(drv, 1, max_vpp=2.0, v_limit=1.5), log, drv.ch1))
    log = []
    drv = StubChannelDriver(log)
    out.append((SiglentSdg3000xSource(drv, 1, ampl_limit=2.0), log, drv.ch1))
    log = []
    scope = StubScope(log)
    out.append((RohdeSchwarzRtb2000AfgSource(scope, max_vpp=2.0), log, scope.afg))
    return out


_IDS = ["afg31000", "sdg3000x", "rtb2000_afg"]


def _events(log):
    return [e if isinstance(e, str) else e[0] if e[0] != "output" else e for e in log]


# ---------------------------------------------------------------- relay sources on stubs


@pytest.mark.parametrize("i", range(3), ids=_IDS)
def test_source_open_set_output_close_order(i):
    src, log, _ = _stub_sources()[i]
    src.open()
    src.set(1e3, 1.0)
    src.set(2e3, 1.5)
    src.output(True)
    src.output(False)
    src.close()
    extra = {0: ["modulation_off"], 1: [], 2: ["burst"]}[i]
    assert _events(log) == [
        "preserve_enter",
        ("output", False),
        *extra,
        "sine",
        "freq",
        "ampl",
        ("output", True),
        ("output", False),
        ("output", False),
        "preserve_exit",
    ]


@pytest.mark.parametrize("i", range(3), ids=_IDS)
def test_source_returns_the_read_back_not_the_request(i):
    src, _, _ = _stub_sources()[i]
    src.open()
    assert src.set(1000.04, 1.234) == {"freq": 1000.0, "amplitude": 1.23}
    assert src.set(2000.06, 1.456) == {"freq": 2000.1, "amplitude": 1.46}
    src.close()


@pytest.mark.parametrize("i", range(3), ids=_IDS)
def test_source_refuses_an_amplitude_above_the_limit_before_any_write(i):
    src, log, _ = _stub_sources()[i]
    src.open()
    log.clear()
    with pytest.raises(ValueError, match="above the limit"):
        src.set(1e3, 2.5)
    with pytest.raises(ValueError, match="amplitude is required"):
        src.set(1e3, None)
    assert log == []
    src.close()


@pytest.mark.parametrize("i", range(3), ids=_IDS)
def test_source_close_turns_the_output_off_after_an_error(i):
    src, log, ch = _stub_sources()[i]
    src.open()
    src.set(1e3, 1.0)
    src.output(True)
    ch.fail_on.add("freq")
    with pytest.raises(RuntimeError):
        src.set(2e3, 1.0)
    log.clear()
    src.close()
    assert _events(log) == [("output", False), "preserve_exit"]


@pytest.mark.parametrize("i", range(3), ids=_IDS)
def test_source_close_leaves_the_preserved_setup_even_when_output_off_fails(i):
    src, log, ch = _stub_sources()[i]
    src.open()
    ch.fail_on.add("output")
    with pytest.raises(RuntimeError):
        src.close()
    assert log[-1] == "preserve_exit"


@pytest.mark.parametrize("i", range(3), ids=_IDS)
def test_source_describe_has_identity_and_settings(i):
    src, _, _ = _stub_sources()[i]
    d = src.describe()
    assert d["class"] in ("StubChannelDriver", "StubScope")
    assert "max_vpp" in d and "offset" in d


def test_preserve_false_skips_the_preserved_setup():
    log: list = []
    src = TektronixAfg31000Source(StubChannelDriver(log), max_vpp=1.0, v_limit=1.0, preserve=False)
    src.open()
    src.close()
    assert "preserve_enter" not in log


def test_sdg_limit_is_half_at_a_50_ohm_load():
    log: list = []
    src = SiglentSdg3000xSource(StubChannelDriver(log, load=50.0), ampl_limit=2.0)
    src.open()
    assert src.describe()["allowed_vpp_at_load"] == 1.0
    with pytest.raises(ValueError, match="above the limit"):
        src.set(1e3, 1.2)
    assert src.set(1e3, 1.0)["amplitude"] == 1.0
    src.close()


def test_sdg_max_vpp_tightens_the_limit():
    log: list = []
    src = SiglentSdg3000xSource(StubChannelDriver(log), ampl_limit=10.0, max_vpp=1.0)
    src.open()
    with pytest.raises(ValueError, match="above the limit"):
        src.set(1e3, 1.5)
    src.close()


# ---------------------------------------------------------------- 33120A, no relay


def test_33120a_open_sets_minimum_after_entering_the_preserved_setup():
    log: list = []
    src = Keysight33120ASource(Stub33120A(log), max_vpp=2.0, load="FIFTY")
    src.open()
    assert log == [
        "preserve_enter",
        ("ampl", "MIN"),
        ("unit", "VPP"),
        ("load", "FIFTY"),
        ("sync", True),
    ]


def test_33120a_output_off_is_minimum_amplitude_and_on_restores_the_request():
    log: list = []
    src = Keysight33120ASource(Stub33120A(log), max_vpp=2.0, sync=False)
    src.open()
    # Until output(True) the amplitude stays at the minimum: no relay, so the
    # output call is the only path to the drive level.
    assert src.set(1e3, 1.0) == {"freq": 1e3, "amplitude": 0.05}
    assert src.output(True) == 1.0
    assert src.set(2e3, 1.5) == {"freq": 2e3, "amplitude": 1.5}
    src.output(False)
    src.output(True)
    src.close()
    assert log == [
        "preserve_enter",
        ("ampl", "MIN"),
        ("unit", "VPP"),
        ("sine", 1e3, 0.05),
        ("ampl", 1.0),
        ("freq", 2e3),
        ("ampl", 1.5),
        ("ampl", "MIN"),
        ("ampl", 1.5),
        ("ampl", "MIN"),
        "preserve_exit",
    ]


def test_33120a_drive_off_sweep_never_leaves_the_minimum(tmp_path):
    from sweep.run import run_sweep

    log: list = []
    src = Keysight33120ASource(Stub33120A(log), max_vpp=2.0, sync=False)
    run_sweep(
        [1e3, 2e3],
        src,
        [lambda f, p: {"x": 1.0}],
        amplitude=1.5,
        drive_off=True,
        folder=tmp_path,
        quiet=True,
        plot=False,
        review=False,
    )
    levels = [e[-1] for e in log if e[0] in ("sine", "ampl")]
    assert levels and all(v in ("MIN", 0.05) for v in levels)


def test_33120a_output_on_without_a_set_writes_nothing():
    log: list = []
    src = Keysight33120ASource(Stub33120A(log), max_vpp=2.0, sync=False)
    src.open()
    log.clear()
    src.output(True)
    assert log == []


def test_33120a_refuses_above_the_limit_and_close_still_sets_minimum():
    log: list = []
    gen = Stub33120A(log)
    src = Keysight33120ASource(gen, max_vpp=1.0, sync=False)
    src.open()
    log.clear()
    with pytest.raises(ValueError, match="above the limit"):
        src.set(1e3, 1.5)
    assert log == []
    src.set(1e3, 0.5)
    gen.fail_on.add("freq")
    with pytest.raises(RuntimeError):
        src.set(2e3, 0.5)
    log.clear()
    src.close()
    assert log == [("ampl", "MIN"), "preserve_exit"]


def test_33120a_describe_says_there_is_no_relay():
    src = Keysight33120ASource(Stub33120A([]), max_vpp=1.0)
    d = src.describe()
    assert d["output_relay"] is False
    assert "minimum amplitude" in d["output_off_means"]
    assert d["minimum_amplitude"] == 0.05
    assert d["restores_found_amplitude_on_close"] is True


# ---------------------------------------------------------------- AFG31000 on FakeAfg


@pytest.fixture
def make_afg(monkeypatch):
    monkeypatch.setattr(afg_module.visa, "ResourceManager", lambda: None)
    monkeypatch.setattr(afg_module, "_BUSY_SETTLE_S", 0.0)
    monkeypatch.setattr(afg_module, "_UPLOAD_RETRY_S", 0.0)

    def factory(model: str = "AFG31102"):
        fake = FakeAfg(model, "TCPIP0::x::5025::SOCKET")
        afg = Tektronix_AFG31000("TCPIP0::x::5025::SOCKET")
        monkeypatch.setattr(afg.link, "_open_resource", fake.open_resource)
        afg.connect()
        return afg, fake

    return factory


def test_afg31000_drives_the_fake_and_restores_it(make_afg):
    afg, fake = make_afg()
    fake.ch[2].output = True
    src = TektronixAfg31000Source(afg, 2, max_vpp=2.0, v_limit=1.5)
    src.open()
    assert fake.ch[2].output is False
    got = src.set(1e3, 1.0)
    assert got == {"freq": 1e3, "amplitude": 1.0}
    assert fake.ch[2].output is False
    src.output(True)
    assert fake.ch[2].output is True
    assert src.set(2e3, 0.5) == {"freq": 2e3, "amplitude": 0.5}
    d = src.describe()
    assert d["load_readback"] == 50.0 and d["voltage_limits"]["high"] == 1.5
    src.close()
    assert (fake.ch[2].freq, fake.ch[2].vpp) == (1e6, 1.0)
    assert (fake.ch[2].lim_low, fake.ch[2].lim_high) == (-5.0, 5.0)
    assert fake.ch[2].output is True  # found on, so the restore puts it back on


def test_afg31000_driver_refuses_a_swing_outside_v_limit(make_afg):
    afg, fake = make_afg()
    src = TektronixAfg31000Source(afg, 1, max_vpp=5.0, v_limit=0.5)
    src.open()
    with pytest.raises(ValueError, match="outside the limit"):
        src.set(1e3, 2.0)
    src.close()
    assert fake.ch[1].output is False


# ---------------------------------------------------------------- SDG3000X on FakeSDG


@pytest.fixture
def make_sdg(monkeypatch):
    monkeypatch.setattr(sdg_module.visa, "ResourceManager", lambda: None)

    def factory():
        fake = FakeSDG()
        sdg = Siglent_SDG3000X("TCPIP::FAKE::INSTR")
        monkeypatch.setattr(sdg.link, "_open_resource", lambda rm, name: fake)
        sdg.connect()
        return sdg, fake

    return factory


def test_sdg3000x_drives_the_fake_and_restores_it(make_sdg):
    sdg, fake = make_sdg()
    src = SiglentSdg3000xSource(sdg, 1, ampl_limit=2.0)
    src.open()
    assert fake.ch["C1"].out == "OFF"
    assert src.set(1e3, 1.5) == {"freq": 1e3, "amplitude": 1.5}
    src.output(True)
    assert fake.ch["C1"].out == "ON"
    assert src.set(5e3, 1.0) == {"freq": 5e3, "amplitude": 1.0}
    src.close()
    assert fake.ch["C1"].out == "OFF"
    assert (fake.ch["C1"].p["FRQ"], fake.ch["C1"].p["AMP"]) == (1000.0, 4.0)
    assert not any("drv_snapshot_" in k for k in fake.fs)


def test_sdg3000x_load_50_halves_the_allowed_amplitude(make_sdg):
    sdg, fake = make_sdg()
    src = SiglentSdg3000xSource(sdg, 1, ampl_limit=2.0, load=50)
    src.open()
    with pytest.raises(ValueError, match="above the limit"):
        src.set(1e3, 1.5)
    src.close()
    assert fake.ch["C1"].load == "HZ"


# ---------------------------------------------------------------- RTB2000 on FakeRTB


@pytest.fixture
def make_rtb(monkeypatch):
    monkeypatch.setattr(rtb_module.visa, "ResourceManager", lambda: None)
    for name in ("_RECORD_LENGTH_POLL_S", "_RESULT_POLL_S", "_DVM_POLL_S", "_ROLL_MARGIN_S"):
        monkeypatch.setattr(rtb_module, name, 0.0)

    def factory():
        fake = FakeRTB()
        scope = RohdeSchwarz_RTB2000("TCPIP::FAKE::5025::SOCKET")
        monkeypatch.setattr(scope.link, "_open_resource", fake.open_resource)
        scope.connect()
        fake.log.clear()
        return scope, fake

    return factory


def test_nearest_record_length_rounds_up_and_caps():
    assert nearest_record_length(1) == 10_000
    assert nearest_record_length(10_000) == 10_000
    assert nearest_record_length(10_001) == 20_000
    assert nearest_record_length(3_000_000) == 5_000_000
    assert nearest_record_length(10**9) == 20_000_000
    assert nearest_record_length(150, choices=(100, 200)) == 200


def test_rtb2000_receiver_returns_a_capture_with_matching_lengths(make_rtb):
    scope, fake = make_rtb()
    rx = RohdeSchwarzRtb2000Receiver(scope, {"V_in": "CH1", "V_out": "ch2"}, trigger="ch1")
    rx.open()
    assert fake.s["CHAN2:STAT"] in ("1", "ON")
    assert fake.s["TRIG:A:MODE"] == "AUTO"
    plan = plan_capture(1e4, CaptureRules(max_record_length=50_000))
    cap = rx.measure(1e4, plan)
    assert isinstance(cap, Capture)
    assert set(cap.signals) == {"V_in", "V_out"}
    assert all(len(v) == len(cap.t) for v in cap.signals.values())
    # The read-backs come from the scope, not from the transferred array, so the
    # sweep loop can catch a transfer shorter than the record.
    assert cap.settings["points_transferred"] == len(cap.t)
    assert (
        cap.settings["record_length"] == int(fake.s["ACQ:POIN"])
        or cap.settings["record_length"] == scope.acq.get_record_length()
    )
    assert cap.settings["sample_rate"] == pytest.approx(scope.acq.get_adc_rate())
    assert cap.settings["div_count"] == 12
    assert cap.settings["acquisition_mode"] and cap.settings["trigger_mode"]
    from sweep.capture import validate_capture

    flags, invalid = validate_capture(plan, cap.settings, dt=cap.t[1] - cap.t[0])
    if cap.settings["record_length"] != len(cap.t):
        assert invalid and "transfer_short" in flags
    assert cap.settings["record_length_requested"] == nearest_record_length(plan.record_length)
    assert cap.settings["time_scale"] == pytest.approx(plan.duration / 12)
    assert cap.full_scale["V_in"] == pytest.approx((-1.0, 1.0))
    assert cap.units == {"V_in": "V", "V_out": "V"}
    assert cap.clipped == {}
    assert any(c == "SING" for c in fake.log)
    assert rx.describe()["channel_setup"]["V_in"]["scale"] == 0.2
    rx.close()
    assert float(fake.s["TIM:SCAL"]) == pytest.approx(1e-4)


def test_rtb2000_receiver_does_not_raise_on_a_clipped_channel(make_rtb):
    scope, fake = make_rtb()
    fake.clip = True
    rx = RohdeSchwarzRtb2000Receiver(scope, {"V": "ch1"})
    rx.open()
    cap = rx.measure(1e4, plan_capture(1e4))
    lo, hi = cap.full_scale["V"]
    assert np.max(cap.signals["V"]) > hi
    rx.close()


def test_rtb2000_afg_source_drives_the_fake_and_restores_it(make_rtb):
    scope, fake = make_rtb()
    src = RohdeSchwarzRtb2000AfgSource(scope, max_vpp=2.0)
    src.open()
    assert fake.s["WGEN:OUTP"] in ("OFF", "0")
    assert src.set(2e3, 1.0) == {"freq": 2e3, "amplitude": 1.0}
    src.output(True)
    assert fake.s["WGEN:OUTP"] in ("ON", "1")
    assert src.set(3e3, 1.5) == {"freq": 3e3, "amplitude": 1.5}
    with pytest.raises(ValueError, match="above the limit"):
        src.set(3e3, 2.5)
    src.close()
    assert fake.s["WGEN:OUTP"] in ("OFF", "0")
    assert float(fake.s["WGEN:FREQ"]) == 1000.0 and float(fake.s["WGEN:VOLT"]) == 0.5


# ---------------------------------------------------------------- scalar readings


def test_scalar_measurement_reads_once_per_point():
    calls = iter([1.0, 2.0])
    m = ScalarMeasurement("V", lambda: next(calls), "V")
    r = m.measure(1e3, None)
    assert isinstance(r, Readings)
    assert r.values == {"V": 1.0} and r.units == {"V": "V"} and r.group is None
    assert r.settings["n"] == 1 and "std" not in r.settings


def test_scalar_measurement_averages_n_and_reports_the_spread():
    calls = iter([1.0, 2.0, 3.0])
    m = ScalarMeasurement("I", lambda: next(calls), "A", n=3)
    r = m.measure(1e3, None)
    assert r.values["I"] == pytest.approx(2.0)
    assert r.settings["std"] == pytest.approx(1.0)
    assert r.settings["readings"] == [1.0, 2.0, 3.0]


def test_scalar_measurement_keeps_an_overload_as_nan():
    calls = iter([1.0, math.nan])
    r = ScalarMeasurement("V", lambda: next(calls), "V", n=2).measure(1e3, None)
    assert math.isnan(r.values["V"])


def test_scalar_measurement_settles_before_reading(monkeypatch):
    order = []
    monkeypatch.setattr(scalar.time, "sleep", lambda s: order.append(("sleep", s)))
    m = ScalarMeasurement("V", lambda: order.append("read") or 1.0, "V", settle_s=0.25)
    m.measure(1e3, None)
    assert order == [("sleep", 0.25), "read"]


def test_scalar_measurement_refuses_n_below_one():
    with pytest.raises(ValueError, match="n must be at least 1"):
        ScalarMeasurement("V", lambda: 1.0, "V", n=0)


def test_scalar_measurement_preserves_only_when_asked():
    log: list = []
    inst = SimpleNamespace(preserved_setup=_preserved(log))
    m = ScalarMeasurement("V", lambda: 1.0, "V", instrument=inst)
    m.open()
    m.close()
    assert log == []
    m = ScalarMeasurement("V", lambda: 1.0, "V", instrument=inst, preserve=True)
    m.open()
    m.close()
    assert log == ["preserve_enter", "preserve_exit"]
    assert m.describe()["preserved"] is True


def _instrument(log=None):
    inst = mock.MagicMock()
    inst.resource_name = "TCPIP::FAKE::INSTR"
    inst.preserved_setup = _preserved(log if log is not None else [])
    for path, value in (
        ("meas_vac", 0.7),
        ("meas_iac", 0.01),
        ("meas_frequency", 1e3),
        ("meas_voltage", 5.0),
        ("meas_current", 0.2),
        ("meas_power", 1.0),
    ):
        getattr(inst, path).return_value = value
        getattr(inst.channel.return_value, path).return_value = value
    inst.channel.return_value.meas_vi.return_value = SimpleNamespace(voltage=5.0, current=0.2)
    return inst


@pytest.mark.parametrize(
    ("helper", "method", "unit", "value", "on_channel"),
    [
        (scalar.siglent_sdm_ac_voltage, "meas_vac", "V", 0.7, False),
        (scalar.siglent_sdm_ac_current, "meas_iac", "A", 0.01, False),
        (scalar.siglent_sdm_frequency, "meas_frequency", "Hz", 1e3, False),
        (scalar.keysight_e364xa_voltage, "meas_voltage", "V", 5.0, True),
        (scalar.keysight_e364xa_current, "meas_current", "A", 0.2, True),
        (scalar.keysight_e364xa_power, "meas_vi", "W", 1.0, True),
        (scalar.owon_odp_voltage, "meas_voltage", "V", 5.0, True),
        (scalar.owon_odp_current, "meas_current", "A", 0.2, True),
        (scalar.owon_odp_power, "meas_power", "W", 1.0, True),
        (scalar.keysight_b2900_voltage, "meas_voltage", "V", 5.0, True),
        (scalar.keysight_b2900_current, "meas_current", "A", 0.2, True),
        (scalar.rigol_dl3000_voltage, "meas_voltage", "V", 5.0, False),
        (scalar.rigol_dl3000_current, "meas_current", "A", 0.2, False),
        (scalar.rigol_dl3000_power, "meas_power", "W", 1.0, False),
    ],
)
def test_scalar_helpers_call_the_named_driver_reading(helper, method, unit, value, on_channel):
    inst = _instrument()
    m = helper(inst, name="X") if not on_channel else helper(inst, 2, name="X")
    assert isinstance(m, ScalarMeasurement)
    r = m.measure(1e3, None)
    assert r.values["X"] == pytest.approx(value) and r.units["X"] == unit
    node = inst.channel.return_value if on_channel else inst
    getattr(node, method).assert_called_once()
    if on_channel:
        inst.channel.assert_called_with(2)
    d = m.describe()
    assert d["resource_name"] == "TCPIP::FAKE::INSTR" and d["unit"] == unit


def test_sdm_helpers_preserve_the_meter_setup_and_pass_options():
    log: list = []
    dmm = _instrument(log)
    m = scalar.siglent_sdm_ac_voltage(dmm, range=2, bandwidth=20, n=2)
    m.open()
    r = m.measure(1e3, None)
    m.close()
    assert log == ["preserve_enter", "preserve_exit"]
    dmm.meas_vac.assert_called_with(range=2, bandwidth=20)
    assert dmm.meas_vac.call_count == 2 and r.values == {"V_ac": 0.7}


def test_supply_helpers_do_not_preserve_by_default():
    log: list = []
    m = scalar.keysight_e364xa_voltage(_instrument(log))
    m.open()
    m.close()
    assert log == [] and m.describe()["preserved"] is False
