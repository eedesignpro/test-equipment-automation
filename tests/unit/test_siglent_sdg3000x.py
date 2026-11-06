# SPDX-License-Identifier: Apache-2.0
# Copyright: see the NOTICE file.
"""Hardware-free tests for the Siglent SDG3000X driver.

Ported from the original repository's tests, renamed to this repository's
driver standard. They run against ``tests/mocks/fake_sdg3000x.py``, which
reproduces the reply shapes and quirks measured there on an SDG3202X. A pass
here says the driver agrees with that fake; every renamed method stays
UNVERIFIED on hardware until the bench checks are rerun under the new names.
"""

from __future__ import annotations

import inspect
import math
import re

import numpy as np
import pytest
import pyvisa as visa

import drivers.siglent_sdg3000x as sdg_module
from common.errors import (
    InstrumentError,
    InstrumentQueryError,
    InstrumentTimeoutError,
    InstrumentUnresponsiveError,
    ValueSnappedWarning,
)
from common.scpi import ScpiEnum
from common.state import BurstState, ModulationState, SweepState, WaveformState
from drivers.siglent_sdg3000x import (
    BurstIdle,
    BurstMode,
    ChannelMode,
    ChirpRepeat,
    Coupling,
    CouplingState,
    CounterMode,
    Direction,
    Edge,
    Function,
    Gate,
    HopType,
    IqWaveform,
    Load,
    ModulationShape,
    ModulationSource,
    ModulationType,
    Polarity,
    Reference,
    RunMode,
    Segment,
    SequenceTriggerSource,
    Siglent_SDG3000X,
    Spacing,
    SweepType,
    TriggerOut,
    TriggerSource,
    WaitEvent,
)
from tests.mocks.fake_sdg3000x import IDN, FakeSDG


@pytest.fixture
def make_sdg(monkeypatch):
    """Factory: a connected driver on a fresh fake, with the connect traffic cleared."""
    monkeypatch.setattr(sdg_module.visa, "ResourceManager", lambda: None)

    def factory(fake: FakeSDG | None = None, resource: str = "TCPIP::FAKE::INSTR"):
        fake = fake or FakeSDG()
        sdg = Siglent_SDG3000X(resource)
        monkeypatch.setattr(sdg.link, "_open_resource", lambda rm, name: fake)
        sdg.connect()
        fake.log.clear()
        return sdg, fake

    return factory


@pytest.fixture
def sdg(make_sdg):
    return make_sdg()[0]


def _writes(fake: FakeSDG) -> list[str]:
    return [c for c in fake.log if not c.endswith("?") and "?" not in c.split()[0]]


# ---- construction and connection ------------------------------------------------------------


def test_init_builds_the_node_tree_with_no_io(monkeypatch):
    def refuse():
        raise AssertionError("__init__ must not open a resource")

    monkeypatch.setattr(sdg_module.visa, "ResourceManager", refuse)
    sdg = Siglent_SDG3000X("TCPIP::192.0.2.1::5025::SOCKET")
    assert not sdg.link.is_connected
    for node in ("ch1", "ch2", "multichannel", "iq", "file", "counter"):
        assert getattr(sdg, node).link is sdg.link
    assert sdg.ch1.sweep.link is sdg.link and not hasattr(sdg, "ch3")


def test_connect_attaches_nodes_and_reads_identity(make_sdg):
    sdg, _ = make_sdg()
    assert (sdg.model, sdg.serial, sdg.firmware) == ("SDG3202X", "SDG3XA0000000", "1.1.1.7R10")
    assert (sdg.num_channels, sdg.max_frequency) == (2, 200e6)
    for node in ("mod", "sweep", "burst", "harmonics", "hopping", "multitone", "multipulse",
                 "sequence"):  # fmt: skip
        assert hasattr(sdg.ch1, node)
    for node in ("multichannel", "iq", "file", "counter", "ch2"):
        assert hasattr(sdg, node)
    assert not hasattr(sdg, "ch3")
    assert sdg.channel(2) is sdg.ch2
    with pytest.raises(ValueError):
        sdg.channel(3)


@pytest.mark.parametrize(
    ("model", "top"), [("SDG3082X", 80e6), ("SDG3162X", 160e6), ("SDG3202X", 200e6)]
)
def test_model_table(make_sdg, model, top):
    sdg, _ = make_sdg(FakeSDG(IDN.replace("SDG3202X", model)))
    assert (sdg.model, sdg.max_frequency, sdg.num_channels) == (model, top, 2)


def test_connect_refuses_another_instrument(make_sdg):
    fake = FakeSDG("Keysight Technologies,33522A,MY0000000,1.0")
    with pytest.raises(InstrumentError, match="not an SDG3082X"):
        make_sdg(fake)
    assert fake.closed


def test_connect_refuses_a_model_missing_from_the_table(make_sdg):
    with pytest.raises(InstrumentError, match="_MODEL_INFO"):
        make_sdg(FakeSDG(IDN.replace("SDG3202X", "SDG3999X")))


def test_connect_drains_the_error_queue(make_sdg):
    fake = FakeSDG()
    fake.errors.append('-113,"Undefined header;*OPT?"')
    sdg, _ = make_sdg(fake)
    assert sdg.errors_on_connect == ['-113,"Undefined header;*OPT?"']
    assert sdg.get_errors() == []


def test_connect_timeout_is_unresponsive(make_sdg):
    fake = FakeSDG()

    def silent(cmd):
        raise visa.VisaIOError(visa.constants.StatusCode.error_timeout)

    fake.query = silent
    with pytest.raises(InstrumentUnresponsiveError, match="do not retry"):
        make_sdg(fake)


def test_connect_sets_newline_terminators(make_sdg):
    fake = FakeSDG()
    fake.write_termination = fake.read_termination = None
    make_sdg(fake)
    assert (fake.write_termination, fake.read_termination) == ("\n", "\n")


def test_disconnect_turns_both_outputs_off(make_sdg):
    sdg, fake = make_sdg()
    sdg.set_output_state_all(True)
    sdg.disconnect()
    assert [c.out for c in fake.ch.values()] == ["OFF", "OFF"] and fake.closed


def test_safe_state_never_raises(make_sdg):
    sdg, fake = make_sdg()

    def broken(cmd):
        raise visa.VisaIOError(visa.constants.StatusCode.error_timeout)

    fake.write = broken
    fake.query = broken
    sdg.safe_state()


def test_main_without_a_resource_prints_usage():
    assert sdg_module._main(["prog"]) == 2


def test_main_runs_read_only_and_leaves_outputs_off(monkeypatch, capsys):
    fake = FakeSDG()
    monkeypatch.setattr(sdg_module.visa, "ResourceManager", lambda: None)
    monkeypatch.setattr(sdg_module._SdgLink, "_open_resource", lambda self, rm, name: fake)
    assert sdg_module._main(["prog", "TCPIP::FAKE::INSTR"]) == 0
    out = capsys.readouterr().out
    assert "SDG3202X" in out and "CH1: SINE" in out
    assert not any(c.startswith("C1:BSWV") for c in _writes(fake))


# ---- the API surface -------------------------------------------------------------------------

_NODE_CLASSES = [
    sdg_module.Siglent_SDG3000X,
    sdg_module._Channel,
    sdg_module._Modulation,
    sdg_module._Sweep,
    sdg_module._Burst,
    sdg_module._Harmonics,
    sdg_module._Hopping,
    sdg_module._Multitone,
    sdg_module._Multipulse,
    sdg_module._Sequence,
    sdg_module._MultiChannel,
    sdg_module._Iq,
    sdg_module._File,
    sdg_module._Counter,
]
_NO_PREFIX = {"connect", "disconnect", "snapshot", "restore", "preserved_setup", "channel",
              "safe_state", "reset", "check_errors"}  # fmt: skip


def _public_methods():
    for cls in _NODE_CLASSES:
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if name.startswith("_") or func.__module__ != sdg_module.__name__:
                continue
            yield cls, name, func


def test_every_public_method_uses_a_standard_prefix():
    for cls, name, _ in _public_methods():
        assert name in _NO_PREFIX or name.split("_")[0] in ("set", "get", "meas", "cmd"), (
            f"{cls.__name__}.{name}"
        )


def test_every_public_method_has_a_short_first_docstring_line():
    for cls, name, func in _public_methods():
        doc = inspect.getdoc(func)
        assert doc, f"{cls.__name__}.{name} has no docstring"
        assert len(doc.splitlines()[0]) <= 90, f"{cls.__name__}.{name}: first line too long"


def test_every_setter_returns_a_value():
    for cls, name, func in _public_methods():
        if name.startswith("set_"):
            ret = inspect.signature(func).return_annotation
            assert ret not in (inspect.Signature.empty, None, "None"), f"{cls.__name__}.{name}"


def test_waveform_helpers_require_an_amplitude_limit():
    for name in ("set_waveform_sine", "set_waveform_square", "set_waveform_ramp",
                 "set_waveform_pulse", "set_waveform_arb", "set_waveform_arb_builtin",
                 "set_waveform_arb_user", "set_waveform_arb_file", "set_chirp", "set_chirp_arb"):  # fmt: skip
        param = inspect.signature(getattr(sdg_module._Channel, name)).parameters["ampl_limit"]
        assert param.kind is param.KEYWORD_ONLY and param.default is param.empty, name
        assert "on" not in inspect.signature(getattr(sdg_module._Channel, name)).parameters


def test_enums_hang_off_the_class():
    for name in ("Function", "Load", "Spacing", "TriggerSource", "ModulationShape", "Gate",
                 "RunMode", "IqWaveform", "Segment"):  # fmt: skip
        assert getattr(Siglent_SDG3000X, name) is getattr(sdg_module, name)
    assert issubclass(Function, ScpiEnum) and str(Function.SINE) == "SINE"


# ---- basic waveforms -------------------------------------------------------------------------


def test_sine_order_and_result(make_sdg):
    sdg, fake = make_sdg()
    state = sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0, offset=0.5, load=50)
    writes = _writes(fake)
    assert writes.index("C1:OUTP LOAD,50") < writes.index("C1:BSWV WVTP,SINE")
    bswv = [c for c in writes if c.startswith("C1:BSWV ")]
    assert bswv.index("C1:BSWV MAX_OUTPUT_AMP,4") < bswv.index("C1:BSWV FRQ,1000")
    assert bswv[-1].startswith("C1:BSWV AMP,")
    assert isinstance(state, WaveformState)
    assert (state.function, state.frequency, state.amplitude, state.offset) == (
        "SINE", 1000.0, 1.0, 0.5,
    )  # fmt: skip
    assert state.output_state is False and sdg.ch1.get_load() == 50.0
    assert "C1:OUTP ON" not in writes


def test_ampl_limit_refuses_before_any_write_at_50_ohm(make_sdg):
    sdg, fake = make_sdg()
    with pytest.raises(ValueError, match="SDG-34"):
        sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=1.5, load=50)
    assert _writes(fake) == []
    assert sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=1.5).amplitude == 1.0  # HIGHZ
    with pytest.raises(ValueError, match="positive"):
        sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=0)


def test_clamp_raises_in_a_helper(sdg):
    with pytest.raises(InstrumentError, match=r"frequency: requested 3e\+08"):
        sdg.ch1.set_waveform_sine(300e6, 1.0, ampl_limit=20.0)


def test_atomic_setter_returns_the_applied_value_and_warns(sdg):
    with pytest.warns(ValueSnappedWarning, match="SDG-01"):
        assert sdg.ch1.set_frequency(300e6) == 200e6


def test_injected_readback_fault(make_sdg):
    sdg, fake = make_sdg()
    fake.override = {"AMP": 0.9}
    with pytest.raises(InstrumentError, match="amplitude"):
        sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0)


def test_instrument_error_carries_the_command(make_sdg):
    sdg, fake = make_sdg()
    fake.fail_on = {"C1:BSWV FRQ,1000"}
    with pytest.raises(InstrumentQueryError, match="C1:BSWV FRQ,1000.*-222"):
        sdg.ch1.set_frequency(1000)


def test_output_on_that_reads_off_raises(make_sdg):
    sdg, fake = make_sdg()
    fake.stuck_output = True
    with pytest.raises(InstrumentError, match="overvoltage"):
        sdg.ch1.set_output_state(True)


def test_square_ramp_and_pulse_fields(sdg):
    assert sdg.ch1.set_waveform_square(1e3, 1.0, ampl_limit=4.0, duty=30.0).duty == 30.0
    assert sdg.ch1.set_waveform_ramp(1e3, 1.0, ampl_limit=4.0, symmetry=20.0).symmetry == 20.0
    p = sdg.ch1.set_waveform_pulse(
        1e3, 2.0, ampl_limit=4.0, offset=1.0, width=1e-4, leading_edge_time=1e-8,
        trailing_edge_time=2e-8,
    )  # fmt: skip
    assert (p.width, p.edge_time, p.trailing_edge_time) == (1e-4, 1e-8, 2e-8)
    with pytest.raises(ValueError):
        sdg.ch1.set_waveform_pulse(1e3, 1.0, ampl_limit=4.0, width=1e-4, duty=10.0)
    with pytest.raises(ValueError):
        sdg.ch1.set_waveform_pulse(1e3, 1.0, ampl_limit=4.0, edge_time=1e-8, leading_edge_time=1e-8)


def test_dc_and_a_getter_on_the_wrong_shape(sdg):
    state = sdg.ch1.set_waveform_dc(1.5)
    assert (state.function, state.offset, state.frequency) == ("DC", 1.5, 0.0)
    with pytest.raises(InstrumentError, match="frequency does not apply to the DC shape"):
        sdg.ch1.get_frequency()


def test_noise_and_its_bandwidth(sdg):
    noise = sdg.ch1.set_waveform_noise(0.1, bandwidth=1e6)
    assert (noise.stdev, noise.mean, noise.bandwidth) == (0.1, 0.0, 1e6)
    assert sdg.ch1.get_waveform().bandwidth == 1e6
    assert sdg.ch1._set_noise_bandwidth(None) is None
    with pytest.raises(InstrumentError, match="not NOISE"):
        sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0) and sdg.ch1.get_noise()


def test_prbs(sdg):
    state = sdg.ch1.set_waveform_prbs(1e6, 1.0, offset=0.5, length=9, differential=True,
                                      polynomial="X9+X5")  # fmt: skip
    assert state.bit_rate == 1e6 and state.length == 9 and state.differential
    assert state.polynomial == "X9+X5"
    assert sdg.ch1.get_waveform().frequency == 1e6


def test_dbm_needs_50_ohm(sdg):
    with pytest.raises(InstrumentError, match="dBm needs the load at 50 ohm"):
        sdg.ch1.set_amplitude_dbm(0)


def test_atomic_waveform_settings(sdg):
    ch = sdg.ch1
    assert ch.set_amplitude(2.0) == 2.0 and ch.get_amplitude() == 2.0
    assert ch.set_offset(0.1) == 0.1 and ch.get_offset() == 0.1
    assert ch.set_phase(45.0) == 45.0 and ch.get_phase() == 45.0
    assert ch.set_frequency(2e3) == 2e3 and ch.get_frequency() == 2e3


# ---- output page -----------------------------------------------------------------------------


def test_output_page_settings(sdg):
    ch = sdg.ch1
    assert ch.set_invert_state(True) is True and ch.get_invert_state() is True
    assert ch.set_combine_state(True) is True and ch.get_combine_state() is True
    assert ch.get_load() == math.inf and ch.set_load(Load.HIGHZ) == math.inf
    assert ch.get_amplitude_limit() == 20.0 and ch.set_amplitude_limit(5.0) == 5.0
    with pytest.raises(ValueError, match="50 ohm or Load.HIGHZ"):
        ch.set_load(75)


def test_noise_sum_and_filter(sdg):
    assert sdg.ch1.set_noise_sum(20.0) == 20.0 and sdg.ch1.set_noise_sum(None) is None
    assert sdg.ch1.set_filter_cutoff(20e6) == 2e7
    with pytest.raises(InstrumentError):
        sdg.ch1.set_filter_cutoff(1e6)  # clamps to 10 MHz (SDG-17)
    assert sdg.ch1.set_filter_cutoff(None) is None


def test_both_outputs(sdg):
    assert sdg.set_output_state_all(True) == [True, True]
    assert sdg.set_output_state_all(False) == [False, False]


def test_sync_out_one_channel(sdg):
    sdg.ch1.set_sync_state(True)
    assert sdg.ch2.set_sync_state(True) is True
    assert sdg.ch1.get_sync_state() is False and sdg.ch2.get_sync_type() == "CH2"
    assert sdg.ch2.set_sync_state(False) is False


def test_sync_out_refused_during_burst(sdg):
    sdg.ch1.set_burst(3, period=1e-3)
    with pytest.raises(InstrumentError, match="SDG-12"):
        sdg.ch1.set_sync_state(True)


# ---- modulation ------------------------------------------------------------------------------


def test_fm_helper_and_record(make_sdg):
    sdg, fake = make_sdg()
    state = sdg.ch1.set_fm(500.0, rate=100.0)
    assert "C1:MDWV FM,SRC,INT" in fake.log
    assert isinstance(state, ModulationState)
    assert (state.mode, state.level, state.rate) == ("FM", 500.0, 100.0)
    assert (state.source, state.shape) == (ModulationSource.INTERNAL, ModulationShape.SINE)
    assert sdg.ch1.mod.get_deviation() == 500.0 and sdg.ch1.mod.get_type() is ModulationType.FM


def test_modulation_setter_needs_modulation_on(sdg):
    with pytest.raises(InstrumentError, match=r"ch1\.mod\.set_state"):
        sdg.ch1.mod.set_depth(50)


def test_shape_change_while_modulating_is_loud_and_a_helper_clears_it(sdg):
    sdg.ch1.set_am(80.0, rate=100.0)
    with pytest.raises(InstrumentError, match="Modulation, sweep or burst"):
        sdg.ch1._set_function(Function.SQUARE)
    assert sdg.ch1.set_waveform_sine(2e3, 1.0, ampl_limit=4.0).function == "SINE"
    assert sdg.ch1.get_modulation().mode is None


def test_am_dsbsc_pm_and_keying(sdg):
    assert sdg.ch1.set_am(50.0, rate=1e3).level == 50.0
    assert sdg.ch1.set_dsbsc(rate=200.0).mode == "DSBSC"
    assert sdg.ch1.set_pm(90.0, rate=10.0).level == 90.0
    fsk = sdg.ch1.set_fsk(2e3, key_rate=100.0)
    assert (fsk.level, fsk.rate) == (2000.0, 100.0)
    assert sdg.ch1.set_ask(key_rate=50.0).rate == 50.0
    assert sdg.ch1.set_psk(key_rate=10.0, polarity=Polarity.NEGATIVE).mode == "PSK"
    assert sdg.ch1.mod.get_polarity() is Polarity.NEGATIVE


def test_pwm_needs_a_pulse_carrier(sdg):
    with pytest.raises(InstrumentError, match="pulse"):
        sdg.ch1.set_pwm(1e-5, rate=100.0)
    sdg.ch1.set_waveform_pulse(1e3, 1.0, ampl_limit=4.0, width=1e-4)
    assert sdg.ch1.set_pwm(1e-5, rate=100.0).mode == "PWM"


@pytest.mark.parametrize("kind", ["AM", "DSBSC", "FM", "PM", "ASK", "FSK", "PSK"])
def test_external_modulation(sdg, kind):
    sdg.ch1.set_waveform_sine(100e3, 1.0, ampl_limit=4.0)
    kw = {
        "FM": {"deviation": 5e3},
        "PM": {"deviation": 90.0},
        "FSK": {"hop_frequency": 50e3},
        "PSK": {"polarity": Polarity.NEGATIVE},
    }.get(kind, {})
    state = sdg.ch1.set_modulation_external(kind, **kw)
    assert state.mode == kind and state.source is ModulationSource.EXTERNAL


def test_external_pwm_needs_a_pulse(sdg):
    sdg.ch1.set_waveform_sine(100e3, 1.0, ampl_limit=4.0)
    with pytest.raises(InstrumentError, match="pulse"):
        sdg.ch1.set_modulation_external(ModulationType.PWM, deviation=1e-5)


def test_am_with_an_external_source_points_to_the_external_helper(sdg):
    with pytest.raises(ValueError, match="set_modulation_external"):
        sdg.ch1.set_am(50.0, rate=1e3, source=ModulationSource.EXTERNAL)


def test_modulation_input_scale(sdg):
    assert sdg.ch1.get_modulation_input_scale().mode is None
    sdg.ch1.set_am(50.0, rate=1e3)
    scale = sdg.ch1.get_modulation_input_scale()
    assert scale.unit == "%/V" and scale.per_volt == pytest.approx(100 / 6.1)
    sdg.ch1.set_fm(600.0, rate=10.0)
    assert sdg.ch1.get_modulation_input_scale().per_volt == pytest.approx(100.0)
    sdg.ch1.set_ask(key_rate=10.0)
    assert "TTL" in sdg.ch1.get_modulation_input_scale().note


def test_modulation_off(sdg):
    sdg.ch1.set_am(50.0, rate=1e3)
    assert sdg.ch1.set_modulation_off().mode is None
    assert not sdg.ch1.mod.get_state()


# ---- sweep -----------------------------------------------------------------------------------


def test_frequency_sweep_and_trigger_out(sdg):
    sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0)
    state = sdg.ch1.set_sweep(100.0, 100e3, 3e-3, spacing=Spacing.LOG,
                              trigger_out=TriggerOut.RISE, trigger_source=TriggerSource.MANUAL)  # fmt: skip
    assert isinstance(state, SweepState)
    assert (state.start, state.stop, state.time, state.spacing) == (100.0, 1e5, 3e-3, Spacing.LOG)
    assert sdg.ch1.sweep.get_trigger_out() is TriggerOut.RISE
    assert sdg.ch1.sweep.get_trigger_source() is TriggerSource.MANUAL


def test_sweep_needs_a_sweepable_carrier(sdg):
    sdg.ch1.set_waveform_pulse(1e3, 1.0, ampl_limit=4.0, width=1e-4)
    with pytest.raises(InstrumentError, match="sweep needs"):
        sdg.ch1.set_sweep(100.0, 1e3, 0.1)


def test_sweep_external_edge(sdg):
    sdg.ch1.set_sweep(1e3, 2e3, 10e-3, trigger_source="external", trigger_edge=Edge.FALL)
    assert sdg.ch1.sweep.get_trigger_source() is TriggerSource.EXTERNAL
    assert sdg.ch1.sweep.get_trigger_edge() is Edge.FALL


def test_amplitude_sweep_and_holds(sdg):
    sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0)
    state = sdg.ch1.set_sweep_amplitude(0.5, 2.0, 10e-3)
    assert (state.sweep_type, state.start, state.stop) == (SweepType.AMPLITUDE, 0.5, 2.0)
    assert sdg.ch1.sweep.set_hold_start(1e-3) == 1e-3
    assert sdg.ch1.sweep.set_return_time(2e-3) == 2e-3
    assert sdg.ch1.sweep.set_hold_end(4e-3) == 4e-3


def test_frequency_and_amplitude_sweep(sdg):
    state = sdg.ch1.set_sweep_frequency_amplitude(1e3, 2e3, 0.5, 1.5, 10e-3)
    assert state.start == 1e3
    assert sdg.ch1.sweep.get_type() is SweepType.FREQUENCY_AMPLITUDE
    assert sdg.ch1.get_sweep_amplitude().stop == 1.5


def test_sweep_and_burst_read_none_while_off(sdg):
    assert sdg.ch1.get_sweep() is None and sdg.ch1.get_burst() is None
    assert sdg.ch1.get_sweep_amplitude() is None


def test_sweep_atomics(sdg):
    sdg.ch1.set_sweep(1e3, 2e3, 1.0)
    sw = sdg.ch1.sweep
    assert sw.set_direction(Direction.UP_DOWN) is Direction.UP_DOWN
    assert sw.set_symmetry(30.0) == 30.0 and sw.set_steps(5) == 5
    assert sw.set_frequency_center(1.5e3) == 1.5e3 and sw.set_frequency_span(500.0) == 500.0
    sw.cmd_trigger()


# ---- burst -----------------------------------------------------------------------------------


def test_ncycle_and_gated_burst(sdg):
    sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0)
    state = sdg.ch1.set_burst(5, period=10e-3, trigger_out=TriggerOut.FALL)
    assert isinstance(state, BurstState)
    assert (state.cycles, state.period, state.mode) == (5.0, 0.01, BurstMode.NCYCLE)
    assert sdg.ch1.burst.get_trigger_out() is TriggerOut.FALL
    gated = sdg.ch1.set_burst_gated(polarity=Polarity.NEGATIVE)
    assert gated.mode is BurstMode.GATED
    assert sdg.ch1.burst.get_gate_polarity() is Polarity.NEGATIVE


def test_burst_external_edge_count_and_idle(sdg):
    state = sdg.ch1.set_burst(3, period=1e-3, trigger_source=TriggerSource.EXTERNAL,
                              trigger_edge=Edge.FALL, count=2, idle=BurstIdle.END)  # fmt: skip
    assert state.idle is BurstIdle.END
    assert sdg.ch1.burst.get_trigger_edge() is Edge.FALL and sdg.ch1.burst.get_count() == 2


def test_burst_count_needs_an_external_trigger(sdg):
    sdg.ch1.set_burst(3, period=1e-3)
    with pytest.raises(InstrumentError, match="SDG-13"):
        sdg.ch1.burst.set_count(2)


def test_infinite_burst_and_bad_cycles(sdg):
    assert sdg.ch1.set_burst(math.inf, period=1e-3).cycles == math.inf
    with pytest.raises(ValueError):
        sdg.ch1.set_burst(2.5, period=1e-3)


# ---- chirp -----------------------------------------------------------------------------------


def test_chirp_on_the_sweep_with_a_period(sdg):
    state = sdg.ch1.set_chirp(1e3, 1e4, 10e-3, ampl=1.0, ampl_limit=4.0,
                              repeat=ChirpRepeat.PERIOD, period=50e-3, load=50)  # fmt: skip
    assert (state.method, state.start, state.stop, state.duration) == ("sweep", 1e3, 1e4, 0.01)
    assert sdg.ch1.sweep.get_hold_end() == pytest.approx(0.04)
    assert state.output_state is False
    with pytest.raises(ValueError, match="period"):
        sdg.ch1.set_chirp(1e3, 1e4, 10e-3, ampl=1.0, ampl_limit=4.0, repeat=ChirpRepeat.PERIOD)


def test_chirp_as_an_arb(make_sdg):
    sdg, fake = make_sdg()
    state = sdg.ch1.set_chirp_arb(1e3, 1e4, 10e-3, ampl=1.0, ampl_limit=4.0, points=4096)
    assert state.method == "arb" and state.points == 4096 and "chirp_c1" in fake.user
    assert sdg.ch1.get_frequency() == pytest.approx(100.0)


# ---- hopping ---------------------------------------------------------------------------------


def test_hopping_list_writes_the_tables(make_sdg):
    sdg, fake = make_sdg()
    state = sdg.ch1.hopping.set_list([1e6, 2e6, 3e6], dwell=1e-3, ampl=1.0, ampl_limit=4.0,
                                     order=[2, 0, 1])  # fmt: skip
    assert state.enabled and state.hop_type is HopType.LIST and state.dwell == 0.001
    assert "C1:FHOP:MFLIst 1,1000000" in fake.log and "C1:FHOP:AFLIst 3,3000000" in fake.log
    assert "C1:FHOP:MOLIst 1,3" in fake.log
    assert sdg.ch1.hopping.set_frequencies([5e6]) == [5e6]


def test_hopping_random_with_avoid(sdg):
    state = sdg.ch1.hopping.set_random(start=1e6, stop=10e6, step=1e6, dwell=1e-3, ampl=1.0,
                                       ampl_limit=4.0, avoid=[(3e6, 4e6)])  # fmt: skip
    assert state.hop_type is HopType.RANDOM and state.avoid_enabled and state.stop == 1e7


def test_hopping_random_list_and_prbs_order(sdg):
    state = sdg.ch1.hopping.set_random_list([1e6, 2e6], dwell=1e-3, ampl=1.0, ampl_limit=4.0)
    assert state.hop_type is HopType.RANDOM_LIST and state.prbs_order == 7
    assert sdg.ch1.hopping.set_prbs_order(9) == 9
    sdg.ch1.hopping.cmd_save("x")
    sdg.ch1.hopping.cmd_recall("x")


# ---- multitone, multipulse -------------------------------------------------------------------


def test_multitone_replaces_defaults_and_switches_mode(sdg):
    state = sdg.ch1.multitone.set_tone_list([1e6, 2e6, 5e6], ampl=1.0)
    assert state.tones == (1e6, 2e6, 5e6) and state.running
    assert sdg.ch1.get_mode() is ChannelMode.MULTITONE
    sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0)
    assert sdg.ch1.get_mode() is ChannelMode.AFG


def test_multitone_ignored_sample_rate_is_loud(sdg):
    with pytest.raises(InstrumentError, match="chooses the rate"):
        sdg.ch1.multitone.set_tone_list([1e6], ampl=1.0, sample_rate=50e6)


def test_multitone_notches_and_range(sdg):
    sdg.ch1.multitone.set_state(True)
    assert sdg.ch1.multitone.set_notches([(1.5e3, 2.5e3)]) == [(1500.0, 2500.0)]
    state = sdg.ch1.multitone.set_tone_range(1e3, 5e3, 5, ampl=1.0, notches=[(2.5e3, 3.5e3)])
    assert state.tones == (1e3, 2e3, 4e3, 5e3)


def test_multipulse_train(sdg):
    state = sdg.ch1.multipulse.set_train([(1e-3, 2e-3), (0.5e-3, 1e-3), (10e-6, 10e-6)],
                                         level_high=1.0, level_low=0.0)  # fmt: skip
    assert state.pulses == ((1e-3, 2e-3), (5e-4, 1e-3), (1e-5, 1e-5)) and state.running
    assert state.sample_rate is None
    assert sdg.ch1.get_mode() is ChannelMode.MULTIPULSE
    assert sdg.ch1.multipulse.set_sample_rate(1e8) == 1e8
    with pytest.raises(ValueError):
        sdg.ch1.multipulse.set_pulses([(1e-3, 1e-3)])


# ---- sequence --------------------------------------------------------------------------------


def test_sequence_one_segment(sdg):
    state = sdg.ch1.sequence.set_segments([Segment("Square", length=32768, loops=2)],
                                          sample_rate=10e6)  # fmt: skip
    assert state.running and state.sample_rate == 1e7 and state.segments[0].waveform == "square"
    assert state.run_mode is RunMode.CONTINUOUS


def test_sequence_rules(sdg):
    seq = sdg.ch1.sequence
    with pytest.raises(ValueError, match="ADVANCED"):
        seq.set_segments([Segment("Sine", length=1024),
                          Segment("Square", length=1024, wait=WaitEvent.MANUAL)],
                         sample_rate=10e6)  # fmt: skip
    with pytest.raises(ValueError, match="next 3"):
        seq.set_segments([Segment("Sine", length=1024), Segment("Square", length=1024, next=3)],
                         sample_rate=10e6)  # fmt: skip
    with pytest.raises(ValueError, match="set_segments"):
        seq.set_run_mode(RunMode.STEP)
    for bad in (32, 100):
        with pytest.raises(ValueError, match="DS p.11"):
            seq.set_segments([Segment("Sine", length=bad)], sample_rate=10e6)
    with pytest.raises(ValueError, match="1024"):
        seq.set_segments([Segment("Sine", length=1024)] * 1025, sample_rate=10e6)


def test_sequence_segment_count_refuses_while_running(sdg):
    sdg.ch1.sequence.set_segments([Segment("Sine", length=1024)], sample_rate=1e6)
    with pytest.raises(InstrumentError, match="SDG-05"):
        sdg.ch1.sequence.get_segment_count()
    assert sdg.ch1.sequence.get_setup().segments is None


def test_sequence_atomics(make_sdg):
    sdg, fake = make_sdg()
    seq = sdg.ch1.sequence
    assert seq.set_hold_user_level(0.5) == 0.5 and seq.get_hold().name == "USER"
    fake.ch["C1"].awg["TRIGger:SOURce"] = "TIME"  # written TIM, read TIME (MEAS)
    assert seq.get_trigger_source() is SequenceTriggerSource.TIMER
    assert seq.set_trigger_source(SequenceTriggerSource.EXTERNAL) is SequenceTriggerSource.EXTERNAL
    with pytest.raises(ValueError, match="read back only"):
        seq.set_trigger_source(SequenceTriggerSource.FREE_RUN)
    assert seq.get_scale() == 100.0
    seq.cmd_save("s1")
    seq.cmd_recall("s1")


_TEMPLATE = """<root>
  <awg_ch_setting>
    <run_mode>0</run_mode>
    <trigger_source>1</trigger_source>
    <trigger_edgeA>0</trigger_edgeA>
    <trigger_out>3</trigger_out>
    <sample_rate>10000000</sample_rate>
  </awg_ch_setting>
  <segment_list>
    <segment_end_numb>0</segment_end_numb>
    <segment_count>1</segment_count>
    <segment_0><waveform>Sine</waveform><loop_count>1</loop_count><wait_event>0</wait_event>\
<goto_numb>0</goto_numb><amplitude>2</amplitude><offset>0</offset><high_level>1</high_level>\
<low_level>-1</low_level></segment_0>
  </segment_list>
</root>"""


def test_awgx_build_and_code_check():
    """The file builder on a template reduced to the tags the original map names (not a saved file)."""
    body = sdg_module._awgx_segment_xml(_TEMPLATE, 0)
    rows = [
        {"body": body, "loops": 2, "wait": "auto", "next": 1, "amplitude": 1.0, "offset": 0.1},
        {"body": body, "loops": 3, "wait": "external", "next": 0, "amplitude": None,
         "offset": None},
    ]  # fmt: skip
    x = sdg_module._awgx_build(_TEMPLATE, rows, run_mode="advanced")
    sdg_module._awgx_check_codes(x, _TEMPLATE)
    assert "<segment_count>2</segment_count>" in x and "<run_mode>5</run_mode>" in x
    assert "<segment_end_numb>1</segment_end_numb>" in x and "<high_level>0.6</high_level>" in x
    seg1 = sdg_module._awgx_segment_xml(x, 1)
    assert "<wait_event>3</wait_event>" in seg1 and "<loop_count>3</loop_count>" in seg1
    for bad in ("<wait_event>3</wait_event>", "<run_mode>5</run_mode>"):
        crash = x.replace(bad, bad.replace(">3<", ">4<").replace(">5<", ">2<"), 1)
        with pytest.raises(ValueError, match="SDG-07"):
            sdg_module._awgx_check_codes(crash, _TEMPLATE)
    with pytest.raises(ValueError, match="differs"):
        sdg_module._awgx_check_codes(
            x.replace("<trigger_out>3</trigger_out>", "<trigger_out>9</trigger_out>"), _TEMPLATE
        )
    assert sdg_module._awgx_test_xml(sdg_module._awgx_pack("s", x.encode())) == x


def test_awgx_check_code_alphabet():
    code = sdg_module._awgx_check_code(b"<x/>")
    assert len(code) == 16 and re.fullmatch(r"[0-9a-z]+", code)
    assert not set(code) & set("01lo")


# ---- two channels, IQ, system ----------------------------------------------------------------


def test_quadrature_and_phase_lock(make_sdg):
    sdg, fake = make_sdg()
    state = sdg.multichannel.set_quadrature(1e3, 1.0, ampl_limit=4.0)
    assert state.phase_locked and state.ch2_phase == 90.0
    assert state.ch1.function == "SINE" and state.ch2.frequency == 1e3
    assert sdg.get_output_state_all() == [False, False]  # no helper energizes
    assert sdg.multichannel.set_phase_locked_state(False) is False


def test_differential_inverts_ch2(sdg):
    sdg.multichannel.set_differential(1e3, 1.0, ampl_limit=4.0, function=Function.SQUARE)
    assert sdg.ch2.get_invert_state() is True and sdg.ch1.get_invert_state() is False
    with pytest.raises(ValueError):
        sdg.multichannel.set_differential(1e3, 1.0, ampl_limit=4.0, function=Function.DC)


def test_coupling_and_track_with_the_stray_token(sdg):
    state = sdg.multichannel.set_coupling(frequency_ratio=2.0, amplitude_deviation=0.5)
    assert state == CouplingState(frequency_ratio=2.0, amplitude_deviation=0.5)
    assert sdg.multichannel.set_track_state(True) is True
    assert sdg.multichannel.get_track_state() is True
    with pytest.raises(ValueError):
        sdg.multichannel.set_coupling(frequency_ratio=2.0, frequency_deviation=1.0)


def test_copy_between_channels(make_sdg):
    sdg, fake = make_sdg()
    sdg.ch1.set_waveform_square(5e3, 1.0, ampl_limit=4.0)
    assert sdg.multichannel.cmd_copy_ch1_to_ch2().function == "SQUARE"
    assert "PACP C2,C1" in fake.log
    sdg.multichannel.cmd_copy_ch2_to_ch1()
    assert "PACP C1,C2" in fake.log


def test_iq_builtin_and_off(sdg):
    state = sdg.iq.set_waveform_builtin(IqWaveform.QAM_16, symbol_rate=1e6, vrms=0.2)
    assert (state.enabled, state.waveform, state.symbol_rate, state.vrms) == (
        True, "16QAM", 1e6, 0.2,
    )  # fmt: skip
    assert sdg.ch2.get_mode() is ChannelMode.IQ
    assert sdg.iq.get_waveform_info().modulation == "16QAM"
    assert sdg.iq.set_state(False) is False
    assert sdg.ch1.get_mode() is ChannelMode.AFG and sdg.ch2.get_mode() is ChannelMode.AFG
    assert sdg.iq.set_waveform_builtin("8PSK", symbol_rate=1e5, vrms=0.1).waveform == "8PSK"


def test_iq_user_file_must_exist_and_adjustments(sdg):
    with pytest.raises(InstrumentError, match="EasyIQ"):
        sdg.iq.set_waveform_user("missing", symbol_rate=1e6, vrms=0.2)
    assert sdg.iq.set_amplitude_dbm(-10.0) == pytest.approx(-10.0)
    assert sdg.iq.set_adjust_angle_q(5.0) == 5.0
    assert sdg.iq.get_sample_rate_limits().maximum == 2e8
    sdg.iq.cmd_trigger()


def test_reference_clock_and_utility_page(sdg):
    assert sdg.set_reference_clock_external() is Reference.EXTERNAL
    assert sdg.set_reference_clock_internal() is Reference.INTERNAL
    assert sdg.set_reference_clock_out_state(True) is True
    assert sdg.get_beeper_state() is True and sdg.get_front_panel_keys_state() is True
    assert sdg.set_beeper_state(True) is True and sdg.set_front_panel_keys_state(True) is True
    assert sdg.get_lan_address() == "192.0.2.10"


# ---- arb -------------------------------------------------------------------------------------


def test_arb_upload_selects_by_path(make_sdg):
    sdg, fake = make_sdg()
    state = sdg.ch1.set_waveform_arb([math.sin(i / 10) for i in range(100)], 2e3, 1.0, name="t1",
                                     ampl_limit=4.0)  # fmt: skip
    assert state.frequency == 2000.0 and state.function == "ARB"
    assert sdg.ch1.get_arb_name() == "t1.bin"
    assert 'C1:ARWV NAME,"Local/t1.bin"' in fake.log
    assert sdg.get_arb_names_user() == ["t1"]


def test_arb_upload_over_the_socket(make_sdg):
    """SDG-04 withdrawn 2026-10-16: a raw-socket upload with 0x0A bytes arrives complete."""
    sdg, _ = make_sdg(resource="TCPIP::192.0.2.1::5025::SOCKET")
    state = sdg.ch1.set_waveform_arb([0, 1, 0.5, -1], 1.0, 1.0, name="t", ampl_limit=4.0)
    assert state.function == "ARB"


def test_arb_builtin_user_and_missing(sdg):
    assert sdg.get_arb_names_builtin() == {"Sine": 0, "StairUp": 2, "Cardiac": 26}
    state = sdg.ch1.set_waveform_arb_builtin("Cardiac", 1e3, 1.0, ampl_limit=4.0)
    assert state.function == "ARB" and sdg.ch1.get_arb_name() == "Cardiac"
    with pytest.raises(InstrumentError, match="Is it stored"):
        sdg.ch1.set_waveform_arb_user("nope", 1e3, 1.0, ampl_limit=4.0)


# ---- harmonics -------------------------------------------------------------------------------


def test_harmonics(sdg):
    sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0)
    state = sdg.ch1.harmonics.set_levels({3: -10.0, 5: (-20.0, 90.0)})
    assert state.enabled and state.levels[0].dbc == -10.0 and state.levels[1].phase == 90.0
    assert sdg.ch1.harmonics.set_state(False) is False
    assert sdg.ch1.harmonics.get_setup().enabled is False
    with pytest.raises(ValueError):
        sdg.ch1.harmonics.set_level(3, dbc=-10.0, vpp=0.1)


def test_harmonics_need_a_sine(sdg):
    sdg.ch1.set_waveform_square(1e3, 1.0, ampl_limit=4.0)
    with pytest.raises(InstrumentError, match="sine carrier"):
        sdg.ch1.harmonics.set_state(True)


# ---- reset, snapshot, restore ----------------------------------------------------------------


def test_reset_keeps_the_clock_source(make_sdg):
    sdg, fake = make_sdg()
    assert sdg.set_reference_clock_external() is Reference.EXTERNAL
    sdg.cmd_reset()  # first reset: *RST, clock written back, default setup saved
    assert sdg.get_reference_clock() is Reference.EXTERNAL
    assert sdg.file.get_size(sdg_module._DEFAULT_SETUP.format(clock="external"))
    n = len(fake.log)
    sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0)
    sdg.cmd_reset()  # later resets: recall, no *RST
    assert "*RST" not in fake.log[n:] and any("LOAD:XML" in c for c in fake.log[n:])
    assert sdg.get_reference_clock() is Reference.EXTERNAL
    assert sdg.get_output_state_all() == [False, False]


def test_snapshot_round_trip_restores_outputs_last(make_sdg):
    sdg, fake = make_sdg()
    sdg.ch1.set_waveform_square(2e3, 1.5, ampl_limit=20.0, offset=0.25, duty=30.0, load=50)
    sdg.ch1.set_output_state(True)
    snap = sdg.snapshot()
    assert snap["setup_file"] in fake.fs and snap["outputs"] == [True, False]
    sdg.cmd_reset()
    sdg.restore(snap)
    w = sdg.ch1.get_waveform()
    assert (w.function, w.frequency, w.amplitude, w.offset, w.duty) == (
        "SQUARE", 2000.0, 1.5, 0.25, 30.0,
    )  # fmt: skip
    assert sdg.ch1.get_load() == 50.0 and sdg.get_output_state_all() == [True, False]


def test_restore_raises_on_a_difference_with_outputs_off(make_sdg):
    sdg, fake = make_sdg()
    sdg.ch1.set_output_state(True)
    snap = sdg.snapshot()
    fake.setups[snap["setup_file"]]["ch"]["C1"].p["FRQ"] = 5e3
    with pytest.raises(InstrumentError, match="CH1 waveform"):
        sdg.restore(snap)
    assert sdg.get_output_state_all() == [False, False]


def test_preserved_setup_restores_and_deletes_the_file(make_sdg):
    sdg, fake = make_sdg()
    with sdg.preserved_setup():
        files = [k for k in fake.fs if "drv_snapshot_" in k]
        sdg.ch1.set_frequency(5e3)
    assert sdg.ch1.get_frequency() == 1000.0
    assert files and not any(k in fake.fs for k in files)


# ---- files -----------------------------------------------------------------------------------


def test_file_write_read_and_odd_length(make_sdg):
    sdg, fake = make_sdg()
    assert sdg.file.cmd_write("Local/x.bin", b"abc") == 3
    assert sdg.file.get_file("Local/x.bin") == b"ab\x00"  # SDG-18
    assert "Local/drv_files_rd.bin" not in fake.fs
    with pytest.raises(InstrumentError, match="no file"):
        sdg.file.get_file("Local/none.bin")


def test_file_copy_move_delete_and_folders(sdg):
    f = sdg.file
    f.cmd_write("Local/a.bin", b"1234")
    f.cmd_copy("Local/a.bin", "Local/b.bin")  # false -108 tolerated (SDG-19)
    f.cmd_move("Local/b.bin", "Local/c.bin")
    assert f.get_size("Local/b.bin") is None and f.get_size("Local/c.bin") == 4
    f.cmd_delete("Local/c.bin")
    f.cmd_make_directory("Local/sub")
    assert f.get_folder_exists("Local/sub")
    f.cmd_remove_directory("Local/sub")
    assert not f.get_folder_exists("Local/sub")
    cat = f.get_catalog()
    assert cat.free == 7000000 and [e.name for e in cat.entries] == ["a.bin"]
    assert f.get_drives() == ["Local"]
    with pytest.raises(ValueError):
        f.get_size("nofolder")


def test_file_arb_csv_layout(make_sdg):
    sdg, fake = make_sdg()
    sdg.file.cmd_write_arb_csv("Local/w.csv", [0.0, 1.0], freq=1e3, ampl=2.0)
    text = fake.fs["Local/w.csv"].decode()
    lines = text.split("\r\n")
    assert lines[0] == "data length,2" and lines[12] == "xpos,value" and lines[5:12] == [""] * 7


def test_setup_save_and_recall(sdg):
    assert sdg.file.cmd_save_setup("Local/s") > 0
    sdg.file.cmd_recall_setup("Local/s.xml")
    with pytest.raises(InstrumentError, match="no setup file"):
        sdg.file.cmd_recall_setup("Local/missing")


# ---- counter ---------------------------------------------------------------------------------


def test_counter_frequency_mode_and_pause_inversion(make_sdg):
    sdg, fake = make_sdg()
    state = sdg.counter.set_frequency_mode(coupling=Coupling.DC, trigger_level=0.25,
                                           reference=1000.0)  # fmt: skip
    assert (state.enabled, state.mode, state.running, state.coupling) == (
        True, CounterMode.FREQUENCY, True, Coupling.DC,
    )  # fmt: skip
    assert (state.trigger_level, state.reference) == (0.25, 1000.0)
    assert sdg.counter.set_run_state(False) is False
    assert fake.counter["CONFig:PAUSe"] == "OFF"  # SDG-27: OFF holds
    assert sdg.counter.meas_width_positive() == pytest.approx(250e-6)
    assert sdg.counter.meas_width_negative() == pytest.approx(750e-6)
    assert sdg.counter.cmd_wait_measurements(3, poll=0) == 3
    stats = sdg.counter.meas_statistics()
    assert stats.count == 3 and stats.mean == 1000.0
    assert sdg.counter.meas_frequency() == 1000.0 and sdg.counter.meas_deviation() == 1.5


def test_counter_wait_times_out(sdg, monkeypatch):
    monkeypatch.setattr(sdg_module.time, "sleep", lambda s: None)
    with pytest.raises(InstrumentTimeoutError, match="3 of 5"):
        sdg.counter.cmd_wait_measurements(5, timeout=0.0)


def test_counter_totalizer_gate(sdg):
    state = sdg.counter.set_totalizer_mode(edge=Edge.FALL, gate=Gate.LEVEL_NEGATIVE)
    assert (state.mode, state.edge, state.gate) == (
        CounterMode.TOTALIZER, Edge.FALL, Gate.LEVEL_NEGATIVE,
    )  # fmt: skip
    assert sdg.counter.set_totalizer_gate(Gate.AFTER_RISE) is Gate.AFTER_RISE
    assert sdg.counter.set_totalizer_gate(Gate.OFF) is Gate.OFF
    assert sdg.counter.meas_total() == 1234
    assert sdg.counter.set_state(False) is False


# ---- sequence file load: trigger source (SDG-37) ---------------------------------------------


def _file_route(monkeypatch, seq, fake, keep_file_value):
    """Stand in for the file load: it sets the trigger source from the file (SDG-37)."""

    def load(segments, mode, name, user):
        fake.ch["C1"].awg["TRIGger:SOURce"] = "EXT"

    monkeypatch.setattr(seq, "_load_file", load)
    monkeypatch.setattr(seq, "get_segments", lambda: [])
    monkeypatch.setattr(seq, "_compare", lambda segments, written: None)
    if keep_file_value:
        write = fake.write

        def drop_after_load(cmd, **kwargs):
            if "TRIGger:SOURce" in cmd and fake.ch["C1"].awg["TRIGger:SOURce"] == "EXT":
                return len(cmd)
            return write(cmd, **kwargs)

        monkeypatch.setattr(fake, "write", drop_after_load)


def test_sequence_file_load_sets_the_trigger_source_again(make_sdg, monkeypatch):
    sdg, fake = make_sdg()
    seq = sdg.ch1.sequence
    _file_route(monkeypatch, seq, fake, keep_file_value=False)
    segments = [Segment("Sine", length=1024), Segment("Square", length=1024)]
    seq.set_segments(segments, sample_rate=1e6, run_mode="triggered", trigger_source="manual")
    assert fake.ch["C1"].awg["TRIGger:SOURce"] == "MAN"


def test_sequence_file_load_raises_if_the_trigger_source_stays_wrong(make_sdg, monkeypatch):
    sdg, fake = make_sdg()
    seq = sdg.ch1.sequence
    _file_route(monkeypatch, seq, fake, keep_file_value=True)
    segments = [Segment("Sine", length=1024), Segment("Square", length=1024)]
    with pytest.raises(InstrumentError, match="after the file load"):
        seq.set_segments(segments, sample_rate=1e6, run_mode="triggered", trigger_source="manual")


def test_arb_codes_span_the_full_range(make_sdg, monkeypatch):
    sdg, fake = make_sdg()
    blocks = []
    write_raw = fake.write_raw

    def keep(data):
        blocks.append(data)
        return write_raw(data)

    monkeypatch.setattr(fake, "write_raw", keep)
    sdg.ch1.set_waveform_arb([0.0, 0.5, 1.0], 1e3, 1.0, ampl_limit=4.0, name="x")
    block = next(b for b in blocks if b"WVDT WVNM,x" in b)
    data = block.split(b"WAVEDATA,", 1)[1].rstrip(b"\n")
    assert np.frombuffer(data, "<i2").tolist() == [-32767, 0, 32767]


def test_harmonic_type_while_off_names_the_cause(make_sdg, monkeypatch):
    sdg, _ = make_sdg()
    monkeypatch.setattr(sdg.ch1.harmonics, "_raw", lambda: {"HARMSTATE": "OFF"})
    with pytest.raises(InstrumentError, match="harmonics are off"):
        sdg.ch1.harmonics.get_type()
