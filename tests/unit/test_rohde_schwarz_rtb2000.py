# SPDX-License-Identifier: Apache-2.0
# Copyright: see the NOTICE file.
"""Hardware-free tests for the Rohde & Schwarz RTB2000 driver.

Ported from the original repository (``tests/test_rtb2000.py``) onto the
renamed API, and extended for the port: the node tree built at connect, the
option gating, the numbers-only input rule, the read-back on every setter,
the generator outputs that a config helper never turns on, and the snapshot
and restore.

The fake (``tests/mocks/fake_rtb2000.py``) is lenient: it echoes what was
written. These tests prove the driver's command sequence and parsing, not
the scope's behaviour. Bench facts stay with their quirk IDs in the driver.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re

import numpy as np
import pytest
import pyvisa as visa

import drivers.rohde_schwarz_rtb2000 as rtb_module
from common.errors import (
    InstrumentError,
    InstrumentQueryError,
    InstrumentTimeoutError,
    InstrumentUnresponsiveError,
    NotConnectedError,
)
from common.node import Node
from common.state import WaveformState
from drivers.rohde_schwarz_rtb2000 import (
    AcquireMode,
    AcquisitionSettings,
    AcquisitionState,
    AfgBurstSettings,
    AuxOutSource,
    BusFrame,
    Capture,
    ChannelSettings,
    CountDirection,
    Coupling,
    FftWindow,
    GridStyle,
    HistorySettings,
    Hysteresis,
    Interpolation,
    Load,
    LogicActivity,
    LogicPodSettings,
    MaskCounts,
    MathOperation,
    MeasStatistics,
    MeasType,
    PatternGenFunction,
    PatternLevel,
    Polarity,
    QuickMeasurement,
    RohdeSchwarz_RTB2000,
    Slope,
    TimeReference,
    TriggerMode,
    TriggerOutSettings,
    UartLine,
    UartTriggerMode,
    VideoField,
    VideoStandard,
    WidthCondition,
    XySettings,
)
from tests.mocks.fake_rtb2000 import IDN, IDN_2CH, OPTIONS, FakeRTB

RESOURCE = "TCPIP::FAKE::5025::SOCKET"


@pytest.fixture
def make_scope(monkeypatch):
    """Connect a driver to a FakeRTB. Returns (driver, fake)."""
    monkeypatch.setattr(rtb_module.visa, "ResourceManager", lambda: None)
    for name in ("_RECORD_LENGTH_POLL_S", "_RESULT_POLL_S", "_DVM_POLL_S", "_ROLL_MARGIN_S"):
        monkeypatch.setattr(rtb_module, name, 0.0)

    def factory(idn: str = IDN, options: str = OPTIONS):
        fake = FakeRTB(idn, options)
        scope = RohdeSchwarz_RTB2000(RESOURCE)
        monkeypatch.setattr(scope.link, "_open_resource", fake.open_resource)
        scope.connect()
        fake.log.clear()
        return scope, fake

    return factory


@pytest.fixture
def scope(make_scope):
    return make_scope()


def _patch_write(monkeypatch, fake: FakeRTB, drop_prefix: str) -> None:
    """Make the fake log, but not apply, every write that starts with ``drop_prefix``."""
    original = fake.write

    def write(cmd: str, **kwargs) -> int:
        if cmd.startswith(drop_prefix):
            fake.log.append(cmd)
            return len(cmd)
        return original(cmd, **kwargs)

    monkeypatch.setattr(fake, "write", write)


# ------------------------------------------------------------- construction


def test_object_before_connection_does_no_io():
    s = RohdeSchwarz_RTB2000("TCPIP::192.0.2.1::5025::SOCKET")
    assert not s.link.is_connected
    assert hasattr(s, "ch4") and hasattr(s, "trig") and hasattr(s, "afg")
    with pytest.raises(NotConnectedError):
        s.link.visa  # noqa: B018 - the property access is the test


def test_connect_reads_identity_and_options(scope):
    s, _ = scope
    assert s.model == "RTB2004" and s.firmware == "03.000"
    assert s.serial == "1333.1005k04/100000"
    assert "K15" in s.options and s.num_channels == 4
    assert [ch.n for ch in s._channels()] == [1, 2, 3, 4]
    assert s.errors_on_connect == []


def test_link_sets_terminators_and_sends_no_device_clear(scope):
    _, fake = scope
    assert fake.write_termination == "\n" and fake.read_termination == "\n"
    assert fake.cleared == 0


def test_two_channel_model_has_no_ch3_or_ch4(make_scope):
    s, _ = make_scope(idn=IDN_2CH)
    assert s.model == "RTB2002"
    assert not hasattr(s, "ch3") and not hasattr(s, "ch4")
    with pytest.raises(ValueError, match="not a channel"):
        s.channel(3)


def test_no_options_means_no_option_nodes(make_scope):
    s, _ = make_scope(options="0")
    assert s.options == []
    for name in ("afg", "pattern", "logic", "history", "bus"):
        assert not hasattr(s, name), name


def test_bus_has_only_the_licensed_protocols(make_scope):
    s, _ = make_scope(options="K2")
    assert hasattr(s.bus, "set_uart") and hasattr(s.bus, "set_uart_trigger")
    for absent in ("set_spi", "set_i2c", "set_can", "set_lin", "set_parallel"):
        assert not hasattr(s.bus, absent), absent
    s, _ = make_scope(options="K1,K3")
    assert hasattr(s.bus, "set_spi") and hasattr(s.bus, "set_can")
    assert not hasattr(s.bus, "set_uart") and not hasattr(s.bus, "set_parallel")
    assert not hasattr(s, "logic")


def test_wrong_instrument_is_refused(monkeypatch):
    monkeypatch.setattr(rtb_module.visa, "ResourceManager", lambda: None)
    fake = FakeRTB(idn="TEKTRONIX,MSO58,C012345,CF:91.1CT FV:2.0")
    s = RohdeSchwarz_RTB2000(RESOURCE)
    monkeypatch.setattr(s.link, "_open_resource", fake.open_resource)
    with pytest.raises(InstrumentError, match="does not scan"):
        s.connect()
    assert not s.link.is_connected


def test_identity_timeout_raises_unresponsive(monkeypatch):
    monkeypatch.setattr(rtb_module.visa, "ResourceManager", lambda: None)
    fake = FakeRTB()

    def query(cmd: str, **kwargs) -> str:
        raise visa.VisaIOError(visa.constants.StatusCode.error_timeout)

    monkeypatch.setattr(fake, "query", query)
    s = RohdeSchwarz_RTB2000(RESOURCE)
    monkeypatch.setattr(s.link, "_open_resource", fake.open_resource)
    with pytest.raises(InstrumentUnresponsiveError, match="do not retry"):
        s.connect()


# ------------------------------------------------------------- safe state


def test_disconnect_turns_generator_and_pattern_outputs_off(scope):
    s, fake = scope
    s.afg.set_output_state(True)
    s.pattern.set_output_state(True)
    s.disconnect()
    assert "WGEN:OUTP OFF" in fake.log and "PGEN:PATT:STAT OFF" in fake.log
    assert fake.s["WGEN:OUTP"] == "OFF" and fake.s["PGEN:PATT:STAT"] == "OFF"
    assert fake.closed


def test_safe_state_without_b6_leaves_the_generator_alone(make_scope):
    s, fake = make_scope(options="K1")
    s.disconnect()
    assert not any(c.startswith(("WGEN", "PGEN")) for c in fake.log)


def test_safe_state_never_raises(scope):
    s, fake = scope

    def broken(cmd: str, **kwargs) -> int:
        raise visa.VisaIOError(visa.constants.StatusCode.error_timeout)

    fake.write = broken
    s.safe_state()


# ------------------------------------------------------------- input rules


def test_numbers_only_no_si_text(scope):
    s, _ = scope
    with pytest.raises(TypeError, match="1e3"):
        s.ch1.set_scale("200m")
    with pytest.raises(TypeError):
        s.horz.set_scale(True)
    with pytest.raises(TypeError):
        s.afg.set_frequency("1k")


def test_numbers_go_out_at_12_significant_digits(scope):
    s, fake = scope
    s.trig.set_level(0.123456789012345, source="ch1")
    assert "TRIG:A:LEV1 0.123456789012" in fake.log


def test_enums_are_attached_and_take_names_or_tokens(scope):
    s, _ = scope
    assert s.Coupling is Coupling and s.MeasType is MeasType and s.Load is Load
    assert s.ch1.set_coupling("ac") is Coupling.AC
    assert s.ch1.set_coupling("DCL") is Coupling.DC
    with pytest.raises(ValueError, match="use one of"):
        s.ch1.set_coupling("fifty")


def test_checked_write_raises_with_the_scope_error(scope):
    s, fake = scope
    fake.refuse["CHAN1:SCAL"] = '-222,"Data out of range"'
    with pytest.raises(InstrumentQueryError, match="-222"):
        s.ch1.set_scale(100.0)


def test_read_back_mismatch_raises(scope, monkeypatch):
    s, fake = scope
    _patch_write(monkeypatch, fake, "TIM:SCAL")
    with pytest.raises(InstrumentError, match="requested 0.005"):
        s.horz.set_scale(5e-3)


# ------------------------------------------------------------- vertical


def test_channel_setters_are_atomic_and_return_the_read_back(scope):
    s, fake = scope
    ch = s.ch2
    assert ch.set_display_state(True) is True
    assert ch.set_attenuation(10) == 10.0
    assert fake.log.index("PROB2:SET:ATT:UNIT V") < fake.log.index("PROB2:SET:ATT:MAN 10")
    assert ch.set_scale(0.5) == 0.5
    assert ch.set_offset(0.1) == pytest.approx(0.1)
    assert ch.set_position(-1.0) == -1.0
    assert ch.set_coupling(Coupling.AC) is Coupling.AC
    assert ch.set_bandwidth_limit_state(True) is True
    assert ch.set_invert(True) is True
    settings = ch.get_settings()
    assert settings == ChannelSettings(
        display_state=True,
        scale=0.5,
        offset=0.1,
        position=-1.0,
        coupling=Coupling.AC,
        bandwidth_limit=True,
        invert=True,
        attenuation=10.0,
        type="HRES",
    )
    assert not hasattr(ch, "apply_setup") and not hasattr(ch, "on")


def test_threshold_find_and_hysteresis(scope):
    s, fake = scope
    assert s.ch1.set_threshold(0.3) == 0.3
    assert s.ch1.set_threshold_find() == 0.3
    assert "CHAN1:THR:FIND" in fake.log
    assert s.ch1.set_hysteresis(Hysteresis.LARGE) is Hysteresis.LARGE


# ------------------------------------------------------------- horizontal


def test_timebase_helper_changes_only_what_is_given(scope):
    s, _ = scope
    h = s.horz.set_timebase(1e-3, position=2e-4)
    assert h.scale == 1e-3 and h.position == 2e-4 and h.roll_auto is False
    assert h.range == pytest.approx(0.012)
    assert h.reference is TimeReference.CENTER
    h = s.horz.set_timebase(1e-4)
    assert h.position == 2e-4


def test_reference_has_three_positions(scope):
    s, fake = scope
    assert s.horz.set_reference(TimeReference.LEFT) is TimeReference.LEFT
    assert "TIM:REF 8.33333333333" in fake.log
    assert s.horz.set_reference("right") is TimeReference.RIGHT


def test_zoom_and_roll(scope):
    s, _ = scope
    s.horz.set_timebase(1e-4)
    z = s.horz.set_zoom(1e-5, time=2.5e-5)
    assert z.scale == 1e-5 and z.time == 2.5e-5
    assert s.horz.set_zoom_state(False) is False
    assert s.horz.get_zoom() is None
    assert s.horz.set_roll_auto_state(True) is True
    assert s.horz.set_roll_minimum(0.1) == 0.1


# ------------------------------------------------------------- acquisition


def test_acquisition_mode_writes_every_channel_type(scope):
    s, fake = scope
    assert s.acq.set_mode(AcquireMode.SAMPLE) is AcquireMode.SAMPLE
    assert all(f"CHAN{n}:TYPE SAMP" in fake.log for n in range(1, 5))
    assert s.acq.set_mode("average") is AcquireMode.AVERAGE
    assert s.acq.set_average_count(16) == 16
    assert s.acq.set_interpolation(Interpolation.SAMPLE_HOLD) is Interpolation.SAMPLE_HOLD


def test_record_length_reads_back(scope):
    s, _ = scope
    assert s.acq.set_record_length(1_000_000) == 1_000_000
    a = s.acq.get_settings()
    assert isinstance(a, AcquisitionSettings) and a.record_length == 1_000_000
    assert a.memory == "DMEM" and a.adc_rate == 6.25e7


def test_record_length_starts_a_stopped_acquisition(scope):
    s, fake = scope
    fake.s["ACQ:STAT"] = "STOP"
    s.acq.set_record_length(10_000)
    assert "RUN" in fake.log


def test_record_length_not_applied_raises_with_quirk(scope, monkeypatch):
    s, fake = scope
    _patch_write(monkeypatch, fake, "ACQ:POIN ")
    with pytest.raises(InstrumentError, match="RTB-08"):
        s.acq.set_record_length(10_000, timeout=0.01)


# ------------------------------------------------------------- trigger


def test_trigger_edge(scope):
    s, _ = scope
    t = s.trig.set_edge("ch1", level=0.25, slope=Slope.FALL, holdoff=1e-6)
    assert t.source == "ch1" and t.level == 0.25 and t.slope is Slope.FALL
    assert t.holdoff == 1e-6 and t.mode is TriggerMode.NORMAL and t.type == "EDGE"
    t = s.trig.set_edge("ext", level=1.0)
    assert t.source == "ext" and t.level == 1.0
    t = s.trig.set_edge("line")
    assert t.source == "line" and t.level is None
    assert t.holdoff is None


def test_trigger_edge_without_level_finds_the_mid_level(scope):
    s, fake = scope
    t = s.trig.set_edge("ch1")
    assert "TRIG:A:FIND" in fake.log and t.level == 0.01


def test_line_source_has_no_level(scope):
    s, _ = scope
    with pytest.raises(ValueError, match="no trigger level"):
        s.trig.set_level(0.5, source="line")


def test_other_triggers(scope):
    s, _ = scope
    t = s.trig.set_pulse_width("ch1", level=0.5, width=10e-6, condition=WidthCondition.LONGER)
    assert t.type.startswith("WIDT")
    assert s.trig.set_runt("ch1", lower=0.25, upper=0.75).type == "RUNT"
    assert s.trig.set_transition("ch1", lower=0.1, upper=0.9, time=500e-9).type.startswith("RIS")
    assert s.trig.set_timeout("ch1", level=0.25, time=500e-6).type.startswith("TIM")
    t = s.trig.set_pattern(
        {"ch1": PatternLevel.HIGH, "ch2": "low"}, levels={"ch1": 0.5, "ch2": 0.5}
    )
    assert t.pattern.startswith("10XX")


def test_video_trigger_and_actions(scope):
    s, _ = scope
    t = s.trig.set_video("ch1")
    assert t.video_standard is VideoStandard.NTSC and t.video_field is VideoField.ALL_LINES
    a = s.trig.set_actions(reference=True)
    assert a.reference and a.enabled and not a.sound
    assert not any(dataclasses.asdict(s.trig.set_actions()).values())


def test_trigger_mode_and_holdoff(scope):
    s, _ = scope
    assert s.trig.set_mode(TriggerMode.AUTO) is TriggerMode.AUTO
    assert s.trig.set_holdoff(2e-6) == 2e-6
    assert s.trig.set_holdoff(None) is None
    assert s.trig.set_hf_reject_state(True) is True


# ------------------------------------------------------------- capture


def test_capture_and_clip(scope):
    s, fake = scope
    s.ch1.set_scale(0.2)
    cap = s.data.meas_waveform(["ch1"])
    assert isinstance(cap, Capture) and len(cap.ch1) == 1200
    assert cap.sample_rate == pytest.approx(1e6)
    assert "ch1" in cap.settings.channels
    assert "CHAN1:DATA:POIN DMAX" in fake.log  # full record while stopped (quirk RTB-04)
    fake.clip = True
    with pytest.raises(InstrumentError, match="clips"):
        s.data.meas_waveform(["ch1"])


def test_capture_record_saves_npz(scope, tmp_path):
    s, _ = scope
    cap = s.data.get_waveform(["ch1", "ch2"], check_clip=False)
    path = cap.save(tmp_path)
    with np.load(path) as saved:
        assert set(saved.files) == {"t", "ch1", "ch2"}
    with pytest.raises(AttributeError):
        cap.ch3  # noqa: B018 - the attribute access is the test


def test_trigger_timeout_clears_the_queue(scope):
    s, fake = scope
    fake.trigger_fires = False
    with pytest.raises(InstrumentTimeoutError, match="CH1"):
        s.data.meas_waveform(["ch1"], timeout=0.1)
    assert fake.cleared >= 1 and "RUN" in fake.log
    assert s.get_errors() == []


def test_roll_capture_ends_running_with_roll_off(scope):
    s, fake = scope
    cap = s.data.meas_waveform_roll(["ch1"], 0.0, scale=1.0)
    assert len(cap.ch1) == 1200
    assert "STOP" in fake.log and "CHAN1:DATA:POIN DEF" in fake.log
    assert fake.s["TIM:ROLL:AUT"] == "OFF" and fake.s["ACQ:STAT"] == "RUN"


def test_auto_range_keeps_bandwidth_and_triggers_normal(scope):
    s, _ = scope
    s.ch2.set_bandwidth_limit_state(True)
    ranges = s.set_auto_range(["ch1", "ch2"], trigger_source="ch1")
    assert set(ranges) == {"ch1", "ch2"}
    assert ranges["ch1"].low == pytest.approx(-0.5, abs=0.01)
    assert s.ch2.get_bandwidth_limit_state() is True
    assert s.trig.get_mode() is TriggerMode.NORMAL


# ------------------------------------------------------------- measurements


def test_meas_node_carries_the_one_shot_measurements(scope):
    s, fake = scope
    assert s.meas.frequency("ch1") == 1e4
    assert "MEAS8:MAIN FREQ" in fake.log and fake.log[-1] == "RUN"
    assert s.meas.phase("ch2", "ch1") == 90.0
    assert "MEAS8:SOUR CH2,CH1" in fake.log
    assert s.meas.custom(MeasType.RMS, "ch1") == pytest.approx(0.3536)


def test_root_has_no_duplicate_measure_methods(scope):
    s, _ = scope
    assert not [n for n in dir(s) if n.startswith(("measure_", "meas_"))]


def test_invalid_measurement_raises(scope):
    s, _ = scope
    with pytest.raises(InstrumentError, match="invalid"):
        s.meas.high("ch1")


def test_delay_needs_a_reference(scope):
    s, _ = scope
    with pytest.raises(ValueError, match="reference"):
        s.meas.set_slot(2, MeasType.DELAY, "ch1")


def test_statistics_use_slot_1(scope):
    s, fake = scope
    stats = s.meas.statistics(MeasType.FREQUENCY, "ch1", count=100)
    assert isinstance(stats, MeasStatistics) and stats.count >= 100 and stats.mean == 1e4
    assert "MEAS1:STAT ON" in fake.log and "MEAS1:STAT OFF" in fake.log


def test_quick_dvm_counter_cursors(scope):
    s, fake = scope
    q = s.meas.quick("ch1")
    assert isinstance(q, QuickMeasurement) and q.frequency == 1e4 and q.mean == 0.25
    assert fake.log[-1].startswith("SYST:ERR") and "MEAS1:AOFF" in fake.log
    assert s.dvm.meas_dc("ch1") == 0.25 and s.dvm.meas_ac_rms("ch1") == 0.354
    assert s.dvm.meas_ac_dc_rms("ch1") == 0.434
    assert s.counter.meas_frequency("ch1") == 1e4 and s.counter.meas_period() == 1e-4
    assert s.cursors.set_time("ch1", -2.5e-5, 7.5e-5).delta == pytest.approx(1e-4)
    assert s.cursors.set_voltage("ch1", 0.6, -0.1).delta == pytest.approx(0.7)


def test_display(scope):
    s, _ = scope
    assert s.disp.set_grid(GridStyle.RETICLE) is GridStyle.RETICLE
    assert s.disp.set_persistence(math.inf) == math.inf
    assert s.disp.set_persistence(0.5) == 0.5
    assert s.disp.set_persistence(None) is None
    assert s.disp.set_xy("ch1", "ch2") == XySettings(xy_state=True, x="ch1", y1="ch2")
    assert s.disp.set_xy_state(False) is False


# ------------------------------------------------------------- analysis and files


def test_math_reference_search_fft_mask(scope):
    s, _ = scope
    assert s.math.set_operation(MathOperation.ADD, "ch1", "ch2").startswith("ADD(CH1,CH2)")
    t, x = s.math.get_waveform()
    assert len(t) == len(x) == 1200
    s.ref1.cmd_save_from("ch1")
    assert len(s.ref1.get_waveform()[1]) == 1200
    hits = s.search.set_width("ch1", level=0.5, width=10e-6)
    assert len(hits) == 3 and hits[0].value == 2e-5 and hits[0].time == -1e-4
    fft = s.fft.set_source("ch1")
    assert fft.window is FftWindow.HANNING and fft.source == "ch1"
    assert len(s.fft.get_spectrum()[0]) == 1200
    assert s.mask.set_from_source("ch1") is True
    s.mask.cmd_run()
    assert s.mask.get_counts() == MaskCounts(acquisitions=26, violations=0)


def test_files(scope):
    s, _ = scope
    assert s.file.cmd_save_setup("TESTS") > 0
    s.file.cmd_recall_setup("TESTS")
    assert s.file.cmd_save_waveform("ch1", "TESTW") > 0
    assert s.file.get_file("TESTW.CSV").startswith(b"t,ch1")
    names = [f.name for f in s.file.get_catalog().files]
    assert "TESTW.CSV" in names and "TESTS.SET" in names
    s.file.cmd_delete("TESTW.CSV")
    assert s.file.get_size("TESTW.CSV") is None
    with pytest.raises(InstrumentError, match="no setup file"):
        s.file.cmd_recall_setup("MISSING")
    assert s.file.get_drives() == ["/INT"]


def test_screenshot(scope, tmp_path):
    s, _ = scope
    p = tmp_path / "s.png"
    assert s.file.get_screenshot(p).startswith(b"\x89PNG") and p.exists()


# ------------------------------------------------------------- history (K15)


def test_history(scope):
    s, _ = scope
    h = s.history.set_segmented_memory(100, record_length=10_000)
    assert h == HistorySettings(count=100, record_length=10_000, fast_segmentation=True)
    assert s.history.cmd_acquire(100) >= 100
    assert list(s.history.get_timestamps()) == [-2e-3, -1e-3, 0.0]
    cap = s.history.get_waveform(-1, ["ch1"])
    assert cap.history_index == -1 and len(cap.ch1) == 1200


# ------------------------------------------------------------- bus (K1, K2, K3, B1)


def test_bus_uart_frames(scope):
    s, fake = scope
    assert s.bus.set_uart("ch1", 115200, threshold=1.0) == "UART"
    assert fake.s["CHAN1:THR"] == "1"
    fake.s |= {
        "BUS1:UART:FCO": "1",
        "BUS1:UART:FRAM1:STAT": "OK",
        "BUS1:UART:FRAM1:STAR": "0",
        "BUS1:UART:FRAM1:WCO": "2",
        "BUS1:UART:FRAM1:WORD1:VAL": "69",
        "BUS1:UART:FRAM1:WORD2:VAL": "80",
    }
    assert s.bus.get_frames() == [BusFrame(start=0.0, state="ok", words=[69, 80])]


def test_bus_frame_detail_is_skipped_on_an_invalid_frame(scope):
    s, fake = scope
    s.bus.set_i2c("ch1", "ch2")
    fake.s |= {"BUS1:TYPE": "I2C", "BUS1:I2C:FCO": "1", "BUS1:I2C:FRAM1:STAT": "UNEX"}
    frames = s.bus.get_frames()
    assert frames[0].state == "unex" and frames[0].address is None
    assert not any("ADDR" in c for c in fake.log)  # quirk RTB-15


def test_bus_frame_query_timeout_reads_empty_and_clears(scope, monkeypatch):
    s, fake = scope
    s.bus.set_i2c("ch1", "ch2")
    fake.s |= {"BUS1:TYPE": "I2C", "BUS1:I2C:FCO": "1", "BUS1:I2C:FRAM1:STAT": "OK"}
    original = fake.query

    def query(cmd: str, **kwargs) -> str:
        if cmd == "BUS1:I2C:FRAM1:ADDR?":
            fake.errors.append('-200,"Execution error"')
            raise visa.VisaIOError(visa.constants.StatusCode.error_timeout)
        return original(cmd, **kwargs)

    monkeypatch.setattr(fake, "query", query)
    frames = s.bus.get_frames()
    assert frames[0].state == "ok" and frames[0].address is None
    assert fake.cleared == 1 and s.get_errors() == []


def test_uart_trigger_writes_bits_in_wire_order(scope):
    s, fake = scope
    u = s.bus.set_uart_trigger(mode=UartTriggerMode.PATTERN, data=b"P", offset=3)
    assert u.pattern == "00001010"  # 0x50 LSB first (quirk RTB-16)
    assert u.mode is UartTriggerMode.PATTERN
    assert not any(c.startswith("TRIG:A:SOUR:UART") for c in fake.log)
    s.bus.set_uart_trigger(mode=UartTriggerMode.START_BIT, line=UartLine.TX)
    assert "TRIG:A:SOUR:UART TX" in fake.log


def test_parallel_bus(scope):
    s, _ = scope
    p = s.bus.set_parallel(["d0", "d1", "d2", "d3"])
    assert p.type == "PAR" and p.width == 4 and p.data == ["d0", "d1", "d2", "d3"]
    p = s.bus.set_parallel(["d0", "d1"], clock="d4", clock_slope=Slope.FALL)
    assert p.clock == "d4" and p.clock_slope is Slope.FALL


# ------------------------------------------------------------- logic (B1)


def test_logic_pod_and_waveform(scope):
    s, fake = scope
    pod = s.logic.set_pod((0, 1, 2, 3))
    assert pod == LogicPodSettings(pod=1, channels=[0, 1, 2, 3], threshold=0.0, hysteresis=None)
    assert "DIG4:DISP OFF" in fake.log and "LOG1:THR TTL" in fake.log
    cap = s.logic.get_waveform(["d0", "d1"])
    assert set(cap.channels) == {"d0", "d1"} and set(np.unique(cap.d0)) == {0, 1}
    assert "FORM ASC" in fake.log


def test_logic_refuses_without_probe_or_across_pods(scope):
    s, fake = scope
    with pytest.raises(ValueError, match="both pods"):
        s.logic.set_pod(("d0", "d8"))
    fake.s["LOG1:PROB"] = "0"
    with pytest.raises(InstrumentError, match="not connected"):
        s.logic.set_pod()


def test_logic_live_level(scope):
    s, fake = scope
    assert s.logic.get_live_level("d0") is LogicActivity.LOW
    fake.s["DIG0:CURR:STAT:MAX"] = "1"
    assert s.logic.get_live_level(0) is LogicActivity.TOGGLE


# ------------------------------------------------------------- generators (B6)


def test_pattern_helpers_set_the_pattern_and_leave_the_output_off(scope):
    s, fake = scope
    p = s.pattern.set_counter(1e3, direction=CountDirection.DOWN)
    assert p.function is PatternGenFunction.COUNTER and p.direction is CountDirection.DOWN
    assert p.output_state is False and "PGEN:PATT:STAT ON" not in fake.log
    sq = s.pattern.set_square(2e3, duty=25.0, inverted=True)
    assert sq.frequency == 2e3 and sq.duty == 25.0 and sq.inverted is True
    assert s.pattern.set_arbitrary([1, 2, 3], 1e-6).values == [1, 2, 3]
    assert s.pattern.set_manual(5) == 5
    assert "PGEN:PATT:STAT ON" not in fake.log
    assert s.pattern.set_output_state(True) is True


def test_afg_helpers_set_the_waveform_and_leave_the_output_off(scope):
    s, fake = scope
    w = s.afg.set_waveform_sine(1e4, 1.0, offset=0.1)
    assert w == WaveformState(
        function="SINE", frequency=1e4, amplitude=1.0, offset=0.1, output_state=False
    )
    assert "WGEN:OUTP ON" not in fake.log
    assert s.afg.set_waveform_pulse(1e3, 1.0, duty=20.0).duty == 20.0
    assert s.afg.set_waveform_dc(0.5).function == "DC"
    assert s.afg.set_output_state(True) is True
    fake.s["WGEN:FUNC"] = "SINUSOID"
    assert s.afg.get_waveform().function == "SINE"


def test_afg_load(scope):
    s, _ = scope
    assert s.afg.set_load(50) == 50.0
    assert s.afg.set_load(Load.HIGHZ) == math.inf
    assert s.afg.set_load(math.inf) == math.inf
    with pytest.raises(ValueError, match="50 ohm"):
        s.afg.set_load(75)


def test_afg_burst(scope):
    s, fake = scope
    assert s.afg.set_burst(3) == AfgBurstSettings(
        enabled=True, cycles=3, phase=0.0, continuous=False
    )
    assert s.afg.set_burst(2, continuous=True).continuous is True
    s.afg.cmd_burst_trigger()
    assert fake.log[-1] == "WGEN:BURS:TRIG:SING"  # no completion wait after it
    assert s.afg.set_burst_state(False) is False


def test_aux_out(scope):
    s, _ = scope
    assert s.aux_out.get_source() is AuxOutSource.REFERENCE
    assert s.aux_out.set_source(AuxOutSource.TRIGGER) is AuxOutSource.TRIGGER
    t = s.aux_out.set_trigger_pulse(width=2e-6, polarity=Polarity.NEGATIVE)
    assert t == TriggerOutSettings(
        source=AuxOutSource.TRIGGER, width=2e-6, polarity=Polarity.NEGATIVE
    )
    assert s.aux_out.set_source("reference") is AuxOutSource.REFERENCE


# ------------------------------------------------------------- system


def test_run_stop_single_and_state(scope):
    s, fake = scope
    s.cmd_stop()
    assert s.get_acquisition_state() is AcquisitionState.STOPPING
    s.cmd_run()
    assert s.get_acquisition_state() is AcquisitionState.RUN
    s.cmd_single()
    assert fake.log[-1] == "SING"
    fake.s["ACQ:STAT"] = "COMPLETE"
    assert s.get_acquisition_state() is AcquisitionState.COMPLETE


def test_reset_waits_and_raises_on_queued_errors(scope):
    s, fake = scope
    s.cmd_reset()
    assert fake.log[:2] == ["*RST", "*OPC?"]
    fake.refuse["*RST"] = '-200,"Execution error"'
    with pytest.raises(InstrumentQueryError, match="-200"):
        s.cmd_reset()


def test_get_errors_drains_the_queue(scope):
    s, fake = scope
    fake.errors = ['-113,"Undefined header"', '-222,"Data out of range"']
    assert s.get_errors() == ['-113,"Undefined header"', '-222,"Data out of range"']
    assert s.get_errors() == []


def test_snapshot_and_restore_put_back_what_was_found(scope):
    s, fake = scope
    before = s.get_settings()
    snap = s.snapshot()
    s.ch1.set_scale(1.0)
    s.ch2.set_display_state(True)
    s.ch3.set_coupling(Coupling.AC)
    s.horz.set_timebase(1e-3, position=1e-4, reference=TimeReference.LEFT)
    s.acq.set_mode(AcquireMode.AVERAGE)
    s.trig.set_edge("ch2", level=0.3, slope=Slope.FALL, holdoff=1e-6)
    s.aux_out.set_source(AuxOutSource.OFF)
    s.afg.set_load(50)
    s.afg.set_waveform_square(5e3, 2.0)
    s.afg.set_output_state(True)
    s.restore(snap)
    after = s.get_settings()
    assert after.channels == before.channels
    assert after.acquisition == before.acquisition
    assert after.trigger == before.trigger
    for name in ("scale", "position", "reference", "roll_auto"):
        assert getattr(after.horizontal, name) == getattr(before.horizontal, name), name
    assert s.aux_out.get_source() is AuxOutSource.REFERENCE
    assert s.afg.get_waveform() == snap["afg"]["waveform"]
    assert s.afg.get_load() == math.inf
    assert fake.s["WGEN:OUTP"] == "OFF"


def test_preserved_setup_restores_on_an_error(scope):
    s, _ = scope
    with pytest.raises(RuntimeError, match="body"):
        with s.preserved_setup():
            s.ch1.set_scale(2.0)
            raise RuntimeError("body failed")
    assert s.ch1.get_scale() == 0.2


# ------------------------------------------------------------- API contract

_SCPI_IN_SUMMARY = re.compile(
    r"\*[A-Za-z]{2,4}\??"
    r"|[A-Za-z]{2,}(?::[A-Za-z0-9<>\[\]|_]+\??)+"
    r"|:[A-Za-z]{3,}\??(?=[)\s.,;]|$)"
)
_SPHINX_ROLE = re.compile(r":(?:class|meth|func|data|attr|mod|obj|ref|exc):`[^`]*`")
_UNPREFIXED = {"connect", "disconnect", "snapshot", "restore", "preserved_setup", "channel"}
_MEAS_NODE_NAMES = {
    "custom",
    "quick",
    "statistics",
    "frequency",
    "period",
    "peak_to_peak",
    "amplitude",
    "high",
    "low",
    "mean",
    "rms",
    "rise_time",
    "fall_time",
    "duty_positive",
    "width_positive",
    "overshoot_positive",
    "phase",
    "delay",
}


def _public_methods(obj):
    """(class name, method name, function) for each method the driver defines on ``obj``."""
    seen = {}
    for klass in reversed(type(obj).__mro__):
        if klass in (object, Node) or klass.__name__ == "Instrument":
            continue
        for name, value in vars(klass).items():
            if not name.startswith("_") and inspect.isfunction(value):
                seen[name] = (klass.__name__, value)
    return [(cls, name, fn) for name, (cls, fn) in sorted(seen.items())]


def _all_nodes():
    s = RohdeSchwarz_RTB2000(RESOURCE)
    nodes = [s] + [v for k, v in vars(s).items() if isinstance(v, Node) and not k.startswith("_")]
    return [(type(node).__name__, node) for node in nodes]


def test_every_public_method_follows_the_naming_and_docstring_rules():
    problems = []
    for node_name, node in _all_nodes():
        for _cls, name, fn in _public_methods(node):
            where = f"{node_name}.{name}"
            doc = inspect.getdoc(fn) or ""
            first = doc.splitlines()[0] if doc else ""
            if not first:
                problems.append(f"{where}: no docstring")
            elif _SCPI_IN_SUMMARY.search(_SPHINX_ROLE.sub("", first)):
                problems.append(f"{where}: SCPI in the first docstring line")
            prefixed = name.startswith(("set_", "get_", "meas_", "cmd_"))
            allowed = name in _UNPREFIXED or name == "safe_state"
            if node_name == "_Measure":
                allowed = allowed or name in _MEAS_NODE_NAMES
            if not prefixed and not allowed:
                problems.append(f"{where}: no verb prefix")
            if name.startswith("set_"):
                ret = inspect.signature(fn).return_annotation
                if ret in (inspect.Signature.empty, None, "None"):
                    problems.append(f"{where}: set_ without a read-back return")
    assert problems == []


def test_banned_names_are_absent():
    banned = re.compile(r"^(on|off|output_on|output_off|is_\w+|apply\w*|measure_\w+|capture\w*)$")
    found = [
        f"{node_name}.{name}"
        for node_name, node in _all_nodes()
        for _cls, name, _fn in _public_methods(node)
        if banned.match(name)
    ]
    assert found == []


def test_records_are_frozen():
    for record in (ChannelSettings, Capture, BusFrame, MeasStatistics, HistorySettings):
        assert dataclasses.is_dataclass(record)
        assert record.__dataclass_params__.frozen, record.__name__


# ---- FFT range (RTB-13) ----------------------------------------------------------------------


def _fft_opens_up(monkeypatch, fake, rounds_needed):
    """The range reaches the request only on round ``rounds_needed`` (MEAS 2026-10-18)."""
    write = fake.write
    fake.s["SPEC:FREQ:STAR"] = "0"
    fake.s["SPEC:FREQ:STOP"] = "2.86E+06"
    state = {"round": 0, "span": 0.0}

    def fake_write(cmd, **kwargs):
        head, _, val = cmd.partition(" ")
        if head == "SPEC:FREQ:SPAN":
            state["span"] = float(val)
        elif head == "SPEC:FREQ:CENT":
            state["round"] += 1
            if state["round"] >= rounds_needed:
                fake.s["SPEC:FREQ:STAR"] = str(float(val) - state["span"] / 2)
                fake.s["SPEC:FREQ:STOP"] = str(float(val) + state["span"] / 2)
        return write(cmd, **kwargs)

    monkeypatch.setattr(fake, "write", fake_write)
    return state


def test_fft_set_range_repeats_until_the_range_reads_back(make_scope, monkeypatch):
    scope, fake = make_scope()
    scope.fft.set_source("ch1")
    state = _fft_opens_up(monkeypatch, fake, rounds_needed=3)
    got = scope.fft.set_range(45e3, 62e3)
    assert state["round"] == 3 and got.start == 45e3 and got.stop == 62e3


def test_fft_set_range_sets_the_rbw_ratio_after_the_range(make_scope, monkeypatch):
    scope, fake = make_scope()
    scope.fft.set_source("ch1")
    _fft_opens_up(monkeypatch, fake, rounds_needed=1)
    fake.log.clear()
    scope.fft.set_range(45e3, 62e3, rbw_ratio=1024)
    writes = [c for c in fake.log if c.startswith("SPEC:FREQ:")]
    centre = next(i for i, c in enumerate(writes) if c.startswith("SPEC:FREQ:CENT"))
    assert writes.index("SPEC:FREQ:BAND:RES:RAT 1024") > centre


def test_fft_set_range_raises_if_the_range_never_reads_back(make_scope, monkeypatch):
    scope, fake = make_scope()
    scope.fft.set_source("ch1")
    _fft_opens_up(monkeypatch, fake, rounds_needed=99)
    with pytest.raises(InstrumentError, match="FFT range reads"):
        scope.fft.set_range(45e3, 62e3)
