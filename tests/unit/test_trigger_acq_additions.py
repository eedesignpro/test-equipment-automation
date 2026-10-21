"""Unit tests for R11 trigger additions (B-event, holdoff BY, user hysteresis,
state) and R13 acquisition additions (sequence, fast-average).

Query responses replay real MSO58 traffic captured in
tests/captures/trig_acq_trace.json (bench session 2026-07-23). The B-trigger
edge event uses the undocumented-but-real TRIGger:B:EDGE branch (ADR-016).
"""

import pytest

from drivers.tektronix_mso.enums import AcqSequenceMode
from tests.mocks.capture_replay import connect_replay

TRACE = "tek_mso/trig_acq_trace.json"


def test_get_state_reads_trigger_state():
    scope, mock = connect_replay(TRACE)
    assert scope.trig.get_state() == "TRIGGER"
    assert "TRIGger:STATE?" in mock.writes


def test_holdoff_by_command_and_readback():
    scope, mock = connect_replay(TRACE)
    result = scope.trig.set_holdoff_by("RANDOM")
    assert "TRIG:A:HOLD:BY RANDom" in mock.writes
    assert result == "RANDOM"


def test_holdoff_by_invalid_raises():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.trig.set_holdoff_by("BOGUS")


def test_user_hysteresis_state_and_value():
    scope, mock = connect_replay(TRACE)
    scope.trig.set_user_hysteresis(True, 120e-3)
    assert "TRIGger:HYSTeresis:USER:STATe ON" in mock.writes
    assert "TRIGger:HYSTeresis:USER:VALue 0.12" in mock.writes


def test_user_hysteresis_state_only_returns_none():
    scope, mock = connect_replay(TRACE)
    result = scope.trig.set_user_hysteresis(False)
    assert result is None
    assert "TRIGger:HYSTeresis:USER:STATe OFF" in mock.writes
    assert not any("USER:VALue" in w and "?" not in w for w in mock.writes)


def test_b_state_write():
    scope, mock = connect_replay(TRACE)
    scope.trig.set_b_state(True)
    assert "TRIGger:B:STATE ON" in mock.writes


def test_b_by_events_count_and_time():
    scope, mock = connect_replay(TRACE)
    scope.trig.set_b_by("EVENTS")
    scope.trig.set_b_events_count(3)
    scope.trig.set_b_time(50e-9)
    assert "TRIGger:B:BY EVENTS" in mock.writes
    assert "TRIGger:B:EVENTS:COUNt 3" in mock.writes
    assert "TRIGger:B:TIMe 5e-08" in mock.writes


def test_set_b_edge_uses_b_edge_branch():
    scope, mock = connect_replay(TRACE)
    level = scope.trig.set_b_edge("CH5", level=0.0, slope="RISING", coupling="DC")
    assert "TRIGger:B:TYPe EDGE" in mock.writes
    assert "TRIGger:B:EDGE:SOUrce CH5" in mock.writes
    assert "TRIGger:B:EDGE:SLOpe RISE" in mock.writes
    assert "TRIGger:B:EDGE:COUPling DC" in mock.writes
    assert "TRIGger:B:LEVel:CH5 0" in mock.writes
    assert level == pytest.approx(0.0)


def test_set_b_edge_no_level_skips_level_write():
    scope, mock = connect_replay(TRACE)
    scope.trig.set_b_edge("CH5", slope="FALLING")
    assert "TRIGger:B:EDGE:SLOpe FALL" in mock.writes
    assert not any(w.startswith("TRIGger:B:LEVel:CH5 ") for w in mock.writes)


# --- acquisition additions -------------------------------------------------


def test_sequence_current_and_mode():
    scope, mock = connect_replay(TRACE)
    assert scope.acq.get_sequence_current() == 0
    assert scope.acq.get_sequence_mode() is AcqSequenceMode.NUMACQS
    assert "ACQ:SEQ:CURRENT?" in mock.writes


def test_set_sequence_mode_command():
    scope, mock = connect_replay(TRACE)
    scope.acq.set_sequence_mode("NUMACQS")
    assert "ACQ:SEQ:MOD NUMACQs" in mock.writes


def test_fast_average_trio_commands():
    scope, mock = connect_replay(TRACE)
    scope.acq.set_fast_average_state(True)
    scope.acq.set_fast_average_limit(16)
    scope.acq.set_fast_average_stopafter(64)
    assert "ACQ:FASTAV:STATE ON" in mock.writes
    assert "ACQ:FASTAV:LIM 16" in mock.writes
    assert "ACQ:FASTAV:STOPA 64" in mock.writes
