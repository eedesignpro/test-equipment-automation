"""Unit tests for the R17 trigger config_* additions (video/timeout/runt/
window/transition). Readback shapes below are bench-confirmed in
tests/captures/trig_sv_verification.json (video: 'NTSC'/'PAL'/'SECAM'
command-acceptance only - no video source available from the AFG; timeout
2e-6s; runt (0.8,-0.8); window (0.5,-0.5); transition 1e-7s).
"""

import pytest

from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource


def _connect(mock):
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    mock.responses.setdefault("*IDN?", lambda _c: "TEKTRONIX,MSO58,C0,CF:91")
    scope.connect()
    mock.writes.clear()
    return scope


# ---------------------------------------------------------------------------
# Video (R17 priority)
# ---------------------------------------------------------------------------


_VIDEO_STANDARD_SCPI = {"NTSC": "NTSc", "PAL": "PAL", "SECAM": "SECAM"}


@pytest.mark.parametrize("standard", ["NTSC", "PAL", "SECAM"])
def test_set_video_standard_command_and_readback(standard):
    mock = MockVisaResource(responses={"TRIG:A:VID:STAN?": standard})
    scope = _connect(mock)
    result = scope.trig.set_video("CH5", standard=standard)
    assert result == standard
    assert "TRIG:A:TYP VIDEO" in mock.writes
    assert "TRIG:A:VID:SOU CH5" in mock.writes
    assert f"TRIG:A:VID:STAN {_VIDEO_STANDARD_SCPI[standard]}" in mock.writes


def test_set_video_default_field_all_lines():
    mock = MockVisaResource(responses={"TRIG:A:VID:STAN?": "NTSC"})
    scope = _connect(mock)
    scope.trig.set_video("CH5")
    assert any("TRIG:A:VID:FIELD" in w for w in mock.writes)
    assert not any("TRIG:A:VID:LINE" in w for w in mock.writes)


def test_set_video_numeric_line_sets_field_numeric_and_line():
    mock = MockVisaResource(responses={"TRIG:A:VID:STAN?": "NTSC"})
    scope = _connect(mock)
    scope.trig.set_video("CH5", line=42)
    assert "TRIG:A:VID:LINE 42" in mock.writes
    numeric_field_writes = [w for w in mock.writes if w.startswith("TRIG:A:VID:FIELD")]
    assert len(numeric_field_writes) == 1


def test_set_video_invalid_standard_raises():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.trig.set_video("CH5", standard="BOGUS")


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_set_timeout_command_sequence_and_readback():
    mock = MockVisaResource(responses={"TRIG:A:TIMEO:TIM?": "2e-06"})
    scope = _connect(mock)
    result = scope.trig.set_timeout("CH5", time=2e-6, polarity="STAYS HIGH")
    assert result == pytest.approx(2e-6)
    assert "TRIG:A:TYP TIMEOUT" in mock.writes
    assert "TRIG:A:TIMEO:SOU CH5" in mock.writes
    assert "TRIG:A:TIMEO:TIM 2e-06" in mock.writes


def test_set_timeout_explicit_level_skips_setlevel():
    mock = MockVisaResource(responses={"TRIG:A:TIMEO:TIM?": "1e-06"})
    scope = _connect(mock)
    scope.trig.set_timeout("CH5", time=1e-6, level=0.4)
    assert "TRIG:A:LEV:CH5 0.4" in mock.writes
    assert "TRIG:A SETLEVEL" not in mock.writes


def test_set_timeout_invalid_polarity_raises():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.trig.set_timeout("CH5", time=1e-6, polarity="BOGUS")


# ---------------------------------------------------------------------------
# Runt
# ---------------------------------------------------------------------------


def test_set_runt_occurs_command_sequence_and_readback():
    mock = MockVisaResource(responses={"TRIG:A:UPP:CH5?": "0.8", "TRIG:A:LOW:CH5?": "-0.8"})
    scope = _connect(mock)
    result = scope.trig.set_runt("CH5", upper=0.8, lower=-0.8)
    assert result == (pytest.approx(0.8), pytest.approx(-0.8))
    assert "TRIG:A:TYP RUNT" in mock.writes
    assert "TRIG:A:UPP:CH5 0.8" in mock.writes
    assert "TRIG:A:LOW:CH5 -0.8" in mock.writes
    assert not any("TRIG:A:RUNT:WID" in w for w in mock.writes)


def test_set_runt_lessthan_requires_width():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.trig.set_runt("CH5", upper=0.8, lower=-0.8, when="LESSTHAN")


def test_set_runt_lessthan_with_width_sends_width_command():
    mock = MockVisaResource(responses={"TRIG:A:UPP:CH5?": "0.8", "TRIG:A:LOW:CH5?": "-0.8"})
    scope = _connect(mock)
    scope.trig.set_runt("CH5", upper=0.8, lower=-0.8, when="LESSTHAN", width=1e-7)
    assert "TRIG:A:RUNT:WID 1e-07" in mock.writes


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


def test_set_window_enters_command_sequence_and_readback():
    mock = MockVisaResource(responses={"TRIG:A:UPP:CH5?": "0.5", "TRIG:A:LOW:CH5?": "-0.5"})
    scope = _connect(mock)
    result = scope.trig.set_window("CH5", upper=0.5, lower=-0.5)
    assert result == (pytest.approx(0.5), pytest.approx(-0.5))
    assert "TRIG:A:TYP WINDOW" in mock.writes
    assert "TRIG:A:WIN:SOU CH5" in mock.writes


def test_set_window_inside_requires_width():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.trig.set_window("CH5", upper=0.5, lower=-0.5, when="INSIDE")


def test_set_window_outside_with_width_sends_width_command():
    mock = MockVisaResource(responses={"TRIG:A:UPP:CH5?": "0.5", "TRIG:A:LOW:CH5?": "-0.5"})
    scope = _connect(mock)
    scope.trig.set_window("CH5", upper=0.5, lower=-0.5, when="OUTSIDE", width=2e-7)
    assert "TRIG:A:WIN:WID 2e-07" in mock.writes


# ---------------------------------------------------------------------------
# Transition (rise/fall time)
# ---------------------------------------------------------------------------


def test_set_transition_command_sequence_and_readback():
    mock = MockVisaResource(responses={"TRIG:A:TRAN:DELT?": "1e-07"})
    scope = _connect(mock)
    result = scope.trig.set_transition("CH5", delta_time=1e-7, upper=0.5, lower=-0.5)
    assert result == pytest.approx(1e-7)
    assert "TRIG:A:TYP TRANSITION" in mock.writes
    assert "TRIG:A:TRAN:SOU CH5" in mock.writes
    assert "TRIG:A:UPP:CH5 0.5" in mock.writes
    assert "TRIG:A:LOW:CH5 -0.5" in mock.writes
    assert "TRIG:A:TRAN:DELT 1e-07" in mock.writes


def test_set_transition_invalid_when_raises():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.trig.set_transition("CH5", delta_time=1e-7, upper=0.5, lower=-0.5, when="BOGUS")
