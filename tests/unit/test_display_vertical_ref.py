"""Unit tests for R14 Display SELect and R15 vertical queries / reference
basics (compound-response parsing, SELect:CH, EXTDBatten, ref waveforms).

Query responses replay real MSO58 traffic captured in
tests/captures/display_vertical_ref_trace.json (bench session 2026-07-23,
CH1 TPP1000 / CH4 TCP202 / CH5 BNC-AFG). SAVe:WAVEform-to-REF is not
implemented on this firmware, so references are created via REF:ADDNew against
the selected source (ADR-020).
"""

from tests.mocks.capture_replay import connect_replay

TRACE = "tek_mso/display_vertical_ref_trace.json"


# --- R14 display select ----------------------------------------------------


def test_display_select_source():
    scope, mock = connect_replay(TRACE)
    assert scope.disp.set_select_source("CH5") == "CH5"
    assert "DISplay:SELect:SOUrce CH5" in mock.writes


def test_display_select_view_and_waveview_source():
    scope, mock = connect_replay(TRACE)
    scope.disp.set_select_view("WAVEVIEW1")
    scope.disp.set_waveview_source("CH5", view=1)
    assert "DISplay:SELect:VIEW WAVEVIEW1" in mock.writes
    assert "DISplay:SELect:WAVEView1:SOUrce CH5" in mock.writes


def test_display_select_math_bus_reference():
    scope, mock = connect_replay(TRACE)
    scope.disp.set_select_reference("NONE")
    assert "DISplay:SELect:REFerence NONE" in mock.writes


# --- R15 vertical queries --------------------------------------------------


def test_get_config_parses_compound_response():
    scope, mock = connect_replay(TRACE)
    fields = scope.ch5.get_config()
    assert isinstance(fields, list)
    assert len(fields) == 59  # captured field count
    assert fields[0] == "PASSED"
    assert "CH5?" in mock.writes


def test_get_probe_config_parses_compound():
    scope, _ = connect_replay(TRACE)
    fields = scope.ch5.get_probe_config()
    assert isinstance(fields, list)
    assert len(fields) == 21


def test_probe_cal_and_selfcal_status():
    scope, _ = connect_replay(TRACE)
    assert scope.ch5.get_probe_cal_status() == "DEFAULT"
    assert scope.ch5.probe.get_selfcal_state() == "PASSED"


def test_analog_bandwidth_query():
    scope, _ = connect_replay(TRACE)
    assert scope.get_analog_bandwidth() == 2.0e9


def test_selected_channel_command_and_readback():
    scope, mock = connect_replay(TRACE)
    assert scope.ch5.set_selected(True) is True
    assert "SELect:CH5 ON" in mock.writes


def test_ext_db_attenuation():
    scope, mock = connect_replay(TRACE)
    assert scope.ch5.set_ext_db_attenuation(0.0) == 0.0
    assert "CH5:PROBEFunc:EXTDBatten 0" in mock.writes


def test_probe_type_current_probe_reports_analog():
    scope, _ = connect_replay(TRACE)
    # TCP202 current probe on CH4 reports ANALOG, not digital.
    assert scope.ch4.get_probe_type() == "ANALOG"


# --- R15 reference basics --------------------------------------------------


def test_ref_save_from_channel_uses_addnew():
    scope, mock = connect_replay(TRACE)
    scope.ref1.cmd_save_from("CH5")
    assert "SELect:CH5 ON" in mock.writes
    assert "DISplay:SELect:SOUrce CH5" in mock.writes
    assert 'REF:ADDNew "REF1"' in mock.writes


def test_ref_label_and_deskew():
    scope, mock = connect_replay(TRACE)
    scope.ref1.set_label("myref")
    scope.ref1.set_deskew(0.0)
    assert 'REF:REF1:LABel:NAMe "myref"' in mock.writes
    assert "REF:REF1:DESKew 0" in mock.writes


def test_ref_delete_and_list():
    scope, mock = connect_replay(TRACE)
    assert scope.ref1.cmd_delete() == []  # REF:LIST? -> NONE after delete
    assert 'REF:DELete "REF1"' in mock.writes
