"""Unit tests for R12 FastFrame + History (timestamp tables) and the R4/R12
segmented-readout DATa:FRAMESTARt/FRAMESTOP additions.

Query responses replay real MSO58 traffic captured in
tests/captures/history_fastframe_trace.json (bench session 2026-07-23). On this
family "History" maps onto the FastFrame group; the History command group
proper is only the timestamp-table trio (ADR-017).
"""

import pytest

from tests.mocks.capture_replay import connect_replay

TRACE = "tek_mso/history_fastframe_trace.json"


def test_fastframe_state_and_count():
    scope, mock = connect_replay(TRACE)
    scope.horz.set_fastframe_state(True)
    assert "HORizontal:FASTframe:STATE ON" in mock.writes
    assert scope.horz.set_fastframe_count(10) == 10
    assert "HORizontal:FASTframe:COUNt 10" in mock.writes


def test_fastframe_max_frames_query():
    scope, _ = connect_replay(TRACE)
    assert scope.horz.get_fastframe_max_frames() == 8000


def test_fastframe_selected():
    scope, mock = connect_replay(TRACE)
    assert scope.horz.set_fastframe_selected(3) == 3
    assert "HORizontal:FASTframe:SELECTED 3" in mock.writes


def test_fastframe_summary_and_multi_mode_commands():
    scope, mock = connect_replay(TRACE)
    scope.horz.set_fastframe_summary("AVERAGE")
    scope.horz.set_fastframe_multi_mode("OVERLAY")
    assert "HORizontal:FASTframe:SUMFrame AVErage" in mock.writes
    assert "HORizontal:FASTframe:MULtipleframes:MODe OVERlay" in mock.writes


def test_fastframe_ref_frame_and_include():
    scope, mock = connect_replay(TRACE)
    scope.horz.set_fastframe_ref_frame(1)
    scope.horz.set_fastframe_ref_include(True)
    assert "HORizontal:FASTframe:REF:FRAme 1" in mock.writes
    assert "HORizontal:FASTframe:REF:INCLUde ON" in mock.writes


def test_fastframe_timestamp_selector():
    scope, mock = connect_replay(TRACE)
    scope.horz.get_fastframe_timestamp("SELECTED")
    scope.horz.get_fastframe_timestamp("DELTA")
    assert "HORizontal:FASTframe:TIMEStamp:SELECTED?" in mock.writes
    assert "HORizontal:FASTframe:TIMEStamp:DELTa?" in mock.writes


def test_fastframe_timestamp_bad_selector_raises():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.horz.get_fastframe_timestamp("BOGUS")


def test_fastframe_xzero_selected():
    scope, mock = connect_replay(TRACE)
    scope.horz.get_fastframe_xzero("SELECTED")
    assert "HORizontal:FASTframe:XZEro:SELECTED?" in mock.writes


def test_data_frame_start_stop():
    scope, mock = connect_replay(TRACE)
    assert scope.data.set_frame_start(1) == 1
    assert scope.data.set_frame_stop(3) == 3
    assert "DATa:FRAMESTARt 1" in mock.writes
    assert "DATa:FRAMESTOP 3" in mock.writes


def test_data_frame_stop_max_token():
    # 'MAX\n' is the real echo the scope returns for a MAX stop (ADR-020 note).
    scope, mock = connect_replay(TRACE, extra={"DATa:FRAMESTOP?": "MAX\n"})
    assert scope.data.set_frame_stop("MAX") == "MAX"
    assert "DATa:FRAMESTOP MAX" in mock.writes


def test_data_frame_stop_bad_token_raises():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.data.set_frame_stop("HALF")


def test_history_timestamp_tables():
    scope, mock = connect_replay(TRACE)
    scope.horz.cmd_add_timestamp_table("TABLE1")
    assert 'TSTamptable:ADDNew "TABLE1"' in mock.writes
    assert scope.horz.get_timestamp_tables() == []  # NONE captured
    scope.horz.cmd_delete_timestamp_table("TABLE1")
    assert 'TSTamptable:DELETE "TABLE1"' in mock.writes
