"""Unit tests for R16 save/recall exports, R17 SV completion, R18 cursors,
R19 zoom, R20 DVM.

Query responses replay real MSO58 traffic captured in
tests/captures/exports_sv_cursor_zoom_dvm_trace.json plus the real exported
measurement-results CSV in tests/captures/measurement_table.csv (bench session
2026-07-23). DVM was NOT enabled on the bench unit, so its methods are gated
off (HardwareNotPresentError) rather than issuing SCPI that would hang.
"""

from pathlib import Path

import pytest

from common.errors import HardwareNotPresentError

from tests.mocks.capture_replay import connect_replay
from tests.mocks.mock_visa import MockVisaResource

TRACE = "tek_mso/exports_sv_cursor_zoom_dvm_trace.json"
CAPTURES = Path(__file__).parent.parent / "captures" / "tek_mso"
MEAS_CSV = (CAPTURES / "measurement_table.csv").read_bytes()


def _ldir(filename: str, size: int) -> str:
    return f'"{filename};FILE;{size};07/23/26 13:52:00"'


# --- R16 exports -----------------------------------------------------------


def test_image_composition_command_and_readback():
    scope, mock = connect_replay(TRACE)
    assert scope.file.set_image_composition("NORMAL") == "NORMAL"
    assert "SAVe:IMAGe:COMPosition NORMal" in mock.writes


def test_image_composition_invalid_raises():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.file.set_image_composition("SEPIA")


def test_get_measurement_table_roundtrip_contains_measurement():
    mock = MockVisaResource(
        responses={
            "*IDN?": "TEKTRONIX,MSO58,C0,CF:91",
            "FILESystem:DIR?": '"pyvisa_tmp"',
            "*OPC?": "1",
            "FILESystem:LDIR?": _ldir("Temp.csv", len(MEAS_CSV)),
        }
    )
    scope = Tektronix_MSO_connect(mock)
    mock.queue_read(MEAS_CSV)
    data = scope.file.get_measurement_table()
    assert data == MEAS_CSV
    assert b"Measurement Results" in data
    assert any(w.startswith("SAVe:EVENTtable:MEASUrement ") for w in mock.writes)


def test_get_peaks_table_and_plot_data_commands():
    mock = MockVisaResource(
        responses={
            "*IDN?": "TEKTRONIX,MSO58,C0,CF:91",
            "FILESystem:DIR?": '"pyvisa_tmp"',
            "*OPC?": "1",
            "FILESystem:LDIR?": _ldir("Temp.csv", 4),
        }
    )
    scope = Tektronix_MSO_connect(mock)
    mock.queue_read(b"abcd")
    scope.file.get_peaks_table()
    mock.queue_read(b"efgh")
    scope.file.get_plot_data()
    assert any(w.startswith("SAVe:EVENTtable:PEAKS ") for w in mock.writes)
    assert any(w.startswith("SAVe:PLOTData ") for w in mock.writes)


# --- R17 SV completion -----------------------------------------------------


def test_sv_span_queries_per_channel():
    scope, mock = connect_replay(TRACE)
    scope.ch5.sv.get_span_above_bw()
    scope.ch5.sv.get_span_below_dc()
    assert "CH5:SV:SPANABovebw?" in mock.writes
    assert "CH5:SV:SPANBELowdc?" in mock.writes


def test_sv_trace_selects_per_channel():
    scope, mock = connect_replay(TRACE)
    scope.ch5.sv.set_rf_normal(True)
    scope.ch5.sv.set_spectrogram(True)
    assert "SV:CH5:SELect:RF_NORMal ON" in mock.writes
    assert "SV:CH5:SELect:SPECtrogram ON" in mock.writes


def test_sv_lock_and_spectrogram_cscale():
    scope, mock = connect_replay(TRACE)
    scope.sv.set_lock_spectrum(False)
    lo, hi = scope.sv.set_spectrogram_cscale(-140, -40)
    assert "SV:LOCKSpectrum OFF" in mock.writes
    assert "SV:SPECtrogram:CSCale:MIN -140" in mock.writes
    assert "SV:SPECtrogram:CSCale:MAX -40" in mock.writes
    assert hi == pytest.approx(-40.0)


def test_sv_spectrogram_cursor_and_rf_phase_master():
    scope, mock = connect_replay(TRACE)
    scope.sv.set_spectrogram_cursor("A", True)
    scope.sv.set_rf_phase_reference_master("NONE")
    assert "SV:SPECtrogram:CURSor:A ON" in mock.writes
    assert "SV:RF_PHASe:REFerence:MASTer NONE" in mock.writes


def test_sv_spectrogram_cursor_bad_arg():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.sv.set_spectrogram_cursor("C", True)


def test_peaks_table_requires_table_name_format():
    scope, mock = connect_replay(TRACE)
    scope.sv.cmd_add_peaks_table("TABLE1")
    assert 'PEAKSTABle:ADDNew "TABLE1"' in mock.writes
    assert scope.sv.get_peaks_tables() == []  # NONE captured
    with pytest.raises(ValueError):
        scope.sv.cmd_add_peaks_table("PK1")


# --- R18 cursors -----------------------------------------------------------


def test_cursor_state_function_and_mode():
    scope, mock = connect_replay(TRACE)
    scope.cursors.set_state(True)
    scope.cursors.set_function("VBARS")
    scope.cursors.set_mode("INDEPENDENT")
    assert "DISplay:WAVEView1:CURSor:CURSOR1:STATE ON" in mock.writes
    assert "DISplay:WAVEView1:CURSor:CURSOR1:FUNCtion VBArs" in mock.writes
    assert "DISplay:WAVEView1:CURSor:CURSOR1:MODe INDEPENDENT" in mock.writes


def test_cursor_source_and_vbar_positions():
    scope, mock = connect_replay(TRACE)
    scope.cursors.set_source_a("CH5")
    scope.cursors.set_vbar_positions(-2e-7, 3e-7)
    assert "DISplay:WAVEView1:CURSor:CURSOR1:ASOUrce CH5" in mock.writes
    assert "DISplay:WAVEView1:CURSor:CURSOR1:VBArs:APOSition -2e-07" in mock.writes
    assert "DISplay:WAVEView1:CURSor:CURSOR1:VBArs:BPOSition 3e-07" in mock.writes


def test_cursor_vbar_delta_readback():
    scope, _ = connect_replay(TRACE)
    assert scope.cursors.get_vbar_delta() == pytest.approx(500e-9)


def test_cursor_invalid_function_raises():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.cursors.set_function("DIAGONAL")


# --- R19 zoom --------------------------------------------------------------


def test_zoom_state_and_scale_commands():
    scope, mock = connect_replay(TRACE)
    scope.zoom.set_state(True)
    scope.zoom.set_horizontal_scale(1e-7)
    scope.zoom.set_horizontal_position(50.0)
    scope.zoom.set_vertical_scale(1.0)
    assert "DISplay:WAVEView1:ZOOM:ZOOM1:STATe ON" in mock.writes
    assert "DISplay:WAVEView1:ZOOM:ZOOM1:HORizontal:SCALe 1e-07" in mock.writes
    assert "DISplay:WAVEView1:ZOOM:ZOOM1:HORizontal:POSition 50" in mock.writes
    assert "DISplay:WAVEView1:ZOOM:ZOOM1:VERTical:SCALe 1" in mock.writes


def test_zoom_indices():
    scope, mock = connect_replay(TRACE)
    # ZOOM2 was not part of the captured session; assert the indexed command is
    # built correctly (the empty readback on this mock then raises).
    with pytest.raises(Exception):  # noqa: B017
        scope.zoom.set_state(True, zoom=2, view=1)
    assert "DISplay:WAVEView1:ZOOM:ZOOM2:STATe ON" in mock.writes


# --- R20 DVM (not installed on bench unit) --------------------------------


def test_dvm_gated_off_when_not_installed():
    scope, _ = connect_replay(TRACE)
    scope.dvm.installed = False
    with pytest.raises(HardwareNotPresentError):
        scope.dvm.get_mode()
    with pytest.raises(HardwareNotPresentError):
        scope.dvm.get_value()


def test_dvm_commands_when_installed():
    scope, mock = connect_replay(TRACE, extra={"DVM:MODe?": "DC", "DVM:SOUrce?": "CH1"})
    scope.dvm.installed = True
    assert scope.dvm.set_mode("DC") == "DC"
    assert scope.dvm.set_source("CH1") == "CH1"
    scope.dvm.reset()
    assert "DVM:MODe DC" in mock.writes
    assert "DVM:SOUrce CH1" in mock.writes
    assert "DVM RESET" in mock.writes


def test_dvm_invalid_mode_raises():
    scope, _ = connect_replay(TRACE)
    scope.dvm.installed = True
    with pytest.raises(ValueError):
        scope.dvm.set_mode("PEAK")


def Tektronix_MSO_connect(mock):
    """Local connect helper mirroring the capture-replay connect for byte tests."""
    from drivers.tektronix_mso import Tektronix_MSO

    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    scope.connect()
    mock.writes.clear()
    return scope
