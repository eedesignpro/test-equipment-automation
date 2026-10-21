"""Unit tests for R7 (per-channel SV center/spectrum-time/squelch) and
R8 (HORizontal:MODe:MANual:CONFIGure, ADR-014).

``capabilities_data_sv_trace.json`` is a results-summary trace (no raw
cmd/resp pairs), so these pin the exact SCPI strings and readbacks recorded
there (sv5_channel_center=1e6, sv5_spectrum_time=50.0, sv5_squelch_state=True/
threshold=0.5, hor_manual_config_default/set_recordlength/set_scale).
"""

import pytest

from drivers.tektronix_mso import Tektronix_MSO
from drivers.tektronix_mso.enums import HorizontalManualConfig

from tests.mocks.mock_visa import MockVisaResource


def _connect(responses=None):
    mock = MockVisaResource(responses=responses or {})
    mock.responses.setdefault("*IDN?", "TEKTRONIX,MSO58,C0,CF:91")
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    scope.connect()
    mock.writes.clear()
    return scope, mock


def test_set_channel_center_unlocks_then_writes_ch_centerf():
    scope, mock = _connect({"CH5:SV:CENTERF?": "1000000.0"})
    result = scope.ch5.sv.set_channel_center(1e6)
    assert mock.writes[:2] == ["SV:LOCKC OFF", "CH5:SV:CENTERF 1e+06"]
    assert result == pytest.approx(1e6)


def test_set_center_locks_and_writes_ch1_only():
    """Global set_center locks all channels together via CH1 (distinct
    command from the per-channel set_channel_center)."""
    scope, mock = _connect()
    scope.ch1.sv.set_center(2e6)
    assert mock.writes == ["SV:LOCKC ON", "CH1:SV:CENTERF 2e+06"]


def test_set_spectrum_time_is_distinct_from_display_position():
    """ADR-013: CH<x>:SV:POSition (spectrum time) vs DISPLAY:SPECVIEW position."""
    scope, mock = _connect(
        {
            "CH5:SV:POSition?": "50.0",
            "DISPLAY:SPECVIEW1:CH5:VERTICAL:POSITION?": "1.5",
        }
    )
    result = scope.ch5.sv.set_spectrum_time(50.0)
    assert mock.writes[:2] == ["CH5:SV:POSition 50", "CH5:SV:POSition?"]
    assert result == pytest.approx(50.0)
    mock.writes.clear()

    scope.ch5.sv.set_position(1.5)
    assert mock.writes[0] == "DISPLAY:SPECVIEW1:CH5:VERTICAL:POSITION 1.5"


def test_squelch_state_and_threshold_command_strings():
    scope, mock = _connect({"SV:CH5:SQUELCH:STATE?": "1", "SV:CH5:SQUELCH:THReshold?": "0.5"})
    state = scope.ch5.sv.set_squelch_state(True)
    threshold = scope.ch5.sv.set_squelch_threshold(0.5)
    assert "SV:CH5:SQUELCH:STATE ON" in mock.writes
    assert "SV:CH5:SQUELCH:THReshold 0.5" in mock.writes
    assert state is True
    assert threshold == pytest.approx(0.5)


def test_squelch_state_off():
    scope, mock = _connect({"SV:CH5:SQUELCH:STATE?": "0"})
    assert scope.ch5.sv.set_squelch_state(False) is False
    assert "SV:CH5:SQUELCH:STATE OFF" in mock.writes


@pytest.mark.parametrize(
    "target,scpi,readback",
    [
        ("SCALE", "HORIZontalscale", "HORIZONTALSCALE"),
        ("RECORD LENGTH", "RECORDLength", "RECORDLENGTH"),
    ],
)
def test_horizontal_manual_config_set(target, scpi, readback):
    scope, mock = _connect({"HOR:MOD:MAN:CONFIG?": readback})
    result = scope.horz.set_manual_config(target)
    assert mock.writes[0] == f"HOR:MOD:MAN:CONFIG {scpi}"
    assert result == readback


def test_horizontal_manual_config_default_is_horizontalscale():
    """Bench default per ADR-014/capture: HORIZONTALSCALE."""
    scope, _ = _connect({"HOR:MOD:MAN:CONFIG?": "HORIZONTALSCALE"})
    assert scope.horz.get_manual_config() is HorizontalManualConfig.SCALE


def test_horizontal_manual_config_does_not_write_hor_mode():
    """A preference, not an action: must not touch HOR:MODE as a side effect."""
    scope, mock = _connect({"HOR:MOD:MAN:CONFIG?": "RECORDLENGTH"})
    scope.horz.set_manual_config("RECORD LENGTH")
    assert not any(w.startswith("HOR:MODE") for w in mock.writes)
