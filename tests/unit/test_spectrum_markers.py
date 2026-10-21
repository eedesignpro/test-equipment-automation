"""Unit tests for the R20 Spectrum View peak-marker expansion.

Peak-list shape/values below are the real bench readback captured in
tests/captures/trig_sv_verification.json (SV:MARKER:PEAKS:FREQ?/AMPLITUDE?
for a 1 MHz AFG carrier into CH5: reference peak at 1.000000 MHz / +6.41 dBm,
side entries at ~896.5 kHz and ~1103.5 kHz window-skirt artifacts; peak count
5, threshold -50 dBm).
"""

import ast
import json
from pathlib import Path

import pytest

from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource

CAPTURES = Path(__file__).parent.parent / "captures" / "tek_mso"


def _connect(mock):
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    mock.responses.setdefault("*IDN?", lambda _c: "TEKTRONIX,MSO58,C0,CF:91")
    scope.connect()
    mock.writes.clear()
    return scope


def _load_verification():
    with open(CAPTURES / "trig_sv_verification.json") as fh:
        return json.load(fh)["results"]


def test_set_peak_count_command_and_readback():
    mock = MockVisaResource(responses={"SV:MARKER:PEAK:MAX?": "5"})
    scope = _connect(mock)
    result = scope.sv.set_peak_count(5)
    assert result == 5
    assert "SV:MARKER:PEAK:MAX 5" in mock.writes


def test_set_peak_threshold_command_and_readback():
    mock = MockVisaResource(responses={"SV:MARKER:PEAK:THR?": "-50"})
    scope = _connect(mock)
    result = scope.sv.set_peak_threshold(-50.0)
    assert result == -50.0
    assert "SV:MARKER:PEAK:THR -50" in mock.writes


def test_set_peak_excursion_command_and_readback():
    mock = MockVisaResource(responses={"SV:MARKER:PEAK:EXCUR?": "10"})
    scope = _connect(mock)
    result = scope.sv.set_peak_excursion(10.0)
    assert result == 10.0
    assert "SV:MARKER:PEAK:EXCUR 10" in mock.writes


def test_set_peak_markers_on_off_and_get_state():
    mock = MockVisaResource(responses={"SV:MARKER:PEAK:STATE?": "1"})
    scope = _connect(mock)
    scope.sv.set_peak_markers(True)
    assert "SV:MARKER:PEAK:STATE ON" in mock.writes
    assert scope.sv.get_peak_markers() is True
    scope.sv.set_peak_markers(False)
    assert "SV:MARKER:PEAK:STATE OFF" in mock.writes


@pytest.mark.parametrize("marker_type", ["ABSOLUTE", "DELTA"])
def test_set_marker_type_command(marker_type):
    mock = MockVisaResource()
    scope = _connect(mock)
    scope.sv.set_marker_type(marker_type)
    assert any(w.startswith("SV:MARKER:TYP") for w in mock.writes)


def test_set_marker_type_invalid_raises():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.sv.set_marker_type("BOGUS")


def test_get_peaks_parses_captured_reference_peak_list():
    """Reproduces the real get_peaks() readback from trig_sv_verification.json."""
    verification = _load_verification()
    captured_repr = verification["sv_get_peaks"]["result"]
    captured_peaks = ast.literal_eval(captured_repr)

    freqs = ",".join(str(f) for f, _ in captured_peaks)
    amps = ",".join(str(a) for _, a in captured_peaks)
    mock = MockVisaResource(
        responses={
            "SV:MARKER:PEAKS:FREQ?": freqs,
            "SV:MARKER:PEAKS:AMPLITUDE?": amps,
        }
    )
    scope = _connect(mock)

    peaks = scope.sv.get_peaks()

    assert len(peaks) == 3
    reference = peaks[1]
    assert reference[0] == pytest.approx(1000000.0)
    assert reference[1] == pytest.approx(6.4113416671753)


def test_get_peaks_empty_list_when_no_markers():
    mock = MockVisaResource(
        responses={"SV:MARKER:PEAKS:FREQ?": "", "SV:MARKER:PEAKS:AMPLITUDE?": ""}
    )
    scope = _connect(mock)
    assert scope.sv.get_peaks() == []


def test_get_reference_frequency_and_amplitude_match_capture():
    verification = _load_verification()
    ref_freq = float(verification["sv_ref_freq"]["result"])
    mock = MockVisaResource(
        responses={
            "SV:MARKER:REFER:FREQ?": str(ref_freq),
            "SV:MARKER:REFER:AMPLITUDE?": "6.41",
        }
    )
    scope = _connect(mock)
    assert scope.sv.get_reference_frequency() == pytest.approx(ref_freq)
    assert scope.sv.get_reference_amplitude() == pytest.approx(6.41)


def test_marker_to_center_sends_refer_command():
    mock = MockVisaResource()
    scope = _connect(mock)
    scope.sv.cmd_marker_to_center()
    assert "SV:MARKER:REFER" in mock.writes


def test_per_channel_units_and_trace_select_require_channel_context():
    """Per-channel SV settings must not be usable on the global sv node."""
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.sv.set_units("DBM")
    with pytest.raises(ValueError):
        scope.sv.set_select_trace("MAX HOLD")


def test_per_channel_units_and_trace_select_on_channel_sv_node():
    mock = MockVisaResource()
    scope = _connect(mock)
    scope.ch5.sv.set_units("DBM")
    assert "SV:CH5:UNI DBM" in mock.writes
    scope.ch5.sv.set_select_trace("MAX HOLD")
    assert any(w.startswith("SV:CH5:SELT") for w in mock.writes)


def test_burst_width_timeout_documented_limitation_matches_capture():
    """Cross-check the documented burst_width limitation against the real
    bench error text captured in trig_sv_verification.json."""
    verification = _load_verification()
    captured_error = verification["meas_burst_width_retest"]["error"]
    assert "no fresh populated result" in captured_error


def test_sv_vertical_scale_and_position_per_channel():
    mock = MockVisaResource(
        responses={
            "DISPLAY:SPECVIEW1:CH5:VERTICAL:SCALE?": "7",
            "DISPLAY:SPECVIEW1:CH5:VERTICAL:POSITION?": "-3.0000",
        }
    )
    scope = _connect(mock)
    assert scope.ch5.sv.set_scale(7) == 7
    assert "DISPLAY:SPECVIEW1:CH5:VERTICAL:SCALE 7" in mock.writes
    assert scope.ch5.sv.set_position(-3) == -3.0
    assert "DISPLAY:SPECVIEW1:CH5:VERTICAL:POSITION -3" in mock.writes


def test_sv_vertical_scale_requires_channel_node():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError, match="per-channel"):
        scope.sv.set_scale(10)
    with pytest.raises(ValueError, match="per-channel"):
        scope.sv.set_position(0)
