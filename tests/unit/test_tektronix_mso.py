"""Unit tests for the Tektronix_MSO driver (IDN parse, coupling, gating)."""

import pytest

from common.errors import HardwareNotPresentError, InstrumentError
from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource


def _connect_scope(monkeypatch, idn, extra_responses=None):
    responses = {"*IDN?": idn}
    if extra_responses:
        responses.update(extra_responses)
    mock = MockVisaResource(responses=responses)
    scope = Tektronix_MSO("MOCK::INSTR")
    monkeypatch.setattr(scope.link, "_open_resource", lambda rm, name: mock)
    scope.connect()
    return scope, mock


@pytest.mark.parametrize(
    "idn,model,count",
    [
        ("TEKTRONIX,MSO58,C012345,CF:91.1CT FV:1.2.3", "MSO58", 8),
        ("TEKTRONIX,MSO44,C0,CF:91", "MSO44", 4),
        ("TEKTRONIX,MSO46,C0,CF:91", "MSO46", 6),
        ("TEKTRONIX,MSO54,C0,CF:91", "MSO54", 4),
        ("TEKTRONIX,MSO64,C0,CF:91", "MSO64", 4),
        ("TEKTRONIX,MSO58B,C0,CF:91", "MSO58B", 8),
    ],
)
def test_idn_parse_channel_count(monkeypatch, idn, model, count):
    scope, _ = _connect_scope(monkeypatch, idn)
    assert scope.model == model
    assert scope.num_channels == count


def test_channel_install_gating(monkeypatch):
    scope, _ = _connect_scope(monkeypatch, "TEKTRONIX,MSO44,C0,CF:91")
    assert scope.ch1.installed and scope.ch4.installed
    assert not scope.ch5.installed
    assert not scope.ch8.installed
    assert not scope.ch5.sv.installed
    with pytest.raises(HardwareNotPresentError):
        scope.ch5.set_scale(0.1)


def test_set_coupling_sends_coup_command(monkeypatch):
    scope, mock = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91")
    scope.ch5.set_coupling("DC REJECT")
    assert mock.writes[-1] == "CH5:COUP DCR"
    scope.ch1.set_coupling("AC")
    assert mock.writes[-1] == "CH1:COUP AC"


def test_set_coupling_invalid_choice_lists_valid_options(monkeypatch):
    scope, _ = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91")
    with pytest.raises(ValueError) as exc:
        scope.ch1.set_coupling("BOGUS")
    msg = str(exc.value)
    assert "AC" in msg and "DC" in msg and "DC_REJECT" in msg


def test_set_scale_readback(monkeypatch):
    scope, mock = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91", {"CH5:SCA?": "0.2"})
    result = scope.ch5.set_scale(0.2)
    assert result == 0.2
    assert "CH5:SCA 0.2" in mock.writes


def test_acquisition_set_type_uses_choice(monkeypatch):
    scope, mock = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91")
    scope.acq.set_type("HIRES")
    assert mock.writes[-1] == "ACQ:MODE HIR"
    with pytest.raises(ValueError):
        scope.acq.set_type("BOGUS")


def test_acquisition_single_and_state(monkeypatch):
    scope, mock = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91", {"ACQ:STATE?": "0"})
    scope.acq.cmd_single()
    assert "ACQ:STOPA SEQ" in mock.writes
    assert "ACQ:SEQ:NUMSEQ 1" in mock.writes
    assert "ACQ:STATE RUN" in mock.writes
    assert scope.acq.get_state() == 0


def test_spectrum_window_command(monkeypatch):
    scope, mock = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91")
    scope.sv.set_window("BLACKMAN")
    assert mock.writes[-1] == "SV:WINDOW BLAC"


def test_spectrum_window_invalid_choice(monkeypatch):
    scope, _ = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91")
    with pytest.raises(ValueError) as exc:
        scope.sv.set_window("BOGUS")
    assert "KAISER" in str(exc.value)


def test_spectrum_center_span_rbw(monkeypatch):
    scope, mock = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91")
    scope.sv.set_center(1e6)
    assert "CH1:SV:CENTERF 1000000" in mock.writes
    scope.sv.set_span(2e6)
    assert mock.writes[-1] == "SV:SPAN 2000000"
    scope.sv.set_rbw(1000)
    assert "SV:RBWM MAN" in mock.writes
    assert mock.writes[-1] == "SV:RBW 1000"


def test_trigger_set_edge_slope_both_sends_either(monkeypatch):
    """Bug-fix regression: TRIG:A:EDGE:SLO must send EITHER for 'BOTH'."""
    scope, mock = _connect_scope(
        monkeypatch, "TEKTRONIX,MSO58,C0,CF:91", {"TRIG:A:LEV:CH5?": "0.1"}
    )
    scope.trig.set_edge("CH5", level=0.1, slope="BOTH", coupling="DC")
    assert "TRIG:A:EDGE:SLO EITHER" in mock.writes


def test_trigger_set_edge_rising_and_falling(monkeypatch):
    scope, mock = _connect_scope(
        monkeypatch, "TEKTRONIX,MSO58,C0,CF:91", {"TRIG:A:LEV:CH5?": "0.1"}
    )
    scope.trig.set_edge("CH5", level=0.1, slope="RISING")
    assert "TRIG:A:EDGE:SLO RISE" in mock.writes
    scope.trig.set_edge("CH5", level=0.1, slope="FALLING")
    assert "TRIG:A:EDGE:SLO FALL" in mock.writes


def test_trigger_set_edge_invalid_slope(monkeypatch):
    scope, _ = _connect_scope(monkeypatch, "TEKTRONIX,MSO58,C0,CF:91", {"TRIG:A:LEV:CH5?": "0.1"})
    with pytest.raises(ValueError):
        scope.trig.set_edge("CH5", level=0.1, slope="BOGUS")


def test_trigger_set_edge_returns_readback_level(monkeypatch):
    scope, mock = _connect_scope(
        monkeypatch, "TEKTRONIX,MSO58,C0,CF:91", {"TRIG:A:LEV:CH5?": "0.25"}
    )
    level = scope.trig.set_edge("CH5", level=0.25, slope="RISING", coupling="DC")
    assert level == 0.25
    assert "TRIG:A:LEV:CH5 0.25" in mock.writes
    assert "TRIG:A:TYP EDGE" in mock.writes


def test_bad_idn_raises(monkeypatch):
    with pytest.raises(RuntimeError):
        _connect_scope(monkeypatch, "SOMEVENDOR,WIDGET,0,0")


def test_check_errors_uses_esr_event_queue(monkeypatch):
    scope, mock = _connect_scope(
        monkeypatch,
        "TEKTRONIX,MSO58,C0,CF:91",
        extra_responses={"*ESR?": "32", "ALLEV?": '113,"Undefined header; Command not found"'},
    )
    with pytest.raises(InstrumentError) as exc:
        scope.check_errors()
    assert "ESR=32" in str(exc.value)
    assert "Undefined header" in str(exc.value)
    assert "SYST:ERR?" not in mock.writes


def test_check_errors_clean_and_opc_bit_ignored(monkeypatch):
    scope, _ = _connect_scope(
        monkeypatch, "TEKTRONIX,MSO58,C0,CF:91", extra_responses={"*ESR?": "0"}
    )
    scope.check_errors()
    scope2, _ = _connect_scope(
        monkeypatch, "TEKTRONIX,MSO58,C0,CF:91", extra_responses={"*ESR?": "1"}
    )
    scope2.check_errors()
