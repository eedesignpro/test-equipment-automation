"""Unit tests for the R18/R19 Vertical + Probe + Horizontal flesh-out."""

import pytest

from common.errors import HardwareNotPresentError
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
# R18: Vertical (label, deskew, offset, termination, probe queries)
# ---------------------------------------------------------------------------


def test_set_label_command_and_readback():
    mock = MockVisaResource(responses={"CH5:LAB:NAM?": '"Vout"'})
    scope = _connect(mock)
    result = scope.ch5.set_label("Vout")
    assert result == "Vout"
    assert 'CH5:LAB:NAM "Vout"' in mock.writes


def test_get_label_strips_quotes():
    mock = MockVisaResource(responses={"CH5:LAB:NAM?": '"Vout"'})
    scope = _connect(mock)
    assert scope.ch5.get_label() == "Vout"


def test_set_and_get_deskew():
    mock = MockVisaResource(responses={"CH5:DESK?": "1e-09"})
    scope = _connect(mock)
    result = scope.ch5.set_deskew(1e-9)
    assert result == pytest.approx(1e-9)
    assert "CH5:DESK 1e-09" in mock.writes
    assert scope.ch5.get_deskew() == pytest.approx(1e-9)


def test_get_offset_and_get_termination():
    mock = MockVisaResource(responses={"CH5:OFFS?": "0.1", "CH5:TER?": "50"})
    scope = _connect(mock)
    assert scope.ch5.get_offset() == 0.1
    assert scope.ch5.get_termination() == 50


def test_set_offset_and_termination_commands():
    mock = MockVisaResource(responses={"CH5:OFFS?": "0.1", "CH5:TER?": "1000000"})
    scope = _connect(mock)
    scope.ch5.set_offset(0.1)
    assert "CH5:OFFS 0.1" in mock.writes
    scope.ch5.set_termination(1e6)
    assert "CH5:TER 1000000" in mock.writes


def test_set_probe_control_and_bandwidth_optimization_choice_maps():
    mock = MockVisaResource()
    scope = _connect(mock)
    scope.ch5.set_probe_control("MANUAL")
    assert "CH5:PROBECO MANual" in mock.writes
    scope.ch5.set_bandwidth_optimization("FLATNESS")
    assert "CH5:BAN:FILT:OPTIMIZATION FLATNESS" in mock.writes
    with pytest.raises(ValueError):
        scope.ch5.set_probe_control("BOGUS")


def test_get_probe_type_direct():
    mock = MockVisaResource(responses={"CH5:PROBETYPE?": "ANALOG"})
    scope = _connect(mock)
    assert scope.ch5.get_probe_type() == "ANALOG"


@pytest.mark.parametrize(
    "method,cmd,resp,expected",
    [
        ("get_id", "CH5:PRO:ID?", '"C164798";"TPP1000"', ("C164798", "TPP1000")),
        ("get_type", "CH5:PRO:ID:TYP?", '"ANALOG"', "ANALOG"),
        ("get_serial", "CH5:PRO:ID:SER?", '"B012345"', "B012345"),
        ("get_gain", "CH5:PRO:GAIN?", "10", 10.0),
        ("get_units", "CH5:PRO:UNI?", '"V"', "V"),
        ("get_resistance", "CH5:PRO:RES?", "1000000", 1e6),
        ("get_status", "CH5:PRO:STAT?", '"OK"', "OK"),
        ("get_degauss_state", "CH5:PRO:DEGAUSS:STATE?", "NOTNEEDED", "NOTNEEDED"),
    ],
)
def test_probe_query_methods(method, cmd, resp, expected):
    mock = MockVisaResource(responses={cmd: resp})
    scope = _connect(mock)
    result = getattr(scope.ch5.probe, method)()
    assert result == expected


def test_probe_set_input_mode_and_forced_range():
    mock = MockVisaResource(responses={"CH5:PRO:INPUTM?": "DIFFERENTIAL", "CH5:PRO:FORCEDR?": "5"})
    scope = _connect(mock)
    scope.ch5.probe.set_input_mode("DIFFERENTIAL")
    assert "CH5:PRO:INPUTM DIFFERENTIAL" in mock.writes
    result = scope.ch5.probe.set_forced_range(5.0)
    assert result == 5.0


def test_probe_autozero_and_degauss_commands():
    mock = MockVisaResource()
    scope = _connect(mock)
    scope.ch5.probe.cmd_autozero()
    assert "CH5:PRO:AUTOZ EXECute" in mock.writes
    scope.ch5.probe.cmd_degauss()
    assert "CH5:PRO:DEGAUSS EXECute" in mock.writes


def test_probe_node_is_gated_by_channel_installed_state():
    scope = _connect(MockVisaResource(responses={"*IDN?": "TEKTRONIX,MSO44,C0,CF:91"}))
    assert scope.ch5.probe.installed is False
    with pytest.raises(HardwareNotPresentError):
        scope.ch5.probe.get_id()


# ---------------------------------------------------------------------------
# R19: Horizontal (div count, roll, acq duration, sample rate, delay mode)
# ---------------------------------------------------------------------------


def test_get_div_count():
    mock = MockVisaResource(responses={"HOR:DIV?": "10"})
    scope = _connect(mock)
    assert scope.horz.get_div_count() == 10


def test_get_roll_mode():
    mock = MockVisaResource(responses={"HOR:ROLL?": "AUTO"})
    scope = _connect(mock)
    assert scope.horz.get_roll() == "AUTO"


def test_get_acq_duration_and_sample_rate_actual():
    mock = MockVisaResource(responses={"HOR:ACQDURATION?": "1.6e-06", "HOR:SAMPLERate?": "6.25e9"})
    scope = _connect(mock)
    assert scope.horz.get_acq_duration() == pytest.approx(1.6e-6)
    assert scope.horz.get_sample_rate_actual() == pytest.approx(6.25e9)


def test_get_delay_mode_and_get_position_get_scale_get_delay():
    mock = MockVisaResource(
        responses={
            "HOR:DEL:MOD?": "1",
            "HOR:POS?": "50",
            "HOR:MOD:SCA?": "1e-06",
            "HOR:DEL:TIM?": "2e-06",
        }
    )
    scope = _connect(mock)
    assert scope.horz.get_delay_mode() is True
    assert scope.horz.get_position() == 50
    assert scope.horz.get_scale() == pytest.approx(1e-6)
    assert scope.horz.get_delay() == pytest.approx(2e-6)


def test_set_mode_choice_and_get_mode():
    mock = MockVisaResource(responses={"HOR:MODE?": "AUTO"})
    scope = _connect(mock)
    result = scope.horz.set_mode("AUTO")
    assert result == "AUTO"
    assert "HOR:MODE AUTO" in mock.writes
    with pytest.raises(ValueError):
        scope.horz.set_mode("BOGUS")
