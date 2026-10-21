"""Unit tests for the AFG subsystem (`_Afg`, `scope.afg`).

``tests/captures/afg_subsystem_trace.json`` is a step-summary trace (labels and
results only, no raw cmd/resp pairs), so it cannot be replayed with
``tests/mocks/capture_replay``. These tests instead pin the exact SCPI command
strings and assert the helpers return the bench-observed read-back values
recorded in that trace (ADR-011), driving a ``MockVisaResource`` whose scripted
responses are those captured values.

The AFG was ported to the design standard on 2026-08-06 (ADR-041/043). The
function choice now lives in the method name, so there is no `set_function`.
Helpers write only the values that differ from the present setting, because a
query costs 0.74 ms and a synchronized write costs about 135 ms
(bench-verified). Tests therefore assert on the set of writes, not on
``writes[0]``, and prime every read-back the helper checks.
"""

import inspect

import pytest

from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource

# Bench-verified read-back defaults. A helper queries before it writes, so
# every value it may compare against has to answer something parseable.
_DEFAULTS = {
    "AFG:FUNCtion?": "SINE",
    "AFG:FREQuency?": "1000.0",
    "AFG:AMPLitude?": "1.0",
    "AFG:OFFSet?": "0.0",
    "AFG:SQUare:DUty?": "50.0",
    "AFG:RAMP:SYMmetry?": "50.0",
    "AFG:PULse:WIDth?": "1e-06",
    "AFG:NOISEAdd:STATE?": "0",
    "AFG:NOISEAdd:PERCent?": "0.0",
    "AFG:OUTPut:STATE?": "1",
    "AFG:HIGHLevel?": "0.5",
    "AFG:LOWLevel?": "-0.5",
    "AFG:OUTPut:LOAd:IMPEDance?": "HIGHZ",
    "AFG:OUTPut:MODe?": "CONTINUOUS",
    "AFG:BURSt:CCOUnt?": "1",
    "AFG:ARBitrary:SOUrce?": "",
}


def _writes(mock):
    """Only the commands that changed state. MockVisaResource logs queries too."""
    return [w for w in mock.writes if not w.rstrip().endswith("?")]


def _connect(responses=None):
    mock = MockVisaResource(responses=responses or {})
    mock.responses.setdefault("*IDN?", "TEKTRONIX,MSO58,C0,CF:91")
    # Keep the connect-time afg/dvm runtime probes (ADR-012) clean so
    # scope.afg.installed is True by the time the test body runs.
    mock.responses.setdefault("AFG:FUNCTION?", "SINE")
    mock.responses.setdefault("DVM:MODE?", "OFF")
    mock.responses.setdefault("*ESR?", "0")
    mock.responses.setdefault("ALLEV?", '0,"No events to report - queue empty"')
    for key, value in _DEFAULTS.items():
        mock.responses.setdefault(key, value)
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    scope.connect()
    mock.writes.clear()
    return scope, mock


# The 14 function tokens, each reachable through its own method. Bench-verified
# accepted and read back as the full uppercase word.
_WAVEFORM_CASES = [
    ("set_waveform_sine", "SINE", "SINE"),
    ("set_waveform_square", "SQUARE", "SQUare"),
    ("set_waveform_pulse", "PULSE", "PULSe"),
    ("set_waveform_ramp", "RAMP", "RAMP"),
    ("set_waveform_sinc", "SINC", "SINC"),
    ("set_waveform_gaussian", "GAUSSIAN", "GAUSsian"),
    ("set_waveform_lorentz", "LORENTZ", "LORENtz"),
    ("set_waveform_exp_rise", "ERISE", "ERISe"),
    ("set_waveform_exp_decay", "EDECAY", "EDECAy"),
    ("set_waveform_haversine", "HAVERSINE", "HAVERSINe"),
    ("set_waveform_cardiac", "CARDIAC", "CARDIac"),
]


@pytest.mark.parametrize("method,token,scpi", _WAVEFORM_CASES)
def test_waveform_method_writes_its_own_function(method, token, scpi):
    """The method name is the function select. No string names a shape.

    The instrument starts on a different function so the write is observable.
    A helper skips the write when the function already matches.
    """
    scope, mock = _connect({"AFG:FUNCtion?": "DC"})
    getattr(scope.afg, method)(2e3, 1.5)
    assert f"AFG:FUNCtion {scpi}" in _writes(mock)


@pytest.mark.parametrize("method,token,scpi", _WAVEFORM_CASES)
def test_waveform_method_reports_its_own_function(method, token, scpi):
    scope, _ = _connect({"AFG:FUNCtion?": token})
    assert getattr(scope.afg, method)(2e3, 1.5).function == token


def test_noise_waveform_selects_noise():
    scope, mock = _connect({"AFG:FUNCtion?": "DC"})
    scope.afg.set_waveform_noise(1.5)
    assert "AFG:FUNCtion NOISe" in _writes(mock)


def test_dc_waveform_selects_dc():
    scope, mock = _connect({"AFG:FUNCtion?": "SINE"})
    scope.afg.set_waveform_dc(0.5)
    assert "AFG:FUNCtion DC" in _writes(mock)


def test_arb_file_sets_source_then_function():
    """ADR-011: the scope does not validate file existence. The path is quoted."""
    scope, mock = _connect({"AFG:FUNCtion?": "SINE", "AFG:ARBitrary:SOUrce?": ""})
    scope.afg.set_waveform_arb_file("C:/pyvisa_tmp/nope.wfm", 1e3, 1.0)
    sent = _writes(mock)
    assert 'AFG:ARBitrary:SOUrce "C:/pyvisa_tmp/nope.wfm"' in sent
    assert "AFG:FUNCtion ARBitrary" in sent
    assert sent.index('AFG:ARBitrary:SOUrce "C:/pyvisa_tmp/nope.wfm"') < sent.index(
        "AFG:FUNCtion ARBitrary"
    )


def test_arb_file_default_is_empty_string():
    """afg_subsystem_trace.json: arb_source_query_default -> ""."""
    scope, _ = _connect({"AFG:ARBitrary:SOUrce?": ""})
    assert scope.afg.get_arb_file() == ""


# ---------- minimal-write behavior (bench-verified cost asymmetry) ----------


def test_helper_skips_a_write_whose_value_already_matches():
    """A query costs 0.74 ms and a write about 135 ms, so equal values skip."""
    scope, mock = _connect()
    scope.afg.set_waveform_sine(1000.0, 1.0, offset=0.0)
    assert _writes(mock) == []


def test_helper_writes_only_the_value_that_differs():
    scope, mock = _connect()
    scope.afg.set_waveform_sine(2000.0, 1.0, offset=0.0)
    assert _writes(mock) == ["AFG:FREQuency 2000"]


def test_helper_rejects_an_unknown_function_token():
    scope, _ = _connect()
    with pytest.raises(ValueError):
        scope.afg._apply("NOT_A_FUNCTION")


# ---------- additive noise ----------


def test_noise_add_value_enables_and_sets_percent():
    scope, mock = _connect({"AFG:NOISEAdd:PERCent?": "0.0", "AFG:NOISEAdd:STATE?": "0"})
    scope.afg.set_waveform_sine(1000.0, 1.0, noise_add=10.0)
    assert "AFG:NOISEAdd:PERCent 10" in mock.writes
    assert "AFG:NOISEAdd:STATE ON" in mock.writes


def test_noise_add_false_disables():
    scope, mock = _connect({"AFG:NOISEAdd:STATE?": "1"})
    scope.afg.set_waveform_sine(1000.0, 1.0, noise_add=False)
    assert "AFG:NOISEAdd:STATE OFF" in mock.writes


def test_noise_add_none_leaves_noise_untouched():
    scope, mock = _connect()
    scope.afg.set_waveform_sine(1000.0, 1.0)
    assert not [w for w in _writes(mock) if w.startswith("AFG:NOISEAdd")]


@pytest.mark.parametrize(
    "method",
    [
        "set_waveform_sine",
        "set_waveform_square",
        "set_waveform_pulse",
        "set_waveform_ramp",
        "set_waveform_arb_file",
        "set_waveform_cardiac",
    ],
)
def test_noise_add_argument_exists_where_hardware_accepts_it(method):
    scope, _ = _connect()
    assert "noise_add" in inspect.signature(getattr(scope.afg, method)).parameters


@pytest.mark.parametrize("method", ["set_waveform_dc", "set_waveform_noise"])
def test_noise_add_argument_absent_where_hardware_declines_it(method):
    """DC and NOISE refuse additive noise and report no error (bench-verified).

    The signature is the only warning a caller gets, so the argument must not
    exist on these two methods.
    """
    scope, _ = _connect()
    assert "noise_add" not in inspect.signature(getattr(scope.afg, method)).parameters


# ---------- state read-back ----------


def test_get_waveform_reports_duty_for_square_only():
    scope, _ = _connect({"AFG:FUNCtion?": "SQUARE", "AFG:SQUare:DUty?": "40.0"})
    state = scope.afg.get_waveform()
    assert state.duty == pytest.approx(40.0)
    assert state.symmetry is None
    assert state.width is None


def test_get_waveform_reports_symmetry_for_ramp_only():
    scope, _ = _connect({"AFG:FUNCtion?": "RAMP", "AFG:RAMP:SYMmetry?": "25.0"})
    state = scope.afg.get_waveform()
    assert state.symmetry == pytest.approx(25.0)
    assert state.duty is None


def test_get_waveform_reports_noise_add_as_none_when_off():
    scope, _ = _connect({"AFG:NOISEAdd:STATE?": "0"})
    assert scope.afg.get_waveform().noise_add is None


# ---------- atomic setters kept public as justified exceptions ----------


@pytest.mark.parametrize(
    "method,cmd,value,readback",
    [
        ("set_frequency", "AFG:FREQuency", 1e6, 1000000.0),
        ("set_amplitude", "AFG:AMPLitude", 1.0, 1.0),
        ("set_offset", "AFG:OFFSet", 0.1, 0.1),
    ],
)
def test_atomic_setter_sends_command_and_returns_bench_readback(method, cmd, value, readback):
    scope, mock = _connect({f"{cmd}?": str(readback)})
    result = getattr(scope.afg, method)(value)
    assert mock.writes[0].startswith(cmd)
    assert result == pytest.approx(readback)


def test_set_levels_writes_both_and_returns_the_pair():
    scope, mock = _connect({"AFG:HIGHLevel?": "0.6", "AFG:LOWLevel?": "-0.6"})
    high, low = scope.afg.set_levels(0.6, -0.6)
    assert "AFG:HIGHLevel 0.6" in mock.writes
    assert "AFG:LOWLevel -0.6" in mock.writes
    assert (high, low) == pytest.approx((0.6, -0.6))


# ---------- output, load, burst ----------


def test_set_output_state_on():
    scope, mock = _connect({"AFG:OUTPut:STATE?": "1"})
    assert scope.afg.set_output_state(True) is True
    assert mock.writes[0] == "AFG:OUTPut:STATE ON"


def test_set_output_state_off():
    scope, mock = _connect({"AFG:OUTPut:STATE?": "0"})
    assert scope.afg.set_output_state(False) is False
    assert mock.writes[0] == "AFG:OUTPut:STATE OFF"


@pytest.mark.parametrize(
    "value,scpi,expected",
    [
        (50, "FIFty", 50.0),
        (float("inf"), "HIGHZ", float("inf")),
        ("HIGHZ", "HIGHZ", float("inf")),
    ],
)
def test_set_load_accepts_ohms_or_token(value, scpi, expected):
    readback = "FIFTY" if scpi == "FIFty" else "HIGHZ"
    scope, mock = _connect({"AFG:OUTPut:LOAd:IMPEDance?": readback})
    assert scope.afg.set_load(value) == expected
    assert f"AFG:OUTPut:LOAd:IMPEDance {scpi}" in mock.writes


def test_set_load_rejects_an_unsupported_resistance():
    scope, _ = _connect()
    with pytest.raises(ValueError):
        scope.afg.set_load(600)


def test_load_enum_is_total_and_reachable_from_the_instance():
    """The enum carries every accepted value, so no caller needs a raw string."""
    scope, _ = _connect()
    # Reached from the instrument handle, not the node: an operator working
    # with an instrument has the whole instrument in scope, and one path to
    # an enum beats two (docs/hardware-not-software.md section 2).
    assert {m.name for m in scope.Load} == {"FIFTY", "HIGHZ"}


def test_set_burst_switches_mode_and_sets_count():
    scope, mock = _connect({"AFG:BURSt:CCOUnt?": "5", "AFG:OUTPut:MODe?": "BURST"})
    assert scope.afg.set_burst(5) == 5
    assert "AFG:OUTPut:MODe BURSt" in mock.writes
    assert "AFG:BURSt:CCOUnt 5" in mock.writes


def test_set_continuous_returns_the_mode_token():
    scope, mock = _connect({"AFG:OUTPut:MODe?": "CONTINUOUS"})
    assert scope.afg.set_continuous() == "CONTINUOUS"
    assert "AFG:OUTPut:MODe CONTinuous" in mock.writes


def test_cmd_burst_trigger_is_set_only_no_query():
    scope, mock = _connect()
    scope.afg.cmd_burst_trigger()
    assert mock.writes == ["AFG:BURSt:TRIGger"]
