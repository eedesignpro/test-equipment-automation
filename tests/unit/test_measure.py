"""Unit tests for the ``meas`` (_Measure) subsystem.

Dispatcher command sequence and timing are replayed from the real bench
capture ``tests/captures/meas_dispatcher_trace.json`` (frequency/amplitude/
pk2pk on CH5, per ADR-007's fresh-acquisition wait). Named-wrapper coverage
is cross-checked against the type tokens implemented in the driver (which
were themselves verified live per ``tests/captures/meas_wrapper_enumeration.
json`` / ADR-008 / ADR-010) - no synthetic instrument *behavior* is invented;
only the timeout-raise path uses a controlled (never-populating) response,
which is pure loop/guard logic per the testing policy.
"""

import json
from collections import deque
from pathlib import Path

import pytest

from common.errors import InstrumentTimeoutError
from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource

CAPTURES = Path(__file__).parent.parent / "captures" / "tek_mso"


def _load_trace():
    with open(CAPTURES / "meas_dispatcher_trace.json") as fh:
        return json.load(fh)


def _queue_mock(trace_entries, extra=None):
    """Build a MockVisaResource that answers each query from an ordered queue.

    Multiple occurrences of the same command are answered in the captured
    order (a per-command FIFO), exactly reproducing the real session.
    """
    queues: dict[str, deque] = {}
    for entry in trace_entries:
        if entry["dir"] == "query":
            queues.setdefault(entry["cmd"], deque()).append(entry["resp"])
    if extra:
        for cmd, resp in extra.items():
            queues.setdefault(cmd, deque()).append(resp)

    def responder(cmd):
        q = queues.get(cmd)
        if q:
            return q.popleft()
        return ""

    return MockVisaResource(responses={cmd: responder for cmd in queues})


def _connect(mock):
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    mock.responses.setdefault("*IDN?", lambda _c: "TEKTRONIX,MSO58,C0,CF:91")
    scope.connect()
    mock.writes.clear()  # drop *IDN?/*OPT? connect-time chatter for clean assertions
    return scope


def test_generic_scpi_is_alias_for_custom():
    trace = _load_trace()
    meas1 = [e for e in trace["trace"][:13]]
    mock = _queue_mock(meas1)
    scope = _connect(mock)

    value = scope.meas.generic_scpi("FREQUENCY", "CH5", num=1)

    assert value == pytest.approx(999999.8730412109)
    expected_cmds = [e["cmd"] for e in meas1]
    assert mock.writes == expected_cmds


def test_custom_dispatcher_command_sequence_meas1_frequency():
    """Replays the exact MEAS1 (frequency) segment of the captured trace."""
    trace = _load_trace()
    meas1 = trace["trace"][:13]
    mock = _queue_mock(meas1)
    scope = _connect(mock)

    value = scope.meas.custom("FREQUENCY", "CH5", num=1)

    assert value == pytest.approx(trace["meas"]["frequency_hz"])
    expected_cmds = [e["cmd"] for e in meas1]
    assert mock.writes == expected_cmds
    # ADR-007: type set + SOUrce, then *OPC?, before the fresh-acquisition poll.
    assert expected_cmds[1] == 'MEASUrement:ADDNew "MEAS1"'
    assert expected_cmds[2] == "MEASUrement:MEAS1:TYPe FREQUENCY"
    assert expected_cmds[3] == "MEASUrement:MEAS1:SOUrce CH5"
    assert expected_cmds[4] == "*OPC?"
    assert expected_cmds[-1] == "MEASUrement:MEAS1:RESUlts:CURRentacq:MEAN?"


def test_custom_dispatcher_meas2_amplitude_persists_slot():
    trace = _load_trace()
    # MEAS2 segment: LIST? "MEAS1" .. through MEAN? (indices 13-27 inclusive).
    meas2 = trace["trace"][13:28]
    mock = _queue_mock(meas2)
    scope = _connect(mock)

    value = scope.meas.custom("AMPLITUDE", "CH5", num=2, persist=True)

    assert value == pytest.approx(trace["meas"]["amplitude_v"])
    assert 'MEASUrement:DELete "MEAS2"' not in mock.writes


def test_custom_dispatcher_meas3_pk2pk_deletes_when_not_persisted():
    trace = _load_trace()
    meas3 = trace["trace"][28:]
    mock = _queue_mock(meas3)
    scope = _connect(mock)

    value = scope.meas.custom("PK2PK", "CH5", num=3, persist=False)

    assert value == pytest.approx(trace["meas"]["pk2pk_v"])
    assert mock.writes[-1] == 'MEASUrement:DELete "MEAS3"'


def test_wait_for_fresh_population_polls_population_then_acq_numac():
    """Verifies polling order matches the captured trace: POPUlation? then NUMAC?."""
    trace = _load_trace()
    meas1 = trace["trace"][:13]
    poll_order = [
        e["cmd"]
        for e in meas1
        if e["cmd"] in ("MEASUrement:MEAS1:RESUlts:CURRentacq:POPUlation?", "ACQ:NUMAC?")
    ]
    # baseline NUMAC?, then (POPU, NUMAC) x3
    assert poll_order[0] == "ACQ:NUMAC?"
    assert (
        poll_order[1:]
        == [
            "MEASUrement:MEAS1:RESUlts:CURRentacq:POPUlation?",
            "ACQ:NUMAC?",
        ]
        * 3
    )


def test_custom_raises_timeout_when_population_never_arrives():
    """Timeout-raise is pure loop/guard logic; population is pinned at 0.

    Real never-populating behavior is confirmed on hardware for BURSTWIDTH on
    a continuous waveform (tests/captures/meas_wrapper_enumeration.json /
    trig_sv_verification.json: "MEAS1 produced no fresh populated result
    after 5s").
    """
    responses = {
        "MEASUrement:LIST?": "NONE",
        "MEASUrement:MEAS1:RESUlts:CURRentacq:POPUlation?": "0",
        "ACQ:NUMAC?": "5",
    }
    mock = MockVisaResource(responses=responses)
    scope = _connect(mock)

    with pytest.raises(InstrumentTimeoutError) as exc:
        scope.meas.custom("BURSTWIDTH", "CH5", num=1, timeout=0.05)
    assert "MEAS1" in str(exc.value)


def test_is_meas_active_and_delete():
    mock = MockVisaResource(responses={"MEASUrement:LIST?": "MEAS1,MEAS2"})
    scope = _connect(mock)
    assert scope.meas.get_meas_active(1) is True
    assert scope.meas.get_meas_active(3) is False
    scope.meas.cmd_delete(2)
    assert 'MEASUrement:DELete "MEAS2"' in mock.writes


# ---------------------------------------------------------------------------
# Named-wrapper TYPe token spot checks (R16), organized per sub-node.
# Physical correctness of these wrappers on live hardware is captured in
# tests/captures/meas_wrapper_enumeration.json (39/40 physically correct;
# burst_width documented as requiring a bursted signal, see docstring).
# ---------------------------------------------------------------------------

_VERT_TOKENS = [
    ("amplitude", "AMPLITUDE"),
    ("maximum", "MAXIMUM"),
    ("minimum", "MINIMUM"),
    ("peak_to_peak", "PK2PK"),
    ("top", "TOP"),
    ("base", "BASE"),
    ("high", "HIGH"),
    ("low", "LOW"),
    ("mean", "MEAN"),
    ("rms", "RMS"),
    ("ac_rms", "ACRMS"),
    ("area", "AREA"),
    ("overshoot_pos", "POVERSHOOT"),
    ("overshoot_neg", "NOVERSHOOT"),
]

_TIME_TOKENS = [
    ("frequency", "FREQUENCY"),
    ("period", "PERIOD"),
    ("rise_time", "RISETIME"),
    ("fall_time", "FALLTIME"),
    ("rise_slew_rate", "RISESLEWRATE"),
    ("fall_slew_rate", "FALLSLEWRATE"),
    ("width_pos", "PWIDTH"),
    ("width_neg", "NWIDTH"),
    ("duty_pos", "PDUTY"),
    ("duty_neg", "NDUTY"),
    ("burst_width", "BURSTWIDTH"),
    ("high_time", "HIGHTIME"),
    ("low_time", "LOWTIME"),
    ("n_period", "NPERIOD"),
    ("data_rate", "DATARATE"),
    ("time_outside_level", "TIMEOUTSIDELEVEL"),
    ("unit_interval", "UNITINTERVAL"),
    ("time_to_max", "TIMETOMAX"),
    ("time_to_min", "TIMETOMIN"),
]

_DUAL_TOKENS = [
    ("delay", "DELAY"),
    ("phase", "PHASE"),
    ("skew", "SKEW"),
    ("setup", "SETUP"),
    ("hold", "HOLD"),
    ("common_mode", "COMMONMODE"),
    ("ac_common_mode", "ACCOMMONMODE"),
]

_LEGACY_TOKENS = [
    ("frequency", "FREQUENCY"),
    ("amplitude", "AMPLITUDE"),
    ("maximum", "MAXIMUM"),
    ("minimum", "MINIMUM"),
    ("peak_to_peak", "PK2PK"),
    ("period", "PERIOD"),
    ("pulse_width_pos", "PWIDTH"),
    ("pulse_width_neg", "NWIDTH"),
    ("mean", "MEAN"),
    ("rms", "RMS"),
    ("ac_rms", "ACRMS"),
]


def _mock_for_single_shot():
    # ACQ:NUMAC? must advance between the baseline read and the first poll
    # (ADR-007's fresh-acquisition check), so hand out an increasing counter.
    numac = iter([1, 2, 2, 2, 2, 2, 2, 2, 2])
    return MockVisaResource(
        responses={
            "MEASUrement:LIST?": "NONE",
            "MEASUrement:MEAS1:RESUlts:CURRentacq:POPUlation?": "1",
            "ACQ:NUMAC?": lambda _cmd: str(next(numac)),
            "MEASUrement:MEAS1:RESUlts:CURRentacq:MEAN?": "1.0",
        }
    )


@pytest.mark.parametrize("method_name,token", _VERT_TOKENS)
def test_meas_vert_wrapper_sends_correct_type_token(method_name, token):
    mock = _mock_for_single_shot()
    scope = _connect(mock)
    getattr(scope.meas.vert, method_name)("CH5")
    assert f"MEASUrement:MEAS1:TYPe {token}" in mock.writes


@pytest.mark.parametrize("method_name,token", _TIME_TOKENS)
def test_meas_time_wrapper_sends_correct_type_token(method_name, token):
    mock = _mock_for_single_shot()
    scope = _connect(mock)
    getattr(scope.meas.time, method_name)("CH5")
    assert f"MEASUrement:MEAS1:TYPe {token}" in mock.writes


@pytest.mark.parametrize("method_name,token", _DUAL_TOKENS)
def test_meas_dual_wrapper_sends_correct_type_token_and_two_sources(method_name, token):
    mock = _mock_for_single_shot()
    scope = _connect(mock)
    getattr(scope.meas.dual, method_name)("CH5", "CH6")
    assert f"MEASUrement:MEAS1:TYPe {token}" in mock.writes
    assert "MEASUrement:MEAS1:SOURCE1 CH5" in mock.writes
    assert "MEASUrement:MEAS1:SOURCE2 CH6" in mock.writes


@pytest.mark.parametrize("method_name,token", _LEGACY_TOKENS)
def test_legacy_meas_wrapper_sends_correct_type_token(method_name, token):
    mock = _mock_for_single_shot()
    scope = _connect(mock)
    getattr(scope.meas, method_name)("CH5")
    assert f"MEASUrement:MEAS1:TYPe {token}" in mock.writes


def test_peak_to_peak_uses_pk2pk_not_pk2p():
    """ADR-008: PK2PK is the manual-documented token, not the plan's PK2P."""
    mock = _mock_for_single_shot()
    scope = _connect(mock)
    scope.meas.peak_to_peak("CH5")
    assert "MEASUrement:MEAS1:TYPe PK2PK" in mock.writes
    assert not any("PK2P " in w and "PK2PK" not in w for w in mock.writes)


def test_enumeration_capture_keys_all_map_to_implemented_wrappers():
    """Every wrapper key exercised live in meas_wrapper_enumeration.json exists."""
    with open(CAPTURES / "meas_wrapper_enumeration.json") as fh:
        enum = json.load(fh)
    scope = _connect(MockVisaResource())
    for key in enum["results"]:
        subnode_name, _, method_name = key.partition(".")
        subnode = getattr(scope.meas, subnode_name)
        assert hasattr(subnode, method_name), f"missing wrapper for capture key {key!r}"


def test_stats_reads_all_accumulated_fields():
    mock = MockVisaResource(
        responses={
            "MEASUrement:MEAS1:RESUlts:CURRentacq:MEAN?": "1.0",
            "MEASUrement:MEAS1:RESUlts:ALLAcqs:MEAN?": "1.1",
            "MEASUrement:MEAS1:RESUlts:ALLAcqs:MINimum?": "0.9",
            "MEASUrement:MEAS1:RESUlts:ALLAcqs:MAXimum?": "1.3",
            "MEASUrement:MEAS1:RESUlts:ALLAcqs:STDDev?": "0.05",
            "MEASUrement:MEAS1:RESUlts:ALLAcqs:POPUlation?": "42",
        }
    )
    scope = _connect(mock)
    stats = scope.meas.get_statistics(1)
    assert stats.value == 1.0
    assert stats.mean == 1.1
    assert stats.minimum == 0.9
    assert stats.maximum == 1.3
    assert stats.stddev == 0.05
    assert stats.population == 42
