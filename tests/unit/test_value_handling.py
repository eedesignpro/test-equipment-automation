"""Unit tests for value-handling layer 1 (R8/R9) and the two security fixes
(R10/R11). Pure-Python logic per the testing policy - no instrument capture
needed.
"""

import warnings

import pytest

from common.errors import InstrumentQueryError, ValueSnappedWarning
from common.link import InstrumentLink
from common.node import Node
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
# R8: ValueSnappedWarning emission (in/out of tolerance, strict raise)
# ---------------------------------------------------------------------------


def test_set_and_verify_in_tolerance_is_silent(make_link):
    link, _ = make_link({"CH1:SCA?": "0.50001"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = link.set_and_verify("CH1:SCA", 0.5)
    assert result == pytest.approx(0.50001)


def test_set_and_verify_out_of_tolerance_warns_not_raises(make_link):
    link, _ = make_link({"CH1:SCA?": "1.0"})
    with pytest.warns(ValueSnappedWarning) as record:
        result = link.set_and_verify("CH1:SCA", 0.5)
    assert result == 1.0
    assert "requested 0.5" in str(record[0].message)
    assert "instrument applied 1" in str(record[0].message)


def test_set_and_verify_strict_raises_instead_of_warning(make_link):
    link, _ = make_link({"CH1:SCA?": "1.0"})
    with pytest.raises(ValueError) as exc:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            link.set_and_verify("CH1:SCA", 0.5, strict=True)
    assert "snapped" in str(exc.value)


def test_set_and_verify_respects_custom_tolerance(make_link):
    link, _ = make_link({"CH1:SCA?": "0.55"})
    # 10% relative tolerance swallows a 10% snap.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = link.set_and_verify("CH1:SCA", 0.5, rel_tol=0.2)
    assert result == 0.55


def test_node_set_and_verify_forwards_tolerance_kwargs(make_link):
    link, mock = make_link({"CH1:SCA?": "1.0"})
    node = Node(link, name="CH1")
    with pytest.warns(ValueSnappedWarning):
        node.set_and_verify("CH1:SCA", 0.5)
    assert mock.writes[0] == "CH1:SCA 0.5"


# ---------------------------------------------------------------------------
# R9: token pass-through (string values bypass numeric snap-check entirely)
# ---------------------------------------------------------------------------


def test_set_and_verify_token_passthrough_no_snap_warning(make_link):
    link, mock = make_link({"CH5:BAN?": "FUL"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = link.set_and_verify("CH5:BAN", "FUL")
    assert result == "FUL"
    assert mock.writes == ["CH5:BAN FUL", "CH5:BAN?"]


def test_bandwidth_full_token_bypasses_numeric_snap_check():
    mock = MockVisaResource(responses={"CH5:BAN?": "FUL"})
    scope = _connect(mock)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = scope.ch5.set_bandwidth("FULL")
    assert result == "FUL"
    assert "CH5:BAN FUL" in mock.writes


def test_bandwidth_numeric_value_still_goes_through_snap_check():
    mock = MockVisaResource(responses={"CH5:BAN?": "500000000.0"})
    scope = _connect(mock)
    with pytest.warns(ValueSnappedWarning):
        result = scope.ch5.set_bandwidth(2e8)
    assert result == 5e8


def test_bandwidth_invalid_token_raises_value_error():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.ch5.set_bandwidth("PROBE_MAX")


# ---------------------------------------------------------------------------
# R10: set_units quote guard (security Low)
# ---------------------------------------------------------------------------


def test_set_units_rejects_embedded_double_quote():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.ch5.set_units('mV"; :SYSTEM:ERROR "injected')


def test_set_units_accepts_clean_string_and_sends_quoted_command():
    mock = MockVisaResource(responses={"CH5:PROBEF:EXTU?": '"mV"'})
    scope = _connect(mock)
    result = scope.ch5.set_units("mV")
    assert result == '"mV"'
    assert 'CH5:PROBEF:EXTU "mV"' in mock.writes


def test_set_units_none_disables_external_units():
    mock = MockVisaResource()
    scope = _connect(mock)
    scope.ch5.set_units(None)
    assert "CH5:PROBEF:EXTU:STATE 0" in mock.writes


def test_set_label_also_rejects_embedded_double_quote():
    """Same quote-injection class fix applied to the label setter."""
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.ch5.set_label('CH5"; :SYSTEM:ERROR "injected')


# ---------------------------------------------------------------------------
# R11: _parse_binary_block truncation raise (security Low)
# ---------------------------------------------------------------------------


def test_parse_binary_block_raises_on_truncated_payload():
    raw = b"#210" + bytes(range(5))  # header declares 10 bytes, only 5 present
    with pytest.raises(InstrumentQueryError) as exc:
        InstrumentLink._parse_binary_block(raw)
    assert "Truncated" in str(exc.value)


def test_parse_binary_block_accepts_exact_and_over_length_payload():
    payload = bytes(range(10))
    raw = b"#210" + payload
    assert InstrumentLink._parse_binary_block(raw) == payload

    raw_with_trailer = b"#210" + payload + b"\n"
    assert InstrumentLink._parse_binary_block(raw_with_trailer) == payload


def test_parse_binary_block_raises_on_missing_header():
    with pytest.raises(InstrumentQueryError):
        InstrumentLink._parse_binary_block(b"no header here")


def test_read_binary_block_raises_on_truncated_live_read(make_link):
    link, mock = make_link()
    mock.queue_read(b"#3100" + bytes(range(5)))  # declares 100 bytes, sends 5
    with pytest.raises(InstrumentQueryError):
        link.read_binary_block()
