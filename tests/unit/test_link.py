"""Unit tests for InstrumentLink helpers, including the three bug fixes."""

import pytest

from common.errors import InstrumentQueryError, NotConnectedError
from common.link import InstrumentLink

from tests.mocks.mock_visa import MockVisaResource


def test_visa_property_raises_when_not_connected():
    link = InstrumentLink("MOCK::INSTR")
    assert link.is_connected is False
    with pytest.raises(NotConnectedError):
        _ = link.visa


def test_connect_disconnect_context_manager(monkeypatch):
    mock = MockVisaResource()
    link = InstrumentLink("MOCK::INSTR")
    monkeypatch.setattr(link, "_open_resource", lambda rm, name: mock)
    with link:
        assert link.is_connected
        assert mock.cleared
        assert mock.timeout == link.default_timeout_ms
    assert link.is_connected is False
    assert mock.closed


def test_query_float_direct_and_tolerant(make_link):
    link, _ = make_link({"VOLT?": "1.5", "SCALE?": '"2.5E-3 V"'})
    assert link.query_float("VOLT?") == 1.5
    assert link.query_float("SCALE?") == pytest.approx(2.5e-3)


def test_query_float_raises_on_garbage(make_link):
    link, _ = make_link({"BAD?": "no-number-here"})
    with pytest.raises(InstrumentQueryError):
        link.query_float("BAD?")


def test_query_int_direct_and_tolerant(make_link):
    link, _ = make_link({"N?": "42", "MSG?": "count=7 items"})
    assert link.query_int("N?") == 42
    assert link.query_int("MSG?") == 7


def test_query_int_raises_on_garbage(make_link):
    link, _ = make_link({"BAD?": "none"})
    with pytest.raises(InstrumentQueryError):
        link.query_int("BAD?")


def test_query_bool_variants(make_link):
    link, _ = make_link({"A?": "1", "B?": "OFF", "C?": "TRUE", "D?": "0"})
    assert link.query_bool("A?") is True
    assert link.query_bool("B?") is False
    assert link.query_bool("C?") is True
    assert link.query_bool("D?") is False


def test_write_choice_valid_and_invalid(make_link):
    link, mock = make_link()
    link.write_choice("CH1:COUP", "DC REJECT", {"AC": "AC", "DC": "DC", "DC REJECT": "DCR"})
    assert mock.writes[-1] == "CH1:COUP DCR"
    with pytest.raises(ValueError) as exc:
        link.write_choice("CH1:COUP", "BOGUS", {"AC": "AC"})
    assert "AC" in str(exc.value)


def test_set_and_verify_returns_readback(make_link):
    link, mock = make_link({"CH1:SCA?": "0.5"})
    result = link.set_and_verify("CH1:SCA", 0.5)
    assert result == 0.5
    assert mock.writes[0] == "CH1:SCA 0.5"


def test_temporary_timeout_restores(make_link):
    link, mock = make_link()
    original = mock.timeout
    with link.temporary_timeout(60000):
        assert mock.timeout == 60000
    assert mock.timeout == original


def test_temporary_timeout_restores_on_exception(make_link):
    link, mock = make_link()
    original = mock.timeout
    with pytest.raises(RuntimeError):
        with link.temporary_timeout(60000):
            raise RuntimeError("boom")
    assert mock.timeout == original


def test_check_errors_clean(make_link):
    link, _ = make_link({"SYST:ERR?": '0,"No error"'})
    link.check_errors()


def test_check_errors_collects_and_raises(make_link):
    responses = iter(['-222,"Data out of range"', '0,"No error"'])
    link, _ = make_link({"SYST:ERR?": lambda _cmd: next(responses)})
    with pytest.raises(InstrumentQueryError) as exc:
        link.check_errors()
    assert "Data out of range" in str(exc.value)


def test_binary_block_roundtrip(make_link):
    link, mock = make_link()
    payload = bytes(range(10))
    link.write_binary_block("CURV", payload)
    sent = mock.raw_writes[-1]
    assert sent.startswith(b"CURV #210")
    assert sent.endswith(payload)

    mock.queue_read(b"#210" + payload + b"\n")
    assert link.read_binary_block() == payload
