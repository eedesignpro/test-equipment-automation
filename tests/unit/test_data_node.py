"""Unit tests for the R4 `data` node (`_Data`, `scope.data`).

``capabilities_data_sv_trace.json`` (T3/T4) is a summary trace, not a raw
op/cmd/resp capture, so it cannot be replayed byte-for-byte. Its
``data_preamble`` block IS the real bench preamble (byt_nr/bn_fmt/byt_or/
ymult/yoff/yzero) though, so the scaling-math test below drives that exact
preamble through a small synthetic binary block and checks the
``y = YZEro + YMUlt * (raw - YOFf)`` formula the audit called out, while the
command-sequencing tests pin the exact write/query order ``get_waveform``
issues (source -> encoding -> width -> STARt/STOP -> preamble -> CURVe?).
"""

import numpy as np

from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource

# Real bench preamble captured on CH5 (capabilities_data_sv_trace.json
# "data_preamble"), narrowed to a 4-point record for a small test payload.
_BENCH_PREAMBLE_RESPONSES = {
    "WFMOutpre:BYT_Nr?": "1",
    "WFMOutpre:BIT_Nr?": "8",
    "WFMOutpre:BN_Fmt?": "RI",
    "WFMOutpre:BYT_Or?": "MSB",
    "WFMOutpre:ENCdg?": "BINARY",
    "WFMOutpre:NR_Pt?": "4",
    "WFMOutpre:PT_Off?": "0",
    "WFMOutpre:XINcr?": "1.6E-10",
    "WFMOutpre:XZEro?": "1.6E-10",
    "WFMOutpre:YMUlt?": "0.02",
    "WFMOutpre:YOFf?": "0.0",
    "WFMOutpre:YZEro?": "0.0",
}


def _connect(responses=None):
    mock = MockVisaResource(responses=responses or {})
    mock.responses.setdefault("*IDN?", "TEKTRONIX,MSO58,C0,CF:91")
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    scope.connect()
    mock.writes.clear()
    return scope, mock


def _binary_block(payload: bytes) -> bytes:
    header = f"#{len(str(len(payload)))}{len(payload)}".encode("ascii")
    return header + payload + b"\n"


def test_get_waveform_scales_raw_codes_with_bench_preamble():
    """y = YZEro + YMUlt * (raw - YOFf) against the real CH5 preamble."""
    responses = dict(_BENCH_PREAMBLE_RESPONSES)
    responses["DATa:SOUrce?"] = "CH5"
    responses["DATa:ENCdg?"] = "RIBinary"
    responses["DATa:WIDth?"] = "1"
    responses["HOR:RECO?"] = "4"
    scope, mock = _connect(responses)

    raw_codes = np.array([-27, 0, 27, 54], dtype=np.int8)
    mock.queue_read(_binary_block(raw_codes.tobytes()))

    waveform = scope.data.get_waveform(source="CH5")

    expected = 0.0 + 0.02 * (raw_codes.astype(np.float64) - 0.0)
    np.testing.assert_allclose(waveform, expected)
    assert waveform.dtype == np.float64


def test_get_waveform_command_sequence_matches_documented_order():
    responses = dict(_BENCH_PREAMBLE_RESPONSES)
    responses["DATa:ENCdg?"] = "RIBinary"
    responses["DATa:WIDth?"] = "1"
    responses["HOR:RECO?"] = "4"
    scope, mock = _connect(responses)
    mock.queue_read(_binary_block(bytes([0, 0, 0, 0])))

    scope.data.get_waveform(source="CH5")

    expected_order = [
        "DATa:SOUrce CH5",
        "DATa:ENCdg RIBinary",
        "DATa:WIDth 1",
        "DATa:STARt 1",
        "DATa:STOP 4",
        "CURVe?",
    ]
    indices = [mock.writes.index(cmd) for cmd in expected_order]
    assert indices == sorted(indices)


def test_get_waveform_keeps_current_source_when_none_given():
    responses = dict(_BENCH_PREAMBLE_RESPONSES)
    responses["DATa:ENCdg?"] = "RIBinary"
    responses["DATa:WIDth?"] = "1"
    responses["HOR:RECO?"] = "4"
    scope, mock = _connect(responses)
    mock.queue_read(_binary_block(bytes([0, 0, 0, 0])))

    scope.data.get_waveform()

    assert not any(w.startswith("DATa:SOUrce ") for w in mock.writes)


def test_set_frame_stop_max_echoes_literal_token():
    """ADR-020: DATa:FRAMESTOP MAX echoes the literal token MAX, not an int."""
    scope, mock = _connect({"DATa:FRAMESTOP?": "MAX"})
    result = scope.data.set_frame_stop("MAX")
    assert mock.writes[0] == "DATa:FRAMESTOP MAX"
    assert result == "MAX"


def test_set_frame_stop_rejects_non_max_string():
    scope, _ = _connect()
    try:
        scope.data.set_frame_stop("LAST")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-'MAX' string frame stop")


def test_set_frame_stop_numeric_returns_int():
    scope, mock = _connect({"DATa:FRAMESTOP?": "10"})
    result = scope.data.set_frame_stop(10)
    assert mock.writes[0] == "DATa:FRAMESTOP 10"
    assert result == 10


def test_get_time_axis_matches_bench_preamble_xincr_xzero():
    scope, _ = _connect(dict(_BENCH_PREAMBLE_RESPONSES))
    preamble = scope.data.get_preamble()
    time_axis = scope.data.get_time_axis(preamble)
    expected = 1.6e-10 + 1.6e-10 * (np.arange(4) - 0)
    np.testing.assert_allclose(time_axis, expected)


def test_get_waveform_codes_returns_16_bit_codes_unchanged_with_preamble():
    """width=2: big-endian int16 codes come back exactly, in native byte order."""
    responses = dict(_BENCH_PREAMBLE_RESPONSES)
    responses.update(
        {
            "WFMOutpre:BYT_Nr?": "2",
            "WFMOutpre:BIT_Nr?": "16",
            "DATa:SOUrce?": "CH5",
            "DATa:ENCdg?": "RIBinary",
            "DATa:WIDth?": "2",
            "HOR:RECO?": "4",
            "WFMOutpre:YMUlt?": "7.8125E-5",
        }
    )
    scope, mock = _connect(responses)
    raw = np.array([-32000, -1, 0, 31999], dtype=">i2")
    mock.queue_read(_binary_block(raw.tobytes()))

    codes, pre = scope.data.get_waveform_codes(source="CH5", width=2)

    assert codes.dtype == np.int16 and codes.dtype.byteorder in ("=", "|", "<")
    np.testing.assert_array_equal(codes, raw.astype(np.int16))
    assert pre["ymult"] == 7.8125e-5 and pre["byt_nr"] == 2
