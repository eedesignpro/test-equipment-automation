"""Unit tests for the ``file`` (_File) subsystem.

Response shapes below (LDIR size-listing format, the trailing terminator
after FILESystem:READFile, a real PNG magic header, and a real .wfm) come
from the live bench session documented in tests/captures/README.md and
context/decisions.md (ADR-005/ADR-006). Command sequencing/paths are
verified against the driver; large binary payloads are the actual captured
screenshot/.wfm bytes, not synthetic ones.
"""

from pathlib import Path

import pytest

from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource

CAPTURES = Path(__file__).parent.parent / "captures" / "tek_mso"
PNG_BYTES = (CAPTURES / "screenshot_mso58.png").read_bytes()
WFM_BYTES = (CAPTURES / "ch5_sine_1mhz.wfm").read_bytes()


def _connect(mock):
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    mock.responses.setdefault("*IDN?", lambda _c: "TEKTRONIX,MSO58,C0,CF:91")
    scope.connect()
    mock.writes.clear()
    return scope


def _ldir_response(filename: str, size: int) -> str:
    return f'"{filename};FILE;{size};07/22/26 17:13:00"'


def test_temp_remote_creates_scratch_dir_when_missing():
    mock = MockVisaResource(
        responses={
            "FILESystem:DIR?": '"foo","bar"',
        }
    )
    scope = _connect(mock)
    remote = scope.file._temp_remote("png")
    assert remote == "C:/pyvisa_tmp/Temp.png"
    assert 'FILESystem:CWD "C:/"' in mock.writes
    assert 'FILESystem:MKDir "C:/pyvisa_tmp"' in mock.writes


def test_temp_remote_skips_mkdir_when_scratch_dir_exists():
    mock = MockVisaResource(responses={"FILESystem:DIR?": '"pyvisa_tmp","other"'})
    scope = _connect(mock)
    scope.file._temp_remote("wfm")
    assert not any("MKDir" in w for w in mock.writes)


def test_drain_terminator_swallows_timeout_when_nothing_pending():
    mock = MockVisaResource()
    scope = _connect(mock)
    # read_buffer is empty, so mock.read_bytes(1) returns b"" - must not raise.
    scope.file._drain_terminator()


def test_get_screenshot_downloads_real_png_and_writes_local_file(tmp_path):
    remote = "C:/pyvisa_tmp/Temp.png"
    mock = MockVisaResource(
        responses={
            "FILESystem:DIR?": '"pyvisa_tmp"',
            "*OPC?": "1",
            "FILESystem:LDIR?": _ldir_response("Temp.png", len(PNG_BYTES)),
        }
    )
    mock.queue_read(PNG_BYTES)  # FILESystem:READFile payload
    mock.queue_read(b"")  # terminator drain read
    scope = _connect(mock)

    local_path = scope.file.get_screenshot("shot", folder_path=str(tmp_path))

    assert Path(local_path).read_bytes() == PNG_BYTES
    assert Path(local_path).read_bytes()[:8] == bytes.fromhex("89504e470d0a1a0a")
    assert 'SAVe:IMAGe "C:/pyvisa_tmp/Temp.png"' in mock.writes
    assert f'FILESystem:READFile "{remote}"' in mock.writes
    assert f'FILESystem:DELEte "{remote}"' in mock.writes


def test_get_waveform_data_returns_real_wfm_bytes_in_memory():
    mock = MockVisaResource(
        responses={
            "FILESystem:DIR?": '"pyvisa_tmp"',
            "*OPC?": "1",
            "FILESystem:LDIR?": _ldir_response("Temp.wfm", len(WFM_BYTES)),
        }
    )
    mock.queue_read(WFM_BYTES)
    mock.queue_read(b"")
    scope = _connect(mock)

    data = scope.file.get_waveform_data("CH5")

    assert data == WFM_BYTES
    assert 'SAVe:WAVEform CH5, "C:/pyvisa_tmp/Temp.wfm"' in mock.writes
    # get_waveform_data never touches local disk.
    assert not any("open(" in w for w in mock.writes)


def test_get_waveform_data_then_parse_to_numpy_roundtrip():
    """The R7 idiom: parse_wfm_data_to_numpy(file.get_waveform_data(...))."""
    mock = MockVisaResource(
        responses={
            "FILESystem:DIR?": '"pyvisa_tmp"',
            "*OPC?": "1",
            "FILESystem:LDIR?": _ldir_response("Temp.wfm", len(WFM_BYTES)),
        }
    )
    mock.queue_read(WFM_BYTES)
    mock.queue_read(b"")
    scope = _connect(mock)

    data = scope.file.get_waveform_data("CH5")
    time_samples, waveforms = scope.file.parse_wfm_data_to_numpy(data)

    assert len(time_samples) == len(waveforms[0]) == 10064


def test_get_file_size_returns_none_when_never_lands():
    mock = MockVisaResource(
        responses={"FILESystem:LDIR?": '"other_file;FILE;10;date"'},
    )
    scope = _connect(mock)
    size = scope.file._get_file_size("C:/pyvisa_tmp/", "Temp.wfm")
    assert size is None


def test_download_raises_query_error_when_size_unresolvable():
    from common.errors import InstrumentQueryError

    mock = MockVisaResource(responses={"FILESystem:LDIR?": '"other;FILE;1;date"'})
    scope = _connect(mock)
    with pytest.raises(InstrumentQueryError):
        scope.file._download("C:/pyvisa_tmp/Temp.wfm")


def test_local_path_rejects_nonexistent_folder():
    mock = MockVisaResource()
    scope = _connect(mock)
    with pytest.raises(ValueError):
        scope.file._local_path("out.png", "Z:/does/not/exist")
