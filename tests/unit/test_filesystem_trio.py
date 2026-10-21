"""Unit tests for the R6 FILESystem COPy/RMDir/REName trio + shared quote helper.

``filesystem_usb_trace.json`` is a results-summary trace (directory listing
snapshots, no raw cmd/resp pairs) so it cannot be replayed via
``capture_replay``. These tests pin the exact quoted SCPI command strings
against a ``MockVisaResource``, using the same USB-drive paths the bench
session exercised (session B, E: drive).
"""

from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource


def _connect():
    mock = MockVisaResource()
    mock.responses.setdefault("*IDN?", "TEKTRONIX,MSO58,C0,CF:91")
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    scope.connect()
    mock.writes.clear()
    return scope, mock


def test_copy_quotes_both_paths():
    scope, mock = _connect()
    scope.file.cmd_copy("E:/orig.set", "E:/copy.set")
    assert mock.writes == ['FILESystem:COPy "E:/orig.set","E:/copy.set"']


def test_rename_quotes_both_paths():
    scope, mock = _connect()
    scope.file.cmd_rename("E:/orig.set", "E:/renamed.set")
    assert mock.writes == ['FILESystem:REName "E:/orig.set","E:/renamed.set"']


def test_rmdir_quotes_path():
    scope, mock = _connect()
    scope.file.cmd_rmdir("E:/scratch")
    assert mock.writes == ['FILESystem:RMDir "E:/scratch"']


def test_copy_rename_rmdir_are_set_only_no_query():
    """None of the trio reads back a value - they are fire-and-forget writes."""
    scope, mock = _connect()
    scope.file.cmd_copy("E:/a", "E:/b")
    scope.file.cmd_rename("E:/b", "E:/c")
    scope.file.cmd_rmdir("E:/emptydir")
    assert len(mock.writes) == 3
    assert all(not w.endswith("?") for w in mock.writes)


def test_copy_handles_paths_with_spaces_via_quote_helper():
    scope, mock = _connect()
    scope.file.cmd_copy("E:/my file.set", "E:/my other file.set")
    assert mock.writes == ['FILESystem:COPy "E:/my file.set","E:/my other file.set"']
