"""Unit tests for Tektronix_MSO.preserved_setup() (R3/T1).

No raw op/cmd/resp capture exists for this context manager - the committed
``tests/captures/preserved_setup_trace.json`` is a step-summary (found_state/
restored booleans), not a replayable query trace - so these tests drive a
``MockVisaResource`` directly and assert on the exact command sequence and,
critically, the exception-path restore ordering the audit called out: save on
enter, then on exit ALWAYS recall (even under an exception) and ALWAYS
attempt to delete the scratch file afterward, with recall attempted before
delete even if recall itself raises.
"""

import pytest

from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource


def _connect(mock):
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    mock.responses.setdefault("*IDN?", "TEKTRONIX,MSO58,C0,CF:91")
    mock.responses.setdefault("FILESystem:DIR?", '"pyvisa_tmp;DIR;0;;"')
    scope.connect()
    mock.writes.clear()
    return scope


def test_preserved_setup_saves_then_recalls_on_normal_exit():
    mock = MockVisaResource()
    scope = _connect(mock)

    with scope.preserved_setup():
        pass

    saves = [w for w in mock.writes if w.startswith("SAVe:SETUp")]
    recalls = [w for w in mock.writes if w.startswith("RECAll:SETUp")]
    deletes = [w for w in mock.writes if w.startswith("FILESystem:DELEte")]
    assert len(saves) == 1
    assert len(recalls) == 1
    assert len(deletes) == 1
    # save must precede recall, recall must precede delete
    assert mock.writes.index(saves[0]) < mock.writes.index(recalls[0])
    assert mock.writes.index(recalls[0]) < mock.writes.index(deletes[0])


def test_preserved_setup_restores_on_exception():
    """The audit's core exception-path requirement: recall still runs."""
    mock = MockVisaResource()
    scope = _connect(mock)

    with pytest.raises(ValueError):
        with scope.preserved_setup():
            raise ValueError("boom inside body")

    recalls = [w for w in mock.writes if w.startswith("RECAll:SETUp")]
    deletes = [w for w in mock.writes if w.startswith("FILESystem:DELEte")]
    assert len(recalls) == 1
    assert len(deletes) == 1


def test_preserved_setup_reraises_the_original_exception_type():
    mock = MockVisaResource()
    scope = _connect(mock)

    class CustomError(Exception):
        pass

    with pytest.raises(CustomError):
        with scope.preserved_setup():
            raise CustomError("distinctive body failure")


def test_preserved_setup_deletes_scratch_even_if_recall_raises():
    """Delete must still be attempted (suppressed) even if recall itself errors."""

    class RecallFailsResource(MockVisaResource):
        def write(self, message, **kwargs):
            if message.startswith("RECAll:SETUp"):
                raise RuntimeError("simulated recall failure")
            return super().write(message, **kwargs)

    mock = RecallFailsResource()
    scope = _connect(mock)

    # RuntimeError from the failed recall propagates out of the context
    # manager (delete is best-effort/suppressed, but recall failure is not).
    with pytest.raises(RuntimeError):
        with scope.preserved_setup():
            pass

    deletes = [w for w in mock.writes if w.startswith("FILESystem:DELEte")]
    assert len(deletes) == 1


def test_preserved_setup_scratch_path_uses_dedicated_directory():
    mock = MockVisaResource()
    scope = _connect(mock)

    with scope.preserved_setup():
        pass

    saves = [w for w in mock.writes if w.startswith("SAVe:SETUp")]
    assert "C:/pyvisa_tmp/" in saves[0]
    assert saves[0].endswith('.set"')


def test_preserved_setup_applies_temporary_timeout_when_given(monkeypatch):
    mock = MockVisaResource()
    scope = _connect(mock)

    seen_timeouts = []
    original = scope.link.temporary_timeout

    def spy(timeout_ms):
        seen_timeouts.append(timeout_ms)
        return original(timeout_ms)

    monkeypatch.setattr(scope.link, "temporary_timeout", spy)

    with scope.preserved_setup(timeout_ms=60000):
        pass

    assert 60000 in seen_timeouts
