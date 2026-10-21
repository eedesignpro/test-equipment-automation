"""Unit tests for Tektronix_MSO.check_errors() 540-class warning filter (R5).

Event codes 540-549 are the "Measurement warning" class (e.g. signal
clipping) and must be surfaced via ``warnings.warn`` rather than raised, so a
benign clipping warning does not abort a script - while a real command/
execution error (CME/EXE, any other event code) must still raise
``InstrumentError``. Verified against a synthetic ALLEV? built from the real
549 "Clipping negative"/540 "Low amplitude" event text recorded in
meas_wrapper_enumeration.json / meas_config_trace.json.
"""

import warnings

import pytest

from common.errors import InstrumentError, InstrumentWarning
from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource


def _connect(responses=None):
    mock = MockVisaResource(responses=responses or {})
    mock.responses.setdefault("*IDN?", "TEKTRONIX,MSO58,C0,CF:91")
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    scope.connect()
    return scope, mock


def test_esr_clean_no_warning_no_raise():
    scope, _ = _connect({"*ESR?": "0"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scope.check_errors()  # must neither warn nor raise


def test_pure_540_event_warns_and_does_not_raise():
    scope, _ = _connect(
        {
            "*ESR?": "16",  # EXE bit set -> triggers the event-queue read
            "ALLEV?": '549,"Clipping negative"',
        }
    )
    with pytest.warns(InstrumentWarning, match="Clipping negative"):
        scope.check_errors()  # must not raise


def test_pure_command_error_raises_instrumenterror():
    scope, _ = _connect({"*ESR?": "32", "ALLEV?": '113,"Undefined header"'})
    with pytest.raises(InstrumentError, match="Undefined header"):
        scope.check_errors()


def test_mixed_540_warning_and_real_error_both_warns_and_raises():
    scope, _ = _connect(
        {
            "*ESR?": "48",  # CME+EXE
            "ALLEV?": '540,"Low amplitude";113,"Undefined header"',
        }
    )
    with pytest.warns(InstrumentWarning, match="Low amplitude"):
        with pytest.raises(InstrumentError, match="Undefined header"):
            scope.check_errors()


def test_event_code_zero_is_dropped():
    """Code 0 ('No events to report') must not be treated as a warning or error."""
    scope, _ = _connect({"*ESR?": "16", "ALLEV?": '0,"No events to report - queue empty"'})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scope.check_errors()  # no events after filtering -> no warn, no raise


@pytest.mark.parametrize("code", [540, 545, 549])
def test_full_measurement_warning_range_boundaries(code):
    scope, _ = _connect({"*ESR?": "16", "ALLEV?": f'{code},"Measurement warning"'})
    with pytest.warns(InstrumentWarning):
        scope.check_errors()


@pytest.mark.parametrize("code", [539, 550])
def test_codes_just_outside_measurement_warning_range_raise(code):
    scope, _ = _connect({"*ESR?": "16", "ALLEV?": f'{code},"Boundary event"'})
    with pytest.raises(InstrumentError):
        scope.check_errors()
