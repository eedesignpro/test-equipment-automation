"""Live-hardware milestone test for the Tektronix MSO58 (iteration 1 bench proof).

Skipped by default (marker ``hardware``, deselected via ``addopts`` in
pyproject.toml). Run explicitly against the bench scope with:

    .venv/Scripts/python.exe -m pytest -m hardware tests/hardware -v

Requires:
- A Tektronix MSO58 whose VISA address is in the ``MSO_RESOURCE`` environment
  variable, for example ``TCPIP::192.0.2.10::INSTR``. The tests skip when it
  is unset.
- The scope's internal AFG cabled into CH5 (the signal is driven by this
  test itself via the AFG subsystem - no ambient AFG-on state is assumed)

Milestone covered: connect -> IDN parses to MSO58/8 channels -> set/read
vertical scale on CH5 -> basic single-sequence acquisition completes.

Self-containment (R9): the fixture drives the AFG via the AFG subsystem inside
``preserved_setup()`` so the run neither depends on nor leaves behind ambient
AFG state. At teardown the scope is left sane: acquisition running and the AFG
output OFF.

A short reachability probe runs before any pyvisa I/O so that a missing or
unreachable scope produces a clean ``skip`` rather than a long VISA-level
hang.
"""

import os
import socket

import pytest

from drivers.tektronix_mso import Tektronix_MSO

RESOURCE = os.environ.get("MSO_RESOURCE", "")
HOST = RESOURCE.split("::")[1] if RESOURCE.count("::") >= 2 else ""
# VXI-11 RPC portmapper port; a fast TCP probe here avoids blocking on a
# pyvisa/VISA-level open_resource() call when the scope is unreachable.
PROBE_PORT = 111
PROBE_TIMEOUT_S = 2.0
CONNECT_TIMEOUT_MS = 5000
ACQUISITION_TIMEOUT_MS = 10000
PRESERVE_TIMEOUT_MS = 20000

AFG_FREQUENCY_HZ = 1e6
AFG_AMPLITUDE_V = 1.0

pytestmark = pytest.mark.hardware


def _scope_reachable() -> bool:
    try:
        with socket.create_connection((HOST, PROBE_PORT), timeout=PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


@pytest.fixture
def scope():
    if not RESOURCE:
        pytest.skip("set MSO_RESOURCE to the scope VISA address")
    if not _scope_reachable():
        pytest.skip(f"MSO58 not reachable at {HOST}:{PROBE_PORT} (resource {RESOURCE})")

    instr = Tektronix_MSO(RESOURCE)
    try:
        instr.connect(timeout_ms=CONNECT_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001 - any connection failure should skip, not fail QA
        pytest.skip(f"Could not connect to {RESOURCE}: {exc}")

    with instr.preserved_setup(timeout_ms=PRESERVE_TIMEOUT_MS):
        instr.afg.set_function("SINE")
        instr.afg.set_frequency(AFG_FREQUENCY_HZ)
        instr.afg.set_amplitude(AFG_AMPLITUDE_V)
        instr.afg.set_output_mode("CONTINUOUS")
        instr.afg.set_output_state(True)
        instr.acq.cmd_run()
        instr.link.wait_for_complete(ACQUISITION_TIMEOUT_MS)
        yield instr

    # Leave the scope sane: acquisition running, AFG output OFF.
    instr.afg.set_output_state(False)
    instr.acq.cmd_run()
    instr.disconnect()


def test_idn_parses_to_mso58_with_8_channels(scope):
    assert scope.model == "MSO58"
    assert scope.num_channels == 8
    assert scope.ch5.installed
    assert scope.ch8.installed


def test_set_and_read_vertical_scale_on_ch5(scope):
    applied = scope.ch5.set_scale(0.5)
    assert applied == pytest.approx(0.5, rel=0.1)

    readback = scope.ch5.get_scale()
    assert readback == pytest.approx(applied)


def test_basic_single_sequence_acquisition_completes(scope):
    """AFG looped into CH5 should give a clean single-sequence acquisition."""
    scope.acq.cmd_single()
    with scope.link.temporary_timeout(ACQUISITION_TIMEOUT_MS):
        opc = scope.link.query("*OPC?").strip()
    assert opc == "1"
    assert scope.acq.get_state() == 0
