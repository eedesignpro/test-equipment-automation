"""Live-hardware tests for meas/file (R14, T13).

Skipped by default (marker ``hardware``). Run explicitly against the bench
scope with:

    .venv/Scripts/python.exe -m pytest -m hardware tests/hardware -v

Requires:
- A Tektronix MSO58 whose VISA address is in the ``MSO_RESOURCE`` environment
  variable, for example ``TCPIP::192.0.2.10::INSTR``. The tests skip when it
  is unset.
- The scope's internal AFG cabled into CH5

The AFG is configured via the AFG subsystem (R1) inside ``preserved_setup()``
as test setup only - no raw ``link.write`` AFG setup and no dependency on
ambient AFG state. At teardown the scope is left sane: the prior setup is
restored and the AFG output is left OFF. Only one VISA connection is held at a
time. Continuous acquisition is started so the measurement dispatcher's
fresh-acquisition wait (ADR-007) has an advancing ``ACQ:NUMAC?`` counter to
observe.
"""

import os
import socket

import pytest

from drivers.tektronix_mso import Tektronix_MSO

RESOURCE = os.environ.get("MSO_RESOURCE", "")
HOST = RESOURCE.split("::")[1] if RESOURCE.count("::") >= 2 else ""
PROBE_PORT = 111
PROBE_TIMEOUT_S = 2.0
CONNECT_TIMEOUT_MS = 5000
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
        # AFG setup via the AFG subsystem (R1).
        instr.afg.set_function("SINE")
        instr.afg.set_frequency(AFG_FREQUENCY_HZ)
        instr.afg.set_amplitude(AFG_AMPLITUDE_V)
        instr.afg.set_output_mode("CONTINUOUS")
        instr.afg.set_output_state(True)
        instr.acq.cmd_run()  # continuous acquisition so ACQ:NUMAC? advances
        instr.link.query("*OPC?")

        yield instr

        instr.acq.cmd_stop()
        instr.meas.cmd_delete("ALL")

    # Leave the scope sane: acquisition running, AFG output OFF.
    instr.afg.set_output_state(False)
    instr.acq.cmd_run()
    instr.disconnect()


def test_meas_frequency_on_ch5_matches_afg_within_tolerance(scope):
    freq = scope.meas.frequency("CH5", num=1, persist=False)
    assert freq == pytest.approx(AFG_FREQUENCY_HZ, rel=0.01)


def test_meas_amplitude_on_ch5_matches_afg_within_tolerance(scope):
    amplitude = scope.meas.amplitude("CH5", num=1, persist=False)
    # The internal AFG's amplitude setting is realized directly (not halved
    # by an external 1 MOhm probe termination) - bench-observed ~1.0 V for a
    # 1.0 V AFG amplitude setting.
    assert amplitude == pytest.approx(AFG_AMPLITUDE_V, rel=0.2)


def test_get_waveform_data_ch5_parses_to_expected_sine(scope):
    data = scope.file.get_waveform_data("CH5")
    assert len(data) > 0

    time_samples, waveforms = scope.file.parse_wfm_data_to_numpy(data)

    assert len(waveforms) == 1
    y = waveforms[0]
    assert len(time_samples) == len(y) > 0
    # A 1 MHz sine at ~1 V amplitude should swing well past +-0.3V.
    assert float(y.max()) > 0.3
    assert float(y.min()) < -0.3


def test_get_screenshot_retrieves_valid_png(scope, tmp_path):
    local_path = scope.file.get_screenshot("hw_test", folder_path=str(tmp_path))
    with open(local_path, "rb") as fh:
        data = fh.read()
    assert data[:8] == bytes.fromhex("89504e470d0a1a0a")
    assert len(data) > 1000
