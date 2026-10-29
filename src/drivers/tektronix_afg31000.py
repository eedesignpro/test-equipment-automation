"""Driver for the Tektronix AFG31000 series arbitrary function generators.

Public import path::

    from drivers.tektronix_afg31000 import Tektronix_AFG31000

One model-gated class covers the series: one or two channels, 25 to 250 MHz.
The bench unit is an AFG31102 (two channels, 100 MHz) on LAN. Pin the raw
socket, ``TCPIP0::<address>::5025::SOCKET``. "Transport" below says why.

Common tasks:
  1. A 1 kHz, 2 Vpp sine on channel 1, clamped at +-1.5 V, then on:
         afg.ch1.set_waveform_sine(1e3, 2.0, v_limit=1.5)
         afg.ch1.set_output_state(True)
  2. Step the frequency in 1 uHz steps without touching anything else:
         for hz in (1000.000001, 1000.000002, 1000.000003):
             afg.ch1.set_frequency(hz)
  3. Ramp linearly from 1 kHz to 2 kHz over 1 s, over and over:
         afg.ch2.set_sweep(1e3, 2e3, 1.0)
  4. Step 100 Hz either side of the carrier ten times a second (FM, square):
         afg.ch2.set_fm(100.0, rate=10.0, shape=afg.ModulationShape.SQUARE)
  5. Play one period of a 60 Hz harmonic series from a numpy array:
         t = np.arange(4000) / 4000
         wave = np.sin(2 * np.pi * t) + 0.2 * np.sin(2 * np.pi * 3 * t)
         afg.ch1.set_waveform_arb(wave, 60.0, 2.0, v_limit=1.2)
  6. Lock to the lab 10 MHz and confirm the instrument is happy with it:
         state = afg.set_reference_clock_external()
         assert not state.frequency_questionable
  7. A 10 us pulse every 1 ms with 20 ns edges, then three-cycle bursts every 10 ms:
         afg.ch1.set_waveform_pulse(1e3, 2.0, v_limit=1.5, width=10e-6, edge_time=20e-9)
         afg.set_trigger_timer(10e-3)
         afg.ch1.set_burst(3)
  8. Put CH2 90 degrees behind CH1 at the same frequency, and line the phases up:
         afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0)
         afg.ch2.set_waveform_sine(1e3, 1.0, v_limit=1.0)
         afg.ch2.set_phase(-90.0)
         afg.interchannel.cmd_phase_sync()

Read these before you energize anything.

Add In can drive channel 1 past its own range, and nothing in the instrument
stops it. The rear Add In signal is summed in at unity gain into 50 ohm (2.03
V/V open circuit), after the amplitude setting and outside the voltage limit.
A full-scale output plus 0.4 V of Add In measured +5.62 V into 50 ohm, past the
5.5 Vrms rating of a 50 ohm scope input for a DC level. Keep the channel 1 peak
plus the Add In peak inside what the load can take. See ``set_external_add_state``.

The voltage limit bounds every setting, and every waveform helper requires one.
``v_limit=1.5`` means +-1.5 V; ``v_limit=(-0.2, 3.0)`` is asymmetric. The
instrument refuses an amplitude or an offset that would cross the limit, and a
limit written below the present swing pulls the amplitude down to fit
(bench-observed 2026-10-28). The limit is referenced to the load setting, like
the amplitude: the instrument stores it as a source voltage, so a 5 V limit at
50 ohm reads 10 V once the load is set to HIGHZ. Set the load first.

Every amplitude is in volts peak to peak, whatever unit the front panel shows.
The front panel unit is a live setting that changes how a bare number is read:
with the unit at Vrms, ``VOLT 0.8`` means 0.8 Vrms. This driver always sends the
unit, so it is immune to that. Keep it in mind with raw SCPI, and know that a
shape change with the panel at Vrms keeps the RMS value constant, so a square
to sine change raises the peak by 41 percent. The helpers write the limit
before the shape for that reason.

Transport. Over VXI-11 (``inst0``) a function change, or turning a sweep or FM
on or off, keeps the instrument busy for up to about 70 ms, and a read issued in
that window fails at once with ``VI_ERROR_INV_PROT`` instead of waiting. The raw
socket simply holds the reply until the instrument is ready. This driver waits
out the window either way, but the socket path is exact and faster. The socket
serves one client at a time: a second one connects but gets no answer until
the first closes (bench-observed 2026-10-28), while VXI-11 is served alongside.

Edit memory is per channel on this firmware. Channel 1 plays only EMEM1 and
channel 2 only EMEM2; the crossed choice is refused (-108). The arbitrary-
waveform helper therefore uploads into the channel's own memory.

Arbitrary data never travels over the raw socket. Its parser ends a message at
any 0x0A byte that lands at the end of a TCP segment, even inside a binary
block, then reads the rest of the data as commands; a stray ``#0`` or quote in
that data left the socket deaf until a restart (bench-observed 2026-10-28,
ADR-055). So an upload goes over a short VXI-11 session of its own, which ends
a message properly. Reading an edit memory back is no safer: over the socket
16000 points came back, but a 65536-point read never answered and also left
the socket deaf, and this firmware gives no way to learn the size first. So the
driver has no public read-back, and :meth:`snapshot` copies each edit memory to
a file on the instrument's own drive instead of reading it.

An amplitude step that changes the internal range mutes the output for 16
to 24 ms; small steps, offset and frequency changes are clean. See
``set_amplitude``. The load setting is best set once, at setup: just after a
change the instrument can briefly report the old setting again, so the
waveform helpers check it and raise rather than write a scaled value.

Only one run mode is active per channel: continuous, one modulation, sweep or
burst. Turning on AM, FM, PM, FSK, PWM, a sweep or a burst ends whichever of
the others was running. The instrument does this silently, with nothing in the
error queue (bench-observed 2026-10-28).

``disconnect()`` turns both outputs off (:meth:`safe_state`). A script that has
to leave an output running must say so and keep the link open, or reopen it.
"""

from __future__ import annotations

import contextlib
import math
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pyvisa as visa

from common.errors import InstrumentError, InstrumentUnresponsiveError
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import ScpiEnum, member_from_response
from common.state import (
    BurstState,
    ModulationState,
    SweepState,
    VoltageLimits,
    WaveformState,
)

_DEFAULT_TIMEOUT_MS = 5000

# model -> capabilities, from the Specification and Performance Verification
# Technical Reference (077-1489-01), tables 5, 9, 10 and 25. ``channels`` gates
# ch2. ``add_in`` gates the rear-panel Add In, which the 25 and 50 MHz models
# do not have. ``v_peak`` is the largest swing into 50 ohm and ``max_sine`` the
# top sine frequency; both are informational, the instrument enforces its own.
_MODEL_INFO: dict[str, dict] = {
    "AFG31021": {"channels": 1, "max_sine": 25e6, "v_peak": 5.0, "add_in": False},
    "AFG31022": {"channels": 2, "max_sine": 25e6, "v_peak": 5.0, "add_in": False},
    "AFG31051": {"channels": 1, "max_sine": 50e6, "v_peak": 5.0, "add_in": False},
    "AFG31052": {"channels": 2, "max_sine": 50e6, "v_peak": 5.0, "add_in": False},
    "AFG31101": {"channels": 1, "max_sine": 100e6, "v_peak": 5.0, "add_in": True},
    "AFG31102": {"channels": 2, "max_sine": 100e6, "v_peak": 5.0, "add_in": True},
    "AFG31151": {"channels": 1, "max_sine": 150e6, "v_peak": 2.5, "add_in": True},
    "AFG31152": {"channels": 2, "max_sine": 150e6, "v_peak": 2.5, "add_in": True},
    "AFG31251": {"channels": 1, "max_sine": 250e6, "v_peak": 2.5, "add_in": True},
    "AFG31252": {"channels": 2, "max_sine": 250e6, "v_peak": 2.5, "add_in": True},
}

_IDN_RE = re.compile(r"\b(AFG31\d{3})\b", re.IGNORECASE)

# A function change, a sweep going on or off, and FM going on or off each keep
# the instrument busy. Bench-measured 2026-10-28 on the bench AFG31102 (FV:1.6.1)
# with projects/afg31000_probe.py, stages "func" and "busy":
#   raw socket: the reply is held, so *OPC? returns late, 60-151 ms after a
#     function change and 70-93 ms after a run-mode change. No read failed.
#   VXI-11: a read inside the window fails at once with VI_ERROR_INV_PROT. 3 of
#     6 reads failed 60 ms after a function change, 1 of 6 at 70 ms, and none of
#     24 between 80 and 120 ms. Run-mode changes failed out to 47 ms. A device
#     clear recovers the link, and the write itself always landed.
# Off the socket the driver waits _BUSY_SETTLE_S, about twice the longest failure
# seen, and then asks *OPC?, polling while the read still fails. USB was not
# measured, so it takes the same safe path.
_BUSY_SETTLE_S = 0.15
# Bounds a hung instrument. It is not a settle time. The longest wait observed
# is *RST, which answered *OPC? after 1.2 s over the socket (harness section
# "system", 2026-10-28); *RST over VXI-11 was not measured.
_BUSY_DEADLINE_S = 3.0

# Instrument storage precision, as (significant digits, finest step). A helper
# compares the present value against the wanted one at this precision, so a
# repeat call writes nothing while any change the instrument can hold still
# writes (design standard 4.5). Bench-measured 2026-10-28, stage "ch2cfg":
# 1234.56789012 Hz read back 1234.567890 and 12345678.9012345 Hz read back
# 12345678.9012 (1 uHz or 12 digits); 1.23456 Vpp read back 1.235 and 0.0123456
# Vpp read back 0.0123 (4 digits or 0.1 mV); offset 0.123456 V read back 0.123
# and -0.0987654 V read back -0.099 (1 mV); a 1.23456 V limit read back 1.235;
# FM deviation 1234.5678 Hz read back 1234.57; FM rate 1234.5678 Hz read back
# 1235; sweep time 0.0123456789 s read back 0.012; sweep start 1000.123456 Hz
# read back 1000.12346.
_Q_FREQUENCY = (12, 1e-6)
_Q_AMPLITUDE = (4, 1e-4)
_Q_VOLTS = (4, 1e-3)
_Q_FM_DEVIATION = (6, 1e-6)
_Q_FM_RATE = (4, 1e-3)
_Q_SWEEP_TIME = (4, 1e-3)
_Q_SWEEP_FREQUENCY = (9, 1e-6)
# Second pass, bench-measured 2026-10-28 (stages shapes2, mod2, burst, chan2,
# noiseadd, followup): phase 12.3456789 deg read back 0.21555 rad, 12.35 deg
# (0.01 deg); PM deviation 12.3456 deg read back 12.3 deg (0.1 deg); AM depth
# 12.3456 read back 12.3, ramp symmetry 12.3456 read back 12.3, PWM deviation
# 12.3456 read back 12.3 (0.1 percent); pulse duty 12.34567 read back 12.346
# (0.001 percent); pulse width 123.456789 us read back 123.46 us (5 digits);
# lead delay 12.3456789 us read back 12.34568 us (7 digits); FSK hop 1234.567891234
# read back 1234.568 (7 digits); AM, PM, PWM and FSK rate 4 digits like FM; burst
# delay 12.3456789 us read back 12.346 us (5 digits); trigger timer 12.3456789 ms
# read back 12.35 ms (4 digits); noise level 12.3456 read back 12 (whole percent).
_Q_PHASE_DEG = (6, 0.01)
_Q_PM_DEG = (6, 0.1)
_Q_TENTH_PERCENT = (6, 0.1)
_Q_DUTY = (6, 1e-3)
_Q_WIDTH = (5, 1e-12)
_Q_DELAY = (7, 1e-12)
_Q_FSK_HOP = (7, 1e-6)
_Q_BURST_DELAY = (5, 1e-12)
_Q_TIMER = (4, 1e-9)
_Q_NOISE = (6, 1.0)
# Pulse edges are TRUNCATED to 4 digits, not rounded: 12.3456 ns read back
# 12.34 ns and 23.4567 ns read back 23.45 ns (stage followup). They are compared
# within one whole step, so a truncated read-back still counts as the same.
_Q_EDGE = (4, 1e-12)
# With the panel unit at Vrms or dBm the amplitude is read as the high level
# minus the low level. Each of those holds 1 mV, so the difference can sit 1 mV
# off: 1 Vpp read back as 0.501 - -0.500 after a load change (stage "ch2cfg").
_AMPLITUDE_SLACK_FROM_LEVELS = 1.5e-3

# OUTPut:IMPedance? answers 99.0e36 for INFinity on this firmware (the manual
# says 9.9E+37). Anything this large is high impedance.
_HIGHZ_THRESHOLD = 9e36

# BURSt:NCYCles? answers 9.9e37 for an infinite burst (stage burst), as the manual says.
_BURST_INFINITE = 9.9e37

# The instrument answers every angle in radians; the driver speaks degrees.
_DEG = 180.0 / math.pi

# Arbitrary waveform data: two bytes per point, big-endian, codes 0 to 16382.
# Bench-measured 2026-10-28, stage "ch2cfg": 0x3FFF read back as 0x3FFE, 0x4000
# as 0x0000 and 0xFFFF as 0x3FFE, with nothing in the error queue, so a code out
# of range is masked or clamped silently. The driver never sends one.
_ARB_CODE_MAX = 16382
_ARB_CODE_MID = _ARB_CODE_MAX // 2
_ARB_POINTS = (2, 131_072)
# Where the driver keeps its own files on the instrument's internal drive, the
# same name the scope driver uses for its scratch folder.
_SCRATCH_DIR = "M:/pyvisa_tmp"
# After a block arrives over VXI-11 the first *OPC? there fails with
# VI_ERROR_INV_PROT, and one more 0.25 s later answers: one failed read for
# 2000, 32000 and 131072 bytes, done in 0.28-0.53 s (2026-10-28). A device clear
# in that window can abort a large block and left the instrument deaf, so the
# upload waits without one.
_UPLOAD_RETRY_S = 0.25
_UPLOAD_DEADLINE_S = 10.0

# Bit 5 of the questionable status register: frequency accuracy questionable.
_QUES_FREQUENCY = 1 << 5

_ERROR_QUEUE_DRAIN = 64

# The CH1=CH2 couplings, by SCPI header: frequency, level, phase.
_COUPLINGS = ("FREQ", "VOLT", "PHAS")

# *TST? answered 0 after 136.4 s over the socket, with both outputs and the load
# settings as they were (harness section selftest, 2026-10-28). The bound is
# about four times that, for a hung instrument, not a settle time.
_SELF_TEST_TIMEOUT_MS = 600_000


def _num(value: float) -> str:
    """Format a number for the instrument at full precision."""
    return f"{value:.12g}"


def _quantum(value: float, digits: int, floor: float) -> float:
    """The step the instrument stores ``value`` at: ``digits`` figures, never finer than ``floor``."""
    if value == 0 or not math.isfinite(value):
        return floor
    return max(floor, 10.0 ** (math.floor(math.log10(abs(value))) - digits + 1))


def _same(present: float, wanted: float, precision: tuple[int, float], slack: float = 0.0) -> bool:
    """Whether the instrument already holds ``wanted``, at its own precision.

    ``slack`` widens the test for a reading coarser than the stored value.
    A value within half a step of ``wanted`` also counts, because the
    instrument rounds a half step up (a 0.5 percent noise level read back 1,
    stage noiseadd) where Python's round() goes to even, and a truncated edge
    sits a whole step from the rounded value.
    """
    step = _quantum(wanted, *precision)
    tolerance = max(step / 2, slack) * (1 + 1e-9)
    nearest = round(wanted / step) * step
    return min(abs(nearest - present), abs(wanted - present)) <= tolerance


def _limit_pair(v_limit: float | tuple[float, float]) -> tuple[float, float]:
    """Turn a scalar or a ``(low, high)`` limit into ``(low, high)`` volts."""
    if isinstance(v_limit, (int, float)) and not isinstance(v_limit, bool):
        if v_limit <= 0:
            raise ValueError(
                f"a scalar v_limit means +-v_limit and must be positive, got {v_limit!r}"
            )
        return -float(v_limit), float(v_limit)
    try:
        low, high = (float(v) for v in v_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"v_limit must be a number or a (low, high) pair, got {v_limit!r}"
        ) from exc
    if low >= high:
        raise ValueError(f"v_limit low ({low:g} V) must be below high ({high:g} V)")
    return low, high


def _check_swing(caller: str, ampl: float | None, offset: float, low: float, high: float) -> None:
    """Refuse, before any write, a swing that the limit would not allow.

    ``ampl`` None is a DC level: the instrument ignores the amplitude there and
    lets the offset reach the limit (bench-observed 2026-10-28, stage followup).
    """
    if ampl is None:
        if not low - 1e-12 <= offset <= high + 1e-12:
            raise ValueError(
                f"{caller}() refused: {offset:+g} V is outside the limit {low:+g} V to "
                f"{high:+g} V. Lower the level or widen v_limit."
            )
        return
    if ampl <= 0:
        raise ValueError(f"{caller}() refused: amplitude must be positive, got {ampl:g} Vpp")
    top = offset + ampl / 2
    bottom = offset - ampl / 2
    if top > high + 1e-12 or bottom < low - 1e-12:
        raise ValueError(
            f"{caller}() refused: {ampl:g} Vpp at {offset:+g} V offset swings "
            f"{bottom:+g} V to {top:+g} V, outside the limit {low:+g} V to {high:+g} V. "
            "Lower the amplitude or the offset, or widen v_limit."
        )


def _check_noise_add(caller: str, noise_add: float | bool | None) -> None:
    """Refuse a noise level the instrument would refuse: 0 to 50 percent (-222 at 51)."""
    if noise_add is None or isinstance(noise_add, bool):
        if noise_add is True:
            raise ValueError(f"{caller}() refused: noise_add=True has no level; pass a percent")
        return
    if not 0 <= noise_add <= 50:
        raise ValueError(f"{caller}() refused: noise_add is 0 to 50 percent, got {noise_add!r}")


def _arb_codes(samples: Sequence[float]) -> bytes:
    """Scale ``samples`` to the instrument's codes, largest magnitude to full scale."""
    data = np.asarray(samples, dtype=float).ravel()
    lo, hi = _ARB_POINTS
    if not lo <= data.size <= hi:
        raise ValueError(f"an arbitrary waveform takes {lo} to {hi} points, got {data.size}")
    if not np.all(np.isfinite(data)):
        raise ValueError("arbitrary waveform samples must all be finite numbers")
    peak = float(np.max(np.abs(data)))
    if peak == 0:
        raise ValueError(
            "all-zero samples have no shape; use set_waveform_sine at a tiny amplitude"
        )
    codes = np.rint(_ARB_CODE_MID + _ARB_CODE_MID * data / peak)
    return np.clip(codes, 0, _ARB_CODE_MAX).astype(">u2").tobytes()


@dataclass(frozen=True)
class FileEntry:
    """One entry of a folder on the instrument's drive.

    The names the instrument lists lag behind the drive: a file stored a second
    earlier was missing, and a deleted one stayed listed, while the free space
    it reports moved at once (bench-observed 2026-10-28). Treat a listing as a
    view of the recent past; a delete of a missing file answers -256.
    """

    name: str
    """The file or folder name, without the folder it is in."""

    is_directory: bool
    """True for a folder."""

    size: int
    """The size in bytes."""


def _catalog(link: InstrumentLink, directory: str) -> list[FileEntry]:
    """List ``directory`` on the instrument's drive, leaving the working folder as found.

    MMEMory:CATalog? lists the working folder only, so this moves there and back.
    """
    found = link.query("MMEM:CDIR?").strip()
    link.write(f'MMEM:CDIR "{directory}"')
    try:
        raw = link.query("MMEM:CAT?").strip()
    finally:
        link.write(f"MMEM:CDIR {found}")
    entries = []
    for item in re.findall(r'"([^"]*)"', raw):
        name, kind, size = (item.rsplit(",", 2) + ["", ""])[:3]
        if name in (".", ".."):
            continue
        entries.append(FileEntry(name, kind.upper() == "DIR", int(size or 0)))
    return entries


def _ensure_scratch_dir(link: InstrumentLink) -> None:
    """Create the driver's scratch folder on the internal drive; a folder already there is fine.

    Making a folder that exists queues no error (bench-observed 2026-10-28), so
    this does not consult the catalog, whose names lag (see FileEntry).
    """
    link.write(f'MMEM:MDIR "{_SCRATCH_DIR}"')
    link.query("*OPC?")


def _drain_errors(link: InstrumentLink) -> list[str]:
    """Read the error queue dry and return what it held."""
    errors: list[str] = []
    for _ in range(_ERROR_QUEUE_DRAIN):
        raw = link.query("SYST:ERR?").strip()
        if not raw or raw.split(",", 1)[0].strip().lstrip("+") in ("0", "-0"):
            break
        errors.append(raw)
    return errors


# ---------------------------------------------------------------------------
# Enums. Every value is the token the instrument ANSWERS with, which it also
# accepts, so a read-back compares straight against the member.
# ---------------------------------------------------------------------------


class Function(ScpiEnum):
    """The output shape. This is the waveform list on the front panel."""

    SINE = "SIN"
    SQUARE = "SQU"
    PULSE = "PULS"
    RAMP = "RAMP"
    NOISE = "PRN"
    DC = "DC"
    SINC = "SINC"
    GAUSSIAN = "GAUS"
    LORENTZ = "LOR"
    EXP_RISE = "ERIS"
    EXP_DECAY = "EDEC"
    HAVERSINE = "HAV"
    ARBITRARY = "EMEM"
    ARB_FILE = "EFIL"


class AmplitudeUnit(ScpiEnum):
    """The unit the front panel shows the amplitude in. The driver always speaks Vpp."""

    VPP = "VPP"
    VRMS = "VRMS"
    DBM = "DBM"


class Load(ScpiEnum):
    """What the far end of the cable terminates into, when it is not a resistance."""

    HIGHZ = "INF"


class ModulationShape(ScpiEnum):
    """The internal modulating waveform."""

    SINE = "SIN"
    SQUARE = "SQU"
    TRIANGLE = "TRI"
    RAMP = "RAMP"
    NEGATIVE_RAMP = "NRAM"
    NOISE = "PRN"
    ARBITRARY = "EMEM"
    ARB_FILE = "EFIL"


class ModulationSource(ScpiEnum):
    """Where the modulating signal comes from. EXTERNAL is the rear Ext Mod input."""

    INTERNAL = "INT"
    EXTERNAL = "EXT"


class Spacing(ScpiEnum):
    """How a sweep steps from its start frequency to its stop frequency."""

    LINEAR = "LIN"
    LOGARITHMIC = "LOG"


class SweepMode(ScpiEnum):
    """Whether a sweep repeats by itself or runs once per trigger."""

    REPEAT = "AUTO"
    TRIGGERED = "MAN"


class RunMode(ScpiEnum):
    """What the channel is doing. Only one run mode is active at a time."""

    CONTINUOUS = "CONT"
    MODULATION = "MOD"
    SWEEP = "SWE"
    BURST = "BURS"


class TriggerSource(ScpiEnum):
    """What triggers a TRIGGERED sweep, for the whole instrument."""

    TIMER = "TIM"
    EXTERNAL = "EXT"


class Reference(ScpiEnum):
    """Which 10 MHz timebase the instrument runs from."""

    INTERNAL = "INT"
    EXTERNAL = "EXT"


class Polarity(ScpiEnum):
    """Whether the output is inverted about its offset."""

    NORMAL = "NORM"
    INVERTED = "INV"


class BurstMode(ScpiEnum):
    """What starts and stops a burst.

    GATED outputs while the gate is true: the front Trigger In, or with the
    trigger source at TIMER the first half of each timer period (user manual).
    """

    TRIGGERED = "TRIG"
    GATED = "GAT"


class BurstIdle(ScpiEnum):
    """What the output holds between bursts: the first point, the offset, or the last point."""

    START = "START"
    DC = "DC"
    END = "END"


class Slope(ScpiEnum):
    """Which edge of the front Trigger In fires; for a gated burst, which level opens the gate."""

    POSITIVE = "POS"
    NEGATIVE = "NEG"


class TriggerOutMode(ScpiEnum):
    """What the front Trigger Out marks during an infinite burst.

    TRIGGER repeats with every cycle; SYNC gives one pulse as the burst starts.
    In every other run mode the two are the same (programmer manual). Both
    observed on the bench, 2026-10-28 (ADR-057).
    """

    TRIGGER = "TRIG"
    SYNC = "SYNC"


class PulseHold(ScpiEnum):
    """Which pulse parameter survives a change of period."""

    WIDTH = "WIDT"
    DUTY_CYCLE = "DUTY"


@dataclass(frozen=True)
class ReferenceClockState:
    """Which timebase is selected, and whether the instrument doubts its frequency."""

    selected: Reference
    """The source that has been chosen."""

    frequency_questionable: bool
    """Bit 5 of the questionable status register: frequency accuracy is in doubt.

    This is the only sign that an external reference has gone: ``selected``
    keeps reading EXTERNAL and nothing reaches the error queue. Bench-observed
    2026-10-28 (probe stage "refloss"): with the GNSS 10 MHz pulled from Ext Ref
    In, the bit rose within 0.2 s, stayed up while the cable was out, and
    cleared within 0.2 s of reconnecting. With the reference present it read
    False on both sources.
    """


_MODULATABLE = {
    Function.SINE,
    Function.SQUARE,
    Function.RAMP,
    Function.SINC,
    Function.GAUSSIAN,
    Function.LORENTZ,
    Function.EXP_RISE,
    Function.EXP_DECAY,
    Function.HAVERSINE,
    Function.ARBITRARY,
    Function.ARB_FILE,
}
_SWEEPABLE = {Function.SINE, Function.SQUARE, Function.RAMP, Function.ARBITRARY}
_PHASELESS = {Function.DC, Function.NOISE, Function.PULSE}
_MODULATION_KINDS = ("AM", "FM", "PM", "FSK", "PWM")
# A burst survives a pulse carrier and ends on noise or DC (stage burst).
_BURSTABLE = _MODULATABLE | {Function.PULSE}
# The instrument refuses AM and FM on a pulse and PWM on anything else (-221,
# stage mod2), so each helper checks the carrier before it writes.
_CARRIER_NEEDED = {
    "AM": (_MODULATABLE, "a sine, square, ramp, built-in or arbitrary"),
    "FM": (_MODULATABLE, "a sine, square, ramp, built-in or arbitrary"),
    "PM": (_MODULATABLE, "a sine, square, ramp, built-in or arbitrary"),
    "FSK": (_MODULATABLE, "a sine, square, ramp, built-in or arbitrary"),
    "PWM": ({Function.PULSE}, "a pulse"),
}


def _token(value, enum_cls: type[ScpiEnum], label: str) -> str:
    """The SCPI token for an enum member, a member name, or a token, any case."""
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    for member in enum_cls:
        if text in (member.name, str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")


class _AfgLink(InstrumentLink):
    """The link, with this instrument's transport behaviour.

    A raw socket gets newline terminators and no device clear, which a socket
    cannot carry. A binary block never goes over the socket: it goes over a
    short VXI-11 session to the same instrument (see the module docstring).
    """

    @property
    def is_socket(self) -> bool:
        return self.resource_name.upper().endswith("::SOCKET")

    def connect(self, timeout_ms: int | None = None) -> None:
        rm = visa.ResourceManager()
        self._instr = self._open_resource(rm, self.resource_name)
        if self.is_socket:
            self._instr.write_termination = "\n"
            self._instr.read_termination = "\n"
        else:
            self._instr.clear()
        self._instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms

    def _block_resource(self) -> str:
        """The VXI-11 resource of the instrument this socket resource reaches."""
        board, host = self.resource_name.split("::")[:2]
        return f"{board}::{host}::inst0::INSTR"

    def write_binary_block(self, cmd: str, data: bytes) -> int:
        """Send ``cmd`` with ``data`` as a binary block, and wait until the instrument has it.

        Raises if the instrument queued an error for it. Over a socket the
        block goes over a VXI-11 session of its own, after everything already
        sent on the socket has run.
        """
        message = cmd.encode("ascii") + self._format_binary_header(len(data)) + bytes(data)
        # The trailing newline ends the message on a link without an END flag.
        message += b"\n"
        if not self.is_socket:
            sent = self.visa.write_raw(message)
            self._wait_for_block(self.visa)
            return sent
        self.query("*OPC?")
        side = self._open_resource(visa.ResourceManager(), self._block_resource())
        try:
            side.timeout = self.visa.timeout
            sent = side.write_raw(message)
            self._wait_for_block(side)
            errors = []
            for _ in range(_ERROR_QUEUE_DRAIN):
                raw = side.query("SYST:ERR?").strip()
                if not raw or raw.split(",", 1)[0].strip().lstrip("+") in ("0", "-0"):
                    break
                # -500 "Power on" is the first read of a fresh VXI-11 queue.
                if not raw.startswith("-500"):
                    errors.append(raw)
        finally:
            side.close()
        if errors:
            raise InstrumentError(f"{cmd.strip()} <{len(data)} bytes> was refused: {errors}")
        return sent

    def _wait_for_block(self, session) -> None:
        """Ask *OPC? until it answers, without the device clear that can abort a block."""
        deadline = time.monotonic() + _UPLOAD_DEADLINE_S
        while True:
            try:
                session.query("*OPC?")
                return
            except visa.VisaIOError as exc:
                if exc.error_code != visa.constants.StatusCode.error_invalid_protocol:
                    raise
                if time.monotonic() > deadline:
                    raise InstrumentUnresponsiveError(
                        f"{self.resource_name} did not finish a binary block in "
                        f"{_UPLOAD_DEADLINE_S:g} s. Stop and check the instrument."
                    ) from exc
                time.sleep(_UPLOAD_RETRY_S)

    def query_binary_block(self, cmd: str) -> bytes:
        """Query a definite-length block, reading exactly the declared byte count."""
        return bytes(self.visa.query_binary_values(cmd, datatype="B", container=bytes))

    def wait_settled(self) -> None:
        """Wait until a function or run-mode change has finished."""
        if self.is_socket:
            self.query("*OPC?")
            return
        deadline = time.monotonic() + _BUSY_DEADLINE_S
        while True:
            time.sleep(_BUSY_SETTLE_S)
            try:
                self.query("*OPC?")
                return
            except visa.VisaIOError as exc:
                if exc.error_code != visa.constants.StatusCode.error_invalid_protocol:
                    raise
                if time.monotonic() > deadline:
                    raise InstrumentUnresponsiveError(
                        f"{self.resource_name} stayed busy for over {_BUSY_DEADLINE_S:g} s "
                        "after a function or run-mode change. Stop and check the instrument."
                    ) from exc
                self.visa.clear()


class _Channel(Node):
    """One output channel.

    A channel node exists only for a channel the instrument has, so nothing
    here can be called and then refuse for lack of hardware.
    """

    def __init__(self, link: _AfgLink, n: int):
        super().__init__(link, name=f"CH{n}")
        self.n = n
        self._src = f"SOUR{n}:"

    # ---------- the output connector ----------

    def set_output_state(self, state: bool) -> bool:
        """Turn this channel's output on or off; returns the read-back.

        The connector follows the call late. Bench-measured 2026-10-28 at
        10 kHz into a scope: the call took 50 to 280 ms to answer, and a 50 ms
        off produced 158 ms without signal, a 200 ms off 309 ms. Allow a few
        hundred milliseconds before a measurement that depends on the change.
        """
        self.write(f"OUTP{self.n} {'ON' if state else 'OFF'}")
        return self.get_output_state()

    def get_output_state(self) -> bool:
        """Whether this channel's output is on."""
        return self.query_bool(f"OUTP{self.n}?")

    def set_load(self, load: float | Load | str) -> float:
        """Tell the generator what the far end terminates into, in ohms; 1 to 10000 or HIGHZ.

        Enum arg: <afg>.Load.<TAB>

        This changes no hardware. The source impedance stays 50 ohm. It sets
        what the generator assumes, so the amplitude, offset and voltage limit
        it reports match what the load sees. Changing it rescales all three,
        because the instrument holds them as source voltages. Returns the
        read-back, where ``float('inf')`` means high impedance.

        Set it once, at setup. Shortly after a change the instrument has been
        seen to report the old setting again for a moment, and a voltage
        written then is scaled for the wrong load (bench-observed 2026-10-28,
        ADR-054). The waveform helpers detect that and raise.
        """
        numeric = isinstance(load, (int, float)) and not isinstance(load, bool)
        wanted = float(load) if numeric else float("inf")
        if not numeric:
            _token(load, Load, "load")  # refuse a bad token before reading anything
        present = self.get_load()
        # A write only when it differs, so a repeat cannot open the stale window.
        if present == wanted or (numeric and abs(present - wanted) <= 1e-3 * wanted):
            return present
        if numeric:
            self.write(f"OUTP{self.n}:IMP {_num(load)}")
        else:
            self.write(f"OUTP{self.n}:IMP {_token(load, Load, 'load')}")
        return self.get_load()

    def get_load(self) -> float:
        """What the generator assumes the far end terminates into, in ohms; inf is high-Z."""
        ohms = self.query_float(f"OUTP{self.n}:IMP?")
        return float("inf") if ohms >= _HIGHZ_THRESHOLD else ohms

    # ---------- the voltage limit ----------

    def set_voltage_limits(self, low: float, high: float) -> VoltageLimits:
        """Clamp the output between ``low`` and ``high`` volts at the load.

        The instrument then refuses any amplitude or offset that would cross
        either limit. A limit written inside the present swing pulls the
        amplitude down to fit. The clamp has no off switch; the widest setting
        is the model's full range. Referenced to the load setting, so set the
        load first.
        """
        if low >= high:
            raise ValueError(f"low ({low:g} V) must be below high ({high:g} V)")
        self._ensure_limits(low, high)
        return self.get_voltage_limits()

    def get_voltage_limits(self) -> VoltageLimits:
        """The output voltage clamp, in volts at the load."""
        return VoltageLimits(
            low=self.query_float(f"{self._src}VOLT:LIM:LOW?"),
            high=self.query_float(f"{self._src}VOLT:LIM:HIGH?"),
            enabled=True,
        )

    def _ensure_limits(self, low: float, high: float) -> None:
        """Write the limits that differ, in an order that keeps low below high."""
        present = self.get_voltage_limits()
        writes = []
        if not _same(present.low, low, _Q_VOLTS):
            writes.append(("LOW", low))
        if not _same(present.high, high, _Q_VOLTS):
            writes.append(("HIGH", high))
        if low >= present.high:
            writes.reverse()  # moving the window up past the old top: raise the top first
        for side, value in writes:
            self._write_checked(
                f"{self._src}VOLT:LIM:{side}", value, _Q_VOLTS, f"{side.lower()} limit"
            )

    # ---------- frequency, amplitude, offset ----------
    #
    # set_frequency and set_amplitude are justified public atomics (design
    # standard 5.3): a loop that steps one must not rewrite the others. Each
    # stays inside the voltage limit already in the instrument, so a level-only
    # write needs no limit argument of its own.

    def set_frequency(self, hz: float) -> float:
        """Set the output frequency in Hz, to 1 uHz or 12 digits. Returns the read-back."""
        return self._write_checked(f"{self._src}FREQ", hz, _Q_FREQUENCY, "frequency")

    def get_frequency(self) -> float:
        """The output frequency in Hz."""
        return self.query_float(f"{self._src}FREQ?")

    def set_amplitude(self, vpp: float) -> float:
        """Set the amplitude in volts peak to peak, inside the present limit. Returns the read-back.

        A step that changes the instrument's internal range mutes the output
        for 16 to 24 ms, and no command holds the range. Bench-measured
        2026-10-28 at 10 kHz into HIGHZ (probe stage "ampstep", ADR-054):
        2 -> 1.9 and 1 -> 0.99 Vpp were clean; 2 -> 1.5, 2 -> 1, 0.2 -> 0.1 and
        0.1 -> 0.2 Vpp went quiet for 24 ms, 0.5 -> 0.45 Vpp for 16 ms. A muting
        step took 27 to 38 ms to answer against 4 to 7 ms for a clean one. Where
        the range edges lie was not mapped.  # UNVERIFIED beyond those steps
        """
        self.write(f"{self._src}VOLT {_num(vpp)}VPP")
        present, slack = self._get_amplitude_and_slack()
        if not _same(present, vpp, _Q_AMPLITUDE, slack):
            self._refused("amplitude", vpp, present)
        return present

    def get_amplitude(self) -> float:
        """The amplitude in volts peak to peak, whatever unit the front panel shows."""
        return self._get_amplitude_and_slack()[0]

    def _get_amplitude_and_slack(self) -> tuple[float, float]:
        """The amplitude in Vpp, and how far that reading can sit from the stored value."""
        unit = self.query(f"{self._src}VOLT:UNIT?").strip().upper()
        if unit == AmplitudeUnit.VPP.value:
            return self.query_float(f"{self._src}VOLT?"), 0.0
        high = self.query_float(f"{self._src}VOLT:HIGH?")
        low = self.query_float(f"{self._src}VOLT:LOW?")
        return high - low, _AMPLITUDE_SLACK_FROM_LEVELS

    def set_offset(self, volts: float) -> float:
        """Set the DC offset in volts, inside the present limit. Returns the read-back."""
        return self._write_checked(f"{self._src}VOLT:OFFS", volts, _Q_VOLTS, "offset")

    def get_offset(self) -> float:
        """The DC offset in volts."""
        return self.query_float(f"{self._src}VOLT:OFFS?")

    def set_amplitude_unit(self, unit: AmplitudeUnit | str) -> AmplitudeUnit:
        """Set the unit the front panel shows the amplitude in; the driver still speaks Vpp.

        Enum arg: <afg>.AmplitudeUnit.<TAB>

        A display preference with one trap: a bare number in raw SCPI is read
        in this unit. Every driver call sends its unit, so none is affected.
        """
        token = _token(unit, AmplitudeUnit, "unit")
        if self.get_amplitude_unit().value != token:
            self.write(f"{self._src}VOLT:UNIT {token}")
        return self.get_amplitude_unit()

    def get_amplitude_unit(self) -> AmplitudeUnit:
        """The unit the front panel shows the amplitude in."""
        raw = self.query(f"{self._src}VOLT:UNIT?")
        return member_from_response(raw, AmplitudeUnit, "amplitude unit")

    def get_run_mode(self) -> RunMode:
        """Whether the channel is continuous, modulated, sweeping or bursting."""
        return member_from_response(self.query(f"{self._src}RMOD?"), RunMode, "run mode")

    # ---------- the waveform, one call per shape ----------
    #
    # Each helper reads the present state and writes only what differs, in an
    # order that is valid at every moment (design standard 4.5 and 6.2):
    #   1. the limit, before any level, so nothing ever runs under an old limit;
    #   2. the shape, then the busy window waited out;
    #   3. the levels: a falling amplitude before the offset, a rising one after,
    #      so the swing never leaves both the old and the new envelope;
    #   4. the frequency.
    # No helper turns the output on (6.2 rule 4).
    #
    # Each helper also reads the load setting before and after its writes. Every
    # voltage is written in units of that setting, so a write that lands while
    # the setting reads differently is scaled by the wrong factor. Bench-observed
    # 2026-10-28 (projects/afg31000_probe.py stage "loadflip", ADR-054): within a
    # quarter second of a 50 ohm -> HIGHZ change, the instrument reported 50 ohm
    # again for 20 to 66 ms, once for several hundred ms, and a 0.5 V limit and a
    # 0.2 Vpp amplitude written then landed at 1.0 V and 0.4 Vpp. What triggers
    # it is not established.  # UNVERIFIED - so the helper detects it, and raises.

    def _set_waveform(
        self,
        function: Function,
        caller: str,
        *,
        freq: float | None,
        ampl: float | None,
        offset: float,
        v_limit: float | tuple[float, float],
        upload: bytes | None = None,
        efile: str | None = None,
        symmetry: float | None = None,
        pulse: dict | None = None,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        low, high = _limit_pair(v_limit)
        _check_swing(caller, ampl, offset, low, high)
        _check_noise_add(caller, noise_add)
        plan = self._plan_pulse(caller, freq, **pulse) if pulse is not None else None
        load = self.get_load()
        self._ensure_limits(low, high)
        if upload is not None:
            self.link.write_binary_block(f"DATA:DATA EMEM{self.n},", upload)
        new_file = efile is not None and self._get_arb_file() != efile
        if new_file:
            self.write(f'{self._src}FUNC:EFIL "{efile}"')
        if upload is not None or new_file or self._get_function() is not function:
            self._set_function(function)
        if plan is not None:
            self._apply_pulse(plan)
        if symmetry is not None:
            self._write_checked(
                f"{self._src}FUNC:RAMP:SYMM", symmetry, _Q_TENTH_PERCENT, "ramp symmetry"
            )
        if ampl is None:
            self._write_checked(f"{self._src}VOLT:OFFS", offset, _Q_VOLTS, "offset")
        else:
            self._ensure_levels(ampl, offset)
        if freq is not None and plan is None:
            self._write_checked(f"{self._src}FREQ", freq, _Q_FREQUENCY, "frequency")
        if noise_add is not None:
            self._ensure_noise_add(noise_add)
        state = self.get_waveform()
        if self.get_load() != load:
            raise InstrumentError(
                f"{caller}() on {self.name}: the load setting read {load:g} ohm before the "
                f"writes and {self.get_load():g} ohm after, so a value written in between may "
                "be scaled for the wrong load. Leave the load alone during a run (set it once "
                "at setup), check get_voltage_limits() and get_waveform(), and call again."
            )
        return state

    def _ensure_levels(self, ampl: float, offset: float) -> None:
        present_ampl, slack = self._get_amplitude_and_slack()
        write_ampl = not _same(present_ampl, ampl, _Q_AMPLITUDE, slack)
        if write_ampl and ampl < present_ampl:
            self.set_amplitude(ampl)
            write_ampl = False
        self._write_checked(f"{self._src}VOLT:OFFS", offset, _Q_VOLTS, "offset")
        if write_ampl:
            self.set_amplitude(ampl)

    def set_waveform_sine(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a sine at ``freq`` Hz and ``ampl`` Vpp, clamped at ``v_limit`` volts.

        ``v_limit`` is +-volts, or a ``(low, high)`` pair. A swing outside it is
        refused before anything is written. Calling again with one changed
        argument moves only that parameter. ``noise_add`` is internal noise in
        percent of the amplitude, 0 to 50; False turns it off and None leaves
        it as it is. The instrument halves the signal to make room for the
        noise while it is on, and still reports the full amplitude: a 1 Vpp
        sine measured 0.50 Vpp under 10 and 50 percent noise (bench-measured
        2026-10-28). AM, FM, PM, FSK, PWM or the sweep turning on or off drops
        noise add silently; a burst keeps it (stages noiseadd, followup2).
        """
        return self._set_waveform(
            Function.SINE,
            "set_waveform_sine",
            freq=freq,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
            noise_add=noise_add,
        )

    def set_waveform_square(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a 50 percent square at ``freq`` Hz and ``ampl`` Vpp, clamped at ``v_limit`` V.

        The square has no duty setting on this series. Use the pulse shape for
        another duty cycle.
        """
        return self._set_waveform(
            Function.SQUARE,
            "set_waveform_square",
            freq=freq,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
            noise_add=noise_add,
        )

    def set_waveform_ramp(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        symmetry: float | None = None,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a ramp; ``symmetry`` is the percent of the cycle spent rising, 0 to 100.

        100 is a rising sawtooth, 0 a falling one and 50 a triangle. None keeps
        the present symmetry. Up to 1 MHz on the AFG31102.
        """
        return self._set_waveform(
            Function.RAMP,
            "set_waveform_ramp",
            freq=freq,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
            symmetry=symmetry,
            noise_add=noise_add,
        )

    def set_waveform_triangle(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a triangle at ``freq`` Hz and ``ampl`` Vpp: the ramp at 50 percent symmetry."""
        return self._set_waveform(
            Function.RAMP,
            "set_waveform_triangle",
            freq=freq,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
            symmetry=50.0,
            noise_add=noise_add,
        )

    def set_waveform_pulse(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        width: float | None = None,
        duty: float | None = None,
        edge_time: float | None = None,
        leading_edge_time: float | None = None,
        trailing_edge_time: float | None = None,
        delay: float | None = None,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a pulse train; ``width`` in seconds or ``duty`` in percent.

        Width and duty are one parameter expressed two ways, so pass one or
        neither. The one passed is the one a later frequency change keeps: the
        helper sets the instrument's hold to match. Neither keeps whichever the
        instrument holds now. ``edge_time`` sets both edges, or set each with
        ``leading_edge_time`` and ``trailing_edge_time``, in seconds; the
        shortest is 4 ns on the AFG31102. ``delay`` is the lead delay in seconds.

        The instrument refuses an edge longer than the width allows, and the
        helper checks the manual's rule first: width at least 0.625 times the
        two edges, and at most the period less 0.8 times them.

        Selecting the pulse shape can carry the previous shape's frequency in
        and clamp a stored width that no longer fits, and the clamp stays
        (bench-observed 2026-10-28, stages shapes2 and followup). So the helper
        reads the pulse settings before the change and writes every one after
        it, in an order where each step is valid.
        """
        return self._set_waveform(
            Function.PULSE,
            "set_waveform_pulse",
            freq=freq,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
            pulse={
                "width": width,
                "duty": duty,
                "edge_time": edge_time,
                "leading": leading_edge_time,
                "trailing": trailing_edge_time,
                "delay": delay,
            },
            noise_add=noise_add,
        )

    def set_waveform_noise(
        self, ampl: float, *, v_limit: float | tuple[float, float], offset: float = 0.0
    ) -> WaveformState:
        """Output noise at ``ampl`` Vpp, clamped at ``v_limit`` volts.

        Noise has no frequency, and this series has no noise bandwidth setting.
        """
        return self._set_waveform(
            Function.NOISE,
            "set_waveform_noise",
            freq=None,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
        )

    def set_waveform_dc(
        self, offset: float, *, v_limit: float | tuple[float, float]
    ) -> WaveformState:
        """Output a steady level of ``offset`` volts, clamped at ``v_limit`` volts.

        The amplitude setting plays no part: a DC level may reach the limit
        itself (bench-observed 2026-10-28).
        """
        return self._set_waveform(
            Function.DC,
            "set_waveform_dc",
            freq=None,
            ampl=None,
            offset=offset,
            v_limit=v_limit,
        )

    def _set_builtin(self, function: Function, caller: str, freq, ampl, v_limit, offset, noise_add):
        return self._set_waveform(
            function,
            caller,
            freq=freq,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
            noise_add=noise_add,
        )

    def set_waveform_sinc(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output sin(x)/x at ``freq`` Hz; ``ampl`` is twice the main lobe's peak, in volts.

        The built-in shapes reach 1 MHz on the AFG31102 (bench-measured). The
        limit check treats the shape as swinging ``ampl``/2 either side of
        ``offset``, which covers the side lobes.
        """
        return self._set_builtin(
            Function.SINC, "set_waveform_sinc", freq, ampl, v_limit, offset, noise_add
        )

    def set_waveform_gaussian(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a gaussian pulse at ``freq`` Hz; ``ampl`` is twice its peak, in volts."""
        return self._set_builtin(
            Function.GAUSSIAN, "set_waveform_gaussian", freq, ampl, v_limit, offset, noise_add
        )

    def set_waveform_lorentz(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a Lorentz pulse at ``freq`` Hz; ``ampl`` is twice its peak, in volts."""
        return self._set_builtin(
            Function.LORENTZ, "set_waveform_lorentz", freq, ampl, v_limit, offset, noise_add
        )

    def set_waveform_exp_rise(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output an exponential rise at ``freq`` Hz; ``ampl`` is twice its peak, in volts."""
        return self._set_builtin(
            Function.EXP_RISE, "set_waveform_exp_rise", freq, ampl, v_limit, offset, noise_add
        )

    def set_waveform_exp_decay(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output an exponential decay at ``freq`` Hz; ``ampl`` is twice its peak, in volts."""
        return self._set_builtin(
            Function.EXP_DECAY, "set_waveform_exp_decay", freq, ampl, v_limit, offset, noise_add
        )

    def set_waveform_haversine(
        self,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a haversine at ``freq`` Hz; ``ampl`` is twice its peak, in volts."""
        return self._set_builtin(
            Function.HAVERSINE, "set_waveform_haversine", freq, ampl, v_limit, offset, noise_add
        )

    def set_waveform_arb(
        self,
        samples: Sequence[float] | None,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Play one period of ``samples``, repeating ``freq`` times a second, at ``ampl`` Vpp.

        ``samples`` holds 2 to 131072 real numbers. They are scaled so the
        largest magnitude reaches ``ampl``/2 either side of ``offset``. The
        upload goes into this channel's own edit memory, which is volatile. Pass
        None to replay what that memory already holds.

        The sample rate follows from the point count and ``freq``: up to 16000
        points play at up to 1 GS/s on this model, more at 250 MS/s (spec table
        4). The frequency reaches 50 MHz at any length on the AFG31102. A new
        upload always restarts the waveform. Uploads of 131072 points took
        0.53 s (bench-measured 2026-10-28).
        """
        upload = None if samples is None else _arb_codes(samples)
        return self._set_waveform(
            Function.ARBITRARY,
            "set_waveform_arb",
            freq=freq,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
            upload=upload,
            noise_add=noise_add,
        )

    def set_waveform_arb_file(
        self,
        path: str,
        freq: float,
        ampl: float,
        *,
        v_limit: float | tuple[float, float],
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Play the .tfwx file at ``path`` on the instrument's drive at ``freq`` Hz and ``ampl`` Vpp.

        ``path`` is a .tfwx file on the internal drive (``M:/...``) or a USB
        stick (``U:/...``), forward slashes. Edit memory is left alone. Save a
        file from edit memory with :meth:`cmd_save_arb_file`.
        """
        return self._set_waveform(
            Function.ARB_FILE,
            "set_waveform_arb_file",
            freq=freq,
            ampl=ampl,
            offset=offset,
            v_limit=v_limit,
            efile=path,
            noise_add=noise_add,
        )

    def get_waveform(self) -> WaveformState:
        """Read the shape now being generated and every parameter that applies.

        ``function`` is the instrument's own token: EMEM is this channel's edit
        memory, EFIL a waveform file. ``noise_add`` is the internal noise level
        in percent, or None while noise add is off.
        """
        function = self._get_function()
        offset = self.get_offset()
        output = self.get_output_state()
        noise = (
            self.query_float(f"SOUR{self.n + 2}:POW?")
            if self._get_combine_feed() in ("NOIS", "BOTH")
            else None
        )
        if function is Function.DC:
            return WaveformState(
                function=function.value,
                frequency=0.0,
                amplitude=0.0,
                offset=offset,
                output_state=output,
                noise_add=noise,
            )
        pulse = function is Function.PULSE
        return WaveformState(
            function=function.value,
            frequency=0.0 if function is Function.NOISE else self.get_frequency(),
            amplitude=self.get_amplitude(),
            offset=offset,
            output_state=output,
            duty=self.query_float(f"{self._src}PULS:DCYC?") if pulse else None,
            symmetry=(
                self.query_float(f"{self._src}FUNC:RAMP:SYMM?")
                if function is Function.RAMP
                else None
            ),
            width=self.query_float(f"{self._src}PULS:WIDT?") if pulse else None,
            edge_time=self.query_float(f"{self._src}PULS:TRAN?") if pulse else None,
            trailing_edge_time=(self.query_float(f"{self._src}PULS:TRAN:TRA?") if pulse else None),
            delay=self.query_float(f"{self._src}PULS:DEL?") if pulse else None,
            noise_add=noise,
        )

    # ---------- pulse internals ----------

    def _plan_pulse(
        self,
        caller: str,
        freq: float,
        *,
        width: float | None,
        duty: float | None,
        edge_time: float | None,
        leading: float | None,
        trailing: float | None,
        delay: float | None,
    ) -> dict:
        """Work out every pulse setting the helper will write, from the present ones.

        Reads before the shape changes, because the change can clamp them.
        Raises ValueError, before any write, for a pulse the instrument would refuse.
        """
        if width is not None and duty is not None:
            raise ValueError(
                f"{caller}() refused: pass width (seconds) or duty (percent), not both; "
                "they are one pulse parameter expressed two ways."
            )
        if edge_time is not None and (leading is not None or trailing is not None):
            raise ValueError(
                f"{caller}() refused: edge_time sets both edges; pass it, or "
                "leading_edge_time and trailing_edge_time, not both."
            )
        if freq <= 0:
            raise ValueError(f"{caller}() refused: frequency must be positive, got {freq:g} Hz")
        src = self._src
        hold = member_from_response(self.query(f"{src}PULS:HOLD?"), PulseHold, "pulse hold")
        if width is not None:
            hold = PulseHold.WIDTH
        elif duty is not None:
            hold = PulseHold.DUTY_CYCLE
        period = 1.0 / freq
        if width is None and duty is None:
            if hold is PulseHold.WIDTH:
                width = self.query_float(f"{src}PULS:WIDT?")
            else:
                duty = self.query_float(f"{src}PULS:DCYC?")
        target_width = width if width is not None else duty / 100.0 * period
        lead = edge_time if edge_time is not None else leading
        trail = edge_time if edge_time is not None else trailing
        lead = self.query_float(f"{src}PULS:TRAN?") if lead is None else lead
        trail = self.query_float(f"{src}PULS:TRAN:TRA?") if trail is None else trail
        delay = self.query_float(f"{src}PULS:DEL?") if delay is None else delay
        edges = lead + trail
        if not 0 < target_width < period:
            raise ValueError(
                f"{caller}() refused: a {target_width:g} s pulse does not fit a {period:g} s period."
            )
        if target_width < 0.625 * edges or target_width > period - 0.8 * edges:
            raise ValueError(
                f"{caller}() refused: a {target_width:g} s width with {lead:g} s and {trail:g} s "
                "edges breaks the instrument's rule: the width must be at least 0.625 times "
                "the two edges, and at most the period less 0.8 times them. Shorten the "
                "edges or change the width."
            )
        if not 0 <= delay <= period:
            raise ValueError(f"{caller}() refused: delay {delay:g} s is outside 0 to {period:g} s")
        return {
            "freq": freq,
            "hold": hold,
            "width": width,
            "duty": duty,
            "lead": lead,
            "trail": trail,
            "delay": delay,
        }

    def _apply_pulse(self, plan: dict) -> None:
        """Write a planned pulse with the pulse shape selected, keeping every step valid.

        Shorter edges go first and longer ones last, so the width always has
        room for them. A shorter period takes the new width or duty first and a
        longer one takes it after, so the width fits the period at each step
        whichever is held: with the duty held a period change scales the width,
        and 99.99 percent of a shorter period can break the edge rule. A period
        change that leaves no room is clamped silently: 400 us held at 10 kHz
        read 99.995 us, and 99.99 percent at 10 MHz read 95 percent, with no
        error (bench-observed 2026-10-28, ADR-056). No step here reaches that.
        """
        src = self._src
        if self._get_pulse_hold() is not plan["hold"]:
            self.write(f"{src}PULS:HOLD {plan['hold'].value}")
        edges = (
            ("PULS:TRAN", plan["lead"], "leading edge"),
            ("PULS:TRAN:TRA", plan["trail"], "trailing edge"),
        )
        present = {h: self.query_float(f"{src}{h}?") for h, _, _ in edges}
        for header, wanted, label in edges:
            if wanted < present[header]:
                self._write_checked(f"{src}{header}", wanted, _Q_EDGE, label, truncates=True)

        def write_width() -> None:
            if plan["width"] is not None:
                self._write_checked(f"{src}PULS:WIDT", plan["width"], _Q_WIDTH, "pulse width")
            else:
                self._write_checked(f"{src}PULS:DCYC", plan["duty"], _Q_DUTY, "pulse duty")

        shorter = plan["freq"] > self.get_frequency()
        if shorter:
            write_width()
            self._write_checked(f"{src}FREQ", plan["freq"], _Q_FREQUENCY, "frequency")
        else:
            self._write_checked(f"{src}FREQ", plan["freq"], _Q_FREQUENCY, "frequency")
            write_width()
        for header, wanted, label in edges:
            if wanted >= present[header]:
                self._write_checked(f"{src}{header}", wanted, _Q_EDGE, label, truncates=True)
        self._write_checked(f"{src}PULS:DEL", plan["delay"], _Q_DELAY, "pulse delay")

    def _get_pulse_hold(self) -> PulseHold:
        """Which pulse parameter survives a change of period (SOUR<n>:PULS:HOLD?)."""
        return member_from_response(self.query(f"{self._src}PULS:HOLD?"), PulseHold, "pulse hold")

    # ---------- noise add and arbitrary files ----------

    def _get_combine_feed(self) -> str:
        """What is summed into the output: "", NOIS, EXT or BOTH (SOUR<n>:COMB:FEED?)."""
        return self.query(f"{self._src}COMB:FEED?").strip().strip('"').upper()

    def _ensure_noise_add(self, noise_add: float | bool) -> None:
        """Turn internal noise add on at ``noise_add`` percent, or off for False.

        Keeps Add In as it is on channel 1. The noise level is SOURce3 for
        channel 1 and SOURce4 for channel 2.
        """
        feed = self._get_combine_feed()
        external = feed in ("EXT", "BOTH")
        noise = noise_add is not False
        if noise:
            self._write_checked(f"SOUR{self.n + 2}:POW", noise_add, _Q_NOISE, "noise level")
        token = {
            (False, False): "",
            (False, True): "NOIS",
            (True, False): "EXT",
            (True, True): "BOTH",
        }[(external, noise)]
        if token != feed:
            self.write(f'{self._src}COMB:FEED "{token}"')

    def _get_arb_file(self) -> str:
        """The waveform file the EFIL shape plays (SOUR<n>:FUNC:EFIL?); "" when none."""
        return self.query(f"{self._src}FUNC:EFIL?").strip().strip('"')

    def cmd_save_arb_file(self, path: str) -> str:
        """Copy this channel's edit memory to a .tfwx file at ``path`` on the instrument's drive.

        ``path`` is ``M:/...`` (internal) or ``U:/...`` (USB), forward slashes.
        An existing file is overwritten. Took 0.32-0.40 s for 1000 to 65536
        points (bench-measured 2026-10-28). Returns ``path``.
        """
        self.write(f'MMEM:STOR:TRAC EMEM{self.n},"{path}"')
        self.query("*OPC?")
        errors = _drain_errors(self.link)
        if errors:
            raise InstrumentError(f"cmd_save_arb_file({path!r}) was refused: {errors}")
        return path

    def _set_function(self, function: Function | str) -> Function:
        """Write the output shape (SOUR<n>:FUNC) and wait out the busy window.

        Private because a bare shape change is not a complete operating point.
        """
        self.write(f"{self._src}FUNC {_token(function, Function, 'function')}")
        self.link.wait_settled()
        return self._get_function()

    def _get_function(self) -> Function:
        """The shape now being generated (SOUR<n>:FUNC?)."""
        return member_from_response(self.query(f"{self._src}FUNC?"), Function, "function")

    def _get_phase(self) -> float:
        """The waveform phase in radians (SOUR<n>:PHAS?)."""
        return self.query_float(f"{self._src}PHAS?")

    def _set_phase(self, radians: float) -> float:
        """Write the waveform phase in radians (SOUR<n>:PHAS <x>RAD)."""
        self.write(f"{self._src}PHAS {_num(radians)}RAD")
        return self._get_phase()

    def _get_edit_memory(self) -> bytes:
        """This channel's edit memory as raw big-endian codes (DATA? EMEM<n>).

        Private, and with no public counterpart, because it can hang the
        instrument. Over the raw socket a read returns at most 16384 points
        (32768 bytes), in about 22 ms. One point more, and the instrument sends
        the block header and nothing after it, and the socket stays deaf until
        a restart: neither a new connection nor a VXI-11 device clear brings it
        back (stage ememread, ADR-057). VXI-11 fails any read of data with a
        0x0A byte (ADR-055). This firmware offers no way to learn the size
        first: DATA:POINts? is undefined (-113), the catalog's names lag, and
        the waveform-list length query knows only imported waveforms. Use it
        only on a memory you know holds 16384 points or fewer.
        """
        return self.link.query_binary_block(f"DATA? EMEM{self.n}")

    def _set_edit_memory(self, data: bytes) -> None:
        """Load raw big-endian codes into this channel's edit memory (DATA:DATA EMEM<n>)."""
        self.link.write_binary_block(f"DATA:DATA EMEM{self.n},", data)

    # ---------- phase and polarity ----------

    def set_phase(self, degrees: float) -> float:
        """Set the waveform phase in degrees, -180 to +180. Returns the read-back in degrees.

        Held to 0.01 degree (bench-measured 2026-10-28). Pulse, noise and DC have
        no phase, so this refuses on them. Two channels at the same frequency
        keep this offset between them once lined up by
        ``afg.interchannel.cmd_phase_sync()``.
        """
        function = self._get_function()
        if function in _PHASELESS:
            raise ValueError(
                f"set_phase() refused: {self.name} is generating {function.name}, which has no "
                "phase. Phase applies to sine, square, ramp, built-in and arbitrary shapes."
            )
        if not -180.0 <= degrees <= 180.0:
            raise ValueError(f"set_phase() refused: {degrees:g} deg is outside -180 to +180")
        return self._write_checked(
            f"{self._src}PHAS", degrees, _Q_PHASE_DEG, "phase", suffix="DEG", read_scale=_DEG
        )

    def get_phase(self) -> float:
        """The waveform phase in degrees."""
        return self._get_phase() * _DEG

    def set_polarity(self, polarity: Polarity | str) -> Polarity:
        """Set whether the output is inverted about its offset; a :class:`Polarity`.

        Enum arg: <afg>.Polarity.<TAB>

        A polarity write keeps the instrument busy for about 70 ms (bench-measured
        2026-10-28), which the call waits out.
        """
        token = _token(polarity, Polarity, "polarity")
        if self.get_polarity().value != token:
            self.write(f"OUTP{self.n}:POL {token}")
            self.link.wait_settled()
        return self.get_polarity()

    def get_polarity(self) -> Polarity:
        """Whether the output is inverted about its offset."""
        return member_from_response(self.query(f"OUTP{self.n}:POL?"), Polarity, "polarity")

    # ---------- modulation ----------
    #
    # One call per mode, each the 33522A's signature. Each writes only what
    # differs and turns its mode on last. Turning one on ends any other mode,
    # the sweep and the burst, silently (bench-observed 2026-10-28, stage mod2).
    #
    # Changing the level of a modulation that is running took 58-70 ms over the
    # socket against 1.5 ms with it off (stage mod2), and over VXI-11 the next read
    # failed at once (stage busy4), so that write waits out a busy window before its
    # read-back. Amplitude and offset writes opened no window over VXI-11.

    def _modulate(
        self,
        kind: str,
        *,
        level: tuple[str, float, tuple[int, float], str, str, float],
        rate: float | None,
        rate_header: str,
        shape: ModulationShape | str | None,
        source: ModulationSource | str | None,
    ) -> ModulationState:
        """Write one modulation's settings that differ, then turn it on.

        ``level`` is (header, value, precision, label, unit suffix, read scale).
        """
        allowed, words = _CARRIER_NEEDED[kind]
        function = self._get_function()
        if function not in allowed:
            raise ValueError(
                f"set_{kind.lower()}() needs {words} carrier; {self.name} is generating "
                f"{function.name}. Set the carrier shape first."
            )
        src = self._src
        if shape is not None:
            token = _token(shape, ModulationShape, "shape")
            if token == ModulationShape.ARBITRARY.value:
                token = f"EMEM{self.n}"  # this channel's own memory, as for the carrier
            present = self.query(f"{src}{kind}:INT:FUNC?").strip().upper()
            # The query answers EMEM for either memory; the write names this one.
            if present != token and not (present == "EMEM" and token.startswith("EMEM")):
                self.write(f"{src}{kind}:INT:FUNC {token}")
        if rate is not None:
            self._write_checked(f"{src}{rate_header}", rate, _Q_FM_RATE, f"{kind} rate")
        if source is not None:
            token = _token(source, ModulationSource, "source")
            if self.query(f"{src}{kind}:SOUR?").strip().upper() != token:
                self.write(f"{src}{kind}:SOUR {token}")
        running = self.query_bool(f"{src}{kind}:STAT?")
        header, value, precision, label, suffix, scale = level
        self._write_checked(
            f"{src}{header}",
            value,
            precision,
            label,
            suffix=suffix,
            read_scale=scale,
            settle=running,
        )
        if not running:
            self.write(f"{src}{kind}:STAT ON")
            self.link.wait_settled()
        return self.get_modulation()

    def set_am(
        self,
        depth: float,
        *,
        rate: float | None = None,
        shape: ModulationShape | str | None = None,
        source: ModulationSource | str | None = None,
    ) -> ModulationState:
        """Amplitude-modulate the carrier to ``depth`` percent, 0 to 120, and turn AM on.

        Enum arg: shape  = <afg>.ModulationShape.<TAB>
                  source = <afg>.ModulationSource.<TAB>

        ``rate`` is the internal modulating frequency in Hz, 1 mHz to 1 MHz.
        EXTERNAL takes this channel's rear Ext Mod input: at 100 % depth, 0 V
        gave half the amplitude, and the swing grew 0.45 Vpp per volt on a
        1 Vpp carrier, so it reaches none and full at about -1.1 and +1.1 V
        (bench, both channels alike, -0.8 to +0.8 V, ADR-057). Settings left as
        None keep their present values. Turning AM on clears noise add.
        """
        return self._modulate(
            "AM",
            level=("AM", depth, _Q_TENTH_PERCENT, "AM depth", "", 1.0),
            rate=rate,
            rate_header="AM:INT:FREQ",
            shape=shape,
            source=source,
        )

    def set_fm(
        self,
        deviation: float,
        *,
        rate: float | None = None,
        shape: ModulationShape | str | None = None,
        source: ModulationSource | str | None = None,
    ) -> ModulationState:
        """Frequency-modulate the carrier by ``deviation`` Hz peak, and turn FM on.

        Enum arg: shape  = <afg>.ModulationShape.<TAB>
                  source = <afg>.ModulationSource.<TAB>

        ``rate`` is the internal modulating frequency in Hz. A SQUARE shape
        steps between the carrier plus and minus ``deviation``. EXTERNAL takes
        this channel's rear Ext Mod input, which is linear but reaches the full
        ``deviation`` at about 1.41 V, not the 1 V of spec table 25:
        bench-measured 2026-10-28 on channel 2, 141.9 Hz/V at a 200 Hz
        setting, 0.15 Hz worst residual over -0.8 V to +0.8 V; channel 1
        measured the same, 141.3-141.6 Hz/V (ADR-057). The deviation
        cannot exceed the carrier frequency (-222). Settings left as None keep
        their present values, and only what differs is written.
        """
        return self._modulate(
            "FM",
            level=("FM", deviation, _Q_FM_DEVIATION, "FM deviation", "", 1.0),
            rate=rate,
            rate_header="FM:INT:FREQ",
            shape=shape,
            source=source,
        )

    def set_pm(
        self,
        deviation: float,
        *,
        rate: float | None = None,
        shape: ModulationShape | str | None = None,
        source: ModulationSource | str | None = None,
    ) -> ModulationState:
        """Phase-modulate the carrier by ``deviation`` degrees peak, 0 to 180, and turn PM on.

        Enum arg: shape  = <afg>.ModulationShape.<TAB>
                  source = <afg>.ModulationSource.<TAB>

        Degrees, as on the front panel and the 33522A. The instrument holds
        0.1 degree and answers in radians; the driver converts both ways
        (bench-measured 2026-10-28: 90 deg read back 1.57080 rad, 181 deg -222).
        EXTERNAL takes this channel's rear Ext Mod input: 52.5 degrees per volt
        at a 90 degree setting, so the full deviation at about 1.7 V (bench,
        both channels alike, ADR-057). With an external source, turning PM
        on moved the output by 180 degrees against the other channel; see
        ``cmd_phase_sync``.
        """
        return self._modulate(
            "PM",
            level=("PM", deviation, _Q_PM_DEG, "PM deviation", "DEG", _DEG),
            rate=rate,
            rate_header="PM:INT:FREQ",
            shape=shape,
            source=source,
        )

    def set_fsk(
        self,
        hop_frequency: float,
        *,
        rate: float | None = None,
        source: ModulationSource | str | None = None,
    ) -> ModulationState:
        """Shift between the carrier and ``hop_frequency`` Hz, and turn FSK on.

        Enum arg: <afg>.ModulationSource.<TAB>

        ``rate`` is how many times a second the internal source shifts, in Hz.
        EXTERNAL keys on the rear Ext Mod input's 3.3 V logic level (spec table
        25). On the bench that input was driven only between -0.8 and +0.8 V,
        and both channels sat at ``hop_frequency`` throughout (ADR-057).
        # UNVERIFIED: which level selects the carrier
        """
        return self._modulate(
            "FSK",
            level=("FSK", hop_frequency, _Q_FSK_HOP, "FSK hop frequency", "", 1.0),
            rate=rate,
            rate_header="FSK:INT:RATE",
            shape=None,
            source=source,
        )

    def set_pwm(
        self,
        deviation: float,
        *,
        rate: float | None = None,
        shape: ModulationShape | str | None = None,
        source: ModulationSource | str | None = None,
    ) -> ModulationState:
        """Pulse-width-modulate by ``deviation`` seconds peak, and turn PWM on.

        Enum arg: shape  = <afg>.ModulationShape.<TAB>
                  source = <afg>.ModulationSource.<TAB>

        Seconds, as on the 33522A. The instrument holds the deviation as a
        percent of the period, so the driver converts at the present frequency;
        a later frequency change keeps the percent, not the seconds. The pulse
        must have room for it: the instrument refuses a deviation larger than
        the width or the time left in the period (-222), and a later duty change
        that leaves it no room shrinks it to fit, silently: 40 % read 10 % after
        the duty went from 50 to 10 % (stage followup2). Only a pulse carrier
        can be width-modulated (-221 otherwise). EXTERNAL takes this channel's
        rear Ext Mod input, which reaches the full deviation at about 1.41 V:
        70.7 us per volt at a 100 us setting (bench, both channels, ADR-057).
        """
        period = 1.0 / self.get_frequency()
        return self._modulate(
            "PWM",
            level=(
                "PWM:DCYC",
                deviation / period * 100.0,
                _Q_TENTH_PERCENT,
                "PWM deviation",
                "",
                1.0,
            ),
            rate=rate,
            rate_header="PWM:INT:FREQ",
            shape=shape,
            source=source,
        )

    def set_modulation_off(self) -> ModulationState:
        """Turn every modulation, the sweep and the burst off, leaving a plain carrier.

        Turning a modulation or the sweep off also clears noise add, silently
        (stages noiseadd, followup2).
        """
        for kind in (*_MODULATION_KINDS, "BURS"):
            if self.query_bool(f"{self._src}{kind}:STAT?"):
                self.write(f"{self._src}{kind}:STAT OFF")
                self.link.wait_settled()
        if self.query(f"{self._src}FREQ:MODE?").strip().upper().startswith("SWE"):
            self.write(f"{self._src}FREQ:MODE CW")
            self.link.wait_settled()
        return self.get_modulation()

    def get_modulation(self) -> ModulationState:
        """Which modulation is running on this channel, and its settings.

        ``level`` is AM depth in percent, FM deviation in Hz, PM deviation in
        degrees, the FSK hop frequency in Hz, or the PWM deviation in seconds at
        the present frequency. ``rate`` is None with an external source.
        """
        src = self._src
        for kind in _MODULATION_KINDS:
            if not self.query_bool(f"{src}{kind}:STAT?"):
                continue
            source = member_from_response(
                self.query(f"{src}{kind}:SOUR?"), ModulationSource, f"{kind} source"
            )
            internal = source is ModulationSource.INTERNAL
            rate_header = "FSK:INT:RATE" if kind == "FSK" else f"{kind}:INT:FREQ"
            if kind == "PM":
                level = self.query_float(f"{src}PM?") * _DEG
            elif kind == "PWM":
                level = self.query_float(f"{src}PWM:DCYC?") / 100.0 / self.get_frequency()
            else:
                level = self.query_float(f"{src}{kind}?")
            return ModulationState(
                mode=kind,
                level=level,
                rate=self.query_float(f"{src}{rate_header}?") if internal else None,
                shape=None
                if kind == "FSK"
                else member_from_response(
                    self.query(f"{src}{kind}:INT:FUNC?"), ModulationShape, f"{kind} shape"
                ),
                source=source,
            )
        return ModulationState(mode=None)

    # ---------- burst ----------

    def set_burst(
        self,
        cycles: float,
        *,
        mode: BurstMode | str = BurstMode.TRIGGERED,
        delay: float | None = None,
        idle: BurstIdle | str | None = None,
        phase: float | None = None,
    ) -> BurstState:
        """Emit ``cycles`` cycles per trigger, 1 to 1000000 or ``float('inf')``, and turn burst on.

        Enum arg: mode = <afg>.BurstMode.<TAB>
                  idle = <afg>.BurstIdle.<TAB>

        The bursts come from the instrument's trigger, shared by both channels:
        set ``afg.set_trigger_source`` and, for TIMER, ``afg.set_trigger_timer``,
        which is then the burst period. ``cmd_trigger()`` fires one burst.
        ``delay`` is the time from the trigger to the first cycle, in seconds,
        TRIGGERED only. ``idle`` is what the output holds between bursts.
        ``phase`` is where each burst starts, in degrees. A burst ends any
        modulation or sweep. Settings left as None keep their present values.
        """
        function = self._get_function()
        if function not in _BURSTABLE:
            raise ValueError(
                f"set_burst() needs a sine, square, ramp, pulse, built-in or arbitrary carrier; "
                f"{self.name} is generating {function.name}. Set the carrier shape first."
            )
        if not (cycles == math.inf or (1 <= cycles <= 1_000_000 and float(cycles).is_integer())):
            raise ValueError(f"set_burst() refused: cycles is 1 to 1000000 or inf, got {cycles!r}")
        if phase is not None and function in _PHASELESS:
            raise ValueError(
                f"set_burst() refused: a {function.name} carrier has no phase; leave phase=None."
            )
        src = self._src
        token = _token(mode, BurstMode, "burst mode")
        if self.query(f"{src}BURS:MODE?").strip().upper() != token:
            # A mode write held the socket 55-58 ms (stage burst), so it is waited out.
            self.write(f"{src}BURS:MODE {token}")
            self.link.wait_settled()
        present = self.query_float(f"{src}BURS:NCYC?")
        wanted = _BURST_INFINITE if cycles == math.inf else float(cycles)
        if present != wanted:
            self.write(f"{src}BURS:NCYC {'INF' if cycles == math.inf else int(cycles)}")
        if delay is not None:
            self._write_checked(f"{src}BURS:TDEL", delay, _Q_BURST_DELAY, "burst delay")
        if phase is not None:
            self.set_phase(phase)
        if not self.query_bool(f"{src}BURS:STAT?"):
            self.write(f"{src}BURS:STAT ON")
            self.link.wait_settled()
        if idle is not None:
            # The instrument takes the idle state only while the burst is on (-221
            # otherwise, stages burst and followup).
            token = _token(idle, BurstIdle, "burst idle")
            if self.query(f"{src}BURS:IDLE?").strip().upper() != token:
                self.write(f"{src}BURS:IDLE {token}")
        return self.get_burst()

    def get_burst(self) -> BurstState:
        """The burst configuration, and whether it is running.

        ``period`` is the instrument's trigger timer, which paces the bursts
        only while the trigger source is TIMER. ``cycles`` is 9.9e37 for an
        infinite burst, as the instrument reports it.
        """
        src = self._src
        function = self._get_function()
        return BurstState(
            enabled=self.query_bool(f"{src}BURS:STAT?"),
            cycles=self.query_float(f"{src}BURS:NCYC?"),
            period=self.link.query_float("TRIG:TIM?"),
            phase=0.0 if function in _PHASELESS else self.get_phase(),
            mode=member_from_response(self.query(f"{src}BURS:MODE?"), BurstMode, "burst mode"),
            delay=self.query_float(f"{src}BURS:TDEL?"),
            idle=member_from_response(self.query(f"{src}BURS:IDLE?"), BurstIdle, "burst idle"),
        )

    def cmd_burst_rearm(self) -> None:
        """Stop an infinite burst and wait for the next trigger to start it again."""
        self.write(f"{self._src}BURS:INFI:REAR")

    # ---------- sweep ----------

    def set_sweep(
        self,
        start: float,
        stop: float,
        time: float,
        *,
        spacing: Spacing | str = Spacing.LINEAR,
        hold_time: float | None = None,
        return_time: float | None = None,
        mode: SweepMode | str = SweepMode.REPEAT,
    ) -> SweepState:
        """Sweep from ``start`` to ``stop`` Hz over ``time`` seconds, and turn the sweep on.

        Enum arg: spacing = <afg>.Spacing.<TAB>
                  mode    = <afg>.SweepMode.<TAB>

        ``hold_time`` dwells at the stop frequency and ``return_time`` sets the
        retrace, both in seconds; None keeps the present values. A sweep ends
        FM. The carrier must be a sine, square, ramp or arbitrary waveform.

        REPEAT runs continuously. TRIGGERED waits at ``start`` and runs one
        sweep per trigger, so it needs ``afg.set_trigger_source(EXTERNAL)``:
        with the source at TIMER the timer re-triggers it and it never waits.
        Bench-observed 2026-10-28 on channel 2: with EXTERNAL it sat at 1000 Hz,
        ``cmd_trigger()`` ran one 1 to 2 kHz sweep, and it returned to wait.
        Turning the sweep on clears noise add (stage followup2).
        """
        function = self._get_function()
        if function not in _SWEEPABLE:
            raise ValueError(
                f"set_sweep() needs a sine, square, ramp or arbitrary carrier; {self.name} is "
                f"generating {function.name}. Call set_waveform_sine(...) first."
            )
        spacing_token = _token(spacing, Spacing, "spacing")
        if self.query(f"{self._src}SWE:SPAC?").strip().upper() != spacing_token:
            self.write(f"{self._src}SWE:SPAC {spacing_token}")
        mode_token = _token(mode, SweepMode, "mode")
        if self.query(f"{self._src}SWE:MODE?").strip().upper() != mode_token:
            self.write(f"{self._src}SWE:MODE {mode_token}")
        self._write_checked(f"{self._src}FREQ:STAR", start, _Q_SWEEP_FREQUENCY, "sweep start")
        self._write_checked(f"{self._src}FREQ:STOP", stop, _Q_SWEEP_FREQUENCY, "sweep stop")
        self._write_checked(f"{self._src}SWE:TIME", time, _Q_SWEEP_TIME, "sweep time")
        if hold_time is not None:
            self._write_checked(f"{self._src}SWE:HTIM", hold_time, _Q_SWEEP_TIME, "hold time")
        if return_time is not None:
            self._write_checked(f"{self._src}SWE:RTIM", return_time, _Q_SWEEP_TIME, "return time")
        if not self.query(f"{self._src}FREQ:MODE?").strip().upper().startswith("SWE"):
            self.write(f"{self._src}FREQ:MODE SWE")
            self.link.wait_settled()
        return self.get_sweep()

    def get_sweep(self) -> SweepState:
        """The frequency sweep configuration, and whether it is running."""
        return SweepState(
            enabled=self.query(f"{self._src}FREQ:MODE?").strip().upper().startswith("SWE"),
            start=self.query_float(f"{self._src}FREQ:STAR?"),
            stop=self.query_float(f"{self._src}FREQ:STOP?"),
            time=self.query_float(f"{self._src}SWE:TIME?"),
            spacing=member_from_response(self.query(f"{self._src}SWE:SPAC?"), Spacing, "spacing"),
            hold_time=self.query_float(f"{self._src}SWE:HTIM?"),
            return_time=self.query_float(f"{self._src}SWE:RTIM?"),
        )

    def _get_sweep_mode(self) -> SweepMode:
        """Whether the sweep repeats or waits for a trigger (SOUR<n>:SWE:MODE?)."""
        return member_from_response(self.query(f"{self._src}SWE:MODE?"), SweepMode, "sweep mode")

    # ---------- plumbing ----------

    def _write_checked(
        self,
        setting: str,
        wanted: float,
        precision: tuple[int, float],
        label: str,
        *,
        truncates: bool = False,
        suffix: str = "",
        read_scale: float = 1.0,
        settle: bool = False,
    ) -> float:
        """Write ``setting`` only if it differs, then confirm the instrument took it.

        ``suffix`` is a unit sent after the number, and ``read_scale`` turns the
        read-back into the unit of ``wanted`` (radians to degrees, say). A value
        the instrument ``truncates`` is the same within a whole step. ``settle``
        waits out a busy window before the read-back.
        """
        slack = _quantum(wanted, *precision) if truncates else 0.0
        present = self.query_float(f"{setting}?") * read_scale
        if _same(present, wanted, precision, slack):
            return present
        self.write(f"{setting} {_num(wanted)}{suffix}")
        if settle:
            self.link.wait_settled()
        present = self.query_float(f"{setting}?") * read_scale
        if not _same(present, wanted, precision, slack):
            self._refused(label, wanted, present)
        return present

    def _refused(self, label: str, wanted: float, present: float) -> None:
        errors = _drain_errors(self.link)
        raise InstrumentError(
            f"{self.name} refused {label} {wanted:g}; it holds {present:g}. "
            f"Instrument errors: {errors or 'none'}. A -222 here usually means the "
            "value is outside the voltage limit or the shape's range."
        )


class _AddInChannel(_Channel):
    """Channel 1 on a model with the rear-panel Add In input.

    Add In sums an external signal into this output: -1 V to +1 V (DC plus
    peak AC), DC to 10 MHz (spec table 25). Channel 2 has no Add In; the
    instrument refuses it there (-150). The manuals give neither the gain nor
    any clamp on the sum, so both were measured; see set_external_add_state.
    """

    def set_external_add_state(self, state: bool) -> bool:
        """Sum the rear-panel Add In signal into this output, or stop summing it.

        Bench-measured 2026-10-28 with the scope's AFG on Add In and channel 1
        into a 1 Mohm scope input (harness section "addin", ADR-054):

        - The Add In signal reaches the output at 2.03 V/V open circuit, which
          is unity into 50 ohm, the same at 0.1, 1 and 4 Vpp settings.
        - The voltage limit does not touch it. With a +0.6 V limit and a 0.5 V
          sine peak, 0.4 V of Add In took the output to 1.36 V.
        - Nothing clips it near full scale. A 20 Vpp open-circuit sine plus
          0.4 V of Add In reached +11.24 / -11.08 V, which is +5.62 / -5.54 V
          into 50 ohm.
        - The input is 50 ohm only while this is on. With it off, a source set
          for a 50 ohm load delivered 0.79 V instead of 0.40 V.

        So size the Add In drive against the load: the channel 1 peak plus the
        Add In peak must stay inside what the far end can take. Keep the drive
        within the +-1 V input range. Internal noise add, if on, stays on: the
        instrument takes and reads back "NOIS", "BOTH", "EXT" and "" on channel
        1 (bench-observed 2026-10-28, stage noiseadd).
        """
        noise = self._get_combine_feed() in ("NOIS", "BOTH")
        token = {
            (False, False): "",
            (True, False): "EXT",
            (False, True): "NOIS",
            (True, True): "BOTH",
        }[(state, noise)]
        self.write(f'{self._src}COMB:FEED "{token}"')
        return self.get_external_add_state()

    def get_external_add_state(self) -> bool:
        """Whether the rear-panel Add In signal is summed into this output."""
        return self._get_combine_feed() in ("EXT", "BOTH")


class _InterChannel(Node):
    """The front panel's InterChannel tab: what ties channel 1 and channel 2 together.

    Present on two-channel models only.
    """

    def __init__(self, link: _AfgLink):
        super().__init__(link, name="INTERCHANNEL")

    def cmd_phase_sync(self) -> None:
        """Line both channels' phases up, as the panel's Align Phase does.

        The instrument stops both outputs' waveforms, restarts them together,
        and keeps each channel's phase setting as the offset between them. A
        frequency change on one channel loses the alignment (user manual). So
        does turning a modulation off: AM, FM or PM off, on either channel, left
        an arbitrary phase between them, 9 times of 9. AM or FM turning on kept
        the alignment, and external PM turning on moved its channel by exactly
        180 degrees (fitted from scope records, harness section extmod,
        ADR-057). The alignment is back as soon as this returns.
        """
        self.write("SOUR1:PHAS:INIT")

    def _set_coupling(self, header: str, state: bool) -> bool:
        if self._get_coupling(header) != state:
            self.write(f"SOUR1:{header}:CONC {'ON' if state else 'OFF'}")
        return self._get_coupling(header)

    def _get_coupling(self, header: str) -> bool:
        return self.query_bool(f"SOUR1:{header}:CONC?")

    def set_frequency_coupling_state(self, state: bool) -> bool:
        """Keep both channels at one frequency (panel: Frequency CH1=CH2); returns the read-back.

        Turning it on copies channel 1's frequency to channel 2. From then on a
        frequency set on either channel moves the other (bench-observed
        2026-10-28, stage chan2).
        """
        return self._set_coupling("FREQ", state)

    def get_frequency_coupling_state(self) -> bool:
        """Whether both channels are kept at one frequency."""
        return self._get_coupling("FREQ")

    def set_level_coupling_state(self, state: bool) -> bool:
        """Keep both channels at one amplitude and offset (panel: Level CH1=CH2).

        Turning it on copies channel 1's amplitude and offset to channel 2, in
        Vpp across shapes, and from then on either channel moves the other. A
        level outside either channel's voltage limit is refused on both
        (bench-observed 2026-10-28, stage followup). With the channels' panel
        amplitude units different, the copied numbers came out unit-dependent,
        so this refuses then: set both units alike first.
        """
        if state:
            units = {self.query(f"SOUR{n}:VOLT:UNIT?").strip().upper() for n in (1, 2)}
            if len(units) > 1:
                raise ValueError(
                    "set_level_coupling_state(True) refused: the two channels show the "
                    f"amplitude in different units ({', '.join(sorted(units))}), and the "
                    "instrument then copies an unrelated number. Call set_amplitude_unit() "
                    "on both channels with the same unit first."
                )
        return self._set_coupling("VOLT", state)

    def get_level_coupling_state(self) -> bool:
        """Whether both channels are kept at one amplitude and offset."""
        return self._get_coupling("VOLT")

    def set_phase_coupling_state(self, state: bool) -> bool:
        """Keep both channels at one phase setting; turning it on copies channel 1's to channel 2."""
        return self._set_coupling("PHAS", state)

    def get_phase_coupling_state(self) -> bool:
        """Whether both channels are kept at one phase setting."""
        return self._get_coupling("PHAS")

    def _copy(self, source: int, target: int) -> None:
        self.write(f"AFGC:CSC CH{source},CH{target}")
        self.link.wait_settled()

    def cmd_copy_ch1_to_ch2(self) -> None:
        """Copy channel 1's whole setup to channel 2, and turn both outputs off.

        Copies the shape, frequency, levels, panel unit, load setting, voltage
        limits, polarity and run mode. The instrument turns BOTH outputs off as
        it copies: with both on, both read off after it, either way round
        (bench-observed 2026-10-28, 0.59-0.73 s). Turn them on again after.
        """
        self._copy(1, 2)

    def cmd_copy_ch2_to_ch1(self) -> None:
        """Copy channel 2's whole setup to channel 1, and turn both outputs off."""
        self._copy(2, 1)


class _Memory(Node):
    """The five setup memories, 0 to 4. Memory 0 is overwritten with the last setup at power off."""

    def __init__(self, link: _AfgLink):
        super().__init__(link, name="MEMORY")

    @staticmethod
    def _check(location: int, low: int = 0) -> int:
        if location not in range(low, 5):
            raise ValueError(f"a setup memory is {low} to 4, got {location!r}")
        return location

    def _checked_write(self, cmd: str) -> None:
        self.write(cmd)
        self.query("*OPC?")
        errors = _drain_errors(self.link)
        if errors:
            raise InstrumentError(f"{cmd} was refused: {errors}")

    def cmd_save_state(self, location: int) -> None:
        """Save every setting to setup memory ``location``, 0 to 4; a locked one refuses.

        Took 0.32 s. A locked memory refuses with -258 (bench-observed 2026-10-28).
        """
        self._checked_write(f"*SAV {self._check(location)}")

    def cmd_recall_state(self, location: int) -> None:
        """Recall setup memory ``location``. It never turns an output on.

        A setup saved with an output on came back with it off (bench-observed
        2026-10-28, 0.13-0.16 s). An empty memory refuses with -256.
        """
        self._checked_write(f"*RCL {self._check(location)}")

    def cmd_delete_state(self, location: int) -> None:
        """Delete setup memory ``location``, so it reads back as not valid."""
        self._checked_write(f"MEM:STAT:DEL {self._check(location)}")

    def get_state_valid(self, location: int) -> bool:
        """Whether setup memory ``location`` holds a saved setup."""
        return self.query_bool(f"MEM:STAT:VAL? {self._check(location)}")

    def set_lock_state(self, location: int, state: bool) -> bool:
        """Protect setup memory ``location``, 1 to 4, from saves and deletes; returns the read-back."""
        self.write(f"MEM:STAT:LOCK {self._check(location, 1)},{'ON' if state else 'OFF'}")
        return self.get_lock_state(location)

    def get_lock_state(self, location: int) -> bool:
        """Whether setup memory ``location`` is protected from saving and deleting."""
        return self.query_bool(f"MEM:STAT:LOCK? {self._check(location, 1)}")

    def set_auto_recall_state(self, state: bool) -> bool:
        """Power on with the setup the instrument had at power off, or with the defaults."""
        self.write(f"MEM:STAT:REC:AUTO {'ON' if state else 'OFF'}")
        return self.get_auto_recall_state()

    def get_auto_recall_state(self) -> bool:
        """Whether the instrument powers on with the setup it had at power off."""
        return self.query_bool("MEM:STAT:REC:AUTO?")


class _File(Node):
    """Files on the instrument's drives: ``M:/`` internal, ``U:/`` a USB stick.

    Paths use forward slashes. Waveform files are .tfwx and setup files .tfs.
    The driver keeps its own files in ``M:/pyvisa_tmp``.
    """

    def __init__(self, link: _AfgLink):
        super().__init__(link, name="FILE")

    def _checked_write(self, cmd: str) -> None:
        self.write(cmd)
        self.query("*OPC?")
        errors = _drain_errors(self.link)
        if errors:
            raise InstrumentError(f"{cmd} was refused: {errors}")

    def get_catalog(self, directory: str = "M:") -> list[FileEntry]:
        """The files and folders in ``directory``, each with its size in bytes.

        The names can lag behind the drive by seconds; see :class:`FileEntry`.
        """
        return _catalog(self.link, directory)

    def cmd_save_setup(self, path: str) -> str:
        """Save every setting to a .tfs setup file at ``path``; returns ``path``. Took 0.32 s.

        On the internal drive the file must sit in the root folder, ``M:/name.tfs``:
        any path in a subfolder was refused with -257 "illegal file name"
        (bench-observed 2026-10-28). Whether USB is the same was not tried.
        """
        self._checked_write(f'SAV:SETU "{path}"')
        return path

    def cmd_recall_setup(self, path: str) -> None:
        """Recall every setting from the .tfs setup file at ``path``. Took 0.17 s.

        It never turns an output on: a file saved with channel 2 on came back
        with it off, as a setup memory does (bench-observed 2026-10-28).
        """
        self._checked_write(f'RECALL:SETU "{path}"')

    def cmd_delete(self, path: str) -> None:
        """Delete the file, or the empty folder, at ``path``. Took 0.17 s."""
        self._checked_write(f'MMEM:DEL "{path}"')

    def cmd_make_directory(self, path: str) -> None:
        """Create a folder at ``path``."""
        self._checked_write(f'MMEM:MDIR "{path}"')


class Tektronix_AFG31000(Instrument):
    """Tektronix AFG31000 series arbitrary function generator.

    ``__init__`` does no I/O. ``connect()`` reads ``*IDN?`` and attaches what the
    model has: ``ch2`` on two-channel models only, and the Add In controls on
    ``ch1`` only on the 100 MHz and faster models.
    """

    # Declared for the API-map generator, which cannot see what connect()
    # attaches. No assignment, so nothing exists until it is real.
    ch1: _AddInChannel  # a plain channel, without Add In, on the 25 and 50 MHz models
    ch2: _Channel  # two-channel models only
    interchannel: _InterChannel  # two-channel models only
    memory: _Memory
    file: _File

    # Enums live on the class so afg.Function.<TAB> resolves from the live object.
    Function = Function
    AmplitudeUnit = AmplitudeUnit
    Load = Load
    ModulationShape = ModulationShape
    ModulationSource = ModulationSource
    Spacing = Spacing
    SweepMode = SweepMode
    RunMode = RunMode
    Reference = Reference
    TriggerSource = TriggerSource
    Polarity = Polarity
    BurstMode = BurstMode
    BurstIdle = BurstIdle
    Slope = Slope
    TriggerOutMode = TriggerOutMode

    def __init__(self, resource_name: str, *, connect_timeout_ms: int = _DEFAULT_TIMEOUT_MS):
        """
        Parameters
        ----------
        resource_name : str
            pyvisa resource for the generator, read off its front panel. The raw
            socket, ``TCPIP0::<address>::5025::SOCKET``, is the recommended one.
        connect_timeout_ms : int
            I/O timeout for connect and normal operation, in milliseconds.
        """
        super().__init__(resource_name, name="AFG")
        self.link = _AfgLink(resource_name, timeout_ms=connect_timeout_ms)
        self._connect_timeout_ms = connect_timeout_ms

        self.model: str | None = None
        self.firmware: str | None = None
        self.options: list[str] = []
        self.num_channels: int = 2
        self.max_frequency: float | None = None
        self.has_add_in: bool = True

        self.ch1 = _AddInChannel(self.link, 1)
        self.interchannel = _InterChannel(self.link)  # removed by connect() on one-channel models
        self.memory = _Memory(self.link)
        self.file = _File(self.link)

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the pinned resource, confirm it is an AFG31000, and attach the channels.

        A VISA timeout on the first ``*IDN?`` raises
        :class:`InstrumentUnresponsiveError` and is never retried. An answer
        from some other instrument raises :class:`InstrumentError`: a wrong
        address is fixed by reading the right one off the front panel, never by
        scanning (ADR-051).
        """
        to = self._connect_timeout_ms if timeout_ms is None else timeout_ms
        try:
            self.link.connect(timeout_ms=to)
            idn = self.link.query("*IDN?").strip()
        except visa.VisaIOError as exc:
            if exc.error_code == visa.constants.StatusCode.error_timeout:
                self.link.disconnect()
                busy = (
                    " The raw socket serves one client at a time, so another session "
                    "holding it causes exactly this; close that session first. A socket "
                    "left deaf by a stalled reply answers again only after a restart "
                    "(ADR-055)."
                    if self.link.is_socket
                    else ""
                )
                raise InstrumentUnresponsiveError(
                    f"{self.resource_name} did not answer *IDN? (VISA timeout).{busy} Check "
                    "the address on the front panel and the cable; do not retry in a loop."
                ) from exc
            raise
        if not _IDN_RE.search(idn):
            self.link.disconnect()
            raise InstrumentError(
                f"{self.resource_name} did not answer with an AFG31000 IDN: {idn!r}. "
                "Check the resource; this driver does not scan for the instrument."
            )
        self.idn = idn
        self._on_connect(idn)
        self.link.write("*CLS")
        return idn

    def _on_connect(self, idn: str) -> None:
        """Parse the IDN and attach the nodes this model has."""
        match = _IDN_RE.search(idn)
        if not match:
            raise InstrumentError(f"Not a recognized AFG31000 IDN: {idn!r}")
        self.model = match.group(1).upper()
        fields = [f.strip() for f in idn.split(",")]
        self.firmware = fields[3] if len(fields) >= 4 else None
        info = _MODEL_INFO.get(self.model)
        if info is None:
            # The family matched but the model is new: assume the smallest
            # feature set, so nothing is attached that might not exist.
            info = {"channels": 1, "max_sine": None, "v_peak": None, "add_in": False}
        self.num_channels = info["channels"]
        self.max_frequency = info["max_sine"]
        self.has_add_in = info["add_in"]

        wanted = _AddInChannel if self.has_add_in else _Channel
        if type(self.ch1) is not wanted:
            self.ch1 = wanted(self.link, 1)
        if self.num_channels == 2:
            if not hasattr(self, "ch2"):
                self.ch2 = _Channel(self.link, 2)
            if not hasattr(self, "interchannel"):
                self.interchannel = _InterChannel(self.link)
        else:
            for name in ("ch2", "interchannel"):
                if hasattr(self, name):
                    delattr(self, name)
        with contextlib.suppress(Exception):
            self.options = self.get_options()

    def safe_state(self) -> None:
        """Turn every output off before the link closes. Best effort; never raises."""
        for ch in self._channels():
            with contextlib.suppress(Exception):
                self.link.write(f"OUTP{ch.n} OFF")

    def _channels(self) -> list[_Channel]:
        found = [self.ch1]
        if hasattr(self, "ch2"):
            found.append(self.ch2)
        return found

    def channel(self, n: int) -> _Channel:
        """Return the channel node for ``n``."""
        channels = self._channels()
        if n not in range(1, len(channels) + 1):
            raise ValueError(
                f"channel must be 1..{len(channels)} on a {self.model or 'this model'}, got {n!r}"
            )
        return channels[n - 1]

    # ---------- both outputs at once ----------

    def set_output_state_all(self, state: bool) -> list[bool]:
        """Turn every channel's output on or off; returns each read-back in order.

        The outputs are switched one at a time, not simultaneously.
        """
        for ch in self._channels():
            ch.set_output_state(state)
        return self.get_output_state_all()

    def get_output_state_all(self) -> list[bool]:
        """Whether each channel's output is on, in channel order."""
        return [ch.get_output_state() for ch in self._channels()]

    # ---------- the 10 MHz reference ----------

    def set_reference_clock_external(self) -> ReferenceClockState:
        """Run the timebase from the 10 MHz on the rear Ext Ref input.

        The input needs 100 mVpp to 5 Vpp within 35 kHz of 10 MHz (spec table
        25). If it is absent, the instrument keeps reporting EXTERNAL and
        raises only ``frequency_questionable``, so check that on the result.
        """
        self.link.write(f"ROSC:SOUR {Reference.EXTERNAL.value}")
        return self.get_reference_clock()

    def set_reference_clock_internal(self) -> ReferenceClockState:
        """Run the timebase from the internal oscillator, ignoring the rear input."""
        self.link.write(f"ROSC:SOUR {Reference.INTERNAL.value}")
        return self.get_reference_clock()

    def get_reference_clock(self) -> ReferenceClockState:
        """Which timebase is selected, and whether the instrument doubts its frequency."""
        selected = member_from_response(self.link.query("ROSC:SOUR?"), Reference, "reference")
        status = self.link.query_int("STAT:QUES:COND?")
        return ReferenceClockState(
            selected=selected, frequency_questionable=bool(status & _QUES_FREQUENCY)
        )

    # ---------- the trigger ----------

    def cmd_trigger(self) -> None:
        """Send a trigger: one burst, or one sweep in TRIGGERED sweep mode."""
        self.link.write("*TRG")

    def set_trigger_source(self, source: TriggerSource | str) -> TriggerSource:
        """Choose what triggers a burst or a TRIGGERED sweep, for both channels.

        Enum arg: <afg>.TriggerSource.<TAB>

        EXTERNAL takes the front-panel Trigger In, and ``cmd_trigger()`` also
        fires it. TIMER fires on the internal timer; see :meth:`set_trigger_timer`.
        """
        self.link.write(f"TRIG:SOUR {_token(source, TriggerSource, 'trigger source')}")
        return self.get_trigger_source()

    def get_trigger_source(self) -> TriggerSource:
        """What triggers a burst or a TRIGGERED sweep."""
        raw = self.link.query("TRIG:SOUR?")
        return member_from_response(raw, TriggerSource, "trigger source")

    def set_trigger_timer(self, seconds: float) -> float:
        """Set the internal trigger period in seconds, 1 us to 500 s; returns the read-back.

        With the trigger source at TIMER this is the burst period of both
        channels. Held to 4 digits (bench-measured 2026-10-28).
        """
        if not 1e-6 <= seconds <= 500.0:
            raise ValueError(f"set_trigger_timer() refused: {seconds:g} s is outside 1 us to 500 s")
        present = self.get_trigger_timer()
        if not _same(present, seconds, _Q_TIMER):
            self.link.write(f"TRIG:TIM {_num(seconds)}")
            present = self.get_trigger_timer()
            if not _same(present, seconds, _Q_TIMER):
                raise InstrumentError(
                    f"the instrument refused trigger timer {seconds:g} s; it holds {present:g} s. "
                    f"Instrument errors: {self.get_errors() or 'none'}."
                )
        return present

    def get_trigger_timer(self) -> float:
        """The internal trigger period in seconds."""
        return self.link.query_float("TRIG:TIM?")

    def set_trigger_slope(self, slope: Slope | str) -> Slope:
        """Choose which edge of the front Trigger In fires; a :class:`Slope`.

        Enum arg: <afg>.Slope.<TAB>

        POSITIVE starts a burst on Trigger In's rising edge and NEGATIVE on its
        falling edge, each about 0.2 us later. In a gated burst it picks the
        level that opens the gate: high for POSITIVE, low for NEGATIVE. Bench,
        2026-10-28, Trigger In driven from Trigger Out (ADR-057).
        """
        token = _token(slope, Slope, "slope")
        if self.get_trigger_slope().value != token:
            self.link.write(f"TRIG:SLOP {token}")
        return self.get_trigger_slope()

    def get_trigger_slope(self) -> Slope:
        """Which edge of the front Trigger In fires."""
        return member_from_response(self.link.query("TRIG:SLOP?"), Slope, "trigger slope")

    def set_trigger_out_mode(self, mode: TriggerOutMode | str) -> TriggerOutMode:
        """Choose what the front Trigger Out marks during an infinite burst.

        Enum arg: <afg>.TriggerOutMode.<TAB>

        Trigger Out follows channel 1's run mode only, as a 0 to 3.3 V logic
        level, even with channel 1's output off. Measured 2026-10-28 (ADR-057):
        one 50 % pulse per period rising at channel 1's phase-0 point, divided
        down above 4.9 MHz and absent above 50 MHz (user manual table 9); one
        per sweep; one per cycle of an internal modulation and nothing for an
        external one; one per burst, rising as it starts. In an infinite burst
        TRIGGER gives one pulse per cycle and SYNC one pulse as it starts, half
        a timer period wide on the timer. With the trigger source EXTERNAL, a
        burst holds it high while Trigger In is high.
        """
        token = _token(mode, TriggerOutMode, "trigger out mode")
        if self.get_trigger_out_mode().value != token:
            self.link.write(f"OUTP:TRIG:MODE {token}")
        return self.get_trigger_out_mode()

    def get_trigger_out_mode(self) -> TriggerOutMode:
        """What the front Trigger Out marks during an infinite burst."""
        raw = self.link.query("OUTP:TRIG:MODE?")
        return member_from_response(raw, TriggerOutMode, "trigger out mode")

    # ---------- system ----------

    def get_errors(self) -> list[str]:
        """Drain and return the error queue; an empty list means clean.

        The queue is per interface: an error from a VXI-11 session does not
        appear on the socket (ADR-054).
        """
        return _drain_errors(self.link)

    def cmd_reset(self) -> None:
        """Reset every setting to the factory default: a 1 MHz, 1 Vpp sine into 50 ohm.

        Took 1.2-1.4 s over the socket and 1.15 s over VXI-11, where it opens
        no busy window (bench-measured 2026-10-28). It keeps the setup memories,
        the panel lock, the key click and the display brightness.
        """
        self.link.write("*RST")
        self.link.wait_settled()

    def cmd_clear_status(self) -> None:
        """Clear the status byte, the event registers and the error queue."""
        self.link.write("*CLS")

    def get_options(self) -> list[str]:
        """The installed options as a list of tokens; empty when there are none."""
        raw = self.link.query("*OPT?").strip().strip('"')
        return [
            o.strip().strip('"') for o in raw.split(",") if o.strip().strip('"') not in ("", "0")
        ]

    def set_front_panel_lock_state(self, state: bool) -> bool:
        """Lock the front panel's keys and knob against use, or unlock them; returns the read-back."""
        if self.get_front_panel_lock_state() != state:
            self.link.write(f"SYST:KLOC {'ON' if state else 'OFF'}")
        return self.get_front_panel_lock_state()

    def get_front_panel_lock_state(self) -> bool:
        """Whether the front panel's keys and knob are locked."""
        return self.link.query_bool("SYST:KLOC?")

    def cmd_beep(self) -> None:
        """Sound the beeper once."""
        self.link.write("SYST:BEEP")

    def self_test(self) -> int:
        """Run the internal self-test; 0 means pass, anything else a failure.

        Took 136 s on the bench AFG31102, during which the instrument answers
        nothing; the call waits up to 10 minutes. The outputs and the load
        settings were as before it afterwards (bench-measured 2026-10-28).
        """
        with self.link.temporary_timeout(_SELF_TEST_TIMEOUT_MS):
            return self.link.query_int("*TST?")

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Read back every setting this driver can change, for an exact restore.

        Each channel's edit memory is copied to a file in ``M:/pyvisa_tmp`` on
        the instrument rather than read over the wire, which a large memory
        would hang (ADR-055); the copy back was byte-exact on the bench.
        :meth:`preserved_setup` deletes those files when it is done; a bare
        snapshot leaves them, named in ``snapshot["files"]``.
        Never uses ``*SAV``, which would overwrite one of the operator's setup
        memories. Takes about 1.5 s.
        """
        _ensure_scratch_dir(self.link)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        snap = self._read_state(stamp)
        return snap

    def _read_state(self, stamp: str | None) -> dict:
        """Every setting; with a ``stamp``, each edit memory copied to a file as well."""
        link = self.link
        snap: dict = {
            "reference": link.query("ROSC:SOUR?").strip(),
            "trigger_source": link.query("TRIG:SOUR?").strip(),
            "trigger_timer": link.query_float("TRIG:TIM?"),
            "trigger_slope": link.query("TRIG:SLOP?").strip(),
            "trigger_out_mode": link.query("OUTP:TRIG:MODE?").strip(),
            "panel_lock": link.query_bool("SYST:KLOC?"),
            "couplings": (
                {h: link.query_bool(f"SOUR1:{h}:CONC?") for h in _COUPLINGS}
                if hasattr(self, "interchannel")
                else {}
            ),
            "files": {},
            "channels": {},
        }
        for ch in self._channels():
            src = ch._src
            function = ch._get_function()
            if stamp is not None:
                path = f"{_SCRATCH_DIR}/snapshot_{stamp}_ch{ch.n}.tfwx"
                ch.cmd_save_arb_file(path)
                snap["files"][ch.n] = path
            mod = {}
            for kind in ("AM", "PM", "PWM"):
                level = "PWM:DCYC" if kind == "PWM" else kind
                mod[kind] = {
                    "level": ch.query_float(f"{src}{level}?"),
                    "rate": ch.query_float(f"{src}{kind}:INT:FREQ?"),
                    "shape": ch.query(f"{src}{kind}:INT:FUNC?").strip(),
                    "source": ch.query(f"{src}{kind}:SOUR?").strip(),
                }
            mod["FSK"] = {
                "level": ch.query_float(f"{src}FSK?"),
                "rate": ch.query_float(f"{src}FSK:INT:RATE?"),
                "source": ch.query(f"{src}FSK:SOUR?").strip(),
            }
            snap["channels"][ch.n] = {
                "output": ch.get_output_state(),
                "load": ch.get_load(),
                "limits": ch.get_voltage_limits(),
                "unit": ch.get_amplitude_unit(),
                "polarity": ch.get_polarity(),
                "function": function,
                "arb_file": ch._get_arb_file(),
                "amplitude_in_unit": ch.query_float(f"{src}VOLT?"),
                "offset": ch.get_offset(),
                "frequency": ch.get_frequency(),
                "phase": ch._get_phase(),
                "symmetry": ch.query_float(f"{src}FUNC:RAMP:SYMM?"),
                "pulse": {
                    "period": ch.query_float(f"{src}PULS:PER?"),
                    "hold": ch._get_pulse_hold(),
                    "width": ch.query_float(f"{src}PULS:WIDT?"),
                    "lead": ch.query_float(f"{src}PULS:TRAN?"),
                    "trail": ch.query_float(f"{src}PULS:TRAN:TRA?"),
                    "delay": ch.query_float(f"{src}PULS:DEL?"),
                },
                "fm": {
                    "deviation": ch.query_float(f"{src}FM?"),
                    "rate": ch.query_float(f"{src}FM:INT:FREQ?"),
                    "shape": ch.query(f"{src}FM:INT:FUNC?").strip(),
                    "source": ch.query(f"{src}FM:SOUR?").strip(),
                },
                "modulation": mod,
                "sweep": ch.get_sweep(),
                "sweep_mode": ch._get_sweep_mode(),
                "burst": {
                    "mode": ch.query(f"{src}BURS:MODE?").strip(),
                    "cycles": ch.query_float(f"{src}BURS:NCYC?"),
                    "delay": ch.query_float(f"{src}BURS:TDEL?"),
                    "idle": ch.query(f"{src}BURS:IDLE?").strip(),
                },
                "states": {
                    k: ch.query_bool(f"{src}{k}:STAT?") for k in (*_MODULATION_KINDS, "BURS")
                },
                "noise_level": ch.query_float(f"SOUR{ch.n + 2}:POW?"),
                "combine": ch._get_combine_feed(),
            }
        return snap

    def restore(self, snapshot: dict) -> None:
        """Restore the settings captured by :meth:`snapshot`, exactly as found.

        Each output goes off first, so nothing is energized while it is being
        configured, and goes back to its found state last. The channel
        couplings go off first and back on near the end, so one channel's
        writes do not move the other. The result is then read back and
        compared with the snapshot, applied once more if it differs, and
        raises if it still differs or if the instrument queued an error. The
        edit memories are reloaded from their files but not compared, because
        reading one back can hang the instrument.
        """
        self._apply_snapshot(snapshot)
        if self._differences(snapshot):
            # A write can land scaled for a stale load setting (see set_load), so
            # the result is checked rather than assumed, and applied once more.
            self._apply_snapshot(snapshot)
            differing = self._differences(snapshot)
            if differing:
                raise InstrumentError(f"restore() could not put back: {differing}")

    def _apply_snapshot(self, snapshot: dict) -> None:
        """Write every setting in ``snapshot``, in the order restore() describes.

        Per channel, three passes. With the pulse shape selected: the pulse
        settings and PWM, because the instrument checks a pulse width against
        the frequency of whatever shape is selected (stage followup). With the
        sine selected: the AM, FM, PM, FSK, sweep and burst settings, which
        then have their widest ranges. With the found shape: the levels, the
        frequency, the phase and the run mode.
        """
        link = self.link
        for header in snapshot["couplings"]:
            link.write(f"SOUR1:{header}:CONC OFF")
        for n, s in snapshot["channels"].items():
            ch = self.channel(n)
            src = ch._src
            ch.set_output_state(False)
            ch.set_modulation_off()
            load = s["load"]
            ch.set_load(Load.HIGHZ if load == float("inf") else load)
            ch._ensure_limits(s["limits"].low, s["limits"].high)
            ch.set_amplitude_unit(s["unit"])
            ch.set_polarity(s["polarity"])
            if n in snapshot["files"]:
                ch.write(f'MMEM:LOAD:TRAC EMEM{n},"{snapshot["files"][n]}"')
                ch.query("*OPC?")
            if s["arb_file"] and s["arb_file"] != ch._get_arb_file():
                # An empty name is refused (-232, 2026-10-28), so "none" cannot
                # be put back; it matters only while the file shape is selected.
                ch.write(f'{src}FUNC:EFIL "{s["arb_file"]}"')

            ch._set_function(Function.PULSE)
            p = s["pulse"]
            ch.write(f"{src}PULS:HOLD {PulseHold.WIDTH.value}")
            ch.write(f"{src}PULS:TRAN MIN")
            ch.write(f"{src}PULS:TRAN:TRA MIN")
            # As in _apply_pulse: a shorter period takes the width first, so the
            # held width always fits the period.
            shorter = 1.0 / p["period"] > ch.get_frequency()
            if shorter:
                ch.write(f"{src}PULS:WIDT {_num(p['width'])}")
            ch.write(f"{src}FREQ {_num(1.0 / p['period'])}")
            if not shorter:
                ch.write(f"{src}PULS:WIDT {_num(p['width'])}")
            ch.write(f"{src}PULS:TRAN {_num(p['lead'])}")
            ch.write(f"{src}PULS:TRAN:TRA {_num(p['trail'])}")
            ch.write(f"{src}PULS:DEL {_num(p['delay'])}")
            ch.write(f"{src}PULS:HOLD {p['hold'].value}")
            pwm = s["modulation"]["PWM"]
            ch.write(f"{src}PWM:DCYC {_num(pwm['level'])}")
            self._restore_internal_mod(ch, "PWM", pwm)

            ch._set_function(Function.SINE)
            fm = s["fm"]
            # The deviation register can hold more than the carrier, but a
            # write of it is refused (-222) unless the carrier is at least as
            # high at that moment; lowering the carrier afterwards keeps it
            # (bench-measured 2026-10-28: 1000 Hz on a 1 kHz carrier accepted,
            # 1001 Hz refused; a 5 kHz deviation survived the carrier going from
            # 10 kHz to 1 kHz).
            ch.write(f"{src}FREQ {_num(max(s['frequency'], fm['deviation']))}")
            ch.write(f"{src}FM {_num(fm['deviation'])}")
            self._restore_internal_mod(ch, "FM", fm)
            mod = s["modulation"]
            ch.write(f"{src}AM {_num(mod['AM']['level'])}")
            self._restore_internal_mod(ch, "AM", mod["AM"])
            ch.write(f"{src}PM {_num(mod['PM']['level'])}RAD")
            self._restore_internal_mod(ch, "PM", mod["PM"])
            ch.write(f"{src}FSK {_num(mod['FSK']['level'])}")
            ch.write(f"{src}FSK:INT:RATE {_num(mod['FSK']['rate'])}")
            ch.write(f"{src}FSK:SOUR {mod['FSK']['source']}")
            sweep = s["sweep"]
            ch.write(f"{src}SWE:SPAC {sweep.spacing.value}")
            ch.write(f"{src}SWE:MODE {s['sweep_mode'].value}")
            ch.write(f"{src}FREQ:STAR {_num(sweep.start)}")
            ch.write(f"{src}FREQ:STOP {_num(sweep.stop)}")
            ch.write(f"{src}SWE:TIME {_num(sweep.time)}")
            ch.write(f"{src}SWE:HTIM {_num(sweep.hold_time)}")
            ch.write(f"{src}SWE:RTIM {_num(sweep.return_time)}")
            # The phase register answers on every shape but is written only on
            # one that has a phase (programmer manual), so it goes in here.
            ch._set_phase(s["phase"])
            burst = s["burst"]
            ch.write(f"{src}BURS:MODE {burst['mode']}")
            link.wait_settled()
            cycles = burst["cycles"]
            ch.write(f"{src}BURS:NCYC {'INF' if cycles >= _HIGHZ_THRESHOLD else int(cycles)}")
            ch.write(f"{src}BURS:TDEL {_num(burst['delay'])}")

            # Entering the pulse can take the previous shape's frequency and clamp
            # the width for good (stages shapes2, followup), and the FM pass may
            # have left the sine higher, so the found frequency goes in first.
            ch.write(f"{src}FREQ {_num(s['frequency'])}")
            ch._set_function(s["function"])
            ch.write(f"{src}FUNC:RAMP:SYMM {_num(s['symmetry'])}")
            # As in _ensure_levels: a falling amplitude before the offset and a
            # rising one after, so the swing never leaves the restored limit.
            volt = f"{src}VOLT {_num(s['amplitude_in_unit'])}{s['unit'].value}"
            falling = s["amplitude_in_unit"] < ch.query_float(f"{src}VOLT?")
            if falling:
                ch.write(volt)
            ch.write(f"{src}VOLT:OFFS {_num(s['offset'])}")
            if not falling:
                ch.write(volt)
            ch.write(f"{src}FREQ {_num(s['frequency'])}")
            for kind, on in s["states"].items():
                if on:
                    ch.write(f"{src}{kind}:STAT ON")
                    link.wait_settled()
            if s["states"]["BURS"]:
                # The idle state is taken only while the burst is on (stage followup).
                ch.write(f"{src}BURS:IDLE {burst['idle']}")
            if sweep.enabled:
                ch.write(f"{src}FREQ:MODE SWE")
                link.wait_settled()
            # Last: a modulation or the sweep turning on clears noise add, while noise
            # add written over a running AM stays (stage followup2, ADR-057).
            ch.write(f"SOUR{n + 2}:POW {_num(s['noise_level'])}")
            if isinstance(ch, _AddInChannel) or s["combine"] in ("", "NOIS"):
                ch.write(f'{src}COMB:FEED "{s["combine"]}"')
        for header, on in snapshot["couplings"].items():
            if on:
                link.write(f"SOUR1:{header}:CONC ON")
        link.write(f"ROSC:SOUR {snapshot['reference']}")
        link.write(f"TRIG:SOUR {snapshot['trigger_source']}")
        link.write(f"TRIG:TIM {_num(snapshot['trigger_timer'])}")
        link.write(f"TRIG:SLOP {snapshot['trigger_slope']}")
        link.write(f"OUTP:TRIG:MODE {snapshot['trigger_out_mode']}")
        link.write(f"SYST:KLOC {'ON' if snapshot['panel_lock'] else 'OFF'}")
        for n, s in snapshot["channels"].items():
            self.channel(n).set_output_state(s["output"])
        errors = self.get_errors()
        if errors:
            raise InstrumentError(f"restore() finished, but the instrument queued: {errors}")

    @staticmethod
    def _restore_internal_mod(ch: _Channel, kind: str, saved: dict) -> None:
        """Write one modulation's internal rate, shape and source from a snapshot."""
        src = ch._src
        ch.write(f"{src}{kind}:INT:FREQ {_num(saved['rate'])}")
        shape = f"EMEM{ch.n}" if saved["shape"] == "EMEM" else saved["shape"]
        ch.write(f"{src}{kind}:INT:FUNC {shape}")
        ch.write(f"{src}{kind}:SOUR {saved['source']}")

    def _differences(self, snapshot: dict) -> list[str]:
        """Which snapshotted settings the instrument does not hold now."""
        now = self._read_state(None)
        found = []
        for n, wanted in snapshot["channels"].items():
            for key, value in wanted.items():
                have = now["channels"][n][key]
                if key == "arb_file" and not value:
                    continue  # an empty file name cannot be written back (-232)
                if key == "burst" and not wanted["states"]["BURS"]:
                    # Idle cannot be written with the burst off, so it is not compared.
                    have = {k: v for k, v in have.items() if k != "idle"}
                    value = {k: v for k, v in value.items() if k != "idle"}
                if have != value:
                    found.append(f"CH{n} {key}")
        for key in (
            "reference",
            "trigger_source",
            "trigger_timer",
            "trigger_slope",
            "trigger_out_mode",
            "panel_lock",
            "couplings",
        ):
            if now[key] != snapshot[key]:
                found.append(key)
        return found

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator:
        """Snapshot every setting this driver touches on enter, and restore it on exit.

        Deletes the snapshot's edit-memory files from the instrument afterwards.
        """
        state = self.snapshot()
        try:
            yield self
        finally:
            try:
                self.restore(state)
            finally:
                for path in state["files"].values():
                    with contextlib.suppress(Exception):
                        self.link.write(f'MMEM:DEL "{path}"')
                        self.link.query("*OPC?")


def _main(argv: list[str] | None = None) -> int:
    """Read-only self-test: identity, options, outputs, reference, error queue.

    Changes no setting. The disconnect at the end turns both outputs off.
    """
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE")
        return 2
    afg = Tektronix_AFG31000(argv[1])
    print(f"connecting to {afg.resource_name} ...")
    idn = afg.connect()
    try:
        print(f"  IDN: {idn}")
        print(f"  model={afg.model}  firmware={afg.firmware}  options={afg.options or '(none)'}")
        print(f"  channels={afg.num_channels}  add_in={afg.has_add_in}")
        for ch in afg._channels():
            wave = ch.get_waveform()
            print(
                f"  {ch.name}: {wave.function} {wave.frequency:.12g} Hz {wave.amplitude:g} Vpp "
                f"offset {wave.offset:g} V, output {'ON' if wave.output_state else 'OFF'}"
            )
        print(f"  reference: {afg.get_reference_clock()}")
        errors = afg.get_errors()
        print(f"  error queue: {errors or 'clean'}")
    finally:
        afg.disconnect()
        print("disconnected (outputs off).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
