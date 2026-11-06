# SPDX-License-Identifier: Apache-2.0
# Copyright: see the NOTICE file.
"""Driver for the Siglent SDG3000X family of two-channel waveform generators.

Public import path::

    from drivers.siglent_sdg3000x import Siglent_SDG3000X

One family class covers the SDG3082X, SDG3162X and SDG3202X. Every model has
two channels, so the constructor builds the whole node tree, with no I/O.
``connect()`` reads ``*IDN?`` and refuses anything else. No address is
built in: read it off the instrument and pass it, VXI-11 or the raw socket.

Common tasks:
  1. A 1 kHz, 1 Vpp sine on channel 1 into 50 ohm, amplitude capped, then on:
         sdg.ch1.set_waveform_sine(1e3, 1.0, ampl_limit=4.0, load=50)
         sdg.ch1.set_output_state(True)
  2. Step the frequency without touching anything else:
         sdg.ch1.set_frequency(2e3)
  3. Sweep 100 Hz to 100 kHz logarithmically over 3 ms:
         sdg.ch1.set_sweep(100.0, 100e3, 3e-3, spacing=sdg.Spacing.LOG)
  4. FM with 500 Hz deviation at a 100 Hz rate:
         sdg.ch1.set_fm(500.0, rate=100.0)
  5. Five-cycle bursts every 10 ms:
         sdg.ch1.set_burst(5, period=10e-3)
  6. CH2 90 degrees behind CH1, both outputs on:
         sdg.multichannel.set_quadrature(1e3, 1.0, ampl_limit=4.0)
         sdg.set_output_state_all(True)

Provenance. This driver is a port of the SDG3000X driver in a private
repository (the original repository), rebuilt on ``common`` and renamed to the
driver standard. Every bench fact below carries the quirk ID it had there
(SDG-nn). The records are in ``docs/quirks/sdg3000x.md`` and ``.jsonl``. They
were observed on an SDG3202X, firmware 1.1.1.7R10 and R12, under the old
method names. The port kept the SCPI each method sends. A rerun under these
names on 2026-11-05, SDG3202X firmware 1.1.1.7R12: every ``get_`` with no
required argument, the write-back of each ``set_`` that has a matching ``get_``
(tools/api_sweep.py), and signal checks on a scope of the basic shapes,
modulation, chirp, burst, hopping, multitone, multipulse, sequence, IQ, phase
and coupling. Absolute output level was not checked in that rerun. Sources cited as: PG = programming guide EN01A, UM = user
manual EN01A, DS = data sheet EN01C, MEAS = a bench measurement in the original
repository.

Read these before you energize anything.

Every setting write takes about 110 ms to execute, VXI-11 and socket alike,
and queued writes run no faster (SDG-23). SCPI suits setup, not streaming
control. Each write here is followed by an operation-complete query and an
error-queue check, as the original driver did after every write.

Out-of-range values are clamped without an error (SDG-01). Every ``set_``
returns the value the instrument holds; an atomic setter warns with
:class:`ValueSnappedWarning` when that differs, and a helper raises
:class:`InstrumentError` instead.

The only output clamp is an amplitude limit in Vpp, held in high-impedance
volts: a 0.6 Vpp limit with the load set to 50 ohm caps the amplitude at
0.3 Vpp (SDG-34). Every waveform helper that has an amplitude requires
``ampl_limit`` in those units and writes it before the levels. Whether the
limit also bounds the offset or a DC level was never observed  # UNVERIFIED

No helper turns an output on. ``set_output_state`` and ``set_output_state_all``
do. ``disconnect()`` turns both outputs off (:meth:`Siglent_SDG3000X.safe_state`).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import math
import re
import tarfile
import time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import TypeVar, cast

import numpy as np
import pyvisa as visa

from common.errors import (
    InstrumentError,
    InstrumentQueryError,
    InstrumentTimeoutError,
    InstrumentUnresponsiveError,
    ValueSnappedWarning,
)
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import ScpiEnum, member_from_response, quote
from common.state import BurstState, ModulationState, SweepState, WaveformState

# The original driver opened every resource with a 3 s timeout.
_DEFAULT_TIMEOUT_MS = 3000

# model -> capabilities. Channels and top frequency from DS p.3.
_MODEL_INFO: dict[str, dict] = {
    "SDG3082X": {"channels": 2, "max_frequency": 80e6},
    "SDG3162X": {"channels": 2, "max_frequency": 160e6},
    "SDG3202X": {"channels": 2, "max_frequency": 200e6},
}

_IDN_RE = re.compile(r"\b(SDG3\d{3}X)\b", re.IGNORECASE)

# A helper raises when a read-back differs from the request by more than this
# fraction. The original driver's own check, not an instrument precision.
_TOLERANCE = 1e-3

# Bounds on a hung instrument, not settle times. The original driver waited this
# long for *RST and a setup recall, and 60 s after an arb upload.
_RESET_TIMEOUT_S = 30.0
_UPLOAD_TIMEOUT_S = 60.0
_SEQUENCE_RUN_TIMEOUT_MS = 60_000
_ERROR_QUEUE_DRAIN = 50

# The counter makes about 0.6 measurements per second, and remote queries
# stall it: 10 queries per second gave none in 20 s (SDG-28). Poll slowly.
_COUNTER_POLL_S = 2.0
_COUNTER_TIMEOUT_S = 20.0

# Setup files this driver writes to the instrument's Local drive. The default
# setup is one file per clock source, so a reset never switches the clock
# (SDG-25). The original driver used the same names, so its saved files are reused.
_DEFAULT_SETUP = "Local/drv_default_{clock}.xml"
_SNAPSHOT_SETUP = "Local/drv_snapshot_{stamp}.xml"
_FILE_READ_TMP = "drv_files_rd"
_SEQUENCE_TMP = "drv_seq_tmp"

# The arb amplitude scale: the instrument maps the data extremes to the
# amplitude, not the code range (SDG-24).
_ARB_CODE_SPAN = 65534
_ARB_CODE_LOW = -32767

# Sequence segment rules (DS p.11).
_SEGMENT_MAX = 1024
_SEGMENT_MIN_POINTS = 64
_SEGMENT_STEP_BELOW_256 = 16

# Reply units the instrument appends to numbers. Anything else after a number
# is a token: "16QAM" stays text.
_REPLY_UNITS = {"", "hz", "v", "vpp", "vrms", "s", "dbm", "dbc", "db", "bps", "%", "min"}
_NUMBER_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-z%]*)$")

# Waveform-page reply keys, for messages.
_BSWV_NAMES = {
    "WVTP": "shape",
    "FRQ": "frequency",
    "PERI": "period",
    "AMP": "amplitude",
    "AMPVRMS": "amplitude in Vrms",
    "AMPDBM": "amplitude in dBm",
    "OFST": "offset",
    "HLEV": "high level",
    "LLEV": "low level",
    "PHSE": "phase",
    "DUTY": "duty cycle",
    "SYM": "symmetry",
    "WIDTH": "width",
    "RISE": "rise time",
    "FALL": "fall time",
    "DLY": "delay",
    "STDEV": "noise standard deviation",
    "MEAN": "noise mean",
    "MAX_OUTPUT_AMP": "amplitude limit",
    "BANDWIDTH": "noise bandwidth",
}


def _num(value: float) -> str:
    """Format a number for the instrument at 12 significant digits."""
    return f"{value:.12g}"


def _close(requested: float, applied: object) -> bool:
    """Whether ``applied`` is a number within the helper tolerance of ``requested``."""
    return isinstance(applied, float) and abs(applied - requested) <= max(
        _TOLERANCE * abs(requested), 1e-9
    )


def _value(token: str) -> float | str:
    """``'1000HZ'`` -> 1000.0, ``'-0S'`` -> 0.0, ``'SINE'`` -> ``'SINE'``."""
    text = token.strip().strip('"')
    match = _NUMBER_RE.match(text)
    if match and match.group(2).lower() in _REPLY_UNITS:
        number = float(match.group(1))
        return 0.0 if number == 0 else number
    return text


def _pairs(tokens: list[str], context: str = "") -> dict:
    """``['A', '1', 'B', 'x']`` -> ``{'A': 1.0, 'B': 'x'}``. A CARR key opens a nested dict."""
    out: dict = {}
    target = out
    i = 0
    while i < len(tokens):
        key = tokens[i].strip()
        if key == "CARR":
            target = out.setdefault("CARR", {})
            i += 1
            continue
        if i + 1 >= len(tokens):
            raise InstrumentQueryError(f"key {key!r} has no value in the reply {context!r}")
        target[key] = _value(tokens[i + 1])
        i += 2
    return out


def _parse_indexed(reply: str) -> list[list[float]]:
    """``'1,1000.0;2,2000.0'`` -> ``[[1000.0], [2000.0]]``, index dropped (MEAS)."""
    out = []
    for item in reply.strip().split(";"):
        if item.strip():
            out.append([float(x) for x in item.split(",")[1:]])
    return out


def _on_off(state: bool) -> str:
    return "ON" if state else "OFF"


# ---------------------------------------------------------------------------
# Enums. Every value is the token the instrument takes.
# ---------------------------------------------------------------------------


class Function(ScpiEnum):
    """The carrier shape: the Waveforms key on the front panel."""

    SINE = "SINE"
    SQUARE = "SQUARE"
    RAMP = "RAMP"
    PULSE = "PULSE"
    NOISE = "NOISE"
    DC = "DC"
    ARB = "ARB"
    PRBS = "PRBS"


class Load(ScpiEnum):
    """What the far end of the cable terminates into, when it is not 50 ohm."""

    HIGHZ = "HZ"


class ChannelMode(ScpiEnum):
    """What a channel is doing: the basic generator, a sequence, IQ, multitone or multipulse."""

    AFG = "AFG"
    SEQUENCE = "AWG"
    IQ = "IQ"
    MULTITONE = "MULTITONE"
    MULTIPULSE = "MULTIPULSE"


class ModulationType(ScpiEnum):
    """The modulation kind on the Mod page."""

    AM = "AM"
    DSBSC = "DSBSC"
    FM = "FM"
    PM = "PM"
    PWM = "PWM"
    ASK = "ASK"
    FSK = "FSK"
    PSK = "PSK"


class ModulationSource(ScpiEnum):
    """Where the modulating signal comes from. EXTERNAL is the rear MODULATION IN."""

    INTERNAL = "INT"
    EXTERNAL = "EXT"
    CH1 = "CH1"
    CH2 = "CH2"


class ModulationShape(ScpiEnum):
    """The internal modulating waveform."""

    SINE = "SINE"
    SQUARE = "SQUARE"
    TRIANGLE = "TRIANGLE"
    UP_RAMP = "UPRAMP"
    DOWN_RAMP = "DNRAMP"
    NOISE = "NOISE"
    ARB = "ARB"


class Polarity(ScpiEnum):
    """PSK polarity, or which gate level runs a gated burst."""

    POSITIVE = "POS"
    NEGATIVE = "NEG"


class TriggerSource(ScpiEnum):
    """What starts a sweep or a burst. MANUAL waits for ``cmd_trigger()``."""

    INTERNAL = "INT"
    EXTERNAL = "EXT"
    MANUAL = "MAN"


class TriggerOut(ScpiEnum):
    """The edge Trig/Sync OUT gives at each sweep or burst start, or none."""

    OFF = "OFF"
    RISE = "RISE"
    FALL = "FALL"


class Edge(ScpiEnum):
    """A rising or falling edge on an input."""

    RISE = "RISE"
    FALL = "FALL"


class Spacing(ScpiEnum):
    """How a frequency sweep steps from start to stop."""

    LINEAR = "LINE"
    LOG = "LOG"
    STEP = "STEP"


class AmplitudeSpacing(ScpiEnum):
    """How an amplitude sweep steps from start to stop."""

    LINEAR = "LINE"
    STEP = "STEP"


class SweepType(ScpiEnum):
    """What a sweep changes: frequency, amplitude, or both together (UM p.73)."""

    FREQUENCY = "FREQ"
    AMPLITUDE = "AMP"
    FREQUENCY_AMPLITUDE = "BOTH"


class Direction(ScpiEnum):
    """Which way a sweep runs."""

    UP = "UP"
    DOWN = "DOWN"
    UP_DOWN = "UP_DOWN"


class BurstMode(ScpiEnum):
    """N cycles per trigger, or output while a gate is open."""

    NCYCLE = "NCYC"
    GATED = "GATE"


class BurstIdle(ScpiEnum):
    """The level between bursts: the carrier at the start phase, or at the burst end (SDG-13)."""

    START = "START"
    END = "END"


class LogicLevel(ScpiEnum):
    """PRBS logic family. Any choice but CUSTOM sets the levels itself."""

    TTL_CMOS = "TTL_CMOS"
    LVTTL_LVCMOS = "LVTTL_LVCMOS"
    ECL = "ECL"
    LVPECL = "LVPECL"
    LVDS = "LVDS"
    CUSTOM = "CUSTOM"


class HarmonicType(ScpiEnum):
    """Which harmonic orders may carry a level."""

    EVEN = "EVEN"
    ODD = "ODD"
    ALL = "ALL"


class ChirpRepeat(ScpiEnum):
    """How a chirp on the sweep hardware repeats."""

    CONTINUOUS = "CONTINUOUS"
    PERIOD = "PERIOD"
    ONCE = "ONCE"
    EXTERNAL = "EXTERNAL"


class HopType(ScpiEnum):
    """The frequency-hopping pattern."""

    LIST = "MANUAL"
    RANDOM = "RHOP"
    RANDOM_LIST = "RLIST"


class PatternTriggerSource(ScpiEnum):
    """What starts a multipulse train or an IQ pattern."""

    INTERNAL = "INT"
    EXTERNAL = "EXT"
    MANUAL = "MAN"
    TIMER = "TIMER"


class RunMode(ScpiEnum):
    """How a sequence plays (MEAS 2026-10-15). STEP and ADVANCED need the file route (SDG-06)."""

    CONTINUOUS = "CONT"
    TRIGGERED = "TCON"
    BURST = "BURS"
    STEP = "STEP"
    ADVANCED = "ADV"


class SequenceTriggerSource(ScpiEnum):
    """What triggers a sequence. FREE_RUN is read back only."""

    MANUAL = "MAN"
    TIMER = "TIM"
    EXTERNAL = "EXT"
    FREE_RUN = "FRUN"


class SequenceEdge(ScpiEnum):
    """The external trigger edge of a sequence."""

    RISE = "RISE"
    FALL = "FALL"
    BOTH = "BOTH"


class Interpolation(ScpiEnum):
    """How a sequence interpolates between points."""

    ZERO = "ZERO"
    LINEAR = "LINEAR"
    SINC = "SINC"
    SINC13 = "SINC13"
    SINC27 = "SINC27"


class LengthIncrease(ScpiEnum):
    """How a segment grows when its length increases."""

    INTERPOLATE = "INT"
    ZERO = "ZERO"
    HOLD_LAST = "HLAS"
    DUPLICATE = "DUPL"


class LengthDecrease(ScpiEnum):
    """How a segment shrinks when its length decreases."""

    DECIMATE = "DECI"
    CUT_TAIL = "CTAI"
    CUT_HEAD = "CHEA"


class SequenceHold(ScpiEnum):
    """The level after a sequence ends. USER holds the level set by ``set_hold_user_level``."""

    MID = "MID"
    START = "START"
    END = "END"
    USER = "USER"


class WaitEvent(ScpiEnum):
    """What a segment waits for before it plays, run mode ADVANCED only.

    No TIMER: its file code is unknown and a wrong code crashes the SCPI server
    (SDG-07).
    """

    AUTO = "AUTO"
    MANUAL = "MAN"
    EXTERNAL = "EXT"


class IqWaveform(ScpiEnum):
    """The built-in IQ waveforms (PG p.58)."""

    ASK_2 = "2ASK"
    ASK_4 = "4ASK"
    ASK_8 = "8ASK"
    BPSK = "BPSK"
    QPSK = "QPSK"
    PSK_8 = "8PSK"
    DBPSK = "DBPSK"
    DQPSK = "DQPSK"
    D8PSK = "D8PSK"
    QAM_8 = "8QAM"
    QAM_16 = "16QAM"
    QAM_32 = "32QAM"
    QAM_64 = "64QAM"
    QAM_128 = "128QAM"
    QAM_256 = "256QAM"
    QAM_16_2 = "16QAM_2"


class Reference(ScpiEnum):
    """Which 10 MHz timebase the instrument runs from."""

    INTERNAL = "INT"
    EXTERNAL = "EXT"


class CounterMode(ScpiEnum):
    """What the rear Counter IN measures."""

    FREQUENCY = "FREQUENCY"
    TOTALIZER = "TOTALIZER"


class Coupling(ScpiEnum):
    """Counter input coupling."""

    AC = "AC"
    DC = "DC"


class Gate(ScpiEnum):
    """When the totalizer counts, gated from front TRIG/SYNC IN."""

    OFF = "OFF"
    LEVEL_POSITIVE = "LEVEL_POSITIVE"
    LEVEL_NEGATIVE = "LEVEL_NEGATIVE"
    AFTER_RISE = "AFTER_RISE"
    AFTER_FALL = "AFTER_FALL"


_SWEEP_TRIGGER_OUT = {TriggerOut.OFF: "OFF", TriggerOut.RISE: "UP", TriggerOut.FALL: "DOWN"}
_KEYING = (ModulationType.ASK, ModulationType.FSK, ModulationType.PSK)
_SWEEPABLE = (Function.SINE, Function.SQUARE, Function.RAMP, Function.ARB)


_E = TypeVar("_E", bound=ScpiEnum)
_T = TypeVar("_T")


def _as_float(value: object) -> float:
    """A reply value as a float; raises when the instrument answered text."""
    if isinstance(value, float):
        return value
    try:
        return float(str(value))
    except ValueError as exc:
        raise InstrumentQueryError(f"expected a number, the instrument answered {value!r}") from exc


def _present(value: _T | None, what: str) -> _T:
    """``value``, or raise when a page reads off right after a helper turned it on."""
    if value is None:
        raise InstrumentError(f"the {what} reads off right after it was turned on")
    return value


def _token(value: object, enum_cls: type[ScpiEnum], label: str) -> str:
    """The SCPI token for an enum member, a member name, or a token, any case."""
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    for member in enum_cls:
        if text in (member.name, str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")


def _member(response: object, enum_cls: type[_E], label: str) -> _E:
    """Resolve a reply to a member, allowing the ``_A`` suffix some replies carry (MEAS)."""
    text = str(response).strip().strip('"').upper()
    try:
        return cast(_E, member_from_response(text, enum_cls, label))
    except ValueError:
        if text.endswith("_A"):
            with contextlib.suppress(ValueError):
                return cast(_E, member_from_response(text[:-2], enum_cls, label))
        valid = ", ".join(m.name for m in enum_cls)
        raise InstrumentQueryError(
            f"the instrument answered {response!r} for {label}; expected one of: {valid}"
        ) from None


# ---------------------------------------------------------------------------
# Records this driver returns that common.state does not hold.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseState:
    """The Gaussian noise output of one channel."""

    stdev: float
    """Standard deviation in volts."""

    mean: float
    """Mean level in volts."""

    bandwidth: float | None
    """Bandwidth limit in Hz; None at full bandwidth."""

    output_state: bool
    """True when the output connector is driven."""


@dataclass(frozen=True)
class PrbsState:
    """The PRBS output of one channel."""

    bit_rate: float
    """Bits per second."""

    length: int
    """The PRBS order: the pattern repeats every 2**length - 1 bits."""

    amplitude: float
    """Volts peak to peak."""

    offset: float
    """Volts."""

    level_high: float
    """Volts."""

    level_low: float
    """Volts."""

    edge_time: float
    """Edge time in seconds."""

    logic_level: LogicLevel
    """The logic family; CUSTOM uses the amplitude and offset."""

    differential: bool
    """True when channel 2 carries the inverted pattern."""

    polynomial: str
    """The generator polynomial as the instrument reports it."""

    output_state: bool
    """True when the output connector is driven."""


@dataclass(frozen=True)
class SweepAmplitudeState:
    """The amplitude half of a sweep."""

    sweep_type: SweepType
    """AMPLITUDE alone, or FREQUENCY_AMPLITUDE together."""

    start: float
    """Start amplitude in Vpp."""

    stop: float
    """Stop amplitude in Vpp."""

    time: float
    """Sweep time in seconds."""

    spacing: AmplitudeSpacing
    """LINEAR or STEP."""

    direction: Direction
    """Which way the amplitude runs."""

    steps: int
    """Steps of a STEP amplitude sweep."""

    symmetry: float
    """Percent of an UP_DOWN sweep spent rising."""


@dataclass(frozen=True)
class ModulationInputScale:
    """What one volt on the rear MODULATION IN does for the present modulation (SDG-09, SDG-10)."""

    mode: str | None
    """The modulation token, or None when nothing modulates."""

    per_volt: float | None
    """Effect per volt in ``unit``; None for a keying mode."""

    unit: str | None
    """Unit of ``per_volt``."""

    full_scale_volts: float | None
    """Input volts that give the full set depth or deviation."""

    note: str = ""
    """The keying input rule, for ASK, FSK and PSK."""


@dataclass(frozen=True)
class HarmonicLevel:
    """One harmonic added to a sine carrier."""

    order: int
    """Harmonic order, 2 to 20."""

    dbc: float
    """Level relative to the carrier in dBc. -80 means none."""

    vpp: float
    """Level in volts peak to peak."""

    phase: float
    """Phase in degrees."""


@dataclass(frozen=True)
class HarmonicsState:
    """The harmonics page of one channel."""

    enabled: bool
    """Whether harmonics are added."""

    harmonic_type: HarmonicType | None
    """Which orders may carry a level; None while off."""

    levels: tuple[HarmonicLevel, ...] = ()
    """The orders read, in the order asked for."""


@dataclass(frozen=True)
class ChirpState:
    """A chirp, as the setting helper left it."""

    method: str
    """``sweep`` for the sweep hardware, ``arb`` for an uploaded waveform."""

    start: float
    """Start frequency in Hz."""

    stop: float
    """Stop frequency in Hz."""

    duration: float
    """Seconds per chirp."""

    spacing: Spacing
    """LINEAR or LOG."""

    repeat: ChirpRepeat | None
    """How it repeats, on the sweep hardware; None for an arb chirp."""

    period: float | None
    """Seconds from one chirp start to the next, for ChirpRepeat.PERIOD."""

    points: int | None
    """Samples in an arb chirp."""

    output_state: bool
    """True when the output connector is driven."""


@dataclass(frozen=True)
class HoppingState:
    """The frequency-hopping page of one channel."""

    enabled: bool
    """Whether hopping runs."""

    hop_type: HopType
    """LIST, RANDOM or RANDOM_LIST."""

    dwell: float
    """Seconds at each frequency."""

    start: float | None = None
    """Random-grid start in Hz; RANDOM only."""

    stop: float | None = None
    """Random-grid stop in Hz; RANDOM only."""

    step: float | None = None
    """Random-grid step in Hz; RANDOM only."""

    prbs_order: int | None = None
    """Order of the pseudo-random sequence; RANDOM and RANDOM_LIST."""

    avoid_enabled: bool | None = None
    """Whether the avoid ranges apply; RANDOM only."""


@dataclass(frozen=True)
class MultitoneState:
    """The multitone page of one channel."""

    enabled: bool
    """Whether multitone is on."""

    running: bool
    """Whether it plays."""

    tones: tuple[float, ...]
    """Tone frequencies in Hz."""

    notches: tuple[tuple[float, float], ...]
    """Notch ranges (low, high) in Hz."""

    amplitude: float
    """Total amplitude in Vpp."""

    sample_rate: float
    """Samples per second, chosen by the instrument (MEAS)."""


@dataclass(frozen=True)
class MultipulseState:
    """The multi-pulse page of one channel."""

    enabled: bool
    """Whether multipulse is on."""

    running: bool
    """Whether it plays."""

    pulses: tuple[tuple[float, float], ...]
    """Each pulse as (width, gap) in seconds."""

    level_high: float
    """Volts."""

    level_low: float
    """Volts."""

    trigger_source: PatternTriggerSource
    """What starts a train."""

    trigger_timer: float
    """Timer period in seconds."""

    trigger_delay: float
    """Delay from the trigger in seconds."""

    sample_rate: float | None
    """Samples per second; None when the instrument chooses."""


@dataclass(frozen=True)
class Segment:
    """One segment of a sequence: one row of the front-panel segment table (UM p.53-54).

    ``waveform`` is a built-in or user arb name. ``next`` is the segment number,
    from 1, played after this one; None means the following segment, and the
    last wraps to 1. ``wait`` applies in RunMode.ADVANCED only.
    """

    waveform: str
    """Built-in or user arb name."""

    length: int
    """Points; at least 64, and a multiple of 16 below 256 (DS p.11)."""

    loops: int = 1
    """Times the segment plays."""

    amplitude: float | None = None
    """Vpp, or None to keep the waveform's own."""

    offset: float | None = None
    """Volts, or None to keep the waveform's own."""

    next: int | None = None
    """The segment number played next, from 1."""

    wait: WaitEvent = WaitEvent.AUTO
    """What the segment waits for."""


@dataclass(frozen=True)
class SequenceState:
    """The sequence page of one channel."""

    running: bool
    """Whether the sequence plays."""

    segments: tuple[Segment, ...] | None
    """The segment table; None while running, when the count query hangs (SDG-05)."""

    sample_rate: float
    """Samples per second."""

    run_mode: RunMode
    """How it plays."""

    trigger_source: SequenceTriggerSource
    """What triggers it."""

    interpolation: Interpolation
    """Interpolation between points."""

    offset: float
    """Volts."""

    hold: SequenceHold
    """The level after the sequence ends."""


@dataclass(frozen=True)
class CouplingState:
    """How CH2 follows CH1, per quantity: by a ratio, by a deviation, or not at all."""

    frequency_ratio: float | None = None
    frequency_deviation: float | None = None
    """Hz."""

    amplitude_ratio: float | None = None
    amplitude_deviation: float | None = None
    """Vpp."""

    phase_ratio: float | None = None
    phase_deviation: float | None = None
    """Degrees."""


@dataclass(frozen=True)
class ChannelPairState:
    """Both channels after a two-channel task."""

    phase_locked: bool
    """Whether the channels are phase locked."""

    ch2_phase: float
    """CH2 phase relative to CH1, degrees."""

    ch1: WaveformState
    """CH1 waveform."""

    ch2: WaveformState
    """CH2 waveform."""


@dataclass(frozen=True)
class IqState:
    """The IQ page."""

    enabled: bool
    """Whether the channels are in IQ mode."""

    source: str
    """``builtin`` or ``userstored``."""

    waveform: str
    """The waveform name the instrument reports."""

    carrier_frequency: float
    """Hz; 0 gives baseband I and Q."""

    symbol_rate: float
    """Symbols per second."""

    sample_rate: float
    """Samples per second."""

    vrms: float
    """IQ amplitude in Vrms, into 50 ohm at any load setting (SDG-22)."""

    trigger_source: PatternTriggerSource
    """What starts the pattern."""

    gain: float
    """I/Q gain balance in dB."""

    offset_i: float
    """I offset in volts."""

    offset_q: float
    """Q offset in volts."""

    angle_q: float
    """Q angle in degrees."""


@dataclass(frozen=True)
class IqWaveformInfo:
    """What the instrument reports about the loaded IQ waveform."""

    symbol_length: float | str | None
    oversampling: float | str | None
    modulation: float | str | None
    filter_type: float | str | None
    filter_alpha: float | str | None


@dataclass(frozen=True)
class SampleRateLimits:
    """The IQ sample-rate range."""

    minimum: float
    """Samples per second."""

    maximum: float
    """Samples per second."""


@dataclass(frozen=True)
class FileEntry:
    """One entry of a folder on the instrument's drive."""

    name: str
    is_directory: bool
    size: int
    """Bytes; 0 for a folder."""

    file_type: str
    """The type the instrument lists, lower case; empty for a folder."""


@dataclass(frozen=True)
class FileCatalog:
    """One folder listing, with the drive's space."""

    free: int
    """Free bytes."""

    used: int
    """Used bytes."""

    entries: tuple[FileEntry, ...]
    """Files and folders."""


@dataclass(frozen=True)
class CounterState:
    """The counter page."""

    enabled: bool
    mode: CounterMode
    running: bool
    coupling: Coupling
    hf_reject: bool
    trigger_level: float
    """Volts."""

    reference: float | None = None
    """Reference frequency in Hz; FREQUENCY mode only."""

    edge: Edge | None = None
    """Counted edge; TOTALIZER mode only."""

    gate: Gate | None = None
    """Totalizer gate; TOTALIZER mode only."""


@dataclass(frozen=True)
class CounterStatistics:
    """Frequency statistics since the last clear."""

    mean: float
    """Hz."""

    maximum: float
    """Hz."""

    minimum: float
    """Hz."""

    sdev: float
    """Hz."""

    count: int
    """Measurements in the statistics."""

    duty_mean: float
    """Percent."""

    deviation_mean: float
    """ppm against the reference."""


# ---------------------------------------------------------------------------
# Sequence files (.awgx). MEAS 2026-10-15, firmware 1.1.1.7R10 and R12: gzip of
# a ustar tar holding <stem>/, <stem>/test.xml, <stem>/xml_check_file.txt (a
# 16-character check code of test.xml) and optional wave files. A load checks
# the code: a mismatch loads nothing and reports no error.
#
# SDG-07 (limit): an out-of-range code loads without error, then the query that
# formats it kills the SCPI server and freezes the front panel until a power
# cycle. Every code written here comes from a file the instrument saved
# (_AWGX_CODES), and _awgx_check_codes refuses anything else.
# ---------------------------------------------------------------------------

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_CONFUSABLE = {"0": "2", "1": "3", "l": "m", "o": "p"}

# Codes the instrument itself wrote into saved files. Crash codes, never
# written: trigger_source 2, wait_event 2 and 4.
_AWGX_CODES = {
    "run_mode": {"continuous": 0, "triggered": 1, "burst": 3, "step": 4, "advanced": 5},
    "trigger_source": {"free_run": 0, "manual": 1, "timer": 3, "external": 4},
    "trigger_edgeA": {"rise": 0, "fall": 1, "both": 2},
    "wait_event": {"auto": 0, "manual": 1, "external": 3},
}

# Folder of user arbs in the instrument's own saves (MEAS).
_AWGX_USER_FOLDER = "/usr/bin/siglent/usr/usr"


def _awgx_check_code(xml: bytes) -> str:
    """The check code of test.xml: MD5, one character per byte (fits 35 of 35 saved files)."""
    out = []
    for b in hashlib.md5(xml).digest():
        ch = chr(b) if chr(b) in _B36 else _B36[b % 36]
        out.append(_CONFUSABLE.get(ch, ch))
    return "".join(out)


def _awgx_unpack(blob: bytes) -> dict[str, bytes]:
    archive = tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob)))
    return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def _awgx_test_xml(blob: bytes) -> str:
    return next(v for k, v in _awgx_unpack(blob).items() if k.endswith("test.xml")).decode()


def _awgx_pack(stem: str, xml: bytes) -> bytes:
    bio = io.BytesIO()
    now = int(time.time())
    with tarfile.open(fileobj=bio, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        folder = tarfile.TarInfo(stem)
        folder.type, folder.mode, folder.mtime = tarfile.DIRTYPE, 0o777, now
        folder.uname, folder.gname = "root", "root"
        archive.addfile(folder)
        for name, data in (
            ("test.xml", xml),
            ("xml_check_file.txt", _awgx_check_code(xml).encode()),
        ):
            info = tarfile.TarInfo(f"{stem}/{name}")
            info.size, info.mode, info.mtime = len(data), 0o777, now
            info.uname, info.gname = "root", "root"
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(bio.getvalue())


def _awgx_segment_xml(template: str, i: int) -> str:
    """The body of <segment_i> in a saved test.xml."""
    match = re.search(rf"<segment_{i}>(.*?)</segment_{i}>", template, re.S)
    if not match:
        raise ValueError(f"no segment_{i} in the template")
    return match.group(1)


def _awgx_set_tag(body: str, tag: str, value: object) -> str:
    new, n = re.subn(rf"<{tag}>[^<]*</{tag}>", f"<{tag}>{value}</{tag}>", body, count=1)
    if n != 1:
        raise ValueError(f"tag <{tag}> not in the template")
    return new


def _awgx_build(template: str, rows: list[dict], *, run_mode: str | None = None) -> str:
    """Replace the segment list of a saved test.xml.

    Each row: body (a <segment_0> body from a save with that waveform and
    length), loops, wait (a code key), next (0-based), amplitude and offset (V,
    or None to keep the body's). ``run_mode``: a code key; None keeps the
    template's.
    """
    parts = []
    for i, row in enumerate(rows):
        body = row["body"]
        body = _awgx_set_tag(body, "loop_count", int(row["loops"]))
        body = _awgx_set_tag(body, "wait_event", _AWGX_CODES["wait_event"][row["wait"]])
        body = _awgx_set_tag(body, "goto_numb", int(row["next"]))
        if row.get("amplitude") is not None or row.get("offset") is not None:
            amp = row.get("amplitude")
            off = row.get("offset")
            if amp is None:
                amp = float(re.search(r"<amplitude>([^<]*)", body).group(1))
            if off is None:
                off = float(re.search(r"<offset>([^<]*)", body).group(1))
            for tag, v in (
                ("amplitude", amp),
                ("offset", off),
                ("high_level", off + amp / 2),
                ("low_level", off - amp / 2),
            ):
                body = _awgx_set_tag(body, tag, f"{float(v):g}")
        parts.append(f"<segment_{i}>{body}</segment_{i}>")
    joined = "\n            ".join(parts)
    xml, n = re.subn(
        r"<segment_count>\d+</segment_count>\s*<segment_0>.*</segment_\d+>",
        f"<segment_count>{len(rows)}</segment_count>\n            {joined}",
        template,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise ValueError("template has no segment list")
    xml = _awgx_set_tag(xml, "segment_end_numb", len(rows) - 1)
    if run_mode is not None:
        xml = _awgx_set_tag(xml, "run_mode", _AWGX_CODES["run_mode"][run_mode])
    return xml


def _awgx_channel_setting(xml: str) -> dict[str, str]:
    block = re.search(r"<awg_ch_setting>(.*?)</awg_ch_setting>", xml, re.S).group(1)
    return dict(re.findall(r"<(\w+)>([^<]*)</\1>", block))


def _awgx_check_codes(xml: str, template: str) -> None:
    """Raise ValueError unless every code is a saved one (SDG-07) and the rest match the template."""
    for name, allowed in _AWGX_CODES.items():
        for v in re.findall(rf"<{name}>([^<]*)</{name}>", xml):
            if int(v) not in allowed.values():
                raise ValueError(
                    f"sequence file: {name} {v} is not a code the instrument saved "
                    f"{sorted(allowed.values())} (SDG-07)"
                )
    saved, built = _awgx_channel_setting(template), _awgx_channel_setting(xml)
    for key, value in saved.items():
        if key not in _AWGX_CODES and key != "sample_rate" and value != built.get(key):
            raise ValueError(
                f"sequence file: awg_ch_setting {key} {built.get(key)} differs from the "
                f"saved {value}"
            )


# ---------------------------------------------------------------------------
# The link.
# ---------------------------------------------------------------------------


class _SdgLink(InstrumentLink):
    """The link, with this instrument's write discipline.

    Opens with newline terminators and no device clear, as the original driver did
    on VXI-11 and the raw socket alike. Whether a device clear is safe here was
    never tried  # UNVERIFIED
    """

    def connect(self, timeout_ms: int | None = None) -> None:
        rm = visa.ResourceManager()
        self._instr = self._open_resource(rm, self.resource_name)
        self._instr.write_termination = "\n"
        self._instr.read_termination = "\n"
        self._instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms

    def query(self, message: str, **kwargs) -> str:
        return self.visa.query(message, **kwargs).strip()

    def query_body(self, cmd: str) -> str:
        """Query and strip the echoed header, such as ``C1:BSWV `` or ``C1SWWV `` (MEAS)."""
        reply = self.query(cmd)
        head, sep, body = reply.partition(" ")
        if sep and "," not in head:
            return body.strip()
        return reply

    def opc(self, timeout_s: float | None = None) -> None:
        """Wait until the instrument has finished every command sent so far."""
        if timeout_s is None:
            reply = self.query("*OPC?")
        else:
            with self.temporary_timeout(int(timeout_s * 1000)):
                reply = self.query("*OPC?")
        if reply != "1":
            raise InstrumentQueryError(f"{self.resource_name}: *OPC? answered {reply!r}")

    def drain_errors(self) -> list[str]:
        """Read the error queue dry; return the entries, oldest first."""
        found = []
        for _ in range(_ERROR_QUEUE_DRAIN):
            reply = self.query("SYST:ERR?")
            if re.match(r"^\+?0\s*,", reply) or reply in ("0", "+0"):
                return found
            found.append(reply)
        raise InstrumentQueryError(
            f"{self.resource_name}: the error queue did not empty after "
            f"{_ERROR_QUEUE_DRAIN} reads: {found[-3:]}"
        )

    def raise_errors(self, context: str, allow: tuple[str, ...] = ()) -> None:
        """Raise if the error queue holds anything but the ``allow`` prefixes."""
        found = [e for e in self.drain_errors() if not (allow and e.startswith(allow))]
        if found:
            raise InstrumentQueryError(
                f"{self.resource_name}: after {context!r}: {'; '.join(found)}"
            )

    def write_checked(self, cmd: str, *, allow: tuple[str, ...] = ()) -> None:
        """Write, wait for completion, and raise on a queued error (MEAS: after every write)."""
        self.write(cmd)
        self.opc()
        self.raise_errors(cmd, allow)

    def write_raw(self, data: bytes) -> int:
        return self.visa.write_raw(data)


class _SdgNode(Node):
    """A node on this instrument's link, so the checked write is in reach."""

    link: _SdgLink


# ---------------------------------------------------------------------------
# Channel sub-pages that switch on and off: modulation, sweep, burst.
# ---------------------------------------------------------------------------


class _StateNode(_SdgNode):
    """Shared on/off for modulation, sweep and burst (state first: PG p.36, p.40, p.49)."""

    _cmd = ""
    _attr = ""

    def __init__(self, ch: _Channel):
        super().__init__(ch.link, name=f"{ch.name} {self._attr}")
        self._ch = ch
        self._c = ch._c

    def _raw(self) -> dict:
        return _pairs(self.link.query_body(f"{self._c}:{self._cmd}?").split(","))

    def get_state(self) -> bool:
        """Whether this page is on."""
        return self._raw().get("STATE") == "ON"

    def set_state(self, state: bool) -> bool:
        """Turn this page on or off; returns the read-back.

        Turning it on first returns the channel to the basic generator if it was
        in another mode. Raises when the instrument refuses, which it does
        silently for a carrier that cannot carry it (MEAS).
        """
        if state and self._ch.get_mode() is not ChannelMode.AFG:
            self._ch._plain()
        self.link.write_checked(f"{self._c}:{self._cmd} STATE,{_on_off(state)}")
        if self.get_state() != state:
            raise InstrumentError(
                f"{self.name} did not turn {_on_off(state).lower()}. Check the carrier "
                "(sweep: sine, square, ramp or arb, PG p.41; PWM: pulse)."
            )
        return state

    def _need_on(self) -> None:
        if not self.get_state():
            raise InstrumentError(
                f"{self.name} is off. Call ch{self._ch.n}.{self._attr}.set_state(True) first, "
                "or a set_ helper on the channel."
            )

    def _set_key(self, key: str, value: str, prefix: str = "") -> float | str:
        self._need_on()
        self.link.write_checked(f"{self._c}:{self._cmd} {prefix}{key},{value}")
        found = self._raw()
        if key not in found:
            raise InstrumentError(f"{self.name}: {key} does not apply now")
        return found[key]

    def _get_key(self, key: str) -> float | str:
        found = self._raw()
        if key not in found:
            raise InstrumentError(
                f"{self.name}: {key} does not apply now (reply keys: {sorted(found)})"
            )
        return found[key]

    def _others_off(self) -> None:
        for part in (self._ch.mod, self._ch.sweep, self._ch.burst):
            if part is not self and part.get_state():
                part.set_state(False)


class _Modulation(_StateNode):
    """The Mod page: AM, DSB-SC, FM, PM, PWM, ASK, FSK, PSK (PG p.36-40).

    The channel's ``set_am`` and its siblings set a whole modulation in one
    call; these setters change one value of the modulation already on.
    """

    _cmd = "MDWV"
    _attr = "mod"

    def _raw(self) -> dict:
        tokens = self.link.query_body(f"{self._c}:MDWV?").split(",")
        if len(tokens) < 2 or tokens[0] != "STATE":
            raise InstrumentQueryError(f"{self.name}: modulation reply starts {tokens[:3]}")
        if tokens[1] == "OFF":
            return {"STATE": "OFF"}
        found: dict = {"STATE": "ON", "TYPE": tokens[2]}
        found.update(_pairs(tokens[3:]))
        return found

    def _key(self, key: str, value: str) -> float | str:
        self._need_on()
        kind = self.get_type()
        assert kind is not None  # _need_on refused a modulation that is off
        return self._set_key(key, value, prefix=f"{kind.value},")

    def set_type(self, mode: ModulationType | str) -> ModulationType:
        """Choose the modulation kind of the modulation already on; keeps the source.

        Enum arg: <sdg>.ModulationType.<TAB>
        """
        token = _token(mode, ModulationType, "modulation type")
        self._need_on()
        source = self._raw().get("SRC", "INT")
        if token in ("ASK", "FSK", "PSK") and source not in ("INT", "EXT"):
            source = "INT"  # PG p.38: keying sources are internal or external only
        self.link.write_checked(f"{self._c}:MDWV {token},SRC,{source}")
        got = self.get_type()
        if got is None or got.value != token:
            raise InstrumentError(
                f"{self.name}: type reads {got} after {token} (PWM needs a pulse carrier, UM p.70)"
            )
        return got

    def get_type(self) -> ModulationType | None:
        """The modulation kind; None while modulation is off."""
        raw = self._raw()
        return _member(raw["TYPE"], ModulationType, "modulation type") if "TYPE" in raw else None

    def set_source(self, source: ModulationSource | str) -> ModulationSource:
        """Choose where the modulating signal comes from.

        Enum arg: <sdg>.ModulationSource.<TAB>

        ASK, FSK and PSK take INTERNAL or EXTERNAL only, and report it as a
        trigger source (MEAS). With CH1 or CH2 the other channel's waveform
        modulates and the rate setting is ignored (SDG-09).
        """
        token = _token(source, ModulationSource, "modulation source")
        kind = self.get_type()
        if kind in _KEYING:
            if token not in ("INT", "EXT"):
                raise ValueError(f"{kind.value} source is INTERNAL or EXTERNAL (PG p.38)")
            self.link.write_checked(f"{self._c}:MDWV {kind.value},SRC,{token}")
            return self.get_source()
        return _member(self._key("SRC", token), ModulationSource, "modulation source")

    def get_source(self) -> ModulationSource:
        """Where the modulating signal comes from."""
        raw = self._raw()
        value = raw.get("SRC", raw.get("TRSR"))
        if value is None:
            raise InstrumentError(f"{self.name}: no source in the reply (modulation off?)")
        return _member(value, ModulationSource, "modulation source")

    def set_shape(self, shape: ModulationShape | str) -> ModulationShape:
        """Choose the internal modulating waveform.

        Enum arg: <sdg>.ModulationShape.<TAB>
        """
        token = _token(shape, ModulationShape, "modulation shape")
        return _member(self._key("MDSP", token), ModulationShape, "modulation shape")

    def get_shape(self) -> ModulationShape:
        """The internal modulating waveform."""
        return _member(self._get_key("MDSP"), ModulationShape, "modulation shape")

    def set_rate(self, hz: float) -> float:
        """Set the modulating frequency in Hz (AM, DSB-SC, FM, PM, PWM; internal source)."""
        return _as_float(self._key("FRQ", _num(hz)))

    def get_rate(self) -> float:
        """The modulating frequency in Hz."""
        return _as_float(self._get_key("FRQ"))

    def set_hop_frequency(self, hz: float) -> float:
        """Set the FSK hop frequency in Hz: the frequency for a 1 (UM p.72)."""
        return _as_float(self._key("HFRQ", _num(hz)))

    def get_hop_frequency(self) -> float:
        """The FSK hop frequency in Hz."""
        return _as_float(self._get_key("HFRQ"))

    def set_key_rate(self, hz: float) -> float:
        """Set the ASK, FSK or PSK key rate in Hz."""
        return _as_float(self._key("KFRQ", _num(hz)))

    def get_key_rate(self) -> float:
        """The ASK, FSK or PSK key rate in Hz."""
        return _as_float(self._get_key("KFRQ"))

    def set_depth(self, percent: float) -> float:
        """Set the AM depth in percent, 0 to 120."""
        return _as_float(self._key("DEPTH", _num(percent)))

    def get_depth(self) -> float:
        """The AM depth in percent."""
        return _as_float(self._get_key("DEPTH"))

    def set_deviation(self, value: float) -> float:
        """Set the deviation: Hz for FM, degrees for PM, seconds for PWM."""
        return _as_float(self._key("DEVI", _num(value)))

    def get_deviation(self) -> float:
        """The deviation: Hz for FM, degrees for PM, seconds for PWM."""
        return _as_float(self._get_key("DEVI"))

    def set_polarity(self, polarity: Polarity | str) -> Polarity:
        """Set the PSK polarity.

        Enum arg: <sdg>.Polarity.<TAB>
        """
        token = _token(polarity, Polarity, "polarity")
        return _member(self._key("PLRT", token), Polarity, "polarity")

    def get_polarity(self) -> Polarity:
        """The PSK polarity."""
        return _member(self._get_key("PLRT"), Polarity, "polarity")


class _Sweep(_StateNode):
    """The Sweep page (PG p.40-48, UM p.73-77).

    The channel's ``set_sweep`` helpers set a whole sweep in one call; these
    setters change one value of the sweep already on.
    """

    _cmd = "SWWV"
    _attr = "sweep"

    def _need_carrier(self) -> None:
        function = self._ch._get_function()
        if function not in _SWEEPABLE:
            raise InstrumentError(
                f"{self._ch.name}: a sweep needs a sine, square, ramp or arb carrier "
                f"(PG p.41); the carrier is {function.name}"
            )

    def _sw(self, key: str, value: str | None = None) -> str:
        self._need_on()
        if value is not None:
            self.link.write_checked(f"{self._c}:SWEep:{key} {value}")
        return self.link.query(f"{self._c}:SWEep:{key}?").strip().strip('"')

    def _swf(self, key: str, value: float | None = None) -> float:
        return float(self._sw(key, None if value is None else _num(value)))

    def set_type(self, sweep_type: SweepType | str) -> SweepType:
        """Choose what the sweep changes: frequency, amplitude, or both (UM p.73).

        Enum arg: <sdg>.SweepType.<TAB>
        """
        token = _token(sweep_type, SweepType, "sweep type")
        return _member(self._sw("TYPE", token), SweepType, "sweep type")

    def get_type(self) -> SweepType:
        """What the sweep changes."""
        return _member(self._sw("TYPE"), SweepType, "sweep type")

    def set_frequency_start(self, hz: float) -> float:
        """Set the start frequency in Hz. Start clamps to the present stop (SDG-15)."""
        return _as_float(self._set_key("START", _num(hz)))

    def get_frequency_start(self) -> float:
        """The start frequency in Hz."""
        return _as_float(self._get_key("START"))

    def set_frequency_stop(self, hz: float) -> float:
        """Set the stop frequency in Hz."""
        return _as_float(self._set_key("STOP", _num(hz)))

    def get_frequency_stop(self) -> float:
        """The stop frequency in Hz."""
        return _as_float(self._get_key("STOP"))

    def set_frequency_center(self, hz: float) -> float:
        """Set the center frequency in Hz."""
        return _as_float(self._set_key("CENTER", _num(hz)))

    def get_frequency_center(self) -> float:
        """The center frequency in Hz."""
        return _as_float(self._get_key("CENTER"))

    def set_frequency_span(self, hz: float) -> float:
        """Set the frequency span in Hz."""
        return _as_float(self._set_key("SPAN", _num(hz)))

    def get_frequency_span(self) -> float:
        """The frequency span in Hz."""
        return _as_float(self._get_key("SPAN"))

    def set_time(self, seconds: float) -> float:
        """Set the time of one sweep in seconds."""
        return _as_float(self._set_key("TIME", _num(seconds)))

    def get_time(self) -> float:
        """The time of one sweep in seconds."""
        return _as_float(self._get_key("TIME"))

    def set_spacing(self, spacing: Spacing | str) -> Spacing:
        """Choose linear, logarithmic or stepped frequency spacing.

        Enum arg: <sdg>.Spacing.<TAB>
        """
        token = _token(spacing, Spacing, "spacing")
        return _member(self._set_key("SWMD", token), Spacing, "spacing")

    def get_spacing(self) -> Spacing:
        """The frequency spacing."""
        return _member(self._get_key("SWMD"), Spacing, "spacing")

    def set_direction(self, direction: Direction | str) -> Direction:
        """Choose which way the frequency runs.

        Enum arg: <sdg>.Direction.<TAB>
        """
        token = _token(direction, Direction, "direction")
        return _member(self._set_key("DIR", token), Direction, "direction")

    def get_direction(self) -> Direction:
        """Which way the frequency runs."""
        return _member(self._get_key("DIR"), Direction, "direction")

    def set_symmetry(self, percent: float) -> float:
        """Set the percent of an UP_DOWN frequency sweep spent rising."""
        return _as_float(self._set_key("SYM", _num(percent)))

    def get_symmetry(self) -> float:
        """The percent of an UP_DOWN frequency sweep spent rising."""
        return _as_float(self._get_key("SYM"))

    def set_steps(self, n: int) -> int:
        """Set the steps of a STEP frequency sweep."""
        return int(self._set_key("STEPNUM", str(int(n))))

    def get_steps(self) -> int:
        """The steps of a STEP frequency sweep."""
        return int(self._get_key("STEPNUM"))

    def set_trigger_source(self, source: TriggerSource | str) -> TriggerSource:
        """Choose what starts a sweep.

        Enum arg: <sdg>.TriggerSource.<TAB>
        """
        token = _token(source, TriggerSource, "trigger source")
        return _member(self._set_key("TRSR", token), TriggerSource, "trigger source")

    def get_trigger_source(self) -> TriggerSource:
        """What starts a sweep."""
        return _member(self._get_key("TRSR"), TriggerSource, "trigger source")

    def set_trigger_out(self, edge: TriggerOut | str) -> TriggerOut:
        """Choose the Trig/Sync OUT edge at each sweep start, internal or manual source.

        Enum arg: <sdg>.TriggerOut.<TAB>

        The edge marks the start of the ramp; the holds follow the ramp (SDG-16).
        """
        member = _member(_token(edge, TriggerOut, "trigger out"), TriggerOut, "trigger out")
        got = self._set_key("TRMD", _SWEEP_TRIGGER_OUT[member])
        return self._trigger_out_from(got)

    def get_trigger_out(self) -> TriggerOut:
        """The Trig/Sync OUT edge at each sweep start."""
        return self._trigger_out_from(self._get_key("TRMD"))

    @staticmethod
    def _trigger_out_from(token: object) -> TriggerOut:
        reverse = {v: k for k, v in _SWEEP_TRIGGER_OUT.items()}
        text = str(token).strip().upper()
        if text not in reverse:
            raise InstrumentQueryError(f"sweep trigger out: unknown reply {token!r}")
        return reverse[text]

    def set_trigger_edge(self, edge: Edge | str) -> Edge:
        """Choose the Trig/Sync IN edge that starts a sweep, external source.

        Enum arg: <sdg>.Edge.<TAB>
        """
        token = _token(edge, Edge, "edge")
        return _member(self._set_key("EDGE", token), Edge, "edge")

    def get_trigger_edge(self) -> Edge:
        """The Trig/Sync IN edge that starts a sweep."""
        return _member(self._get_key("EDGE"), Edge, "edge")

    def set_hold_start(self, seconds: float) -> float:
        """Set the time held at the start value before each sweep, in seconds (UM p.75).

        Trig/Sync OUT marks the ramp start; the holds follow the ramp (SDG-16).
        """
        return self._swf("SHTime", seconds)

    def get_hold_start(self) -> float:
        """The time held at the start value, in seconds."""
        return self._swf("SHTime")

    def set_hold_end(self, seconds: float) -> float:
        """Set the time held at the stop value after each sweep, in seconds (UM p.75)."""
        self._need_on()
        self.link.write_checked(f"{self._c}:SWEep:EHTime {_num(seconds)}")
        return self.get_hold_end()

    def get_hold_end(self) -> float:
        """The time held at the stop value, in seconds."""
        return float(self.link.query(f"{self._c}:SWEep:EHTime?").strip('"'))

    def set_return_time(self, seconds: float) -> float:
        """Set the return time from the stop value to the start value, in seconds (UM p.75)."""
        return self._swf("RTime", seconds)

    def get_return_time(self) -> float:
        """The return time from the stop value to the start value, in seconds."""
        return self._swf("RTime")

    def set_amplitude_start(self, vpp: float) -> float:
        """Set the amplitude-sweep start in Vpp."""
        return self._swf("SAMPlitude", vpp)

    def get_amplitude_start(self) -> float:
        """The amplitude-sweep start in Vpp."""
        return self._swf("SAMPlitude")

    def set_amplitude_stop(self, vpp: float) -> float:
        """Set the amplitude-sweep stop in Vpp."""
        return self._swf("EAMPlitude", vpp)

    def get_amplitude_stop(self) -> float:
        """The amplitude-sweep stop in Vpp."""
        return self._swf("EAMPlitude")

    def set_amplitude_center(self, vpp: float) -> float:
        """Set the amplitude-sweep center in Vpp."""
        return self._swf("CAMPlitude", vpp)

    def get_amplitude_center(self) -> float:
        """The amplitude-sweep center in Vpp."""
        return self._swf("CAMPlitude")

    def set_amplitude_span(self, vpp: float) -> float:
        """Set the amplitude-sweep span in Vpp."""
        return self._swf("ASPan", vpp)

    def get_amplitude_span(self) -> float:
        """The amplitude-sweep span in Vpp."""
        return self._swf("ASPan")

    def set_amplitude_spacing(self, spacing: AmplitudeSpacing | str) -> AmplitudeSpacing:
        """Choose linear or stepped amplitude spacing.

        Enum arg: <sdg>.AmplitudeSpacing.<TAB>
        """
        token = _token(spacing, AmplitudeSpacing, "amplitude spacing")
        return _member(self._sw("AMODe", token), AmplitudeSpacing, "amplitude spacing")

    def get_amplitude_spacing(self) -> AmplitudeSpacing:
        """The amplitude spacing."""
        return _member(self._sw("AMODe"), AmplitudeSpacing, "amplitude spacing")

    def set_amplitude_steps(self, n: int) -> int:
        """Set the steps of a STEP amplitude sweep, 2 to 1024."""
        return int(float(self._sw("ASNumber", str(int(n)))))

    def get_amplitude_steps(self) -> int:
        """The steps of a STEP amplitude sweep."""
        return int(float(self._sw("ASNumber")))

    def set_amplitude_direction(self, direction: Direction | str) -> Direction:
        """Choose which way the amplitude runs.

        Enum arg: <sdg>.Direction.<TAB>
        """
        token = _token(direction, Direction, "direction")
        return _member(self._sw("ADIRection", token), Direction, "direction")

    def get_amplitude_direction(self) -> Direction:
        """Which way the amplitude runs."""
        return _member(self._sw("ADIRection"), Direction, "direction")

    def set_amplitude_symmetry(self, percent: float) -> float:
        """Set the percent of an UP_DOWN amplitude sweep spent rising."""
        return self._swf("ASYMmetry", percent)

    def get_amplitude_symmetry(self) -> float:
        """The percent of an UP_DOWN amplitude sweep spent rising."""
        return self._swf("ASYMmetry")

    def cmd_trigger(self) -> None:
        """Run one sweep, trigger source MANUAL."""
        self._need_on()
        self.link.write_checked(f"{self._c}:SWWV MTRIG")


class _Burst(_StateNode):
    """The Burst page (PG p.48-51, UM p.78-80).

    The channel's ``set_burst`` helpers set a whole burst in one call; these
    setters change one value of the burst already on.
    """

    _cmd = "BTWV"
    _attr = "burst"

    def set_mode(self, mode: BurstMode | str) -> BurstMode:
        """Choose N cycles per trigger, or output while a gate is open.

        Enum arg: <sdg>.BurstMode.<TAB>
        """
        token = _token(mode, BurstMode, "burst mode")
        return _member(self._set_key("GATE_NCYC", token), BurstMode, "burst mode")

    def get_mode(self) -> BurstMode:
        """N cycles per trigger, or gated."""
        return _member(self._get_key("GATE_NCYC"), BurstMode, "burst mode")

    @staticmethod
    def _cycles_from(value: object) -> float:
        if str(value).strip().upper() == "INF":
            return math.inf
        return _as_float(value)

    def set_cycles(self, cycles: float) -> float:
        """Set the cycles per burst, a whole number or ``math.inf``."""
        token = "INF" if cycles == math.inf else str(int(cycles))
        return self._cycles_from(self._set_key("TIME", token))

    def get_cycles(self) -> float:
        """The cycles per burst; ``math.inf`` for an infinite burst."""
        return self._cycles_from(self._get_key("TIME"))

    def set_period(self, seconds: float) -> float:
        """Set the burst period in seconds, internal trigger."""
        return _as_float(self._set_key("PRD", _num(seconds)))

    def get_period(self) -> float:
        """The burst period in seconds."""
        return _as_float(self._get_key("PRD"))

    def set_phase(self, degrees: float) -> float:
        """Set the start phase of each burst, in degrees."""
        return _as_float(self._set_key("STPS", _num(degrees)))

    def get_phase(self) -> float:
        """The start phase of each burst, in degrees."""
        return _as_float(self._get_key("STPS"))

    def set_trigger_source(self, source: TriggerSource | str) -> TriggerSource:
        """Choose what starts a burst.

        Enum arg: <sdg>.TriggerSource.<TAB>
        """
        token = _token(source, TriggerSource, "trigger source")
        return _member(self._set_key("TRSR", token), TriggerSource, "trigger source")

    def get_trigger_source(self) -> TriggerSource:
        """What starts a burst."""
        return _member(self._get_key("TRSR"), TriggerSource, "trigger source")

    def set_trigger_out(self, edge: TriggerOut | str) -> TriggerOut:
        """Choose the Trig/Sync OUT edge at each burst start, internal or manual source.

        Enum arg: <sdg>.TriggerOut.<TAB>
        """
        token = _token(edge, TriggerOut, "trigger out")
        return _member(self._set_key("TRMD", token), TriggerOut, "trigger out")

    def get_trigger_out(self) -> TriggerOut:
        """The Trig/Sync OUT edge at each burst start."""
        return _member(self._get_key("TRMD"), TriggerOut, "trigger out")

    def set_trigger_edge(self, edge: Edge | str) -> Edge:
        """Choose the Trig/Sync IN edge that starts a burst, external source.

        Enum arg: <sdg>.Edge.<TAB>
        """
        token = _token(edge, Edge, "edge")
        return _member(self._set_key("EDGE", token), Edge, "edge")

    def get_trigger_edge(self) -> Edge:
        """The Trig/Sync IN edge that starts a burst."""
        return _member(self._get_key("EDGE"), Edge, "edge")

    def set_count(self, n: int) -> int:
        """Set the bursts per trigger, ``period`` apart; external or manual trigger only.

        Written as COUNT and read back as COUNTER; with the internal trigger
        the write is ignored without an error (SDG-13).
        """
        if self.get_trigger_source() is TriggerSource.INTERNAL:
            raise InstrumentError(
                f"{self.name}: the burst count needs an EXTERNAL or MANUAL trigger source "
                "(SDG-13); set_trigger_source first"
            )
        self._need_on()
        self.link.write_checked(f"{self._c}:BTWV COUNT,{int(n)}")
        return int(self._raw()["COUNTER"])

    def get_count(self) -> int:
        """The bursts per trigger."""
        return int(self._get_key("COUNTER"))

    def set_idle(self, idle: BurstIdle | str) -> BurstIdle:
        """Choose the level between bursts: the carrier at the start phase, or at the burst end.

        Enum arg: <sdg>.BurstIdle.<TAB>

        Other words are ignored without an error (SDG-13).
        """
        token = _token(idle, BurstIdle, "burst idle")
        return _member(self._set_key("HOLD", token), BurstIdle, "burst idle")

    def get_idle(self) -> BurstIdle:
        """The level between bursts."""
        return _member(self._get_key("HOLD"), BurstIdle, "burst idle")

    def set_delay(self, seconds: float) -> float:
        """Set the delay from the trigger to the first cycle, in seconds."""
        return _as_float(self._set_key("DLAY", _num(seconds)))

    def get_delay(self) -> float:
        """The delay from the trigger to the first cycle, in seconds."""
        return _as_float(self._get_key("DLAY"))

    def set_gate_polarity(self, polarity: Polarity | str) -> Polarity:
        """Choose which gate level runs a gated burst.

        Enum arg: <sdg>.Polarity.<TAB>
        """
        token = _token(polarity, Polarity, "gate polarity")
        return _member(self._set_key("PLRT", token), Polarity, "gate polarity")

    def get_gate_polarity(self) -> Polarity:
        """Which gate level runs a gated burst."""
        return _member(self._get_key("PLRT"), Polarity, "gate polarity")

    def cmd_trigger(self) -> None:
        """Run one burst, trigger source MANUAL."""
        self._need_on()
        self.link.write_checked(f"{self._c}:BTWV MTRIG")


# ---------------------------------------------------------------------------
# Other channel pages.
# ---------------------------------------------------------------------------


class _Harmonics(_SdgNode):
    """Harmonics added to a sine carrier: order 2 to 20, level in dBc or Vpp (UM p.36-37)."""

    def __init__(self, ch: _Channel):
        super().__init__(ch.link, name=f"{ch.name} harmonics")
        self._ch = ch
        self._c = ch._c

    def _raw(self) -> dict:
        body = self.link.query_body(f"{self._c}:HARM?")
        return _pairs(body.split(","), body)

    def _write(self, arg: str) -> None:
        self.link.write_checked(f"{self._c}:HARM {arg}")

    def get_state(self) -> bool:
        """Whether harmonics are added."""
        return self._raw()["HARMSTATE"] == "ON"

    def set_state(self, state: bool) -> bool:
        """Turn the harmonics on or off; on needs a sine carrier (PG p.31)."""
        if state and self._ch._get_function() is not Function.SINE:
            raise InstrumentError(f"{self._ch.name}: harmonics need a sine carrier (PG p.31)")
        self._write(f"HARMSTATE,{_on_off(state)}")
        if self.get_state() != state:
            raise InstrumentError(f"{self.name} did not turn {_on_off(state).lower()}")
        return state

    def set_type(self, harmonic_type: HarmonicType | str) -> HarmonicType:
        """Choose which orders may carry a level: even, odd or all.

        Enum arg: <sdg>.HarmonicType.<TAB>
        """
        self._write(f"HARMTYPE,{_token(harmonic_type, HarmonicType, 'harmonic type')}")
        return self.get_type()

    def get_type(self) -> HarmonicType:
        """Which orders may carry a level."""
        return _member(self._raw()["HARMTYPE"], HarmonicType, "harmonic type")

    def set_level(
        self,
        order: int,
        *,
        dbc: float | None = None,
        vpp: float | None = None,
        phase: float | None = None,
    ) -> HarmonicLevel:
        """Set one harmonic's level, in dBc or Vpp but not both, and its phase in degrees."""
        if dbc is not None and vpp is not None:
            raise ValueError("give dbc or vpp, not both")
        arg = f"HARMORDER,{int(order)}"
        if dbc is not None:
            arg += f",HARMDBC,{_num(dbc)}"
        if vpp is not None:
            arg += f",HARMAMP,{_num(vpp)}"
        if phase is not None:
            arg += f",HARMPHASE,{_num(phase)}"
        self._write(arg)
        got = self.get_level(order)
        for name, wanted, have in (
            ("dbc", dbc, got.dbc),
            ("vpp", vpp, got.vpp),
            ("phase", phase, got.phase),
        ):
            if wanted is not None and not _close(wanted, have):
                raise InstrumentError(
                    f"{self.name} {order}: {name} requested {wanted}, applied {have} "
                    "(the type allows even, odd or all orders)"
                )
        return got

    def get_level(self, order: int) -> HarmonicLevel:
        """Read one harmonic's level; this selects that order on the panel."""
        self._write(f"HARMORDER,{int(order)}")
        found = self._raw()
        if int(found.get("HARMORDER", -1)) != int(order):
            raise InstrumentError(
                f"{self.name}: order reads {found.get('HARMORDER')} after {order} "
                f"(type {found.get('HARMTYPE')})"
            )
        return HarmonicLevel(
            order=int(order),
            dbc=found["HARMDBC"],
            vpp=found["HARMAMP"],
            phase=found["HARMPHASE"],
        )

    def set_levels(
        self,
        levels: dict[int, float | tuple[float, float]],
        *,
        harmonic_type: HarmonicType | str = HarmonicType.ALL,
    ) -> HarmonicsState:
        """Turn harmonics on with ``{order: dbc}`` or ``{order: (dbc, phase)}``.

        Other orders keep their level.
        """
        self.set_state(True)
        self.set_type(harmonic_type)
        for order, value in levels.items():
            dbc, phase = value if isinstance(value, tuple) else (value, None)
            self.set_level(order, dbc=dbc, phase=phase)
        return self.get_setup(list(levels))

    def get_setup(self, orders: Sequence[int] | None = None) -> HarmonicsState:
        """Whether harmonics are on, the type, and the levels of ``orders``."""
        found = self._raw()
        if found["HARMSTATE"] != "ON":
            return HarmonicsState(enabled=False, harmonic_type=None)
        return HarmonicsState(
            enabled=True,
            harmonic_type=_member(found["HARMTYPE"], HarmonicType, "harmonic type"),
            levels=tuple(self.get_level(o) for o in (orders or ())),
        )


class _Hopping(_SdgNode):
    """The Hopping page (UM p.81-84, PG p.60-69). Needs the basic generator and a sine.

    The instrument has no query for the frequency, order and avoid tables (MEAS:
    -113), so their setters return the tables written.
    """

    def __init__(self, ch: _Channel):
        super().__init__(ch.link, name=f"{ch.name} hopping")
        self._ch = ch
        self._c = ch._c

    def _q(self, part: str) -> str:
        return self.link.query(f"{self._c}:FHOP:{part}?").strip('"')

    def _w(self, part: str) -> None:
        self.link.write_checked(f"{self._c}:FHOP:{part}")

    def _prepare(
        self,
        caller: str,
        ampl: float,
        offset: float,
        ampl_limit: float,
        load: float | Load | str | None,
    ) -> None:
        ch = self._ch
        ch._check_limit(caller, ampl, ampl_limit, load)
        ch._plain()
        if load is not None:
            ch.set_load(load)
        ch._set_function(Function.SINE)
        ch._write_limit(ampl_limit)
        for key, value in (("OFST", offset), ("AMP", ampl)):
            got = ch._set_bswv(key, value, warn=False)
            if not _close(value, got):
                raise InstrumentError(
                    f"{caller}(): {_BSWV_NAMES[key]} requested {value:g}, applied {got}"
                )

    def _finish(self, dwell: float) -> HoppingState:
        self.set_dwell(dwell)
        self.set_state(True)
        return self.get_setup()

    def set_list(
        self,
        frequencies: Sequence[float],
        *,
        dwell: float,
        ampl: float,
        ampl_limit: float,
        offset: float = 0.0,
        order: Sequence[int] | None = None,
        load: float | Load | str | None = None,
    ) -> HoppingState:
        """Hop through ``frequencies`` in Hz, in list order or in ``order``, ``dwell`` s each.

        ``order`` holds 0-based positions in ``frequencies``. ``ampl`` is in
        Vpp and ``ampl_limit`` is the amplitude limit in high-impedance Vpp
        (SDG-34). The output is not turned on.
        """
        self._prepare("set_list", ampl, offset, ampl_limit, load)
        self._w("TYPE MANUAL")
        self.set_frequencies(frequencies)
        self.set_order(order if order is not None else list(range(len(frequencies))))
        return self._finish(dwell)

    def set_random(
        self,
        *,
        start: float,
        stop: float,
        step: float,
        dwell: float,
        ampl: float,
        ampl_limit: float,
        offset: float = 0.0,
        avoid: Sequence[tuple[float, float]] | None = None,
        prbs_order: int = 7,
        load: float | Load | str | None = None,
    ) -> HoppingState:
        """Hop at random on the grid ``start`` to ``stop`` in ``step`` Hz, skipping ``avoid``."""
        self._prepare("set_random", ampl, offset, ampl_limit, load)
        self._w("TYPE RHOP")
        for part, value in (("SFREquency", start), ("EFREquency", stop), ("FSTep", step)):
            self._w(f"{part} {_num(value)}")
            got = float(self._q(part))
            if not _close(value, got):
                raise InstrumentError(f"{self.name} {part}: requested {value:g}, applied {got}")
        self._w(f"RPATtern {int(prbs_order)}")
        self.set_avoid(avoid or [])
        return self._finish(dwell)

    def set_random_list(
        self,
        frequencies: Sequence[float],
        *,
        dwell: float,
        ampl: float,
        ampl_limit: float,
        offset: float = 0.0,
        prbs_order: int = 7,
        load: float | Load | str | None = None,
    ) -> HoppingState:
        """Hop through ``frequencies`` in Hz in a random order, ``dwell`` s each."""
        self._prepare("set_random_list", ampl, offset, ampl_limit, load)
        self._w("TYPE RLIST")
        self.set_frequencies(frequencies)
        self._w(f"RLPAttern {int(prbs_order)}")
        return self._finish(dwell)

    def set_state(self, state: bool) -> bool:
        """Turn hopping on or off; returns the read-back."""
        self._w(f"SWITch {_on_off(state)}")
        if self.get_state() != state:
            raise InstrumentError(f"{self.name} did not turn {_on_off(state).lower()}")
        return state

    def get_state(self) -> bool:
        """Whether hopping runs."""
        return self._q("SWITch") == "ON"

    def set_dwell(self, seconds: float) -> float:
        """Set the time at each frequency, in seconds."""
        self._w(f"TIME {_num(seconds)}")
        return self.get_dwell()

    def get_dwell(self) -> float:
        """The time at each frequency, in seconds."""
        return float(self._q("TIME"))

    def _set_table(self, kind: str, clear: str, modify: str, add: str, items: list[str]) -> None:
        if not items:
            raise ValueError(f"{kind}: give at least one item")
        self._w(clear)
        self._w(f"{modify} 1,{items[0]}")
        for i, item in enumerate(items[1:], 2):
            self._w(f"{add} {i},{item}")

    def set_frequencies(self, frequencies: Sequence[float]) -> list[float]:
        """Write the frequency table in Hz; returns the values written (no query exists)."""
        values = [float(f) for f in frequencies]
        self._set_table("frequencies", "CFLIst", "MFLIst", "AFLIst", [_num(f) for f in values])
        return values

    def set_order(self, order: Sequence[int]) -> list[int]:
        """Write the order table as 0-based positions; returns the values written."""
        values = [int(i) for i in order]
        self._set_table("order", "COLIst", "MOLIst", "AOLIst", [str(i + 1) for i in values])
        return values

    def set_avoid(self, ranges: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        """Write the avoid ranges (low, high) in Hz; empty turns avoid off. Returns them."""
        values = [(float(lo), float(hi)) for lo, hi in ranges]
        self._w("CALIst")
        for lo, hi in values:
            self._w(f"AALIst {_num(lo)},{_num(hi)}")
        self._w(f"ALSTate {_on_off(bool(values))}")
        return values

    def _prbs_header(self) -> str:
        return "RLPAttern" if self._q("TYPE") == "RLIST" else "RPATtern"

    def set_prbs_order(self, n: int) -> int:
        """Set the order of the pseudo-random sequence for the present random type."""
        header = self._prbs_header()
        self._w(f"{header} {int(n)}")
        return int(self._q(header))

    def get_prbs_order(self) -> int:
        """The order of the pseudo-random sequence for the present random type."""
        return int(self._q(self._prbs_header()))

    def cmd_save(self, name: str) -> None:
        """Save the three tables as ``Local/<name>_frequencies.hop``, ``_order``, ``_avoid``."""
        for cmd, part in (("SFLIst", "frequencies"), ("SOLIst", "order"), ("SALIst", "avoid")):
            self._w(f'{cmd} "Local/{name}_{part}.hop"')

    def cmd_recall(self, name: str) -> None:
        """Recall tables saved by ``cmd_save(name)``."""
        for cmd, part in (("LFLIst", "frequencies"), ("LOLIst", "order"), ("LALIst", "avoid")):
            self._w(f'{cmd} "Local/{name}_{part}.hop"')

    def get_setup(self) -> HoppingState:
        """The hopping state, type, dwell and random-grid settings."""
        hop_type = _member(self._q("TYPE"), HopType, "hop type")
        extra: dict = {}
        if hop_type is HopType.RANDOM:
            extra = {
                "start": float(self._q("SFREquency")),
                "stop": float(self._q("EFREquency")),
                "step": float(self._q("FSTep")),
                "prbs_order": int(self._q("RPATtern")),
                "avoid_enabled": self._q("ALSTate") == "ON",
            }
        elif hop_type is HopType.RANDOM_LIST:
            extra = {"prbs_order": int(self._q("RLPAttern"))}
        return HoppingState(
            enabled=self.get_state(), hop_type=hop_type, dwell=self.get_dwell(), **extra
        )


class _Multitone(_SdgNode):
    """The Multitone page (UM p.85-88, PG p.70-71).

    On puts the channel in sequence mode (MEAS); off returns it to the basic
    generator.
    """

    def __init__(self, ch: _Channel):
        super().__init__(ch.link, name=f"{ch.name} multitone")
        self._ch = ch
        self._c = ch._c

    def _q(self, part: str) -> str:
        return self.link.query(f"{self._c}:MTONE:{part}?")

    def _w(self, part: str) -> None:
        self.link.write_checked(f"{self._c}:MTONE:{part}")

    def set_tone_list(
        self,
        frequencies: Sequence[float],
        *,
        ampl: float,
        sample_rate: float | None = None,
        load: float | Load | str | None = None,
    ) -> MultitoneState:
        """Play tones at ``frequencies`` in Hz with a total ``ampl`` in Vpp; output not turned on.

        The instrument picks the sample rate and ignores writes to it (MEAS), so
        ``sample_rate`` raises if it is not applied.
        """
        ch = self._ch
        ch._plain()
        if load is not None:
            ch.set_load(load)
        self.set_state(True)
        self.set_tones(frequencies)
        self.set_notches([])
        got = self.set_amplitude(ampl)
        if not _close(ampl, got):
            raise InstrumentError(f"{self.name}: amplitude {got}, requested {ampl:g}")
        if sample_rate is not None:
            got = self.set_sample_rate(sample_rate)
            if not _close(sample_rate, got):
                raise InstrumentError(
                    f"{self.name}: sample rate {got:g} Sa/s, requested {sample_rate:g}. "
                    "The instrument chooses the rate (MEAS)."
                )
        self.set_run_state(True)
        return self.get_setup()

    def set_tone_range(
        self,
        start: float,
        stop: float,
        count: int,
        *,
        ampl: float,
        notches: Sequence[tuple[float, float]] | None = None,
        sample_rate: float | None = None,
        load: float | Load | str | None = None,
    ) -> MultitoneState:
        """Play ``count`` equally spaced tones from ``start`` to ``stop`` Hz, minus ``notches``."""
        if count < 1:
            raise ValueError("count must be at least 1")
        tones = (
            [start]
            if count == 1
            else [start + i * (stop - start) / (count - 1) for i in range(count)]
        )
        tones = [f for f in tones if not any(lo <= f <= hi for lo, hi in (notches or ()))]
        return self.set_tone_list(tones, ampl=ampl, sample_rate=sample_rate, load=load)

    def set_state(self, state: bool) -> bool:
        """Turn multitone on, or off and back to the basic generator; returns the read-back."""
        if not state:
            self.set_run_state(False)
        self._w(f"STAte {_on_off(state)}")
        if self.get_state() != state:
            raise InstrumentError(f"{self.name} did not turn {_on_off(state).lower()}")
        if not state:
            self._ch._afg()
        return state

    def get_state(self) -> bool:
        """Whether multitone is on."""
        return self._q("STAte") == "ON"

    def set_run_state(self, state: bool) -> bool:
        """Start or stop playing; returns the read-back. Raises if a start does not take."""
        self._w(f"RSTAte {_on_off(state)}")
        got = self.get_run_state()
        if state and not got:
            raise InstrumentError(f"{self.name} did not start")
        return got

    def get_run_state(self) -> bool:
        """Whether multitone plays."""
        return self._q("RSTAte") == "ON"

    def _replace(
        self, clear: str, add: str, delete: str, query: str, items: list[str], expect: list
    ) -> list:
        self._w(clear)
        defaults = len(_parse_indexed(self._q(query)))  # a clear leaves default items (MEAS)
        for item in items:
            self._w(f"{add} {item}")
        for i in range(defaults, 0, -1):
            self._w(f"{delete} {i}")
        got = _parse_indexed(self._q(query))
        flat = [tuple(g) if len(g) > 1 else g[0] for g in got]
        mismatch = len(flat) != len(expect) or any(
            (not all(_close(a, b) for a, b in zip(e, g)))
            if isinstance(e, tuple)
            else not _close(e, g)
            for e, g in zip(expect, flat)
        )
        if mismatch:
            raise InstrumentError(f"{self.name} {query}: reads {flat}, requested {expect}")
        return flat

    def set_tones(self, frequencies: Sequence[float]) -> list[float]:
        """Replace the tone table with ``frequencies`` in Hz; returns the read-back."""
        values = [float(f) for f in frequencies]
        return self._replace(
            "CTLIst", "ATLIst", "DTLIst", "TLISt", [_num(f) for f in values], values
        )

    def get_tones(self) -> list[float]:
        """The tone frequencies in Hz."""
        return [g[0] for g in _parse_indexed(self._q("TLISt"))]

    def set_notches(self, ranges: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        """Replace the notch ranges (low, high) in Hz; empty turns notches off.

        The notch state goes on before the list; otherwise the list write fails
        with -200 (MEAS).
        """
        values = [(float(lo), float(hi)) for lo, hi in ranges]
        if not values:
            self._w("NSTAte OFF")
            return []
        self._w("NSTAte ON")
        return self._replace(
            "CNLIst",
            "ANLIst",
            "DNLIst",
            "NLISt",
            [f"{_num(lo)},{_num(hi)}" for lo, hi in values],
            values,
        )

    def get_notches(self) -> list[tuple[float, float]]:
        """The notch ranges (low, high) in Hz; empty while notches are off."""
        if self._q("NSTAte") != "ON":
            return []
        return [tuple(g) for g in _parse_indexed(self._q("NLISt"))]

    def set_amplitude(self, vpp: float) -> float:
        """Set the total amplitude in Vpp."""
        self._w(f"AMPlitude {_num(vpp)}")
        return self.get_amplitude()

    def get_amplitude(self) -> float:
        """The total amplitude in Vpp."""
        return float(self._q("AMPlitude"))

    def set_sample_rate(self, rate: float) -> float:
        """Write the sample rate in Sa/s; the instrument keeps its own choice (MEAS)."""
        self._w(f"SRATe {_num(rate)}")
        return self.get_sample_rate()

    def get_sample_rate(self) -> float:
        """The sample rate in Sa/s."""
        return float(self._q("SRATe"))

    def get_setup(self) -> MultitoneState:
        """The whole multitone page."""
        return MultitoneState(
            enabled=self.get_state(),
            running=self.get_run_state(),
            tones=tuple(self.get_tones()),
            notches=tuple(self.get_notches()),
            amplitude=self.get_amplitude(),
            sample_rate=self.get_sample_rate(),
        )


class _Multipulse(_SdgNode):
    """The Multi Pulse page (UM p.89, PG p.72-73).

    On puts the channel in sequence mode (MEAS); off returns it to the basic
    generator.
    """

    def __init__(self, ch: _Channel):
        super().__init__(ch.link, name=f"{ch.name} multipulse")
        self._ch = ch
        self._c = ch._c

    def _q(self, part: str) -> str:
        return self.link.query(f"{self._c}:MPULse:{part}?")

    def _w(self, part: str) -> None:
        self.link.write_checked(f"{self._c}:MPULse:{part}")

    def set_train(
        self,
        pulses: Sequence[tuple[float, float]],
        *,
        level_high: float,
        level_low: float,
        trigger_source: PatternTriggerSource | str = PatternTriggerSource.INTERNAL,
        timer: float | None = None,
        delay: float = 0.0,
        sample_rate: float | None = None,
        load: float | Load | str | None = None,
    ) -> MultipulseState:
        """Play a pulse train: ``pulses`` as (width, gap) in seconds, 2 to 30 of them.

        Enum arg: trigger_source = <sdg>.PatternTriggerSource.<TAB>

        ``sample_rate`` None lets the instrument choose. The output is not
        turned on. Whether the amplitude limit applies in this mode was never
        observed  # UNVERIFIED
        """
        ch = self._ch
        ch._plain()
        if load is not None:
            ch.set_load(load)
        self.set_state(True)
        self.set_pulses(pulses)
        self.set_level_high(level_high)
        self.set_level_low(level_low)
        self.set_trigger_source(trigger_source)
        if timer is not None:
            self.set_trigger_timer(timer)
        self.set_trigger_delay(delay)
        self.set_sample_rate(sample_rate)
        self.set_run_state(True)
        return self.get_setup()

    def set_state(self, state: bool) -> bool:
        """Turn multipulse on, or off and back to the basic generator; returns the read-back."""
        if not state:
            self.set_run_state(False)
        self._w(f"STAte {_on_off(state)}")
        if self.get_state() != state:
            raise InstrumentError(f"{self.name} did not turn {_on_off(state).lower()}")
        if not state:
            self._ch._afg()
        return state

    def get_state(self) -> bool:
        """Whether multipulse is on."""
        return self._q("STAte") == "ON"

    def set_run_state(self, state: bool) -> bool:
        """Start or stop playing; returns the read-back. Raises if a start does not take."""
        self._w(f"RSTAte {_on_off(state)}")
        got = self.get_run_state()
        if state and not got:
            raise InstrumentError(f"{self.name} did not start")
        return got

    def get_run_state(self) -> bool:
        """Whether multipulse plays."""
        return self._q("RSTAte") == "ON"

    def cmd_trigger(self) -> None:
        """Play one pulse train, trigger source MANUAL.

        The command is a guess: the PG p.73 example shows a different header,
        and the original driver never checked it  # UNVERIFIED
        """
        self.link.write_checked(f"{self._c}:AWG:TRIGger:IMMediate")

    def set_pulses(self, pulses: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        """Write the pulses as (width, gap) in seconds, 2 to 30 (PG p.72); returns the read-back."""
        values = [(float(w), float(g)) for w, g in pulses]
        if not 2 <= len(values) <= 30:
            raise ValueError("pulses: give 2 to 30 (width, gap) pairs (PG p.72)")
        self._w(f"PNUMber {len(values)}")
        for i, (width, gap) in enumerate(values, 1):
            self._w(f"MPULse {i},{_num(width)},{_num(gap)}")
        got = self.get_pulses()
        if len(got) != len(values) or any(
            not (_close(a, c) and _close(b, d)) for (a, b), (c, d) in zip(values, got)
        ):
            raise InstrumentError(f"{self.name}: reads {got}, requested {values}")
        return got

    def get_pulses(self) -> list[tuple[float, float]]:
        """The pulses as (width, gap) in seconds."""
        return [tuple(g) for g in _parse_indexed(self._q("PULse"))]

    def set_level_high(self, volts: float) -> float:
        """Set the high level in volts."""
        self._w(f"HLEVel {_num(volts)}")
        return self.get_level_high()

    def get_level_high(self) -> float:
        """The high level in volts."""
        return float(self._q("HLEVel"))

    def set_level_low(self, volts: float) -> float:
        """Set the low level in volts."""
        self._w(f"LLEVel {_num(volts)}")
        return self.get_level_low()

    def get_level_low(self) -> float:
        """The low level in volts."""
        return float(self._q("LLEVel"))

    def set_trigger_source(self, source: PatternTriggerSource | str) -> PatternTriggerSource:
        """Choose what starts a pulse train.

        Enum arg: <sdg>.PatternTriggerSource.<TAB>
        """
        self._w(f"TRIGger:SOURce {_token(source, PatternTriggerSource, 'trigger source')}")
        return self.get_trigger_source()

    def get_trigger_source(self) -> PatternTriggerSource:
        """What starts a pulse train."""
        return _member(self._q("TRIGger:SOURce"), PatternTriggerSource, "trigger source")

    def set_trigger_timer(self, seconds: float) -> float:
        """Set the timer period in seconds."""
        self._w(f"TRIGger:TIMer {_num(seconds)}")
        return self.get_trigger_timer()

    def get_trigger_timer(self) -> float:
        """The timer period in seconds."""
        return float(self._q("TRIGger:TIMer"))

    def set_trigger_delay(self, seconds: float) -> float:
        """Set the delay from the trigger in seconds."""
        self._w(f"TRIGger:DELAy {_num(seconds)}")
        return self.get_trigger_delay()

    def get_trigger_delay(self) -> float:
        """The delay from the trigger in seconds."""
        return float(self._q("TRIGger:DELAy"))

    def set_sample_rate(self, rate: float | None) -> float | None:
        """Set the sample rate in Sa/s; None lets the instrument choose."""
        if rate is None:
            self._w("SRate:TYPe AUTO")
        else:
            self._w("SRate:TYPe CUSTOM")
            self._w(f"SRate {_num(rate)}")
        return self.get_sample_rate()

    def get_sample_rate(self) -> float | None:
        """The sample rate in Sa/s; None when the instrument chooses."""
        if self._q("SRate:TYPe") == "AUTO":
            return None
        return float(self._q("SRate"))

    def get_setup(self) -> MultipulseState:
        """The whole multi-pulse page."""
        return MultipulseState(
            enabled=self.get_state(),
            running=self.get_run_state(),
            pulses=tuple(self.get_pulses()),
            level_high=self.get_level_high(),
            level_low=self.get_level_low(),
            trigger_source=self.get_trigger_source(),
            trigger_timer=self.get_trigger_timer(),
            trigger_delay=self.get_trigger_delay(),
            sample_rate=self.get_sample_rate(),
        )


class _Sequence(_SdgNode):
    """Sequence mode: the front-panel Seq key (UM p.49-57, PG p.52-54).

    One segment table per channel. A sequence file also holds scenario and
    sequence levels, jumps and markers; they load but do not play on this model
    (SDG-33). In sequence mode Trig/Sync OUT sends nothing (SDG-30).

    MEAS 2026-10-15, firmware 1.1.1.7R10: segment insert, delete, clear, start
    segment and amplitude-scale writes all fail with -200, so a table of more
    than one segment goes through a sequence file (SDG-06).
    """

    _SCPI_RUN = (RunMode.CONTINUOUS, RunMode.TRIGGERED, RunMode.BURST)  # what SCPI sets (SDG-06)

    def __init__(self, ch: _Channel):
        super().__init__(ch.link, name=f"{ch.name} sequence")
        self._ch = ch
        self._c = ch._c

    def _q(self, part: str) -> str:
        return self.link.query(f"{self._c}:AWG:{part}?")

    def _w(self, part: str) -> None:
        self.link.write_checked(f"{self._c}:AWG:{part}")

    def _enter(self) -> None:
        if self._ch.get_mode() is not ChannelMode.SEQUENCE:
            self._ch._plain()
            self.link.write_checked(f"{self._c}:CPARam MODE,AWG")

    def set_segments(
        self,
        segments: Sequence[Segment],
        *,
        sample_rate: float,
        run_mode: RunMode | str = RunMode.CONTINUOUS,
        trigger_source: SequenceTriggerSource | str = SequenceTriggerSource.MANUAL,
        trigger_edge: SequenceEdge | str | None = None,
        burst_count: int | None = None,
        burst_period: float | None = None,
        interpolation: Interpolation | str = Interpolation.ZERO,
        offset: float = 0.0,
        load: float | Load | str | None = None,
        name: str = "drv_seq",
    ) -> SequenceState:
        """Load ``segments`` as the sequence, read every segment back, and start it playing.

        Enum arg: run_mode       = <sdg>.RunMode.<TAB>
                  trigger_source = <sdg>.SequenceTriggerSource.<TAB>
                  interpolation  = <sdg>.Interpolation.<TAB>

        One segment in run mode CONTINUOUS, TRIGGERED or BURST goes over SCPI.
        Anything else is refused by SCPI (SDG-06), so the driver saves the page
        from the instrument, builds ``Local/<name>.AWGx`` from that save with
        only saved codes (SDG-07), and loads it. User arbs always go by file,
        because SCPI ignores their names (SDG-29).

        Run modes (MEAS 2026-10-15): TRIGGERED waits for the first trigger and
        then plays on; BURST plays ``burst_count`` passes ``burst_period`` apart
        per trigger, ignores a trigger inside a pass and re-arms about 13 us
        after a pass, so a retriggered burst never chains without a gap (SDG-36;
        for a gapless loop use TRIGGERED or CONTINUOUS); STEP plays one segment per trigger; ADVANCED plays the
        chain, and a segment with a ``wait`` holds until its event. The output
        is not turned on. Whether the amplitude limit applies in this mode was
        never observed  # UNVERIFIED
        """
        segments = list(segments)
        mode = _member(_token(run_mode, RunMode, "run mode"), RunMode, "run mode")
        if not segments:
            raise ValueError("a sequence needs at least one segment")
        n = len(segments)
        if n > _SEGMENT_MAX:
            raise ValueError(f"{n} segments: the instrument holds {_SEGMENT_MAX} (DS p.11)")
        for k, seg in enumerate(segments, start=1):
            points = int(seg.length)
            if points < _SEGMENT_MIN_POINTS or (points < 256 and points % _SEGMENT_STEP_BELOW_256):
                raise ValueError(
                    f"segment {k}: length {points} points; the instrument needs at least 64, "
                    "and a multiple of 16 below 256 (DS p.11)"
                )
            if seg.next is not None and not 1 <= seg.next <= n:
                raise ValueError(f"segment {k}: next {seg.next} is not a segment number 1..{n}")
            if seg.wait is not WaitEvent.AUTO and mode is not RunMode.ADVANCED:
                raise ValueError(
                    f"segment {k}: wait {seg.wait.name} needs run_mode ADVANCED (UM p.53)"
                )
        self._enter()
        if load is not None:
            self._ch.set_load(load)
        self.set_run_state(False)
        self.set_sample_rate(sample_rate)
        self.set_run_mode(mode if mode in self._SCPI_RUN else RunMode.CONTINUOUS)
        source_set = None
        edge_set = None
        if mode is not RunMode.CONTINUOUS:
            source_set = self.set_trigger_source(trigger_source)
        if trigger_edge is not None:
            edge_set = self.set_trigger_edge(trigger_edge)
        self.set_interpolation(interpolation)
        self.set_offset(offset)
        user = set(self._ch._sdg.get_arb_names_user())
        first = segments[0]
        one_by_scpi = (
            n == 1
            and mode in self._SCPI_RUN
            and first.wait is WaitEvent.AUTO
            and first.next in (None, 1)
            and first.waveform not in user
            and self.get_segment_count() == 1  # SCPI cannot delete segments
        )
        if one_by_scpi:
            self.set_segment(1, first)
        else:
            self._load_file(segments, mode, name, user)
            # MEAS 2026-10-18: after the load the trigger source reads the file's
            # value, not the one set before the load (SDG-37). Set both again.
            if source_set is not None and self.set_trigger_source(trigger_source) is not source_set:
                raise InstrumentError(
                    f"{self.name}: trigger source reads {self.get_trigger_source().name} "
                    f"after the file load, requested {source_set.name}"
                )
            if edge_set is not None and self.set_trigger_edge(trigger_edge) is not edge_set:
                raise InstrumentError(
                    f"{self.name}: trigger edge reads {self.get_trigger_edge().name} "
                    f"after the file load, requested {edge_set.name}"
                )
        if burst_count is not None:
            self.set_burst_count(burst_count)
        if burst_period is not None:
            self.set_burst_period(burst_period)
        written = self.get_segments()  # read back while stopped (SDG-05)
        self._compare(segments, written)
        if self.get_run_mode() is not mode:
            raise InstrumentError(
                f"{self.name}: run mode reads {self.get_run_mode().name}, requested {mode.name}"
            )
        self.set_run_state(True)
        state = self.get_setup()
        return replace(state, segments=tuple(written))

    def _save_xml(self) -> str:
        self._w(f'SAVE PATH,"Local/{_SEQUENCE_TMP}.AWGx",SWFS,"FALSE"')
        # MEAS: the saved name is lower case.
        return _awgx_test_xml(self._ch._sdg.file.get_file(f"Local/{_SEQUENCE_TMP}.awgx"))

    def _load_file(self, segments: list[Segment], mode: RunMode, name: str, user: set[str]) -> None:
        def key(seg: Segment) -> tuple:
            return ("<user>" if seg.waveform in user else seg.waveform.lower(), int(seg.length))

        # One save per built-in waveform and length. User arbs of one length
        # share a save, because their path is set per segment.
        bodies: dict = {}
        for seg in segments:
            k = key(seg)
            if k not in bodies:
                if seg.waveform not in user:
                    self._w(f"SEGMent1:WAVeform {seg.waveform}")
                self._w(f"SEGMent1:LENGth {int(seg.length)}")
                bodies[k] = _awgx_segment_xml(self._save_xml(), 0)
        template = self._save_xml()
        n = len(segments)

        def body(seg: Segment) -> str:
            text = bodies[key(seg)]
            if seg.waveform in user:  # the path the instrument saves (SDG-29)
                text = _awgx_set_tag(text, "waveform", f"{_AWGX_USER_FOLDER}/{seg.waveform}.bin")
            return text

        rows = [
            {
                "body": body(seg),
                "loops": seg.loops,
                "wait": seg.wait.name.lower(),
                "next": (seg.next - 1) if seg.next is not None else (k + 1) % n,
                "amplitude": seg.amplitude,
                "offset": seg.offset,
            }
            for k, seg in enumerate(segments)
        ]
        xml = _awgx_build(template, rows, run_mode=mode.name.lower())
        _awgx_check_codes(xml, template)
        files = self._ch._sdg.file
        files.cmd_write(f"Local/{name}.AWGx", _awgx_pack(name, xml.encode()))
        self._w(f'LOAD "Local/{name}.AWGx"')
        files.cmd_delete(f"Local/{_SEQUENCE_TMP}.awgx")

    def _compare(self, wanted: list[Segment], got: list[Segment]) -> None:
        n = len(wanted)
        bad = len(got) != n
        for k, (w, g) in enumerate(zip(wanted, got), start=1):
            nxt = w.next if w.next is not None else k % n + 1
            bad |= (
                g.waveform.lower() != w.waveform.lower()
                or g.length != int(w.length)
                or g.loops != int(w.loops)
                or g.wait is not w.wait
                or (n > 1 and g.next != nxt)
                or (w.amplitude is not None and not _close(w.amplitude, g.amplitude))
                or (w.offset is not None and not _close(w.offset, g.offset))
            )
        if bad:
            raise InstrumentError(f"{self.name}: reads {got}, requested {wanted}")

    def set_run_state(self, state: bool) -> bool:
        """Start or stop the sequence; returns the read-back. Raises if a start does not take."""
        # MEAS 2026-11-05: the start of a 15 Mpoint sequence holds *OPC? beyond the 3 s default.
        with self.link.temporary_timeout(_SEQUENCE_RUN_TIMEOUT_MS):
            self._w(f"STATE {'RUN' if state else 'STOP'}")
        got = self.get_run_state()
        if state and not got:
            raise InstrumentError(f"{self.name} did not start")
        return got

    def get_run_state(self) -> bool:
        """Whether the sequence plays."""
        return self._q("STATE") == "RUN"

    def cmd_trigger(self) -> None:
        """Send the sequence a manual trigger."""
        self._w("TRIGger:IMMediate")

    def get_segment_count(self) -> int:
        """The segments in the sequence; stopped only (SDG-05).

        While the sequence plays, the count query gets no reply and hangs the
        link until the timeout, so this refuses instead.
        """
        if self.get_run_state():
            raise InstrumentError(
                f"{self.name}: the segment count gets no reply while the sequence plays "
                "(SDG-05); set_run_state(False) first"
            )
        return int(self._q("SEGMent:COUNt"))

    def set_segment(self, index: int, segment: Segment) -> Segment:
        """Change one existing segment: waveform, length, loops, levels and next.

        A wait other than AUTO is refused over SCPI (-200, MEAS), and a user
        arb name is ignored without an error (SDG-29); give both to
        ``set_segments``. The read-back raises on a mismatch.
        """
        if segment.wait is not WaitEvent.AUTO:
            raise ValueError(
                f"{self.name}: SCPI refuses a wait event (-200, MEAS); use set_segments"
            )
        count = self.get_segment_count()
        if not 1 <= index <= count:
            raise ValueError(
                f"{self.name}: segment {index} does not exist ({count} segment(s); an insert "
                "fails over SCPI, MEAS)"
            )
        p = f"SEGMent{index}"
        self._w(f"{p}:WAVeform {segment.waveform}")
        self._w(f"{p}:LENGth {int(segment.length)}")
        self._w(f"{p}:LOOP {int(segment.loops)}")
        if segment.amplitude is not None:
            self._w(f"{p}:AMPlitude {_num(segment.amplitude)}")
        if segment.offset is not None:
            self._w(f"{p}:OFFset {_num(segment.offset)}")
        if segment.next is not None:
            self._w(f"{p}:GOTO {int(segment.next)}")
        got = self.get_segment(index)
        if (
            got.waveform.lower() != segment.waveform.lower()
            or got.length != int(segment.length)
            or got.loops != int(segment.loops)
        ):
            raise InstrumentError(f"{self.name}: segment {index} reads {got}, requested {segment}")
        return got

    def get_segment(self, index: int) -> Segment:
        """Read one segment, numbered from 1."""
        p = f"SEGMent{index}"
        return Segment(
            waveform=self._q(f"{p}:WAVeform"),
            length=int(float(self._q(f"{p}:LENGth"))),
            loops=int(float(self._q(f"{p}:LOOP"))),
            amplitude=float(self._q(f"{p}:AMPlitude")),
            offset=float(self._q(f"{p}:OFFset")),
            next=int(float(self._q(f"{p}:GOTO"))),
            wait=_member(self._q(f"{p}:WAITEvent"), WaitEvent, "wait event"),
        )

    def get_segments(self) -> list[Segment]:
        """Read every segment; stopped only (SDG-05)."""
        return [self.get_segment(i) for i in range(1, self.get_segment_count() + 1)]

    def set_sample_rate(self, rate: float) -> float:
        """Set the sample rate in Sa/s, 10 mSa/s to 600 MSa/s (UM p.50)."""
        self._w(f"SRATE {_num(rate)}")
        return self.get_sample_rate()

    def get_sample_rate(self) -> float:
        """The sample rate in Sa/s."""
        return float(self._q("SRATE"))

    def set_run_mode(self, mode: RunMode | str) -> RunMode:
        """Choose CONTINUOUS, TRIGGERED or BURST; STEP and ADVANCED go through set_segments.

        Enum arg: <sdg>.RunMode.<TAB>

        SCPI refuses STEP and ADVANCED with -200 (SDG-06).
        """
        member = _member(_token(mode, RunMode, "run mode"), RunMode, "run mode")
        if member not in self._SCPI_RUN:
            raise ValueError(
                f"{self.name}: run mode {member.name} is set only through set_segments (SDG-06)"
            )
        self._w(f"RMODe {member.value}")
        return self.get_run_mode()

    def get_run_mode(self) -> RunMode:
        """How the sequence plays."""
        return _member(self._q("RMODe"), RunMode, "run mode")

    def set_trigger_source(self, source: SequenceTriggerSource | str) -> SequenceTriggerSource:
        """Choose what triggers the sequence: MANUAL, TIMER or EXTERNAL.

        Enum arg: <sdg>.SequenceTriggerSource.<TAB>

        Written TIM, read back TIME; TIMer is refused with -224 (MEAS).
        """
        token = _token(source, SequenceTriggerSource, "trigger source")
        if token == SequenceTriggerSource.FREE_RUN.value:
            raise ValueError("FREE_RUN is read back only; choose MANUAL, TIMER or EXTERNAL")
        self._w(f"TRIGger:SOURce {token}")
        return self.get_trigger_source()

    def get_trigger_source(self) -> SequenceTriggerSource:
        """What triggers the sequence."""
        reply = self._q("TRIGger:SOURce").strip().upper()
        if reply == "TIME":
            return SequenceTriggerSource.TIMER
        return _member(reply, SequenceTriggerSource, "trigger source")

    def set_trigger_edge(self, edge: SequenceEdge | str) -> SequenceEdge:
        """Choose the external trigger edge.

        Enum arg: <sdg>.SequenceEdge.<TAB>

        The header is spelled as the PG prints it; the expected spelling is
        undefined, -113 (MEAS, SDG-06).
        """
        self._w(f"TRIGAger:SLOPe {_token(edge, SequenceEdge, 'edge')}")
        return self.get_trigger_edge()

    def get_trigger_edge(self) -> SequenceEdge:
        """The external trigger edge."""
        return _member(self._q("TRIGAger:SLOPe"), SequenceEdge, "edge")

    def set_trigger_timer(self, seconds: float) -> float:
        """Set the trigger timer period in seconds."""
        self._w(f"TRIGger:TIMer {_num(seconds)}")
        return self.get_trigger_timer()

    def get_trigger_timer(self) -> float:
        """The trigger timer period in seconds."""
        return float(self._q("TRIGger:TIMer"))

    def set_trigger_delay(self, seconds: float) -> float:
        """Set the trigger delay in seconds."""
        self._w(f"TRIGger:DELAy {_num(seconds)}")
        return self.get_trigger_delay()

    def get_trigger_delay(self) -> float:
        """The trigger delay in seconds."""
        return float(self._q("TRIGger:DELAy"))

    def set_burst_count(self, n: int) -> int:
        """Set the passes per trigger in run mode BURST, 1 to 65535; it clamps.

        The instrument refuses this with -200 while the sequence runs (SDG-36).
        """
        self._w(f"BURSt:COUNt {int(n)}")
        return self.get_burst_count()

    def get_burst_count(self) -> int:
        """The passes per trigger in run mode BURST."""
        return int(float(self._q("BURSt:COUNt")))

    def set_burst_period(self, seconds: float) -> float:
        """Set the time between pass starts in run mode BURST, in seconds.

        The minimum is the pass time + 1 us; it clamps. Passes start every 1.0101
        x this period, and the instrument needs about 13 us between passes. The
        instrument refuses this with -200 while the sequence runs (SDG-36).
        """
        self._w(f"BURSt:PERiod {_num(seconds)}")
        return self.get_burst_period()

    def get_burst_period(self) -> float:
        """The time between passes in run mode BURST, in seconds."""
        return float(self._q("BURSt:PERiod"))

    def set_interpolation(self, kind: Interpolation | str) -> Interpolation:
        """Choose the interpolation between points.

        Enum arg: <sdg>.Interpolation.<TAB>
        """
        self._w(f"INTPtype {_token(kind, Interpolation, 'interpolation')}")
        return self.get_interpolation()

    def get_interpolation(self) -> Interpolation:
        """The interpolation between points."""
        return _member(self._q("INTPtype"), Interpolation, "interpolation")

    def set_length_increase(self, kind: LengthIncrease | str) -> LengthIncrease:
        """Choose how a segment grows when its length increases.

        Enum arg: <sdg>.LengthIncrease.<TAB>
        """
        self._w(f"INCReasing {_token(kind, LengthIncrease, 'length increase')}")
        return self.get_length_increase()

    def get_length_increase(self) -> LengthIncrease:
        """How a segment grows when its length increases."""
        return _member(self._q("INCReasing"), LengthIncrease, "length increase")

    def set_length_decrease(self, kind: LengthDecrease | str) -> LengthDecrease:
        """Choose how a segment shrinks when its length decreases.

        Enum arg: <sdg>.LengthDecrease.<TAB>
        """
        self._w(f"DECReasing {_token(kind, LengthDecrease, 'length decrease')}")
        return self.get_length_decrease()

    def get_length_decrease(self) -> LengthDecrease:
        """How a segment shrinks when its length decreases."""
        return _member(self._q("DECReasing"), LengthDecrease, "length decrease")

    def get_scale(self) -> float:
        """The amplitude scale in percent. SCPI refuses a write of it (-200, SDG-06)."""
        return float(self._q("SCALe")) * 100

    def set_offset(self, volts: float) -> float:
        """Set the sequence offset in volts."""
        self._w(f"OFFset {_num(volts)}")
        return self.get_offset()

    def get_offset(self) -> float:
        """The sequence offset in volts."""
        return float(self._q("OFFset"))

    def set_hold(self, hold: SequenceHold | str) -> SequenceHold:
        """Choose the level after the sequence ends: MID, START, END, or USER.

        Enum arg: <sdg>.SequenceHold.<TAB>

        USER holds the level of ``set_hold_user_level``.
        """
        self._w(f"HOLDtype {_token(hold, SequenceHold, 'hold')}")
        return self.get_hold()

    def get_hold(self) -> SequenceHold:
        """The level after the sequence ends."""
        return _member(self._q("HOLDtype"), SequenceHold, "hold")

    def set_hold_user_level(self, volts: float) -> float:
        """Hold ``volts`` after the sequence ends; selects the USER hold."""
        self._w("HOLDtype USER")
        self._w(f"USRHOLD {_num(volts)}")
        return self.get_hold_user_level()

    def get_hold_user_level(self) -> float:
        """The USER hold level in volts."""
        return float(self._q("USRHOLD"))

    def cmd_save(self, name: str, *, with_waveforms: bool = True) -> None:
        """Save the sequence page as ``Local/<name>.AWGx``.

        The instrument refuses a save while the sequence plays (-200, MEAS
        2026-10-17), so this stops it first and restarts it after.
        """
        running = self.get_run_state()
        self.set_run_state(False)
        swfs = "TRUE" if with_waveforms else "FALSE"
        self._w(f"SAVE PATH,{quote(f'Local/{name}.AWGx')},SWFS,{quote(swfs)}")
        if running:
            self.set_run_state(True)

    def cmd_recall(self, name: str) -> None:
        """Load ``Local/<name>.AWGx``; the sequence stays stopped.

        The instrument refuses a load while a sequence plays (-200, MEAS
        2026-10-17), so this stops it first.
        """
        self._enter()
        self.set_run_state(False)
        self._w(f'LOAD "Local/{name}.AWGx"')

    def get_setup(self) -> SequenceState:
        """The sequence page; the segments only while stopped (SDG-05)."""
        running = self.get_run_state()
        return SequenceState(
            running=running,
            segments=None if running else tuple(self.get_segments()),
            sample_rate=self.get_sample_rate(),
            run_mode=self.get_run_mode(),
            trigger_source=self.get_trigger_source(),
            interpolation=self.get_interpolation(),
            offset=self.get_offset(),
            hold=self.get_hold(),
        )


# ---------------------------------------------------------------------------
# One output channel.
# ---------------------------------------------------------------------------


class _Channel(_SdgNode):
    """One output channel.

    The waveform helpers set the mode, the load, the shape, the amplitude
    limit and the levels with the amplitude last, read everything back, and
    raise above a 0.1 percent difference. They never turn the output on.
    """

    def __init__(self, sdg: Siglent_SDG3000X, n: int):
        super().__init__(sdg.link, name=f"CH{n}")
        self._sdg = sdg
        self.n = n
        self._c = f"C{n}"
        self.mod = _Modulation(self)
        self.sweep = _Sweep(self)
        self.burst = _Burst(self)
        self.harmonics = _Harmonics(self)
        self.hopping = _Hopping(self)
        self.multitone = _Multitone(self)
        self.multipulse = _Multipulse(self)
        self.sequence = _Sequence(self)

    # ---------- raw page access ----------

    def _bswv(self) -> dict:
        return _pairs(self.link.query_body(f"{self._c}:BSWV?").split(","), f"{self._c}:BSWV?")

    def _bswv_value(self, key: str) -> float | str:
        found = self._bswv()
        if key not in found:
            raise InstrumentError(
                f"{self.name}: {_BSWV_NAMES.get(key, key)} does not apply to the "
                f"{found.get('WVTP')} shape"
            )
        return found[key]

    def _set_bswv(self, key: str, value: float, *, warn: bool = True) -> float:
        self.link.write_checked(f"{self._c}:BSWV {key},{_num(value)}")
        found = self._bswv()
        if key not in found:
            raise InstrumentError(
                f"{self.name}: {_BSWV_NAMES.get(key, key)} does not apply to the "
                f"{found.get('WVTP')} shape"
            )
        got = found[key]
        if warn and isinstance(got, float) and not _close(value, got):
            warnings.warn(
                ValueSnappedWarning(
                    f"{self.name}: {_BSWV_NAMES.get(key, key)} requested {value:g}, the "
                    f"instrument holds {got:g}; it clamps without an error (SDG-01)"
                ),
                stacklevel=3,
            )
        return _as_float(got)

    def _outp(self) -> dict:
        body = self.link.query_body(f"{self._c}:OUTP?")  # 'OFF,LOAD,HZ,PLRT,NOR' (MEAS)
        tokens = body.split(",")
        found = _pairs(tokens[1:], body)
        load = found.get("LOAD")
        return {
            "state": tokens[0].strip() == "ON",
            "load": math.inf if str(load).upper() == "HZ" else _as_float(load),
            "inverted": str(found.get("PLRT")).upper() == "INVT",
        }

    # ---------- mode ----------

    def get_mode(self) -> ChannelMode:
        """What the channel is doing: basic generator, sequence, IQ, multitone or multipulse."""
        token = self.link.query_body(f"{self._c}:CPARam?").split(",")[-1]
        mode = _member(token, ChannelMode, "channel mode")
        if mode is ChannelMode.SEQUENCE:
            if self.multitone.get_state():
                return ChannelMode.MULTITONE
            if self.multipulse.get_state():
                return ChannelMode.MULTIPULSE
        return mode

    def _afg(self) -> None:
        if self.link.query_body(f"{self._c}:CPARam?") != "MODE,AFG":
            self.link.write_checked(f"{self._c}:CPARam MODE,AFG")

    def _plain(self) -> None:
        """The basic generator, with every other mode and page off.

        Leaving IQ mode turns both outputs off, as the original driver did.
        """
        mode = self.get_mode()
        if mode is ChannelMode.MULTITONE:
            self.multitone.set_state(False)
        elif mode is ChannelMode.MULTIPULSE:
            self.multipulse.set_state(False)
        elif mode is ChannelMode.SEQUENCE:
            self.sequence.set_run_state(False)
        elif mode is ChannelMode.IQ:
            self._sdg.iq.set_state(False)
        self._afg()
        if self.hopping.get_state():
            self.hopping.set_state(False)
        for part in (self.mod, self.sweep, self.burst):
            if part.get_state():
                part.set_state(False)

    # ---------- the output connector ----------

    def set_output_state(self, state: bool) -> bool:
        """Turn this channel's output on or off; returns the read-back.

        Raises if the output reads off after an on, which overvoltage
        protection causes (UM p.99).
        """
        # MEAS 2026-11-05: with a 15 Mpoint sequence loaded, this takes 2.2 s against the 3 s default.
        with self.link.temporary_timeout(_SEQUENCE_RUN_TIMEOUT_MS):
            self.link.write_checked(f"{self._c}:OUTP {_on_off(state)}")
        got = self.get_output_state()
        if got != state:
            hint = " Check the overvoltage protection (UM p.99)." if state else ""
            raise InstrumentError(
                f"{self.name}: the output reads {_on_off(got)} after {_on_off(state)}.{hint}"
            )
        return got

    def get_output_state(self) -> bool:
        """Whether this channel's output is on."""
        return self._outp()["state"]

    @staticmethod
    def _load_ohms(load: float | Load | str) -> float:
        numeric = isinstance(load, (int, float)) and not isinstance(load, bool)
        if numeric:
            if float(load) != 50.0:
                raise ValueError(
                    f"load {load!r}: this driver sets 50 ohm or Load.HIGHZ, the pair the original "
                    "driver checked"
                )
            return 50.0
        _token(load, Load, "load")
        return math.inf

    def set_load(self, load: float | Load | str) -> float:
        """Tell the generator the far end is 50 ohm or high impedance; returns ohms, inf for HIGHZ.

        Enum arg: <sdg>.Load.<TAB>

        This changes no hardware. The load scales the voltage the instrument
        reports for one amplitude setting (UM p.34).
        """
        ohms = self._load_ohms(load)
        token = "50" if ohms == 50.0 else Load.HIGHZ.value
        self.link.write_checked(f"{self._c}:OUTP LOAD,{token}")
        return self.get_load()

    def get_load(self) -> float:
        """The load the generator assumes, in ohms; inf for high impedance."""
        return self._outp()["load"]

    def set_invert_state(self, state: bool) -> bool:
        """Invert the output about its offset, or not; returns the read-back."""
        self.link.write_checked(f"{self._c}:OUTP PLRT,{'INVT' if state else 'NOR'}")
        return self.get_invert_state()

    def get_invert_state(self) -> bool:
        """Whether the output is inverted."""
        return self._outp()["inverted"]

    def set_amplitude_limit(self, vpp: float) -> float:
        """Set the amplitude limit in Vpp into high impedance; returns the read-back.

        The limit is in high-impedance volts: a 0.6 limit with the load at 50
        ohm clamps a request to 0.3 Vpp, with no error (SDG-34).
        """
        return self._set_bswv("MAX_OUTPUT_AMP", vpp)

    def get_amplitude_limit(self) -> float:
        """The amplitude limit in Vpp into high impedance."""
        return _as_float(self._bswv_value("MAX_OUTPUT_AMP"))

    def set_combine_state(self, state: bool) -> bool:
        """Add the other channel to this output, or not; returns the read-back (UM p.95).

        MEAS: the other channel adds at its 50 ohm level (CH2 0.3 Vpp at load
        HIGHZ added 0.147 Vpp into CH1's 50 ohm).
        """
        self.link.write_checked(f"{self._c}:CMBN {_on_off(state)}")
        return self.get_combine_state()

    def get_combine_state(self) -> bool:
        """Whether the other channel is added to this output."""
        return self.link.query(f"{self._c}:CMBN?").split()[-1] == "ON"

    def set_noise_sum(self, snr_db: float | None) -> float | None:
        """Add Gaussian noise at ``snr_db`` dB signal to noise, or None for off (PG p.21)."""
        if snr_db is None:
            self.link.write_checked(f"{self._c}:NOISE_ADD STATE,OFF")
        else:
            self.link.write_checked(f"{self._c}:NOISE_ADD STATE,ON,RATIO_DB,{_num(snr_db)}")
        got = self.get_noise_sum()
        if (snr_db is None) != (got is None) or (snr_db is not None and not _close(snr_db, got)):
            raise InstrumentError(f"{self.name}: noise sum requested {snr_db}, applied {got}")
        return got

    def get_noise_sum(self) -> float | None:
        """The added noise in dB signal to noise; None while off."""
        found = _pairs(self.link.query_body(f"{self._c}:NOISE_ADD?").split(","))
        return found["RATIO_DB"] if found["STATE"] == "ON" else None

    def set_filter_cutoff(self, hz: float | None) -> float | None:
        """Low-pass the output at ``hz``, 10 to 200 MHz, or None for no filter.

        A cutoff below 10 MHz clamps to 10 MHz, and the 20 MHz setting loses
        1.3 dB at 2 MHz (SDG-17).
        """
        if hz is None:
            self.link.write_checked(f"{self._c}:FILT STATE,OFF")
        else:
            self.link.write_checked(f"{self._c}:FILT STATE,ON")
            self.link.write_checked(f"{self._c}:FILT COFF_FRQ,{_num(hz)}")
        got = self.get_filter_cutoff()
        if (hz is None) != (got is None) or (hz is not None and not _close(hz, got)):
            raise InstrumentError(f"{self.name}: filter cutoff requested {hz}, applied {got}")
        return got

    def get_filter_cutoff(self) -> float | None:
        """The output filter cutoff in Hz; None while off."""
        found = _pairs(self.link.query_body(f"{self._c}:FILT?").split(","))
        return found["COFF_FRQ"] if found["STATE"] == "ON" else None

    def _sync(self) -> list[str]:
        return self.link.query_body(f"{self._c}:SYNC?").split(",")  # 'ON,TYPE,CH1' (MEAS)

    def set_sync_state(self, state: bool) -> bool:
        """Drive the front Trig/Sync OUT from this channel, or stop; returns the read-back.

        One channel holds the connector, so on here turns the other channel's
        sync off. The instrument refuses sync, with no error, while a burst or
        a sweep is on, because they use the connector (SDG-12). In sequence
        mode the sync query gets no reply (SDG-30).
        """
        self.link.write_checked(f"{self._c}:SYNC {_on_off(state)}")
        got = self.get_sync_state()
        if got != state:
            raise InstrumentError(
                f"{self.name}: sync out reads {_on_off(got)} after {_on_off(state)}. A burst or "
                "sweep holds Trig/Sync OUT (SDG-12); use its trigger out instead."
            )
        return got

    def get_sync_state(self) -> bool:
        """Whether this channel drives the front Trig/Sync OUT."""
        return self._sync()[0] == "ON"

    def get_sync_type(self) -> str:
        """What the sync marks, as the instrument reports it; the type is front-panel only."""
        reply = self._sync()
        return reply[2] if len(reply) > 2 else ""

    # ---------- the waveform: shape and atomic settings ----------

    def _get_function(self) -> Function:
        return _member(self._bswv()["WVTP"], Function, "shape")

    def _set_function(self, function: Function | str) -> Function:
        """Select the shape (BSWV WVTP); raises when the instrument ignores it."""
        token = _token(function, Function, "shape")
        self.link.write_checked(f"{self._c}:BSWV WVTP,{token}")
        got = self._get_function()
        if got.value != token:
            raise InstrumentError(
                f"{self.name}: the shape reads {got.name} after {token}. Modulation, sweep or "
                "burst may be on (MEAS)."
            )
        return got

    def set_frequency(self, hz: float) -> float:
        """Set the output frequency in Hz; returns the read-back.

        A justified public atomic (design standard 5.3): a loop that steps
        frequency must not rewrite the others.
        """
        return self._set_bswv("FRQ", hz)

    def get_frequency(self) -> float:
        """The output frequency in Hz."""
        return _as_float(self._bswv_value("FRQ"))

    def set_period(self, seconds: float) -> float:
        """Set the output period in seconds; returns the read-back."""
        return self._set_bswv("PERI", seconds)

    def get_period(self) -> float:
        """The output period in seconds."""
        return _as_float(self._bswv_value("PERI"))

    def set_amplitude(self, vpp: float) -> float:
        """Set the amplitude in Vpp into the set load; returns the read-back."""
        return self._set_bswv("AMP", vpp)

    def get_amplitude(self) -> float:
        """The amplitude in Vpp into the set load."""
        return _as_float(self._bswv_value("AMP"))

    def set_amplitude_vrms(self, vrms: float) -> float:
        """Set the amplitude in Vrms; returns the read-back."""
        return self._set_bswv("AMPVRMS", vrms)

    def get_amplitude_vrms(self) -> float:
        """The amplitude in Vrms."""
        return _as_float(self._bswv_value("AMPVRMS"))

    def set_amplitude_dbm(self, dbm: float) -> float:
        """Set the amplitude in dBm; needs the load at 50 ohm (UM p.34)."""
        if self.get_load() == math.inf:
            raise InstrumentError(f"{self.name}: dBm needs the load at 50 ohm; it is HIGHZ")
        return self._set_bswv("AMPDBM", dbm)

    def get_amplitude_dbm(self) -> float:
        """The amplitude in dBm."""
        return _as_float(self._bswv_value("AMPDBM"))

    def set_offset(self, volts: float) -> float:
        """Set the DC offset in volts; returns the read-back."""
        return self._set_bswv("OFST", volts)

    def get_offset(self) -> float:
        """The DC offset in volts."""
        return _as_float(self._bswv_value("OFST"))

    def set_level_high(self, volts: float) -> float:
        """Set the high level in volts; returns the read-back."""
        return self._set_bswv("HLEV", volts)

    def get_level_high(self) -> float:
        """The high level in volts."""
        return _as_float(self._bswv_value("HLEV"))

    def set_level_low(self, volts: float) -> float:
        """Set the low level in volts; returns the read-back."""
        return self._set_bswv("LLEV", volts)

    def get_level_low(self) -> float:
        """The low level in volts."""
        return _as_float(self._bswv_value("LLEV"))

    def set_phase(self, degrees: float) -> float:
        """Set the phase in degrees; it means something in phase-locked mode (UM p.35)."""
        return self._set_bswv("PHSE", degrees)

    def get_phase(self) -> float:
        """The phase in degrees."""
        return _as_float(self._bswv_value("PHSE"))

    def _set_duty(self, percent: float) -> float:
        """Square and pulse duty cycle in percent (BSWV DUTY)."""
        return self._set_bswv("DUTY", percent)

    def _set_symmetry(self, percent: float) -> float:
        """Ramp symmetry in percent (BSWV SYM)."""
        return self._set_bswv("SYM", percent)

    def _set_width(self, seconds: float) -> float:
        """Pulse positive width in seconds (BSWV WIDTH)."""
        return self._set_bswv("WIDTH", seconds)

    def _set_edge_rise(self, seconds: float) -> float:
        """Pulse rise time 10-90 percent in seconds (BSWV RISE)."""
        return self._set_bswv("RISE", seconds)

    def _set_edge_fall(self, seconds: float) -> float:
        """Pulse fall time 90-10 percent in seconds (BSWV FALL)."""
        return self._set_bswv("FALL", seconds)

    def _set_delay(self, seconds: float) -> float:
        """Pulse delay in seconds (BSWV DLY)."""
        return self._set_bswv("DLY", seconds)

    def _set_noise_stdev(self, volts: float) -> float:
        """Noise standard deviation in volts (BSWV STDEV)."""
        return self._set_bswv("STDEV", volts)

    def _set_noise_mean(self, volts: float) -> float:
        """Noise mean in volts (BSWV MEAN)."""
        return self._set_bswv("MEAN", volts)

    def _set_noise_bandwidth(self, hz: float | None) -> float | None:
        """Noise bandwidth in Hz, None for full (BSWV BANDSTATE, BANDWIDTH). 1 MHz accepted (MEAS)."""
        if hz is None:
            self.link.write_checked(f"{self._c}:BSWV BANDSTATE,OFF")
        else:
            self.link.write_checked(f"{self._c}:BSWV BANDSTATE,ON")
            self._set_bswv("BANDWIDTH", hz)
        return self._get_noise_bandwidth()

    def _get_noise_bandwidth(self) -> float | None:
        found = self._bswv()
        if "BANDSTATE" not in found:
            raise InstrumentError(f"{self.name}: the noise bandwidth applies to the noise shape")
        return found["BANDWIDTH"] if found["BANDSTATE"] == "ON" else None

    # ---------- the waveform helpers ----------

    def _check_limit(
        self,
        caller: str,
        ampl: float,
        ampl_limit: float,
        load: float | Load | str | None,
    ) -> None:
        """Refuse, before any write, an amplitude the limit would clamp (SDG-34)."""
        if ampl_limit <= 0:
            raise ValueError(f"{caller}() refused: ampl_limit must be positive, got {ampl_limit!r}")
        ohms = self._load_ohms(load) if load is not None else self.get_load()
        # The limit is in high-impedance volts: at 50 ohm the largest amplitude
        # is half of it (SDG-34, 0.6 Vpp limit -> 0.3 Vpp at 50 ohm).
        allowed = ampl_limit if ohms == math.inf else ampl_limit / 2
        if ampl > allowed * (1 + 1e-9):
            raise ValueError(
                f"{caller}() refused: {ampl:g} Vpp is above what ampl_limit={ampl_limit:g} allows "
                f"at this load ({allowed:g} Vpp; the limit is in high-impedance volts, SDG-34). "
                "Lower the amplitude or raise ampl_limit."
            )

    def _write_limit(self, ampl_limit: float) -> None:
        got = self._set_bswv("MAX_OUTPUT_AMP", ampl_limit, warn=False)
        if not _close(ampl_limit, got):
            raise InstrumentError(
                f"{self.name}: amplitude limit requested {ampl_limit:g}, applied {got}"
            )

    def _set_waveform(
        self,
        function: Function,
        caller: str,
        values: list[tuple[str, float]],
        *,
        load: float | Load | str | None,
        ampl_limit: float | None = None,
        ampl: float | None = None,
        select=None,
    ) -> None:
        """Set a shape in the safe order and check the read-back.

        Mode first, then the load, the shape, the limit, and the levels with
        the amplitude last (design standard 6.2). ``values`` is in that order.
        """
        if ampl_limit is not None:
            self._check_limit(caller, ampl, ampl_limit, load)
        self._plain()
        if load is not None:
            self.set_load(load)
        self._set_function(function)
        if select is not None:
            select()
        if ampl_limit is not None:
            self._write_limit(ampl_limit)
        for key, value in values:
            self.link.write_checked(f"{self._c}:BSWV {key},{_num(value)}")
        got = self._bswv()
        bad = [
            f"{_BSWV_NAMES[k]}: requested {v:g}, applied {got.get(k)}"
            for k, v in values
            if not _close(v, got.get(k))
        ]
        if bad:
            raise InstrumentError(
                f"{caller}() on {self.name}: "
                + "; ".join(bad)
                + ". The instrument clamps out-of-range values without an error (SDG-01)."
            )

    def set_waveform_sine(
        self,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        offset: float = 0.0,
        load: float | Load | str | None = None,
    ) -> WaveformState:
        """Output a sine at ``freq`` Hz and ``ampl`` Vpp under ``ampl_limit`` Vpp into high-Z.

        ``ampl_limit`` is the amplitude limit in high-impedance volts (SDG-34),
        so at 50 ohm it allows half its value. ``load`` is 50 or Load.HIGHZ,
        or None to keep it.
        """
        self._set_waveform(
            Function.SINE,
            "set_waveform_sine",
            [("FRQ", freq), ("OFST", offset), ("AMP", ampl)],
            load=load,
            ampl_limit=ampl_limit,
            ampl=ampl,
        )
        return self.get_waveform()

    def set_waveform_square(
        self,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        offset: float = 0.0,
        duty: float = 50.0,
        load: float | Load | str | None = None,
    ) -> WaveformState:
        """Output a square at ``freq`` Hz and ``ampl`` Vpp with ``duty`` percent."""
        self._set_waveform(
            Function.SQUARE,
            "set_waveform_square",
            [("FRQ", freq), ("DUTY", duty), ("OFST", offset), ("AMP", ampl)],
            load=load,
            ampl_limit=ampl_limit,
            ampl=ampl,
        )
        return self.get_waveform()

    def set_waveform_ramp(
        self,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        offset: float = 0.0,
        symmetry: float = 50.0,
        load: float | Load | str | None = None,
    ) -> WaveformState:
        """Output a ramp; ``symmetry`` is the percent of the cycle spent rising."""
        self._set_waveform(
            Function.RAMP,
            "set_waveform_ramp",
            [("FRQ", freq), ("SYM", symmetry), ("OFST", offset), ("AMP", ampl)],
            load=load,
            ampl_limit=ampl_limit,
            ampl=ampl,
        )
        return self.get_waveform()

    def set_waveform_pulse(
        self,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        offset: float = 0.0,
        width: float | None = None,
        duty: float | None = None,
        edge_time: float | None = None,
        leading_edge_time: float | None = None,
        trailing_edge_time: float | None = None,
        load: float | Load | str | None = None,
    ) -> WaveformState:
        """Output a pulse train; ``width`` in seconds or ``duty`` in percent, not both.

        ``edge_time`` sets both edges; or set the rise with
        ``leading_edge_time`` and the fall with ``trailing_edge_time``, in
        seconds.
        """
        if width is not None and duty is not None:
            raise ValueError("give width or duty, not both")
        if edge_time is not None and (
            leading_edge_time is not None or trailing_edge_time is not None
        ):
            raise ValueError("give edge_time, or leading/trailing_edge_time, not both")
        rise = edge_time if edge_time is not None else leading_edge_time
        fall = edge_time if edge_time is not None else trailing_edge_time
        values = [("FRQ", freq)]
        if width is not None:
            values.append(("WIDTH", width))
        if duty is not None:
            values.append(("DUTY", duty))
        if rise is not None:
            values.append(("RISE", rise))
        if fall is not None:
            values.append(("FALL", fall))
        values += [("OFST", offset), ("AMP", ampl)]
        self._set_waveform(
            Function.PULSE,
            "set_waveform_pulse",
            values,
            load=load,
            ampl_limit=ampl_limit,
            ampl=ampl,
        )
        return self.get_waveform()

    def set_waveform_dc(
        self, level: float, *, load: float | Load | str | None = None
    ) -> WaveformState:
        """Output a DC level in volts.

        No amplitude limit: the instrument's only clamp is on amplitude, and
        whether it bounds a DC level was never observed  # UNVERIFIED
        """
        self._set_waveform(Function.DC, "set_waveform_dc", [("OFST", level)], load=load)
        return self.get_waveform()

    def set_waveform_noise(
        self,
        stdev: float,
        *,
        mean: float = 0.0,
        bandwidth: float | None = None,
        load: float | Load | str | None = None,
    ) -> NoiseState:
        """Output Gaussian noise of ``stdev`` volts about ``mean``; ``bandwidth`` Hz or None.

        No amplitude limit: whether the limit bounds noise was never observed
        # UNVERIFIED
        """
        self._set_waveform(
            Function.NOISE,
            "set_waveform_noise",
            [("STDEV", stdev), ("MEAN", mean)],
            load=load,
        )
        got = self._set_noise_bandwidth(bandwidth)
        if (bandwidth is None) != (got is None) or (
            bandwidth is not None and not _close(bandwidth, got)
        ):
            raise InstrumentError(
                f"set_waveform_noise() on {self.name}: bandwidth requested {bandwidth}, "
                f"applied {got}"
            )
        return self.get_noise()

    def get_noise(self) -> NoiseState:
        """The noise settings; raises unless the shape is noise."""
        found = self._bswv()
        if found.get("WVTP") != Function.NOISE.value:
            raise InstrumentError(f"{self.name}: the shape is {found.get('WVTP')}, not NOISE")
        return NoiseState(
            stdev=found["STDEV"],
            mean=found["MEAN"],
            bandwidth=found.get("BANDWIDTH") if found.get("BANDSTATE") == "ON" else None,
            output_state=self.get_output_state(),
        )

    def set_waveform_prbs(
        self,
        bit_rate: float,
        ampl: float,
        *,
        offset: float = 0.0,
        length: int = 7,
        edge_time: float | None = None,
        logic_level: LogicLevel | str = LogicLevel.CUSTOM,
        differential: bool = False,
        polynomial: str | None = None,
        seed: int | None = None,
        load: float | Load | str | None = None,
    ) -> PrbsState:
        """Output a PRBS at ``bit_rate`` bit/s and ``ampl`` Vpp; ``length`` is the order, 3 to 32.

        Enum arg: logic_level = <sdg>.LogicLevel.<TAB>

        A logic level other than CUSTOM sets the levels itself and ignores
        ``ampl`` and ``offset``. ``differential`` puts the inverted pattern on
        CH2. ``polynomial`` such as ``"X7+X6"`` (the instrument appends +1)
        and ``seed`` set a custom generator. No amplitude limit: the PRBS page
        reports none, and whether one applies was never observed  # UNVERIFIED
        """
        level = _token(logic_level, LogicLevel, "logic level")
        custom = level == LogicLevel.CUSTOM.value
        self._plain()
        if load is not None:
            self.set_load(load)
        self._set_function(Function.PRBS)
        w = self.link.write_checked
        c = self._c
        w(f"{c}:BSWV LOGICLEVEL,{level}")
        w(f"{c}:BSWV BITRATE,{_num(bit_rate)}")
        w(f"{c}:BSWV LENGTH,{int(length)}")
        if edge_time is not None:
            w(f"{c}:BSWV EDGE,{_num(edge_time)}")
        if custom:
            w(f"{c}:BSWV OFST,{_num(offset)}")
            w(f"{c}:BSWV AMP,{_num(ampl)}")
        w(f"{c}:BSWV DIFFSTATE,{_on_off(differential)}")
        if polynomial is not None:
            w(f"{c}:PRBS:FSTAte ON")
            w(f"{c}:PRBS:FORMula {quote(polynomial)}")
        else:
            w(f"{c}:PRBS:FSTAte OFF")
        if seed is not None:
            w(f"{c}:PRBS:SEED {int(seed)}")
        found = self._bswv()
        wanted = [("BITRATE", bit_rate), ("LENGTH", float(length))]
        if custom:
            wanted += [("AMP", ampl), ("OFST", offset)]
        bad = [
            f"{k}: requested {v:g}, applied {found.get(k)}"
            for k, v in wanted
            if not _close(v, found.get(k))
        ]
        if found.get("DIFFSTATE") != _on_off(differential):
            bad.append(f"differential reads {found.get('DIFFSTATE')}")
        if polynomial is not None:
            got = self._get_prbs_polynomial().replace(" ", "").upper()
            want = polynomial.replace(" ", "").upper()
            if got not in (want, want + "+1"):
                bad.append(f"polynomial reads {got!r}")
        if bad:
            raise InstrumentError(f"set_waveform_prbs() on {self.name}: " + "; ".join(bad))
        return self.get_prbs()

    def _get_prbs_polynomial(self) -> str:
        return self.link.query(f"{self._c}:PRBS:FORMula?").strip().strip('"')

    def get_prbs(self) -> PrbsState:
        """The PRBS settings; raises unless the shape is PRBS."""
        found = self._bswv()
        if found.get("WVTP") != Function.PRBS.value:
            raise InstrumentError(f"{self.name}: the shape is {found.get('WVTP')}, not PRBS")
        return PrbsState(
            bit_rate=found["BITRATE"],
            length=int(found["LENGTH"]),
            amplitude=found["AMP"],
            offset=found["OFST"],
            level_high=found["HLEV"],
            level_low=found["LLEV"],
            edge_time=found["EDGE"],
            logic_level=_member(found["LOGICLEVEL"], LogicLevel, "logic level"),
            differential=found["DIFFSTATE"] == "ON",
            polynomial=self._get_prbs_polynomial(),
            output_state=self.get_output_state(),
        )

    def _set_arb_builtin(self, name: str | int) -> str:
        """Select a built-in arb by name or index (ARWV NAME / INDEX)."""
        if isinstance(name, int):
            self.link.write_checked(f"{self._c}:ARWV INDEX,{name}")
        else:
            self.link.write_checked(f"{self._c}:ARWV NAME,{name}")
        got = self.get_arb_name()
        if isinstance(name, str) and got.lower() != name.lower():
            raise InstrumentError(f"{self.name}: the arb reads {got!r} after {name!r}")
        return got

    def _set_arb_user(self, name: str) -> str:
        """Select a stored user arb as Local/<name>.bin (ARWV NAME).

        The bare name selects nothing and reports no error (MEAS), so the path
        form is used and the selection read back.
        """
        self.link.write_checked(f'{self._c}:ARWV NAME,"Local/{name}.bin"')
        got = self.get_arb_name()
        if got.lower() != f"{name}.bin".lower():
            raise InstrumentError(
                f"{self.name}: the arb reads {got!r} after user arb {name!r}. Is it stored? "
                "See get_arb_names_user()."
            )
        return got

    def _set_arb_file(self, path: str) -> str:
        """Select an arb file by its file-manager path (ARWV NAME, PG p.32).

        The selection must name the file's stem. A plain CSV selects nothing
        with no error; only the EasyWaveX layout loads (SDG-32).
        """
        self.link.write_checked(f"{self._c}:ARWV NAME,{quote(path)}")
        got = self.get_arb_name()
        stem = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if not got.lower().startswith(stem.lower()):
            raise InstrumentError(f"{self.name}: the arb reads {got!r} after file {path!r}")
        return got

    def get_arb_name(self) -> str:
        """The selected arb: a built-in name such as ``StairUp``, or ``name.bin`` for a user arb."""
        found = _pairs(self.link.query_body(f"{self._c}:ARWV?").split(","))
        return str(found.get("NAME"))

    def _set_arb_levels(
        self,
        caller: str,
        freq: float,
        ampl: float,
        ampl_limit: float,
        offset: float,
        load: float | Load | str | None,
        select,
    ) -> WaveformState:
        self._set_waveform(
            Function.ARB,
            caller,
            [("FRQ", freq), ("OFST", offset), ("AMP", ampl)],
            load=load,
            ampl_limit=ampl_limit,
            ampl=ampl,
            select=select,
        )
        return self.get_waveform()

    def set_waveform_arb_builtin(
        self,
        name: str | int,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        offset: float = 0.0,
        load: float | Load | str | None = None,
    ) -> WaveformState:
        """Play a built-in arb, by name such as ``"StairUp"`` or by index, at ``freq`` Hz."""
        return self._set_arb_levels(
            "set_waveform_arb_builtin",
            freq,
            ampl,
            ampl_limit,
            offset,
            load,
            lambda: self._set_arb_builtin(name),
        )

    def set_waveform_arb_user(
        self,
        name: str,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        offset: float = 0.0,
        load: float | Load | str | None = None,
    ) -> WaveformState:
        """Play a user arb stored on the instrument, by name, at ``freq`` Hz."""
        return self._set_arb_levels(
            "set_waveform_arb_user",
            freq,
            ampl,
            ampl_limit,
            offset,
            load,
            lambda: self._set_arb_user(name),
        )

    def set_waveform_arb_file(
        self,
        path: str,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        offset: float = 0.0,
        load: float | Load | str | None = None,
    ) -> WaveformState:
        """Play an arb file such as ``"Local/folder/wave.bin"`` at ``freq`` Hz.

        The instrument converts a .csv, .dat or .mat file to .bin on selection
        (UM p.103-104). A CSV loads only in the EasyWaveX layout (SDG-32); see
        ``file.cmd_write_arb_csv``.
        """
        return self._set_arb_levels(
            "set_waveform_arb_file",
            freq,
            ampl,
            ampl_limit,
            offset,
            load,
            lambda: self._set_arb_file(path),
        )

    def set_waveform_arb(
        self,
        samples: Sequence[float],
        freq: float,
        ampl: float,
        *,
        name: str,
        ampl_limit: float,
        offset: float = 0.0,
        load: float | Load | str | None = None,
    ) -> WaveformState:
        """Store one period of ``samples`` as user arb ``name`` and play it at ``freq`` Hz.

        The samples' peak to peak maps to ``ampl``: the instrument scales the
        data extremes to the amplitude, so two channels with different data
        spans get different scales (SDG-24). The upload works over the raw
        socket as well as VXI-11 (SDG-04 withdrawn).
        """
        values = np.asarray(samples, dtype=np.float64).ravel()
        if len(values) < 2:
            raise ValueError("samples: give at least 2 points")
        self._check_limit("set_waveform_arb", ampl, ampl_limit, load)
        lo, hi = float(values.min()), float(values.max())
        span = (hi - lo) or 1.0
        codes = np.round((values - lo) / span * _ARB_CODE_SPAN + _ARB_CODE_LOW)
        data = codes.astype("<i2").tobytes()
        head = (
            f"{self._c}:WVDT WVNM,{name},FRQ,{_num(freq)},AMP,{_num(ampl)},"
            f"OFST,{_num(offset)},PHASE,0,WAVEDATA,"
        )
        self.link.write_raw(head.encode() + data + b"\n")
        self.link.opc(_UPLOAD_TIMEOUT_S)
        self.link.raise_errors(f"WVDT {name} ({len(values)} points)")
        # The basic-generator reset in the helper also stops a running sweep,
        # which would otherwise sweep the arb rate (MEAS).
        return self._set_arb_levels(
            "set_waveform_arb",
            freq,
            ampl,
            ampl_limit,
            offset,
            load,
            lambda: self._set_arb_user(name),
        )

    def get_waveform(self) -> WaveformState:
        """Read the shape now generated and every parameter that applies to it.

        DC reports its level as the offset. Noise reports its mean as the
        offset and amplitude 0; use ``get_noise``. PRBS reports its bit rate
        as the frequency; use ``get_prbs`` for the rest.
        """
        found = self._bswv()
        output = self.get_output_state()
        function = _member(found["WVTP"], Function, "shape")

        def number(key: str) -> float | None:
            v = found.get(key)
            return v if isinstance(v, float) else None

        if function is Function.DC:
            return WaveformState(
                function=function.value,
                frequency=0.0,
                amplitude=0.0,
                offset=number("OFST") or 0.0,
                output_state=output,
            )
        if function is Function.NOISE:
            return WaveformState(
                function=function.value,
                frequency=0.0,
                amplitude=0.0,
                offset=number("MEAN") or 0.0,
                output_state=output,
                bandwidth=number("BANDWIDTH") if found.get("BANDSTATE") == "ON" else None,
            )
        if function is Function.PRBS:
            return WaveformState(
                function=function.value,
                frequency=number("BITRATE") or 0.0,
                amplitude=number("AMP") or 0.0,
                offset=number("OFST") or 0.0,
                output_state=output,
                edge_time=number("EDGE"),
            )
        pulse = function is Function.PULSE
        return WaveformState(
            function=function.value,
            frequency=number("FRQ") or 0.0,
            amplitude=number("AMP") or 0.0,
            offset=number("OFST") or 0.0,
            output_state=output,
            duty=number("DUTY") if function in (Function.SQUARE, Function.PULSE) else None,
            symmetry=number("SYM") if function is Function.RAMP else None,
            width=number("WIDTH") if pulse else None,
            edge_time=number("RISE") if pulse else None,
            trailing_edge_time=number("FALL") if pulse else None,
            delay=number("DLY") if pulse else None,
        )

    # ---------- modulation ----------

    def _modulate(
        self,
        kind: ModulationType,
        params: list[tuple],
        source: ModulationSource | str,
        shape: ModulationShape | str | None = None,
    ) -> ModulationState:
        token = _token(source, ModulationSource, "modulation source")
        if token == ModulationSource.EXTERNAL.value:
            raise ValueError(
                f"set_{kind.name.lower()}() takes an internal or channel source; use "
                f"set_modulation_external({kind.name}) for the rear MODULATION IN"
            )
        m = self.mod
        m._others_off()
        m.set_state(True)
        m.set_type(kind)
        m.set_source(token)
        if shape is not None and token == ModulationSource.INTERNAL.value:
            m.set_shape(shape)
        bad = []
        for setter, label, value in params:
            got = setter(value)
            if not _close(value, got):
                bad.append(f"{label}: requested {value:g}, applied {got}")
        if bad:
            raise InstrumentError(f"set_{kind.name.lower()}() on {self.name}: " + "; ".join(bad))
        return self.get_modulation()

    def set_am(
        self,
        depth: float,
        *,
        rate: float,
        shape: ModulationShape | str = ModulationShape.SINE,
        source: ModulationSource | str = ModulationSource.INTERNAL,
    ) -> ModulationState:
        """Amplitude-modulate the carrier to ``depth`` percent at ``rate`` Hz, and turn AM on.

        Enum arg: shape  = <sdg>.ModulationShape.<TAB>
                  source = <sdg>.ModulationSource.<TAB>

        Each setting write takes about 110 ms (SDG-23), so SCPI cannot stream
        the depth; use the rear MODULATION IN for that.
        """
        m = self.mod
        return self._modulate(
            ModulationType.AM,
            [(m.set_depth, "depth", depth), (m.set_rate, "rate", rate)],
            source,
            shape,
        )

    def set_dsbsc(
        self,
        *,
        rate: float,
        shape: ModulationShape | str = ModulationShape.SINE,
        source: ModulationSource | str = ModulationSource.INTERNAL,
    ) -> ModulationState:
        """Double-sideband suppressed-carrier modulate at ``rate`` Hz, and turn it on."""
        return self._modulate(
            ModulationType.DSBSC, [(self.mod.set_rate, "rate", rate)], source, shape
        )

    def set_fm(
        self,
        deviation: float,
        *,
        rate: float,
        shape: ModulationShape | str = ModulationShape.SINE,
        source: ModulationSource | str = ModulationSource.INTERNAL,
    ) -> ModulationState:
        """Frequency-modulate by ``deviation`` Hz at ``rate`` Hz, and turn FM on."""
        m = self.mod
        return self._modulate(
            ModulationType.FM,
            [(m.set_rate, "rate", rate), (m.set_deviation, "deviation", deviation)],
            source,
            shape,
        )

    def set_pm(
        self,
        deviation: float,
        *,
        rate: float,
        shape: ModulationShape | str = ModulationShape.SINE,
        source: ModulationSource | str = ModulationSource.INTERNAL,
    ) -> ModulationState:
        """Phase-modulate by ``deviation`` degrees at ``rate`` Hz, and turn PM on.

        SDG-11 (R12, internal and external): the instrument applies the
        deviation to the size of the modulating signal and adds 180 degrees
        while it is negative. At each zero crossing the carrier makes a burst
        of fast cycles, then runs 180 degrees shifted. PM is clean only for a
        modulating signal of one polarity.
        """
        m = self.mod
        return self._modulate(
            ModulationType.PM,
            [(m.set_rate, "rate", rate), (m.set_deviation, "deviation", deviation)],
            source,
            shape,
        )

    def set_pwm(
        self,
        deviation: float,
        *,
        rate: float,
        shape: ModulationShape | str = ModulationShape.SINE,
        source: ModulationSource | str = ModulationSource.INTERNAL,
    ) -> ModulationState:
        """Width-modulate a pulse carrier by ``deviation`` seconds at ``rate`` Hz, and turn PWM on."""
        if self._get_function() is not Function.PULSE:
            raise InstrumentError(
                f"{self.name}: PWM needs a pulse carrier (UM p.70); use set_waveform_pulse first"
            )
        m = self.mod
        return self._modulate(
            ModulationType.PWM,
            [(m.set_rate, "rate", rate), (m.set_deviation, "deviation", deviation)],
            source,
            shape,
        )

    def set_ask(
        self,
        *,
        key_rate: float,
        source: ModulationSource | str = ModulationSource.INTERNAL,
    ) -> ModulationState:
        """Amplitude-shift key at ``key_rate`` Hz, and turn ASK on.

        External keying input: TTL and diode clamped; no negative voltage on
        MODULATION IN in a keying mode (SDG-10).
        """
        return self._modulate(
            ModulationType.ASK, [(self.mod.set_key_rate, "key rate", key_rate)], source
        )

    def set_fsk(
        self,
        hop_frequency: float,
        *,
        key_rate: float,
        source: ModulationSource | str = ModulationSource.INTERNAL,
    ) -> ModulationState:
        """Frequency-shift key between the carrier and ``hop_frequency`` Hz at ``key_rate`` Hz.

        The carrier is the 0 and the hop frequency the 1. External keying
        input: TTL and diode clamped; no negative voltage on MODULATION IN in
        a keying mode (SDG-10).
        """
        m = self.mod
        return self._modulate(
            ModulationType.FSK,
            [
                (m.set_hop_frequency, "hop frequency", hop_frequency),
                (m.set_key_rate, "key rate", key_rate),
            ],
            source,
        )

    def set_psk(
        self,
        *,
        key_rate: float,
        polarity: Polarity | str = Polarity.POSITIVE,
        source: ModulationSource | str = ModulationSource.INTERNAL,
    ) -> ModulationState:
        """Phase-shift key at ``key_rate`` Hz with ``polarity``, and turn PSK on.

        Enum arg: polarity = <sdg>.Polarity.<TAB>

        External keying input: TTL and diode clamped; no negative voltage on
        MODULATION IN in a keying mode (SDG-10).
        """
        self._modulate(ModulationType.PSK, [(self.mod.set_key_rate, "key rate", key_rate)], source)
        self.mod.set_polarity(polarity)
        return self.get_modulation()

    def set_modulation_external(
        self,
        mode: ModulationType | str,
        *,
        deviation: float | None = None,
        hop_frequency: float | None = None,
        polarity: Polarity | str | None = None,
    ) -> ModulationState:
        """Modulate the carrier from the rear MODULATION IN.

        Enum arg: mode = <sdg>.ModulationType.<TAB>

        ``deviation`` is the full-scale deviation for FM (Hz), PM (degrees) or
        PWM (seconds). ``hop_frequency`` is the FSK 1 frequency. ``polarity``
        is for PSK. Analog modes: 10 kohm, +-6 V; AM 16.4 percent per volt, FM
        and PWM reach the deviation at 6 V, PM reaches half of it (SDG-09).
        ASK, FSK and PSK: a TTL key, diode clamped, no negative voltage; 0 to
        3 V keys and +-1 V does not (SDG-10). PM steps 180 degrees at each zero
        crossing of the input (SDG-11). PWM needs a pulse carrier.
        """
        kind = _member(_token(mode, ModulationType, "modulation type"), ModulationType, "type")
        if kind is ModulationType.PWM and self._get_function() is not Function.PULSE:
            raise InstrumentError(f"{self.name}: PWM needs a pulse carrier (UM p.70)")
        m = self.mod
        m._others_off()
        m.set_state(True)
        m.set_type(kind)
        m.set_source(ModulationSource.EXTERNAL)
        bad = []
        for setter, label, value in (
            (m.set_deviation, "deviation", deviation),
            (m.set_hop_frequency, "hop frequency", hop_frequency),
        ):
            if value is not None:
                got = setter(value)
                if not _close(value, got):
                    bad.append(f"{label}: requested {value:g}, applied {got}")
        if polarity is not None:
            m.set_polarity(polarity)
        if bad:
            raise InstrumentError(
                f"set_modulation_external({kind.name}) on {self.name}: " + "; ".join(bad)
            )
        if m.get_source() is not ModulationSource.EXTERNAL:
            raise InstrumentError(f"{self.name}: the modulation source reads {m.get_source().name}")
        return self.get_modulation()

    def get_modulation_input_scale(self) -> ModulationInputScale:
        """What one volt on MODULATION IN does for the present modulation (MEAS, SDG-09).

        AM 16.4 percent per volt (100 percent at 6.1 V); DSB-SC 0.163 of the
        carrier amplitude per volt (full at 6.12 V); FM and PWM reach the set
        deviation at 6 V; PM reaches half the deviation at 6 V (deviation/12
        degrees per volt, one polarity only, SDG-11). ASK, FSK, PSK: a TTL key,
        no scale (SDG-10).
        """
        kind = self.mod.get_type()
        if kind is None:
            return ModulationInputScale(None, None, None, None)
        if kind in _KEYING:
            return ModulationInputScale(
                kind.value,
                None,
                None,
                None,
                "TTL: high >= 2 V, low <= 0.8 V; no negative voltage (SDG-10)",
            )
        if kind is ModulationType.AM:
            return ModulationInputScale(kind.value, 100 / 6.1, "%/V", 6.1)
        if kind is ModulationType.DSBSC:
            return ModulationInputScale(kind.value, 1 / 6.12, "carrier amplitude/V", 6.12)
        deviation = self.mod.get_deviation()
        if kind is ModulationType.PM:
            return ModulationInputScale(kind.value, deviation / 12, "deg/V", 12.0)
        unit = "Hz/V" if kind is ModulationType.FM else "s/V"
        return ModulationInputScale(kind.value, deviation / 6, unit, 6.0)

    def set_modulation_off(self) -> ModulationState:
        """Turn the modulation, the sweep and the burst off, leaving a plain carrier."""
        for part in (self.mod, self.sweep, self.burst):
            if part.get_state():
                part.set_state(False)
        return self.get_modulation()

    def get_modulation(self) -> ModulationState:
        """Which modulation runs on this channel, and its settings.

        ``level`` is AM depth in percent, FM deviation in Hz, PM deviation in
        degrees, PWM deviation in seconds, or the FSK hop frequency in Hz; None
        for DSB-SC, ASK and PSK. ``rate`` is the modulating frequency, or the
        key rate for ASK, FSK and PSK; None with an external source.
        """
        raw = self.mod._raw()
        if raw.get("STATE") != "ON":
            return ModulationState(mode=None)
        kind = _member(raw["TYPE"], ModulationType, "modulation type")
        level_key = {
            ModulationType.AM: "DEPTH",
            ModulationType.FM: "DEVI",
            ModulationType.PM: "DEVI",
            ModulationType.PWM: "DEVI",
            ModulationType.FSK: "HFRQ",
        }.get(kind)
        rate_key = "KFRQ" if kind in _KEYING else "FRQ"
        source = raw.get("SRC", raw.get("TRSR"))
        level = raw.get(level_key) if level_key else None
        rate = raw.get(rate_key)
        return ModulationState(
            mode=kind.value,
            level=level if isinstance(level, float) else None,
            rate=rate if isinstance(rate, float) else None,
            shape=_member(raw["MDSP"], ModulationShape, "shape") if "MDSP" in raw else None,
            source=_member(source, ModulationSource, "source") if source is not None else None,
        )

    # ---------- sweep ----------

    def _sweep_trigger(
        self,
        source: TriggerSource | str,
        out: TriggerOut | str,
        edge: Edge | str,
    ) -> None:
        sw = self.sweep
        got = sw.set_trigger_source(source)
        if got is TriggerSource.EXTERNAL:
            sw.set_trigger_edge(edge)
        else:
            sw.set_trigger_out(out)

    def set_sweep(
        self,
        start: float,
        stop: float,
        time: float,
        *,
        spacing: Spacing | str = Spacing.LINEAR,
        direction: Direction | str = Direction.UP,
        trigger_source: TriggerSource | str = TriggerSource.INTERNAL,
        trigger_out: TriggerOut | str = TriggerOut.OFF,
        trigger_edge: Edge | str = Edge.RISE,
    ) -> SweepState:
        """Sweep from ``start`` to ``stop`` Hz over ``time`` seconds, and turn the sweep on.

        Enum arg: spacing        = <sdg>.Spacing.<TAB>
                  trigger_source = <sdg>.TriggerSource.<TAB>

        ``trigger_edge`` applies to an EXTERNAL source (Trig/Sync IN) and
        ``trigger_out`` to the others (Trig/Sync OUT). The carrier must be a
        sine, square, ramp or arb (PG p.41). The start is written twice,
        because the instrument clamps it to the old stop (SDG-15).
        """
        sw = self.sweep
        sw._need_carrier()
        sw._others_off()
        sw.set_state(True)
        self.link.write_checked(f"{self._c}:SWEep:TYPE FREQ")
        sw.set_frequency_start(start)
        got_stop = sw.set_frequency_stop(stop)
        got_start = sw.set_frequency_start(start)  # again: start clamps to the old stop (SDG-15)
        checks = [
            ("start", start, got_start),
            ("stop", stop, got_stop),
            ("time", time, sw.set_time(time)),
        ]
        sw.set_spacing(spacing)
        sw.set_direction(direction)
        self._sweep_trigger(trigger_source, trigger_out, trigger_edge)
        bad = [f"{n}: requested {r:g}, applied {a}" for n, r, a in checks if not _close(r, a)]
        if bad:
            raise InstrumentError(f"set_sweep() on {self.name}: " + "; ".join(bad))
        return _present(self.get_sweep(), "sweep")

    def _sweep_amplitude_writes(
        self,
        start: float,
        stop: float,
        spacing: AmplitudeSpacing | str,
        steps: int | None,
        direction: Direction | str,
        symmetry: float,
    ) -> list[tuple[str, float, float]]:
        sw = self.sweep
        sw.set_amplitude_start(start)
        got_stop = sw.set_amplitude_stop(stop)
        got_start = sw.set_amplitude_start(start)  # again: start may clamp to the old stop
        sw.set_amplitude_spacing(spacing)
        if steps is not None:
            sw.set_amplitude_steps(steps)
        sw.set_amplitude_direction(direction)
        sw.set_amplitude_symmetry(symmetry)
        return [("amplitude start", start, got_start), ("amplitude stop", stop, got_stop)]

    def set_sweep_amplitude(
        self,
        start: float,
        stop: float,
        time: float,
        *,
        spacing: AmplitudeSpacing | str = AmplitudeSpacing.LINEAR,
        steps: int | None = None,
        direction: Direction | str = Direction.UP,
        symmetry: float = 50.0,
        trigger_source: TriggerSource | str = TriggerSource.INTERNAL,
        trigger_out: TriggerOut | str = TriggerOut.OFF,
        trigger_edge: Edge | str = Edge.RISE,
    ) -> SweepAmplitudeState:
        """Sweep the amplitude from ``start`` to ``stop`` Vpp over ``time`` seconds.

        Enum arg: spacing = <sdg>.AmplitudeSpacing.<TAB>

        The carrier keeps its frequency: turning the sweep on moves the carrier
        to the sweep center (SDG-15), so the helper writes the center to the
        carrier frequency and reads it back. The amplitude limit still clamps.
        """
        sw = self.sweep
        sw._need_carrier()
        carrier = self.get_frequency()
        sw._others_off()
        sw.set_state(True)
        sw.set_type(SweepType.AMPLITUDE)
        sw.set_frequency_center(carrier)
        if not _close(carrier, self.get_frequency()):
            raise InstrumentError(
                f"set_sweep_amplitude() on {self.name}: the carrier reads "
                f"{self.get_frequency()}, requested {carrier:g}"
            )
        checks = self._sweep_amplitude_writes(start, stop, spacing, steps, direction, symmetry)
        checks.append(("time", time, sw.set_time(time)))
        self._sweep_trigger(trigger_source, trigger_out, trigger_edge)
        bad = [f"{n}: requested {r:g}, applied {a}" for n, r, a in checks if not _close(r, a)]
        if bad:
            raise InstrumentError(f"set_sweep_amplitude() on {self.name}: " + "; ".join(bad))
        return _present(self.get_sweep_amplitude(), "amplitude sweep")

    def set_sweep_frequency_amplitude(
        self,
        frequency_start: float,
        frequency_stop: float,
        amplitude_start: float,
        amplitude_stop: float,
        time: float,
        *,
        spacing: Spacing | str = Spacing.LINEAR,
        direction: Direction | str = Direction.UP,
        trigger_source: TriggerSource | str = TriggerSource.INTERNAL,
        trigger_out: TriggerOut | str = TriggerOut.OFF,
        trigger_edge: Edge | str = Edge.RISE,
    ) -> SweepState:
        """Sweep frequency and amplitude together over ``time`` seconds.

        Returns the frequency half; ``get_sweep_amplitude`` reads the other.
        """
        self.set_sweep(
            frequency_start,
            frequency_stop,
            time,
            spacing=spacing,
            direction=direction,
            trigger_source=trigger_source,
            trigger_out=trigger_out,
            trigger_edge=trigger_edge,
        )
        self.sweep.set_type(SweepType.FREQUENCY_AMPLITUDE)
        checks = self._sweep_amplitude_writes(
            amplitude_start, amplitude_stop, AmplitudeSpacing.LINEAR, None, direction, 50.0
        )
        bad = [f"{n}: requested {r:g}, applied {a}" for n, r, a in checks if not _close(r, a)]
        if bad:
            raise InstrumentError(
                f"set_sweep_frequency_amplitude() on {self.name}: " + "; ".join(bad)
            )
        return _present(self.get_sweep(), "sweep")

    def get_sweep(self) -> SweepState | None:
        """The frequency sweep; None while the sweep is off, when the instrument reports none.

        ``hold_time`` is the hold at the stop value and ``return_time`` the
        retrace, both in seconds.
        """
        raw = self.sweep._raw()
        if raw.get("STATE") != "ON":
            return None
        return SweepState(
            enabled=True,
            start=raw["START"],
            stop=raw["STOP"],
            time=raw["TIME"],
            spacing=_member(raw["SWMD"], Spacing, "spacing"),
            hold_time=self.sweep.get_hold_end(),
            return_time=self.sweep.get_return_time(),
        )

    def get_sweep_amplitude(self) -> SweepAmplitudeState | None:
        """The amplitude half of the sweep; None while the sweep is off or sweeps frequency only."""
        sw = self.sweep
        if not sw.get_state():
            return None
        sweep_type = sw.get_type()
        if sweep_type is SweepType.FREQUENCY:
            return None
        return SweepAmplitudeState(
            sweep_type=sweep_type,
            start=sw.get_amplitude_start(),
            stop=sw.get_amplitude_stop(),
            time=sw.get_time(),
            spacing=sw.get_amplitude_spacing(),
            direction=sw.get_amplitude_direction(),
            steps=sw.get_amplitude_steps(),
            symmetry=sw.get_amplitude_symmetry(),
        )

    # ---------- burst ----------

    def set_burst(
        self,
        cycles: float,
        *,
        period: float,
        phase: float = 0.0,
        trigger_source: TriggerSource | str = TriggerSource.INTERNAL,
        trigger_out: TriggerOut | str = TriggerOut.OFF,
        trigger_edge: Edge | str = Edge.RISE,
        count: int | None = None,
        idle: BurstIdle | str | None = None,
    ) -> BurstState:
        """Emit ``cycles`` carrier cycles every ``period`` seconds, or per trigger; burst on.

        Enum arg: trigger_source = <sdg>.TriggerSource.<TAB>
                  idle           = <sdg>.BurstIdle.<TAB>

        ``cycles`` is a whole number or ``math.inf``. With an EXTERNAL or
        MANUAL trigger, ``count`` bursts run per trigger, ``period`` apart
        (SDG-13). ``trigger_edge`` applies to EXTERNAL and ``trigger_out`` to
        the others. ``idle`` is the level between bursts.
        """
        if not (cycles == math.inf or (cycles >= 1 and float(cycles).is_integer())):
            raise ValueError(
                f"set_burst() refused: cycles is a whole number or inf, got {cycles!r}"
            )
        b = self.burst
        b._others_off()
        b.set_state(True)
        b.set_mode(BurstMode.NCYCLE)
        got_cycles = b.set_cycles(cycles)
        got_period = b.set_period(period)
        b.set_phase(phase)
        got_source = b.set_trigger_source(trigger_source)
        if got_source is TriggerSource.EXTERNAL:
            b.set_trigger_edge(trigger_edge)
        else:
            b.set_trigger_out(trigger_out)
        if count is not None and b.set_count(count) != count:
            raise InstrumentError(
                f"set_burst() on {self.name}: count {b.get_count()} (requested {count})"
            )
        if idle is not None:
            b.set_idle(idle)
        if got_cycles != float(cycles) or not _close(period, got_period):
            raise InstrumentError(
                f"set_burst() on {self.name}: cycles {got_cycles} (requested {cycles}), period "
                f"{got_period} (requested {period:g})"
            )
        return _present(self.get_burst(), "burst")

    def set_burst_gated(
        self,
        *,
        polarity: Polarity | str = Polarity.POSITIVE,
        trigger_source: TriggerSource | str = TriggerSource.EXTERNAL,
    ) -> BurstState:
        """Run the carrier while the gate is open at ``polarity``, and turn burst on.

        Enum arg: polarity = <sdg>.Polarity.<TAB>

        After the gate closes the carrier completes its cycle (SDG-14).
        """
        b = self.burst
        b._others_off()
        b.set_state(True)
        b.set_mode(BurstMode.GATED)
        b.set_gate_polarity(polarity)
        b.set_trigger_source(trigger_source)
        return _present(self.get_burst(), "burst")

    def get_burst(self) -> BurstState | None:
        """The burst; None while burst is off, when the instrument reports no settings.

        ``cycles`` is ``math.inf`` for an infinite burst. A value the reply
        does not carry in this mode reads NaN.
        """
        raw = self.burst._raw()
        if raw.get("STATE") != "ON":
            return None

        def number(key: str) -> float:
            v = raw.get(key)
            return v if isinstance(v, float) else math.nan

        return BurstState(
            enabled=True,
            cycles=_Burst._cycles_from(raw["TIME"]) if "TIME" in raw else math.nan,
            period=number("PRD"),
            phase=number("STPS"),
            mode=_member(raw["GATE_NCYC"], BurstMode, "burst mode") if "GATE_NCYC" in raw else None,
            delay=raw.get("DLAY") if isinstance(raw.get("DLAY"), float) else None,
            idle=_member(raw["HOLD"], BurstIdle, "burst idle") if "HOLD" in raw else None,
        )

    # ---------- chirp ----------

    def set_chirp(
        self,
        start: float,
        stop: float,
        duration: float,
        *,
        ampl: float,
        ampl_limit: float,
        offset: float = 0.0,
        spacing: Spacing | str = Spacing.LINEAR,
        repeat: ChirpRepeat | str = ChirpRepeat.CONTINUOUS,
        period: float | None = None,
        trigger_out: TriggerOut | str = TriggerOut.RISE,
        load: float | Load | str | None = None,
    ) -> ChirpState:
        """Chirp a sine from ``start`` to ``stop`` Hz in ``duration`` s on the sweep hardware.

        Enum arg: repeat = <sdg>.ChirpRepeat.<TAB>

        CONTINUOUS runs back to back; PERIOD runs one chirp every ``period``
        seconds; ONCE runs one per ``sweep.cmd_trigger()``; EXTERNAL one per
        Trig/Sync IN edge. The output is not turned on.
        """
        how = _member(_token(repeat, ChirpRepeat, "repeat"), ChirpRepeat, "repeat")
        law = _member(_token(spacing, Spacing, "spacing"), Spacing, "spacing")
        if law is Spacing.STEP:
            raise ValueError("a chirp is LINEAR or LOG")
        if how is ChirpRepeat.PERIOD and (period is None or period < duration):
            raise ValueError("repeat PERIOD needs period >= duration")
        self.set_waveform_sine(start, ampl, ampl_limit=ampl_limit, offset=offset, load=load)
        source = {
            ChirpRepeat.CONTINUOUS: TriggerSource.INTERNAL,
            ChirpRepeat.PERIOD: TriggerSource.INTERNAL,
            ChirpRepeat.ONCE: TriggerSource.MANUAL,
            ChirpRepeat.EXTERNAL: TriggerSource.EXTERNAL,
        }[how]
        sweep = self.set_sweep(
            start, stop, duration, spacing=law, trigger_source=source, trigger_out=trigger_out
        )
        hold = period - duration if how is ChirpRepeat.PERIOD else 0.0
        got = self.sweep.set_hold_end(hold)
        if not _close(hold, got) and not (hold == 0 and got == 0):
            raise InstrumentError(f"set_chirp() on {self.name}: hold {got} s, requested {hold:g} s")
        return ChirpState(
            method="sweep",
            start=sweep.start,
            stop=sweep.stop,
            duration=sweep.time,
            spacing=law,
            repeat=how,
            period=period,
            points=None,
            output_state=self.get_output_state(),
        )

    def set_chirp_arb(
        self,
        start: float,
        stop: float,
        duration: float,
        *,
        ampl: float,
        ampl_limit: float,
        offset: float = 0.0,
        spacing: Spacing | str = Spacing.LINEAR,
        points: int | None = None,
        load: float | Load | str | None = None,
    ) -> ChirpState:
        """Play an exact, phase-continuous chirp computed here as user arb ``chirp_c<n>``.

        ``points`` defaults to 20 samples per period of the higher frequency,
        held between 2**14 and 2**20. The output is not turned on.
        """
        law = _member(_token(spacing, Spacing, "spacing"), Spacing, "spacing")
        if law is Spacing.STEP:
            raise ValueError("a chirp is LINEAR or LOG")
        n = points or int(min(max(20 * max(start, stop) * duration, 2**14), 2**20))
        samples = []
        for i in range(n):
            t = i * duration / n
            if law is Spacing.LINEAR:
                ph = 2 * math.pi * (start * t + (stop - start) * t * t / (2 * duration))
            else:
                k = (stop / start) ** (1 / duration)
                ph = 2 * math.pi * start * (k**t - 1) / math.log(k)
            samples.append(math.sin(ph))
        self.set_waveform_arb(
            samples,
            1 / duration,
            ampl,
            name=f"chirp_{self._c.lower()}",
            ampl_limit=ampl_limit,
            offset=offset,
            load=load,
        )
        return ChirpState(
            method="arb",
            start=start,
            stop=stop,
            duration=duration,
            spacing=law,
            repeat=None,
            period=None,
            points=n,
            output_state=self.get_output_state(),
        )


# ---------------------------------------------------------------------------
# Instrument-wide pages.
# ---------------------------------------------------------------------------


class _MultiChannel(_SdgNode):
    """The Multi-Channel page: two-channel tasks (UM p.94-96, PG p.24-26)."""

    def __init__(self, sdg: Siglent_SDG3000X):
        super().__init__(sdg.link, name="multichannel")
        self._sdg = sdg

    def _coup(self) -> dict:
        # 'TRACE,ON,0,FCOUP,...' has a stray 0 after TRACE,ON (MEAS)
        clean: list[str] = []
        for t in self.link.query("G1:COUP?").split(","):
            if len(clean) % 2 == 0 and _value(t) == 0.0 and not t.isalpha():
                continue
            clean.append(t)
        return _pairs(clean)

    def _pair(self, ch2_phase: float) -> ChannelPairState:
        return ChannelPairState(
            phase_locked=self.get_phase_locked_state(),
            ch2_phase=ch2_phase,
            ch1=self._sdg.ch1.get_waveform(),
            ch2=self._sdg.ch2.get_waveform(),
        )

    def set_quadrature(
        self,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        offset: float = 0.0,
        load: float | Load | str | None = None,
    ) -> ChannelPairState:
        """Put a sine on both channels with CH2 90 degrees from CH1, phase locked and aligned."""
        self.set_track_state(False)
        for ch in (self._sdg.ch1, self._sdg.ch2):
            ch.set_waveform_sine(freq, ampl, ampl_limit=ampl_limit, offset=offset, load=load)
        return self.set_phase_offset(90.0)

    def set_differential(
        self,
        freq: float,
        ampl: float,
        *,
        ampl_limit: float,
        function: Function | str = Function.SINE,
        offset: float = 0.0,
        load: float | Load | str | None = None,
    ) -> ChannelPairState:
        """Make CH2 the inverted copy of CH1, phase locked and aligned: sine, square or ramp.

        Enum arg: function = <sdg>.Function.<TAB>
        """
        shape = _member(_token(function, Function, "shape"), Function, "shape")
        helpers = {
            Function.SINE: "set_waveform_sine",
            Function.SQUARE: "set_waveform_square",
            Function.RAMP: "set_waveform_ramp",
        }
        if shape not in helpers:
            raise ValueError("set_differential() takes SINE, SQUARE or RAMP")
        self.set_track_state(False)
        for ch in (self._sdg.ch1, self._sdg.ch2):
            getattr(ch, helpers[shape])(freq, ampl, ampl_limit=ampl_limit, offset=offset, load=load)
        self._sdg.ch2.set_invert_state(True)
        self._sdg.ch1.set_invert_state(False)
        return self.set_phase_offset(0.0)

    def set_phase_offset(self, degrees: float) -> ChannelPairState:
        """Set CH2's phase relative to CH1 in degrees, phase locked and aligned.

        The alignment goes first: aligning after the phase write undoes it (MEAS).
        """
        self.set_phase_locked_state(True)
        self.cmd_phase_sync()
        self._sdg.ch1._set_bswv("PHSE", 0.0)
        got = self._sdg.ch2._set_bswv("PHSE", degrees % 360)
        return self._pair(got)

    def set_coupling(
        self,
        *,
        frequency_ratio: float | None = None,
        frequency_deviation: float | None = None,
        amplitude_ratio: float | None = None,
        amplitude_deviation: float | None = None,
        phase_ratio: float | None = None,
        phase_deviation: float | None = None,
    ) -> CouplingState:
        """Make CH2 follow CH1 per quantity by a ratio or a deviation; None turns one off.

        Deviations are in Hz, Vpp and degrees. Track goes off first.
        """
        self.set_track_state(False)
        for label, state, ratio, ratio_v, dev, dev_v in (
            ("frequency", "FCOUP", "FRAT", frequency_ratio, "FDEV", frequency_deviation),
            ("amplitude", "ACOUP", "ARAT", amplitude_ratio, "ADEV", amplitude_deviation),
            ("phase", "PCOUP", "PRAT", phase_ratio, "PDEV", phase_deviation),
        ):
            if ratio_v is not None and dev_v is not None:
                raise ValueError(f"{label}: give a ratio or a deviation, not both")
            if ratio_v is None and dev_v is None:
                self.link.write_checked(f"G1:COUP {state},OFF")
                continue
            self.link.write_checked(f"G1:COUP {state},ON")
            key, value = (ratio, ratio_v) if ratio_v is not None else (dev, dev_v)
            self.link.write_checked(f"G1:COUP {key},{_num(value)}")
            got = self._coup().get(key)
            if not _close(value, got):
                raise InstrumentError(f"coupling {label}: {key} reads {got}, requested {value}")
        return self.get_coupling()

    def get_coupling(self) -> CouplingState:
        """How CH2 follows CH1, per quantity."""
        found = self._coup()
        values: dict = {}
        for label, state, ratio, dev in (
            ("frequency", "FCOUP", "FRAT", "FDEV"),
            ("amplitude", "ACOUP", "ARAT", "ADEV"),
            ("phase", "PCOUP", "PRAT", "PDEV"),
        ):
            if found.get(state) != "ON":
                continue
            if ratio in found:
                values[f"{label}_ratio"] = found[ratio]
            else:
                values[f"{label}_deviation"] = found.get(dev)
        return CouplingState(**values)

    def cmd_copy_ch1_to_ch2(self) -> WaveformState:
        """Copy every CH1 parameter to CH2, once (PG p.26); returns CH2's waveform."""
        self.link.write_checked("PACP C2,C1")
        return self._sdg.ch2.get_waveform()

    def cmd_copy_ch2_to_ch1(self) -> WaveformState:
        """Copy every CH2 parameter to CH1, once (PG p.26); returns CH1's waveform."""
        self.link.write_checked("PACP C1,C2")
        return self._sdg.ch1.get_waveform()

    def cmd_phase_sync(self) -> None:
        """Align the phase of both channels now (PG p.24).

        This also copies the CH2 phase setting to CH1 (SDG-38): set a phase
        offset after this call, never before.
        """
        self.link.write_checked("G1:EQPHASE")

    def set_phase_locked_state(self, state: bool) -> bool:
        """Lock the two channels' phase, or run them independently; returns the read-back."""
        self.link.write_checked(f"G1:MODE {'PHASELOCKED' if state else 'INDEPENDENT'}")
        return self.get_phase_locked_state()

    def get_phase_locked_state(self) -> bool:
        """Whether the two channels are phase locked."""
        reply = self.link.query("G1:MODE?").split()[-1].upper()  # PHASE-LOCKED (MEAS)
        return reply.replace("-", "") == "PHASELOCKED"

    def set_track_state(self, state: bool) -> bool:
        """Make CH2 copy CH1 continuously, or stop; this forces phase lock (UM p.96)."""
        self.link.write_checked(f"G1:COUP TRACE,{_on_off(state)}")
        return self.get_track_state()

    def get_track_state(self) -> bool:
        """Whether CH2 copies CH1 continuously."""
        return self._coup().get("TRACE") == "ON"


class _Iq(_SdgNode):
    """IQ mode: I on CH1, Q on CH2, one set of settings (UM p.58-63, PG p.55-59).

    Every IQ level assumes a 50 ohm load on both outputs at any load setting;
    into high impedance the level is 6 dB higher (SDG-22). Whether the
    amplitude limit applies in IQ mode was never observed  # UNVERIFIED
    """

    def __init__(self, sdg: Siglent_SDG3000X):
        super().__init__(sdg.link, name="iq")
        self._sdg = sdg

    def _q(self, part: str) -> str:
        return self.link.query(f"IQ:{part}?").strip('"')

    def _w(self, part: str) -> None:
        self.link.write_checked(f"IQ:{part}")

    def _num_q(self, part: str) -> float:
        return float(_value(self._q(part)))

    def _enter(self) -> None:
        for ch in (self._sdg.ch1, self._sdg.ch2):
            if ch.get_mode() not in (ChannelMode.AFG, ChannelMode.IQ):
                ch._plain()
        self.link.write_checked("C1:CPARam MODE,IQ")
        if self.link.query_body("C2:CPARam?") != "MODE,IQ":
            raise InstrumentError("CH2 did not follow CH1 into IQ mode")

    def _load(
        self,
        command: str,
        symbol_rate: float,
        vrms: float,
        carrier: float,
        trigger_source: PatternTriggerSource | str,
    ) -> IqState:
        self._enter()
        self._w(command)
        self.set_symbol_rate(symbol_rate)
        self.set_carrier_frequency(carrier)
        got = self.set_amplitude_vrms(vrms)
        if not _close(vrms, got):
            raise InstrumentError(f"IQ amplitude {got}, requested {vrms:g}")
        self.set_trigger_source(trigger_source)
        return self.get_setup()

    def set_waveform_builtin(
        self,
        waveform: IqWaveform | str,
        *,
        symbol_rate: float,
        vrms: float,
        carrier: float = 0.0,
        trigger_source: PatternTriggerSource | str = PatternTriggerSource.INTERNAL,
    ) -> IqState:
        """Play a built-in IQ waveform at ``symbol_rate`` symbols/s and ``vrms`` Vrms.

        Enum arg: waveform = <sdg>.IqWaveform.<TAB>

        ``carrier`` 0 gives baseband I and Q; above 0 an IF signal (UM p.60).
        Levels assume 50 ohm loads at any load setting (SDG-22). The outputs
        are not turned on.
        """
        token = _token(waveform, IqWaveform, "IQ waveform")
        state = self._load(
            f"WAVEload:BUILtin {quote(token)}", symbol_rate, vrms, carrier, trigger_source
        )
        if state.waveform != token:
            raise InstrumentError(f"IQ waveform reads {state.waveform!r}, requested {token!r}")
        return state

    def set_waveform_user(
        self,
        path: str,
        *,
        symbol_rate: float,
        vrms: float,
        carrier: float = 0.0,
        trigger_source: PatternTriggerSource | str = PatternTriggerSource.INTERNAL,
    ) -> IqState:
        """Play a user IQ file made by EasyIQ: ``"Local/x.arb"``, or a bare name in Local.

        The file must exist, and the driver never writes one: the format is
        undocumented (SDG-21).
        """
        if "/" not in path:
            path = f"Local/{path if path.lower().endswith('.arb') else path + '.arb'}"
        if self._sdg.file.get_size(path) is None:
            raise InstrumentError(f"no IQ file {path!r} on the instrument; make it with EasyIQ")
        return self._load(
            f"WAVEload:USERstored {quote(path)}", symbol_rate, vrms, carrier, trigger_source
        )

    def set_state(self, state: bool) -> bool:
        """Put both channels in IQ mode, or back to the basic generator with outputs off."""
        if state:
            self._enter()
            return self.get_state()
        self._sdg.ch1.set_output_state(False)
        self._sdg.ch2.set_output_state(False)
        self.link.write_checked("C1:CPARam MODE,AFG")
        if self.link.query_body("C2:CPARam?") != "MODE,AFG":
            self.link.write_checked("C2:CPARam MODE,AFG")
        return self.get_state()

    def get_state(self) -> bool:
        """Whether the channels are in IQ mode."""
        return self.link.query_body("C1:CPARam?") == "MODE,IQ"

    def cmd_trigger(self) -> None:
        """Send the IQ pattern a manual trigger.

        The command is not in the PG and the original driver never checked it
        # UNVERIFIED
        """
        self._w("TRIGger:IMMediate")

    def get_waveform_info(self) -> IqWaveformInfo:
        """What the instrument reports about the loaded IQ waveform."""
        tokens = self._q("WAVeinfo").split(",")  # 'WAVE_INFO,SYMBOL_LENTH,1024,...' (MEAS)
        raw = _pairs(tokens[1:])
        return IqWaveformInfo(
            symbol_length=raw.get("SYMBOL_LENTH"),
            oversampling=raw.get("OVER_SAMPLING"),
            modulation=raw.get("MODULATION"),
            filter_type=raw.get("FILTER_TYPE"),
            filter_alpha=raw.get("FILTER_ALPHA"),
        )

    def get_sample_rate_limits(self) -> SampleRateLimits:
        """The IQ sample-rate range in Sa/s."""
        found = _pairs(self._q("FrequencySamplingLimit").split(","))
        return SampleRateLimits(minimum=float(found["MIN"]), maximum=float(found["MAX"]))

    def set_carrier_frequency(self, hz: float) -> float:
        """Set the IQ carrier frequency in Hz; 0 gives baseband."""
        self._w(f"CENTerfreq {_num(hz)}")
        return self.get_carrier_frequency()

    def get_carrier_frequency(self) -> float:
        """The IQ carrier frequency in Hz."""
        return self._num_q("CENTerfreq")

    def set_symbol_rate(self, rate: float) -> float:
        """Set the symbol rate in symbols per second."""
        self._w(f"SYMBolrate {_num(rate)}")
        return self.get_symbol_rate()

    def get_symbol_rate(self) -> float:
        """The symbol rate in symbols per second."""
        return self._num_q("SYMBolrate")

    def set_sample_rate(self, rate: float) -> float:
        """Set the IQ sample rate in Sa/s."""
        self._w(f"SAMPlerate {_num(rate)}")
        return self.get_sample_rate()

    def get_sample_rate(self) -> float:
        """The IQ sample rate in Sa/s."""
        return self._num_q("SAMPlerate")

    def set_amplitude_vrms(self, vrms: float) -> float:
        """Set the IQ amplitude sqrt(I^2 + Q^2) in Vrms, 50 ohm terms (SDG-22)."""
        self._w(f"AMPLitude {_num(vrms)}")
        return self.get_amplitude_vrms()

    def get_amplitude_vrms(self) -> float:
        """The IQ amplitude in Vrms."""
        return self._num_q("AMPLitude")

    def set_amplitude_dbm(self, dbm: float) -> float:
        """Set the IQ amplitude in dBm, 50 ohm reference at any load setting; returns dBm.

        MEAS: -10 dBm reads 0.0707 Vrms. Into high impedance the level is 6 dB
        higher (SDG-22).
        """
        self._w(f"AMPLitude {_num(dbm)}dBm")
        return self.get_amplitude_dbm()

    def get_amplitude_dbm(self) -> float:
        """The IQ amplitude in dBm into 50 ohm."""
        v = self.get_amplitude_vrms()
        return 10 * math.log10(v * v / 50 / 1e-3) if v > 0 else -math.inf

    def set_trigger_source(self, source: PatternTriggerSource | str) -> PatternTriggerSource:
        """Choose what starts the IQ pattern.

        Enum arg: <sdg>.PatternTriggerSource.<TAB>
        """
        self._w(f"TRIGger:SOURce {_token(source, PatternTriggerSource, 'trigger source')}")
        return self.get_trigger_source()

    def get_trigger_source(self) -> PatternTriggerSource:
        """What starts the IQ pattern."""
        return _member(self._q("TRIGger:SOURce"), PatternTriggerSource, "trigger source")

    def _adjust(self, key: str, value: float, label: str) -> float:
        """Write a compensation value and read it back; it clamps silently (MEAS, SDG-35)."""
        self._w(f"IQADjustment:{key} {_num(value)}")
        got = self._num_q(f"IQADjustment:{key}")
        if not _close(value, got):
            raise InstrumentError(f"IQ {label}: requested {value}, applied {got} (it clamps)")
        return got

    def set_adjust_gain(self, db: float) -> float:
        """Set the I/Q gain balance in dB; +1 dB moved the I/Q rms ratio by +1.01 dB (MEAS)."""
        return self._adjust("GAIN", db, "gain balance")

    def get_adjust_gain(self) -> float:
        """The I/Q gain balance in dB."""
        return self._num_q("IQADjustment:GAIN")

    def set_adjust_offset_i(self, volts: float) -> float:
        """Set the I offset in volts, 50 ohm terms (SDG-22).

        MEAS: 0.05 V moved the I mean by +50 mV into 50 ohm, and Q by 0.3 mV.
        """
        return self._adjust("IOFFset", volts, "I offset")

    def get_adjust_offset_i(self) -> float:
        """The I offset in volts."""
        return self._num_q("IQADjustment:IOFFset")

    def set_adjust_offset_q(self, volts: float) -> float:
        """Set the Q offset in volts, 50 ohm terms (SDG-22).

        MEAS: 0.05 V moved the Q mean by +99 mV into 1 Mohm, and I by 0.1 mV.
        """
        return self._adjust("QOFFset", volts, "Q offset")

    def get_adjust_offset_q(self) -> float:
        """The Q offset in volts."""
        return self._num_q("IQADjustment:QOFFset")

    def set_adjust_angle_q(self, degrees: float) -> float:
        """Set the Q angle in degrees, -10 to 10; 45 reads back 10 with no error (SDG-35).

        MEAS at 10 degrees: baseband I rms times cos(10 degrees), I-Q
        correlation unchanged; IF 1 MHz carrier phase between CH1 and CH2
        unchanged (89 degrees).
        """
        return self._adjust("QSKew", degrees, "Q angle")

    def get_adjust_angle_q(self) -> float:
        """The Q angle in degrees."""
        return self._num_q("IQADjustment:QSKew")

    def get_setup(self) -> IqState:
        """The whole IQ page."""
        wave = self._q("WAVEload").split()  # 'BUILTIN 16QAM' (MEAS)
        return IqState(
            enabled=self.get_state(),
            source=wave[0].lower(),
            waveform=wave[-1],
            carrier_frequency=self.get_carrier_frequency(),
            symbol_rate=self.get_symbol_rate(),
            sample_rate=self.get_sample_rate(),
            vrms=self.get_amplitude_vrms(),
            trigger_source=self.get_trigger_source(),
            gain=self.get_adjust_gain(),
            offset_i=self.get_adjust_offset_i(),
            offset_q=self.get_adjust_offset_q(),
            angle_q=self.get_adjust_angle_q(),
        )


class _File(_SdgNode):
    """The file system: drives Local, U-disk0 (USB) and net_storage (PG p.85-91).

    Paths are ``<drive>/<folder>/<name>``, such as ``Local/tests/run1.xml``.
    Every action reads the listing back. Copy and move report a false -108
    (SDG-19); the listing decides. A file read goes through a temporary copy;
    an odd-length file loses its last byte, restored as 0x00 (SDG-18).
    """

    DRIVES = ("Local", "U-disk0", "net_storage")

    def __init__(self, sdg: Siglent_SDG3000X):
        super().__init__(sdg.link, name="file")
        self._sdg = sdg

    @staticmethod
    def _split(path: str) -> tuple[str, str]:
        folder, _, name = path.rstrip("/").rpartition("/")
        if not folder:
            raise ValueError(f"path {path!r}: give '<drive>/<name>', drive one of {_File.DRIVES}")
        return folder, name

    def get_catalog(self, folder: str = "Local") -> FileCatalog:
        """List one folder, with the drive's free and used bytes."""
        reply = self.link.query(f"MMEMory:CATalog? {quote(folder)}")
        head = reply.split('"', 1)[0].strip().rstrip(",").split(",")
        entries = []
        for item in re.findall(r'"([^"]*)"', reply):
            parts = item.split(",")
            if len(parts) == 3:
                entries.append(FileEntry(parts[0], False, int(parts[2]), parts[1].lower()))
            else:
                entries.append(FileEntry(parts[0], True, 0, ""))
        return FileCatalog(free=int(head[0]), used=int(head[1]), entries=tuple(entries))

    def get_drives(self) -> list[str]:
        """The drives present: Local always; the others when they list anything (MEAS)."""
        found = ["Local"]
        for drive in self.DRIVES[1:]:
            cat = self.get_catalog(drive)
            if cat.free or cat.used or cat.entries:
                found.append(drive)
        return found

    def get_size(self, path: str) -> int | None:
        """A file's size in bytes, or None when it does not exist."""
        folder, name = self._split(path)
        for entry in self.get_catalog(folder).entries:
            if entry.name == name and not entry.is_directory:
                return entry.size
        return None

    def get_folder_exists(self, path: str) -> bool:
        """Whether a folder exists."""
        folder, name = self._split(path)
        return any(e.name == name and e.is_directory for e in self.get_catalog(folder).entries)

    def get_file(self, path: str) -> bytes:
        """Read a file's bytes; an odd-length file's last byte reads 0x00 (SDG-18)."""
        size = self.get_size(path)
        if size is None:
            raise InstrumentError(f"no file {path!r} on the instrument")
        tmp = f"Local/{_FILE_READ_TMP}.bin"
        if self.get_size(tmp) is not None:
            self.link.write_checked(f"MMEMory:DELete {quote(tmp)}")
        self.link.write_checked(f"MMEMory:COPY {quote(path)},{quote(tmp)}", allow=("-108",))
        try:
            self.link.write(f"WVDT? USER,{_FILE_READ_TMP}")
            raw = self.link.read_raw()
            self.link.drain_errors()  # the read queues a false -108 (SDG-19)
        finally:
            self.link.write_checked(f"MMEMory:DELete {quote(tmp)}")
        if b"WAVEDATA," not in raw:
            raise InstrumentQueryError(f"file read reply starts {raw[:60]!r}")
        data = raw.split(b"WAVEDATA,", 1)[1]
        data = data[:-1] if data.endswith(b"\n") else data
        if len(data) == size - 1:
            data += b"\x00"
        if len(data) != size:
            raise InstrumentError(f"read {len(data)} bytes of {path!r}; the listing says {size}")
        return data

    def cmd_write_arb_csv(
        self,
        path: str,
        samples: Sequence[float],
        *,
        freq: float = 1000.0,
        ampl: float = 2.0,
        offset: float = 0.0,
    ) -> int:
        """Write ``samples`` as an arb CSV in the EasyWaveX layout, the one that loads (SDG-32).

        Header lines (data length, frequency, amp, offset, phase), 7 empty
        lines, ``xpos,value``, then one ``time,value`` row per sample, CR LF.
        The header amplitude applies on selection; a waveform helper then sets
        the level. Returns the size.
        """
        x = [float(v) for v in samples]
        lines = [
            f"data length,{len(x)}",
            f"frequency,{freq:.9f}",
            f"amp,{ampl:.9f}",
            f"offset,{offset:.9f}",
            "phase,0.000000000",
        ]
        lines += [""] * 7 + ["xpos,value"]
        lines += [f"{i / len(x) / freq:.9e},{v:.6f}" for i, v in enumerate(x)]
        return self.cmd_write(path, ("\r\n".join(lines) + "\r\n").encode("ascii"))

    def cmd_write(self, path: str, data: bytes) -> int:
        """Write ``data`` to ``path``; returns the size the listing reports."""
        n = str(len(data))
        self.link.write_raw(
            f"MMEMory:TRANsfer {quote(path)},#{len(n)}{n}".encode() + bytes(data) + b"\n"
        )
        self.link.opc()
        self.link.raise_errors(f"MMEMory:TRANsfer {path}")
        got = self.get_size(path)
        if got != len(data):
            raise InstrumentError(f"{path!r} lists {got} bytes after writing {len(data)}")
        return got

    def cmd_delete(self, path: str) -> None:
        """Delete a file."""
        self.link.write_checked(f"MMEMory:DELete {quote(path)}")
        if self.get_size(path) is not None:
            raise InstrumentError(f"{path!r} is still listed after the delete")

    def cmd_copy(self, source: str, target: str) -> None:
        """Copy a file; the false -108 is ignored and the listing confirms (SDG-19)."""
        self.link.write_checked(f"MMEMory:COPY {quote(source)},{quote(target)}", allow=("-108",))
        if self.get_size(target) is None:
            raise InstrumentError(f"{target!r} is not listed after the copy")

    def cmd_move(self, source: str, target: str) -> None:
        """Move or rename a file; the false -108 is ignored and the listing confirms (SDG-19)."""
        self.link.write_checked(f"MMEMory:MOVE {quote(source)},{quote(target)}", allow=("-108",))
        if self.get_size(target) is None or self.get_size(source) is not None:
            raise InstrumentError(f"the move {source!r} -> {target!r} is not in the listing")

    def cmd_make_directory(self, path: str) -> None:
        """Create a folder."""
        self.link.write_checked(f"MMEMory:MDIRectory {quote(path)}")
        if not self.get_folder_exists(path):
            raise InstrumentError(f"folder {path!r} is not listed after the create")

    def cmd_remove_directory(self, path: str) -> None:
        """Delete an empty folder."""
        self.link.write_checked(f"MMEMory:RDIRectory {quote(path)}")
        if self.get_folder_exists(path):
            raise InstrumentError(f"folder {path!r} is still listed (not empty?)")

    def cmd_save_setup(self, path: str) -> int:
        """Save the whole instrument setup as ``.xml``, like the front-panel Save; returns bytes."""
        if not path.lower().endswith(".xml"):
            path += ".xml"
        self.link.write_checked(f"MMEMory:SAVE:XML {quote(path)}")
        size = self.get_size(path)
        if not size:
            raise InstrumentError(f"setup {path!r} is not listed after the save")
        return size

    def cmd_recall_setup(self, path: str) -> None:
        """Recall a setup saved by ``cmd_save_setup``; the outputs read off afterwards (SDG-20)."""
        if not path.lower().endswith(".xml"):
            path += ".xml"
        if self.get_size(path) is None:
            raise InstrumentError(f"no setup file {path!r} on the instrument")
        self.link.write_checked(f"MMEMory:LOAD:XML {quote(path)}")


class _Counter(_SdgNode):
    """The frequency counter on the rear Counter IN (UM p.90-93, PG p.74-75).

    Input: +-2.5 V DC coupled, 5 Vpp AC coupled, 1 Mohm. The totalizer gate
    goes to the front TRIG/SYNC IN. The firmware differs from the PG:

    - Edge and gate settings sit under TOTalizer, not CONFig (SDG-26).
    - The pause control is inverted (SDG-27); ``set_run_state`` hides it.
    - There is no pulse-width query; the widths are duty times period.
    - About 0.6 measurements per second, and frequent remote queries stall
      the counter (SDG-28). After a change, ``cmd_wait_measurements``, then read.
    """

    _C = "SENSe:COUNTer:"

    def __init__(self, sdg: Siglent_SDG3000X):
        super().__init__(sdg.link, name="counter")

    def _w(self, key: str, value: str) -> None:
        self.link.write_checked(f"{self._C}{key} {value}")

    def _q(self, key: str) -> str:
        return self.link.query(f"{self._C}{key}?").strip()

    def _f(self, key: str) -> float:
        return float(self._q(key))

    def set_state(self, state: bool) -> bool:
        """Turn the counter on or off; returns the read-back."""
        self._w("CONFig:STATE", _on_off(state))
        if self.get_state() != state:
            raise InstrumentError(f"the counter did not turn {_on_off(state).lower()}")
        return state

    def get_state(self) -> bool:
        """Whether the counter is on."""
        return self._q("CONFig:STATE").upper() == "ON"

    def set_run_state(self, state: bool) -> bool:
        """Count, or hold the count and statistics; returns the read-back.

        The firmware inverts its pause control: pause ON counts and pause OFF
        holds (SDG-27). This method speaks the front panel.
        """
        self._w("CONFig:PAUSe", _on_off(state))
        return self.get_run_state()

    def get_run_state(self) -> bool:
        """Whether the counter counts (SDG-27)."""
        return self._q("CONFig:PAUSe").upper() == "ON"

    def cmd_clear(self) -> None:
        """Clear the statistics and the total."""
        self.link.write_checked(f"{self._C}CLEar")

    def _set_mode(self, mode: CounterMode | str) -> CounterMode:
        """Select FREQUENCY or TOTALIZER (CONFig:MODE)."""
        self._w("CONFig:MODE", _token(mode, CounterMode, "counter mode"))
        return self.get_mode()

    def get_mode(self) -> CounterMode:
        """FREQUENCY or TOTALIZER."""
        return _member(self._q("CONFig:MODE"), CounterMode, "counter mode")

    def set_coupling(self, coupling: Coupling | str) -> Coupling:
        """Choose AC or DC input coupling.

        Enum arg: <sdg>.Coupling.<TAB>
        """
        self._w("CONFig:COUPLing", _token(coupling, Coupling, "coupling"))
        return self.get_coupling()

    def get_coupling(self) -> Coupling:
        """The input coupling."""
        return _member(self._q("CONFig:COUPLing"), Coupling, "coupling")

    def set_hf_reject_state(self, state: bool) -> bool:
        """Turn the high-frequency reject filter on or off; returns the read-back."""
        self._w("CONFig:HFREJect", _on_off(state))
        return self.get_hf_reject_state()

    def get_hf_reject_state(self) -> bool:
        """Whether the high-frequency reject filter is on."""
        return self._q("CONFig:HFREJect").upper() == "ON"

    def set_trigger_level(self, volts: float) -> float:
        """Set the input trigger level in volts; raises beyond 10 mV off."""
        self._w("CONFig:TLEVel", _num(volts))
        got = self.get_trigger_level()
        if abs(got - volts) > 0.01:  # the original driver's check, not an instrument step
            raise InstrumentError(f"counter trigger level reads {got} V after {volts} V")
        return got

    def get_trigger_level(self) -> float:
        """The input trigger level in volts."""
        return self._f("CONFig:TLEVel")

    def set_reference_frequency(self, hz: float) -> float:
        """Set the reference frequency in Hz for ``meas_deviation``."""
        self._w("FREQuency:RFREQuency", _num(hz))
        got = self.get_reference_frequency()
        if abs(got - hz) > 1e-9 * max(hz, 1):
            raise InstrumentError(f"counter reference reads {got} Hz after {hz} Hz")
        return got

    def get_reference_frequency(self) -> float:
        """The reference frequency in Hz."""
        return self._f("FREQuency:RFREQuency")

    def set_frequency_mode(
        self,
        *,
        coupling: Coupling | str = Coupling.DC,
        hf_reject: bool = False,
        trigger_level: float = 0.0,
        reference: float | None = None,
    ) -> CounterState:
        """Count frequency: counter on, settings read back, statistics cleared, counting."""
        self.set_state(True)
        self._set_mode(CounterMode.FREQUENCY)
        self.set_coupling(coupling)
        self.set_hf_reject_state(hf_reject)
        self.set_trigger_level(trigger_level)
        if reference is not None:
            self.set_reference_frequency(reference)
        self.set_run_state(True)
        self.cmd_clear()
        return self.get_setup()

    def set_totalizer_mode(
        self,
        *,
        edge: Edge | str = Edge.RISE,
        trigger_level: float = 0.0,
        hf_reject: bool = False,
        gate: Gate | str = Gate.OFF,
    ) -> CounterState:
        """Count edges: counter on, settings read back, total cleared, counting.

        Enum arg: gate = <sdg>.Gate.<TAB>
        """
        self.set_state(True)
        self._set_mode(CounterMode.TOTALIZER)
        self.set_totalizer_edge(edge)
        self.set_hf_reject_state(hf_reject)
        self.set_trigger_level(trigger_level)
        self.set_totalizer_gate(gate)
        self.set_run_state(True)
        self.cmd_clear()
        return self.get_setup()

    def cmd_wait_measurements(
        self,
        n: int = 1,
        *,
        timeout: float = _COUNTER_TIMEOUT_S,
        poll: float = _COUNTER_POLL_S,
    ) -> int:
        """Block until the statistics hold at least ``n`` measurements; returns the count.

        The counter makes about 0.6 measurements per second, and remote
        queries stall it: 10 queries per second gave none in 20 s (SDG-28). So
        this polls slowly; do not query the counter in a tight loop.
        """
        start = time.monotonic()
        while True:
            got = int(self._f("FREQuency:SNUMber"))
            if got >= n:
                return got
            if time.monotonic() - start > timeout:
                raise InstrumentTimeoutError(
                    f"the counter holds {got} of {n} measurements after {timeout} s"
                )
            time.sleep(poll)

    def meas_frequency(self) -> float:
        """The latest frequency in Hz; call ``cmd_wait_measurements`` first after a change."""
        return self._f("FREQuency")

    def meas_period(self) -> float:
        """The latest period in seconds."""
        return self._f("FREQuency:PERiod")

    def meas_duty(self) -> float:
        """The latest duty cycle in percent."""
        return self._f("FREQuency:DUTY")

    def meas_width_positive(self) -> float:
        """The positive pulse width in seconds, from duty times period (no width query)."""
        return self.meas_duty() / 100 * self.meas_period()

    def meas_width_negative(self) -> float:
        """The negative pulse width in seconds, from duty times period (no width query)."""
        return (1 - self.meas_duty() / 100) * self.meas_period()

    def meas_deviation(self) -> float:
        """The deviation from the reference frequency in ppm."""
        return self._f("FREQuency:FDEViation")

    def meas_statistics(self) -> CounterStatistics:
        """The frequency statistics since the last clear."""
        return CounterStatistics(
            mean=self._f("FREQuency:MEAN"),
            maximum=self._f("FREQuency:MAX"),
            minimum=self._f("FREQuency:MIN"),
            sdev=self._f("FREQuency:SDEV"),
            count=int(self._f("FREQuency:SNUMber")),
            duty_mean=self._f("FREQuency:DUTY:MEAN"),
            deviation_mean=self._f("FREQuency:MEAN:FDEViation"),
        )

    def set_totalizer_edge(self, edge: Edge | str) -> Edge:
        """Choose which input edge the totalizer counts.

        Enum arg: <sdg>.Edge.<TAB>
        """
        self._w("TOTalizer:EDGE", _token(edge, Edge, "edge"))
        return self.get_totalizer_edge()

    def get_totalizer_edge(self) -> Edge:
        """Which input edge the totalizer counts."""
        return _member(self._q("TOTalizer:EDGE"), Edge, "edge")

    def set_totalizer_gate(self, gate: Gate | str) -> Gate:
        """Choose when the totalizer counts, gated from front TRIG/SYNC IN.

        Enum arg: <sdg>.Gate.<TAB>

        LEVEL_ counts while the gate sits at that level; AFTER_ counts after
        that gate edge.
        """
        member = _member(_token(gate, Gate, "gate"), Gate, "gate")
        if member is Gate.OFF:
            self._w("TOTalizer:GATE:STATe", "OFF")
        else:
            self._w("TOTalizer:GATE:STATe", "ON")
            if member in (Gate.LEVEL_POSITIVE, Gate.LEVEL_NEGATIVE):
                self._w("TOTalizer:GATE:MODE", "LEVEL")
                polarity = "POSITIVE" if member is Gate.LEVEL_POSITIVE else "NEGATIVE"
                self._w("TOTalizer:GATE:POLarity", polarity)
            else:
                self._w("TOTalizer:GATE:MODE", "AFTER_EDGE")
                self._w("TOTalizer:GATE:EDGE", "RISE" if member is Gate.AFTER_RISE else "FALL")
        got = self.get_totalizer_gate()
        if got is not member:
            raise InstrumentError(f"counter gate reads {got.name} after {member.name}")
        return got

    def get_totalizer_gate(self) -> Gate:
        """When the totalizer counts."""
        if self._q("TOTalizer:GATE:STATe").upper() != "ON":
            return Gate.OFF
        if self._q("TOTalizer:GATE:MODE").upper() == "LEVEL":
            positive = self._q("TOTalizer:GATE:POLarity").upper().startswith("POS")
            return Gate.LEVEL_POSITIVE if positive else Gate.LEVEL_NEGATIVE
        # The edge query fails with -200 in LEVEL mode (SDG-26), so it is read only here.
        rise = self._q("TOTalizer:GATE:EDGE").upper() == "RISE"
        return Gate.AFTER_RISE if rise else Gate.AFTER_FALL

    def meas_total(self) -> int:
        """The edges counted since the last clear."""
        return int(float(self._q("TOTalizer")))

    def get_setup(self) -> CounterState:
        """The whole counter page."""
        mode = self.get_mode()
        common = {
            "enabled": self.get_state(),
            "mode": mode,
            "running": self.get_run_state(),
            "coupling": self.get_coupling(),
            "hf_reject": self.get_hf_reject_state(),
            "trigger_level": self.get_trigger_level(),
        }
        if mode is CounterMode.FREQUENCY:
            return CounterState(**common, reference=self.get_reference_frequency())
        return CounterState(
            **common, edge=self.get_totalizer_edge(), gate=self.get_totalizer_gate()
        )


# ---------------------------------------------------------------------------
# The instrument.
# ---------------------------------------------------------------------------


class Siglent_SDG3000X(Instrument):
    """Siglent SDG3000X family waveform generator: SDG3082X, SDG3162X, SDG3202X.

    ``__init__`` builds the node tree with no I/O: every model has two channels.
    ``connect()`` reads ``*IDN?`` and refuses any other instrument.
    """

    link: _SdgLink

    # The node tree, built in __init__ with no I/O.
    ch1: _Channel
    ch2: _Channel
    multichannel: _MultiChannel
    iq: _Iq
    file: _File
    counter: _Counter

    # Enums live on the class so sdg.Function.<TAB> resolves from the live object.
    Function = Function
    Load = Load
    ChannelMode = ChannelMode
    ModulationType = ModulationType
    ModulationSource = ModulationSource
    ModulationShape = ModulationShape
    Polarity = Polarity
    TriggerSource = TriggerSource
    TriggerOut = TriggerOut
    Edge = Edge
    Spacing = Spacing
    AmplitudeSpacing = AmplitudeSpacing
    SweepType = SweepType
    Direction = Direction
    BurstMode = BurstMode
    BurstIdle = BurstIdle
    LogicLevel = LogicLevel
    HarmonicType = HarmonicType
    ChirpRepeat = ChirpRepeat
    HopType = HopType
    PatternTriggerSource = PatternTriggerSource
    RunMode = RunMode
    SequenceTriggerSource = SequenceTriggerSource
    SequenceEdge = SequenceEdge
    Interpolation = Interpolation
    LengthIncrease = LengthIncrease
    LengthDecrease = LengthDecrease
    SequenceHold = SequenceHold
    WaitEvent = WaitEvent
    IqWaveform = IqWaveform
    Reference = Reference
    CounterMode = CounterMode
    Coupling = Coupling
    Gate = Gate
    Segment = Segment

    def __init__(self, resource_name: str, *, connect_timeout_ms: int = _DEFAULT_TIMEOUT_MS):
        """
        Parameters
        ----------
        resource_name : str
            pyvisa resource for the generator, read off its front panel: VXI-11
            ``TCPIP0::<address>::INSTR`` or the raw socket
            ``TCPIP0::<address>::5025::SOCKET``.
        connect_timeout_ms : int
            I/O timeout for connect and normal operation, in milliseconds.
        """
        super().__init__(resource_name, name="SDG")
        self.link = _SdgLink(resource_name, timeout_ms=connect_timeout_ms)
        self._connect_timeout_ms = connect_timeout_ms
        self.model: str | None = None
        self.serial: str | None = None
        self.firmware: str | None = None
        self.num_channels: int = 0
        self.max_frequency: float | None = None
        self.errors_on_connect: list[str] = []
        # Every model in _MODEL_INFO has two channels (DS p.3), so the whole
        # tree is built here, with no I/O; connect() checks the identity.
        self.ch1 = _Channel(self, 1)
        self.ch2 = _Channel(self, 2)
        self.multichannel = _MultiChannel(self)
        self.iq = _Iq(self)
        self.file = _File(self)
        self.counter = _Counter(self)

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the pinned resource and confirm it is an SDG3082X, SDG3162X or SDG3202X.

        A VISA timeout on the identity query raises :class:`InstrumentUnresponsiveError`
        and is never retried. Another instrument, or an SDG3000X model missing
        from the model table, raises :class:`InstrumentError`. A wrong address
        is fixed by reading the right one off the front panel, never by
        scanning (ADR-051). The error queue is drained into
        ``errors_on_connect``.
        """
        to = self._connect_timeout_ms if timeout_ms is None else timeout_ms
        try:
            self.link.connect(timeout_ms=to)
            idn = self.link.query("*IDN?").strip()
        except visa.VisaIOError as exc:
            if exc.error_code == visa.constants.StatusCode.error_timeout:
                self.link.disconnect()
                raise InstrumentUnresponsiveError(
                    f"{self.resource_name} did not answer *IDN? (VISA timeout). Check the "
                    "address on the front panel and the cable; do not retry in a loop."
                ) from exc
            raise
        match = _IDN_RE.search(idn)
        if not match or match.group(1).upper() not in _MODEL_INFO:
            self.link.disconnect()
            raise InstrumentError(
                f"{self.resource_name} answered {idn!r}, which is not an SDG3082X, SDG3162X or "
                "SDG3202X. Check the resource; this driver does not scan for the instrument. "
                "For a new SDG3000X model, add it to _MODEL_INFO from its data sheet."
            )
        self.idn = idn
        self._on_connect(idn)
        self.errors_on_connect = self.link.drain_errors()
        return idn

    def _on_connect(self, idn: str) -> None:
        """Parse the IDN and record the model facts."""
        self.model = _IDN_RE.search(idn).group(1).upper()
        fields = [f.strip() for f in idn.split(",")]
        self.serial = fields[2] if len(fields) >= 3 else None
        self.firmware = ",".join(fields[3:]) if len(fields) >= 4 else None
        info = _MODEL_INFO[self.model]
        self.num_channels = info["channels"]
        self.max_frequency = info["max_frequency"]

    def safe_state(self) -> None:
        """Turn every output off before the link closes. Best effort; never raises."""
        with contextlib.suppress(Exception):
            self.link.write("OUT_BOTHCH OFF")
        for n in range(1, self.num_channels + 1):
            with contextlib.suppress(Exception):
                self.link.write(f"C{n}:OUTP OFF")
        # Waits for the writes to run before the link closes. Whether the
        # instrument would drop them on a close was never observed  # UNVERIFIED
        with contextlib.suppress(Exception):
            self.link.query("*OPC?")

    def _channels(self) -> list[_Channel]:
        return [self.ch1, self.ch2]

    def channel(self, n: int) -> _Channel:
        """Return the channel node for ``n``, 1 or 2."""
        channels = self._channels()
        if n not in range(1, len(channels) + 1):
            raise ValueError(f"channel must be 1..{len(channels)}, got {n!r}")
        return channels[n - 1]

    # ---------- both outputs ----------

    def set_output_state_all(self, state: bool) -> list[bool]:
        """Turn both outputs on or off in one command, like the front panel; returns each read-back."""
        with self.link.temporary_timeout(_SEQUENCE_RUN_TIMEOUT_MS):
            self.link.write_checked(f"OUT_BOTHCH {_on_off(state)}")
        got = self.get_output_state_all()
        bad = [n for n, on in enumerate(got, 1) if on != state]
        if bad:
            raise InstrumentError(f"outputs {bad} did not turn {_on_off(state).lower()}")
        return got

    def get_output_state_all(self) -> list[bool]:
        """Whether each output is on, in channel order."""
        return [ch.get_output_state() for ch in self._channels()]

    # ---------- the 10 MHz reference and the utility page ----------

    def _rosc(self) -> tuple[str, dict]:
        tokens = self.link.query_body("ROSC?").split(",")  # 'INT,10MOUT,OFF,AMPL,31dBm' (MEAS)
        return tokens[0], _pairs(tokens[1:])

    def set_reference_clock_external(self) -> Reference:
        """Run the timebase from the rear 10MHz IN.

        No query reports the input's presence or lock (SDG-08). Each switch
        without a 10 MHz input flashes a clock message that never reaches the
        error queue (SDG-25, SDG-31).
        """
        self.link.write_checked(f"ROSC {Reference.EXTERNAL.value}")
        return self.get_reference_clock()

    def set_reference_clock_internal(self) -> Reference:
        """Run the timebase from the internal oscillator."""
        self.link.write_checked(f"ROSC {Reference.INTERNAL.value}")
        return self.get_reference_clock()

    def get_reference_clock(self) -> Reference:
        """Which timebase is selected."""
        return _member(self._rosc()[0], Reference, "reference")

    def set_reference_clock_out_state(self, state: bool) -> bool:
        """Drive the rear 10MHz OUT, or stop; returns the read-back."""
        self.link.write_checked(f"ROSC 10MOUT,{_on_off(state)}")
        return self.get_reference_clock_out_state()

    def get_reference_clock_out_state(self) -> bool:
        """Whether the rear 10MHz OUT is driven."""
        return self._rosc()[1].get("10MOUT") == "ON"

    def set_beeper_state(self, state: bool) -> bool:
        """Turn the key beeper on or off; returns the read-back."""
        self.link.write_checked(f"BUZZ {_on_off(state)}")
        return self.get_beeper_state()

    def get_beeper_state(self) -> bool:
        """Whether the key beeper is on."""
        return self.link.query("BUZZ?").split()[-1] == "ON"

    def set_front_panel_keys_state(self, state: bool) -> bool:
        """Let the front-panel keys work, or lock them out; returns the read-back.

        True means the keys work, as the original driver read the setting; that
        polarity was never checked at the panel  # UNVERIFIED
        """
        self.link.write_checked(f"KEY {_on_off(state)}")
        return self.get_front_panel_keys_state()

    def get_front_panel_keys_state(self) -> bool:
        """Whether the front-panel keys work."""
        return self.link.query("KEY?").split()[-1] == "ON"

    def get_lan_address(self) -> str:
        """The LAN IP address the instrument reports."""
        return self.link.query("SYST:COMM:LAN:IPAD?").strip('"')

    def get_arb_names_builtin(self) -> dict[str, int]:
        """The built-in arb names, each with its index."""
        reply = self.link.query("STL? BUILDIN").removeprefix("STL")
        tokens = [t.strip() for t in reply.split(",")]
        found = {
            tokens[i + 1]: int(tokens[i][1:])
            for i in range(0, len(tokens) - 1, 2)
            if tokens[i].startswith("M")
        }
        return dict(sorted(found.items(), key=lambda kv: kv[1]))

    def get_arb_names_user(self) -> list[str]:
        """The user arbs stored on the instrument."""
        reply = self.link.query("STL? USER")
        return [t.strip() for t in reply.split("WVNM", 1)[-1].split(",") if t.strip()]

    # ---------- system ----------

    def get_errors(self) -> list[str]:
        """Drain and return the error queue; an empty list means clean."""
        return self.link.drain_errors()

    def cmd_reset(self) -> None:
        """Return to the default settings without switching the reference clock; outputs off.

        A reset recalls a setup saved right after one factory reset, one file
        per clock source, so it never changes the clock. The factory reset
        itself selects the internal clock (SDG-25), and each switch to external
        without a 10 MHz input flashes a clock message. The first reset for a
        clock source without its file runs the factory reset once, writes the
        clock back, and saves the file. Every setting the original snapshot read
        equalled the factory default after the recall (MEAS 2026-10-15 22:51).
        """
        clock = self.get_reference_clock()
        path = _DEFAULT_SETUP.format(clock=clock.name.lower())
        if self.file.get_size(path) is None:
            self.link.write("*RST")
            self.link.opc(_RESET_TIMEOUT_S)
            self.link.raise_errors("*RST")
            if clock is Reference.EXTERNAL:
                self.set_reference_clock_external()
            self.file.cmd_save_setup(path)
        else:
            self.file.cmd_recall_setup(path)
            self.link.opc(_RESET_TIMEOUT_S)
            self.link.raise_errors("setup recall")
        if self.get_reference_clock() is not clock:
            raise InstrumentError(f"the clock source changed from {clock.name} in the reset")
        self.set_output_state_all(False)

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Save the whole setup to a file on the instrument and read back what can be compared.

        The setup goes to ``Local/drv_snapshot_<time>.xml``, named in
        ``snapshot["setup_file"]``. :meth:`preserved_setup` deletes it when it
        is done; a bare snapshot leaves it. The output states are recorded
        apart, because a recall leaves the outputs off (SDG-20).
        """
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = _SNAPSHOT_SETUP.format(stamp=stamp)
        self.file.cmd_save_setup(path)
        return {
            "setup_file": path,
            "outputs": self.get_output_state_all(),
            "state": self._read_state(),
        }

    def _read_state(self) -> dict:
        """The settings restore() compares: per channel mode and output, and the AFG pages.

        The waveform, modulation, sweep and burst pages are read in the basic
        generator only, as the original driver did; in sequence mode some queries
        hang (SDG-05, SDG-30).
        """
        state: dict = {
            "reference": self.link.query_body("ROSC?"),
            "phase_locked": self.multichannel.get_phase_locked_state(),
            "coupling": self.link.query("G1:COUP?"),
            "channels": {},
        }
        for ch in self._channels():
            mode = ch.get_mode()
            outp = ch._outp()
            entry: dict = {"mode": mode, "load": outp["load"], "inverted": outp["inverted"]}
            if mode is ChannelMode.AFG:
                entry["waveform"] = ch._bswv()
                entry["modulation"] = ch.mod._raw()
                entry["sweep"] = ch.sweep._raw()
                entry["burst"] = ch.burst._raw()
            state["channels"][ch.n] = entry
        return state

    def restore(self, snapshot: dict) -> None:
        """Put back the setup captured by :meth:`snapshot`, outputs last, exactly as found.

        Both outputs go off first. The setup file is recalled, which restored
        every setting the original snapshot compared (SDG-25) and leaves the
        outputs off (SDG-20). The result is read back and compared; on a
        difference this raises with the outputs still off. Whether the recall
        also restores sequence, IQ, hopping and counter settings was never
        observed  # UNVERIFIED
        """
        self.set_output_state_all(False)
        self.file.cmd_recall_setup(snapshot["setup_file"])
        self.link.opc(_RESET_TIMEOUT_S)
        now = self._read_state()
        wanted = snapshot["state"]
        differing = [k for k in ("reference", "phase_locked", "coupling") if now[k] != wanted[k]]
        for n, entry in wanted["channels"].items():
            for key, value in entry.items():
                if now["channels"].get(n, {}).get(key) != value:
                    differing.append(f"CH{n} {key}")
        if differing:
            raise InstrumentError(
                f"restore() could not put back {differing}; the outputs are left off"
            )
        for ch, on in zip(self._channels(), snapshot["outputs"]):
            if on:
                ch.set_output_state(True)

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator[Siglent_SDG3000X]:
        """Snapshot the setup on enter and restore it on exit; deletes the setup file after."""
        state = self.snapshot()
        try:
            yield self
        finally:
            try:
                self.restore(state)
            finally:
                with contextlib.suppress(Exception):
                    self.file.cmd_delete(state["setup_file"])


def _main(argv: list[str] | None = None) -> int:
    """Read-only self-test: identity, both channels' waveform and output, reference, errors.

    Changes no setting. The disconnect at the end turns both outputs off.
    """
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE")
        return 2
    sdg = Siglent_SDG3000X(argv[1])
    print(f"connecting to {sdg.resource_name} ...")
    idn = sdg.connect()
    try:
        print(f"  IDN: {idn}")
        print(f"  model={sdg.model}  firmware={sdg.firmware}  channels={sdg.num_channels}")
        print(f"  errors at connect: {sdg.errors_on_connect or 'none'}")
        for ch in sdg._channels():
            mode = ch.get_mode()
            if mode is ChannelMode.AFG:
                wave = ch.get_waveform()
                print(
                    f"  {ch.name}: {wave.function} {wave.frequency:.12g} Hz {wave.amplitude:g} Vpp "
                    f"offset {wave.offset:g} V, output {'ON' if wave.output_state else 'OFF'}"
                )
            else:
                print(
                    f"  {ch.name}: mode {mode.name}, output "
                    f"{'ON' if ch.get_output_state() else 'OFF'}"
                )
        print(f"  reference: {sdg.get_reference_clock().name}")
        errors = sdg.get_errors()
        print(f"  error queue: {errors or 'clean'}")
    finally:
        sdg.disconnect()
        print("disconnected (outputs off).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
