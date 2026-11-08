# SPDX-License-Identifier: Apache-2.0
# Copyright: see the NOTICE file.
"""Driver for the Rohde & Schwarz RTB2000 oscilloscopes (RTB2002, RTB2004).

Public import path::

    from drivers.rohde_schwarz_rtb2000 import RohdeSchwarz_RTB2000

One model-gated class covers the family. ``__init__`` does no I/O.
``connect()`` reads ``*IDN?`` and ``*OPT?``, then keeps ``ch3`` and ``ch4`` on
the four-channel model only, and keeps each option node only when the option
list shows it: ``afg`` and ``pattern`` need B6, ``logic`` needs B1,
``history`` needs K15, and ``bus`` needs one of K1, K2, K3 or B1, with only the
protocols those options decode.

Port note. This driver was ported from the original repository (commit
8970360, ``rohde_schwarz/rtb2000.py``) onto ``common.Instrument`` and renamed
to ``docs/driver-api-design.md``. Every bench fact cited below was observed in
that repository under its old method names. A rerun under these names on
2026-11-05, RTB2004 firmware 03.000: every ``get_`` with no required argument,
the write-back of each ``set_`` that has a matching ``get_`` (tools/api_sweep.py),
and signal checks of capture, measurements, voltmeter, counter, FFT, record
length, zoom, setup files, screenshot and the built-in generator. Bus decode,
logic, pattern, mask and history were read back only. Quirk IDs (RTB-nn) refer to
``docs/quirks/rtb2000.md``; "UM p.NNN" cites the R&S RTB2 user manual page;
"MEAS" cites an original bench measurement.

Common tasks:
  1. CH1 at 200 mV/div, with a normal-mode edge trigger at 0 V:
         scope.ch1.set_scale(0.2)
         scope.trig.set_edge("ch1", level=0.0)
  2. One acquisition of CH1 and CH2, read to the PC:
         cap = scope.data.meas_waveform(["ch1", "ch2"])
         cap.t, cap.ch1, cap.ch2
  3. The frequency of the signal on CH1:
         hz = scope.meas.frequency("ch1")
  4. Fit CH1 and CH2 to their signals and trigger on CH1 at its mid level:
         scope.set_auto_range(["ch1", "ch2"], trigger_source="ch1")
  5. A screenshot, saved on the PC:
         png = scope.file.get_screenshot("scope.png")
  6. A 10 kHz, 1 Vpp sine from the built-in generator (option B6), then on:
         scope.afg.set_waveform_sine(10e3, 1.0)
         scope.afg.set_output_state(True)

The scope sources nothing except the built-in generator at [Gen Out] and the
pattern generator at P0..P3 (both option B6). ``disconnect()`` turns both off
(:meth:`RohdeSchwarz_RTB2000.safe_state`). Acquisition and channel settings
stay as they are. [Aux Out] keeps its mode through a reset (quirk RTB-11): on
the original bench it carries the 10 MHz reference to the signal generator, and
``safe_state`` leaves it alone.

Every checked write is followed by an operation-complete query and an error
queue drain, as the original driver did, so a refused setting raises at the call
that sent it. Numbers go in as numbers in base units only: V, s, Hz, percent.
Text such as "1k" raises ``TypeError``.

Not in the driver: the scope's own Bode plot (option K36). Once on, it cannot
be left over SCPI, and leaving it resets every scope setting (quirk RTB-01;
removed from the original driver 2026-10-16 at the operator's request). Three original
records were withdrawn as not reproduced: RTB-02, RTB-03 and RTB-05.
"""

from __future__ import annotations

import contextlib
import math
import numbers
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pyvisa as visa

from common.errors import (
    InstrumentError,
    InstrumentQueryError,
    InstrumentTimeoutError,
    InstrumentUnresponsiveError,
)
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import ScpiEnum, member_from_response, quote
from common.state import WaveformState

# original core/instrument.py opened every instrument with a 3000 ms timeout.
_DEFAULT_TIMEOUT_MS = 3000

# model -> capabilities. ``channels`` gates ch3 and ch4. ``bandwidth`` is the
# base-model bandwidth in Hz; option B243 raises it (original RTB2000Spec).
_MODEL_INFO: dict[str, dict] = {
    "RTB2002": {"channels": 2, "bandwidth": 70e6},
    "RTB2004": {"channels": 4, "bandwidth": 70e6},
}

_IDN_RE = re.compile(r"\b(RTB2\d{3})\b", re.IGNORECASE)

# node -> the options that bring it. Any one listed option attaches the node.
# B6 is the function and pattern generator, B1 the logic probes (MSO), K15
# history and segmented memory, K1 SPI and I2C, K2 UART, K3 CAN and LIN
# (original rtb2000.py class docstrings, UM p.284, p.293, p.372, p.475, p.533).
_OPTION_NODES: dict[str, tuple[str, ...]] = {
    "afg": ("B6",),
    "pattern": ("B6",),
    "logic": ("B1",),
    "history": ("K15",),
    "bus": ("K1", "K2", "K3", "B1"),
}

# The scope answers 9.91E37 for an invalid measurement (original rtb2000.py).
_INVALID = 9.9e37
# original _setf default read-back tolerance, relative.
_TOLERANCE = 1e-3
# original core/scpi.py read the error queue at most 50 times.
_ERROR_QUEUE_DRAIN = 50
# original core/instrument.py waited up to 30 s for *RST. UNVERIFIED: no reset
# duration was recorded.
_RESET_TIMEOUT_S = 30.0
# UNVERIFIED: the original driver waited up to 20 s for autoset; no duration was recorded.
_AUTOSET_TIMEOUT_S = 20.0
# UNVERIFIED: the original driver waited up to 10 s for each auto-range acquisition.
_RANGE_ACQUISITION_TIMEOUT_S = 10.0
# Detail queries on an invalid bus frame never answer (quirk RTB-15), so each
# frame query gets 1.5 s (original Bus.get_frames).
_BUS_QUERY_TIMEOUT_MS = 1500
# Each query while acquiring misses more history triggers, so the history wait
# polls every 0.5 s (quirk RTB-10).
_HISTORY_POLL_S = 0.5
# UNVERIFIED: poll interval of the record-length read-back; only the delay
# itself is observed (quirk RTB-08).
_RECORD_LENGTH_POLL_S = 0.05
# UNVERIFIED: poll intervals for results that need an acquisition to fill in
# (quick measurement, counter, search, statistics); no settle was recorded.
_RESULT_POLL_S = 0.2
# UNVERIFIED: voltmeter result poll interval; no settle was recorded.
_DVM_POLL_S = 0.1
# UNVERIFIED: the original driver waited up to 3 s for every quick-measurement result to fill in.
_QUICK_TIMEOUT_S = 3.0
# UNVERIFIED: the original driver gave a logic channel's ASCII read 30 s; no duration was recorded.
_LOGIC_READ_TIMEOUT_MS = 30000
# UNVERIFIED: the original driver ran roll mode 0.5 s past the requested duration before STOP.
_ROLL_MARGIN_S = 0.5
# original clip heuristic: a channel clips when more than 1e-4 of its samples lie
# beyond 4.9 divisions from the offset. Not a bench-measured threshold.
_CLIP_DIVISIONS = 4.9
_CLIP_FRACTION = 1e-4
# Measurement slot the one-shot measurement methods use (original MEAS_SLOT).
_MEAS_SLOT = 8
# The record covers 12 divisions (quirk RTB-04); the reference positions are
# one division from the left, the centre, one division from the right (quirk
# RTB-07, UM p.322: exactly 8.33, 50 and 91.67 percent).
_REFERENCE_PERCENT: dict[str, float] = {"LEFT": 100 / 12, "CENTER": 50.0, "RIGHT": 1100 / 12}


# ---------------------------------------------------------------------------
# Enums. Each value is the SCPI token the driver writes, unless the class says
# otherwise. Read-backs are matched against the token, the member name, or a
# long form the scope answers with.
# ---------------------------------------------------------------------------


class Coupling(ScpiEnum):
    """Analog input coupling. The input is 1 Mohm; a 50 ohm load is wiring."""

    DC = "DCL"
    AC = "ACL"
    GND = "GND"


class Hysteresis(ScpiEnum):
    """Hysteresis of a digitizing threshold."""

    SMALL = "SMAL"
    MEDIUM = "MED"
    LARGE = "LARG"


class TimeReference(ScpiEnum):
    """Horizontal reference point. The value is a label, not a SCPI token (quirk RTB-07)."""

    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class AcquireMode(ScpiEnum):
    """Acquisition mode for all channels. The value is a label, not one SCPI token."""

    SAMPLE = "SAMPLE"
    HIGH_RES = "HIGH_RES"
    PEAK = "PEAK"
    AVERAGE = "AVERAGE"
    ENVELOPE = "ENVELOPE"


class Interpolation(ScpiEnum):
    """How the display fills between samples."""

    SINX = "SINX"
    LINEAR = "LIN"
    SAMPLE_HOLD = "SMHD"


class AcquisitionState(ScpiEnum):
    """Whether the scope acquires, is stopping, has completed, or broke off."""

    RUN = "RUN"
    STOPPING = "STOP"
    COMPLETE = "COMP"
    BREAK = "BRE"


class TriggerMode(ScpiEnum):
    """AUTO free-runs without a trigger; NORMAL waits for one."""

    AUTO = "AUTO"
    NORMAL = "NORM"


class Slope(ScpiEnum):
    """Edge direction for the edge trigger, search and the parallel-bus clock."""

    RISE = "POS"
    FALL = "NEG"
    EITHER = "EITH"


class Edge(ScpiEnum):
    """A rising or a falling edge, where the scope offers no 'either'."""

    RISE = "POS"
    FALL = "NEG"


class TriggerCoupling(ScpiEnum):
    """Edge-trigger coupling."""

    DC = "DC"
    AC = "AC"
    LF_REJECT = "LFR"


class Polarity(ScpiEnum):
    """Pulse polarity for the width and runt trigger, search and the trigger-out pulse."""

    POSITIVE = "POS"
    NEGATIVE = "NEG"


class WidthCondition(ScpiEnum):
    """How a pulse width or time compares with the set value."""

    LONGER = "LONG"
    SHORTER = "SHOR"
    WITHIN = "WITH"
    OUTSIDE = "OUTS"


class TimeoutCondition(ScpiEnum):
    """Whether the timeout trigger waits for the signal to stay high or low."""

    HIGH = "HIGH"
    LOW = "LOW"


class PatternLevel(ScpiEnum):
    """One channel's state in a pattern trigger."""

    HIGH = "1"
    LOW = "0"
    DONT_CARE = "X"


class PatternLogic(ScpiEnum):
    """How a pattern trigger combines the channel states."""

    AND = "AND"
    OR = "OR"


class VideoStandard(ScpiEnum):
    """Video trigger standard (UM p.336)."""

    PAL = "PAL"
    NTSC = "NTSC"
    SECAM = "SEC"
    PAL_M = "PALM"
    I576 = "I576"
    P720 = "P720"
    P1080 = "P1080"
    I1080 = "I1080"


class VideoModulation(ScpiEnum):
    """Video modulation polarity. POSITIVE means negative-going sync pulses."""

    POSITIVE = "POS"
    NEGATIVE = "NEG"


class VideoField(ScpiEnum):
    """Which lines or field the video trigger fires on."""

    ALL_LINES = "ALIN"
    LINE = "LINE"
    EVEN = "EVEN"
    ODD = "ODD"
    ALL = "ALL"


class MeasType(ScpiEnum):
    """Automatic measurement type for a measurement slot (UM p.385-397)."""

    FREQUENCY = "FREQ"
    PERIOD = "PER"
    PEAK_TO_PEAK = "PEAK"
    PEAK_UPPER = "UPE"
    PEAK_LOWER = "LPE"
    HIGH = "HIGH"
    LOW = "LOW"
    AMPLITUDE = "AMPL"
    MEAN = "MEAN"
    RMS = "RMS"
    RISE_TIME = "RTIM"
    FALL_TIME = "FTIM"
    DUTY_POSITIVE = "PDCY"
    DUTY_NEGATIVE = "NDCY"
    WIDTH_POSITIVE = "PPW"
    WIDTH_NEGATIVE = "NPW"
    OVERSHOOT_POSITIVE = "POV"
    OVERSHOOT_NEGATIVE = "NOV"
    DELAY = "DEL"
    PHASE = "PHAS"
    STD_DEV = "STDD"
    CYCLE_MEAN = "CYCM"
    CYCLE_RMS = "CYCR"


class VoltmeterType(ScpiEnum):
    """What a digital voltmeter slot reads (UM p.421)."""

    DC = "DC"
    AC_DC_RMS = "ACDC"
    AC_RMS = "ACRM"


class GridStyle(ScpiEnum):
    """Display grid."""

    LINES = "LIN"
    RETICLE = "RET"
    NONE = "NONE"


class DisplayStyle(ScpiEnum):
    """Whether the waveform is drawn as vectors or dots."""

    VECTORS = "VECT"
    DOTS = "DOTS"


class MathOperation(ScpiEnum):
    """Math waveform operation. The value is the expression template."""

    ADD = "ADD({a},{b})"
    SUBTRACT = "SUB({a},{b})"
    MULTIPLY = "MUL({a},{b})"
    DIVIDE = "DIV({a},{b})"
    INVERT = "INV({a})"
    ABSOLUTE = "ABS({a})"
    SQUARE = "SQR({a})"
    DERIVATIVE = "DERI({a})"
    INTEGRAL = "INTG({a})"


class FftWindow(ScpiEnum):
    """FFT window."""

    RECTANGULAR = "RECT"
    HAMMING = "HAMM"
    HANNING = "HANN"
    BLACKMAN_HARRIS = "BLAC"
    FLATTOP = "FLAT"


class FftScale(ScpiEnum):
    """FFT magnitude scale."""

    LINEAR = "LIN"
    DBM = "DBM"
    DBV = "DBV"
    DBUV = "DBUV"


class BusFormat(ScpiEnum):
    """Number format of the decoded bus words on the screen."""

    HEX = "HEX"
    DECIMAL = "DEC"
    BINARY = "BIN"
    ASCII = "ASC"
    OCTAL = "OCT"


class Parity(ScpiEnum):
    """UART parity."""

    NONE = "NONE"
    EVEN = "EVEN"
    ODD = "ODD"


class IdleLevel(ScpiEnum):
    """Idle level of a UART or LIN line."""

    HIGH = "IDLH"
    LOW = "IDLL"


class UartLine(ScpiEnum):
    """Which UART line a bus trigger watches."""

    RX = "RX"
    TX = "TX"


class UartTriggerMode(ScpiEnum):
    """What the UART bus trigger fires on."""

    BURST_START = "BST"
    START_BIT = "SBIT"
    SYMBOL = "SYMB"
    PATTERN = "PATT"
    PARITY_ERROR = "PRER"
    STOP_ERROR = "SPER"
    BREAK = "BRE"


class CanSignal(ScpiEnum):
    """Which CAN line the probe is on."""

    CAN_H = "CANH"
    CAN_L = "CANL"


class LinStandard(ScpiEnum):
    """LIN protocol version."""

    V1X = "V1X"
    V2X = "V2X"
    J2602 = "J2602"
    AUTO = "AUTO"


class LogicThreshold(ScpiEnum):
    """Logic pod threshold preset: TTL 1.4 V, ECL -1.3 V, CMOS 2.5 V, or a user level."""

    TTL = "TTL"
    ECL = "ECL"
    CMOS = "CMOS"
    USER = "USER"


class LogicActivity(ScpiEnum):
    """Live state of a logic channel since the previous read."""

    LOW = "LOW"
    HIGH = "HIGH"
    TOGGLE = "TOGGLE"


class PatternGenFunction(ScpiEnum):
    """Pattern generator function (UM p.293-298)."""

    SQUARE = "SQU"
    COUNTER = "COUN"
    ARBITRARY = "ARB"
    MANUAL = "MAN"
    SPI = "SPI"
    I2C = "I2C"
    UART = "UART"
    CAN = "CAN"
    LIN = "LIN"
    I2S = "I2S"
    TDM = "TDM"


class CountDirection(ScpiEnum):
    """Counting direction of the pattern generator's 4-bit counter."""

    UP = "UPW"
    DOWN = "DOWN"


class AfgFunction(ScpiEnum):
    """Built-in generator function (UM p.546)."""

    DC = "DC"
    SINE = "SIN"
    SQUARE = "SQU"
    PULSE = "PULS"
    TRIANGLE = "TRI"
    RAMP = "RAMP"
    SINC = "SINC"
    EXPONENTIAL = "EXP"
    ARBITRARY = "ARB"


class Load(ScpiEnum):
    """What the built-in generator assumes its output drives."""

    FIFTY = "R50"
    HIGHZ = "HIGHZ"


class AuxOutSource(ScpiEnum):
    """What drives [Aux Out] (UM p.465). The mode survives a reset (quirk RTB-11)."""

    OFF = "OFF"
    TRIGGER = "TRIG"
    REFERENCE = "REF"
    MASK = "MASK"
    GENERATOR = "GEN"


# Long forms the scope answers with that neither the token nor the member name
# matches (original _Map extra tables).
_SINE_EXTRA = {"SINUSOID": AfgFunction.SINE, "SINUSOIDAL": AfgFunction.SINE}
_PATTERN_EXTRA = {"SQUAREWAVE": PatternGenFunction.SQUARE}
_LOGIC_THRESHOLD_EXTRA = {"MAN": LogicThreshold.USER, "MANUAL": LogicThreshold.USER}
_TRIGGER_SOURCE_EXTRA = {"PATT": "pattern", "NONE": "none", "SBUS1": "bus1", "SBUS2": "bus2"}


# ---------------------------------------------------------------------------
# Records returned by the driver. common.state has no scope records beyond
# TimebaseState and WaveformState; TimebaseState does not fit this scope (its
# position is a percent of the record, the RTB2000 position is seconds), so
# the records below are defined here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelSettings:
    """Every vertical setting of one analog channel."""

    display_state: bool
    """Whether the channel is on."""
    scale: float
    """Volts per division at the probe tip."""
    offset: float
    """Offset in volts."""
    position: float
    """Vertical position in divisions."""
    coupling: Coupling
    bandwidth_limit: bool
    """True when the 20 MHz bandwidth limit is on."""
    invert: bool
    attenuation: float
    """Probe attenuation ratio, for example 10 for a 10:1 probe."""
    type: str
    """Channel type token: SAMP, HRES or PDET."""


@dataclass(frozen=True)
class HorizontalSettings:
    """The horizontal setting. The screen is 12 divisions (quirk RTB-04)."""

    scale: float
    """Seconds per division."""
    position: float
    """Time from the trigger to the reference point, in seconds."""
    reference: TimeReference
    roll_auto: bool
    """Whether automatic roll mode is on."""
    range: float
    """Time across the 12 screen divisions, in seconds."""
    time_acquired: float
    """Time the hardware records; it can exceed the screen (quirk RTB-04)."""


@dataclass(frozen=True)
class AcquisitionSettings:
    """Every acquisition setting."""

    mode: AcquireMode
    record_length: int
    """Points per acquisition."""
    interpolation: Interpolation
    sample_rate: float
    """Display sample rate in samples per second."""
    adc_rate: float
    """Rate of the transferred record in samples per second (quirk RTB-04)."""
    average_count: int
    memory: str
    """Memory mode token, for example DMEM."""


@dataclass(frozen=True)
class TriggerSettings:
    """The A trigger. Fields that do not apply to the type or source are None."""

    type: str
    """Trigger type token as the scope reports it, for example EDGE or WIDT."""
    mode: TriggerMode
    source: str
    """Source in lower case: ch1..ch4, ext, line, d0..d15, pattern, none, bus1."""
    slope: Slope
    holdoff: float | None
    """Holdoff in seconds; None when holdoff is off."""
    level: float | None = None
    """Level in volts for an analog or external source of an edge, width or timeout trigger."""
    coupling: TriggerCoupling | None = None
    pattern: str | None = None
    """Channel pattern of a pattern trigger, for example 10XX."""
    video_standard: VideoStandard | None = None
    video_field: VideoField | None = None


@dataclass(frozen=True)
class ScopeSettings:
    """Every vertical, horizontal, acquisition and trigger setting."""

    channels: dict[str, ChannelSettings]
    """Per channel, keyed ch1..ch4."""
    horizontal: HorizontalSettings
    acquisition: AcquisitionSettings
    trigger: TriggerSettings


@dataclass(frozen=True)
class ZoomSettings:
    """The zoom window."""

    scale: float
    """Seconds per division in the zoom window."""
    time: float
    """Centre of the zoom window, seconds from the trigger."""
    position: float
    """Zoom position as the scope reports it."""


@dataclass(frozen=True)
class MeasSlot:
    """One measurement slot's setup."""

    slot: int
    kind: MeasType
    source: str
    reference: str | None


@dataclass(frozen=True)
class MeasStatistics:
    """Statistics of one measurement over many acquisitions."""

    mean: float
    stddev: float
    minimum: float
    maximum: float
    count: int
    """Acquisitions the statistics cover."""


@dataclass(frozen=True)
class QuickMeasurement:
    """The nine results of the front-panel Quick Measurement, in volts and seconds."""

    peak_to_peak: float
    peak_upper: float
    peak_lower: float
    rms: float
    mean: float
    period: float
    frequency: float
    rise_time: float
    fall_time: float


@dataclass(frozen=True)
class DvmSlot:
    """One digital voltmeter slot's setup."""

    slot: int
    source: str
    kind: VoltmeterType


@dataclass(frozen=True)
class TimeCursors:
    """Two time cursors, seconds from the trigger."""

    t1: float
    t2: float
    delta: float
    """t2 minus t1, in seconds."""
    inverse_delta: float
    """1 / delta, in Hz."""


@dataclass(frozen=True)
class VoltageCursors:
    """Two voltage cursors, in volts."""

    v1: float
    v2: float
    delta: float
    """The voltage difference the scope reports, in volts."""


@dataclass(frozen=True)
class DisplaySettings:
    """Display settings."""

    persistence: float | None
    """Seconds; math.inf for infinite; None when off."""
    grid: GridStyle
    style: DisplayStyle
    intensity_waveform: float
    """Percent."""
    intensity_grid: float
    """Percent."""
    date_time: bool


@dataclass(frozen=True)
class XySettings:
    """The XY display."""

    xy_state: bool
    x: str
    y1: str


@dataclass(frozen=True)
class SearchResult:
    """One search hit (UM p.370)."""

    time: float
    """Seconds from the trigger."""
    type: str
    polarity: str
    value: float
    """Pulse width in seconds for a width search; peak voltage for a peak search."""


@dataclass(frozen=True)
class FftSettings:
    """The scope FFT. Span and RBW follow the time record (quirk RTB-13)."""

    source: str
    window: FftWindow
    scale: FftScale
    start: float
    """Hz."""
    stop: float
    """Hz."""
    rbw: float
    """Resolution bandwidth in Hz."""


@dataclass(frozen=True)
class MaskCounts:
    """Mask test counters."""

    acquisitions: int
    violations: int


@dataclass(frozen=True)
class FileEntry:
    """One file on a scope drive."""

    name: str
    size: int | None
    """Bytes; None for a folder."""


@dataclass(frozen=True)
class FileCatalog:
    """A scope folder listing."""

    used: int
    """Bytes used on the drive."""
    free: int
    """Bytes free on the drive."""
    files: list[FileEntry]


@dataclass(frozen=True)
class BusFrame:
    """One decoded bus frame. Fields the protocol does not have are None or empty.

    Detail fields are filled only for frames whose state is "ok": on any other
    frame the scope does not answer them (quirk RTB-15).
    """

    start: float
    """Seconds from the trigger."""
    state: str
    """Frame state in lower case, for example ok."""
    words: list[int | None] = field(default_factory=list)
    """Data words or bytes as integers."""
    address: int | None = None
    """I2C address as the scope reports it: the 8-bit form (quirk RTB-15)."""
    access: str | None = None
    """I2C access, read or write."""
    identifier: int | None = None
    """CAN or LIN identifier."""
    frame_type: str | None = None
    """CAN frame type."""
    dlc: int | None = None
    """CAN data length code."""
    checksum: int | None = None
    """CAN CRC or LIN checksum."""
    stop: float | None = None
    """End of a parallel-bus word, seconds from the trigger."""


@dataclass(frozen=True)
class ParallelBusSettings:
    """A parallel bus on logic channels."""

    type: str
    width: int
    data: list[str]
    """Logic channel per bit, LSB first."""
    clock: str | None = None
    clock_slope: Slope | None = None


@dataclass(frozen=True)
class UartTriggerSettings:
    """The UART serial-bus trigger."""

    type: str
    mode: UartTriggerMode
    pattern: str
    """Bit pattern in wire order, LSB first per symbol (quirk RTB-16)."""


@dataclass(frozen=True)
class TriggerActions:
    """Actions on each trigger event."""

    enabled: bool
    sound: bool
    reference: bool
    screenshot: bool
    waveform: bool


@dataclass(frozen=True)
class LogicPodSettings:
    """One logic pod's setup."""

    pod: int
    channels: list[int]
    """Logic channels shown, 0..15."""
    threshold: float
    """Threshold in volts."""
    hysteresis: Hysteresis | None


@dataclass(frozen=True)
class PatternSettings:
    """The pattern generator. Fields the function does not use are None."""

    output_state: bool
    function: PatternGenFunction
    frequency: float | None = None
    """Hz."""
    duty: float | None = None
    """Percent."""
    inverted: bool | None = None
    direction: CountDirection | None = None
    values: list[int] | None = None
    sample_time: float | None = None
    """Seconds per arbitrary-pattern value."""
    burst_cycles: int | None = None


@dataclass(frozen=True)
class AfgBurstSettings:
    """A burst of the built-in generator."""

    enabled: bool
    cycles: int
    phase: float
    """Start phase in degrees."""
    continuous: bool
    """True when bursts repeat; False when each waits for cmd_burst_trigger()."""


@dataclass(frozen=True)
class TriggerOutSettings:
    """[Aux Out] as a pulse per trigger event."""

    source: AuxOutSource
    width: float
    """Pulse width in seconds."""
    polarity: Polarity


@dataclass(frozen=True)
class HistorySettings:
    """Segmented memory for the history."""

    count: int
    """History acquisitions the memory holds."""
    record_length: int
    fast_segmentation: bool


@dataclass(frozen=True)
class SignalRange:
    """The span of a signal from one acquisition, in volts."""

    low: float
    high: float


@dataclass(frozen=True)
class Capture:
    """One acquisition on the PC.

    ``t`` is in seconds with the trigger at 0. ``channels`` holds one array per
    source, keyed by name, and each is also an attribute: ``cap.ch1``.
    """

    t: np.ndarray
    channels: dict[str, np.ndarray]
    settings: ScopeSettings
    header: dict[str, list[float]] = field(default_factory=dict)
    """Per source: start time, stop time, points, values per sample."""
    history_index: int | None = None
    """History acquisition index for a history read; None otherwise."""

    def __getattr__(self, name: str) -> np.ndarray:
        channels = self.__dict__.get("channels", {})
        if name in channels:
            return channels[name]
        raise AttributeError(name)

    @property
    def sample_rate(self) -> float:
        """Samples per second, from the time axis."""
        return 1.0 / (self.t[1] - self.t[0])

    def save(self, folder: str | Path) -> Path:
        """Write ``trace.npz`` with t and each channel into ``folder``; returns the path."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "trace.npz"
        np.savez_compressed(path, t=self.t, **self.channels)
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    """Format a number for the instrument at 12 significant digits."""
    return f"{value:.12g}"


def _number(value, label: str) -> float:
    """Return ``value`` as a float; refuse text, so "1k" never reaches the scope."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{label} takes a number in base units (V, s, Hz, percent), got {value!r}. "
            "Text such as '1k' is not accepted; write 1e3."
        )
    return float(value)


def _close(want: float, got: float, rel: float = _TOLERANCE) -> bool:
    return abs(got - want) <= max(rel * abs(want), 1e-12)


_E = TypeVar("_E", bound=ScpiEnum)


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


def _member(value, enum_cls: type[_E], label: str) -> _E:
    """The enum member for a member, a member name, or a token, any case."""
    token = _token(value, enum_cls, label)
    for member in enum_cls:
        if str(member.value) == token:
            return member
    raise ValueError(f"invalid {label} {value!r}")  # pragma: no cover - _token checked it


def _answer(
    response: str, enum_cls: type[_E], label: str, extra: Mapping[str, _E] | None = None
) -> _E:
    """Resolve the scope's answer to a member, including the long forms in ``extra``."""
    text = response.strip().strip('"').upper()
    if extra and text in extra:
        return extra[text]
    return cast(_E, member_from_response(text, enum_cls, label))


def _source(name: str, label: str = "source") -> str:
    """The scope token for a source name such as "ch1", "math1" or "d3"."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} must be a source name such as 'ch1', got {name!r}")
    return name.strip().upper()


def _channel_number(name: str) -> int:
    """The number of an analog channel name, "ch1".."ch4"."""
    match = re.fullmatch(r"ch([1-4])", str(name).strip().lower())
    if not match:
        raise ValueError(f"expected an analog channel 'ch1'..'ch4', got {name!r}")
    return int(match.group(1))


def _logic_number(name) -> int:
    """The number of a logic channel given as 3 or "d3"."""
    text = str(name).strip().lower().removeprefix("d")
    if not text.isdigit() or not 0 <= int(text) <= 15:
        raise ValueError(f"expected a logic channel 0..15 or 'd0'..'d15', got {name!r}")
    return int(text)


def _to_int(text: str) -> int | None:
    """A decoded bus value as an integer: decimal, float or hex text; None when neither."""
    text = text.strip().strip('"')
    try:
        return int(float(text))
    except ValueError:
        try:
            return int(text, 16)
        except ValueError:
            return None


def _drain_errors(link: InstrumentLink) -> list[str]:
    """Read the error queue dry and return what it held, oldest first."""
    errors: list[str] = []
    for _ in range(_ERROR_QUEUE_DRAIN):
        raw = link.query("SYST:ERR?").strip()
        if not raw or raw.split(",", 1)[0].strip().lstrip("+") in ("0", "-0"):
            break
        errors.append(raw)
    return errors


class _RtbLink(InstrumentLink):
    """The pyvisa link, with the original driver's checked write and wait."""

    def connect(self, timeout_ms: int | None = None) -> None:
        rm = visa.ResourceManager()
        instr = self._open_resource(rm, self.resource_name)
        # original core/instrument.py open_resource set "\n" both ways and sent no
        # device clear on open; this keeps both.
        instr.write_termination = "\n"
        instr.read_termination = "\n"
        instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms
        self._instr = instr

    def ask(self, cmd: str) -> str:
        """Query and strip the reply."""
        return self.query(cmd).strip()

    def opc(self, timeout_s: float | None = None) -> None:
        """Wait for operation complete; ``timeout_s`` replaces the I/O timeout for the wait."""
        if timeout_s is None:
            reply = self.ask("*OPC?")
        else:
            with self.temporary_timeout(int(timeout_s * 1000)):
                reply = self.ask("*OPC?")
        if reply != "1":
            raise InstrumentQueryError(f"*OPC? replied {reply!r}, expected '1'")

    def drain_errors(self) -> list[str]:
        return _drain_errors(self)

    def write_checked(self, cmd: str) -> None:
        """Write, wait for completion, and raise when the error queue holds anything."""
        self.write(cmd)
        self.opc()
        errors = self.drain_errors()
        if errors:
            raise InstrumentQueryError(f"the scope reported after {cmd!r}: {'; '.join(errors)}")

    def query_floats(self, cmd: str) -> np.ndarray:
        """Query a REAL,32 little-endian binary block as a float array."""
        values = self.visa.query_binary_values(cmd, datatype="f", is_big_endian=False)
        return np.array(values, dtype=float)


class _RtbNode(Node):
    """Base of every RTB2000 node: the shared link and a reference to the scope."""

    link: _RtbLink

    def __init__(self, scope: RohdeSchwarz_RTB2000, name: str = "", n: int = 0):
        super().__init__(scope.link, name=name)
        self._scope = scope
        self.n = n

    def _w(self, cmd: str) -> None:
        self.link.write_checked(cmd)

    def _q(self, cmd: str) -> str:
        return self.link.ask(cmd)

    def _qf(self, cmd: str) -> float:
        return self.link.query_float(cmd)

    def _set_token(self, cmd: str, token: str) -> str:
        """Write ``cmd token`` and return the stripped read-back."""
        self._w(f"{cmd} {token}")
        return self._q(f"{cmd}?")

    def _set_bool(self, cmd: str, state: bool) -> bool:
        self._w(f"{cmd} {'ON' if state else 'OFF'}")
        return self._q(f"{cmd}?").strip('"').upper() in ("1", "ON")

    def _get_bool(self, cmd: str) -> bool:
        return self._q(cmd).strip('"').upper() in ("1", "ON")

    def _set_float(self, cmd: str, value: float, what: str, rel: float = _TOLERANCE) -> float:
        """Write a number, read it back, and raise when the scope holds another value."""
        self._w(f"{cmd} {_num(value)}")
        got = self._qf(f"{cmd}?")
        if not _close(value, got, rel):
            raise InstrumentError(f"{what}: requested {value:g}, the scope applied {got:g}")
        return got

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


# ---------------------------------------------------------------------------
# Vertical
# ---------------------------------------------------------------------------


class _Channel(_RtbNode):
    """One analog channel (UM p.314-321). Each setting is atomic by design standard 4.3.

    The input is 1 Mohm. A 50 ohm load is wiring: on the original bench CH1 has an
    external feed-through.
    """

    def __init__(self, scope: RohdeSchwarz_RTB2000, n: int):
        super().__init__(scope, f"CH{n}", n)
        self._c = f"CHAN{n}"

    def set_display_state(self, state: bool) -> bool:
        """Turn this channel on or off; returns the read-back."""
        return self._set_bool(f"{self._c}:STAT", state)

    def get_display_state(self) -> bool:
        """Whether this channel is on."""
        return self._get_bool(f"{self._c}:STAT?")

    def set_scale(self, volts_per_div: float) -> float:
        """Set the vertical scale in V/div at the probe tip; returns the read-back.

        The attenuation is included. The scope holds 3 significant digits
        (MEAS), so the read-back is checked to 0.6 percent.
        """
        v = _number(volts_per_div, "volts_per_div")
        return self._set_float(f"{self._c}:SCAL", v, f"CH{self.n} scale", rel=6e-3)

    def get_scale(self) -> float:
        """The vertical scale in V/div at the probe tip."""
        return self._qf(f"{self._c}:SCAL?")

    def set_offset(self, volts: float) -> float:
        """Set the vertical offset in volts; returns the read-back.

        The offset steps in fractions of a division, so the read-back is
        checked to 2 percent of the scale (original tolerance).
        """
        v = _number(volts, "volts")
        self._w(f"{self._c}:OFFS {_num(v)}")
        got = self.get_offset()
        if abs(got - v) > max(6e-3 * abs(v), 0.02 * self.get_scale()):
            raise InstrumentError(f"CH{self.n} offset: requested {v:g}, the scope applied {got:g}")
        return got

    def get_offset(self) -> float:
        """The vertical offset in volts."""
        return self._qf(f"{self._c}:OFFS?")

    def set_position(self, divisions: float) -> float:
        """Set the vertical position in divisions; returns the read-back."""
        d = _number(divisions, "divisions")
        return self._set_float(f"{self._c}:POS", d, f"CH{self.n} position", rel=1e-2)

    def get_position(self) -> float:
        """The vertical position in divisions."""
        return self._qf(f"{self._c}:POS?")

    def set_coupling(self, coupling: Coupling | str) -> Coupling:
        """Set the input coupling: DC, AC or GND; returns the read-back.

        Enum arg: <scope>.Coupling.<TAB>
        """
        got = self._set_token(f"{self._c}:COUP", _token(coupling, Coupling, "coupling"))
        return _answer(got, Coupling, "coupling")

    def get_coupling(self) -> Coupling:
        """The input coupling."""
        return _answer(self._q(f"{self._c}:COUP?"), Coupling, "coupling")

    def set_bandwidth_limit_state(self, state: bool) -> bool:
        """Turn the 20 MHz bandwidth limit on or off; returns the read-back."""
        got = self._set_token(f"{self._c}:BAND", "B20" if state else "FULL")
        return got.upper().startswith("B20")

    def get_bandwidth_limit_state(self) -> bool:
        """Whether the 20 MHz bandwidth limit is on."""
        return self._q(f"{self._c}:BAND?").upper().startswith("B20")

    def set_invert(self, state: bool) -> bool:
        """Invert the displayed waveform, or show it normal; returns the read-back."""
        got = self._set_token(f"{self._c}:POL", "INV" if state else "NORM")
        return got.upper().startswith("INV")

    def get_invert(self) -> bool:
        """Whether the displayed waveform is inverted."""
        return self._q(f"{self._c}:POL?").upper().startswith("INV")

    def set_attenuation(self, ratio: float) -> float:
        """Set the probe attenuation ratio, for example 10 for 10:1; returns the read-back.

        A manual setting in volts. The original channel helper wrote it before
        the scale because it rescales the display.
        """
        # UNVERIFIED: the "set attenuation first" order is an original code note,
        # not a recorded observation.
        r = _number(ratio, "ratio")
        self._w(f"PROB{self.n}:SET:ATT:UNIT V")
        return self._set_float(f"PROB{self.n}:SET:ATT:MAN", r, f"CH{self.n} attenuation")

    def get_attenuation(self) -> float:
        """The probe attenuation ratio."""
        return self._qf(f"PROB{self.n}:SET:ATT:MAN?")

    def set_threshold(self, volts: float) -> float:
        """Set the digitizing threshold in volts for pattern triggers and bus decode.

        The level turns this analog channel into a logic signal (UM p.318).
        Returns the read-back.
        """
        v = _number(volts, "volts")
        return self._set_float(f"{self._c}:THR", v, f"CH{self.n} threshold", rel=2e-2)

    def get_threshold(self) -> float:
        """The digitizing threshold in volts."""
        return self._qf(f"{self._c}:THR?")

    def set_threshold_find(self) -> float:
        """Let the scope set the threshold from the signal on screen; returns it in volts.

        It is not the 50 percent point: two runs on a square set 60 and 53
        percent of the span, and a channel ranged for another signal gives a
        wrong level (quirk RTB-12). Range the channel first.
        """
        self._w(f"{self._c}:THR:FIND")
        return self.get_threshold()

    def set_hysteresis(self, size: Hysteresis | str) -> Hysteresis:
        """Set the threshold hysteresis; returns the read-back.

        Enum arg: <scope>.Hysteresis.<TAB>
        """
        got = self._set_token(f"{self._c}:THR:HYST", _token(size, Hysteresis, "hysteresis"))
        return _answer(got, Hysteresis, "hysteresis")

    def get_hysteresis(self) -> Hysteresis:
        """The threshold hysteresis."""
        return _answer(self._q(f"{self._c}:THR:HYST?"), Hysteresis, "hysteresis")

    def get_settings(self) -> ChannelSettings:
        """Every vertical setting of this channel."""
        return ChannelSettings(
            display_state=self.get_display_state(),
            scale=self.get_scale(),
            offset=self.get_offset(),
            position=self.get_position(),
            coupling=self.get_coupling(),
            bandwidth_limit=self.get_bandwidth_limit_state(),
            invert=self.get_invert(),
            attenuation=self.get_attenuation(),
            type=self._q(f"{self._c}:TYPE?").upper(),
        )


# ---------------------------------------------------------------------------
# Horizontal and acquisition
# ---------------------------------------------------------------------------


class _Horizontal(_RtbNode):
    """Horizontal settings (UM p.321-323). The screen is 12 divisions (quirk RTB-04)."""

    def set_scale(self, seconds_per_div: float) -> float:
        """Set the time scale in s/div; returns the read-back.

        The scope takes 1-2-5 steps and values between, held to 3 significant
        digits (MEAS).
        """
        v = _number(seconds_per_div, "seconds_per_div")
        return self._set_float("TIM:SCAL", v, "time scale", rel=6e-3)

    def get_scale(self) -> float:
        """The time scale in s/div."""
        return self._qf("TIM:SCAL?")

    def set_position(self, seconds: float) -> float:
        """Set the time from the trigger to the reference point in seconds; returns it."""
        v = _number(seconds, "seconds")
        self._w(f"TIM:POS {_num(v)}")
        got = self.get_position()
        if abs(got - v) > max(6e-3 * abs(v), 1e-3 * self.get_scale()):
            raise InstrumentError(f"time position: requested {v:g}, the scope applied {got:g}")
        return got

    def get_position(self) -> float:
        """The time from the trigger to the reference point in seconds."""
        return self._qf("TIM:POS?")

    def set_reference(self, reference: TimeReference | str) -> TimeReference:
        """Set the reference point: LEFT, CENTER or RIGHT; returns the read-back.

        Enum arg: <scope>.TimeReference.<TAB>

        LEFT and RIGHT are one division in from the screen edge. The scope has
        no other positions (quirk RTB-07).
        """
        member = _member(reference, TimeReference, "time reference")
        percent = _REFERENCE_PERCENT[member.value]
        self._set_float("TIM:REF", percent, "time reference", rel=2e-3)
        return self.get_reference()

    def get_reference(self) -> TimeReference:
        """The reference point, as the nearest of LEFT, CENTER and RIGHT."""
        percent = self._qf("TIM:REF?")
        label = min(_REFERENCE_PERCENT, key=lambda k: abs(_REFERENCE_PERCENT[k] - percent))
        return TimeReference(label)

    def set_roll_auto_state(self, state: bool) -> bool:
        """Turn automatic roll mode at slow time scales on or off; returns the read-back."""
        return self._set_bool("TIM:ROLL:AUT", state)

    def get_roll_auto_state(self) -> bool:
        """Whether automatic roll mode is on."""
        return self._get_bool("TIM:ROLL:AUT?")

    def set_roll_minimum(self, seconds_per_div: float) -> float:
        """Set the time scale in s/div at and above which roll mode starts; returns it."""
        v = _number(seconds_per_div, "seconds_per_div")
        return self._set_float("TIM:ROLL:MTIM", v, "roll minimum", rel=1e-2)

    def get_roll_minimum(self) -> float:
        """The time scale in s/div at and above which roll mode starts."""
        return self._qf("TIM:ROLL:MTIM?")

    def set_zoom(self, scale: float, *, time: float = 0.0) -> ZoomSettings:
        """Open the zoom window at ``scale`` s/div, centred ``time`` s from the trigger.

        UM p.347. Returns the read-back.
        """
        s = _number(scale, "scale")
        t = _number(time, "time")
        self._set_bool("TIM:ZOOM:STAT", True)
        self._set_float("TIM:ZOOM:SCAL", s, "zoom scale", rel=6e-3)
        self._w(f"TIM:ZOOM:TIME {_num(t)}")
        got = self._qf("TIM:ZOOM:TIME?")
        if abs(got - t) > max(1e-2 * abs(t), 1e-3 * self.get_scale()):
            raise InstrumentError(f"zoom time: requested {t:g}, the scope applied {got:g}")
        zoom = self.get_zoom()
        if zoom is None:
            raise InstrumentError("the zoom window reads off after it was turned on")
        return zoom

    def get_zoom(self) -> ZoomSettings | None:
        """The zoom window, or None when the zoom is off."""
        if not self._get_bool("TIM:ZOOM:STAT?"):
            return None
        return ZoomSettings(
            scale=self._qf("TIM:ZOOM:SCAL?"),
            time=self._qf("TIM:ZOOM:TIME?"),
            position=self._qf("TIM:ZOOM:POS?"),
        )

    def set_zoom_state(self, state: bool) -> bool:
        """Open or close the zoom window with its present settings; returns the read-back."""
        return self._set_bool("TIM:ZOOM:STAT", state)

    def get_range(self) -> float:
        """The time across the 12 screen divisions, in seconds (quirk RTB-04)."""
        return self._qf("TIM:RANG?")

    def get_time_acquired(self) -> float:
        """The time the hardware records, in seconds; it can exceed the screen (RTB-04)."""
        return self._qf("TIM:RAT?")

    def set_timebase(
        self,
        scale: float,
        *,
        position: float | None = None,
        reference: TimeReference | str | None = None,
    ) -> HorizontalSettings:
        """Set the time scale in s/div, with roll mode off; returns every horizontal setting.

        ``position`` (s) and ``reference`` change only when given. The record
        length is an acquisition setting: see ``acq.set_record_length``.
        """
        self.set_roll_auto_state(False)
        if reference is not None:
            self.set_reference(reference)
        self.set_scale(scale)
        if position is not None:
            self.set_position(position)
        return self.get_timebase()

    def get_timebase(self) -> HorizontalSettings:
        """Every horizontal setting."""
        return HorizontalSettings(
            scale=self.get_scale(),
            position=self.get_position(),
            reference=self.get_reference(),
            roll_auto=self.get_roll_auto_state(),
            range=self.get_range(),
            time_acquired=self.get_time_acquired(),
        )


class _Acquisition(_RtbNode):
    """Acquisition settings (UM p.324-329, p.372-374). Each setting is atomic."""

    def set_mode(self, mode: AcquireMode | str) -> AcquireMode:
        """Set the acquisition mode for all channels; returns the read-back.

        Enum arg: <scope>.AcquireMode.<TAB>

        HIGH_RES is an averaged decimation that low-passes the trace. Use
        SAMPLE for raw samples.
        """
        member = _member(mode, AcquireMode, "acquisition mode")
        acq_type = {AcquireMode.AVERAGE: "AVER", AcquireMode.ENVELOPE: "ENV"}.get(member, "REFR")
        self._w(f"ACQ:TYPE {acq_type}")
        self._w("ACQ:PEAK " + ("AUTO" if member is AcquireMode.PEAK else "OFF"))
        self._w("ACQ:HRES " + ("AUTO" if member is AcquireMode.HIGH_RES else "OFF"))
        ctype = {AcquireMode.HIGH_RES: "HRES", AcquireMode.PEAK: "PDET"}.get(member, "SAMP")
        for ch in self._scope._channels():
            self._w(f"CHAN{ch.n}:TYPE {ctype}")
        got = self.get_mode()
        if got is not member:
            raise InstrumentError(f"acquisition mode reads {got.name} after {member.name}")
        return got

    def get_mode(self) -> AcquireMode:
        """The acquisition mode."""
        acq_type = self._q("ACQ:TYPE?").upper()
        if acq_type.startswith("AVER"):
            return AcquireMode.AVERAGE
        if acq_type.startswith("ENV"):
            return AcquireMode.ENVELOPE
        ctype = self._q("CHAN1:TYPE?").upper()
        by_type = {"HRES": AcquireMode.HIGH_RES, "PDET": AcquireMode.PEAK}
        return by_type.get(ctype, AcquireMode.SAMPLE)

    def set_average_count(self, count: int) -> int:
        """Set the number of acquisitions averaged in AVERAGE mode; returns the read-back."""
        n = int(_number(count, "count"))
        return int(self._set_float("ACQ:AVER:COUN", float(n), "average count"))

    def get_average_count(self) -> int:
        """The number of acquisitions averaged in AVERAGE mode."""
        return int(self._qf("ACQ:AVER:COUN?"))

    def set_record_length(self, points: int, *, timeout: float = 3.0) -> int:
        """Set the points per acquisition, 10 k to 20 M predefined values; returns it.

        UM p.324. The scope applies a new record length at the next
        acquisition and reads back the old value until then (quirk RTB-08).
        So this starts the acquisition when it is not running and polls the
        read-back for ``timeout`` seconds. With no acquisition, for example a
        normal trigger and no signal, it raises.
        """
        n = int(_number(points, "points"))
        if self._scope.get_acquisition_state() is not AcquisitionState.RUN:
            self._scope.cmd_run()
        if self._q("ACQ:MEM?").upper() != "DMEM":
            self._w("ACQ:MEM DMEM")
        self._w(f"ACQ:POIN {n}")
        t0 = time.monotonic()
        while True:
            got = self.get_record_length()
            if got == n:
                return got
            if time.monotonic() - t0 > timeout:
                raise InstrumentError(
                    f"record length reads {got} {timeout} s after writing {n}: it applies at "
                    "the next acquisition (quirk RTB-08). Check the trigger, or use a "
                    "predefined value."
                )
            time.sleep(_RECORD_LENGTH_POLL_S)

    def get_record_length(self) -> int:
        """The points per acquisition."""
        return int(self._qf("ACQ:POIN?"))

    def set_interpolation(self, kind: Interpolation | str) -> Interpolation:
        """Set the interpolation between samples; returns the read-back.

        Enum arg: <scope>.Interpolation.<TAB>
        """
        got = self._set_token("ACQ:INT", _token(kind, Interpolation, "interpolation"))
        return _answer(got, Interpolation, "interpolation")

    def get_interpolation(self) -> Interpolation:
        """The interpolation between samples."""
        return _answer(self._q("ACQ:INT?"), Interpolation, "interpolation")

    def get_sample_rate(self) -> float:
        """The display sample rate in samples per second.

        A transferred record runs at ``get_adc_rate()`` instead (quirk RTB-04).
        """
        return self._qf("ACQ:SRAT?")

    def get_adc_rate(self) -> float:
        """The sample rate of a transferred record in samples per second."""
        return self._qf("ACQ:POIN:ARAT?")

    def get_settings(self) -> AcquisitionSettings:
        """Every acquisition setting."""
        return AcquisitionSettings(
            mode=self.get_mode(),
            record_length=self.get_record_length(),
            interpolation=self.get_interpolation(),
            sample_rate=self.get_sample_rate(),
            adc_rate=self.get_adc_rate(),
            average_count=self.get_average_count(),
            memory=self._q("ACQ:MEM?").upper(),
        )


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

_LEVEL_INDEX = {"ch1": 1, "ch2": 2, "ch3": 3, "ch4": 4, "ext": 5}


class _Trigger(_RtbNode):
    """The A trigger (UM p.330-347). Each trigger type has one helper."""

    @staticmethod
    def _level_index(source: str) -> int:
        try:
            return _LEVEL_INDEX[source.strip().lower()]
        except KeyError:
            raise ValueError(
                f"source {source!r} has no trigger level; use ch1..ch4 or ext"
            ) from None

    def set_mode(self, mode: TriggerMode | str) -> TriggerMode:
        """Set the trigger mode: AUTO free-runs, NORMAL waits; returns the read-back.

        Enum arg: <scope>.TriggerMode.<TAB>
        """
        got = self._set_token("TRIG:A:MODE", _token(mode, TriggerMode, "trigger mode"))
        return _answer(got, TriggerMode, "trigger mode")

    def get_mode(self) -> TriggerMode:
        """The trigger mode."""
        return _answer(self._q("TRIG:A:MODE?"), TriggerMode, "trigger mode")

    def set_source(self, source: str) -> str:
        """Set the trigger source: ch1..ch4, ext, line or d0..d15; returns the read-back."""
        got = self._set_token("TRIG:A:SOUR", _source(source))
        return _TRIGGER_SOURCE_EXTRA.get(got.upper(), got.lower())

    def get_source(self) -> str:
        """The trigger source in lower case; pattern, none or bus1 for those trigger types."""
        got = self._q("TRIG:A:SOUR?")
        return _TRIGGER_SOURCE_EXTRA.get(got.upper(), got.lower())

    def set_level(self, volts: float, *, source: str | None = None) -> float:
        """Set the trigger level in volts for ``source``, default the present source.

        The line source has no level. The level steps in fractions of a
        division, so the read-back is checked to 2 percent of the source's
        V/div. Returns the read-back.
        """
        v = _number(volts, "volts")
        src = (source or self.get_source()).strip().lower()
        i = self._level_index(src)
        self._w(f"TRIG:A:LEV{i} {_num(v)}")
        got = self._qf(f"TRIG:A:LEV{i}?")
        scale = self._scope._channel(src).get_scale() if src.startswith("ch") else 1.0
        if abs(got - v) > max(1e-2 * abs(v), 0.02 * scale):
            raise InstrumentError(
                f"trigger level {src}: requested {v:g}, the scope applied {got:g}"
            )
        return got

    def get_level(self, *, source: str | None = None) -> float:
        """The trigger level in volts for ``source``, default the present source."""
        return self._qf(f"TRIG:A:LEV{self._level_index(source or self.get_source())}?")

    def set_level_find(self) -> float:
        """Set the level to the signal's mid level, the front-panel 'Set to 50%'; returns it."""
        self._w("TRIG:A:FIND")
        return self.get_level()

    def set_slope(self, slope: Slope | str) -> Slope:
        """Set the edge slope: RISE, FALL or EITHER; returns the read-back.

        Enum arg: <scope>.Slope.<TAB>
        """
        got = self._set_token("TRIG:A:EDGE:SLOP", _token(slope, Slope, "slope"))
        return _answer(got, Slope, "slope")

    def get_slope(self) -> Slope:
        """The edge slope."""
        return _answer(self._q("TRIG:A:EDGE:SLOP?"), Slope, "slope")

    def set_coupling(self, coupling: TriggerCoupling | str) -> TriggerCoupling:
        """Set the edge-trigger coupling: DC, AC or LF_REJECT; returns the read-back.

        Enum arg: <scope>.TriggerCoupling.<TAB>
        """
        token = _token(coupling, TriggerCoupling, "trigger coupling")
        return _answer(self._set_token("TRIG:A:EDGE:COUP", token), TriggerCoupling, "coupling")

    def get_coupling(self) -> TriggerCoupling:
        """The edge-trigger coupling."""
        return _answer(self._q("TRIG:A:EDGE:COUP?"), TriggerCoupling, "trigger coupling")

    def set_hf_reject_state(self, state: bool) -> bool:
        """Turn the edge trigger's high-frequency reject filter on or off; returns it."""
        return self._set_bool("TRIG:A:EDGE:FILT:HFR", state)

    def get_hf_reject_state(self) -> bool:
        """Whether the high-frequency reject filter is on."""
        return self._get_bool("TRIG:A:EDGE:FILT:HFR?")

    def set_noise_reject_state(self, state: bool) -> bool:
        """Turn the edge trigger's noise reject on or off; returns the read-back."""
        return self._set_bool("TRIG:A:EDGE:FILT:NREJ", state)

    def get_noise_reject_state(self) -> bool:
        """Whether the noise reject is on."""
        return self._get_bool("TRIG:A:EDGE:FILT:NREJ?")

    def set_holdoff(self, seconds: float | None) -> float | None:
        """Set the holdoff time in seconds, or None for no holdoff; returns the read-back."""
        if seconds is None:
            self._w("TRIG:A:HOLD:MODE OFF")
            return self.get_holdoff()
        v = _number(seconds, "seconds")
        self._w("TRIG:A:HOLD:MODE TIME")
        return self._set_float("TRIG:A:HOLD:TIME", v, "holdoff", rel=1e-2)

    def get_holdoff(self) -> float | None:
        """The holdoff time in seconds, or None when holdoff is off."""
        if self._q("TRIG:A:HOLD:MODE?").upper() == "OFF":
            return None
        return self._qf("TRIG:A:HOLD:TIME?")

    def set_edge(
        self,
        source: str,
        *,
        level: float | None = None,
        slope: Slope | str = Slope.RISE,
        coupling: TriggerCoupling | str = TriggerCoupling.DC,
        hf_reject: bool = False,
        noise_reject: bool = False,
        holdoff: float | None = None,
        mode: TriggerMode | str = TriggerMode.NORMAL,
    ) -> TriggerSettings:
        """Edge trigger on ``source``: ch1..ch4, ext, line or d0..d15; returns the read-back.

        ``level=None`` finds the mid level on an analog channel; ext and line
        keep their level. A logic channel uses its pod threshold, see
        ``logic.set_threshold``. NORMAL mode is the default, so a capture
        proves the trigger fired.
        """
        src = source.strip().lower()
        self._w("TRIG:A:TYPE EDGE")
        self.set_source(src)
        self.set_slope(slope)
        if not src.startswith("d") and src != "line":
            self.set_coupling(coupling)
            self.set_hf_reject_state(hf_reject)
            self.set_noise_reject_state(noise_reject)
            if level is None and src != "ext":
                self.set_level_find()
            elif level is not None:
                self.set_level(level, source=src)
        self.set_holdoff(holdoff)
        self.set_mode(mode)
        return self.get_settings()

    def _set_type(self, word: str, source: str, mode) -> None:
        self._w(f"TRIG:A:TYPE {word}")
        got = self._q("TRIG:A:TYPE?").upper()
        if not word.upper().startswith(got[:3]):
            raise InstrumentError(f"trigger type reads {got!r} after {word!r}")
        self.set_source(source)
        self.set_mode(mode)

    def _set_time(self, cmd: str, value: float, what: str) -> float:
        return self._set_float(cmd, _number(value, what), what, rel=2e-2)

    def set_pulse_width(
        self,
        source: str,
        *,
        level: float,
        width: float,
        polarity: Polarity | str = Polarity.POSITIVE,
        condition: WidthCondition | str = WidthCondition.LONGER,
        delta: float | None = None,
        mode: TriggerMode | str = TriggerMode.NORMAL,
    ) -> TriggerSettings:
        """Width trigger: fire at the end of a pulse whose width in s meets ``condition``.

        ``delta`` (s) is the tolerance for WITHIN and OUTSIDE. Returns the read-back.
        """
        self._set_type("WIDT", source, mode)
        self.set_level(level, source=source)
        self._w(f"TRIG:A:WIDT:POL {_token(polarity, Polarity, 'polarity')}")
        self._w(f"TRIG:A:WIDT:RANG {_token(condition, WidthCondition, 'condition')}")
        self._set_time("TRIG:A:WIDT:WIDT", width, "width")
        if delta is not None:
            self._set_time("TRIG:A:WIDT:DELT", delta, "width delta")
        return self.get_settings()

    def set_runt(
        self,
        source: str,
        *,
        lower: float,
        upper: float,
        polarity: Polarity | str = Polarity.POSITIVE,
        condition: WidthCondition | str = WidthCondition.LONGER,
        width: float = 0.0,
        delta: float | None = None,
        mode: TriggerMode | str = TriggerMode.NORMAL,
    ) -> TriggerSettings:
        """Runt trigger: a pulse crosses ``lower`` (V) but not ``upper`` (V); returns it.

        Positive polarity as described; negative mirrors it. ``width`` and
        ``delta`` are in seconds; a zero width leaves the width alone.
        """
        self._set_type("RUNT", source, mode)
        i = self._level_index(source)
        for which, v in (("UPP", upper), ("LOW", lower)):
            what = f"runt {which.lower()}"
            self._set_float(f"TRIG:A:LEV{i}:RUNT:{which}", _number(v, what), what, rel=2e-2)
        self._w(f"TRIG:A:RUNT:POL {_token(polarity, Polarity, 'polarity')}")
        self._w(f"TRIG:A:RUNT:RANG {_token(condition, WidthCondition, 'condition')}")
        if width:
            self._set_time("TRIG:A:RUNT:WIDT", width, "runt width")
        if delta is not None:
            self._set_time("TRIG:A:RUNT:DELT", delta, "runt delta")
        return self.get_settings()

    def set_transition(
        self,
        source: str,
        *,
        lower: float,
        upper: float,
        time: float,
        edge: Edge | str = Edge.RISE,
        condition: WidthCondition | str = WidthCondition.LONGER,
        delta: float | None = None,
        mode: TriggerMode | str = TriggerMode.NORMAL,
    ) -> TriggerSettings:
        """Rise or fall time trigger between ``lower`` and ``upper`` (V); returns it.

        ``time`` and ``delta`` are in seconds.
        """
        self._set_type("RIS", source, mode)
        i = self._level_index(source)
        for which, v in (("UPP", upper), ("LOW", lower)):
            what = f"rise time {which.lower()}"
            self._set_float(f"TRIG:A:LEV{i}:RIS:{which}", _number(v, what), what, rel=2e-2)
        self._w(f"TRIG:A:RIS:SLOP {_token(edge, Edge, 'edge')}")
        self._w(f"TRIG:A:RIS:RANG {_token(condition, WidthCondition, 'condition')}")
        self._set_time("TRIG:A:RIS:TIME", time, "rise time")
        if delta is not None:
            self._set_time("TRIG:A:RIS:DELT", delta, "rise time delta")
        return self.get_settings()

    def set_timeout(
        self,
        source: str,
        *,
        level: float,
        time: float,
        condition: TimeoutCondition | str = TimeoutCondition.HIGH,
        mode: TriggerMode | str = TriggerMode.NORMAL,
    ) -> TriggerSettings:
        """Timeout trigger: the signal stays above or below ``level`` (V) for ``time`` (s)."""
        self._set_type("TIM", source, mode)
        self.set_level(level, source=source)
        self._w(f"TRIG:A:TIM:RANG {_token(condition, TimeoutCondition, 'condition')}")
        self._set_time("TRIG:A:TIM:TIME", time, "timeout")
        return self.get_settings()

    def set_pattern(
        self,
        states: Mapping[str, PatternLevel | str],
        *,
        levels: Mapping[str, float] | None = None,
        logic: PatternLogic | str = PatternLogic.AND,
        when_true: bool = True,
        mode: TriggerMode | str = TriggerMode.NORMAL,
    ) -> TriggerSettings:
        """Pattern trigger on the analog channels; returns the read-back with the pattern.

        ``states`` maps "ch1".."ch4" to HIGH, LOW or DONT_CARE; a channel not
        named is DONT_CARE. ``levels`` maps a channel to its level in volts.
        """
        wanted = {k.strip().lower(): v for k, v in states.items()}
        bits = "".join(
            _token(wanted.get(f"ch{ch.n}", PatternLevel.DONT_CARE), PatternLevel, "pattern level")
            for ch in self._scope._channels()
        )
        self._w("TRIG:A:TYPE LOG")
        self._w(f"TRIG:A:PATT:SOUR {quote(bits)}")
        self._w(f"TRIG:A:PATT:FUNC {_token(logic, PatternLogic, 'pattern logic')}")
        self._w(f"TRIG:A:PATT:COND {quote('TRUE' if when_true else 'FALSE')}")
        self._w("TRIG:A:PATT:MODE OFF")
        for name, v in (levels or {}).items():
            self.set_level(v, source=name)
        self.set_mode(mode)
        got = self._q("TRIG:A:PATT:SOUR?").strip('"')
        if not got.upper().startswith(bits):
            raise InstrumentError(f"pattern reads {got!r}, requested {bits!r}")
        settings = self.get_settings()
        return TriggerSettings(**{**settings.__dict__, "pattern": got})

    def set_video(
        self,
        source: str,
        *,
        standard: VideoStandard | str = VideoStandard.NTSC,
        modulation: VideoModulation | str = VideoModulation.POSITIVE,
        field: VideoField | str = VideoField.ALL_LINES,
        line: int | None = None,
        mode: TriggerMode | str = TriggerMode.NORMAL,
    ) -> TriggerSettings:
        """Video (TV) trigger on ``source`` (UM p.336); returns the read-back.

        POSITIVE modulation means negative-going sync pulses. ``line`` picks
        the line for the LINE field.
        """
        self._set_type("TV", source, mode)
        self._w(f"TRIG:A:TV:STAN {_token(standard, VideoStandard, 'video standard')}")
        self._w(f"TRIG:A:TV:POL {_token(modulation, VideoModulation, 'video modulation')}")
        self._w(f"TRIG:A:TV:FIEL {_token(field, VideoField, 'video field')}")
        if line is not None:
            self._w(f"TRIG:A:TV:LINE {int(line)}")
        settings = self.get_settings()
        got_standard = _answer(self._q("TRIG:A:TV:STAN?"), VideoStandard, "video standard")
        got_field = _answer(self._q("TRIG:A:TV:FIEL?"), VideoField, "video field")
        return TriggerSettings(
            **{**settings.__dict__, "video_standard": got_standard, "video_field": got_field}
        )

    def set_actions(
        self,
        *,
        sound: bool = False,
        reference: bool = False,
        screenshot: str | None = None,
        waveform: str | None = None,
    ) -> TriggerActions:
        """Set the actions on each trigger: beep, copy to a reference, save a file.

        ``screenshot`` and ``waveform`` are destination paths on the scope.
        The waveform-file destination is refused remotely (-256, -200; quirk
        RTB-16). A trigger-out pulse is not offered here: see
        ``aux_out.set_trigger_pulse``. All off turns actions off. Returns the
        read-back.
        """
        on = bool(sound or reference or screenshot or waveform)
        self._w(f"TRIG:EVEN {'ON' if on else 'OFF'}")
        self._w(f"TRIG:EVEN:SOUN {'ON' if sound else 'OFF'}")
        self._w(f"TRIG:EVEN:REFS {'ON' if reference else 'OFF'}")
        if screenshot:
            self._w(f"TRIG:EVEN:SCRS:DEST {quote(screenshot)}")
        self._w(f"TRIG:EVEN:SCRS {'ON' if screenshot else 'OFF'}")
        if waveform:
            self._w(f"TRIG:EVEN:WFMS:DEST {quote(waveform)}")
        self._w(f"TRIG:EVEN:WFMS {'ON' if waveform else 'OFF'}")
        return TriggerActions(
            enabled=self._get_bool("TRIG:EVEN?"),
            sound=self._get_bool("TRIG:EVEN:SOUN?"),
            reference=self._get_bool("TRIG:EVEN:REFS?"),
            screenshot=self._get_bool("TRIG:EVEN:SCRS?"),
            waveform=self._get_bool("TRIG:EVEN:WFMS?"),
        )

    def get_settings(self) -> TriggerSettings:
        """The trigger type, mode, source, slope and holdoff, with level and coupling."""
        src = self.get_source()
        trig_type = self._q("TRIG:A:TYPE?").upper()
        level = coupling = None
        if src in _LEVEL_INDEX and trig_type.startswith(("EDGE", "WIDT", "TIM")):
            level = self.get_level(source=src)
            coupling = self.get_coupling()
        return TriggerSettings(
            type=trig_type,
            mode=self.get_mode(),
            source=src,
            slope=self.get_slope(),
            holdoff=self.get_holdoff(),
            level=level,
            coupling=coupling,
        )


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

_QUICK_FIELDS = (
    "peak_to_peak",
    "peak_upper",
    "peak_lower",
    "rms",
    "mean",
    "period",
    "frequency",
    "rise_time",
    "fall_time",
)


class _Measure(_RtbNode):
    """The Meas menu: measurement slots 1 to 8 (UM p.385-397).

    The node names the subsystem, so the one-shot measurements carry no
    prefix: ``scope.meas.frequency("ch1")``. Each takes one single
    acquisition in slot 8, reads the result, and leaves the scope running.
    """

    def set_slot(
        self, n: int, kind: MeasType | str, source: str, *, reference: str | None = None
    ) -> MeasSlot:
        """Turn on measurement slot ``n`` (1..8) with ``kind`` on ``source``; returns it.

        Enum arg: <scope>.MeasType.<TAB>

        DELAY and PHASE need a ``reference`` source.
        """
        member = _member(kind, MeasType, "measurement")
        if member in (MeasType.DELAY, MeasType.PHASE) and reference is None:
            raise ValueError(f"{member.name} needs a reference source")
        self._w(f"MEAS{n} ON")
        src = _source(source) + (f",{_source(reference, 'reference')}" if reference else "")
        self._w(f"MEAS{n}:SOUR {src}")
        self._w(f"MEAS{n}:MAIN {member.value}")
        got = self._q(f"MEAS{n}:MAIN?").upper()
        if not member.value.startswith(got[:4]) and not got.startswith(member.value):
            raise InstrumentError(f"measurement slot {n} reads {got!r} after {member.name}")
        return MeasSlot(slot=n, kind=member, source=source, reference=reference)

    def get_slot_result(self, n: int) -> float | None:
        """The present result of slot ``n``, or None when the scope reports it invalid."""
        v = self._qf(f"MEAS{n}:RES?")
        return None if abs(v) >= _INVALID else v

    def get_slot_statistics(self, n: int) -> MeasStatistics:
        """The statistics of slot ``n`` since they were last reset."""
        return MeasStatistics(
            mean=self._qf(f"MEAS{n}:RES:AVG?"),
            stddev=self._qf(f"MEAS{n}:RES:STDD?"),
            minimum=self._qf(f"MEAS{n}:RES:NPE?"),
            maximum=self._qf(f"MEAS{n}:RES:PPE?"),
            count=int(self._qf(f"MEAS{n}:RES:WFMC?")),
        )

    def set_slot_state(self, n: int, state: bool) -> bool:
        """Turn measurement slot ``n`` on or off; returns the read-back."""
        self._w(f"MEAS{n} {'ON' if state else 'OFF'}")
        # UNVERIFIED: the original driver never read the slot state back; this query
        # form was not run on the bench.
        return self._get_bool(f"MEAS{n}:ENAB?")

    def quick(self, source: str) -> QuickMeasurement:
        """Run the front-panel Quick Measurement on ``source``; returns its nine results.

        Uses slot 1, in the order the scope returns them (MEAS). Waits up to
        3 s for every result to be valid, then turns quick measurement off.
        """
        self._w(f"MEAS1:SOUR {_source(source)}")
        self._w("MEAS1:AON")
        try:
            t0 = time.monotonic()
            while True:
                vals = [float(x) for x in self._q("MEAS1:ARES?").split(",")]
                if len(vals) == len(_QUICK_FIELDS) and all(abs(v) < _INVALID for v in vals):
                    return QuickMeasurement(**dict(zip(_QUICK_FIELDS, vals, strict=True)))
                if time.monotonic() - t0 > _QUICK_TIMEOUT_S:
                    raise InstrumentError(f"quick measurement incomplete: {vals}")
                time.sleep(_RESULT_POLL_S)
        finally:
            self._w("MEAS1:AOFF")

    def custom(
        self,
        kind: MeasType | str,
        source: str,
        *,
        reference: str | None = None,
        timeout: float = 5.0,
    ) -> float:
        """Measure any ``kind`` on ``source`` from one single acquisition, in base units.

        Enum arg: <scope>.MeasType.<TAB>

        Raises when the result is invalid, for example with no edge or
        period on screen. The scope is left running.
        """
        member = _member(kind, MeasType, "measurement")
        self.set_slot(_MEAS_SLOT, member, source, reference=reference)
        self._scope.cmd_single()
        self._scope.cmd_wait_acquisition(timeout)
        try:
            v = self.get_slot_result(_MEAS_SLOT)
        finally:
            self.link.write("RUN")
        if v is None:
            raise InstrumentError(
                f"{member.name} on {source} is invalid (no edge or period on screen?)"
            )
        return v

    def frequency(self, source: str) -> float:
        """Measure the frequency of ``source`` in Hz."""
        return self.custom(MeasType.FREQUENCY, source)

    def period(self, source: str) -> float:
        """Measure the period of ``source`` in seconds."""
        return self.custom(MeasType.PERIOD, source)

    def peak_to_peak(self, source: str) -> float:
        """Measure the peak-to-peak voltage of ``source`` in volts."""
        return self.custom(MeasType.PEAK_TO_PEAK, source)

    def amplitude(self, source: str) -> float:
        """Measure the amplitude of a square wave on ``source``, high minus low, in volts."""
        return self.custom(MeasType.AMPLITUDE, source)

    def high(self, source: str) -> float:
        """Measure the high level of ``source`` in volts."""
        return self.custom(MeasType.HIGH, source)

    def low(self, source: str) -> float:
        """Measure the low level of ``source`` in volts."""
        return self.custom(MeasType.LOW, source)

    def mean(self, source: str) -> float:
        """Measure the mean voltage of ``source`` in volts."""
        return self.custom(MeasType.MEAN, source)

    def rms(self, source: str) -> float:
        """Measure the RMS voltage of ``source`` in volts."""
        return self.custom(MeasType.RMS, source)

    def rise_time(self, source: str) -> float:
        """Measure the rise time of ``source`` in seconds."""
        return self.custom(MeasType.RISE_TIME, source)

    def fall_time(self, source: str) -> float:
        """Measure the fall time of ``source`` in seconds."""
        return self.custom(MeasType.FALL_TIME, source)

    def duty_positive(self, source: str) -> float:
        """Measure the positive duty cycle of ``source`` in percent."""
        return self.custom(MeasType.DUTY_POSITIVE, source)

    def width_positive(self, source: str) -> float:
        """Measure the positive pulse width of ``source`` in seconds."""
        return self.custom(MeasType.WIDTH_POSITIVE, source)

    def overshoot_positive(self, source: str) -> float:
        """Measure the positive overshoot of ``source`` in percent."""
        return self.custom(MeasType.OVERSHOOT_POSITIVE, source)

    def phase(self, source: str, reference: str) -> float:
        """Measure the phase of ``source`` against ``reference`` in degrees."""
        return self.custom(MeasType.PHASE, source, reference=reference)

    def delay(self, source: str, reference: str) -> float:
        """Measure the delay of ``source`` against ``reference`` in seconds."""
        return self.custom(MeasType.DELAY, source, reference=reference)

    def statistics(
        self,
        kind: MeasType | str,
        source: str,
        *,
        reference: str | None = None,
        count: int = 100,
        timeout: float = 30.0,
    ) -> MeasStatistics:
        """Measure ``kind`` on ``source`` over ``count`` acquisitions; returns the statistics.

        Enum arg: <scope>.MeasType.<TAB>

        Uses slot 1: the scope refuses statistics (-221) unless slot 1 is on,
        and statistics apply to every active slot (quirk RTB-09).
        """
        member = _member(kind, MeasType, "measurement")
        self.set_slot(1, member, source, reference=reference)
        self._w("MEAS1:STAT ON")
        self._w("MEAS1:STAT:RES")
        self._scope.cmd_run()
        t0 = time.monotonic()
        while True:
            stats = self.get_slot_statistics(1)
            if stats.count >= count:
                self._w("MEAS1:STAT OFF")
                return stats
            if time.monotonic() - t0 > timeout:
                raise InstrumentTimeoutError(
                    f"{stats.count} of {count} acquisitions in {timeout} s; check the trigger"
                )
            time.sleep(_RESULT_POLL_S)


class _Dvm(_RtbNode):
    """Digital voltmeter, slots 1 to 4 (UM p.421). It reads a channel's acquired waveform."""

    def set_slot(
        self, n: int, source: str, kind: VoltmeterType | str = VoltmeterType.DC
    ) -> DvmSlot:
        """Turn on voltmeter slot ``n`` (1..4) reading ``kind`` on ``source``; returns it.

        Enum arg: <scope>.VoltmeterType.<TAB>
        """
        member = _member(kind, VoltmeterType, "voltmeter type")
        self._w(f"DVM{n}:SOUR {_source(source)}")
        self._w(f"DVM{n}:TYPE {member.value}")
        self._w(f"DVM{n}:ENAB ON")
        got = _answer(self._q(f"DVM{n}:TYPE?"), VoltmeterType, "voltmeter type")
        if got is not member:
            raise InstrumentError(f"voltmeter {n} type reads {got.name} after {member.name}")
        return DvmSlot(slot=n, source=source, kind=got)

    def get_slot_result(self, n: int, *, timeout: float = 3.0) -> float:
        """The present result of slot ``n`` in volts; waits up to ``timeout`` s for a valid one."""
        t0 = time.monotonic()
        while True:
            value, ok = self._q(f"DVM{n}:RES:STAT?").split(",")
            if ok.strip() == "1" and abs(float(value)) < _INVALID:
                return float(value)
            if time.monotonic() - t0 > timeout:
                raise InstrumentError(f"voltmeter {n} has no valid result (status {ok.strip()})")
            time.sleep(_DVM_POLL_S)

    def set_slot_state(self, n: int, state: bool) -> bool:
        """Turn voltmeter slot ``n`` on or off; returns the read-back."""
        # UNVERIFIED: the original driver never read this state back; the query form
        # was not run on the bench.
        return self._set_bool(f"DVM{n}:ENAB", state)

    def _meas(self, source: str, kind: VoltmeterType) -> float:
        self.set_slot(1, source, kind)
        try:
            return self.get_slot_result(1)
        finally:
            self._w("DVM1:ENAB OFF")

    def meas_dc(self, source: str) -> float:
        """Measure the mean voltage of the acquired waveform on ``source`` in volts."""
        return self._meas(source, VoltmeterType.DC)

    def meas_ac_dc_rms(self, source: str) -> float:
        """Measure the RMS voltage, AC plus DC, of ``source`` in volts."""
        return self._meas(source, VoltmeterType.AC_DC_RMS)

    def meas_ac_rms(self, source: str) -> float:
        """Measure the RMS voltage of the AC part of ``source`` in volts."""
        return self._meas(source, VoltmeterType.AC_RMS)


_COUNTER_MIN = 1e-30  # below any real frequency in Hz or period in s


class _Counter(_RtbNode):
    """Trigger counter (UM p.422): frequency and period of a channel or the trigger source."""

    def set_state(self, state: bool) -> bool:
        """Turn the trigger counter on or off; returns the read-back."""
        return self._set_bool("TCO:ENAB", state)

    def get_state(self) -> bool:
        """Whether the trigger counter is on."""
        return self._get_bool("TCO:ENAB?")

    def set_source(self, source: str) -> str:
        """Set the counter source: ch1..ch4, or "trigger"; returns the read-back in lower case."""
        token = "TRIG" if source.strip().lower() == "trigger" else _source(source)
        return self._set_token("TCO:SOUR", token).lower()

    def _read(self, query: str, timeout: float) -> float:
        t0 = time.monotonic()
        while True:
            v = self._qf(query)
            # MEAS 2026-11-05: right after a mode change the counter read 1.29e-217, not 0.
            if _COUNTER_MIN < abs(v) < _INVALID:
                return v
            if time.monotonic() - t0 > timeout:
                raise InstrumentError(f"the counter has no valid result ({v})")
            time.sleep(_RESULT_POLL_S)

    def meas_frequency(self, source: str = "trigger", *, timeout: float = 3.0) -> float:
        """Turn the counter on, and measure the frequency of ``source`` in Hz."""
        self.set_state(True)
        self.set_source(source)
        return self._read("TCO:RES:FREQ?", timeout)

    def meas_period(self, source: str = "trigger", *, timeout: float = 3.0) -> float:
        """Turn the counter on, and measure the period of ``source`` in seconds."""
        self.set_state(True)
        self.set_source(source)
        return self._read("TCO:RES:PER?", timeout)


class _Cursors(_RtbNode):
    """Cursor set 1 (UM p.143-147): time cursors or voltage cursors."""

    def _setup(self, source: str, function: str) -> None:
        self._w("CURS1:STAT ON")
        self._w(f"CURS1:FUNC {function}")
        self._w(f"CURS1:SOUR {_source(source)}")

    def set_time(self, source: str, t1: float, t2: float) -> TimeCursors:
        """Place two time cursors on ``source`` at ``t1`` and ``t2`` s from the trigger.

        Returns the read-back with the time difference and its inverse.
        """
        self._setup(source, "VERT")
        got1 = self._set_float("CURS1:X1P", _number(t1, "t1"), "cursor t1", rel=1e-2)
        got2 = self._set_float("CURS1:X2P", _number(t2, "t2"), "cursor t2", rel=1e-2)
        return TimeCursors(
            t1=got1,
            t2=got2,
            delta=self._qf("CURS1:XDEL?"),
            inverse_delta=self._qf("CURS1:XDEL:INV?"),
        )

    def set_voltage(self, source: str, v1: float, v2: float) -> VoltageCursors:
        """Place two voltage cursors on ``source`` at ``v1`` and ``v2`` volts; returns them."""
        self._setup(source, "HOR")
        self._w(f"CURS1:Y1P {_num(_number(v1, 'v1'))}")
        self._w(f"CURS1:Y2P {_num(_number(v2, 'v2'))}")
        return VoltageCursors(
            v1=self._qf("CURS1:Y1P?"),
            v2=self._qf("CURS1:Y2P?"),
            delta=self._qf("CURS1:YDEL?"),
        )

    def set_state(self, state: bool) -> bool:
        """Show or hide the cursors; returns the read-back."""
        # UNVERIFIED: the original driver only wrote OFF and never read this back.
        return self._set_bool("CURS1:STAT", state)


class _Display(_RtbNode):
    """Display settings (UM p.453-457) and the XY display (UM p.420)."""

    def set_persistence(self, seconds: float | None) -> float | None:
        """Set the persistence in s (0.05 to 12.8), math.inf for infinite, None for off.

        Returns the read-back in the same form.
        """
        if seconds is None:
            self._w("DISP:PERS:TYPE OFF")
        elif math.isinf(_number(seconds, "seconds")):
            self._w("DISP:PERS:TYPE INF")
        else:
            self._w("DISP:PERS:TYPE TIME")
            self._set_float("DISP:PERS:TIME", float(seconds), "persistence", rel=0.05)
        return self.get_persistence()

    def get_persistence(self) -> float | None:
        """The persistence in seconds; math.inf for infinite; None when off."""
        kind = self._q("DISP:PERS:TYPE?").upper()
        if kind == "OFF":
            return None
        if kind.startswith("INF"):
            return math.inf
        return self._qf("DISP:PERS:TIME?")

    def set_grid(self, grid: GridStyle | str) -> GridStyle:
        """Set the grid: LINES, RETICLE or NONE; returns the read-back.

        Enum arg: <scope>.GridStyle.<TAB>
        """
        got = self._set_token("DISP:GRID:STYL", _token(grid, GridStyle, "grid"))
        return _answer(got, GridStyle, "grid")

    def get_grid(self) -> GridStyle:
        """The grid."""
        return _answer(self._q("DISP:GRID:STYL?"), GridStyle, "grid")

    def set_style(self, style: DisplayStyle | str) -> DisplayStyle:
        """Draw the waveform as VECTORS or DOTS; returns the read-back.

        Enum arg: <scope>.DisplayStyle.<TAB>
        """
        got = self._set_token("DISP:STYL", _token(style, DisplayStyle, "style"))
        return _answer(got, DisplayStyle, "style")

    def get_style(self) -> DisplayStyle:
        """Whether the waveform is drawn as vectors or dots."""
        return _answer(self._q("DISP:STYL?"), DisplayStyle, "style")

    def set_intensity_waveform(self, percent: float) -> float:
        """Set the waveform intensity in percent; returns the read-back."""
        p = _number(percent, "percent")
        return self._set_float("DISP:INT:WAV", p, "waveform intensity", rel=0.02)

    def set_intensity_grid(self, percent: float) -> float:
        """Set the grid intensity in percent; returns the read-back."""
        p = _number(percent, "percent")
        return self._set_float("DISP:INT:GRID", p, "grid intensity", rel=0.02)

    def set_date_time_state(self, state: bool) -> bool:
        """Show or hide the date and time on the screen; returns the read-back."""
        return self._set_bool("DISP:DTIM", state)

    def set_xy_state(self, state: bool) -> bool:
        """Switch to the XY display, or back to the YT display; returns the read-back."""
        got = self._set_token("DISP:MODE", "XY" if state else "YT")
        return got.upper() == "XY"

    def set_xy(self, x: str, y1: str, y2: str | None = None) -> XySettings:
        """Show the XY display: ``x`` on the horizontal axis, ``y1`` and ``y2`` vertical.

        UM p.420. Returns the read-back.
        """
        self.set_xy_state(True)
        self._w(f"DISP:XY:XSO {_source(x)}")
        self._w(f"DISP:XY:Y1S {_source(y1)}")
        if y2:
            self._w(f"DISP:XY:Y2S {_source(y2)}")
        return XySettings(
            xy_state=self._q("DISP:MODE?").upper() == "XY",
            x=self._q("DISP:XY:XSO?").lower(),
            y1=self._q("DISP:XY:Y1S?").lower(),
        )

    def cmd_clear(self) -> None:
        """Clear the screen and the persistence."""
        self._w("DISP:CLE")

    def get_settings(self) -> DisplaySettings:
        """Every display setting."""
        return DisplaySettings(
            persistence=self.get_persistence(),
            grid=self.get_grid(),
            style=self.get_style(),
            intensity_waveform=self._qf("DISP:INT:WAV?"),
            intensity_grid=self._qf("DISP:INT:GRID?"),
            date_time=self._get_bool("DISP:DTIM?"),
        )


# ---------------------------------------------------------------------------
# Math, reference, search, FFT, mask
# ---------------------------------------------------------------------------


def _read_trace(link: _RtbLink, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Time or frequency axis and REAL,32 data of a math, reference or spectrum trace."""
    link.write("FORM:DATA REAL,32")
    link.write("FORM:BORD LSBF")
    head = [float(x) for x in link.ask(f"{prefix}:DATA:HEAD?").split(",")]
    data = link.query_floats(f"{prefix}:DATA?")
    if int(head[2]) != len(data):
        raise InstrumentQueryError(
            f"{prefix} header says {int(head[2])} points, received {len(data)}"
        )
    return np.linspace(head[0], head[1], len(data), endpoint=False), data


class _Math(_RtbNode):
    """Math waveform 1 (UM p.349-352)."""

    def set_operation(
        self, operation: MathOperation | str, source1: str, source2: str | None = None
    ) -> str:
        """Show math waveform 1 as ``operation`` of the sources; returns the scope's expression.

        Enum arg: <scope>.MathOperation.<TAB>
        """
        member = _member(operation, MathOperation, "math operation")
        b = _source(source2, "source2") if source2 else ""
        expr = member.value.format(a=_source(source1, "source1"), b=b)
        self._w(f"CALC:MATH1:EXPR:DEF {quote(expr)}")
        self._w("CALC:MATH1:STAT ON")
        got = self._q("CALC:MATH1:EXPR:DEF?").strip('"')
        if not got.upper().startswith(expr):
            raise InstrumentError(f"math reads {got!r} after {expr!r}")
        return got

    def set_state(self, state: bool) -> bool:
        """Show or hide math waveform 1; returns the read-back."""
        # UNVERIFIED: the original driver only wrote OFF and never read this back.
        return self._set_bool("CALC:MATH1:STAT", state)

    def get_waveform(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the math waveform of the last acquisition as (t in s, values)."""
        return _read_trace(self.link, "CALC:MATH1")


class _Reference(_RtbNode):
    """One reference waveform, ref1 to ref4 (UM p.353-356)."""

    def __init__(self, scope: RohdeSchwarz_RTB2000, n: int):
        super().__init__(scope, f"REF{n}", n)

    def cmd_save_from(self, source: str) -> None:
        """Copy the present waveform of ``source`` into this reference and show it."""
        self._w(f"REFC{self.n}:SOUR {_source(source)}")
        self._w(f"REFC{self.n}:UPD")
        self._w(f"REFC{self.n}:STAT ON")

    def set_display_state(self, state: bool) -> bool:
        """Show or hide this reference waveform; returns the read-back."""
        # UNVERIFIED: the original driver never read this state back.
        return self._set_bool(f"REFC{self.n}:STAT", state)

    def get_waveform(self) -> tuple[np.ndarray, np.ndarray]:
        """Read this reference waveform as (t in s, values)."""
        return _read_trace(self.link, f"REFC{self.n}")


class _Search(_RtbNode):
    """Search in the acquired waveform (UM p.357-371). Run it on a stopped acquisition."""

    def _start(self, source: str, condition: str) -> None:
        self._w("SEAR:STAT ON")
        self._w(f"SEAR:SOUR {_source(source)}")
        self._w(f"SEAR:COND {condition}")

    def set_width(
        self,
        source: str,
        *,
        level: float,
        width: float,
        polarity: Polarity | str = Polarity.POSITIVE,
        condition: WidthCondition | str = WidthCondition.LONGER,
        delta: float | None = None,
    ) -> list[SearchResult]:
        """Search ``source`` for pulses whose width in s meets ``condition``; returns the hits.

        ``level`` is in volts, ``delta`` in seconds.
        """
        self._start(source, "WIDT")
        self._w(f"SEAR:TRIG:WIDT:POL {_token(polarity, Polarity, 'polarity')}")
        self._w(f"SEAR:TRIG:WIDT:LEV {_num(_number(level, 'level'))}")
        self._w(f"SEAR:TRIG:WIDT:RANG {_token(condition, WidthCondition, 'condition')}")
        self._w(f"SEAR:TRIG:WIDT:WIDT {_num(_number(width, 'width'))}")
        if delta is not None:
            self._w(f"SEAR:TRIG:WIDT:DELT {_num(_number(delta, 'delta'))}")
        return self.get_results()

    def set_edge(
        self, source: str, *, level: float, slope: Slope | str = Slope.RISE
    ) -> list[SearchResult]:
        """Search ``source`` for edges through ``level`` volts; returns the hits."""
        self._start(source, "EDGE")
        self._w(f"SEAR:TRIG:EDGE:SLOP {_token(slope, Slope, 'slope')}")
        self._w(f"SEAR:TRIG:EDGE:LEV {_num(_number(level, 'level'))}")
        return self.get_results()

    def get_count(self) -> int:
        """The number of search hits."""
        return int(self._qf("SEAR:RCO?"))

    def get_results(self, *, timeout: float = 3.0) -> list[SearchResult]:
        """The search hits; waits up to ``timeout`` s for the first one, else empty."""
        t0 = time.monotonic()
        while self.get_count() == 0 and time.monotonic() - t0 < timeout:
            time.sleep(_RESULT_POLL_S)
        if self.get_count() == 0:
            return []
        f = self._q("SEAR:RES:ALL?").split(",")
        return [
            SearchResult(
                time=float(f[i + 1]),
                type=f[i + 3].strip().lower(),
                polarity=f[i + 4].strip().lower(),
                value=float(f[i + 5]),
            )
            for i in range(0, len(f) - 5, 6)
        ]

    def set_state(self, state: bool) -> bool:
        """Turn the search on or off; returns the read-back."""
        # UNVERIFIED: the original driver only wrote OFF and never read this back.
        return self._set_bool("SEAR:STAT", state)


_FFT_RANGE_ROUNDS = 8  # MEAS 2026-10-18: the range opens up within a few rounds
_FFT_RATIO_POLL_S = 0.1


class _Fft(_RtbNode):
    """The scope FFT (UM p.408-418).

    The frequency range follows the time record: a stop frequency or span the
    record cannot give is refused with -222 (quirk RTB-13).
    """

    def set_source(
        self,
        source: str,
        *,
        window: FftWindow | str = FftWindow.HANNING,
        scale: FftScale | str = FftScale.DBV,
    ) -> FftSettings:
        """Show the FFT of ``source`` with ``window`` and ``scale``; returns the read-back.

        Enum args: <scope>.FftWindow.<TAB>, <scope>.FftScale.<TAB>
        """
        self._w("SPEC:STAT ON")
        self._w(f"SPEC:SOUR {_source(source)}")
        self._w(f"SPEC:FREQ:WIND:TYPE {_token(window, FftWindow, 'window')}")
        self._w(f"SPEC:FREQ:MAGN:SCAL {_token(scale, FftScale, 'magnitude scale')}")
        return self.get_settings()

    def get_settings(self) -> FftSettings:
        """The FFT source, window, scale, span and resolution bandwidth."""
        return FftSettings(
            source=self._q("SPEC:SOUR?").lower(),
            window=_answer(self._q("SPEC:FREQ:WIND:TYPE?"), FftWindow, "window"),
            scale=_answer(self._q("SPEC:FREQ:MAGN:SCAL?"), FftScale, "magnitude scale"),
            start=self._qf("SPEC:FREQ:STAR?"),
            stop=self._qf("SPEC:FREQ:STOP?"),
            rbw=self._qf("SPEC:FREQ:BAND?"),
        )

    def set_range(self, start: float, stop: float, *, rbw_ratio: int | None = None) -> FftSettings:
        """Set the frequency range in Hz; returns the read-back with the RBW it gives.

        The scope sets its timebase from the FFT range. Each write clamps to what
        the present timebase allows and moves the timebase, so the range opens up
        over a few rounds (MEAS 2026-10-18, RTB-13). A -222 inside a round is
        expected; the read-back after the rounds decides. ``rbw_ratio`` is span /
        RBW (1024 gives RBW 16.6 Hz on a 17 kHz span).
        """
        span, centre = float(stop - start), float(start + stop) / 2
        tolerance = 1e-3 * span

        def reached(got: FftSettings) -> bool:
            return abs(got.start - start) <= tolerance and abs(got.stop - stop) <= tolerance

        for _ in range(_FFT_RANGE_ROUNDS):
            self.link.write(f"SPEC:FREQ:SPAN {_num(span)}")
            self.link.write(f"SPEC:FREQ:CENT {_num(centre)}")
            self.link.drain_errors()
            got = self.get_settings()
            if reached(got):
                break
        if rbw_ratio is not None:  # after the range: the rounds reset the RBW
            self._w("SPEC:FREQ:BAND:RES:AUTO OFF")
            # MEAS 2026-11-05: a ratio written at once after AUTO OFF can give -222; the
            # same write holds some 100 ms later. The read-back decides.
            for _ in range(_FFT_RANGE_ROUNDS):
                self.link.write(f"SPEC:FREQ:BAND:RES:RAT {int(rbw_ratio)}")
                self.link.drain_errors()
                ratio = self._qf("SPEC:FREQ:BAND:RES:RAT?")
                if abs(ratio - rbw_ratio) <= 0.02 * rbw_ratio:
                    break
                time.sleep(_FFT_RATIO_POLL_S)
            else:
                raise InstrumentError(f"{self.name}: RBW ratio reads {ratio:g}, requested {rbw_ratio}")
            # MEAS 2026-11-05: a ratio write can move the range. Span and centre written
            # again bring it back and keep the ratio.
            for _ in range(_FFT_RANGE_ROUNDS):
                got = self.get_settings()
                if reached(got):
                    break
                self.link.write(f"SPEC:FREQ:SPAN {_num(span)}")
                self.link.write(f"SPEC:FREQ:CENT {_num(centre)}")
                self.link.drain_errors()
                time.sleep(_FFT_RATIO_POLL_S)
        if not reached(got):
            raise InstrumentError(
                f"{self.name}: FFT range reads {got.start:g}..{got.stop:g} Hz, "
                f"requested {start:g}..{stop:g}"
            )
        return got

    def get_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the spectrum as (frequency in Hz, magnitude in the set scale)."""
        return _read_trace(self.link, "SPEC:WAV:SPEC")

    def set_state(self, state: bool) -> bool:
        """Show or hide the FFT; returns the read-back."""
        # UNVERIFIED: the original driver only wrote OFF and never read this back.
        return self._set_bool("SPEC:STAT", state)


class _Mask(_RtbNode):
    """Mask test (UM p.403-407)."""

    def set_from_source(self, source: str, *, y_width: float = 0.4, x_width: float = 0.1) -> bool:
        """Make a mask from the present waveform of ``source``; returns the mask state.

        The mask is the waveform widened by ``y_width`` divisions vertically
        and ``x_width`` divisions horizontally.
        """
        self._w("MASK:STAT ON")
        self._w(f"MASK:SOUR {_source(source)}")
        self._w(f"MASK:YWID {_num(_number(y_width, 'y_width'))}")
        self._w(f"MASK:XWID {_num(_number(x_width, 'x_width'))}")
        self._w("MASK:CHC")
        # UNVERIFIED: the original driver never read the mask state back.
        return self._get_bool("MASK:STAT?")

    def cmd_run(self) -> None:
        """Reset the mask counters and start the mask test."""
        self._w("MASK:RES:COUN")
        self._w("MASK:TEST RUN")

    def cmd_stop(self) -> None:
        """Stop the mask test."""
        self._w("MASK:TEST STOP")

    def get_counts(self) -> MaskCounts:
        """The acquisitions tested and the violations found."""
        return MaskCounts(
            acquisitions=int(self._qf("MASK:COUN?")), violations=int(self._qf("MASK:VCO?"))
        )

    def set_state(self, state: bool) -> bool:
        """Turn the mask test on, or stop it and turn it off; returns the read-back."""
        if not state:
            self.cmd_stop()
        # UNVERIFIED: the original driver only wrote OFF and never read this back.
        return self._set_bool("MASK:STAT", state)


# ---------------------------------------------------------------------------
# Files and waveform data
# ---------------------------------------------------------------------------


class _File(_RtbNode):
    """Scope files (UM p.447-453): drives "/INT" and "/USB_FRONT", plus screenshots."""

    def get_drives(self) -> list[str]:
        """The scope's drives, for example ["/INT"]."""
        return [d.strip().strip('"') for d in self._q("MMEM:DRIV?").split(",")]

    def get_catalog(self, folder: str = "/INT") -> FileCatalog:
        """The files in ``folder`` on the scope, with the drive's used and free bytes."""
        reply = self._q(f"MMEM:CAT? {quote(folder)}")
        head = reply.partition('"')[0]
        used, free = (int(x) for x in head.strip().rstrip(",").split(",")[:2])
        files = []
        for item in re.findall(r'"([^"]*)"', reply):
            parts = item.split(",")
            size = int(parts[-1]) if parts[-1].strip().isdigit() else None
            files.append(FileEntry(name=parts[0], size=size))
        return FileCatalog(used=used, free=free, files=files)

    def _cd(self, folder: str) -> None:
        self._w(f"MMEM:CDIR {quote(folder)}")

    def get_size(self, name: str, *, folder: str = "/INT") -> int | None:
        """The size in bytes of file ``name`` in ``folder``, or None when it is not there."""
        for entry in self.get_catalog(folder).files:
            if entry.name.upper() == name.upper():
                return entry.size
        return None

    def cmd_save_setup(self, name: str, *, folder: str = "/INT") -> int:
        """Save the whole scope setup as ``name``.SET on the scope; returns its size in bytes."""
        name = name if name.upper().endswith(".SET") else name + ".SET"
        self._cd(folder)
        self._w(f"MMEM:STOR:STAT 1,{quote(name)}")
        size = self.get_size(name, folder=folder)
        if not size:
            raise InstrumentError(f"setup {folder}/{name} is not listed after the save")
        return size

    def cmd_recall_setup(self, name: str, *, folder: str = "/INT") -> None:
        """Recall the scope setup ``name``.SET from ``folder``."""
        name = name if name.upper().endswith(".SET") else name + ".SET"
        if self.get_size(name, folder=folder) is None:
            raise InstrumentError(f"no setup file {folder}/{name} on the scope")
        self._cd(folder)
        self._w(f"MMEM:LOAD:STAT 1,{quote(name)}")

    def cmd_save_waveform(self, source: str, name: str, *, folder: str = "/INT") -> int:
        """Export the waveform of ``source`` to ``name``.CSV on the scope; returns its size.

        Export while acquiring. After a stop the scope refused it with -200
        and -410; a later retest gave -200 both stopped and running, cause not
        found (quirk RTB-14).
        """
        stem = name.rsplit(".", 1)[0]
        self._w(f"EXP:WAV:SOUR {_source(source)}")
        self._w("FORM CSV")
        try:
            self._w(f"EXP:WAV:NAME {quote(f'{folder}/{stem}')}")
            self._w("EXP:WAV:SAVE")
        finally:
            self._w("FORM:DATA REAL,32")
        size = self.get_size(stem + ".CSV", folder=folder)
        if not size:
            raise InstrumentError(f"{folder}/{stem}.CSV is not listed after the export")
        return size

    def get_file(self, name: str, *, folder: str = "/INT") -> bytes:
        """Read file ``name`` from ``folder`` on the scope; returns its bytes.

        The block is read by its byte count, then the terminator: a line-based
        read stops at a newline inside a text file and leaves the terminator
        unread, which shifts every later reply (MEAS).
        """
        self._cd(folder)
        self.link.write(f"MMEM:DATA? {quote(name)}")
        head = self.link.read_bytes(2)
        if head[:1] != b"#":
            self.link.visa.clear()
            raise InstrumentQueryError(f"the file read reply starts {head!r}")
        n = int(self.link.read_bytes(int(head[1:2])))
        data = self.link.read_bytes(n)
        self.link.read_bytes(1)
        return data

    def cmd_delete(self, name: str, *, folder: str = "/INT") -> None:
        """Delete file ``name`` from ``folder`` on the scope."""
        self._cd(folder)
        self._w(f"MMEM:DEL {quote(name)}")
        if self.get_size(name, folder=folder) is not None:
            raise InstrumentError(f"{folder}/{name} is still listed after the delete")

    def get_screenshot(self, path: str | Path | None = None) -> bytes:
        """Read a PNG of the screen; returns its bytes and writes ``path`` on the PC when given."""
        self.link.write("HCOP:LANG PNG")
        data = self.link.visa.query_binary_values("HCOP:DATA?", datatype="B", container=bytes)
        if path is not None:
            Path(path).write_bytes(data)
        return data


class _Data(_RtbNode):
    """Analog waveform transfer to the PC."""

    def get_waveform(self, channels: Sequence[str], *, check_clip: bool = True) -> Capture:
        """Read the held acquisition of ``channels`` without a new one; returns a Capture.

        Use it after ``cmd_single()`` and ``cmd_wait_acquisition()`` for a
        capture that must start before the stimulus. Each channel is read at
        the full record for the screen range, which the scope gives only
        while stopped (quirk RTB-04). With ``check_clip`` it raises when a
        channel clips. Ends with the scope running.
        """
        scope = self._scope
        self.link.write("FORM:DATA REAL,32")
        self.link.write("FORM:BORD LSBF")
        out: dict[str, np.ndarray] = {}
        heads: dict[str, list[float]] = {}
        try:
            for name in channels:
                out[name], heads[name] = scope._read_channel(_channel_number(name))
            settings = scope.get_settings()
        finally:
            scope.cmd_run()
        if check_clip:
            for name in channels:
                c = settings.channels[name]
                edge = float(np.mean(np.abs(out[name] - c.offset) > _CLIP_DIVISIONS * c.scale))
                if edge > _CLIP_FRACTION:
                    raise InstrumentError(
                        f"{name} clips: {edge:.2%} of samples at the screen edge "
                        f"({c.scale} V/div, offset {c.offset} V); use set_auto_range"
                    )
        errors = self.link.drain_errors()
        if errors:
            raise InstrumentQueryError(f"the scope reported after the read: {'; '.join(errors)}")
        h = heads[channels[0]]
        t = np.linspace(h[0], h[1], len(out[channels[0]]), endpoint=False)
        return Capture(t, out, settings, heads)

    def meas_waveform(
        self, channels: Sequence[str], *, timeout: float = 5.0, check_clip: bool = True
    ) -> Capture:
        """Take one single acquisition and read ``channels``; returns a Capture.

        Raises InstrumentTimeoutError when the acquisition does not complete
        in ``timeout`` s (NORMAL trigger mode), and InstrumentError when a
        channel clips. Ends with the scope running.
        """
        self._scope.cmd_single()
        self._scope.cmd_wait_acquisition(timeout)
        return self.get_waveform(channels, check_clip=check_clip)

    def meas_waveform_roll(
        self, channels: Sequence[str], duration: float, *, scale: float | None = None
    ) -> Capture:
        """Record ``channels`` in roll mode for ``duration`` s; returns the screen points.

        ``scale`` is the time scale in s/div, default ``duration`` / 12. The
        trigger goes to AUTO. Ends with the scope running and roll mode off.
        """
        scope = self._scope
        d = _number(duration, "duration")
        tb = _number(scale, "scale") if scale is not None else d / 12
        scope.horz.set_roll_auto_state(True)
        if tb < scope.horz.get_roll_minimum():
            scope.horz.set_roll_minimum(tb)
        scope.horz.set_scale(tb)
        scope.trig.set_mode(TriggerMode.AUTO)
        scope.cmd_run()
        time.sleep(d + _ROLL_MARGIN_S)
        scope.cmd_stop()
        self.link.write("FORM:DATA REAL,32")
        self.link.write("FORM:BORD LSBF")
        out: dict[str, np.ndarray] = {}
        heads: dict[str, list[float]] = {}
        try:
            for name in channels:
                n = _channel_number(name)
                self.link.write(f"CHAN{n}:DATA:POIN DEF")
                heads[name] = [float(x) for x in self.link.ask(f"CHAN{n}:DATA:HEAD?").split(",")]
                out[name] = self.link.query_floats(f"CHAN{n}:DATA?")
            settings = scope.get_settings()
        finally:
            scope.cmd_run()
            scope.horz.set_roll_auto_state(False)
        h = heads[channels[0]]
        t = np.linspace(h[0], h[1], len(out[channels[0]]), endpoint=False)
        return Capture(t, out, settings, heads)


# ---------------------------------------------------------------------------
# Option nodes: history (K15), logic (B1), pattern generator and AFG (B6)
# ---------------------------------------------------------------------------


class _History(_RtbNode):
    """History and segmented memory, option K15 (UM p.372-382).

    One stored acquisition is a history acquisition (the manual says
    "segment"). Index 0 is the newest, -1, -2 ... are older (UM p.376).
    """

    def set_fast_segmentation_state(self, state: bool) -> bool:
        """Turn fast segmentation on or off; returns the read-back.

        Fast segmentation acquires into memory as fast as it can, with no
        display until a stop.
        """
        return self._set_bool("ACQ:SEGM:STAT", state)

    def get_fast_segmentation_state(self) -> bool:
        """Whether fast segmentation is on."""
        return self._get_bool("ACQ:SEGM:STAT?")

    def set_segmented_memory(self, count: int, *, record_length: int = 10_000) -> HistorySettings:
        """Turn on fast segmentation with ``count`` history acquisitions; returns the read-back.

        Each acquisition holds ``record_length`` points, 10 k minimum (1 k
        gives -222, quirk RTB-10). The scope must be acquiring: the record
        length applies at the next acquisition (quirk RTB-08). Fewer
        acquisitions fit at longer record lengths (UM p.324).
        """
        n = int(_number(count, "count"))
        acq = self._scope.acq
        acq.set_record_length(record_length)
        self._w("ACQ:MEM MAN")
        self._w(f"ACQ:COUN {n}")
        got = int(self._qf("ACQ:COUN?"))
        if got < n:
            raise InstrumentError(
                f"{got} history acquisitions available, requested {n}; fewer fit at longer "
                "record lengths (UM p.324)"
            )
        return HistorySettings(
            count=got,
            record_length=acq.get_record_length(),
            fast_segmentation=self.set_fast_segmentation_state(True),
        )

    def get_available_count(self) -> int:
        """The number of history acquisitions stored now."""
        return int(self._qf("ACQ:AVA?"))

    def cmd_acquire(self, count: int, *, timeout: float = 10.0) -> int:
        """Run until more than ``count`` history acquisitions are stored, then stop.

        Returns the number stored. Fast segmentation still misses some
        triggers (89 of 100 at 10 ms bursts), and each query while running
        misses more, so this polls every 0.5 s (quirk RTB-10). A stop drops
        the acquisition in progress (MEAS), so the wait runs one past
        ``count``.
        """
        scope = self._scope
        scope.cmd_run()
        t0 = time.monotonic()
        while True:
            n = self.get_available_count()
            if n > count:
                scope.cmd_stop()
                return self.get_available_count()
            if time.monotonic() - t0 > timeout:
                scope.cmd_stop()
                raise InstrumentTimeoutError(
                    f"{n} of {count} history acquisitions in {timeout} s; check the trigger"
                )
            time.sleep(_HISTORY_POLL_S)

    def get_timestamps(self) -> np.ndarray:
        """Time of each history acquisition relative to the newest, in s, oldest first."""
        reply = self._q("CHAN1:HIST:TSR:ALL?")
        return np.array([float(x) for x in reply.split(",") if x.strip()])

    def get_waveform(self, index: int, channels: Sequence[str]) -> Capture:
        """Read history acquisition ``index`` (0 newest, negative older) of ``channels``.

        The scope must be stopped. Returns a Capture with ``history_index`` set.
        """
        scope = self._scope
        channel_numbers = [_channel_number(name) for name in channels]
        for n in channel_numbers:
            self._w(f"CHAN{n}:HIST:CURR {int(index)}")
        self.link.write("FORM:DATA REAL,32")
        self.link.write("FORM:BORD LSBF")
        out: dict[str, np.ndarray] = {}
        heads: dict[str, list[float]] = {}
        for name, n in zip(channels, channel_numbers, strict=True):
            out[name], heads[name] = scope._read_channel(n)
        h = heads[channels[0]]
        t = np.linspace(h[0], h[1], len(out[channels[0]]), endpoint=False)
        return Capture(t, out, scope.get_settings(), heads, history_index=int(index))


class _Logic(_RtbNode):
    """Logic channels D0..D7 (pod 1) and D8..D15 (pod 2), option B1 (UM p.533-541).

    Threshold, hysteresis and decimation apply per pod; display and labels
    per channel.
    """

    def get_probe_connected(self, pod: int = 1) -> bool:
        """Whether a logic probe is plugged into ``pod``."""
        return self._get_bool(f"LOG{pod}:PROB?")

    def set_pod_state(self, pod: int, state: bool) -> bool:
        """Turn logic pod ``pod`` (1 or 2) on or off; returns the read-back."""
        return self._set_bool(f"LOG{pod}:STAT", state)

    def get_pod_state(self, pod: int = 1) -> bool:
        """Whether logic pod ``pod`` is on."""
        return self._get_bool(f"LOG{pod}:STAT?")

    def set_display_state(self, channel: int | str, state: bool) -> bool:
        """Show or hide logic channel ``channel`` (0..15 or "d0".."d15"); returns it."""
        m = _logic_number(channel)
        return self._set_bool(f"DIG{m}:DISP", state)

    def get_display_state(self, channel: int | str) -> bool:
        """Whether logic channel ``channel`` is shown."""
        return self._get_bool(f"DIG{_logic_number(channel)}:DISP?")

    def set_threshold(self, volts: float, *, pod: int = 1) -> float:
        """Set a user threshold in volts for logic pod ``pod``; returns the read-back."""
        v = _number(volts, "volts")
        self._w(f"LOG{pod}:THR USER")
        return self._set_float(f"DIG{(pod - 1) * 8}:THR", v, "logic threshold", rel=0.02)

    def set_threshold_preset(self, preset: LogicThreshold | str, *, pod: int = 1) -> float:
        """Set a threshold preset for pod ``pod``: TTL, ECL or CMOS; returns the level in V.

        Enum arg: <scope>.LogicThreshold.<TAB>
        """
        self._w(f"LOG{pod}:THR {_token(preset, LogicThreshold, 'logic threshold')}")
        return self.get_threshold(pod=pod)

    def get_threshold(self, *, pod: int = 1) -> float:
        """The threshold of logic pod ``pod`` in volts."""
        return self._qf(f"DIG{(pod - 1) * 8}:THR?")

    def get_threshold_preset(self, *, pod: int = 1) -> LogicThreshold:
        """The threshold preset of logic pod ``pod``; USER for a level in volts."""
        # UNVERIFIED: the original driver never ran this query; the long forms come
        # from its read-back table only.
        raw = self._q(f"LOG{pod}:THR?")
        return _answer(raw, LogicThreshold, "logic threshold", _LOGIC_THRESHOLD_EXTRA)

    def set_hysteresis(self, size: Hysteresis | str, *, pod: int = 1) -> Hysteresis:
        """Set the hysteresis of logic pod ``pod``; returns the read-back.

        Enum arg: <scope>.Hysteresis.<TAB>
        """
        got = self._set_token(f"LOG{pod}:HYST", _token(size, Hysteresis, "hysteresis"))
        return _answer(got, Hysteresis, "hysteresis")

    def set_label(self, channel: int | str, text: str | None) -> str | None:
        """Label logic channel ``channel``; None hides the label. Returns the read-back."""
        m = _logic_number(channel)
        if text is None:
            self._w(f"DIG{m}:LAB:STAT OFF")
            return None
        self._w(f"DIG{m}:LAB {quote(text)}")
        self._w(f"DIG{m}:LAB:STAT ON")
        got = self._q(f"DIG{m}:LAB?").strip('"')
        if got != text:
            raise InstrumentError(f"d{m} label reads {got!r} after {text!r}")
        return got

    def get_live_level(self, channel: int | str) -> LogicActivity:
        """The live state of logic channel ``channel`` with no acquisition: LOW, HIGH or TOGGLE.

        UM p.534. The scope reports the lowest and highest level seen since
        the previous read, so the first read after a change can say TOGGLE.
        Read again for the settled state (MEAS 2026-10-16: settled after 0.5 s
        and one discarded read).
        """
        m = _logic_number(channel)
        lo = self._q(f"DIG{m}:CURR:STAT:MIN?")
        hi = self._q(f"DIG{m}:CURR:STAT:MAX?")
        by_pair = {("0", "0"): LogicActivity.LOW, ("1", "1"): LogicActivity.HIGH}
        return by_pair.get((lo, hi), LogicActivity.TOGGLE)

    def set_pod(
        self,
        channels: Sequence[int | str] = (0, 1, 2, 3),
        *,
        threshold: LogicThreshold | float = LogicThreshold.TTL,
        hysteresis: Hysteresis | str | None = None,
        labels: Mapping[int, str] | None = None,
    ) -> LogicPodSettings:
        """Turn on the pod of ``channels`` and show only those; returns the read-back.

        ``channels`` are numbers 0..15 or names "d0".."d15", all on one pod.
        ``threshold`` is a preset or a level in volts. Raises when the logic
        probe is not plugged in.
        """
        wanted = [_logic_number(c) for c in channels]
        pod = 1 if wanted[0] < 8 else 2
        if any((1 if m < 8 else 2) != pod for m in wanted):
            raise ValueError(f"channels {channels!r} span both pods; pass one pod at a time")
        if not self.get_probe_connected(pod):
            raise InstrumentError(f"logic probe pod {pod} is not connected")
        if not self.set_pod_state(pod, True):
            raise InstrumentError(f"logic pod {pod} did not turn on")
        for m in range((pod - 1) * 8, pod * 8):
            self.set_display_state(m, m in wanted)
        if isinstance(threshold, LogicThreshold):
            level = self.set_threshold_preset(threshold, pod=pod)
        else:
            level = self.set_threshold(threshold, pod=pod)
        hyst = self.set_hysteresis(hysteresis, pod=pod) if hysteresis is not None else None
        for m, text in (labels or {}).items():
            self.set_label(m, text)
        return LogicPodSettings(pod=pod, channels=wanted, threshold=level, hysteresis=hyst)

    def get_waveform(self, channels: Sequence[int | str] = ("d0", "d1", "d2", "d3")) -> Capture:
        """Read the held acquisition's logic ``channels`` as 0/1 samples; returns a Capture.

        No new acquisition. Each channel is read over the screen range. The
        Capture's channels are keyed "d0".."d15". Ends with the scope running.
        """
        scope = self._scope
        self.link.write("FORM ASC")
        out: dict[str, np.ndarray] = {}
        heads: dict[str, list[float]] = {}
        try:
            for name in channels:
                m = _logic_number(name)
                key = f"d{m}"
                self.link.write(f"DIG{m}:DATA:POIN DMAX")
                head = [float(x) for x in self.link.ask(f"DIG{m}:DATA:HEAD?").split(",")]
                with self.link.temporary_timeout(_LOGIC_READ_TIMEOUT_MS):
                    reply = self.link.ask(f"DIG{m}:DATA?")
                values = np.array(
                    [int(float(v)) for v in reply.split(",") if v.strip()], dtype=np.uint8
                )
                if int(head[2]) != len(values):
                    raise InstrumentQueryError(
                        f"D{m} header says {int(head[2])} points, received {len(values)}"
                    )
                out[key], heads[key] = values, head
            settings = scope.get_settings()
        finally:
            scope.cmd_run()
        errors = self.link.drain_errors()
        if errors:
            raise InstrumentQueryError(f"the scope reported after the read: {'; '.join(errors)}")
        first = next(iter(out))
        h = heads[first]
        t = np.linspace(h[0], h[1], len(out[first]), endpoint=False)
        return Capture(t, out, settings, heads)

    def meas_waveform(
        self, channels: Sequence[int | str] = ("d0", "d1", "d2", "d3"), *, timeout: float = 5.0
    ) -> Capture:
        """Take one single acquisition and read logic ``channels``; returns a Capture.

        Set the trigger first, for example ``trig.set_edge("d3", slope=...)``.
        Ends with the scope running.
        """
        self._scope.cmd_single()
        self._scope.cmd_wait_acquisition(timeout)
        return self.get_waveform(channels)


class _PatternGenerator(_RtbNode):
    """Pattern generator on the front pins P0..P3 (S0..S3 in the manual), option B6.

    UM p.293-298, p.555-562. The helpers set the pattern and leave the output
    as it is: ``set_output_state(True)`` drives the pins (design standard 6.2
    rule 4).
    """

    def set_output_state(self, state: bool) -> bool:
        """Drive the pattern pins, or stop driving them; returns the read-back."""
        return self._set_bool("PGEN:PATT:STAT", state)

    def get_output_state(self) -> bool:
        """Whether the pattern pins are driven."""
        return self._get_bool("PGEN:PATT:STAT?")

    def _set_function(self, function: PatternGenFunction | str) -> PatternGenFunction:
        """Select the pattern function (PGEN:FUNC); returns the read-back."""
        self._w(f"PGEN:FUNC {_token(function, PatternGenFunction, 'pattern function')}")
        return self.get_function()

    def get_function(self) -> PatternGenFunction:
        """The pattern generator function."""
        raw = self._q("PGEN:FUNC?")
        return _answer(raw, PatternGenFunction, "pattern function", _PATTERN_EXTRA)

    def set_square(
        self, freq: float, *, duty: float = 50.0, inverted: bool = False
    ) -> PatternSettings:
        """Set a square wave on pin P0 at ``freq`` Hz and ``duty`` percent; returns it.

        UM p.293. P1..P3 stay low.
        """
        # UNVERIFIED: the original driver turned the output on at the end of this helper; that
        # these writes land with the output off was not run on the bench.
        self._set_function(PatternGenFunction.SQUARE)
        f = self._set_float("PGEN:PATT:FREQ", _number(freq, "freq"), "pattern frequency", 0.01)
        d = self._set_float("PGEN:PATT:SQU:DCYC", _number(duty, "duty"), "pattern duty", 0.01)
        self._w(f"PGEN:PATT:SQU:POL {'INV' if inverted else 'NORM'}")
        return PatternSettings(
            output_state=self.get_output_state(),
            function=PatternGenFunction.SQUARE,
            frequency=f,
            duty=d,
            inverted=not self._q("PGEN:PATT:SQU:POL?").upper().startswith("NORM"),
        )

    def set_counter(
        self, freq: float, *, direction: CountDirection | str = CountDirection.UP
    ) -> PatternSettings:
        """Set a 4-bit counter on P0..P3 that steps at ``freq`` Hz; returns the read-back.

        Enum arg: <scope>.CountDirection.<TAB>

        P0 toggles at freq / 2 and P3 at freq / 16.
        """
        # UNVERIFIED: the original driver turned the output on at the end of this helper.
        member = _member(direction, CountDirection, "count direction")
        self._set_function(PatternGenFunction.COUNTER)
        f = self._set_float("PGEN:PATT:COUN:FREQ", _number(freq, "freq"), "counter frequency", 0.01)
        self._w(f"PGEN:PATT:COUN:DIR {member.value}")
        got = _answer(self._q("PGEN:PATT:COUN:DIR?"), CountDirection, "count direction")
        if got is not member:
            raise InstrumentError(f"counter direction reads {got.name} after {member.name}")
        return PatternSettings(
            output_state=self.get_output_state(),
            function=PatternGenFunction.COUNTER,
            frequency=f,
            direction=got,
        )

    def set_arbitrary(
        self,
        values: Sequence[int],
        sample_time: float,
        *,
        burst_cycles: int | None = None,
        idle_time: float | None = None,
    ) -> PatternSettings:
        """Set an arbitrary 4-bit pattern on P0..P3, one value per ``sample_time`` s.

        ``values`` are 0..15, 1 to 2048 of them, repeated; ``sample_time`` is
        20 ns to 42 s. With ``burst_cycles`` the pattern plays that many times,
        then idles ``idle_time`` s. The data is a plain value list and the
        length follows it (MEAS). Returns the read-back.
        """
        # UNVERIFIED: the original driver turned the output on at the end of this helper.
        v = [int(x) for x in values]
        if not 1 <= len(v) <= 2048 or any(not 0 <= x <= 15 for x in v):
            raise ValueError("arbitrary pattern: 1 to 2048 values, each 0..15")
        self._set_function(PatternGenFunction.ARBITRARY)
        self._w("PGEN:PATT:TRIG:MODE CONT")
        st = self._set_float(
            "PGEN:PATT:STIM", _number(sample_time, "sample_time"), "pattern sample time", 0.01
        )
        self._w(f"PGEN:PATT:ARB:DATA:LENG {len(v)}")
        self._w("PGEN:PATT:ARB:DATA " + ",".join(map(str, v)))
        got = [int(float(x)) for x in self._q("PGEN:PATT:ARB:DATA?").split(",") if x.strip()]
        if got != v:
            raise InstrumentError(f"arbitrary pattern reads {got[:16]}... after {v[:16]}...")
        self._w(f"PGEN:PATT:BURS:STAT {'ON' if burst_cycles else 'OFF'}")
        if burst_cycles:
            self._w(f"PGEN:PATT:BURS:NCYC {int(burst_cycles)}")
            if idle_time is not None:
                it = _number(idle_time, "idle_time")
                self._set_float("PGEN:PATT:ITIM", it, "pattern idle time", 0.01)
        return PatternSettings(
            output_state=self.get_output_state(),
            function=PatternGenFunction.ARBITRARY,
            values=got,
            sample_time=st,
            burst_cycles=burst_cycles,
        )

    def set_manual(self, value: int) -> int:
        """Set a static 4-bit value on P0..P3, bit 0 on P0; returns the read-back."""
        # UNVERIFIED: the original driver turned the output on at the end of this helper.
        if not 0 <= int(value) <= 15:
            raise ValueError("manual pattern: 0..15")
        self._set_function(PatternGenFunction.MANUAL)
        for s in range(4):
            self._w(f"PGEN:MAN:STAT{s} {'ON' if int(value) >> s & 1 else 'OFF'}")
        got = sum(self._get_bool(f"PGEN:MAN:STAT{s}?") << s for s in range(4))
        if got != int(value):
            raise InstrumentError(f"manual pattern reads {got} after {value}")
        return got

    def get_settings(self) -> PatternSettings:
        """The pattern function and output state, with frequency for square and counter."""
        f = self.get_function()
        out = self.get_output_state()
        if f is PatternGenFunction.COUNTER:
            direction = _answer(self._q("PGEN:PATT:COUN:DIR?"), CountDirection, "direction")
            freq = self._qf("PGEN:PATT:COUN:FREQ?")
            return PatternSettings(out, f, frequency=freq, direction=direction)
        if f is PatternGenFunction.SQUARE:
            freq = self._qf("PGEN:PATT:FREQ?")
            duty = self._qf("PGEN:PATT:SQU:DCYC?")
            return PatternSettings(out, f, frequency=freq, duty=duty)
        return PatternSettings(out, f)


_AFG_PERIODIC = {
    "sine": AfgFunction.SINE,
    "square": AfgFunction.SQUARE,
    "triangle": AfgFunction.TRIANGLE,
    "ramp": AfgFunction.RAMP,
    "sinc": AfgFunction.SINC,
    "exponential": AfgFunction.EXPONENTIAL,
}


class _Afg(_RtbNode):
    """Built-in function generator, option B6 (UM p.284-292, p.546-554).

    The waveform leaves at [Gen Out], and at [Aux Out] with
    ``aux_out.set_source(AuxOutSource.GENERATOR)``. The waveform helpers set
    the waveform and leave the output as it is: ``set_output_state(True)``
    drives the connector (design standard 6.2 rule 4). Set the load first:
    the amplitude is into that load.
    """

    def set_output_state(self, state: bool) -> bool:
        """Drive [Gen Out], or stop driving it; returns the read-back."""
        return self._set_bool("WGEN:OUTP", state)

    def get_output_state(self) -> bool:
        """Whether [Gen Out] is driven."""
        return self._get_bool("WGEN:OUTP?")

    def _set_function(self, function: AfgFunction | str) -> AfgFunction:
        """Select the generator function (WGEN:FUNC); returns the read-back."""
        got = self._set_token("WGEN:FUNC", _token(function, AfgFunction, "generator function"))
        return _answer(got, AfgFunction, "generator function", _SINE_EXTRA)

    def set_frequency(self, hz: float) -> float:
        """Set the frequency in Hz without changing the function; returns the read-back.

        Public so a loop that steps frequency writes nothing else (design
        standard 5.3).
        """
        return self._set_float("WGEN:FREQ", _number(hz, "hz"), "generator frequency", rel=0.01)

    def get_frequency(self) -> float:
        """The frequency in Hz."""
        return self._qf("WGEN:FREQ?")

    def set_amplitude(self, volts: float) -> float:
        """Set the amplitude in Vpp, 0.06 to 6 (UM p.546), into the set load; returns it."""
        v = _number(volts, "volts")
        return self._set_float("WGEN:VOLT", v, "generator amplitude", rel=0.01)

    def get_amplitude(self) -> float:
        """The amplitude in Vpp."""
        return self._qf("WGEN:VOLT?")

    def set_offset(self, volts: float) -> float:
        """Set the DC offset in volts; returns the read-back, checked to 10 mV."""
        v = _number(volts, "volts")
        self._w(f"WGEN:VOLT:OFFS {_num(v)}")
        got = self.get_offset()
        if abs(got - v) > 0.01:
            raise InstrumentError(f"generator offset: requested {v:g}, the scope applied {got:g}")
        return got

    def get_offset(self) -> float:
        """The DC offset in volts."""
        return self._qf("WGEN:VOLT:OFFS?")

    def set_load(self, load: float | Load | str) -> float:
        """Set the load the amplitude is into: 50 ohm, or Load.HIGHZ; returns ohms.

        Enum arg: <scope>.Load.<TAB>

        Returns 50.0, or math.inf for high impedance.
        """
        if isinstance(load, numbers.Real) and not isinstance(load, bool):
            if math.isinf(load):
                token = Load.HIGHZ.value
            elif float(load) == 50.0:
                token = Load.FIFTY.value
            else:
                raise ValueError(f"load must be 50 ohm or Load.HIGHZ, got {load!r}")
        else:
            token = _token(load, Load, "load")
        self._w(f"WGEN:OUTP:LOAD {token}")
        return self.get_load()

    def get_load(self) -> float:
        """The load the amplitude is into, in ohms: 50.0, or math.inf for high impedance."""
        return 50.0 if self._q("WGEN:OUTP:LOAD?").upper().startswith("R50") else math.inf

    def _set_duty(self, percent: float) -> float:
        """Set the pulse duty cycle in percent (WGEN:FUNC:PULS:DCYC); returns the read-back."""
        p = _number(percent, "percent")
        return self._set_float("WGEN:FUNC:PULS:DCYC", p, "generator duty", rel=0.01)

    def _set_periodic(self, function: AfgFunction, freq: float, ampl: float, offset: float) -> None:
        # Write order as in the original helper, which ran on the bench: function,
        # frequency, amplitude, offset. The original helper set the load between
        # function and frequency; here the load is set separately, first.
        self._set_function(function)
        self.set_frequency(freq)
        self.set_amplitude(ampl)
        self.set_offset(offset)

    def set_waveform_sine(self, freq: float, ampl: float, *, offset: float = 0.0) -> WaveformState:
        """Output a sine at ``freq`` Hz and ``ampl`` Vpp; returns the read-back."""
        self._set_periodic(AfgFunction.SINE, freq, ampl, offset)
        return self.get_waveform()

    def set_waveform_square(
        self, freq: float, ampl: float, *, offset: float = 0.0
    ) -> WaveformState:
        """Output a square wave at ``freq`` Hz and ``ampl`` Vpp; returns the read-back."""
        self._set_periodic(AfgFunction.SQUARE, freq, ampl, offset)
        return self.get_waveform()

    def set_waveform_pulse(
        self, freq: float, ampl: float, *, offset: float = 0.0, duty: float | None = None
    ) -> WaveformState:
        """Output a pulse at ``freq`` Hz and ``ampl`` Vpp, ``duty`` in percent; returns it."""
        self._set_periodic(AfgFunction.PULSE, freq, ampl, offset)
        if duty is not None:
            self._set_duty(duty)
        return self.get_waveform()

    def set_waveform_triangle(
        self, freq: float, ampl: float, *, offset: float = 0.0
    ) -> WaveformState:
        """Output a triangle at ``freq`` Hz and ``ampl`` Vpp; returns the read-back."""
        self._set_periodic(AfgFunction.TRIANGLE, freq, ampl, offset)
        return self.get_waveform()

    def set_waveform_ramp(self, freq: float, ampl: float, *, offset: float = 0.0) -> WaveformState:
        """Output a ramp at ``freq`` Hz and ``ampl`` Vpp; returns the read-back."""
        self._set_periodic(AfgFunction.RAMP, freq, ampl, offset)
        return self.get_waveform()

    def set_waveform_sinc(self, freq: float, ampl: float, *, offset: float = 0.0) -> WaveformState:
        """Output a sinc pulse at ``freq`` Hz and ``ampl`` Vpp; returns the read-back."""
        self._set_periodic(AfgFunction.SINC, freq, ampl, offset)
        return self.get_waveform()

    def set_waveform_exponential(
        self, freq: float, ampl: float, *, offset: float = 0.0
    ) -> WaveformState:
        """Output an exponential at ``freq`` Hz and ``ampl`` Vpp; returns the read-back."""
        self._set_periodic(AfgFunction.EXPONENTIAL, freq, ampl, offset)
        return self.get_waveform()

    def set_waveform_dc(self, offset: float) -> WaveformState:
        """Output a DC level of ``offset`` volts; returns the read-back."""
        self._set_function(AfgFunction.DC)
        self.set_offset(offset)
        return self.get_waveform()

    def get_waveform(self) -> WaveformState:
        """The complete output setting; duty is filled for a pulse only."""
        raw = self._q("WGEN:FUNC?")
        function = _answer(raw, AfgFunction, "generator function", _SINE_EXTRA)
        duty = self._qf("WGEN:FUNC:PULS:DCYC?") if function is AfgFunction.PULSE else None
        return WaveformState(
            function=function.name,
            frequency=self.get_frequency(),
            amplitude=self.get_amplitude(),
            offset=self.get_offset(),
            output_state=self.get_output_state(),
            duty=duty,
        )

    def set_burst(
        self, cycles: int = 1, *, phase: float = 0.0, continuous: bool = False
    ) -> AfgBurstSettings:
        """Turn on bursts of ``cycles`` waveform cycles from ``phase`` degrees; returns them.

        UM p.290, p.550. With ``continuous`` the bursts repeat; without it,
        each burst waits for ``cmd_burst_trigger()``.
        """
        self._w("WGEN:BURS ON")
        n = int(self._set_float("WGEN:BURS:NCYC", float(int(cycles)), "burst cycles"))
        ph = _number(phase, "phase")
        if ph:
            self._set_float("WGEN:BURS:PHAS", ph, "burst phase", rel=0.01)
        mode = self._set_token("WGEN:BURS:TRIG", "CONT" if continuous else "SING").upper()
        got = AfgBurstSettings(
            enabled=self._get_bool("WGEN:BURS?"),
            cycles=n,
            # UNVERIFIED: the original driver read the phase back only after writing a
            # nonzero phase.
            phase=self._qf("WGEN:BURS:PHAS?"),
            continuous=not mode.startswith("SING"),
        )
        if not got.enabled or got.continuous != continuous:
            raise InstrumentError(f"generator burst reads {got}")
        return got

    def set_burst_state(self, state: bool) -> bool:
        """Turn burst mode on or off with its present settings; returns the read-back."""
        return self._set_bool("WGEN:BURS", state)

    def cmd_burst_trigger(self) -> None:
        """Fire one burst. Requires burst mode without ``continuous``.

        Sent with no completion wait: with a single acquisition pending, the
        wait also waits for that acquisition and stalls if it never triggers
        (original driver note).
        """
        self.link.write("WGEN:BURS:TRIG:SING")


class _AuxOut(_RtbNode):
    """The [Aux Out] connector (UM p.87, p.465)."""

    def set_source(self, source: AuxOutSource | str) -> AuxOutSource:
        """Set what drives [Aux Out]; returns the read-back.

        Enum arg: <scope>.AuxOutSource.<TAB>

        OFF, a pulse per TRIGGER, the 10 MHz REFERENCE, MASK violations, or
        the GENERATOR waveform. The mode survives a reset (quirk RTB-11).
        """
        got = self._set_token("TRIG:OUT:MODE", _token(source, AuxOutSource, "aux out source"))
        return _answer(got, AuxOutSource, "aux out source")

    def get_source(self) -> AuxOutSource:
        """What drives [Aux Out]."""
        return _answer(self._q("TRIG:OUT:MODE?"), AuxOutSource, "aux out source")

    def set_trigger_pulse(
        self, *, width: float = 1e-6, polarity: Polarity | str = Polarity.POSITIVE
    ) -> TriggerOutSettings:
        """Drive [Aux Out] with a pulse of ``width`` s per trigger event; returns it.

        The pulse runs asynchronously to the acquisition (UM p.87, p.465).
        """
        member = _member(polarity, Polarity, "polarity")
        source = self.set_source(AuxOutSource.TRIGGER)
        w = self._set_float("TRIG:OUT:PLEN", _number(width, "width"), "trigger out width", 0.01)
        got = _answer(self._set_token("TRIG:OUT:POL", member.value), Polarity, "polarity")
        if got is not member:
            raise InstrumentError(f"trigger out polarity reads {got.name} after {member.name}")
        return TriggerOutSettings(source=source, width=w, polarity=got)


# ---------------------------------------------------------------------------
# Serial and parallel bus decode on bus 1 (options K1, K2, K3, B1)
# ---------------------------------------------------------------------------


class _BusBase(_RtbNode):
    """Bus 1 decode (UM p.475-545). The protocols present follow the options."""

    _B = 1

    def _bw(self, cmd: str) -> None:
        self._w(f"BUS{self._B}:{cmd}")

    def _bq(self, cmd: str) -> str:
        return self._q(f"BUS{self._B}:{cmd}?").strip('"')

    def _start(self, kind: str, sources: Sequence[str | None], threshold: float) -> str:
        self._bw(f"TYPE {kind}")
        got = self._bq("TYPE")
        if got.upper()[:3] != kind[:3]:
            raise InstrumentError(f"bus type reads {got!r} after {kind!r}")
        for src in sources:
            if src and src.strip().lower().startswith("ch"):
                self._scope._channel(src).set_threshold(threshold)
        self._bw("STAT ON")
        return got.upper()

    def set_state(self, state: bool) -> bool:
        """Turn bus 1 decode on or off; returns the read-back."""
        # UNVERIFIED: the original driver only wrote OFF and never read this back.
        return self._set_bool(f"BUS{self._B}:STAT", state)

    def set_format(self, fmt: BusFormat | str) -> BusFormat:
        """Set the number format of decoded words on the screen; returns the read-back.

        Enum arg: <scope>.BusFormat.<TAB>
        """
        self._bw(f"FORM {_token(fmt, BusFormat, 'bus format')}")
        return _answer(self._bq("FORM"), BusFormat, "bus format")

    def get_frames(self) -> list[BusFrame]:
        """The decoded frames of the last acquisition, values as integers. Stop first.

        Detail fields are read only for frames whose state is "ok": on any
        other frame the scope does not answer and the query times out (quirk
        RTB-15). Each query gets 1.5 s, and a timed-out one reads empty after
        the link and error queue are cleared.
        """
        kind = self._bq("TYPE").upper()

        def q(cmd: str) -> str:
            try:
                with self.link.temporary_timeout(_BUS_QUERY_TIMEOUT_MS):
                    return self._bq(cmd)
            except (visa.VisaIOError, InstrumentQueryError):
                with contextlib.suppress(Exception):
                    self.link.visa.clear()
                    self.link.drain_errors()
                return ""

        def count(cmd: str) -> int:
            return _to_int(q(cmd)) or 0

        def start(prefix: str, n: int) -> float:
            return float(q(f"{prefix}:FRAM{n}:STAR") or "nan")

        out: list[BusFrame] = []
        if kind.startswith("UART"):
            for n in range(1, count("UART:FCO") + 1):
                state = q(f"UART:FRAM{n}:STAT").lower()
                words = [
                    _to_int(q(f"UART:FRAM{n}:WORD{o}:VAL"))
                    for o in range(1, count(f"UART:FRAM{n}:WCO") + 1)
                ]
                out.append(BusFrame(start=start("UART", n), state=state, words=words))
        elif kind.startswith(("SSPI", "SPI")):
            for n in range(1, count("SPI:FCO") + 1):
                state = q(f"SPI:FRAM{n}:STAT").lower()
                words = []
                if state == "ok":
                    words = [
                        _to_int(q(f"SPI:FRAM{n}:WORD{o}:MOSI"))
                        for o in range(1, count(f"SPI:FRAM{n}:WCO") + 1)
                    ]
                out.append(BusFrame(start=start("SPI", n), state=state, words=words))
        elif kind.startswith("I2C"):
            for n in range(1, count("I2C:FCO") + 1):
                state = q(f"I2C:FRAM{n}:STAT").lower()
                frame = BusFrame(start=start("I2C", n), state=state)
                if state == "ok":
                    frame = BusFrame(
                        start=frame.start,
                        state=state,
                        address=_to_int(q(f"I2C:FRAM{n}:ADDR")),
                        access=q(f"I2C:FRAM{n}:ACC").lower(),
                        words=[
                            _to_int(q(f"I2C:FRAM{n}:BYTE{o}:VAL"))
                            for o in range(1, count(f"I2C:FRAM{n}:BCO") + 1)
                        ],
                    )
                out.append(frame)
        elif kind.startswith("CAN"):
            for n in range(1, count("CAN:FCO") + 1):
                state = q(f"CAN:FRAM{n}:STAT").lower()
                frame = BusFrame(start=start("CAN", n), state=state)
                if state == "ok":
                    frame = BusFrame(
                        start=frame.start,
                        state=state,
                        frame_type=q(f"CAN:FRAM{n}:TYPE").lower(),
                        identifier=_to_int(q(f"CAN:FRAM{n}:IDV")),
                        dlc=_to_int(q(f"CAN:FRAM{n}:DLCV")),
                        checksum=_to_int(q(f"CAN:FRAM{n}:CSV")),
                        words=[
                            _to_int(q(f"CAN:FRAM{n}:BYTE{o}:VAL"))
                            for o in range(1, count(f"CAN:FRAM{n}:BCO") + 1)
                        ],
                    )
                out.append(frame)
        elif kind.startswith("LIN"):
            for n in range(1, count("LIN:FCO") + 1):
                state = q(f"LIN:FRAM{n}:STAT").lower()
                frame = BusFrame(start=start("LIN", n), state=state)
                if state == "ok":
                    frame = BusFrame(
                        start=frame.start,
                        state=state,
                        identifier=_to_int(q(f"LIN:FRAM{n}:IDV")),
                        checksum=_to_int(q(f"LIN:FRAM{n}:CSV")),
                        words=[
                            _to_int(q(f"LIN:FRAM{n}:BYTE{o}:VAL"))
                            for o in range(1, count(f"LIN:FRAM{n}:BCO") + 1)
                        ],
                    )
                out.append(frame)
        elif kind.startswith(("PAR", "CPAR")):
            p = "CPAR" if kind.startswith("CPAR") else "PAR"
            for n in range(1, count(f"{p}:FCO") + 1):
                state = q(f"{p}:FRAM{n}:STAT").lower()
                data: list[int | None] = [
                    int(float(v)) for v in q(f"{p}:FRAM{n}:DATA").split(",") if v.strip()
                ]
                stop = float(q(f"{p}:FRAM{n}:STOP") or "nan")
                out.append(BusFrame(start=start(p, n), state=state, words=data, stop=stop))
        else:
            raise InstrumentError(f"no frame readout for bus type {kind!r}")
        return out


class _BusUart(_BusBase):
    """UART decode and the UART bus trigger, option K2."""

    def set_uart(
        self,
        rx: str,
        baud: float,
        *,
        threshold: float = 0.5,
        bits: int = 8,
        parity: Parity | str = Parity.NONE,
        stop_bits: float = 1,
        idle: IdleLevel | str = IdleLevel.HIGH,
        tx: str | None = None,
    ) -> str:
        """Decode a UART on ``rx`` at ``baud`` bit/s; returns the bus type read back.

        ``threshold`` (V) is written to every analog source channel.
        ``stop_bits`` is 1, 1.5 or 2.
        """
        stop_tokens = {1.0: "B1", 1.5: "B1_5", 2.0: "B2"}
        sb = _number(stop_bits, "stop_bits")
        if sb not in stop_tokens:
            raise ValueError(f"stop_bits must be 1, 1.5 or 2, got {stop_bits!r}")
        got = self._start("UART", [rx, tx], _number(threshold, "threshold"))
        self._bw(f"UART:RX:SOUR {_source(rx, 'rx')}")
        self._bw(f"UART:TX:SOUR {_source(tx, 'tx') if tx else 'NONE'}")
        self._bw(f"UART:POL {_token(idle, IdleLevel, 'idle level')}")
        self._bw(f"UART:SSIZ {int(bits)}")
        self._bw(f"UART:PAR {_token(parity, Parity, 'parity')}")
        self._bw(f"UART:SBIT {stop_tokens[sb]}")
        self._bw(f"UART:BAUD {int(_number(baud, 'baud'))}")
        return got

    def set_uart_trigger(
        self,
        *,
        mode: UartTriggerMode | str = UartTriggerMode.PATTERN,
        data: bytes = b"",
        offset: int = 0,
        line: UartLine | str = UartLine.RX,
        trigger_mode: TriggerMode | str = TriggerMode.NORMAL,
    ) -> UartTriggerSettings:
        """Trigger on the UART decoded by ``set_uart``; returns the read-back.

        Enum args: <scope>.UartTriggerMode.<TAB>, <scope>.UartLine.<TAB>

        For PATTERN the trigger matches the symbols in ``data`` (1 to 4) after
        ``offset`` ignored symbols. The scope matches each symbol LSB first,
        the order on the wire, so the driver writes them that way (quirk
        RTB-16). RX is the default line and its header is refused with -113,
        so the line is written for TX only (quirk RTB-16).
        """
        trigger_line = _member(line, UartLine, "uart line")
        member = _member(mode, UartTriggerMode, "uart trigger mode")
        self._w("TRIG:A:TYPE BUS")
        self._w("TRIG:A:SOUR SBUS1")
        if trigger_line is UartLine.TX:
            self._w("TRIG:A:SOUR:UART TX")
        self._w(f"TRIG:A:UART:MODE {member.value}")
        if member is UartTriggerMode.PATTERN:
            if not 1 <= len(data) <= 4:
                raise ValueError("UART trigger pattern: 1 to 4 symbols")
            bits = "".join(f"{b:08b}"[::-1] for b in data)
            self._w(f"TRIG:A:UART:PATT {quote(bits)}")
            self._w(f"TRIG:A:UART:PLEN {len(data)}")
            self._w(f"TRIG:A:UART:POFF {int(offset)}")
        self._scope.trig.set_mode(trigger_mode)
        return UartTriggerSettings(
            type=self._q("TRIG:A:TYPE?").upper(),
            mode=_answer(self._q("TRIG:A:UART:MODE?"), UartTriggerMode, "uart trigger mode"),
            pattern=self._q("TRIG:A:UART:PATT?").strip('"'),
        )


class _BusSpiI2c(_BusBase):
    """SPI and I2C decode, option K1."""

    def set_spi(
        self,
        clock: str,
        data: str,
        *,
        threshold: float = 0.5,
        word_size: int = 8,
        lsb_first: bool = False,
        clock_edge: Edge | str = Edge.RISE,
        idle_time: float | None = None,
    ) -> str:
        """Decode SPI without chip select on ``clock`` and ``data``; returns the bus type.

        The burst idle time in s separates frames. ``threshold`` (V) is
        written to every analog source channel.
        """
        got = self._start("SSPI", [clock, data], _number(threshold, "threshold"))
        self._bw(f"SSPI:CLOC:SOUR {_source(clock, 'clock')}")
        self._bw(f"SSPI:CLOC:POL {_token(clock_edge, Edge, 'clock edge')}")
        self._bw(f"SSPI:DATA:SOUR {_source(data, 'data')}")
        self._bw("SSPI:DATA:POL POS")
        self._bw(f"SSPI:BORD {'LSBF' if lsb_first else 'MSBF'}")
        self._bw(f"SSPI:SSIZ {int(word_size)}")
        if idle_time is not None:
            self._bw(f"SSPI:BIT {_num(_number(idle_time, 'idle_time'))}")
        return got

    def set_i2c(self, scl: str, sda: str, *, threshold: float = 0.5) -> str:
        """Decode I2C on ``scl`` and ``sda``; returns the bus type read back.

        ``threshold`` (V) is written to every analog source channel.
        """
        got = self._start("I2C", [scl, sda], _number(threshold, "threshold"))
        self._bw(f"I2C:CLOC:SOUR {_source(scl, 'scl')}")
        self._bw(f"I2C:DATA:SOUR {_source(sda, 'sda')}")
        return got


class _BusCanLin(_BusBase):
    """CAN and LIN decode, option K3."""

    def set_can(
        self,
        data: str,
        bitrate: float,
        *,
        threshold: float = 0.5,
        signal: CanSignal | str = CanSignal.CAN_H,
        sample_point: float = 50,
    ) -> str:
        """Decode CAN on ``data`` at ``bitrate`` bit/s; returns the bus type read back.

        ``sample_point`` is in percent of the bit. ``threshold`` (V) is
        written to the source channel.
        """
        got = self._start("CAN", [data], _number(threshold, "threshold"))
        self._bw(f"CAN:DATA:SOUR {_source(data, 'data')}")
        self._bw(f"CAN:TYPE {_token(signal, CanSignal, 'can signal')}")
        self._bw(f"CAN:SAMP {int(_number(sample_point, 'sample_point'))}")
        self._bw(f"CAN:BITR {int(_number(bitrate, 'bitrate'))}")
        return got

    def set_lin(
        self,
        data: str,
        bitrate: float,
        *,
        threshold: float = 0.5,
        standard: LinStandard | str = LinStandard.V2X,
        idle: IdleLevel | str = IdleLevel.HIGH,
    ) -> str:
        """Decode LIN on ``data`` at ``bitrate`` bit/s; returns the bus type read back.

        ``threshold`` (V) is written to the source channel.
        """
        got = self._start("LIN", [data], _number(threshold, "threshold"))
        self._bw(f"LIN:DATA:SOUR {_source(data, 'data')}")
        self._bw(f"LIN:POL {_token(idle, IdleLevel, 'idle level')}")
        self._bw(f"LIN:STAN {_token(standard, LinStandard, 'lin standard')}")
        self._bw(f"LIN:BITR {int(_number(bitrate, 'bitrate'))}")
        return got


class _BusParallel(_BusBase):
    """Parallel bus decode on logic channels, option B1 (UM p.541-545)."""

    def set_parallel(
        self,
        data: Sequence[str],
        *,
        clock: str | None = None,
        clock_slope: Slope | str = Slope.RISE,
        chip_select: str | None = None,
        chip_select_active_low: bool = False,
    ) -> ParallelBusSettings:
        """Decode a parallel bus with one logic channel per bit in ``data``, LSB first.

        Without ``clock``: an unclocked bus, a word at every change, up to 4
        bits. Pattern generator outputs change within about 1 ns of each
        other, so an unclocked bus can decode a one-sample intermediate word
        at a multi-bit change (MEAS: 7 -> 0 -> 8, 0.8 ns). With ``clock``: a
        word per clock edge, up to 15 lines with the clock. Chip-select
        enable is undefined on firmware 03.000 (-113, MEAS), so the chip
        select stays enabled on its source. Returns the read-back.
        """
        kind = "CPAR" if clock else "PAR"
        self._bw(f"TYPE {kind}")
        got_type = self._bq("TYPE").upper()
        if got_type[:4] != kind[:4] and not (kind == "PAR" and got_type.startswith("PAR")):
            raise InstrumentError(f"bus type reads {got_type!r} after {kind!r}")
        self._bw(f"{kind}:WIDT {len(data)}")
        for bit, src in enumerate(data):
            self._bw(f"{kind}:DATA{bit}:SOUR {_source(src)}")
        if clock:
            self._bw(f"CPAR:CLOC:SOUR {_source(clock, 'clock')}")
            self._bw(f"CPAR:CLOC:SLOP {_token(clock_slope, Slope, 'clock slope')}")
            if chip_select:
                self._bw(f"CPAR:CS:SOUR {_source(chip_select, 'chip_select')}")
                self._bw(f"CPAR:CS:POL {'NEG' if chip_select_active_low else 'POS'}")
        self._bw("STAT ON")
        width = int(float(self._bq(f"{kind}:WIDT")))
        sources = [self._bq(f"{kind}:DATA{b}:SOUR").lower() for b in range(len(data))]
        got = ParallelBusSettings(type=self._bq("TYPE").upper(), width=width, data=sources)
        if clock:
            got = ParallelBusSettings(
                type=got.type,
                width=width,
                data=sources,
                clock=self._bq("CPAR:CLOC:SOUR").lower(),
                clock_slope=_answer(self._bq("CPAR:CLOC:SLOP"), Slope, "clock slope"),
            )
        wanted = [d.strip().lower() for d in data]
        if width != len(data) or sources != wanted or (clock and got.clock != clock.lower()):
            raise InstrumentError(f"parallel bus reads {got}, requested data {data}, clock {clock}")
        return got


class _Bus(_BusUart, _BusSpiI2c, _BusCanLin, _BusParallel):
    """Bus 1 decode with every protocol. ``connect()`` keeps only the licensed ones."""


# option -> protocol class, for the bus node connect() builds.
_BUS_PROTOCOLS: tuple[tuple[str, type[_BusBase]], ...] = (
    ("K2", _BusUart),
    ("K1", _BusSpiI2c),
    ("K3", _BusCanLin),
    ("B1", _BusParallel),
)
_BUS_CLASSES: dict[tuple[type, ...], type[_BusBase]] = {}


def _bus_class(options: Sequence[str]) -> type[_BusBase] | None:
    """The bus node class with only the protocols ``options`` license; None for none."""
    mixins = tuple(cls for option, cls in _BUS_PROTOCOLS if option in options)
    if not mixins:
        return None
    if len(mixins) == len(_BUS_PROTOCOLS):
        return _Bus
    if mixins not in _BUS_CLASSES:
        _BUS_CLASSES[mixins] = type("_Bus", mixins, {"__doc__": _Bus.__doc__})
    return _BUS_CLASSES[mixins]


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class RohdeSchwarz_RTB2000(Instrument):
    """Rohde & Schwarz RTB2000 oscilloscope: RTB2002 and RTB2004.

    ``__init__`` does no I/O. It builds the tree of the largest model with
    every option, so autocomplete and the API map see the whole family.
    ``connect()`` reads ``*IDN?`` and ``*OPT?`` and then keeps only what this
    unit has: ``ch3`` and ``ch4`` on the RTB2004, and each option node when
    the option list shows its option.
    """

    # Declared for the API-map generator and for readers.
    ch1: _Channel
    ch2: _Channel
    ch3: _Channel  # RTB2004 only
    ch4: _Channel  # RTB2004 only
    horz: _Horizontal
    acq: _Acquisition
    trig: _Trigger
    meas: _Measure
    dvm: _Dvm
    counter: _Counter
    cursors: _Cursors
    disp: _Display
    math: _Math
    ref1: _Reference
    ref2: _Reference
    ref3: _Reference
    ref4: _Reference
    search: _Search
    fft: _Fft
    mask: _Mask
    file: _File
    data: _Data
    aux_out: _AuxOut
    afg: _Afg  # option B6
    pattern: _PatternGenerator  # option B6
    logic: _Logic  # option B1
    history: _History  # option K15
    bus: _BusBase  # options K1, K2, K3, B1; only the licensed protocols

    # Enums live on the class so scope.Coupling.<TAB> resolves from the live object.
    Coupling = Coupling
    Hysteresis = Hysteresis
    TimeReference = TimeReference
    AcquireMode = AcquireMode
    Interpolation = Interpolation
    AcquisitionState = AcquisitionState
    TriggerMode = TriggerMode
    Slope = Slope
    Edge = Edge
    TriggerCoupling = TriggerCoupling
    Polarity = Polarity
    WidthCondition = WidthCondition
    TimeoutCondition = TimeoutCondition
    PatternLevel = PatternLevel
    PatternLogic = PatternLogic
    VideoStandard = VideoStandard
    VideoModulation = VideoModulation
    VideoField = VideoField
    MeasType = MeasType
    VoltmeterType = VoltmeterType
    GridStyle = GridStyle
    DisplayStyle = DisplayStyle
    MathOperation = MathOperation
    FftWindow = FftWindow
    FftScale = FftScale
    BusFormat = BusFormat
    Parity = Parity
    IdleLevel = IdleLevel
    UartLine = UartLine
    UartTriggerMode = UartTriggerMode
    CanSignal = CanSignal
    LinStandard = LinStandard
    LogicThreshold = LogicThreshold
    LogicActivity = LogicActivity
    PatternGenFunction = PatternGenFunction
    CountDirection = CountDirection
    AfgFunction = AfgFunction
    Load = Load
    AuxOutSource = AuxOutSource

    def __init__(self, resource_name: str, *, timeout_ms: int = _DEFAULT_TIMEOUT_MS):
        """
        Parameters
        ----------
        resource_name : str
            pyvisa resource for the scope, read off its front panel. There is
            no default address and no scan (ADR-051, ADR-052).
        timeout_ms : int
            I/O timeout in milliseconds.
        """
        super().__init__(resource_name, name="RTB2000")
        self.link = _RtbLink(resource_name, timeout_ms=timeout_ms)
        self._timeout_ms = timeout_ms
        self.model: str | None = None
        self.serial: str | None = None
        self.firmware: str | None = None
        self.options: list[str] = []
        self.num_channels: int = 4
        self.errors_on_connect: list[str] = []

        self.ch1 = _Channel(self, 1)
        self.ch2 = _Channel(self, 2)
        self.ch3 = _Channel(self, 3)
        self.ch4 = _Channel(self, 4)
        self.horz = _Horizontal(self, "HORIZONTAL")
        self.acq = _Acquisition(self, "ACQUISITION")
        self.trig = _Trigger(self, "TRIGGER")
        self.meas = _Measure(self, "MEASUREMENT")
        self.dvm = _Dvm(self, "DVM")
        self.counter = _Counter(self, "COUNTER")
        self.cursors = _Cursors(self, "CURSOR")
        self.disp = _Display(self, "DISPLAY")
        self.math = _Math(self, "MATH")
        self.ref1 = _Reference(self, 1)
        self.ref2 = _Reference(self, 2)
        self.ref3 = _Reference(self, 3)
        self.ref4 = _Reference(self, 4)
        self.search = _Search(self, "SEARCH")
        self.fft = _Fft(self, "FFT")
        self.mask = _Mask(self, "MASK")
        self.file = _File(self, "FILE")
        self.data = _Data(self, "DATA")
        self.aux_out = _AuxOut(self, "AUXOUT")
        self.afg = _Afg(self, "AFG")
        self.pattern = _PatternGenerator(self, "PATTERN")
        self.logic = _Logic(self, "LOGIC")
        self.history = _History(self, "HISTORY")
        self.bus = _Bus(self, "BUS1")

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the pinned resource, confirm it is an RTB2000, and build the node tree.

        A VISA timeout on the first identity query raises
        InstrumentUnresponsiveError and is never retried. An answer from any
        other instrument raises InstrumentError: a wrong address is fixed by
        reading the right one off the front panel, never by scanning (ADR-051).
        """
        to = self._timeout_ms if timeout_ms is None else timeout_ms
        try:
            self.link.connect(timeout_ms=to)
            idn = self.link.ask("*IDN?")
        except visa.VisaIOError as exc:
            if exc.error_code == visa.constants.StatusCode.error_timeout:
                self.link.disconnect()
                raise InstrumentUnresponsiveError(
                    f"{self.resource_name} did not answer *IDN? (VISA timeout). Check the "
                    "address on the front panel and the cable; do not retry in a loop."
                ) from exc
            raise
        if not _IDN_RE.search(idn) or "ROHDE" not in idn.upper():
            self.link.disconnect()
            raise InstrumentError(
                f"{self.resource_name} did not answer with an R&S RTB2000 IDN: {idn!r}. "
                "Check the resource; this driver does not scan for the instrument."
            )
        self.idn = idn
        self._on_connect(idn)
        return idn

    def _on_connect(self, idn: str) -> None:
        """Parse the IDN and the options, and keep the nodes this unit has."""
        match = _IDN_RE.search(idn)
        if not match:
            raise InstrumentError(f"not a recognized RTB2000 IDN: {idn!r}")
        model = match.group(1).upper()
        self.model = model
        fields = [f.strip() for f in idn.split(",")]
        self.serial = fields[2] if len(fields) >= 3 else None
        self.firmware = fields[3] if len(fields) >= 4 else None
        # A family member missing from the table gets the smallest feature set,
        # so nothing is attached that might not exist.
        info = _MODEL_INFO.get(model, {"channels": 2, "bandwidth": None})
        self.num_channels = info["channels"]
        try:
            self.options = self.get_options()
        except (visa.VisaIOError, InstrumentQueryError):
            self.options = []

        for n in (3, 4):
            name = f"ch{n}"
            if n <= self.num_channels:
                if not hasattr(self, name):
                    setattr(self, name, _Channel(self, n))
            elif hasattr(self, name):
                delattr(self, name)

        factories = {
            "afg": lambda: _Afg(self, "AFG"),
            "pattern": lambda: _PatternGenerator(self, "PATTERN"),
            "logic": lambda: _Logic(self, "LOGIC"),
            "history": lambda: _History(self, "HISTORY"),
        }
        for name, make in factories.items():
            present = any(option in self.options for option in _OPTION_NODES[name])
            if present and not hasattr(self, name):
                setattr(self, name, make())
            elif not present and hasattr(self, name):
                delattr(self, name)

        bus_class = _bus_class(self.options)
        if bus_class is None:
            if hasattr(self, "bus"):
                del self.bus
        elif type(getattr(self, "bus", None)) is not bus_class:
            self.bus = bus_class(self, "BUS1")

        self.errors_on_connect = self.get_errors()

    def safe_state(self) -> None:
        """Turn the generator and pattern outputs off before the link closes; never raises.

        Best effort. Acquisition, channel and [Aux Out] settings stay.
        """
        if hasattr(self, "afg"):
            with contextlib.suppress(Exception):
                self.link.write("WGEN:OUTP OFF")
                self.link.opc()
        if hasattr(self, "pattern"):
            with contextlib.suppress(Exception):
                self.link.write("PGEN:PATT:STAT OFF")
                self.link.opc()
        with contextlib.suppress(Exception):
            self.link.drain_errors()

    def _channels(self) -> list[_Channel]:
        return [getattr(self, f"ch{n}") for n in range(1, 5) if hasattr(self, f"ch{n}")]

    def _channel(self, name: str) -> _Channel:
        n = _channel_number(name)
        if not hasattr(self, f"ch{n}"):
            raise ValueError(f"{name!r} is not a channel on a {self.model or 'this model'}")
        return getattr(self, f"ch{n}")

    def channel(self, n: int) -> _Channel:
        """Return the analog channel node for ``n``."""
        return self._channel(f"ch{n}")

    # ---------- system ----------

    def get_options(self) -> list[str]:
        """The installed options as a list of tokens, for example ["K1", "B6"]."""
        raw = self.link.ask("*OPT?").strip('"')
        return [o.strip() for o in raw.split(",") if o.strip() and o.strip() != "0"]

    def get_errors(self) -> list[str]:
        """Drain and return the error queue, oldest first; an empty list means clean."""
        return self.link.drain_errors()

    def cmd_reset(self) -> None:
        """Reset every setting to the factory default and wait for it to finish.

        [Aux Out] keeps its mode through the reset, so the 10 MHz reference to
        the signal generator survives (quirk RTB-11). Raises when the error
        queue holds anything afterwards.
        """
        self.link.write("*RST")
        self.link.opc(_RESET_TIMEOUT_S)
        errors = self.get_errors()
        if errors:
            raise InstrumentQueryError(f"the scope reported after the reset: {'; '.join(errors)}")

    # ---------- front-panel keys ----------

    def cmd_run(self) -> None:
        """Start acquiring continuously, the front-panel Run key."""
        self.link.write_checked("RUN")

    def cmd_stop(self) -> None:
        """Stop acquiring, the front-panel Stop key."""
        self.link.write_checked("STOP")

    def cmd_single(self) -> None:
        """Start one acquisition and return at once; ``cmd_wait_acquisition()`` blocks."""
        self.link.write("SING")

    def cmd_autoset(self) -> None:
        """Run the front-panel Autoset and wait for it to finish."""
        self.link.write("AUT")
        self.link.opc(_AUTOSET_TIMEOUT_S)
        errors = self.get_errors()
        if errors:
            raise InstrumentQueryError(f"the scope reported after autoset: {'; '.join(errors)}")

    def get_acquisition_state(self) -> AcquisitionState:
        """Whether the scope acquires, is stopping, has completed, or broke off."""
        raw = self.link.ask("ACQ:STAT?")
        return _answer(raw, AcquisitionState, "acquisition state")

    def cmd_wait_acquisition(self, timeout: float = 5.0) -> None:
        """Wait up to ``timeout`` s for the single acquisition that ``cmd_single()`` started.

        On a timeout this clears the link and the error queue, restarts the
        acquisition, and raises InstrumentTimeoutError naming the trigger
        source and mode. Once the scope did not trigger with correct
        settings until an autoset (quirk RTB-06).
        """
        try:
            self.link.opc(timeout)
        except (visa.VisaIOError, InstrumentQueryError) as exc:
            with contextlib.suppress(Exception):
                self.link.visa.clear()
            with contextlib.suppress(Exception):
                self.get_errors()
            self.link.write("RUN")
            source = self.link.ask("TRIG:A:SOUR?")
            mode = self.link.ask("TRIG:A:MODE?")
            raise InstrumentTimeoutError(
                f"no acquisition in {timeout} s (trigger source {source}, mode {mode}). Check "
                "the trigger; if every setting is correct, cmd_autoset() once restored "
                "triggering (quirk RTB-06)."
            ) from exc

    # ---------- task helpers ----------

    def _read_channel(self, n: int) -> tuple[np.ndarray, list[float]]:
        """Read channel ``n`` at the full record for the screen range (quirk RTB-04)."""
        self.link.write(f"CHAN{n}:DATA:POIN DMAX")
        head = [float(x) for x in self.link.ask(f"CHAN{n}:DATA:HEAD?").split(",")]
        data = self.link.query_floats(f"CHAN{n}:DATA?")
        if int(head[2]) != len(data):
            raise InstrumentQueryError(
                f"CH{n} header says {int(head[2])} points, received {len(data)}"
            )
        return data, head

    def _range(self, n: int) -> SignalRange:
        """Min and max of channel ``n`` from one free-running acquisition."""
        self.link.write("SING")
        self.link.opc(_RANGE_ACQUISITION_TIMEOUT_S)
        self.link.write(f"CHAN{n}:DATA:POIN DEF")
        x = self.link.query_floats(f"CHAN{n}:DATA?")
        return SignalRange(float(np.percentile(x, 0.1)), float(np.percentile(x, 99.9)))

    def set_auto_range(
        self, channels: Sequence[str], *, trigger_source: str, fill_div: float = 6.0
    ) -> dict[str, SignalRange]:
        """Fit each channel's scale and offset to its signal, then trigger on the source.

        Each signal fills ``fill_div`` of the 10 divisions, centred; the edge
        trigger goes to NORMAL mode at the trigger source's mid level. Only
        scale, offset and the channel state change: coupling, bandwidth,
        attenuation and invert stay as set. Returns each channel's range in V.
        """
        fill = _number(fill_div, "fill_div")
        self.link.write("FORM:DATA REAL,32")
        self.link.write("FORM:BORD LSBF")
        self.trig.set_mode(TriggerMode.AUTO)
        for name in channels:
            ch = self._channel(name)
            ch.set_display_state(True)
            ch.set_scale(5.0 * ch.get_attenuation())
            ch.set_offset(0.0)
        ranges: dict[str, SignalRange] = {}
        r = SignalRange(0.0, 0.0)
        for name in channels:
            ch = self._channel(name)
            for _ in range(3):  # coarse, then fine
                r = self._range(ch.n)
                ch.set_scale(max((r.high - r.low) / fill, 1e-3))
                ch.set_offset((r.high + r.low) / 2)
            ranges[name] = r
        self.link.write("RUN")
        src = trigger_source.strip().lower()
        r = ranges.get(src)
        self.trig.set_edge(src, level=(r.high + r.low) / 2 if r else None)
        return ranges

    def get_settings(self) -> ScopeSettings:
        """Every vertical, horizontal, acquisition and trigger setting."""
        return ScopeSettings(
            channels={f"ch{ch.n}": ch.get_settings() for ch in self._channels()},
            horizontal=self.horz.get_timebase(),
            acquisition=self.acq.get_settings(),
            trigger=self.trig.get_settings(),
        )

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Read back every setting :meth:`restore` puts back, for an exact restore.

        Covers the channels, the horizontal and acquisition settings, the
        trigger type, mode, source, slope, coupling, holdoff and the level of
        every source, [Aux Out], the run state, and the generator outputs.
        The settings special to a width, runt, rise-time, timeout, pattern or
        video trigger are not captured.
        """
        snap: dict = {
            "acquisition_state": self.get_acquisition_state(),
            "channels": {f"ch{ch.n}": ch.get_settings() for ch in self._channels()},
            "horizontal": self.horz.get_timebase(),
            "acquisition": self.acq.get_settings(),
            "trigger": self.trig.get_settings(),
            "trigger_levels": {
                i: self.link.query_float(f"TRIG:A:LEV{i}?")
                for i in [*range(1, self.num_channels + 1), 5]
            },
            "aux_out": self.aux_out.get_source(),
        }
        if hasattr(self, "afg"):
            snap["afg"] = {"waveform": self.afg.get_waveform(), "load": self.afg.get_load()}
        if hasattr(self, "pattern"):
            snap["pattern_output"] = self.pattern.get_output_state()
        return snap

    def restore(self, snapshot: dict) -> None:
        """Put back the settings captured by :meth:`snapshot`, exactly as found.

        The generator outputs go off first and back to their found state
        last, so nothing is driven while it is set up. A record length
        applies only at the next acquisition (quirk RTB-08), so the scope runs
        while it is restored, and goes back to the found run state at the end.
        Raises when the error queue holds anything afterwards.
        """
        # UNVERIFIED: this restore sequence was written in the port; the original
        # driver had no restore, so no run of it has been observed.
        if "afg" in snapshot and hasattr(self, "afg"):
            self.afg.set_output_state(False)
        if "pattern_output" in snapshot and hasattr(self, "pattern"):
            self.pattern.set_output_state(False)
        for name, s in snapshot["channels"].items():
            ch = self._channel(name)
            ch.set_attenuation(s.attenuation)
            ch.set_coupling(s.coupling)
            ch.set_bandwidth_limit_state(s.bandwidth_limit)
            ch.set_invert(s.invert)
            ch.set_scale(s.scale)
            ch.set_position(s.position)
            ch.set_offset(s.offset)
            ch.set_display_state(s.display_state)
        h = snapshot["horizontal"]
        self.horz.set_roll_auto_state(h.roll_auto)
        self.horz.set_reference(h.reference)
        self.horz.set_scale(h.scale)
        self.horz.set_position(h.position)
        a = snapshot["acquisition"]
        self.acq.set_mode(a.mode)
        self.acq.set_interpolation(a.interpolation)
        self.acq.set_average_count(a.average_count)
        if self.acq.get_record_length() != a.record_length:
            self.acq.set_record_length(a.record_length)
        t = snapshot["trigger"]
        # UNVERIFIED: that the scope takes its own trigger-type answer as input.
        self.link.write_checked(f"TRIG:A:TYPE {t.type}")
        if t.source in _LEVEL_INDEX or t.source == "line" or t.source.startswith("d"):
            self.trig.set_source(t.source)
        self.trig.set_slope(t.slope)
        if t.coupling is not None:
            self.trig.set_coupling(t.coupling)
        for i, level in snapshot["trigger_levels"].items():
            self.link.write_checked(f"TRIG:A:LEV{i} {_num(level)}")
        self.trig.set_holdoff(t.holdoff)
        self.trig.set_mode(t.mode)
        self.aux_out.set_source(snapshot["aux_out"])
        if "afg" in snapshot and hasattr(self, "afg"):
            afg = snapshot["afg"]
            wave = afg["waveform"]
            self.afg.set_load(afg["load"])
            self.afg._set_function(AfgFunction[wave.function])
            if wave.function != AfgFunction.DC.name:
                self.afg.set_frequency(wave.frequency)
                self.afg.set_amplitude(wave.amplitude)
            self.afg.set_offset(wave.offset)
            if wave.duty is not None:
                self.afg._set_duty(wave.duty)
            self.afg.set_output_state(bool(wave.output_state))
        if "pattern_output" in snapshot and hasattr(self, "pattern"):
            self.pattern.set_output_state(snapshot["pattern_output"])
        if snapshot["acquisition_state"] is not AcquisitionState.RUN:
            self.cmd_stop()
        errors = self.get_errors()
        if errors:
            raise InstrumentError(f"restore() finished, but the scope queued: {errors}")

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator[RohdeSchwarz_RTB2000]:
        """Snapshot the settings this driver restores on enter, and restore them on exit."""
        state = self.snapshot()
        try:
            yield self
        finally:
            self.restore(state)


def _main(argv: list[str] | None = None) -> int:
    """Read-only self-test: identity, options, channels, timebase, trigger, error queue.

    Changes no setting. The disconnect at the end turns the generator and
    pattern outputs off.
    """
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE")
        return 2
    scope = RohdeSchwarz_RTB2000(argv[1])
    print(f"connecting to {scope.resource_name} ...")
    idn = scope.connect()
    try:
        print(f"  IDN: {idn}")
        print(f"  model={scope.model}  firmware={scope.firmware}")
        print(f"  options={scope.options or '(none)'}")
        print(f"  errors at connect: {scope.errors_on_connect or 'none'}")
        print(f"  acquisition: {scope.get_acquisition_state().name}")
        for ch in scope._channels():
            s = ch.get_settings()
            print(
                f"  {ch.name}: {'on' if s.display_state else 'off'} {s.scale:g} V/div "
                f"offset {s.offset:g} V {s.coupling.name}"
            )
        h = scope.horz.get_timebase()
        print(f"  timebase: {h.scale:g} s/div, position {h.position:g} s, {h.reference.name}")
        print(f"  trigger: {scope.trig.get_settings()}")
        print(f"  aux out: {scope.aux_out.get_source().name}")
        if hasattr(scope, "afg"):
            print(f"  generator: {scope.afg.get_waveform()}")
        errors = scope.get_errors()
        print(f"  error queue: {errors or 'clean'}")
    finally:
        scope.disconnect()
        print("disconnected (generator and pattern outputs off).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
