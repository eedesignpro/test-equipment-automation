"""Driver for the Keysight B2900 series precision Source/Measure Units (SMU).

Public import path::

    from drivers.keysight_b2900 import Keysight_B2900

Shape follows ``docs/driver-api-design.md`` (section 8, B2902B). The force
function is chosen by the method name - ``set_source_voltage`` forces volts with
a required current compliance, ``set_source_current`` forces amperes with a
required voltage compliance - and each returns a :class:`SourceState`. Sweeps
and lists follow the same split. Measurements are ``meas_*``.

One family-named class covers the whole B29xx series. ``connect()`` reads
``*IDN?`` and attaches ``ch2`` only on the dual-channel models, so a B2901 has no
``ch2`` attribute at all rather than one that raises. ``ch1`` exists before
connect so autocomplete and the API-map generator can see the channel shape.

Units carry no scalar prefix (design standard 4.7): volts, amperes, ohms, watts,
seconds. The B2900 is already prefix-free, so nothing is converted here.

Compliance takes two forms. A scalar is a symmetric limit of plus and minus that
value. A ``(min, max)`` tuple is an asymmetric limit and ``min`` is negative,
for example ``(-0.5, 2.0)``.

Task recipes::

    smu.connect()

    # 1. Force 1 V with a 10 mA compliance and read back what flows:
    smu.ch1.set_source_voltage(1.0, i_limit=0.010)
    smu.ch1.set_output_state(True)
    r = smu.ch1.meas_vi()                       # r.voltage, r.current

    # 2. Step that level while the output stays live (no flag needed):
    for v in (1.0, 1.5, 2.0):
        smu.ch1.set_source_level(v)
        print(smu.ch1.meas_current())

    # 3. Force 50 mA with a 6 V ceiling, and ask whether it hit the limit:
    smu.ch1.set_source_current(0.050, v_limit=6.0)
    smu.ch1.get_compliance_tripped()

    # 4. Sweep 0 V to 5 V in 21 steps and collect both arrays in one pass:
    smu.ch1.set_sweep_voltage(0.0, 5.0, 21, i_limit=0.010)
    smu.ch1.set_output_state(True)
    pts = smu.ch1.meas_sweep()                  # pts.voltage, pts.current

    # 5. Everything off:
    smu.set_output_state_all(False)

Safe operating area, continuous DC (bench-verified envelope, ADR-040): 3.03 A at
or below 6 V, 1.515 A at or below 21 V, 0.105 A at or below 210 V. An open
INTERLOCK caps the output at +/-42 V.

Bench observations behind the choices in this file (ADR-042, ADR-046). All
measured on the bench B2902B over VXI-11 on 2026-08-10 with
``projects/smu_probe.py``:

- A query costs about 4.1 ms and a write about 1.6 ms, so a read-before-write
  check COSTS more than the write it would avoid. This driver does not diff
  against present state. That is the opposite of the MSO (ADR-044); the numbers
  do not transfer between instruments.
- A same-value level write costs about 1.6 ms and a new-value write about
  4.4 ms, so the instrument already skips the work for a redundant write.
- ``OUTP ON`` followed by ``*OPC?`` takes about 77 ms and ``OUTP OFF`` about
  37 ms. Relay writes here are synchronized with ``*OPC?`` rather than a sleep.
- A real source-range change answers an immediately following query in about
  20 ms with no error. The B2900 has NO deaf window; do not copy the E3646A
  range settle (ADR-026/027) into this driver.
- Switching the source function on a LIVE output applies the stale level of the
  new function: a channel measured -125 uA as a 0 V source, and switching it to
  current force applied a parked +1.000007 mA immediately. A helper therefore
  refuses a live output when, and only when, something would actually switch:
  the force type, sweep or list mode, a range, or the output shape. Moving a
  level or a limit on a live output is ordinary operation and needs no flag, so
  a helper called with unchanged mode and ranges simply proceeds, and
  ``set_source_level`` never switches anything at all.
- With source auto-ranging on, the instrument changes range by itself to follow
  the commanded level: the source range read 0.1 at 0.05 A and 0.1 A, and 1 at
  0.2 A and 0.5 A. A loop that must not range-switch pins the range once.
- Compliance read-back has a trap. Writing ``:PROT:POS`` alone makes the
  symmetric ``:PROT?`` query return the 9.91E37 overload sentinel. This driver
  always reads the POS and NEG pair and never the symmetric query.
- NPLC is shared across VOLT, CURR and RES; writing one changes all three, so
  ``set_nplc`` takes no function argument.
- There is no channel-list output form: ``OUTP:STAT ON,(@1,2)`` returns
  -108 "Parameter not allowed". ``set_output_state_all`` writes each channel.
- ``DISP:VIEW ROLL`` returns -222 "Data out of range" on the B2902B, so ROLL is
  absent from :class:`_Display.View` rather than present and broken.

Channel addressing follows the instrument's two conventions (ADR-040):
``:SOURce[c]:``, ``:SENSe[c]:`` and ``:OUTPut[c]`` take a numeric channel
suffix, while ``:MEASure? / :READ? / :FETCh? / :INITiate`` take the ``(@c)``
channel-list parameter. The channel node builds both so callers never see it.

Command surface follows the Keysight B2900B/BL SCPI Command Reference.

Common tasks:
  1. Force 200 mV with a 10 mA compliance and read the current:
         smu.ch1.set_source_voltage(0.2, i_limit=10e-3)
         smu.ch1.set_output_state(True)
         v, i = smu.ch1.meas_vi()
  2. Step the level on a live output, without touching mode or range:
         for volts in (0.2, 0.4, 0.6):
             smu.ch1.set_source_level(volts)
  3. Sweep an IV curve and collect both columns:
         smu.ch1.set_sweep_voltage(0, 1.0, 21, i_limit=10e-3)
         reading = smu.ch1.meas_sweep()
  4. Measure through 4-wire sense, where leads are landed:
         smu.ch1.set_4w_sense_state(True)
  5. Find out whether the channel is sitting at its limit:
         smu.ch1.get_compliance_tripped()
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import pyvisa as visa

from common.errors import InstrumentError, InstrumentUnresponsiveError
from common.instrument import Instrument
from common.node import Node
from common.scpi import ScpiEnum, member_from_response

# Generous LAN I/O timeout. The link is fast; this only bounds a hung transfer.
_DEFAULT_TIMEOUT_MS = 10000

# A sweep of many points at a high NPLC outlives the normal timeout.
_SWEEP_TIMEOUT_MS = 120_000

# A reading at or above this magnitude is the "no data / not a number" sentinel
# (9.91E37 ASCII / NaN IEEE-754); it is mapped to float('nan'), never returned
# as a real value (ADR-038). Verified live: a channel held in voltage compliance
# returned -9.9E37 for its current.
_OVERLOAD = 9.9e37

# model prefix (first 5 chars, e.g. "B2902") -> channel count. The B2900 has no
# channel-count query, so the model is the authority (ADR-040). The API-map
# generator reads this table to print the channel gate, so keep the gating in
# data and not in an if-statement (design standard 9.1 rule 4).
_MODEL_INFO: dict[str, dict] = {
    "B2901": {"channels": 1},
    "B2902": {"channels": 2},
    "B2911": {"channels": 1},
    "B2912": {"channels": 2},
}

_IDN_RE = re.compile(r"(B29\d\d[A-Z]{0,2})", re.IGNORECASE)


def _onoff(state: bool) -> str:
    return "ON" if state else "OFF"


def _num(value: float) -> str:
    return f"{float(value):g}"


def _token(value, enum_cls: type[ScpiEnum], label: str) -> str:
    """Return the SCPI token for an enum member, member name, or token string.

    Accepts ``Spacing.LOG``, ``"LOG"``, ``"log"`` and ``"LOGARITHMIC"`` alike so
    a plain-string call site keeps working (design standard 4.6).
    """
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    for member in enum_cls:
        if text in (member.name, str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(f"{m.name}={m.value}" for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")


def _nan_if_overload(value: float) -> float:
    """Map the 9.91E37 / NaN overflow sentinel to ``float('nan')`` (ADR-038)."""
    return float("nan") if value != value or abs(value) >= _OVERLOAD else value


def _parse_reading(raw: str) -> list[float]:
    """Parse a comma-separated NR3 measurement response, overload -> NaN."""
    return [_nan_if_overload(float(tok)) for tok in raw.strip().split(",") if tok.strip()]


# ---------------------------------------------------------------- state types


@dataclass(frozen=True)
class Reading:
    """A voltage and current measured in one acquisition.

    Iterates over the measured quantities in field order, which the ``meas_``
    name spells, so ``v, i = ch.meas_vi()`` works and ``r.voltage`` still does.
    A field added later that the name does not spell is not iterated, so no
    unpacking call site breaks.
    """

    voltage: float  # volts
    current: float  # amperes

    def __iter__(self) -> Iterator[float]:
        """Yield voltage then current, the quantities ``meas_vi`` names."""
        yield self.voltage
        yield self.current


@dataclass(frozen=True)
class SweepReading:
    """The measured points of one sweep or list, as parallel columns.

    Columns rather than a list of points, because the operator plots them:
    ``plot(*pts)`` or ``plot(pts.voltage, pts.current)``.
    """

    voltage: list[float] = field(default_factory=list)  # volts
    current: list[float] = field(default_factory=list)  # amperes

    def __iter__(self) -> Iterator[list[float]]:
        """Yield the voltage column then the current column."""
        yield self.voltage
        yield self.current


@dataclass(frozen=True)
class SourceState:
    """The force function, its level, the compliance in force, and the ranges.

    ``level`` is in the forced quantity's unit: volts when ``function`` is VOLT,
    amperes when it is CURR. Exactly one of ``i_limit`` / ``v_limit`` is set:
    the one that limits the forced quantity. A limit is a float for a symmetric
    limit or a ``(min, max)`` tuple when the two polarities differ.

    ``v_range`` and ``i_range`` are the ranges in force for those two
    quantities, wherever they live: the forced quantity's range comes from the
    source, the other from the measurement side.
    """

    function: str  # "VOLT" or "CURR"
    level: float
    i_limit: float | tuple[float, float] | None = None  # amperes
    v_limit: float | tuple[float, float] | None = None  # volts
    v_range: float = float("nan")  # volts
    i_range: float = float("nan")  # amperes
    v_range_auto: bool = False
    i_range_auto: bool = False
    output_state: bool = False


@dataclass(frozen=True)
class SweepState:
    """A staircase-sweep configuration read back from the instrument."""

    function: str  # "VOLT" or "CURR"
    start: float  # volts or amperes
    stop: float  # volts or amperes
    points: int
    spacing: str
    direction: str
    stair: str
    ranging: str


@dataclass(frozen=True)
class ShapeState:
    """The source output shape and, for PULSe, its width and delay."""

    shape: str  # "DC" or "PULS"
    width: float  # seconds
    delay: float  # seconds


@dataclass(frozen=True)
class TriggerState:
    """The acquisition-layer trigger configuration."""

    source: str
    count: int
    delay: float  # seconds
    interval: float  # seconds, the TIMer period


# ---------------------------------------------------------------- sub-nodes


# ---------------------------------------------------------------------------
# Enums
#
# Module level, and attached to Keysight_B2900 below, so an operator reaches
# them as smu.<Enum>.<TAB> from the instrument handle whatever node the method
# sits on. They are deliberately NOT repeated on the nodes: one path to a
# thing (docs/hardware-not-software.md section 2).
# ---------------------------------------------------------------------------


class TraceFeed(ScpiEnum):
    """What data fills the buffer."""

    SENSE = "SENS"
    MATH = "MATH"
    LIMIT = "LIM"


class Control(ScpiEnum):
    """Whether the buffer accepts new data."""

    NEXT = "NEXT"
    NEVER = "NEV"


class Statistic(ScpiEnum):
    """The statistic computed over the buffered data."""

    MEAN = "MEAN"
    SDEV = "SDEV"
    PKPK = "PKPK"
    MINIMUM = "MIN"
    MAXIMUM = "MAX"


class Timestamp(ScpiEnum):
    """How buffered timestamps are reported."""

    DELTA = "DELT"
    ABSOLUTE = "ABS"


class CalcFeed(ScpiEnum):
    """Which quantity feeds the limit test and math engine."""

    MATH = "MATH"
    VOLTAGE = "VOLT"
    CURRENT = "CURR"
    RESISTANCE = "RES"


class LimitFunction(ScpiEnum):
    """What a limit test checks."""

    LIMIT = "LIM"
    COMPLIANCE = "COMP"


class Data(ScpiEnum):
    """Wire format of measurement data."""

    ASCII = "ASC"
    REAL32 = "REAL,32"
    REAL64 = "REAL,64"


class ByteOrder(ScpiEnum):
    """Byte order of binary measurement data."""

    NORMAL = "NORM"
    SWAPPED = "SWAP"


class NumberBase(ScpiEnum):
    """Number base used for status-register and digital responses."""

    ASCII = "ASC"
    BINARY = "BIN"
    OCTAL = "OCT"
    HEXADECIMAL = "HEX"


class SenseElement(ScpiEnum):
    """A field that a measurement response can carry."""

    VOLTAGE = "VOLT"
    CURRENT = "CURR"
    RESISTANCE = "RES"
    TIME = "TIME"
    STATUS = "STAT"
    SOURCE = "SOUR"


class CalcElement(ScpiEnum):
    """A field that a calculation response can carry."""

    CALCULATE = "CALC"
    TIME = "TIME"
    STATUS = "STAT"


class View(ScpiEnum):
    """Front-panel layout.

    ROLL is absent on purpose: the bench B2902B answers -222 "Data out of
    range" to it, so a member for it would be a lie (design standard 5.1).
    """

    SINGLE1 = "SING1"
    SINGLE2 = "SING2"
    DUAL = "DUAL"
    GRAPH = "GRAP"


class StatusRegister(ScpiEnum):
    """Which summary register to address."""

    MEASUREMENT = "MEASurement"
    OPERATION = "OPERation"
    QUESTIONABLE = "QUEStionable"


class Spacing(ScpiEnum):
    """Sweep step spacing."""

    LINEAR = "LIN"
    LOGARITHMIC = "LOG"


class Direction(ScpiEnum):
    """Sweep direction."""

    UP = "UP"
    DOWN = "DOWN"


class Stair(ScpiEnum):
    """Whether a sweep returns to its start."""

    SINGLE = "SING"
    DOUBLE = "DOUB"


class Ranging(ScpiEnum):
    """How the source picks ranges during a sweep."""

    BEST = "BEST"
    FIXED = "FIX"
    AUTO = "AUTO"


class OffMode(ScpiEnum):
    """What the output presents once it is turned off."""

    NORMAL = "NORM"
    ZERO = "ZERO"
    HIGH_Z = "HIZ"


class LowTerminal(ScpiEnum):
    """Whether the low terminal floats or is grounded."""

    FLOAT = "FLO"
    GROUND = "GRO"


class TriggerSource(ScpiEnum):
    """What advances the trigger model."""

    AUTO_INTERNAL = "AINT"
    BUS = "BUS"
    TIMER = "TIM"
    INTERNAL1 = "INT1"
    INTERNAL2 = "INT2"
    LAN = "LAN"
    EXTERNAL1 = "EXT1"
    EXTERNAL2 = "EXT2"
    EXTERNAL3 = "EXT3"
    EXTERNAL4 = "EXT4"
    EXTERNAL5 = "EXT5"
    EXTERNAL6 = "EXT6"
    EXTERNAL7 = "EXT7"
    EXTERNAL8 = "EXT8"
    EXTERNAL9 = "EXT9"
    EXTERNAL10 = "EXT10"
    EXTERNAL11 = "EXT11"
    EXTERNAL12 = "EXT12"
    EXTERNAL13 = "EXT13"
    EXTERNAL14 = "EXT14"


class TriggerAction(ScpiEnum):
    """Which layer of the trigger model a call addresses."""

    ALL = "ALL"
    ACQUIRE = "ACQ"
    TRANSIENT = "TRAN"


class _Trace(Node):
    """Data buffer for one channel.

    Write access is gated by the feed-control mode: set it to NEXT to arm the
    capture (it reverts to NEVer by itself when the buffer fills).
    """

    def __init__(self, link, c: int):
        super().__init__(link, name=f"CH{c}:TRACe")
        self.c = c
        self._t = f"TRAC{c}:"

    def set_feed(self, source: TraceFeed) -> TraceFeed:
        """Select what data fills the buffer; a :class:`Feed`.

        Enum arg: <smu>.TraceFeed.<TAB>
        """
        self.write(f"{self._t}FEED {_token(source, TraceFeed, 'feed')}")
        return self.get_feed()

    def get_feed(self) -> TraceFeed:
        """What data currently fills the buffer."""
        return member_from_response(self.query(f"{self._t}FEED?").strip(), TraceFeed, "feed")

    def set_control(self, control: Control) -> Control:
        """Arm or stop buffer writes; a :class:`Control`.

        Enum arg: <smu>.Control.<TAB>
        """
        self.write(f"{self._t}FEED:CONT {_token(control, Control, 'control')}")
        return self.get_control()

    def get_control(self) -> Control:
        """Whether the buffer is armed for new data."""
        return member_from_response(self.query(f"{self._t}FEED:CONT?").strip(), Control, "control")

    def set_points(self, points: int) -> int:
        """Set the buffer size in points; allowed only while writes are stopped."""
        self.write(f"{self._t}POIN {int(points)}")
        return self.get_points()

    def get_points(self) -> int:
        """The configured buffer size in points."""
        return self.query_int(f"{self._t}POIN?")

    def get_points_actual(self) -> int:
        """The number of points currently stored."""
        return self.query_int(f"{self._t}POIN:ACT?")

    def get_free(self) -> tuple[int, int]:
        """Return (available, total) buffer points."""
        avail, total = _parse_reading(self.query(f"{self._t}FREE?"))
        return int(avail), int(total)

    def get_data(self, offset: int | None = None, size: int | None = None) -> list[float]:
        """Return the buffered data; the overload sentinel reads as NaN."""
        arg = ""
        if offset is not None:
            arg = f" {int(offset)}" + (f",{int(size)}" if size is not None else "")
        return _parse_reading(self.query(f"{self._t}DATA?{arg}"))

    def set_statistic(self, operation: Statistic) -> Statistic:
        """Select the statistic computed over the buffer; a :class:`Statistic`.

        Enum arg: <smu>.Statistic.<TAB>
        """
        self.write(f"{self._t}STAT:FORM {_token(operation, Statistic, 'statistic')}")
        return self.get_statistic()

    def get_statistic(self) -> Statistic:
        """Which statistic is selected."""
        return member_from_response(
            self.query(f"{self._t}STAT:FORM?").strip(), Statistic, "statistic"
        )

    def get_statistic_data(self) -> list[float]:
        """Return the selected statistic computed over the buffer."""
        return _parse_reading(self.query(f"{self._t}STAT:DATA?"))

    def set_timestamp(self, rule: Timestamp) -> Timestamp:
        """Select how buffered timestamps are reported; a :class:`Timestamp`.

        Enum arg: <smu>.Timestamp.<TAB>
        """
        self.write(f"{self._t}TST:FORM {_token(rule, Timestamp, 'timestamp')}")
        return self.get_timestamp()

    def get_timestamp(self) -> Timestamp:
        """How buffered timestamps are reported."""
        return member_from_response(
            self.query(f"{self._t}TST:FORM?").strip(), Timestamp, "timestamp"
        )

    def cmd_clear(self) -> None:
        """Empty the buffer; allowed only while writes are stopped."""
        self.write(f"{self._t}CLE")


class _Calc(Node):
    """Math and limit-test calculation for one channel.

    Covers the math-expression engine, the null offset, and the per-index limit
    tests. The composite-limit and digital-GPIO bit-pattern surface is out of
    scope (rear-panel GPIO, see context/smu-verify-gaps.md).
    """

    def __init__(self, link, c: int):
        super().__init__(link, name=f"CH{c}:CALCulate")
        self.c = c
        self._q = f"CALC{c}:"

    def set_feed(self, source: CalcFeed) -> CalcFeed:
        """Select the quantity the limit test and math engine read; a :class:`Feed`.

        Enum arg: <smu>.CalcFeed.<TAB>
        """
        self.write(f"{self._q}FEED {_token(source, CalcFeed, 'feed')}")
        return self.get_feed()

    def get_feed(self) -> CalcFeed:
        """Which quantity feeds the limit test and math engine."""
        return member_from_response(self.query(f"{self._q}FEED?").strip(), CalcFeed, "feed")

    def get_data(self, offset: int | None = None, size: int | None = None) -> list[float]:
        """Return the limit-test result data; the overload sentinel reads as NaN."""
        arg = ""
        if offset is not None:
            arg = f" {int(offset)}" + (f",{int(size)}" if size is not None else "")
        return _parse_reading(self.query(f"{self._q}DATA?{arg}"))

    # -- math expression engine --

    def set_math_name(self, name: str) -> str:
        """Select the math expression to use, by name."""
        self.write(f'{self._q}MATH:NAME "{name}"')
        return self.get_math_name()

    def get_math_name(self) -> str:
        """The name of the selected math expression."""
        return self.query(f"{self._q}MATH:NAME?").strip().strip('"')

    def set_math_expression(self, definition: str) -> str:
        """Define the selected math expression; the definition is parenthesized."""
        self.write(f"{self._q}MATH:DEF {definition}")
        return self.get_math_expression()

    def get_math_expression(self) -> str:
        """The definition of the selected math expression."""
        return self.query(f"{self._q}MATH:DEF?").strip()

    def set_math_state(self, state: bool) -> bool:
        """Enable or disable the math expression."""
        self.write(f"{self._q}MATH:STAT {_onoff(state)}")
        return self.get_math_state()

    def get_math_state(self) -> bool:
        """Whether the math expression is enabled."""
        return self.query_bool(f"{self._q}MATH:STAT?")

    def set_math_units(self, unit: str) -> str:
        """Set the label shown for the math result's unit."""
        self.write(f'{self._q}MATH:UNIT "{unit}"')
        return self.get_math_units()

    def get_math_units(self) -> str:
        """The label shown for the math result's unit."""
        return self.query(f"{self._q}MATH:UNIT?").strip().strip('"')

    def get_math_catalog(self) -> list[str]:
        """List the predefined and user math expression names."""
        raw = self.query(f"{self._q}MATH:CAT?").strip()
        return [t.strip().strip('"') for t in raw.split(",") if t.strip()]

    def get_math_data(self) -> list[float]:
        """Return the math result data; the overload sentinel reads as NaN."""
        return _parse_reading(self.query(f"{self._q}MATH:DATA?"))

    # -- null offset --

    def set_offset(self, value: float) -> float:
        """Set the null offset subtracted from the limit-test input."""
        self.write(f"{self._q}OFFS {_num(value)}")
        return self.get_offset()

    def get_offset(self) -> float:
        """The null offset subtracted from the limit-test input."""
        return self.query_float(f"{self._q}OFFS?")

    def set_offset_state(self, state: bool) -> bool:
        """Enable or disable the null offset."""
        self.write(f"{self._q}OFFS:STAT {_onoff(state)}")
        return self.get_offset_state()

    def get_offset_state(self) -> bool:
        """Whether the null offset is applied."""
        return self.query_bool(f"{self._q}OFFS:STAT?")

    def cmd_acquire_offset(self) -> None:
        """Set the null offset to the latest reading."""
        self.write(f"{self._q}OFFS:ACQ")

    # -- per-index limit test --

    def set_limit_function(self, kind: LimitFunction, m: int = 1) -> LimitFunction:
        """Set what limit test ``m`` checks; a :class:`LimitFunction`.

        Enum arg: <smu>.LimitFunction.<TAB>
        """
        tok = _token(kind, LimitFunction, "limit function")
        self.write(f"{self._q}LIM{int(m)}:FUNC {tok}")
        return self.get_limit_function(m)

    def get_limit_function(self, m: int = 1) -> LimitFunction:
        """What limit test ``m`` checks."""
        return member_from_response(
            self.query(f"{self._q}LIM{int(m)}:FUNC?").strip(), LimitFunction, "limit function"
        )

    def set_limit_lower(self, value: float, m: int = 1) -> float:
        """Set the lower bound of limit test ``m``."""
        self.write(f"{self._q}LIM{int(m)}:LOW {_num(value)}")
        return self.get_limit_lower(m)

    def get_limit_lower(self, m: int = 1) -> float:
        """The lower bound of limit test ``m``."""
        return self.query_float(f"{self._q}LIM{int(m)}:LOW?")

    def set_limit_upper(self, value: float, m: int = 1) -> float:
        """Set the upper bound of limit test ``m``."""
        self.write(f"{self._q}LIM{int(m)}:UPP {_num(value)}")
        return self.get_limit_upper(m)

    def get_limit_upper(self, m: int = 1) -> float:
        """The upper bound of limit test ``m``."""
        return self.query_float(f"{self._q}LIM{int(m)}:UPP?")

    def set_limit_state(self, state: bool, m: int = 1) -> bool:
        """Enable or disable limit test ``m``."""
        self.write(f"{self._q}LIM{int(m)}:STAT {_onoff(state)}")
        return self.get_limit_state(m)

    def get_limit_state(self, m: int = 1) -> bool:
        """Whether limit test ``m`` is enabled."""
        return self.query_bool(f"{self._q}LIM{int(m)}:STAT?")

    def get_limit_failed(self, m: int = 1) -> bool:
        """Whether limit test ``m`` has failed."""
        return self.query_bool(f"{self._q}LIM{int(m)}:FAIL?")


class _Format(Node):
    """How the instrument formats the data it returns."""

    def __init__(self, link):
        super().__init__(link, name="FORMat")

    def set_data(self, fmt: Data) -> Data:
        """Set the wire format of measurement data; a :class:`Data`.

        Enum arg: <smu>.Data.<TAB>

        Anything other than ASCII returns binary blocks that this driver's
        parsers do not decode. Change it only for a raw-link transfer.
        """
        self.write(f"FORM:DATA {_token(fmt, Data, 'data format')}")
        return self.get_data()

    def get_data(self) -> Data:
        """The wire format of measurement data."""
        return member_from_response(self.query("FORM:DATA?").strip(), Data, "data")

    def set_byte_order(self, order: ByteOrder) -> ByteOrder:
        """Set the byte order of binary data; a :class:`ByteOrder`.

        Enum arg: <smu>.ByteOrder.<TAB>
        """
        self.write(f"FORM:BORD {_token(order, ByteOrder, 'byte order')}")
        return self.get_byte_order()

    def get_byte_order(self) -> ByteOrder:
        """The byte order of binary data."""
        return member_from_response(self.query("FORM:BORD?").strip(), ByteOrder, "byte order")

    def set_elements_sense(self, *elements: SenseElement) -> list[str]:
        """Choose which fields a measurement response carries; the order is fixed.

        ``meas_vi`` reads this list to find its columns, so keep VOLTAGE and
        CURRENT enabled for it to work.
        """
        toks = self._normalize(elements, SenseElement, "sense")
        self.write(f"FORM:ELEM:SENS {','.join(toks)}")
        return self.get_elements_sense()

    def get_elements_sense(self) -> list[str]:
        """Which fields a measurement response carries, in order."""
        raw = self.query("FORM:ELEM:SENS?").strip()
        return [t.strip().upper() for t in raw.split(",") if t.strip()]

    def set_elements_calc(self, *elements: CalcElement) -> list[str]:
        """Choose which fields a calculation response carries."""
        toks = self._normalize(elements, CalcElement, "calc")
        self.write(f"FORM:ELEM:CALC {','.join(toks)}")
        return self.get_elements_calc()

    def get_elements_calc(self) -> list[str]:
        """Which fields a calculation response carries, in order."""
        raw = self.query("FORM:ELEM:CALC?").strip()
        return [t.strip().upper() for t in raw.split(",") if t.strip()]

    def set_status_register(self, fmt: NumberBase) -> NumberBase:
        """Set the number base of status-register responses; a :class:`Register`.

        Enum arg: <smu>.NumberBase.<TAB>
        """
        self.write(f"FORM:SREG {_token(fmt, NumberBase, 'register format')}")
        return self.get_status_register()

    def get_status_register(self) -> NumberBase:
        """The number base of status-register responses."""
        return member_from_response(self.query("FORM:SREG?").strip(), NumberBase, "status register")

    def set_digital(self, fmt: NumberBase) -> NumberBase:
        """Set the number base of digital-port responses; a :class:`Register`.

        Enum arg: <smu>.NumberBase.<TAB>
        """
        self.write(f"FORM:DIG {_token(fmt, NumberBase, 'register format')}")
        return self.get_digital()

    def get_digital(self) -> NumberBase:
        """The number base of digital-port responses."""
        return member_from_response(self.query("FORM:DIG?").strip(), NumberBase, "digital")

    @staticmethod
    def _normalize(elements, enum_cls, label: str) -> list[str]:
        if not elements:
            raise ValueError(f"specify at least one {label} element")
        return [_token(e, enum_cls, f"{label} element") for e in elements]


class _Display(Node):
    """Front-panel display control."""

    def __init__(self, link):
        super().__init__(link, name="DISPlay")

    def set_state(self, state: bool) -> bool:
        """Turn the front-panel display on or off."""
        self.write(f"DISP:ENAB {_onoff(state)}")
        return self.get_state()

    def get_state(self) -> bool:
        """Whether the front-panel display is on."""
        return self.query_bool("DISP:ENAB?")

    def set_digits(self, digits: int) -> int:
        """Set the displayed resolution, 4 to 7 digits."""
        self.write(f"DISP:DIG {int(digits)}")
        return self.get_digits()

    def get_digits(self) -> int:
        """The displayed resolution in digits."""
        return self.query_int("DISP:DIG?")

    def set_view(self, mode: View) -> View:
        """Set the front-panel layout; a :class:`View`.

        Enum arg: <smu>.View.<TAB>
        """
        self.write(f"DISP:VIEW {_token(mode, View, 'view')}")
        return self.get_view()

    def get_view(self) -> View:
        """The front-panel layout."""
        return member_from_response(self.query("DISP:VIEW?").strip(), View, "view")

    def set_zoom_state(self, state: bool) -> bool:
        """Turn the zoomed measurement view on or off."""
        self.write(f"DISP:ZOOM {_onoff(state)}")
        return self.get_zoom_state()

    def get_zoom_state(self) -> bool:
        """Whether the zoomed measurement view is on."""
        return self.query_bool("DISP:ZOOM?")

    def set_color_set(self, color: int) -> int:
        """Select display color set 1 or 2."""
        self.write(f"DISP:CSET {int(color)}")
        return self.get_color_set()

    def get_color_set(self) -> int:
        """The selected display color set."""
        return self.query_int("DISP:CSET?")

    def set_text(self, text: str) -> str:
        """Show a custom message, up to 32 characters."""
        self.write(f'DISP:TEXT:DATA "{text}"')
        return self.get_text()

    def get_text(self) -> str:
        """The custom message."""
        return self.query("DISP:TEXT:DATA?").strip().strip('"')

    def set_text_state(self, state: bool) -> bool:
        """Show or hide the custom message."""
        self.write(f"DISP:TEXT:STAT {_onoff(state)}")
        return self.get_text_state()

    def get_text_state(self) -> bool:
        """Whether the custom message is shown."""
        return self.query_bool("DISP:TEXT:STAT?")

    def get_values(self) -> str:
        """The values currently shown on the front panel."""
        return self.query("DISP:DATA?").strip()


class _Status(Node):
    """Status-register access for the three summary registers.

    The detailed QUEStionable sub-registers (CAL/CURR/TEMP/TEST/VOLT) are out of
    scope; see context/smu-verify-gaps.md.
    """

    def __init__(self, link):
        super().__init__(link, name="STATus")

    def get_condition(self, register: StatusRegister) -> int:
        """Read a condition register without clearing it; a :class:`Register`.

        Enum arg: <smu>.StatusRegister.<TAB>
        """
        return self.query_int(f"STAT:{self._reg(register)}:COND?")

    def get_event(self, register: StatusRegister) -> int:
        """Read and clear an event register; a :class:`Register`.

        Enum arg: <smu>.StatusRegister.<TAB>
        """
        return self.query_int(f"STAT:{self._reg(register)}:EVEN?")

    def set_enable(self, register: StatusRegister, mask: int) -> int:
        """Set a summary enable mask; a :class:`Register`.

        Enum arg: <smu>.StatusRegister.<TAB>
        """
        reg = self._reg(register)
        self.write(f"STAT:{reg}:ENAB {int(mask)}")
        return self.get_enable(register)

    def get_enable(self, register: StatusRegister) -> int:
        """Read a summary enable mask; a :class:`Register`.

        Enum arg: <smu>.StatusRegister.<TAB>
        """
        return self.query_int(f"STAT:{self._reg(register)}:ENAB?")

    def set_ptransition(self, register: StatusRegister, mask: int) -> int:
        """Set a positive-transition filter; a :class:`Register`.

        Enum arg: <smu>.StatusRegister.<TAB>
        """
        reg = self._reg(register)
        self.write(f"STAT:{reg}:PTR {int(mask)}")
        return self.get_ptransition(register)

    def get_ptransition(self, register: StatusRegister) -> int:
        """Read a positive-transition filter; a :class:`Register`.

        Enum arg: <smu>.StatusRegister.<TAB>
        """
        return self.query_int(f"STAT:{self._reg(register)}:PTR?")

    def set_ntransition(self, register: StatusRegister, mask: int) -> int:
        """Set a negative-transition filter; a :class:`Register`.

        Enum arg: <smu>.StatusRegister.<TAB>
        """
        reg = self._reg(register)
        self.write(f"STAT:{reg}:NTR {int(mask)}")
        return self.get_ntransition(register)

    def get_ntransition(self, register: StatusRegister) -> int:
        """Read a negative-transition filter; a :class:`Register`.

        Enum arg: <smu>.StatusRegister.<TAB>
        """
        return self.query_int(f"STAT:{self._reg(register)}:NTR?")

    def cmd_preset(self) -> None:
        """Return the status filters and enables to their default state."""
        self.write("STAT:PRES")

    def _reg(self, register: StatusRegister) -> str:
        return _token(register, StatusRegister, "status register")


class _SmuChannel(Node):
    """One channel of the SMU: its source, its measurement, and its output relay.

    The force function is selected by calling :meth:`set_source_voltage` or
    :meth:`set_source_current`; there is no function argument anywhere. Both
    refuse a live output by default, because switching function while energized
    applies the new function's stale level (measured on the bench; see the
    module docstring).

    Every helper writes range and mode first, then the limit, then the level,
    and none of them closes the output relay (design standard 6.2).
    """

    _ACTION_NODE = {"ALL": "", "ACQ": ":ACQuire", "TRAN": ":TRANsient"}

    trace: _Trace
    calc: _Calc

    def __init__(self, link, c: int):
        super().__init__(link, name=f"CH{c}")
        self.c = c
        self._src = f"SOUR{c}:"  # source prefix, e.g. "SOUR2:VOLT"
        self._sens = f"SENS{c}:"  # sense prefix, e.g. "SENS2:FUNC"
        self._out = f"OUTP{c}"  # output relay root, e.g. "OUTP2"
        self._chan = f"(@{c})"  # channel-list parameter, e.g. "(@2)"
        self._sweep_points = 0  # the instrument cannot report this; see get_sweep
        self.trace = _Trace(link, c)
        self.calc = _Calc(link, c)

    # ---------- output relay ----------

    def set_output_state(self, state: bool) -> bool:
        """Open or close the output relay, then wait for it to finish.

        Synchronized with ``*OPC?`` rather than a sleep: measured 77 ms to close
        and 37 ms to open on the bench B2902B.
        """
        self.write(f"{self._out} {_onoff(state)}")
        self.query("*OPC?")
        return self.get_output_state()

    def get_output_state(self) -> bool:
        """Whether the output relay is closed."""
        return self.query_bool(f"{self._out}?")

    def set_output_off_mode(self, mode: OffMode) -> OffMode:
        """Set what the output presents once off; an :class:`OffMode`.

        Enum arg: <smu>.OffMode.<TAB>
        """
        self.write(f"{self._out}:OFF:MODE {_token(mode, OffMode, 'off mode')}")
        return self.get_output_off_mode()

    def get_output_off_mode(self) -> OffMode:
        """What the output presents once off."""
        return member_from_response(
            self.query(f"{self._out}:OFF:MODE?").strip(), OffMode, "output off mode"
        )

    def set_output_low_terminal(self, terminal: LowTerminal) -> LowTerminal:
        """Float or ground the low terminal; a :class:`LowTerminal`.

        Enum arg: <smu>.LowTerminal.<TAB>

        The output must already be off or the instrument rejects the change.
        """
        tok = _token(terminal, LowTerminal, "low terminal")
        self.write(f"{self._out}:LOW {tok}")
        return self.get_output_low_terminal()

    def get_output_low_terminal(self) -> LowTerminal:
        """Whether the low terminal floats or is grounded."""
        return member_from_response(
            self.query(f"{self._out}:LOW?").strip(), LowTerminal, "output low terminal"
        )

    def set_output_protection_state(self, state: bool) -> bool:
        """Enable or disable automatic output-off on a compliance trip."""
        self.write(f"{self._out}:PROT {_onoff(state)}")
        return self.get_output_protection_state()

    def get_output_protection_state(self) -> bool:
        """Whether the output turns itself off on a compliance trip."""
        return self.query_bool(f"{self._out}:PROT?")

    # ---------- force a level ----------

    def set_source_voltage(
        self,
        volts: float,
        *,
        i_limit: float | tuple[float, float],
        v_range: float | None = None,
        i_range: float | None = None,
        allow_live: bool = False,
    ) -> SourceState:
        """Force ``volts`` V with a required current compliance in A.

        ``i_limit`` is a float for a symmetric limit or a ``(min, max)`` tuple
        for an asymmetric one, where ``min`` is negative. ``v_range`` fixes the
        source range and ``i_range`` the current-measurement range; either left
        at None selects automatic ranging for that quantity.

        Live outputs are fine as long as nothing has to switch. The call is
        refused on a live output only when it would change the force type, leave
        a sweep or list, or change a range, because the instrument applies the
        new setting's stale level as it switches. To move only the level, call
        :meth:`set_source_level`.
        """
        self._apply_force(
            "set_source_voltage", "VOLT", volts, "CURR", i_limit, v_range, i_range, allow_live
        )
        return self.get_source()

    def set_source_current(
        self,
        amps: float,
        *,
        v_limit: float | tuple[float, float],
        i_range: float | None = None,
        v_range: float | None = None,
        allow_live: bool = False,
    ) -> SourceState:
        """Force ``amps`` A with a required voltage compliance in V.

        ``v_limit`` is a float for a symmetric limit or a ``(min, max)`` tuple
        for an asymmetric one, where ``min`` is negative. ``i_range`` fixes the
        source range and ``v_range`` the voltage-measurement range; either left
        at None selects automatic ranging for that quantity.

        Note that the source current range also sets how much internal drop sits
        inside a 2-wire voltage measurement: on the bench B2902B the 100 mA
        range adds about 81 mOhm that the 1 A range does not.

        Refused on a live output under the same conditions as
        :meth:`set_source_voltage`, and only those. To move only the level, call
        :meth:`set_source_level`.
        """
        self._apply_force(
            "set_source_current", "CURR", amps, "VOLT", v_limit, v_range, i_range, allow_live
        )
        return self.get_source()

    def set_source_level(self, value: float) -> float:
        """Move the present force function's level, in that function's unit.

        This is the loop-stepping call: it writes the level and nothing else, so
        it cannot change the force type, the ranges or the limits, and it needs
        no flag on a live output. Volts while forcing voltage, amperes while
        forcing current.

        One caveat, measured on the bench: with source auto-ranging on, the
        instrument changes range by itself to hold the new level. Pin the range
        through ``v_range`` / ``i_range`` first when a loop must not range-switch.
        """
        func = self._get_source_function()
        self.write(f"{self._src}{func} {_num(value)}")
        return self.query_float(f"{self._src}{func}?")

    def get_source_level(self) -> float:
        """The present force function's level, in that function's unit."""
        func = self._get_source_function()
        return self.query_float(f"{self._src}{func}?")

    def get_source(self) -> SourceState:
        """Read the force function, its level, the compliance and the ranges."""
        func = self._get_source_function()
        level = self.query_float(f"{self._src}{func}?")
        i_limit = self._get_compliance("CURR") if func == "VOLT" else None
        v_limit = self._get_compliance("VOLT") if func == "CURR" else None
        if func == "VOLT":
            v_range = self.query_float(f"{self._src}VOLT:RANG?")
            v_auto = self.query_bool(f"{self._src}VOLT:RANG:AUTO?")
            i_range = self.query_float(f"{self._sens}CURR:RANG?")
            i_auto = self.query_bool(f"{self._sens}CURR:RANG:AUTO?")
        else:
            i_range = self.query_float(f"{self._src}CURR:RANG?")
            i_auto = self.query_bool(f"{self._src}CURR:RANG:AUTO?")
            v_range = self.query_float(f"{self._sens}VOLT:RANG?")
            v_auto = self.query_bool(f"{self._sens}VOLT:RANG:AUTO?")
        return SourceState(
            function=func,
            level=level,
            i_limit=i_limit,
            v_limit=v_limit,
            v_range=v_range,
            i_range=i_range,
            v_range_auto=v_auto,
            i_range_auto=i_auto,
            output_state=self.get_output_state(),
        )

    def get_compliance_tripped(self) -> bool:
        """Whether the channel is sitting at its compliance limit.

        Reads the limit that applies to the present force function: the current
        limit while forcing voltage, the voltage limit while forcing current.
        Both directions verified on the bench.
        """
        limited = "CURR" if self._get_source_function() == "VOLT" else "VOLT"
        return self.query_bool(f"{self._sens}{limited}:PROT:TRIP?")

    # ---------- output shape ----------

    def set_source_shape_dc(self, *, allow_live: bool = False) -> ShapeState:
        """Make the source output a continuous DC level.

        Refused on a live output only when the shape actually changes.
        """
        self._apply_shape("set_source_shape_dc", "DC", allow_live)
        return self.get_source_shape()

    def set_source_shape_pulse(
        self, width: float, *, delay: float | None = None, allow_live: bool = False
    ) -> ShapeState:
        """Make the source output a pulse of ``width`` seconds; minimum 50 us.

        Refused on a live output only when the shape actually changes; adjusting
        the width or delay of a pulse already running needs no flag.
        """
        self._apply_shape("set_source_shape_pulse", "PULS", allow_live)
        if delay is not None:
            self.write(f"{self._src}PULS:DEL {_num(delay)}")
        self.write(f"{self._src}PULS:WIDT {_num(width)}")
        return self.get_source_shape()

    def get_source_shape(self) -> ShapeState:
        """Read the output shape and the pulse width and delay in seconds."""
        return ShapeState(
            shape=self.query(f"{self._src}FUNC:SHAP?").strip(),
            width=self.query_float(f"{self._src}PULS:WIDT?"),
            delay=self.query_float(f"{self._src}PULS:DEL?"),
        )

    # ---------- measurement setup ----------

    def set_measure_functions(self, *functions: str) -> list[str]:
        """Choose which quantities an acquisition measures: VOLT, CURR or RES.

        Quantities not named are turned off, so the call states the whole set.
        """
        wanted = [self._meas_func(f) for f in functions]
        if not wanted:
            raise ValueError("specify at least one of VOLT/CURR/RES")
        unwanted = [f for f in ("VOLT", "CURR", "RES") if f not in wanted]
        self.write(f"{self._sens}FUNC:ON {','.join(f'"{f}"' for f in wanted)}")
        if unwanted:
            self.write(f"{self._sens}FUNC:OFF {','.join(f'"{f}"' for f in unwanted)}")
        return self.get_measure_functions()

    def get_measure_functions(self) -> list[str]:
        """Which quantities an acquisition measures."""
        raw = self.query(f"{self._sens}FUNC:ON?").strip()
        return [t.strip().strip('"').upper() for t in raw.split(",") if t.strip()]

    def set_nplc(self, nplc: float) -> float:
        """Set the integration time in power-line cycles.

        One setting covers voltage, current and resistance; the instrument
        shares it across all three (verified on the bench).
        """
        self.write(f"{self._sens}CURR:NPLC {_num(nplc)}")
        return self.get_nplc()

    def get_nplc(self) -> float:
        """The integration time in power-line cycles."""
        return self.query_float(f"{self._sens}CURR:NPLC?")

    def set_nplc_auto_state(self, state: bool) -> bool:
        """Let the instrument pick the integration time by itself."""
        self.write(f"{self._sens}CURR:NPLC:AUTO {_onoff(state)}")
        return self.get_nplc_auto_state()

    def get_nplc_auto_state(self) -> bool:
        """Whether the instrument picks the integration time by itself."""
        return self.query_bool(f"{self._sens}CURR:NPLC:AUTO?")

    def set_aperture(self, seconds: float) -> float:
        """Set the integration time in seconds; shared like :meth:`set_nplc`."""
        self.write(f"{self._sens}CURR:APER {_num(seconds)}")
        return self.get_aperture()

    def get_aperture(self) -> float:
        """The integration time in seconds."""
        return self.query_float(f"{self._sens}CURR:APER?")

    def set_4w_sense_state(self, state: bool) -> bool:
        """Turn 4-wire (Kelvin) sensing on or off.

        With 4-wire on, the channel regulates and measures at the sense leads
        instead of its own terminals. Turning it on with no sense leads attached
        leaves the sense input floating, so only enable it on a channel that is
        actually wired for it.
        """
        self.write(f"{self._sens}REM {_onoff(state)}")
        return self.get_4w_sense_state()

    def get_4w_sense_state(self) -> bool:
        """Whether 4-wire sensing is on."""
        return self.query_bool(f"{self._sens}REM?")

    # ---------- spot measurements ----------

    def meas_voltage(self) -> float:
        """Measure the voltage in V; an overload reads as NaN."""
        return _parse_reading(self.query(f":MEAS:VOLT? {self._chan}"))[0]

    def meas_current(self) -> float:
        """Measure the current in A; an overload reads as NaN."""
        return _parse_reading(self.query(f":MEAS:CURR? {self._chan}"))[0]

    def meas_resistance(
        self, *, offset_compensated: bool | None = None, auto_source: bool | None = None
    ) -> float:
        """Measure the resistance in ohms; an overload reads as NaN.

        ``offset_compensated`` cancels a thermal EMF by measuring twice.
        ``auto_source`` lets the instrument choose the test level rather than
        using the level already set. Either left at None keeps the present
        setting.
        """
        if auto_source is not None:
            self.write(f"{self._sens}RES:MODE {'AUTO' if auto_source else 'MAN'}")
        if offset_compensated is not None:
            self.write(f"{self._sens}RES:OCOM {_onoff(offset_compensated)}")
        return _parse_reading(self.query(f":MEAS:RES? {self._chan}"))[0]

    def meas_vi(self) -> Reading:
        """Measure voltage and current in one acquisition; an overload reads NaN.

        Both quantities come from a single acquisition, so they describe the
        same instant.
        """
        elements = self.query("FORM:ELEM:SENS?").strip()
        columns = [t.strip().upper() for t in elements.split(",") if t.strip()]
        missing = [c for c in ("VOLT", "CURR") if c not in columns]
        if missing:
            raise InstrumentError(
                f"meas_vi() needs {' and '.join(missing)} in the measurement result "
                f"elements, which are currently {columns}. Call "
                "smu.format.set_elements_sense('VOLT', 'CURR') first."
            )
        values = _parse_reading(self.query(f":MEAS? {self._chan}"))
        return Reading(voltage=values[columns.index("VOLT")], current=values[columns.index("CURR")])

    # ---------- sweeps and lists ----------

    def set_sweep_voltage(
        self,
        start: float,
        stop: float,
        points: int,
        *,
        i_limit: float | tuple[float, float],
        spacing: Spacing = Spacing.LINEAR,
        direction: Direction = Direction.UP,
        stair: Stair = Stair.SINGLE,
        ranging: Ranging = Ranging.BEST,
        allow_live: bool = False,
    ) -> SweepState:
        """Configure a voltage staircase from ``start`` to ``stop`` V in ``points`` steps.

        Enum arg: spacing   = <smu>.Spacing.<TAB>
                  direction = <smu>.Direction.<TAB>
                  stair     = <smu>.Stair.<TAB>
                  ranging   = <smu>.Ranging.<TAB>

        Sets the matching transient and acquire trigger counts, so the sweep is
        ready to run. Close the output relay, then call :meth:`meas_sweep`.
        """
        self._require_off("set_sweep_voltage", allow_live)
        return self._configure_sweep(
            "VOLT", start, stop, points, i_limit, spacing, direction, stair, ranging
        )

    def set_sweep_current(
        self,
        start: float,
        stop: float,
        points: int,
        *,
        v_limit: float | tuple[float, float],
        spacing: Spacing = Spacing.LINEAR,
        direction: Direction = Direction.UP,
        stair: Stair = Stair.SINGLE,
        ranging: Ranging = Ranging.BEST,
        allow_live: bool = False,
    ) -> SweepState:
        """Configure a current staircase from ``start`` to ``stop`` A in ``points`` steps.

        Enum arg: spacing   = <smu>.Spacing.<TAB>
                  direction = <smu>.Direction.<TAB>
                  stair     = <smu>.Stair.<TAB>
                  ranging   = <smu>.Ranging.<TAB>
        """
        self._require_off("set_sweep_current", allow_live)
        return self._configure_sweep(
            "CURR", start, stop, points, v_limit, spacing, direction, stair, ranging
        )

    def get_sweep(self) -> SweepState:
        """Read back the staircase-sweep configuration.

        ``points`` is the value this driver last wrote: the instrument's own
        points query needs a MIN/MAX/DEF argument and does not report the
        present setting (verified in iteration 8).
        """
        func = self._get_source_function()
        return SweepState(
            function=func,
            start=self.query_float(f"{self._src}{func}:STAR?"),
            stop=self.query_float(f"{self._src}{func}:STOP?"),
            points=self._sweep_points,
            spacing=self.query(f"{self._src}SWE:SPAC?").strip(),
            direction=self.query(f"{self._src}SWE:DIR?").strip(),
            stair=self.query(f"{self._src}SWE:STA?").strip(),
            ranging=self.query(f"{self._src}SWE:RANG?").strip(),
        )

    def set_list_voltage(
        self,
        values: list[float],
        *,
        i_limit: float | tuple[float, float],
        allow_live: bool = False,
    ) -> int:
        """Program an arbitrary list of voltage steps in V; returns the step count."""
        self._require_off("set_list_voltage", allow_live)
        return self._configure_list("VOLT", values, i_limit)

    def set_list_current(
        self,
        values: list[float],
        *,
        v_limit: float | tuple[float, float],
        allow_live: bool = False,
    ) -> int:
        """Program an arbitrary list of current steps in A; returns the step count."""
        self._require_off("set_list_current", allow_live)
        return self._configure_list("CURR", values, v_limit)

    def get_list(self) -> list[float]:
        """Read back the programmed list steps, in the force function's unit."""
        func = self._get_source_function()
        return _parse_reading(self.query(f"{self._src}LIST:{func}?"))

    def meas_sweep(self) -> SweepReading:
        """Run the configured sweep or list and collect both measured columns.

        One atomic initiate-and-fetch, then the second column from the same run.
        A separate initiate followed by a fetch races the acquisition buffer and
        returns stale data (verified in iteration 8).
        """
        with self.link.temporary_timeout(_SWEEP_TIMEOUT_MS):
            voltage = _parse_reading(self.query(f":READ:ARR:VOLT? {self._chan}"))
            current = _parse_reading(self.query(f":FETC:ARR:CURR? {self._chan}"))
        return SweepReading(voltage=voltage, current=current)

    # ---------- trigger model ----------

    def set_trigger(
        self,
        count: int,
        *,
        source: TriggerSource = TriggerSource.AUTO_INTERNAL,
        delay: float | None = None,
        interval: float | None = None,
        action: TriggerAction = TriggerAction.ALL,
    ) -> TriggerState:
        """Set how many triggers to take and what supplies them.

        Enum arg: source = <smu>.TriggerSource.<TAB>
                  action = <smu>.TriggerAction.<TAB>

        ``interval`` is the period in seconds and applies when ``source`` is
        TIMER. ``action`` selects the acquisition layer, the source layer, or
        both.
        """
        node = self._action_node(action)
        self.write(f"TRIG{self.c}{node}:SOUR {_token(source, TriggerSource, 'trigger source')}")
        self.write(f"TRIG{self.c}{node}:COUN {int(count)}")
        if delay is not None:
            self.write(f"TRIG{self.c}{node}:DEL {_num(delay)}")
        if interval is not None:
            self.write(f"TRIG{self.c}{node}:TIM {_num(interval)}")
        return self.get_trigger()

    def get_trigger(self) -> TriggerState:
        """Read the acquisition layer's trigger configuration."""
        return TriggerState(
            source=self.query(f"TRIG{self.c}:ACQ:SOUR?").strip(),
            count=self.query_int(f"TRIG{self.c}:ACQ:COUN?"),
            delay=self.query_float(f"TRIG{self.c}:ACQ:DEL?"),
            interval=self.query_float(f"TRIG{self.c}:ACQ:TIM?"),
        )

    def cmd_initiate(self, action: TriggerAction = TriggerAction.ALL) -> None:
        """Start the trigger model, arming the channel to source or acquire.

        Enum arg: <smu>.TriggerAction.<TAB>
        """
        self.write(f":INIT{self._action_node(action)} {self._chan}")

    def cmd_abort(self, action: TriggerAction = TriggerAction.ALL) -> None:
        """Stop the trigger model and return the channel to idle.

        Enum arg: <smu>.TriggerAction.<TAB>
        """
        self.write(f":ABOR{self._action_node(action)} {self._chan}")

    def cmd_trigger(self, action: TriggerAction = TriggerAction.ALL) -> None:
        """Send this channel one immediate trigger.

        Enum arg: <smu>.TriggerAction.<TAB>
        """
        self.write(f":TRIG{self._action_node(action)} {self._chan}")

    def cmd_wait_idle(self, action: TriggerAction = TriggerAction.ALL) -> bool:
        """Block until the channel returns to idle, then report that it did.

        Enum arg: <smu>.TriggerAction.<TAB>
        """
        return self.query_bool(f":IDLE{self.c}{self._action_node(action)}?")

    # ---------- underscored plumbing (design standard 5.3) ----------

    def _require_off(self, method: str, allow_live: bool) -> None:
        """Refuse a live output outright; for helpers that always switch mode."""
        if not allow_live and self.get_output_state():
            raise InstrumentError(
                f"{method}() refuses a live output: {self.name} would apply the new "
                "mode's stale level the moment the mode changes. Call "
                "set_output_state(False) first, or pass allow_live=True."
            )

    def _refuse_live_switch(self, method: str, changes: list[str], allow_live: bool) -> None:
        """Refuse a live output only when something would actually switch.

        A level or limit write is safe on a live output and needs no flag. What
        is not safe is a force-type change, leaving a sweep or list, or a range
        change: the instrument applies the new setting's stale level as it
        switches (bench-verified, see the module docstring). Only those raise.
        """
        if not changes or allow_live or not self.get_output_state():
            return
        raise InstrumentError(
            f"{method}() refuses a live output because it would change "
            f"{', and '.join(changes)}, and {self.name} applies the new setting's "
            "stale level as it switches. Use set_source_level() to move only the "
            "level, or call set_output_state(False) first, or pass allow_live=True."
        )

    def _plan_ranges(
        self, *, source: str, v_range: float | None, i_range: float | None
    ) -> tuple[list[str], list[str]]:
        """Return the range writes that would actually change something.

        A range value equal to the present read-back is not a change, so a loop
        that passes the same range every pass never trips the live refusal.
        """
        writes: list[str] = []
        changes: list[str] = []
        for quantity, value in (("VOLT", v_range), ("CURR", i_range)):
            root = self._src if quantity == source else self._sens
            auto_now = self.query_bool(f"{root}{quantity}:RANG:AUTO?")
            if value is None:
                if not auto_now:
                    writes.append(f"{root}{quantity}:RANG:AUTO ON")
                    changes.append(f"the {quantity} range to automatic")
                continue
            present = self.query_float(f"{root}{quantity}:RANG?")
            if auto_now or not math.isclose(present, float(value), rel_tol=1e-9, abs_tol=1e-15):
                writes.append(f"{root}{quantity}:RANG:AUTO OFF")
                writes.append(f"{root}{quantity}:RANG {_num(value)}")
                changes.append(f"the {quantity} range to {_num(value)}")
        return writes, changes

    def _apply_force(
        self,
        method: str,
        func: str,
        level: float,
        limit_quantity: str,
        limit: float | tuple[float, float],
        v_range: float | None,
        i_range: float | None,
        allow_live: bool,
    ) -> None:
        """Switch what needs switching, then write the limit, then the level.

        Range and mode first, limit before level, level last, and nothing here
        closes the output relay (design standard 6.2). Writes that would change
        nothing are skipped, so a live call that only moves the level issues no
        switching command at all.
        """
        writes: list[str] = []
        changes: list[str] = []
        if self._get_source_function() != func:
            writes.append(f"{self._src}FUNC:MODE {func}")
            changes.append(f"the force type to {func}")
        if self.query(f"{self._src}{func}:MODE?").strip().upper() != "FIX":
            writes.append(f"{self._src}{func}:MODE FIX")
            changes.append("the source out of sweep or list mode")
        range_writes, range_changes = self._plan_ranges(
            source=func, v_range=v_range, i_range=i_range
        )
        writes += range_writes
        changes += range_changes

        self._refuse_live_switch(method, changes, allow_live)

        for cmd in writes:
            self.write(cmd)
        self._set_compliance(limit_quantity, limit)
        self.write(f"{self._src}{func} {_num(level)}")

    @staticmethod
    def _meas_func(func: str) -> str:
        """Normalize a measurement-quantity name to VOLT, CURR or RES."""
        f = str(func).strip().upper()
        if f in ("V", "VOLT", "VOLTAGE"):
            return "VOLT"
        if f in ("I", "CURR", "CURRENT"):
            return "CURR"
        if f in ("R", "RES", "RESISTANCE"):
            return "RES"
        raise ValueError(f"measurement quantity must be VOLT/CURR/RES, got {func!r}")

    def _action_node(self, action) -> str:
        return self._ACTION_NODE[_token(action, TriggerAction, "trigger action")]

    def _get_source_function(self) -> str:
        """Read the force function (SOUR<c>:FUNC:MODE?)."""
        return self.query(f"{self._src}FUNC:MODE?").strip().upper()

    def _set_source_function(self, func: str) -> None:
        """Select the force function (SOUR<c>:FUNC:MODE VOLT|CURR)."""
        self.write(f"{self._src}FUNC:MODE {func}")

    def _apply_shape(self, method: str, shape: str, allow_live: bool) -> None:
        """Change the output shape (SOUR<c>:FUNC:SHAPe), only if it differs."""
        if self.query(f"{self._src}FUNC:SHAP?").strip().upper().startswith(shape):
            return
        self._refuse_live_switch(method, [f"the output shape to {shape}"], allow_live)
        self.write(f"{self._src}FUNC:SHAP {shape}")

    def _set_compliance(self, quantity: str, limit: float | tuple[float, float]) -> None:
        """Write the compliance for VOLT or CURR (SENS<c>:<q>:PROT[:POS|:NEG]).

        A scalar goes to the symmetric node, which relinks both polarities. A
        pair writes the two polarities separately. Verified on the bench: after
        a POS-only write the symmetric query answers with the 9.91E37 overload
        sentinel, so this driver never reads it back.
        """
        if isinstance(limit, (tuple, list)):
            low, high = (float(v) for v in limit)
            self.write(f"{self._sens}{quantity}:PROT:NEG {_num(-abs(low))}")
            self.write(f"{self._sens}{quantity}:PROT:POS {_num(abs(high))}")
        else:
            self.write(f"{self._sens}{quantity}:PROT {_num(abs(float(limit)))}")

    def _get_compliance(self, quantity: str) -> float | tuple[float, float]:
        """Read the compliance as a scalar when symmetric, else a (min, max) pair."""
        pos = self.query_float(f"{self._sens}{quantity}:PROT:POS?")
        neg = self.query_float(f"{self._sens}{quantity}:PROT:NEG?")
        if math.isclose(abs(pos), abs(neg), rel_tol=1e-9, abs_tol=1e-12):
            return abs(pos)
        return (-abs(neg), abs(pos))

    def _configure_sweep(
        self,
        func: str,
        start: float,
        stop: float,
        points: int,
        limit,
        spacing,
        direction,
        stair,
        ranging,
    ) -> SweepState:
        """Write a staircase configuration and match the trigger counts to it."""
        n = int(points)
        if n < 1:
            raise ValueError(f"a sweep needs at least one point, got {points!r}")
        self._set_source_function(func)
        self.write(f"{self._src}{func}:MODE SWE")
        self.write(f"{self._src}SWE:SPAC {_token(spacing, Spacing, 'spacing')}")
        self.write(f"{self._src}SWE:DIR {_token(direction, Direction, 'direction')}")
        self.write(f"{self._src}SWE:STA {_token(stair, Stair, 'stair')}")
        self.write(f"{self._src}SWE:RANG {_token(ranging, Ranging, 'ranging')}")
        self._set_compliance("CURR" if func == "VOLT" else "VOLT", limit)
        self.write(f"{self._src}{func}:STAR {_num(start)}")
        self.write(f"{self._src}{func}:STOP {_num(stop)}")
        self.write(f"{self._src}SWE:POIN {n}")
        self._sweep_points = n
        self.write(f"TRIG{self.c}:TRAN:COUN {n}")
        self.write(f"TRIG{self.c}:ACQ:COUN {n}")
        return self.get_sweep()

    def _configure_list(self, func: str, values: list[float], limit) -> int:
        """Write a list configuration and match the trigger counts to it."""
        if not values:
            raise ValueError("a list needs at least one value")
        self._set_source_function(func)
        self.write(f"{self._src}{func}:MODE LIST")
        self._set_compliance("CURR" if func == "VOLT" else "VOLT", limit)
        self.write(f"{self._src}LIST:{func} {','.join(_num(v) for v in values)}")
        n = self.query_int(f"{self._src}LIST:{func}:POIN?")
        self.write(f"TRIG{self.c}:TRAN:COUN {n}")
        self.write(f"TRIG{self.c}:ACQ:COUN {n}")
        return n

    def _append_list(self, func: str, values: list[float]) -> int:
        """Append points to the programmed list (SOUR<c>:LIST:<f>:APPend)."""
        f = self._meas_func(func)
        if not values:
            raise ValueError("append needs at least one value")
        self.write(f"{self._src}LIST:{f}:APP {','.join(_num(v) for v in values)}")
        return self.query_int(f"{self._src}LIST:{f}:POIN?")

    def _set_arm_count(self, count: int, action=TriggerAction.ALL) -> None:
        """Set the arm-layer count (ARM<c><action>:COUNt)."""
        self.write(f"ARM{self.c}{self._action_node(action)}:COUN {int(count)}")

    def _set_arm_source(self, source, action=TriggerAction.ALL) -> None:
        """Set the arm-layer source (ARM<c><action>:SOURce)."""
        tok = _token(source, TriggerSource, "trigger source")
        self.write(f"ARM{self.c}{self._action_node(action)}:SOUR {tok}")

    # ---------- the atomic layer the helpers compose (design standard 5.3) ----------
    #
    # The underscore is a completion tier, not a fence. A helper covers the
    # common case; these reach one value at a time when the helper composes the
    # wrong set. They are supported, and a script or a bench session may call
    # them directly.

    def _set_source_range(self, quantity: str, value: float) -> float:
        """Fix the source range for VOLT or CURR (SOUR<c>:<q>:RANGe)."""
        q = self._meas_func(quantity)
        self.write(f"{self._src}{q}:RANG:AUTO OFF")
        self.write(f"{self._src}{q}:RANG {_num(value)}")
        return self.query_float(f"{self._src}{q}:RANG?")

    def _set_source_range_auto(self, quantity: str, state: bool) -> bool:
        """Turn source auto-ranging on or off (SOUR<c>:<q>:RANGe:AUTO)."""
        q = self._meas_func(quantity)
        self.write(f"{self._src}{q}:RANG:AUTO {_onoff(state)}")
        return self.query_bool(f"{self._src}{q}:RANG:AUTO?")

    def _set_sense_range(self, quantity: str, value: float) -> float:
        """Fix the measurement range for VOLT, CURR or RES (SENS<c>:<q>:RANGe)."""
        q = self._meas_func(quantity)
        self.write(f"{self._sens}{q}:RANG:AUTO OFF")
        self.write(f"{self._sens}{q}:RANG {_num(value)}")
        return self.query_float(f"{self._sens}{q}:RANG?")

    def _set_sense_range_auto(self, quantity: str, state: bool) -> bool:
        """Turn measurement auto-ranging on or off (SENS<c>:<q>:RANGe:AUTO)."""
        q = self._meas_func(quantity)
        self.write(f"{self._sens}{q}:RANG:AUTO {_onoff(state)}")
        return self.query_bool(f"{self._sens}{q}:RANG:AUTO?")

    def _set_compliance_positive(self, quantity: str, value: float) -> float:
        """Set one polarity of the compliance (SENS<c>:<q>:PROTection:POSitive)."""
        q = self._meas_func(quantity)
        self.write(f"{self._sens}{q}:PROT:POS {_num(abs(value))}")
        return self.query_float(f"{self._sens}{q}:PROT:POS?")

    def _set_compliance_negative(self, quantity: str, value: float) -> float:
        """Set one polarity of the compliance (SENS<c>:<q>:PROTection:NEGative)."""
        q = self._meas_func(quantity)
        self.write(f"{self._sens}{q}:PROT:NEG {_num(-abs(value))}")
        return self.query_float(f"{self._sens}{q}:PROT:NEG?")

    def _get_measure_function_state(self, quantity: str) -> bool:
        """Whether one measurement quantity is enabled (SENS<c>:FUNCtion:STATe?)."""
        return self.query_bool(f'{self._sens}FUNC:STAT? "{self._meas_func(quantity)}"')

    def _set_source_mode(self, quantity: str, mode: str) -> str:
        """Select FIXed, SWEep or LIST for a quantity (SOUR<c>:<q>:MODE)."""
        q = self._meas_func(quantity)
        self.write(f"{self._src}{q}:MODE {mode}")
        return self.query(f"{self._src}{q}:MODE?").strip()

    def _set_sweep_start(self, quantity: str, value: float) -> float:
        """Set the sweep start level (SOUR<c>:<q>:STARt)."""
        q = self._meas_func(quantity)
        self.write(f"{self._src}{q}:STAR {_num(value)}")
        return self.query_float(f"{self._src}{q}:STAR?")

    def _set_sweep_stop(self, quantity: str, value: float) -> float:
        """Set the sweep stop level (SOUR<c>:<q>:STOP)."""
        q = self._meas_func(quantity)
        self.write(f"{self._src}{q}:STOP {_num(value)}")
        return self.query_float(f"{self._src}{q}:STOP?")

    def _set_sweep_points(self, points: int) -> int:
        """Set the sweep step count (SOUR<c>:SWEep:POINts).

        Returns the value written. The instrument's own points query needs a
        MIN/MAX/DEF argument and does not report the present setting.
        """
        n = int(points)
        self.write(f"{self._src}SWE:POIN {n}")
        self._sweep_points = n
        return n

    def _set_pulse_width(self, seconds: float) -> float:
        """Set the pulse width (SOUR<c>:PULSe:WIDTh); minimum 50 us."""
        self.write(f"{self._src}PULS:WIDT {_num(seconds)}")
        return self.query_float(f"{self._src}PULS:WIDT?")

    def _set_pulse_delay(self, seconds: float) -> float:
        """Set the pulse delay (SOUR<c>:PULSe:DELay)."""
        self.write(f"{self._src}PULS:DEL {_num(seconds)}")
        return self.query_float(f"{self._src}PULS:DEL?")

    def _set_trigger_count(self, count: int, action=TriggerAction.ALL) -> None:
        """Set the trigger count alone (TRIG<c><action>:COUNt)."""
        self.write(f"TRIG{self.c}{self._action_node(action)}:COUN {int(count)}")

    def _set_trigger_delay(self, seconds: float, action=TriggerAction.ALL) -> None:
        """Set the trigger delay alone (TRIG<c><action>:DELay)."""
        self.write(f"TRIG{self.c}{self._action_node(action)}:DEL {_num(seconds)}")

    def _set_trigger_timer(self, seconds: float, action=TriggerAction.ALL) -> None:
        """Set the timer-trigger period alone (TRIG<c><action>:TIMer)."""
        self.write(f"TRIG{self.c}{self._action_node(action)}:TIM {_num(seconds)}")

    def _set_trigger_source(self, source, action=TriggerAction.ALL) -> None:
        """Set the trigger source alone (TRIG<c><action>:SOURce)."""
        tok = _token(source, TriggerSource, "trigger source")
        self.write(f"TRIG{self.c}{self._action_node(action)}:SOUR {tok}")

    def _measure_all(self) -> list[float]:
        """Measure every enabled quantity in one acquisition (MEASure? (@c)).

        Returns the values in the instrument's result-element order, so read
        ``smu.format.get_elements_sense()`` to know what the columns are.
        """
        return _parse_reading(self.query(f":MEAS? {self._chan}"))

    def _read(self) -> list[float]:
        """Initiate and fetch every enabled quantity (READ? (@c))."""
        return _parse_reading(self.query(f":READ? {self._chan}"))

    def _read_scalar(self, quantity: str) -> float:
        """Initiate and fetch one quantity (READ:<q>? (@c))."""
        return _parse_reading(self.query(f":READ:{self._meas_func(quantity)}? {self._chan}"))[0]

    def _read_array(self, quantity: str) -> list[float]:
        """Initiate and fetch one quantity's whole array (READ:ARRay:<q>? (@c))."""
        with self.link.temporary_timeout(_SWEEP_TIMEOUT_MS):
            return _parse_reading(
                self.query(f":READ:ARR:{self._meas_func(quantity)}? {self._chan}")
            )

    def _fetch(self) -> list[float]:
        """Re-read the last acquisition without triggering a new one (FETCh? (@c))."""
        return _parse_reading(self.query(f":FETC? {self._chan}"))

    def _fetch_scalar(self, quantity: str) -> float:
        """Re-read one quantity from the last acquisition (FETCh:<q>? (@c))."""
        return _parse_reading(self.query(f":FETC:{self._meas_func(quantity)}? {self._chan}"))[0]

    def _fetch_array(self, quantity: str) -> list[float]:
        """Re-read one quantity's whole array (FETCh:ARRay:<q>? (@c))."""
        with self.link.temporary_timeout(_SWEEP_TIMEOUT_MS):
            return _parse_reading(
                self.query(f":FETC:ARR:{self._meas_func(quantity)}? {self._chan}")
            )


# ---------------------------------------------------------------- instrument


class Keysight_B2900(Instrument):
    """Keysight B2900 series precision Source/Measure Unit.

    ``connect()`` opens the pinned LAN resource, validates ``*IDN?``, attaches
    ``ch2`` on the dual-channel models and clears the status and error queues. A
    VISA timeout on first contact is fatal: the operator must power cycle, and
    the link is never retry-hammered.
    """

    # Reachable as smu.<Enum> from the instrument handle; see the note above.
    ByteOrder = ByteOrder
    CalcElement = CalcElement
    CalcFeed = CalcFeed
    Control = Control
    Data = Data
    Direction = Direction
    LimitFunction = LimitFunction
    LowTerminal = LowTerminal
    NumberBase = NumberBase
    OffMode = OffMode
    Ranging = Ranging
    SenseElement = SenseElement
    Spacing = Spacing
    Stair = Stair
    Statistic = Statistic
    StatusRegister = StatusRegister
    Timestamp = Timestamp
    TraceFeed = TraceFeed
    TriggerAction = TriggerAction
    TriggerSource = TriggerSource
    View = View

    ch1: _SmuChannel
    ch2: _SmuChannel  # attached by connect() on the B2902 and B2912 only
    format: _Format
    display: _Display
    status: _Status

    def __init__(
        self,
        resource_name: str,
        *,
        connect_timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        """
        Parameters
        ----------
        resource_name : str
            pyvisa resource for the SMU (default the bench VXI-11 string).
        connect_timeout_ms : int
            I/O timeout used for connect and normal operation, in milliseconds.
        """
        super().__init__(resource_name, name="SMU")
        self.link.default_timeout_ms = connect_timeout_ms
        self._connect_timeout_ms = connect_timeout_ms

        self.model: str | None = None
        self.firmware: str | None = None
        self.num_channels: int | None = None

        # ch1 exists on every model in the family, so it is built here and the
        # API-map generator can walk it offline. Model-dependent channels are
        # attached by connect(), which is the only place *IDN? is known.
        self.ch1 = _SmuChannel(self.link, 1)
        self.format = _Format(self.link)
        self.display = _Display(self.link)
        self.status = _Status(self.link)

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the pinned resource, validate the identity, attach the channels.

        A VISA timeout is fatal (:class:`InstrumentUnresponsiveError`): the
        operator must power cycle, and the link is never retried. An identity
        from something other than a B29xx raises :class:`InstrumentError`, since
        a wrong resource is a wiring fact to fix rather than one to probe around.
        """
        to = self._connect_timeout_ms if timeout_ms is None else timeout_ms
        try:
            self.link.connect(timeout_ms=to)
            idn = self.link.query("*IDN?").strip()
        except visa.VisaIOError as exc:
            if exc.error_code == visa.constants.StatusCode.error_timeout:
                raise InstrumentUnresponsiveError(
                    f"{self.resource_name} did not answer *IDN? (VISA timeout). Stop "
                    "all work - the operator must power cycle the SMU."
                ) from exc
            raise
        if not _IDN_RE.search(idn):
            self.link.disconnect()
            raise InstrumentError(
                f"{self.resource_name} did not answer with a B29xx identity: {idn!r}. "
                "Check the resource; this driver does not scan for the instrument."
            )
        self.idn = idn
        self._on_connect(idn)
        self.link.write("*CLS")
        return idn

    def _on_connect(self, idn: str) -> None:
        """Parse the identity and attach the channels this model actually has."""
        match = _IDN_RE.search(idn)
        if not match:
            raise InstrumentError(f"Not a recognized B29xx identity: {idn!r}")
        model = match.group(1).upper()
        self.model = model
        fields = [f.strip() for f in idn.split(",")]
        self.firmware = fields[3] if len(fields) >= 4 else None

        info = _MODEL_INFO.get(model[:5])
        if info is not None:
            self.num_channels = info["channels"]
        elif model[4:5].isdigit():
            # Fallback: the 4th model digit is the channel count (B290x/B291x).
            self.num_channels = 2 if model[4] == "2" else 1
        else:
            self.num_channels = 2

        # A single-channel model gets no ch2 attribute at all, rather than one
        # that raises on every call (design standard 5.1).
        if self.num_channels >= 2:
            if not isinstance(getattr(self, "ch2", None), _SmuChannel):
                self.ch2 = _SmuChannel(self.link, 2)
        elif isinstance(getattr(self, "ch2", None), _SmuChannel):
            del self.ch2

    def safe_state(self) -> None:
        """Open every channel's output relay before the link closes.

        The SMU sources real power, so this de-energizes on the way out.
        Best-effort: it never raises out of disconnect.
        """
        for ch in self.channels:
            with contextlib.suppress(Exception):
                self.link.write(f"OUTP{ch.c} OFF")

    @property
    def channels(self) -> list[_SmuChannel]:
        """The channel nodes this model has, in order."""
        found = [self.ch1]
        ch2 = getattr(self, "ch2", None)
        if isinstance(ch2, _SmuChannel):
            found.append(ch2)
        return found

    def channel(self, n: int) -> _SmuChannel:
        """Return the channel node for ``n``."""
        for ch in self.channels:
            if ch.c == n:
                return ch
        have = [ch.c for ch in self.channels]
        raise ValueError(
            f"channel must be one of {have} on this {self.model or 'B29xx'}, got {n!r}"
        )

    # ---------- output enable ----------

    def set_output_state_all(self, state: bool) -> list[bool]:
        """Open or close every channel's output relay.

        Written per channel and then synchronized once: the instrument rejects
        a channel-list output write with -108 "Parameter not allowed" (verified
        on the bench), so this is not simultaneous in hardware.
        """
        for ch in self.channels:
            self.link.write(f"OUTP{ch.c} {_onoff(state)}")
        self.link.query("*OPC?")
        return self.get_output_state_all()

    def get_output_state_all(self) -> list[bool]:
        """Whether each channel's output relay is closed, in channel order."""
        return [ch.get_output_state() for ch in self.channels]

    # ---------- system ----------

    def cmd_reset(self) -> None:
        """Reset to the power-on default state and wait for it to finish."""
        self.link.write("*RST")
        self.link.query("*OPC?")

    def cmd_clear_status(self) -> None:
        """Clear the status byte and every event register."""
        self.link.write("*CLS")

    def cmd_system_preset(self) -> None:
        """Preset the instrument settings and the front panel."""
        self.link.write("SYST:PRES")

    def cmd_trigger_bus(self) -> None:
        """Send a bus trigger to every channel whose trigger source is BUS."""
        self.link.write("*TRG")

    def cmd_beep(self, frequency: float = 440.0, duration: float = 0.2) -> None:
        """Sound a beep of ``frequency`` Hz for ``duration`` seconds."""
        self.link.write(f"SYST:BEEP {_num(frequency)},{_num(duration)}")

    def cmd_save_state(self, slot: int) -> None:
        """Save the full instrument state to memory slot 0 to 4."""
        self.link.write(f"*SAV {self._check_slot(slot)}")

    def cmd_recall_state(self, slot: int) -> None:
        """Recall a saved instrument state from memory slot 0 to 4.

        This restores the saved output on/off state too, which on tied-together
        channels can re-energize both at once.
        """
        self.link.write(f"*RCL {self._check_slot(slot)}")
        self.link.query("*OPC?")

    def set_beeper_state(self, state: bool) -> bool:
        """Enable or disable the beeper."""
        self.link.write(f"SYST:BEEP:STAT {_onoff(state)}")
        return self.get_beeper_state()

    def get_beeper_state(self) -> bool:
        """Whether the beeper is enabled."""
        return self.link.query_bool("SYST:BEEP:STAT?")

    def set_line_frequency(self, hz: int) -> int:
        """Set the power-line frequency in Hz, 50 or 60; it survives a reset."""
        if hz not in (50, 60):
            raise ValueError(f"line frequency must be 50 or 60, got {hz!r}")
        self.link.write(f"SYST:LFR {hz}")
        return self.get_line_frequency()

    def get_line_frequency(self) -> int:
        """The power-line frequency in Hz that integration times are counted in."""
        return self.link.query_int("SYST:LFR?")

    def get_scpi_version(self) -> str:
        """The SCPI version the instrument claims."""
        return self.link.query("SYST:VERS?").strip()

    def get_interlock_tripped(self) -> bool:
        """Whether the safety interlock is open, which caps the output at 42 V."""
        return self.link.query_bool("SYST:INT:TRIP?")

    def get_status_byte(self) -> int:
        """Read the status byte."""
        return self.link.query_int("*STB?")

    def get_event_status(self) -> int:
        """Read and clear the standard event status register."""
        return self.link.query_int("*ESR?")

    def get_errors(self) -> list[str]:
        """Drain and return the error queue; an empty list means clean."""
        found: list[str] = []
        for _ in range(50):
            resp = self.link.query("SYST:ERR?").strip()
            code = resp.split(",", 1)[0].strip().lstrip("+")
            if code == "0":
                break
            found.append(resp)
        return found

    def self_test(self) -> int:
        """Run the internal self-test; 0 means pass."""
        with self.link.temporary_timeout(60_000):
            return self.link.query_int("*TST?")

    def _get_error_count(self) -> int:
        """How many entries the error queue holds (SYSTem:ERRor:COUNt?)."""
        return self.link.query_int("SYST:ERR:COUN?")

    def _get_error(self) -> tuple[int, str]:
        """Pop one entry off the error queue as (code, message) (SYSTem:ERRor?)."""
        resp = self.link.query("SYST:ERR?").strip()
        code, _, message = resp.partition(",")
        return int(code), message.strip().strip('"')

    @staticmethod
    def _check_slot(slot: int) -> int:
        if not 0 <= int(slot) <= 4:
            raise ValueError(f"state slot must be 0..4, got {slot!r}")
        return int(slot)

    # ---------- snapshot / restore (bench-state discipline) ----------

    def snapshot(self) -> dict:
        """Read back every setting this driver can change, for an exact restore.

        Read-back queries only, so no memory slot is clobbered. Faithful to what
        was found, never to a default.
        """
        snap: dict = {"channels": {}}
        for ch in self.channels:
            snap["channels"][ch.c] = {
                "function": ch._get_source_function(),
                "shape": ch.query(f"{ch._src}FUNC:SHAP?").strip(),
                "voltage": ch.query_float(f"{ch._src}VOLT?"),
                "current": ch.query_float(f"{ch._src}CURR?"),
                "volt_mode": ch.query(f"{ch._src}VOLT:MODE?").strip(),
                "curr_mode": ch.query(f"{ch._src}CURR:MODE?").strip(),
                "i_limit": ch._get_compliance("CURR"),
                "v_limit": ch._get_compliance("VOLT"),
                "src_volt_range": ch.query_float(f"{ch._src}VOLT:RANG?"),
                "src_volt_auto": ch.query_bool(f"{ch._src}VOLT:RANG:AUTO?"),
                "src_curr_range": ch.query_float(f"{ch._src}CURR:RANG?"),
                "src_curr_auto": ch.query_bool(f"{ch._src}CURR:RANG:AUTO?"),
                "sens_volt_range": ch.query_float(f"{ch._sens}VOLT:RANG?"),
                "sens_volt_auto": ch.query_bool(f"{ch._sens}VOLT:RANG:AUTO?"),
                "sens_curr_range": ch.query_float(f"{ch._sens}CURR:RANG?"),
                "sens_curr_auto": ch.query_bool(f"{ch._sens}CURR:RANG:AUTO?"),
                "nplc": ch.get_nplc(),
                "four_wire": ch.get_4w_sense_state(),
                "measure": ch.get_measure_functions(),
                "off_mode": ch.get_output_off_mode(),
                "output": ch.get_output_state(),
            }
        return snap

    def restore(self, snapshot: dict) -> None:
        """Restore the settings captured by :meth:`snapshot`, exactly as found.

        Every output is opened first, then the settings are written, then the
        output relays return to the state they were found in. Restoring two
        channels that were both energized re-energizes both; the caller owns
        that. :meth:`disconnect` de-energizes regardless.
        """
        for ch in self.channels:
            state = snapshot.get("channels", {}).get(ch.c)
            if state is None:
                continue
            ch.set_output_state(False)
            ch._set_source_function(state["function"])
            ch.write(f"{ch._src}FUNC:SHAP {state['shape']}")
            for quantity, key in (("VOLT", "volt_mode"), ("CURR", "curr_mode")):
                ch.write(f"{ch._src}{quantity}:MODE {state[key]}")
            for root, quantity, rkey, akey in (
                (ch._src, "VOLT", "src_volt_range", "src_volt_auto"),
                (ch._src, "CURR", "src_curr_range", "src_curr_auto"),
                (ch._sens, "VOLT", "sens_volt_range", "sens_volt_auto"),
                (ch._sens, "CURR", "sens_curr_range", "sens_curr_auto"),
            ):
                ch.write(f"{root}{quantity}:RANG {_num(state[rkey])}")
                ch.write(f"{root}{quantity}:RANG:AUTO {_onoff(state[akey])}")
            ch._set_compliance("CURR", state["i_limit"])
            ch._set_compliance("VOLT", state["v_limit"])
            ch.set_nplc(state["nplc"])
            ch.set_4w_sense_state(state["four_wire"])
            ch.set_output_off_mode(state["off_mode"])
            if state["measure"]:
                ch.set_measure_functions(*state["measure"])
            ch.write(f"{ch._src}VOLT {_num(state['voltage'])}")
            ch.write(f"{ch._src}CURR {_num(state['current'])}")
        for ch in self.channels:
            state = snapshot.get("channels", {}).get(ch.c)
            if state is not None and state["output"]:
                ch.set_output_state(True)

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator:
        """Context manager: snapshot on enter, restore the exact state on exit."""
        snap = self.snapshot()
        try:
            yield snap
        finally:
            self.restore(snap)


# ---------------------------------------------------------------- self-test


def _demo_source_measure(smu: Keysight_B2900) -> None:
    """Safe CH1-only source and measure exercise (energizes CH1 briefly).

    CH1 forces 0.2 V with a 1 mA compliance into the tied node while CH2 stays
    open, so no current fight is possible. Ends with the output off.
    """
    ch = smu.ch1
    smu.cmd_reset()
    print(f"  measure functions   -> {ch.set_measure_functions('VOLT', 'CURR')}")
    print(f"  NPLC                -> {ch.set_nplc(1):g}")
    state = ch.set_source_voltage(0.2, i_limit=1e-3)
    print(f"  set_source_voltage  -> {state.function} {state.level:.4f} V, limit {state.i_limit} A")
    ch.set_output_state(True)
    try:
        r = ch.meas_vi()
        print(f"  meas_vi()           -> {r.voltage:.4f} V, {r.current:.3e} A")
        print(f"  meas_voltage()      -> {ch.meas_voltage():.4f} V")
        print(f"  meas_current()      -> {ch.meas_current():.3e} A")
        print(f"  compliance tripped  -> {ch.get_compliance_tripped()}")
    finally:
        print(f"  output off          -> {ch.set_output_state(False)}")
    smu.cmd_reset()


def _demo_sweep(smu: Keysight_B2900) -> None:
    """Safe current staircase and list on CH1 into a CH2 zero-volt clamp.

    CH1 sweeps current into CH2, which holds the tied node at 0 V with generous
    compliance. Both channels stay near 0 V and at or below 50 mA.
    """
    if len(smu.channels) < 2:
        print("  (skipped: this demo needs the tied CH1/CH2 pair of a 2-channel model)")
        return
    src, clamp = smu.ch1, smu.channel(2)
    smu.cmd_reset()
    clamp.set_measure_functions("CURR")
    clamp.set_source_voltage(0.0, i_limit=0.5)
    src.set_measure_functions("VOLT", "CURR")

    sweep = src.set_sweep_current(0.0, 0.05, 5, v_limit=6.0)
    print(
        f"  set_sweep_current   -> {sweep.function} {sweep.start}..{sweep.stop} A x{sweep.points}"
    )
    clamp.set_output_state(True)
    src.set_output_state(True)
    try:
        pts = src.meas_sweep()
        print(f"  staircase currents  -> {[round(a * 1e3, 3) for a in pts.current]} mA")
        src.set_output_state(False)  # the list helper changes mode, so de-energize first
        n = src.set_list_current([0.010, 0.030, 0.020], v_limit=6.0)
        src.set_output_state(True)
        pts = src.meas_sweep()
        print(f"  list of {n} currents  -> {[round(a * 1e3, 3) for a in pts.current]} mA")
    finally:
        src.set_output_state(False)
        clamp.set_output_state(False)
    smu.cmd_reset()


def _demo_config(smu: Keysight_B2900) -> None:
    """Non-energizing read-back of the format, display, trace and calc surface."""
    smu.cmd_reset()
    print(f"  format data         -> {smu.format.set_data(smu.format.Data.ASCII)}")
    print(f"  format elements     -> {smu.format.set_elements_sense('VOLT', 'CURR')}")
    print(f"  format byte order   -> {smu.format.set_byte_order('NORMAL')}")
    print(f"  display view        -> {smu.display.set_view(smu.display.View.DUAL)}")
    print(f"  display digits      -> {smu.display.set_digits(6)}")
    print(f"  trace free          -> {smu.ch1.trace.get_free()}")
    print(f"  trace feed          -> {smu.ch1.trace.set_feed('SENSE')}")
    print(f"  trace points        -> {smu.ch1.trace.set_points(100)}")
    print(f"  math catalog        -> {smu.ch1.calc.get_math_catalog()}")
    print(f"  calc feed           -> {smu.ch1.calc.set_feed('CURRENT')}")
    print(f"  limit function      -> {smu.ch1.calc.set_limit_function('LIMIT')}")
    print(
        f"  limit upper/lower   -> {smu.ch1.calc.set_limit_upper(1e-3)}"
        f"/{smu.ch1.calc.set_limit_lower(-1e-3)}"
    )
    smu.cmd_reset()


def _demo_state(smu: Keysight_B2900) -> None:
    """Non-energizing status read-back plus a snapshot and restore round trip."""
    smu.cmd_reset()
    print(f"  interlock tripped   -> {smu.get_interlock_tripped()}")
    print(f"  status byte / event -> {smu.get_status_byte()} / {smu.get_event_status()}")
    reg = smu.status.Register
    print(
        f"  OPER/MEAS/QUES cond -> {smu.status.get_condition(reg.OPERATION)}/"
        f"{smu.status.get_condition(reg.MEASUREMENT)}/"
        f"{smu.status.get_condition(reg.QUESTIONABLE)}"
    )
    print(f"  QUES enable set     -> {smu.status.set_enable(reg.QUESTIONABLE, 0)}")
    snap = smu.snapshot()
    found = snap["channels"][1]["function"]
    print(f"  snapshot CH1 force  -> {found}")
    smu.ch1.set_source_current(0.01, v_limit=2.0)
    print(f"  mutated CH1 force   -> {smu.ch1.get_source().function}")
    smu.restore(snap)
    print(f"  restored CH1 force  -> {smu.ch1.get_source().function} (== {found})")
    smu.cmd_save_state(4)
    smu.cmd_recall_state(4)
    print("  save/recall slot 4  -> ok")
    smu.cmd_reset()


def _main(argv: list[str] | None = None) -> int:
    """Live self-test against the bench B2900.

    Default: read-mostly identity, channel gating, version, line frequency and
    relay state. It touches no source or sense setting and restores as found.
    Pass ``demo`` for a CH1-only source and measure exercise, ``sweep`` for a
    current staircase and list into a CH2 clamp, ``config`` for a non-energizing
    format and display read-back, or ``state`` for status plus snapshot and
    restore. The ``demo`` and ``sweep`` modes energize briefly; every mode ends
    with the outputs off.
    """
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE [demo] [sweep] [config] [state]")
        return 2
    resource = argv[1]
    modes = {name: name in argv[2:] for name in ("demo", "sweep", "config", "state")}
    smu = Keysight_B2900(resource)
    print(f"connecting to B2900 on {smu.resource_name} ...")
    idn = smu.connect()
    try:
        print(f"  identity: {idn}")
        print(f"  model={smu.model}  firmware={smu.firmware}  channels={smu.num_channels}")
        print(f"  has ch2: {hasattr(smu, 'ch2')}")
        print(
            f"  SCPI version: {smu.get_scpi_version()}  "
            f"line frequency: {smu.get_line_frequency():g} Hz"
        )
        for ch in smu.channels:
            print(f"  {ch.name} output: {'ON' if ch.get_output_state() else 'OFF'}")
        if modes["demo"]:
            print("  --- source and measure demo (CH1 only) ---")
            _demo_source_measure(smu)
        if modes["sweep"]:
            print("  --- sweep demo (CH1 source, CH2 clamp) ---")
            _demo_sweep(smu)
        if modes["config"]:
            print("  --- config demo (format/display/trace/calc) ---")
            _demo_config(smu)
        if modes["state"]:
            print("  --- state demo (status + snapshot/restore) ---")
            _demo_state(smu)
        errors = smu.get_errors()
        print(f"  error queue: {errors or 'clean'}")
    finally:
        smu.disconnect()
        print("disconnected (outputs off).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
