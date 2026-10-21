"""Driver for the Keysight/Agilent 33500 and 33600 series Trueform waveform generators.

Public import path::

    from drivers.keysight_33522a import Keysight_33500

One model-gated class covers both the 335xx and 336xx families; they share a
command set. The bench unit is a 33522A: two channels, 30 MHz, 250 MSa/s, on
LAN, on VXI-11 (``TCPIP0::<address>::inst0::INSTR``).

Common tasks:
  1. Put a 1 kHz 2 Vpp sine on channel 1 and turn it on:
         awg.ch1.set_waveform_sine(1e3, 2.0)
         awg.ch1.set_output_state(True)
  2. Step frequency without disturbing anything else:
         for hz in (1e3, 2e3, 5e3):
             awg.ch1.set_frequency(hz)
  3. Two channels 90 degrees apart:
         awg.ch1.set_waveform_sine(1e4, 1.0)
         awg.ch2.set_waveform_sine(1e4, 1.0)
         awg.ch2.set_phase(90)
         awg.cmd_phase_sync()
  4. Amplitude-modulate, then go back to a plain carrier:
         awg.ch1.set_am(50, rate=1e3)
         awg.ch1.set_modulation_off()
  5. Lock the timebase to the lab 10 MHz reference and confirm it took:
         state = awg.set_reference_clock_external()
         assert state.in_use is awg.Reference.EXTERNAL

Two behaviours are worth knowing before reading further.

The waveform helpers do not use ``APPLy``. ``APPLy`` closes the output relay as
a side effect (bench-verified 2026-08-11: the relay went from open to closed),
which would make every configuration call energize the output. They write the
individual settings instead, and only the ones that differ from what the
instrument already holds, so calling a helper again with one changed argument
moves one parameter and leaves the output otherwise continuous.

The reference clock has three separate facts: what is selected, what is in use,
and whether the instrument may switch by itself. They can disagree. With
automatic selection on, this unit reported INTERNAL while genuinely running on
the scope's external 10 MHz. ``get_reference_clock()`` returns all three.

This is a fast LAN instrument with no inter-command pacing and no post-write
deaf window, unlike the RS-232 33120A and E3646A, so it uses the plain
:class:`common.link.InstrumentLink` (ADR-039). It has a real per-channel output
relay, so :meth:`safe_state` opens both before the link closes; the 33120A has
no relay and leaves its output running.
"""

from __future__ import annotations

import contextlib

from collections.abc import Iterator, Sequence
import re

import numpy as np
import pyvisa as visa

from dataclasses import dataclass

from common.errors import InstrumentError, InstrumentUnresponsiveError
from common.instrument import Instrument
from common.node import Node
from common.scpi import ScpiEnum
from common.state import BurstState, ModulationState, SweepState, WaveformState

# Generous LAN I/O timeout. The link is fast; this only bounds a hung transfer.
_DEFAULT_TIMEOUT_MS = 5000

# model -> capabilities. channels/max_freq/max_srate/arb_points are informational
# (exposed via attributes); the instrument enforces its own limits. Both series
# share the command set, so a new model is a table row, not a new driver.
#   series: "33500" or "33600" (affects only informational caps, not commands).
_MODEL_INFO: dict[str, dict] = {
    # 33500 series (Trueform, 1 uHz resolution).
    "33509B": {
        "channels": 1,
        "max_freq": 20e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33510B": {
        "channels": 2,
        "max_freq": 20e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33511B": {
        "channels": 1,
        "max_freq": 20e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33512B": {
        "channels": 2,
        "max_freq": 20e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33519B": {
        "channels": 1,
        "max_freq": 30e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33520B": {
        "channels": 2,
        "max_freq": 30e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33521A": {
        "channels": 1,
        "max_freq": 30e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33521B": {
        "channels": 1,
        "max_freq": 30e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33522A": {
        "channels": 2,
        "max_freq": 30e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    "33522B": {
        "channels": 2,
        "max_freq": 30e6,
        "max_srate": 250e6,
        "arb_points": 1_000_000,
        "series": "33500",
    },
    # 33600 series (Trueform, higher bandwidth / sample rate).
    "33611A": {
        "channels": 1,
        "max_freq": 80e6,
        "max_srate": 1e9,
        "arb_points": 4_000_000,
        "series": "33600",
    },
    "33612A": {
        "channels": 2,
        "max_freq": 80e6,
        "max_srate": 1e9,
        "arb_points": 4_000_000,
        "series": "33600",
    },
    "33621A": {
        "channels": 1,
        "max_freq": 120e6,
        "max_srate": 1e9,
        "arb_points": 4_000_000,
        "series": "33600",
    },
    "33622A": {
        "channels": 2,
        "max_freq": 120e6,
        "max_srate": 1e9,
        "arb_points": 4_000_000,
        "series": "33600",
    },
}

# 335xx or 336xx, with the alpha suffix. Matches inside a full IDN field.
_IDN_RE = re.compile(r"(33[56]\d\d[A-Z])", re.IGNORECASE)


def _onoff(state: bool) -> str:
    return "ON" if state else "OFF"


def _num(value: float | str) -> str:
    """Format a numeric SCPI argument: a token (MIN/MAX/DEF) or a number."""
    if isinstance(value, str):
        return value
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Enums
#
# Every member's value is the token the instrument RETURNS from a query, not
# the long spelling. The short form is equally valid to write, so one token
# serves both directions and a read-back compares straight against the member.
# That symmetry is what lets the waveform helpers decide whether a write is
# needed at all (design standard 4.5).
# ---------------------------------------------------------------------------


class Function(ScpiEnum):
    """The output shape. This is the FUNCTION menu on the front panel."""

    SINE = "SIN"
    SQUARE = "SQU"
    TRIANGLE = "TRI"
    RAMP = "RAMP"
    PULSE = "PULS"
    PRBS = "PRBS"
    NOISE = "NOIS"
    ARBITRARY = "ARB"
    DC = "DC"


class AmplitudeUnit(ScpiEnum):
    """How an amplitude argument is interpreted.

    DBM is rejected into a high-impedance load and silently becomes VPP.
    """

    VPP = "VPP"
    VRMS = "VRMS"
    DBM = "DBM"


class Load(ScpiEnum):
    """What the far end of the cable terminates into.

    Only the high-impedance case needs a name; every other termination is a
    resistance in ohms. This does not change the fixed 50 ohm source impedance.
    It tells the generator what to assume, so the amplitude it reports matches
    what the load actually sees.
    """

    HIGHZ = "INF"


class Polarity(ScpiEnum):
    """Whether the output is inverted about its offset."""

    NORMAL = "NORM"
    INVERTED = "INV"


class FrequencyMode(ScpiEnum):
    """What drives the output frequency."""

    FIXED = "FIX"
    SWEEP = "SWE"
    LIST = "LIST"


class ModulationShape(ScpiEnum):
    """The internal modulating waveform for AM, FM, PM, PWM and SUM."""

    SINE = "SIN"
    SQUARE = "SQU"
    RAMP = "RAMP"
    NEGATIVE_RAMP = "NRAM"
    TRIANGLE = "TRI"
    NOISE = "NOIS"
    PRBS = "PRBS"
    ARBITRARY = "ARB"


class ModulationSource(ScpiEnum):
    """Where the modulating signal comes from.

    CH1 and CH2 apply to AM, FM, PM, PWM and SUM only. FSK and BPSK take
    INTERNAL or EXTERNAL, and reject a channel.
    """

    INTERNAL = "INT"
    EXTERNAL = "EXT"
    CH1 = "CH1"
    CH2 = "CH2"


class Spacing(ScpiEnum):
    """How a sweep steps from its start frequency to its stop frequency."""

    LINEAR = "LIN"
    LOGARITHMIC = "LOG"


class BurstMode(ScpiEnum):
    """What starts and stops a burst."""

    TRIGGERED = "TRIG"
    GATED = "GAT"


class TriggerSource(ScpiEnum):
    """What advances a sweep, a burst or a frequency list."""

    IMMEDIATE = "IMM"
    EXTERNAL = "EXT"
    TIMER = "TIM"
    BUS = "BUS"


class Slope(ScpiEnum):
    """Which edge a trigger acts on."""

    POSITIVE = "POS"
    NEGATIVE = "NEG"


class ArbFilter(ScpiEnum):
    """The reconstruction filter applied to arbitrary-waveform playback."""

    NORMAL = "NORM"
    STEP = "STEP"
    OFF = "OFF"


class ArbAdvance(ScpiEnum):
    """What advances the arbitrary waveform from one point to the next."""

    SAMPLE_RATE = "SRAT"
    TRIGGER = "TRIG"


class PrbsSequence(ScpiEnum):
    """The pseudo-random binary sequence length.

    The number after PN is the shift-register length in bits. A 33600 accepts
    PN3 through PN32; these members are the set a 33500 accepts.
    """

    PN7 = "PN7"
    PN9 = "PN9"
    PN11 = "PN11"
    PN15 = "PN15"
    PN20 = "PN20"
    PN23 = "PN23"


class PulseHold(ScpiEnum):
    """Which pulse parameter survives a change of period."""

    WIDTH = "WIDT"
    DUTY_CYCLE = "DCYC"


class SyncMode(ScpiEnum):
    """What the Sync connector marks."""

    NORMAL = "NORM"
    CARRIER = "CARR"
    MARKER = "MARK"


class Track(ScpiEnum):
    """Whether channel 2 copies channel 1."""

    OFF = "OFF"
    ON = "ON"
    INVERTED = "INV"


class Combine(ScpiEnum):
    """Which channel is summed onto another channel's connector."""

    NONE = "NONE"
    CH1 = "CH1"
    CH2 = "CH2"


class AngleUnit(ScpiEnum):
    """The unit every phase argument and return is expressed in."""

    DEGREE = "DEG"
    RADIAN = "RAD"
    SECOND = "SEC"


class Reference(ScpiEnum):
    """Which 10 MHz timebase the instrument locks to."""

    INTERNAL = "INT"
    EXTERNAL = "EXT"


class ByteOrder(ScpiEnum):
    """The byte order of a binary block transfer."""

    NORMAL = "NORM"
    SWAPPED = "SWAP"


class ChannelId(ScpiEnum):
    """A channel named as an argument, for Sync, trigger-out and the display."""

    CH1 = "CH1"
    CH2 = "CH2"


def _token(value, enum_cls: type[ScpiEnum], label: str) -> str:
    """Return the SCPI token for an enum member, a member name, or a token string.

    Accepts any of the three case-insensitively, so a caller may pass
    ``Function.SINE``, ``"SINE"`` or ``"sin"``. A bad value names every valid
    option in the message rather than failing at the instrument.
    """
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    for member in enum_cls:
        if text in (member.name, str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")


def _member(value, enum_cls: type[ScpiEnum], label: str):
    """Return the enum member for an enum member, a member name, or a token."""
    return enum_cls(_token(value, enum_cls, label))


# FUNCtion accepts these; the query returns the short token (left column value).
_FUNCTION = {
    "SIN": "SIN",
    "SINE": "SIN",
    "SINUSOID": "SIN",
    "SQU": "SQU",
    "SQUARE": "SQU",
    "TRI": "TRI",
    "TRIANGLE": "TRI",
    "RAMP": "RAMP",
    "PULS": "PULS",
    "PULSE": "PULS",
    "PRBS": "PRBS",
    "NOIS": "NOIS",
    "NOISE": "NOIS",
    "ARB": "ARB",
    "ARBITRARY": "ARB",
    "DC": "DC",
}

_UNIT = {"VPP": "VPP", "VRMS": "VRMS", "DBM": "DBM"}

_FREQ_MODE = {
    "CW": "CW",
    "FIX": "FIXed",
    "FIXED": "FIXed",
    "SWE": "SWEep",
    "SWEEP": "SWEep",
    "LIST": "LIST",
}

_POLARITY = {"NORM": "NORMal", "NORMAL": "NORMal", "INV": "INVerted", "INVERTED": "INVerted"}

_OUTPUT_MODE = {"NORM": "NORMal", "NORMAL": "NORMal", "GAT": "GATed", "GATED": "GATed"}

# OUTP:LOAD tokens (numbers pass through). HIGHZ is the friendly name for INF.
_LOAD_TOKENS = {
    "INF": "INFinity",
    "INFINITY": "INFinity",
    "HIGHZ": "INFinity",
    "MIN": "MINimum",
    "MINIMUM": "MINimum",
    "MAX": "MAXimum",
    "MAXIMUM": "MAXimum",
    "DEF": "DEFault",
    "DEFAULT": "DEFault",
}

# The instrument returns 9.9E37 for a high-impedance (INF) load.
_HIGHZ_SENTINEL = 9.9e37

# ---------------------------------------------------------------------------
# Minimal-write precision.
#
# A waveform helper writes a value only when the instrument is not already
# there, so it must decide whether the value it holds and the value asked for
# are the same. That decision needs the instrument's own precision.
#
# This instrument rounds each quantity to a fixed number of SIGNIFICANT DIGITS
# when it stores it, and then reports the rounded value at full width on the
# wire: writing 1234.5678 Hz and reading back gives +1.23457000000000E+03.
# So the test is not "are these two floats close" but "do they agree at the
# precision the instrument keeps". Formatting both sides to that many digits
# reproduces the instrument's own rounding exactly, which means a repeat call
# writes nothing at all, while any change the hardware can represent still
# writes. An absolute tolerance cannot do both: it either rewrites constantly
# on values the instrument rounds, or silently swallows a small real change.
#
# Bench-measured 2026-08-11 on the bench 33522A (fw 3.05-1.19-2.00-52-00) with
# projects/awg_probe.py, which writes a spread of awkward values per quantity
# and reports the highest precision at which the write and the read-back still
# agree. See ADR-050. A guessed absolute tolerance was 30x too tight and would
# have rewritten every parameter on every call.
# ---------------------------------------------------------------------------
_SIG_FREQUENCY = 6  # frequency, bit rate, sample rate, noise bandwidth
_SIG_VOLT = 5  # amplitude, offset, high and low level
_SIG_PERCENT = 4  # duty cycle, ramp symmetry
_SIG_DEGREE = 5  # phase
_SIG_WIDTH = 6  # pulse width
_SIG_EDGE_TIME = 4  # pulse and PRBS edge time - coarser than the width it belongs to


def _same(present: float, wanted: float, digits: int) -> bool:
    """Whether the instrument already holds ``wanted``, at its own precision.

    Compares the two values formatted to ``digits`` significant figures, which
    is how the instrument stores them. Equal means no write is needed.
    """
    return f"{present:.{digits}g}" == f"{wanted:.{digits}g}"


@dataclass(frozen=True)
class VoltageLimits:
    """The clamp on how far the output may swing.

    The limits keep their values when the clamp is switched off, so ``low`` and
    ``high`` stay meaningful while ``enabled`` is False.
    """

    low: float
    """Lowest output voltage allowed, in volts."""

    high: float
    """Highest output voltage allowed, in volts."""

    enabled: bool
    """Whether the clamp is in force."""


# Internal modulating-waveform shapes (AM/FM/PM/PWM/SUM internal function).
_MOD_FUNCTION = {
    "SIN": "SIN",
    "SINE": "SIN",
    "SINUSOID": "SIN",
    "SQU": "SQU",
    "SQUARE": "SQU",
    "RAMP": "RAMP",
    "NRAM": "NRAM",
    "NRAMP": "NRAM",
    "TRI": "TRI",
    "TRIANGLE": "TRI",
    "NOIS": "NOIS",
    "NOISE": "NOIS",
    "PRBS": "PRBS",
    "ARB": "ARB",
    "ARBITRARY": "ARB",
}

# Modulation source. AM/FM/PM/PWM/SUM allow CH1/CH2; FSK/BPSK only INT/EXT.
_MOD_SOURCE = {
    "INT": "INTernal",
    "INTERNAL": "INTernal",
    "EXT": "EXTernal",
    "EXTERNAL": "EXTernal",
    "CH1": "CH1",
    "CH2": "CH2",
}
_MOD_SOURCE_INTEXT = {
    "INT": "INTernal",
    "INTERNAL": "INTernal",
    "EXT": "EXTernal",
    "EXTERNAL": "EXTernal",
}

_SWEEP_SPACING = {
    "LIN": "LINear",
    "LINEAR": "LINear",
    "LOG": "LOGarithmic",
    "LOGARITHMIC": "LOGarithmic",
}
_BURST_MODE = {"TRIG": "TRIGgered", "TRIGGERED": "TRIGgered", "GAT": "GATed", "GATED": "GATed"}
_TRIG_SOURCE = {
    "IMM": "IMMediate",
    "IMMEDIATE": "IMMediate",
    "EXT": "EXTernal",
    "EXTERNAL": "EXTernal",
    "TIM": "TIMer",
    "TIMER": "TIMer",
    "BUS": "BUS",
}
_ARB_FILTER = {"NORM": "NORMal", "NORMAL": "NORMal", "STEP": "STEP", "OFF": "OFF"}
_ARB_ADVANCE = {"TRIG": "TRIGger", "TRIGGER": "TRIGger", "SRAT": "SRATe", "SRATE": "SRATe"}
_ARB_DAC_MAX = 32767

# The 33500/33600 error queue holds 20 entries before it overflows.
_ERROR_QUEUE_DEPTH = 20


class _ModBase(Node):
    """Base for a per-channel modulation subsystem (``SOURce<n>:<PREFIX>:...``).

    Carries the channel's SOURce prefix and the subsystem keyword (e.g. ``AM``),
    plus the state/source controls common to every modulation type. Enable
    modulation LAST (after configuring the other parameters): only one modulation
    mode may be active at once, and enabling it turns off sweep/burst.
    """

    def __init__(self, link, src: str, prefix: str, name: str, allow_channel_source: bool = True):
        super().__init__(link, name=name)
        self._p = f"{src}{prefix}"
        self._allow_channel_source = allow_channel_source

    def set_state(self, state: bool) -> bool:
        """Turn this modulation on or off; configure the other parameters first."""
        self.write(f"{self._p}:STAT {_onoff(state)}")
        return self.get_state()

    def get_state(self) -> bool:
        """Whether this modulation is on."""
        return self.query_bool(f"{self._p}:STAT?")

    def set_source(self, source: ModulationSource | str) -> ModulationSource:
        """Choose where the modulating signal comes from; a :class:`ModulationSource`.

        Enum arg: <awg>.ModulationSource.<TAB>

        FSK and BPSK reject CH1 and CH2, so this refuses them there rather than
        letting the instrument queue an error.
        """
        member = _member(source, ModulationSource, "source")
        if not self._allow_channel_source and member in (
            ModulationSource.CH1,
            ModulationSource.CH2,
        ):
            raise ValueError(
                f"{self.name} accepts only INTERNAL or EXTERNAL, not {member.name}. "
                "A channel may modulate AM, FM, PM, PWM or SUM."
            )
        self.write(f"{self._p}:SOUR {member.value}")
        return self.get_source()

    def get_source(self) -> ModulationSource:
        """Where the modulating signal comes from."""
        return _member(self.query(f"{self._p}:SOUR?").strip(), ModulationSource, "source")


class _InternalMod(_ModBase):
    """A modulation that has its own internal generator: AM, FM, PM, PWM and SUM.

    Adds the internal modulating frequency and the internal modulating shape.
    """

    def set_internal_frequency(self, frequency: float | str) -> float:
        """Set the internal modulating frequency in Hz."""
        self.write(f"{self._p}:INT:FREQ {_num(frequency)}")
        return self.get_internal_frequency()

    def get_internal_frequency(self, limit: str | None = None) -> float:
        """The internal modulating frequency in Hz; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:INT:FREQ?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_internal_function(self, function: ModulationShape | str) -> ModulationShape:
        """Set the internal modulating shape; a :class:`ModulationShape`.

        Enum arg: <awg>.ModulationShape.<TAB>
        """
        self.write(f"{self._p}:INT:FUNC {_token(function, ModulationShape, 'function')}")
        return self.get_internal_function()

    def get_internal_function(self) -> ModulationShape:
        """The internal modulating shape."""
        return _member(self.query(f"{self._p}:INT:FUNC?").strip(), ModulationShape, "function")


class _AM(_InternalMod):
    """Amplitude modulation (AM)."""

    def set_depth(self, percent: float | str) -> float:
        """Set the AM depth in percent, 0-120."""
        self.write(f"{self._p}:DEPT {_num(percent)}")
        return self.get_depth()

    def get_depth(self, limit: str | None = None) -> float:
        """Query the AM depth in percent; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:DEPT?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_dssc(self, state: bool) -> bool:
        """Enable/disable double-sideband suppressed-carrier AM."""
        self.write(f"{self._p}:DSSC {_onoff(state)}")
        return self.get_dssc()

    def get_dssc(self) -> bool:
        """Query the DSSC state."""
        return self.query_bool(f"{self._p}:DSSC?")


class _FM(_InternalMod):
    """Frequency modulation (FM)."""

    def set_deviation(self, hertz: float | str) -> float:
        """Set the peak frequency deviation in Hz."""
        self.write(f"{self._p}:DEV {_num(hertz)}")
        return self.get_deviation()

    def get_deviation(self, limit: str | None = None) -> float:
        """Query the peak frequency deviation in Hz; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:DEV?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)


class _PM(_InternalMod):
    """Phase modulation (PM)."""

    def set_deviation(self, degrees: float | str) -> float:
        """Set the peak phase deviation in degrees, 0-360."""
        self.write(f"{self._p}:DEV {_num(degrees)}")
        return self.get_deviation()

    def get_deviation(self, limit: str | None = None) -> float:
        """Query the peak phase deviation in degrees; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:DEV?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)


class _PWM(_InternalMod):
    """Pulse-width modulation (PWM); valid only when the function is PULSe."""

    def set_deviation(self, seconds: float | str) -> float:
        """Set the pulse-width deviation in seconds."""
        self.write(f"{self._p}:DEV {_num(seconds)}")
        return self.get_deviation()

    def get_deviation(self, limit: str | None = None) -> float:
        """Query the pulse-width deviation in seconds; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:DEV?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_deviation_duty_cycle(self, percent: float | str) -> float:
        """Set the duty-cycle deviation in percent, 0-50."""
        self.write(f"{self._p}:DEV:DCYC {_num(percent)}")
        return self.get_deviation_duty_cycle()

    def get_deviation_duty_cycle(self, limit: str | None = None) -> float:
        """Query the duty-cycle deviation in percent; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:DEV:DCYC?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)


class _SUM(_InternalMod):
    """Additive modulation (SUM): sums a second waveform onto the carrier."""

    def set_amplitude(self, percent: float | str) -> float:
        """Set the SUM signal amplitude in percent of carrier, 0-100."""
        self.write(f"{self._p}:AMPL {_num(percent)}")
        return self.get_amplitude()

    def get_amplitude(self, limit: str | None = None) -> float:
        """Query the SUM signal amplitude in percent; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:AMPL?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)


class _FSK(_ModBase):
    """Frequency-shift keying (FSK): source is INTERNAL/EXTERNAL only."""

    def set_hop_frequency(self, frequency: float | str) -> float:
        """Set the FSK alternate ("hop") frequency in Hz."""
        self.write(f"{self._p}:FREQ {_num(frequency)}")
        return self.get_hop_frequency()

    def get_hop_frequency(self, limit: str | None = None) -> float:
        """Query the FSK hop frequency in Hz; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:FREQ?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_rate(self, rate: float | str) -> float:
        """Set the internal FSK shift rate in Hz, 0.125 mHz-1 MHz."""
        self.write(f"{self._p}:INT:RATE {_num(rate)}")
        return self.get_rate()

    def get_rate(self, limit: str | None = None) -> float:
        """Query the internal FSK shift rate in Hz; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:INT:RATE?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)


class _BPSK(_ModBase):
    """Binary phase-shift keying (BPSK): source is INTERNAL/EXTERNAL only."""

    def set_phase(self, degrees: float | str) -> float:
        """Set the BPSK phase-shift angle in degrees, 0-360."""
        self.write(f"{self._p}:PHAS {_num(degrees)}")
        return self.get_phase()

    def get_phase(self, limit: str | None = None) -> float:
        """Query the BPSK phase-shift angle in degrees; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:PHAS?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_rate(self, rate: float | str) -> float:
        """Set the internal BPSK shift rate in Hz, 1 mHz-1 MHz."""
        self.write(f"{self._p}:INT:RATE {_num(rate)}")
        return self.get_rate()

    def get_rate(self, limit: str | None = None) -> float:
        """Query the internal BPSK shift rate in Hz; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._p}:INT:RATE?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)


class _Sweep(Node):
    """Frequency sweep for one channel (SWEep + FREQuency start/stop/center/span)."""

    def __init__(self, link, src: str, name: str):
        super().__init__(link, name=name)
        self._src = src

    def set_state(self, state: bool) -> bool:
        """Enable/disable the sweep. Configure the span/time first."""
        self.write(f"{self._src}SWE:STAT {_onoff(state)}")
        return self.get_state()

    def get_state(self) -> bool:
        """Query whether the sweep is enabled."""
        return self.query_bool(f"{self._src}SWE:STAT?")

    def set_start_frequency(self, hertz: float | str) -> float:
        """Set the sweep start frequency in Hz."""
        self.write(f"{self._src}FREQ:STAR {_num(hertz)}")
        return self.get_start_frequency()

    def get_start_frequency(self, limit: str | None = None) -> float:
        """Query the sweep start frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}FREQ:STAR?" + (f" {limit}" if limit else ""))

    def set_stop_frequency(self, hertz: float | str) -> float:
        """Set the sweep stop frequency in Hz."""
        self.write(f"{self._src}FREQ:STOP {_num(hertz)}")
        return self.get_stop_frequency()

    def get_stop_frequency(self, limit: str | None = None) -> float:
        """Query the sweep stop frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}FREQ:STOP?" + (f" {limit}" if limit else ""))

    def set_center_frequency(self, hertz: float | str) -> float:
        """Set the sweep center frequency in Hz."""
        self.write(f"{self._src}FREQ:CENT {_num(hertz)}")
        return self.get_center_frequency()

    def get_center_frequency(self, limit: str | None = None) -> float:
        """Query the sweep center frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}FREQ:CENT?" + (f" {limit}" if limit else ""))

    def set_span(self, hertz: float | str) -> float:
        """Set the sweep frequency span in Hz; negative sweeps down."""
        self.write(f"{self._src}FREQ:SPAN {_num(hertz)}")
        return self.get_span()

    def get_span(self, limit: str | None = None) -> float:
        """Query the sweep frequency span in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}FREQ:SPAN?" + (f" {limit}" if limit else ""))

    def set_time(self, seconds: float | str) -> float:
        """Set the sweep time (start->stop) in seconds."""
        self.write(f"{self._src}SWE:TIME {_num(seconds)}")
        return self.get_time()

    def get_time(self, limit: str | None = None) -> float:
        """Query the sweep time in seconds; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}SWE:TIME?" + (f" {limit}" if limit else ""))

    def set_hold_time(self, seconds: float | str) -> float:
        """Set the sweep hold time at the stop frequency in seconds."""
        self.write(f"{self._src}SWE:HTIM {_num(seconds)}")
        return self.get_hold_time()

    def get_hold_time(self, limit: str | None = None) -> float:
        """Query the sweep hold time in seconds; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}SWE:HTIM?" + (f" {limit}" if limit else ""))

    def set_return_time(self, seconds: float | str) -> float:
        """Set the sweep return (retrace) time in seconds."""
        self.write(f"{self._src}SWE:RTIM {_num(seconds)}")
        return self.get_return_time()

    def get_return_time(self, limit: str | None = None) -> float:
        """Query the sweep return time in seconds; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}SWE:RTIM?" + (f" {limit}" if limit else ""))

    def set_spacing(self, spacing: "Spacing | str") -> "Spacing":
        """Set how the sweep steps from start to stop; a :class:`Spacing`.

        Enum arg: <awg>.Spacing.<TAB>
        """
        self.write(f"{self._src}SWE:SPAC {_token(spacing, Spacing, 'spacing')}")
        return self.get_spacing()

    def get_spacing(self) -> "Spacing":
        """How the sweep steps from its start frequency to its stop frequency."""
        return _member(self.query(f"{self._src}SWE:SPAC?").strip(), Spacing, "spacing")

    def set_marker_frequency(self, hertz: float | str) -> float:
        """Set the sweep marker frequency in Hz; SYNC goes low here."""
        self.write(f"{self._src}MARK:FREQ {_num(hertz)}")
        return self.get_marker_frequency()

    def get_marker_frequency(self, limit: str | None = None) -> float:
        """Query the sweep marker frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}MARK:FREQ?" + (f" {limit}" if limit else ""))


class _Burst(Node):
    """Burst mode for one channel (BURSt subsystem + burst marker cycle)."""

    def __init__(self, link, src: str, name: str):
        super().__init__(link, name=name)
        self._src = src

    def set_state(self, state: bool) -> bool:
        """Enable/disable burst. Configure the other params first."""
        self.write(f"{self._src}BURS:STAT {_onoff(state)}")
        return self.get_state()

    def get_state(self) -> bool:
        """Query whether burst is enabled."""
        return self.query_bool(f"{self._src}BURS:STAT?")

    def set_mode(self, mode: "BurstMode | str") -> "BurstMode":
        """Set what starts and stops a burst; a :class:`BurstMode`.

        Enum arg: <awg>.BurstMode.<TAB>
        """
        self.write(f"{self._src}BURS:MODE {_token(mode, BurstMode, 'mode')}")
        return self.get_mode()

    def get_mode(self) -> "BurstMode":
        """What starts and stops a burst."""
        return _member(self.query(f"{self._src}BURS:MODE?").strip(), BurstMode, "mode")

    def set_cycles(self, cycles: int | str) -> float:
        """Set cycles per burst, 1 to 1e8; accepts INF/MIN/MAX."""
        if isinstance(cycles, str):
            token = {
                "INF": "INFinity",
                "INFINITY": "INFinity",
                "MIN": "MINimum",
                "MAX": "MAXimum",
            }.get(cycles.upper(), cycles)
            self.write(f"{self._src}BURS:NCYC {token}")
        else:
            self.write(f"{self._src}BURS:NCYC {int(cycles)}")
        return self.get_cycles()

    def get_cycles(self, limit: str | None = None) -> float:
        """Query cycles per burst; ``limit`` in {MIN, MAX}. (INFinity reads as 9.9E37.)"""
        return self.query_float(f"{self._src}BURS:NCYC?" + (f" {limit}" if limit else ""))

    def set_period(self, seconds: float | str) -> float:
        """Set the internal burst period in seconds, IMM trigger only."""
        self.write(f"{self._src}BURS:INT:PER {_num(seconds)}")
        return self.get_period()

    def get_period(self, limit: str | None = None) -> float:
        """Query the internal burst period in seconds; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}BURS:INT:PER?" + (f" {limit}" if limit else ""))

    def set_phase(self, degrees: float | str) -> float:
        """Set the starting burst phase in the current angle unit."""
        self.write(f"{self._src}BURS:PHAS {_num(degrees)}")
        return self.get_phase()

    def get_phase(self, limit: str | None = None) -> float:
        """Query the starting burst phase; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}BURS:PHAS?" + (f" {limit}" if limit else ""))

    def set_gate_polarity(self, polarity: "Polarity | str") -> "Polarity":
        """Set which gate level enables the burst; a :class:`Polarity`.

        Enum arg: <awg>.Polarity.<TAB>
        """
        self.write(f"{self._src}BURS:GATE:POL {_token(polarity, Polarity, 'polarity')}")
        return self.get_gate_polarity()

    def get_gate_polarity(self) -> "Polarity":
        """Which level of the external gate enables the burst."""
        return _member(self.query(f"{self._src}BURS:GATE:POL?").strip(), Polarity, "polarity")

    def set_marker_cycle(self, cycle: int | str) -> float:
        """Set the burst marker cycle where SYNC goes low, 2..NCYC+1."""
        self.write(f"{self._src}MARK:CYC {_num(cycle) if isinstance(cycle, str) else int(cycle)}")
        return self.get_marker_cycle()

    def get_marker_cycle(self, limit: str | None = None) -> float:
        """Query the burst marker cycle; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}MARK:CYC?" + (f" {limit}" if limit else ""))


class _FreqList(Node):
    """Frequency-list mode for one channel (LIST subsystem).

    Activate with the channel's :meth:`_Channel.set_frequency_mode` ("LIST"); the
    list steps at :meth:`set_dwell` under IMMediate trigger.
    """

    def __init__(self, link, src: str, name: str):
        super().__init__(link, name=name)
        self._src = src

    def set_frequencies(self, frequencies: Sequence[float]) -> int:
        """Set the frequency list, 1 to 128 values in Hz. Returns the count."""
        values = [float(f) for f in frequencies]
        if not 1 <= len(values) <= 128:
            raise ValueError(f"list must have 1-128 frequencies, got {len(values)}")
        payload = ",".join(f"{v:g}" for v in values)
        self.write(f"{self._src}LIST:FREQ {payload}")
        return self.get_points()

    def get_frequencies(self) -> list[float]:
        """Query the frequency list as a list of Hz values."""
        resp = self.query(f"{self._src}LIST:FREQ?").strip()
        return [float(x) for x in resp.split(",") if x.strip()]

    def get_points(self) -> int:
        """Query the number of frequencies in the list."""
        return int(round(self.query_float(f"{self._src}LIST:FREQ:POIN?")))

    def set_dwell(self, seconds: float | str) -> float:
        """Set the per-frequency dwell time in seconds."""
        self.write(f"{self._src}LIST:DWEL {_num(seconds)}")
        return self.get_dwell()

    def get_dwell(self, limit: str | None = None) -> float:
        """Query the dwell time in seconds; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}LIST:DWEL?" + (f" {limit}" if limit else ""))


class _Trigger(Node):
    """Per-channel trigger system (TRIGger<n> + INITiate<n>)."""

    def __init__(self, link, n: int, name: str):
        super().__init__(link, name=name)
        self.n = n

    def set_source(self, source: "TriggerSource | str") -> "TriggerSource":
        """Set what advances a sweep, a burst or a list; a :class:`TriggerSource`.

        Enum arg: <awg>.TriggerSource.<TAB>
        """
        self.write(f"TRIG{self.n}:SOUR {_token(source, TriggerSource, 'source')}")
        return self.get_source()

    def get_source(self) -> "TriggerSource":
        """What advances a sweep, a burst or a list."""
        return _member(self.query(f"TRIG{self.n}:SOUR?").strip(), TriggerSource, "source")

    def set_slope(self, slope: "Slope | str") -> "Slope":
        """Set which edge of the external trigger input fires; a :class:`Slope`.

        Enum arg: <awg>.Slope.<TAB>
        """
        self.write(f"TRIG{self.n}:SLOP {_token(slope, Slope, 'slope')}")
        return self.get_slope()

    def get_slope(self) -> "Slope":
        """Which edge of the external trigger input fires."""
        return _member(self.query(f"TRIG{self.n}:SLOP?").strip(), Slope, "slope")

    def set_delay(self, seconds: float | str) -> float:
        """Set the trigger delay in seconds, 0 to 1000 s."""
        self.write(f"TRIG{self.n}:DEL {_num(seconds)}")
        return self.get_delay()

    def get_delay(self, limit: str | None = None) -> float:
        """Query the trigger delay in seconds; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"TRIG{self.n}:DEL?" + (f" {limit}" if limit else ""))

    def set_count(self, count: int | str) -> float:
        """Set how many triggers to accept before the channel idles again.

        Applies only when continuous re-arming is off.
        """
        self.write(f"TRIG{self.n}:COUN {_num(count) if isinstance(count, str) else int(count)}")
        return self.get_count()

    def get_count(self, limit: str | None = None) -> float:
        """Query the trigger count; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"TRIG{self.n}:COUN?" + (f" {limit}" if limit else ""))

    def set_timer(self, seconds: float | str) -> float:
        """Set the trigger timer in seconds, used when source is TIMER."""
        self.write(f"TRIG{self.n}:TIM {_num(seconds)}")
        return self.get_timer()

    def get_timer(self, limit: str | None = None) -> float:
        """Query the trigger timer in seconds; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"TRIG{self.n}:TIM?" + (f" {limit}" if limit else ""))

    def set_level(self, volts: float | str) -> float:
        """Set the trigger output level / input threshold in volts."""
        self.write(f"TRIG{self.n}:LEV {_num(volts)}")
        return self.get_level()

    def get_level(self, limit: str | None = None) -> float:
        """Query the trigger level in volts; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"TRIG{self.n}:LEV?" + (f" {limit}" if limit else ""))

    def set_continuous_state(self, state: bool) -> bool:
        """Set the continuous-initiate state for this channel.

        ON = always re-arm to wait-for-trigger; OFF = idle until :meth:`cmd_initiate`.
        """
        self.write(f"INIT{self.n}:CONT {_onoff(state)}")
        return self.get_continuous_state()

    def get_continuous_state(self) -> bool:
        """Query the continuous-initiate state."""
        return self.query_bool(f"INIT{self.n}:CONT?")

    def cmd_initiate(self) -> None:
        """Arm this channel so the next trigger runs its sweep, burst or list.

        Only has an effect when continuous re-arming is off.
        """
        self.write(f"INIT{self.n}:IMM")


class _Display(Node):
    """Front-panel display control (DISPlay subsystem)."""

    def set_state(self, state: bool) -> bool:
        """Enable/disable the front-panel display. Disabling speeds up I/O."""
        self.write(f"DISP {_onoff(state)}")
        return self.get_state()

    def get_state(self) -> bool:
        """Query whether the display is enabled."""
        return self.query_bool("DISP?")

    def set_text(self, text: str) -> str:
        """Show a message on the display, up to 40 chars. Returns read-back."""
        self.write(f'DISP:TEXT "{text}"')
        return self.get_text()

    def get_text(self) -> str:
        """Query the displayed message."""
        return self.query("DISP:TEXT?").strip().strip('"')

    def cmd_clear_text(self) -> None:
        """Clear the display message."""
        self.write("DISP:TEXT:CLE")

    def set_focus(self, channel: "ChannelId | str | int") -> "ChannelId":
        """Bring a channel to the front of a two-channel display.

        Enum arg: <awg>.ChannelId.<TAB>
        """
        self.write(f"DISP:FOC {_ch_token(channel)}")
        return _member(self.query("DISP:FOC?").strip(), ChannelId, "channel")


class _Memory(Node):
    """Instrument-state storage (*SAV/*RCL + MEMory:STATe)."""

    def cmd_save_state(self, location: int) -> None:
        """Save the present state to storage location 0-4."""
        self.link.write(f"*SAV {self._check_location(location)}")
        self.link.query("*OPC?")

    def cmd_recall_state(self, location: int) -> None:
        """Recall a stored state from location 0-4."""
        self.link.write(f"*RCL {self._check_location(location)}")
        self.link.query("*OPC?")

    def cmd_delete_state(self, location: int) -> None:
        """Delete a stored state, resetting its name to default."""
        self.write(f"MEM:STAT:DEL {self._check_location(location)}")

    def set_name(self, location: int, name: str) -> str:
        """Name a storage location, <=12 chars starting with a letter."""
        self.write(f"MEM:STAT:NAME {self._check_location(location)},{name}")
        return self.get_name(location)

    def get_name(self, location: int) -> str:
        """Query a storage location's name."""
        return self.query(f"MEM:STAT:NAME? {self._check_location(location)}").strip().strip('"')

    def get_catalog(self) -> list[str]:
        """List the names of locations 0-4."""
        resp = self.query("MEM:STAT:CAT?").strip()
        return [s.strip().strip('"') for s in resp.split(",")]

    def get_state_valid(self, location: int) -> bool:
        """Return True if a valid state is stored at ``location``."""
        return self.query_bool(f"MEM:STAT:VAL? {self._check_location(location)}")

    def get_state_count(self) -> int:
        """Return the number of state-storage locations, always 5."""
        return self.query_int("MEM:NST?")

    def set_auto_recall_state(self, state: bool) -> bool:
        """Enable/disable auto-recall of location 0 at power-on."""
        self.write(f"MEM:STAT:REC:AUTO {_onoff(state)}")
        return self.query_bool("MEM:STAT:REC:AUTO?")

    @staticmethod
    def _check_location(location: int) -> int:
        if location not in (0, 1, 2, 3, 4):
            raise ValueError(f"state location must be 0-4, got {location!r}")
        return location


class _Status(Node):
    """Status/event registers (IEEE-488.2 common + STATus operation/questionable)."""

    def cmd_clear(self) -> None:
        """Clear the status byte and all event registers."""
        self.link.write("*CLS")

    def cmd_preset(self) -> None:
        """Clear the operation and questionable enable registers."""
        self.write("STAT:PRES")

    def get_event_status(self) -> int:
        """Query the Standard Event register."""
        return self.query_int("*ESR?")

    def set_event_status_enable(self, value: int) -> int:
        """Set the Standard Event enable mask."""
        self.link.write(f"*ESE {int(value)}")
        return self.get_event_status_enable()

    def get_event_status_enable(self) -> int:
        """The mask deciding which standard events reach the status byte."""
        return self.query_int("*ESE?")

    def get_status_byte(self) -> int:
        """Query the Status Byte summary register."""
        return self.query_int("*STB?")

    def set_service_request_enable(self, value: int) -> int:
        """Set the Service Request enable mask."""
        self.link.write(f"*SRE {int(value)}")
        return self.get_service_request_enable()

    def get_service_request_enable(self) -> int:
        """The mask deciding which status-byte bits request service."""
        return self.query_int("*SRE?")

    def set_power_on_status_clear(self, state: bool) -> bool:
        """Set power-on status clear."""
        self.link.write(f"*PSC {1 if state else 0}")
        return self.query_bool("*PSC?")

    def get_operation_condition(self) -> int:
        """Query the Operation condition register."""
        return self.query_int("STAT:OPER:COND?")

    def get_operation_event(self) -> int:
        """Query and clear the Operation event register."""
        return self.query_int("STAT:OPER?")

    def set_operation_enable(self, value: int) -> int:
        """Set the Operation enable mask."""
        self.write(f"STAT:OPER:ENAB {int(value)}")
        return self.query_int("STAT:OPER:ENAB?")

    def get_questionable_condition(self) -> int:
        """Query the Questionable condition register."""
        return self.query_int("STAT:QUES:COND?")

    def get_questionable_event(self) -> int:
        """Query and clear the Questionable event register."""
        return self.query_int("STAT:QUES?")

    def set_questionable_enable(self, value: int) -> int:
        """Set the Questionable enable mask."""
        self.write(f"STAT:QUES:ENAB {int(value)}")
        return self.query_int("STAT:QUES:ENAB?")


@dataclass(frozen=True)
class ReferenceState:
    """Which 10 MHz timebase the instrument is locked to, and how it got there.

    ``selected`` and ``in_use`` can differ. With ``automatic`` on, the
    instrument takes an external reference whenever one is present, whatever is
    selected: the bench 33522A reported INTERNAL while genuinely running on the
    scope's 10 MHz (bench-observed 2026-08-11). Anything that depends on a known
    timebase must read ``in_use``.
    """

    selected: "Reference"
    """The source that has been chosen."""

    in_use: "Reference"
    """The source actually driving the timebase now."""

    automatic: bool
    """Whether the instrument may switch sources by itself."""


class _ReferenceClock(Node):
    """The 10 MHz timebase source (ROSCillator subsystem)."""

    def _set_source(self, source: "Reference | str") -> "Reference":
        """Write the selected reference source (ROSC:SOUR), leaving AUTO alone."""
        self.write(f"ROSC:SOUR {_token(source, Reference, 'source')}")
        return self._get_source()

    def _get_source(self) -> "Reference":
        """Query the selected reference source (ROSC:SOUR?)."""
        return _member(self.query("ROSC:SOUR?").strip(), Reference, "source")

    def _get_in_use(self) -> "Reference":
        """Query the reference source actually in use (ROSC:SOUR:CURR?)."""
        return _member(self.query("ROSC:SOUR:CURR?").strip(), Reference, "source")

    def _set_auto_state(self, state: bool) -> bool:
        """Write automatic reference selection (ROSC:SOUR:AUTO)."""
        self.write(f"ROSC:SOUR:AUTO {_onoff(state)}")
        return self._get_auto_state()

    def _get_auto_state(self) -> bool:
        """Query automatic reference selection (ROSC:SOUR:AUTO?)."""
        return self.query_bool("ROSC:SOUR:AUTO?")


class _Calibration(Node):
    """Calibration subsystem - QUERY ONLY (never performs or unsecures a cal)."""

    def get_count(self) -> int:
        """Query the calibration count."""
        return self.query_int("CAL:COUN?")

    def get_string(self) -> str:
        """Query the stored calibration message."""
        return self.query("CAL:STR?").strip().strip('"')

    def get_secure_state(self) -> bool:
        """Query whether calibration is secured."""
        return self.query_bool("CAL:SEC:STAT?")


class _Arb(Node):
    """Arbitrary-waveform engine for one channel (DATA + FUNCtion:ARBitrary).

    Upload points to volatile memory (:meth:`set_data_dac` for 16-bit DAC codes via
    an IEEE binary block, or :meth:`set_data_floats` for a CSV of -1..1 values),
    :meth:`set_selected` the waveform, then :meth:`output` it. Playback speed is set by
    :meth:`set_sample_rate` or :meth:`set_frequency` (frequency = sample_rate /
    number_of_points). Names are unquoted, up to 12 characters.
    """

    def __init__(self, link, src: str, n: int, name: str):
        super().__init__(link, name=name)
        self._src = src
        self.n = n

    def set_data_dac(self, name: str, codes: Sequence[int], clear: bool = True) -> int:
        """Upload 16-bit DAC codes to volatile memory.

        ``codes`` are integers in [-32767, 32767] (auto-clipped/rounded); full
        scale maps to +/- the set amplitude/2. Sent big-endian (FORM:BORD NORM).
        With ``clear`` the channel's volatile memory is cleared first (a name that
        already exists would otherwise error). Returns the point count.
        """
        arr = np.asarray(codes)
        n = int(arr.size)
        if n < 8:
            raise ValueError(f"arb needs at least 8 points, got {n}")
        clipped = np.clip(np.round(arr), -_ARB_DAC_MAX, _ARB_DAC_MAX).astype(">i2")
        raw = clipped.tobytes()
        if clear:
            self.cmd_clear()
        self._set_byte_order(ByteOrder.NORMAL)
        header = f"#{len(str(len(raw)))}{len(raw)}".encode("ascii")
        self.link.visa.write_raw(
            f"{self._src}DATA:ARB{self.n}:DAC {name},".encode("ascii") + header + raw + b"\n"
        )
        self.link.query("*OPC?")
        return n

    def set_data_floats(self, name: str, data: Sequence[float], clear: bool = True) -> int:
        """Upload a waveform as a CSV of floats.

        ``data`` is any real sequence; it is auto-scaled so the largest magnitude
        maps to full scale (+/-1). Returns the point count.
        """
        arr = np.asarray(data, dtype=float)
        n = int(arr.size)
        if n < 8:
            raise ValueError(f"arb needs at least 8 points, got {n}")
        peak = float(np.max(np.abs(arr)))
        if peak == 0.0:
            raise ValueError("waveform is all zeros - cannot scale")
        payload = ",".join(f"{v:.6f}" for v in (arr / peak))
        if clear:
            self.cmd_clear()
        self.link.write(f"{self._src}DATA:ARB{self.n} {name},{payload}")
        self.link.query("*OPC?")
        return n

    def set_selected(self, name: str) -> str:
        """Select the active arbitrary waveform by name. Returns read-back."""
        self.write(f'{self._src}FUNC:ARB "{name}"')
        return self.get_selected()

    def get_selected(self) -> str:
        """Query the selected arbitrary-waveform name."""
        return self.query(f"{self._src}FUNC:ARB?").strip().strip('"')

    def set_sample_rate(self, sample_rate: float | str) -> float:
        """Set the arb sample rate in Sa/s."""
        self.write(f"{self._src}FUNC:ARB:SRAT {_num(sample_rate)}")
        return self.get_sample_rate()

    def get_sample_rate(self, limit: str | None = None) -> float:
        """Query the arb sample rate in Sa/s; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}FUNC:ARB:SRAT?" + (f" {limit}" if limit else ""))

    def set_frequency(self, frequency: float | str) -> float:
        """Set the arb frequency in Hz; = sample_rate/points."""
        self.write(f"{self._src}FUNC:ARB:FREQ {_num(frequency)}")
        return self.get_frequency()

    def get_frequency(self, limit: str | None = None) -> float:
        """Query the arb frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}FUNC:ARB:FREQ?" + (f" {limit}" if limit else ""))

    def get_period(self, limit: str | None = None) -> float:
        """Query the arb period in seconds."""
        return self.query_float(f"{self._src}FUNC:ARB:PER?" + (f" {limit}" if limit else ""))

    def set_peak_to_peak(self, volts: float | str) -> float:
        """Set the arb peak-to-peak voltage."""
        self.write(f"{self._src}FUNC:ARB:PTP {_num(volts)}")
        return self.get_peak_to_peak()

    def get_peak_to_peak(self, limit: str | None = None) -> float:
        """Query the arb peak-to-peak voltage; ``limit`` in {MIN, MAX}."""
        return self.query_float(f"{self._src}FUNC:ARB:PTP?" + (f" {limit}" if limit else ""))

    def set_filter(self, mode: "ArbFilter | str") -> "ArbFilter":
        """Set the reconstruction filter applied to playback; an :class:`ArbFilter`.

        Enum arg: <awg>.ArbFilter.<TAB>
        """
        self.write(f"{self._src}FUNC:ARB:FILT {_token(mode, ArbFilter, 'mode')}")
        return self.get_filter()

    def get_filter(self) -> "ArbFilter":
        """The reconstruction filter applied to playback."""
        return _member(self.query(f"{self._src}FUNC:ARB:FILT?").strip(), ArbFilter, "mode")

    def set_advance(self, mode: "ArbAdvance | str") -> "ArbAdvance":
        """Set what advances the waveform one point; an :class:`ArbAdvance`.

        Enum arg: <awg>.ArbAdvance.<TAB>
        """
        self.write(f"{self._src}FUNC:ARB:ADV {_token(mode, ArbAdvance, 'mode')}")
        return self.get_advance()

    def get_advance(self) -> "ArbAdvance":
        """What advances the waveform from one point to the next."""
        return _member(self.query(f"{self._src}FUNC:ARB:ADV?").strip(), ArbAdvance, "mode")

    def get_points(self) -> int:
        """Query the number of points in the active arb."""
        return int(round(self.query_float(f"{self._src}FUNC:ARB:POIN?")))

    def get_catalog(self) -> list[str]:
        """List volatile-memory arb/sequence names."""
        resp = self.query(f"{self._src}DATA:VOL:CAT?").strip()
        return [s.strip().strip('"') for s in resp.split(",") if s.strip().strip('"')]

    def get_free(self) -> int:
        """Query the free volatile-memory point count."""
        return int(round(self.query_float(f"{self._src}DATA:VOL:FREE?")))

    def cmd_clear(self) -> None:
        """Clear the channel's volatile waveform memory."""
        self.write(f"{self._src}DATA:VOL:CLE")
        self.link.query("*OPC?")

    def _set_byte_order(self, order: ByteOrder | str) -> ByteOrder:
        """Set the byte order of a binary block transfer (FORM:BORD)."""
        self.link.write(f"FORM:BORD {_token(order, ByteOrder, 'order')}")
        return self._get_byte_order()

    def _get_byte_order(self) -> ByteOrder:
        """Query the byte order of a binary block transfer (FORM:BORD?)."""
        return _member(self.link.query("FORM:BORD?").strip(), ByteOrder, "order")

    def get_attribute_points(self, name: str = "") -> int:
        """Number of points in a stored arb; default = active."""
        arg = f' "{name}"' if name else ""
        return int(round(self.query_float(f"{self._src}DATA:ATTR:POIN?{arg}")))

    def get_attribute_average(self, name: str = "") -> float:
        """Arithmetic mean of a stored arb's points."""
        arg = f' "{name}"' if name else ""
        return self.query_float(f"{self._src}DATA:ATTR:AVER?{arg}")

    def get_attribute_peak_to_peak(self, name: str = "") -> float:
        """Peak-to-peak of a stored arb, 0..1 normalized."""
        arg = f' "{name}"' if name else ""
        return self.query_float(f"{self._src}DATA:ATTR:PTP?{arg}")

    def get_attribute_crest_factor(self, name: str = "") -> float:
        """Crest factor (peak/RMS) of a stored arb."""
        arg = f' "{name}"' if name else ""
        return self.query_float(f"{self._src}DATA:ATTR:CFAC?{arg}")


class _Channel(Node):
    """One output channel of the generator.

    A channel node exists only for a channel the instrument actually has, so
    there is nothing here that can be called and then refuse. The node owns the
    channel's whole surface, so setting a channel up reads top to bottom with no
    select-then-act step.
    """

    def __init__(self, link, n: int):
        super().__init__(link, name=f"CH{n}")
        self.n = n
        self._src = f"SOUR{n}:"  # source-command prefix, e.g. "SOUR2:FREQ"

        # Per-channel modulation subsystems (configure params, then set_state(True)).
        self.am = _AM(link, self._src, "AM", name=f"CH{n}:AM")
        self.fm = _FM(link, self._src, "FM", name=f"CH{n}:FM")
        self.pm = _PM(link, self._src, "PM", name=f"CH{n}:PM")
        self.pwm = _PWM(link, self._src, "PWM", name=f"CH{n}:PWM")
        self.sum = _SUM(link, self._src, "SUM", name=f"CH{n}:SUM")
        self.fsk = _FSK(link, self._src, "FSK", name=f"CH{n}:FSK", allow_channel_source=False)
        self.bpsk = _BPSK(link, self._src, "BPSK", name=f"CH{n}:BPSK", allow_channel_source=False)

        # Per-channel sweep / burst / frequency-list / trigger / arbitrary.
        self.sweep = _Sweep(link, self._src, name=f"CH{n}:SWEEP")
        self.burst = _Burst(link, self._src, name=f"CH{n}:BURST")
        self.freq_list = _FreqList(link, self._src, name=f"CH{n}:LIST")
        self.trigger = _Trigger(link, n, name=f"CH{n}:TRIGGER")
        self.arb = _Arb(link, self._src, n, name=f"CH{n}:ARB")

    # ---------- the output connector ----------

    def set_output_state(self, state: bool) -> bool:
        """Open or close this channel's output relay.

        The relay switches without zeroing the signal first, so the output can
        glitch as it operates. Where a clean transition matters, drop the
        amplitude and zero the offset before calling this.
        """
        self.write(f"OUTP{self.n} {_onoff(state)}")
        return self.get_output_state()

    def get_output_state(self) -> bool:
        """Whether this channel's output relay is closed."""
        return self.query_bool(f"OUTP{self.n}?")

    def set_load(self, load: float | Load | str) -> float:
        """Set what the far end of the cable terminates into, in ohms.

        Enum arg: <awg>.Load.<TAB>

        Takes a resistance from 1 to 10000 ohm, or ``Load.HIGHZ``. This does not
        change the fixed 50 ohm source impedance. It tells the generator what to
        assume, so the amplitude it reports matches what the load actually sees.
        Returns the read-back, where ``float('inf')`` means high impedance.
        """
        if isinstance(load, (int, float)) and not isinstance(load, bool):
            self.write(f"OUTP{self.n}:LOAD {_num(load)}")
        else:
            self.write(f"OUTP{self.n}:LOAD {_token(load, Load, 'load')}")
        return self.get_load()

    def get_load(self) -> float:
        """What the far end terminates into, in ohms; ``float('inf')`` is high-Z."""
        ohms = self.query_float(f"OUTP{self.n}:LOAD?")
        return float("inf") if ohms >= _HIGHZ_SENTINEL else ohms

    def set_polarity(self, polarity: Polarity | str) -> Polarity:
        """Set whether the output is inverted about its offset; a :class:`Polarity`.

        Enum arg: <awg>.Polarity.<TAB>
        """
        self.write(f"OUTP{self.n}:POL {_token(polarity, Polarity, 'polarity')}")
        return self.get_polarity()

    def get_polarity(self) -> Polarity:
        """Whether the output is inverted about its offset."""
        return _member(self.query(f"OUTP{self.n}:POL?").strip(), Polarity, "polarity")

    def set_gating_state(self, state: bool) -> bool:
        """Gate the output on the external trigger input.

        While gating is on, the waveform reaches the connector only when the
        trigger input is asserted. This is neither the output relay nor the
        burst gate.

        Named for what it does rather than for its command: the scope's AFG has
        a differently-behaved setting of the same original name, and one name
        must mean one thing across these drivers (design standard 3.6).
        """
        self.write(f"OUTP{self.n}:MODE {'GAT' if state else 'NORM'}")
        return self.get_gating_state()

    def get_gating_state(self) -> bool:
        """Whether the output is gated on the external trigger input."""
        return self.query(f"OUTP{self.n}:MODE?").strip().upper().startswith("GAT")

    def set_sync_mode(self, mode: SyncMode | str) -> SyncMode:
        """Set what the Sync connector marks while this channel drives it.

        Enum arg: <awg>.SyncMode.<TAB>

        Per channel. It takes effect when the one physical Sync connector is
        pointed at this channel with ``awg.sync.set_source``.
        """
        self.write(f"OUTP{self.n}:SYNC:MODE {_token(mode, SyncMode, 'mode')}")
        return self.get_sync_mode()

    def get_sync_mode(self) -> SyncMode:
        """What the Sync connector marks while this channel drives it."""
        return _member(self.query(f"OUTP{self.n}:SYNC:MODE?").strip(), SyncMode, "mode")

    def set_sync_polarity(self, polarity: Polarity | str) -> Polarity:
        """Set the Sync pulse polarity for this channel; a :class:`Polarity`.

        Enum arg: <awg>.Polarity.<TAB>
        """
        self.write(f"OUTP{self.n}:SYNC:POL {_token(polarity, Polarity, 'polarity')}")
        return self.get_sync_polarity()

    def get_sync_polarity(self) -> Polarity:
        """The Sync pulse polarity for this channel."""
        return _member(self.query(f"OUTP{self.n}:SYNC:POL?").strip(), Polarity, "polarity")

    # ---------- function ----------

    def _set_function(self, function: Function | str) -> Function:
        """Write the output shape (SOUR<c>:FUNC).

        Private because a bare shape change is not a complete operating point:
        it leaves whatever frequency, amplitude and offset the previous shape
        was using. The ``set_waveform_*`` helpers own the shape change and land
        every coupled value with it.
        """
        self.write(f"{self._src}FUNC {_token(function, Function, 'function')}")
        return self._get_function()

    def _get_function(self) -> Function:
        """The shape now being generated."""
        return _member(self.query(f"{self._src}FUNC?").strip(), Function, "function")

    # ---------- frequency ----------

    def set_frequency(self, frequency: float | str) -> float:
        """Set the output frequency in Hz (accepts MIN/MAX/DEF). Returns read-back."""
        self.write(f"{self._src}FREQ {_num(frequency)}")
        return self.get_frequency()

    def get_frequency(self, limit: str | None = None) -> float:
        """Query the output frequency in Hz; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FREQ?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_period(self, period: float) -> float:
        """Set the output period in seconds; the reciprocal of the frequency."""
        self.set_frequency(1.0 / period)
        return self.get_period()

    def get_period(self) -> float:
        """Query the output period in seconds (computed: 1/frequency)."""
        return 1.0 / self.get_frequency()

    def _set_frequency_mode(self, mode: FrequencyMode | str) -> FrequencyMode:
        """Write what drives the output frequency (SOUR<c>:FREQ:MODE).

        Private because the sweep and list helpers select the mode themselves;
        setting it alone arms a sweep whose span may be from another session.
        """
        self.write(f"{self._src}FREQ:MODE {_token(mode, FrequencyMode, 'mode')}")
        return self._get_frequency_mode()

    def _get_frequency_mode(self) -> FrequencyMode:
        """What drives the output frequency: a fixed value, a sweep, or a list."""
        token = self.query(f"{self._src}FREQ:MODE?").strip().upper()
        # The instrument answers CW for the fixed case; FIX is the same state.
        return (
            FrequencyMode.FIXED if token.startswith("CW") else _member(token, FrequencyMode, "mode")
        )

    # ---------- amplitude / offset / levels ----------

    def set_amplitude(self, amplitude: float | str) -> float:
        """Set the output amplitude in the current unit (accepts MIN/MAX/DEF).

        Interpreted in the units from :meth:`set_amplitude_unit` (default Vpp) and
        scaled by the load setting (see :meth:`set_load`). Returns the read-back.
        """
        self.write(f"{self._src}VOLT {_num(amplitude)}")
        return self.get_amplitude()

    def get_amplitude(self, limit: str | None = None) -> float:
        """Query the output amplitude in the current unit; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}VOLT?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_offset(self, offset: float | str) -> float:
        """Set the DC offset in volts (accepts MIN/MAX/DEF). Returns the read-back.

        The offset range is coupled to amplitude (``|Voffset| < Vmax - Vpp/2``);
        an out-of-range value is clamped by the instrument, which queues a
        "Data out of range" error surfaced by :meth:`Keysight_33500.check_errors`.
        """
        self.write(f"{self._src}VOLT:OFFS {_num(offset)}")
        return self.get_offset()

    def get_offset(self, limit: str | None = None) -> float:
        """Query the DC offset in volts; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}VOLT:OFFS?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_high_level(self, volts: float | str) -> float:
        """Set the waveform high level in volts. Returns the read-back.

        Setting high/low levels also sets amplitude and offset. High must stay at
        least 1 mV above low; the instrument adjusts and queues an error otherwise.
        """
        self.write(f"{self._src}VOLT:HIGH {_num(volts)}")
        return self.get_high_level()

    def get_high_level(self, limit: str | None = None) -> float:
        """Query the waveform high level in volts; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}VOLT:HIGH?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_low_level(self, volts: float | str) -> float:
        """Set the waveform low level in volts. Returns the read-back."""
        self.write(f"{self._src}VOLT:LOW {_num(volts)}")
        return self.get_low_level()

    def get_low_level(self, limit: str | None = None) -> float:
        """Query the waveform low level in volts; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}VOLT:LOW?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def set_amplitude_unit(self, unit: AmplitudeUnit | str) -> AmplitudeUnit:
        """Set how an amplitude argument is read; an :class:`AmplitudeUnit`.

        Enum arg: <awg>.AmplitudeUnit.<TAB>

        DBM is meaningless into a high-impedance load, and the instrument turns
        it into VPP without complaining.
        """
        self.write(f"{self._src}VOLT:UNIT {_token(unit, AmplitudeUnit, 'unit')}")
        return self.get_amplitude_unit()

    def get_amplitude_unit(self) -> AmplitudeUnit:
        """How an amplitude argument is read."""
        return _member(self.query(f"{self._src}VOLT:UNIT?").strip(), AmplitudeUnit, "unit")

    def _set_voltage_autorange(self, value: bool | str) -> bool:
        """Set voltage autoranging.

        ``True``/``False`` enable/disable; the string ``"ONCE"`` performs an
        immediate autorange then leaves it OFF. Disabling it avoids attenuator
        glitches when stepping amplitude. Returns the read-back state.
        """
        if isinstance(value, str):
            if value.upper() != "ONCE":
                raise ValueError("string value must be 'ONCE'; else pass a bool")
            self.write(f"{self._src}VOLT:RANG:AUTO ONCE")
        else:
            self.write(f"{self._src}VOLT:RANG:AUTO {_onoff(value)}")
        return self._get_voltage_autorange()

    def _get_voltage_autorange(self) -> bool:
        """Query whether voltage autoranging is enabled."""
        return self.query_bool(f"{self._src}VOLT:RANG:AUTO?")

    def _set_voltage_limit_high(self, volts: float | str) -> float:
        """Set the high output-voltage limit in volts.

        Only enforced while :meth:`_set_voltage_limits_state` is on. Limits are
        referenced to the current load setting.
        """
        self.write(f"{self._src}VOLT:LIM:HIGH {_num(volts)}")
        return self._get_voltage_limit_high()

    def _get_voltage_limit_high(self, limit: str | None = None) -> float:
        """Query the high output-voltage limit in volts; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}VOLT:LIM:HIGH?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def _set_voltage_limit_low(self, volts: float | str) -> float:
        """Set the low output-voltage limit in volts."""
        self.write(f"{self._src}VOLT:LIM:LOW {_num(volts)}")
        return self._get_voltage_limit_low()

    def _get_voltage_limit_low(self, limit: str | None = None) -> float:
        """Query the low output-voltage limit in volts; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}VOLT:LIM:LOW?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def _set_voltage_limits_state(self, state: bool) -> bool:
        """Enable/disable output voltage limits. Read-back state."""
        self.write(f"{self._src}VOLT:LIM:STAT {_onoff(state)}")
        return self._get_voltage_limits_state()

    def _get_voltage_limits_state(self) -> bool:
        """Query whether output voltage limits are enabled."""
        return self.query_bool(f"{self._src}VOLT:LIM:STAT?")

    # ---------- the waveform, one call per shape (design standard 8) ----------
    #
    # These helpers never use APPLy. APPLy closes the output relay as a side
    # effect, which breaks the rule that a config helper never energizes
    # (design standard 6.2 rule 4), and it rewrites every parameter every time,
    # which makes a one-argument change discontinuous.
    #
    # Each helper instead reads the present state and writes only what differs,
    # in an order that is valid at every moment: shape first, then the shape's
    # own parameters, then amplitude, then offset (its valid range depends on
    # the amplitude), then frequency last. A second call with one changed
    # argument therefore moves one parameter and leaves the output otherwise
    # continuous (design standard 4.5).

    def _set_waveform(
        self,
        function: Function,
        *,
        freq: float | None = None,
        ampl: float | None = None,
        offset: float | None = None,
        shape_writes: "tuple[tuple[str, float, float, int], ...]" = (),
    ) -> WaveformState:
        """Land a complete operating point, writing only what is not already set.

        ``shape_writes`` carries the per-shape parameters as
        ``(command_suffix, wanted, present, significant_digits)``.

        The order depends on whether the shape is changing, because a shape
        change is the one moment the new shape's registers still hold another
        session's values (design standard 6.3).

        Changing shape: frequency and the shape's own parameters are written
        FIRST, while the old shape is still selected, so the new shape starts
        from correct values. Writing them afterwards means the instrument spends
        a moment in the new shape with a stale width or symmetry, and it clamps
        against them. Bench-observed 2026-08-11: selecting pulse with a stale
        100 us width against a 100 us period queued -221 "pulse width decreased
        due to period" and then "both edge times decreased due to pulse width",
        and the clamped values did not recover when the real width arrived.

        Not changing shape: everything is already consistent, so the writes go
        in the ordinary order and only the differing values move.
        """
        changing_shape = self._get_function() is not function

        def write_shape_params() -> None:
            for suffix, wanted, present, digits in shape_writes:
                if wanted is not None and not _same(present, wanted, digits):
                    self.write(f"{self._src}{suffix} {_num(wanted)}")

        def write_frequency() -> None:
            if freq is not None and not _same(self.get_frequency(), freq, _SIG_FREQUENCY):
                self.write(f"{self._src}FREQ {_num(freq)}")

        if changing_shape:
            write_frequency()
            write_shape_params()
            self.write(f"{self._src}FUNC {function.value}")
        else:
            write_shape_params()

        # Levels last, and amplitude before offset: the valid offset range
        # depends on the amplitude, so the other order can clamp the offset.
        if ampl is not None and not _same(self.get_amplitude(), ampl, _SIG_VOLT):
            self.write(f"{self._src}VOLT {_num(ampl)}")
        if offset is not None and not _same(self.get_offset(), offset, _SIG_VOLT):
            self.write(f"{self._src}VOLT:OFFS {_num(offset)}")
        if not changing_shape:
            write_frequency()
        return self.get_waveform()

    def set_waveform_sine(self, freq: float, ampl: float, *, offset: float = 0.0) -> WaveformState:
        """Output a sine wave at ``freq`` Hz and ``ampl`` volts peak to peak."""
        return self._set_waveform(Function.SINE, freq=freq, ampl=ampl, offset=offset)

    def set_waveform_square(
        self, freq: float, ampl: float, *, offset: float = 0.0, duty: float | None = None
    ) -> WaveformState:
        """Output a square wave; ``duty`` is the high time as a percent.

        Leaving ``duty`` at None keeps whatever duty cycle the square-wave
        register already holds, so stepping frequency alone does not reset it.
        """
        return self._set_waveform(
            Function.SQUARE,
            freq=freq,
            ampl=ampl,
            offset=offset,
            shape_writes=(("FUNC:SQU:DCYC", duty, self._get_square_duty_cycle(), _SIG_PERCENT),)
            if duty is not None
            else (),
        )

    def set_waveform_triangle(
        self, freq: float, ampl: float, *, offset: float = 0.0
    ) -> WaveformState:
        """Output a triangle wave at ``freq`` Hz and ``ampl`` volts peak to peak.

        A triangle is a ramp at 50 percent symmetry on this instrument, so this
        selects RAMP and sets the symmetry.
        """
        return self._set_waveform(
            Function.RAMP,
            freq=freq,
            ampl=ampl,
            offset=offset,
            shape_writes=(("FUNC:RAMP:SYMM", 50.0, self._get_ramp_symmetry(), _SIG_PERCENT),),
        )

    def set_waveform_ramp(
        self, freq: float, ampl: float, *, offset: float = 0.0, symmetry: float | None = None
    ) -> WaveformState:
        """Output a ramp; ``symmetry`` is the percent of the cycle spent rising.

        100 percent is a rising sawtooth, 0 a falling one, 50 a triangle.
        Leaving it at None keeps the present symmetry.
        """
        return self._set_waveform(
            Function.RAMP,
            freq=freq,
            ampl=ampl,
            offset=offset,
            shape_writes=(("FUNC:RAMP:SYMM", symmetry, self._get_ramp_symmetry(), _SIG_PERCENT),)
            if symmetry is not None
            else (),
        )

    def set_waveform_pulse(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        width: float | None = None,
        duty: float | None = None,
        edge_time: float | None = None,
    ) -> WaveformState:
        """Output a pulse train; ``width`` in seconds or ``duty`` in percent.

        Width and duty are the same hardware parameter expressed two ways, so
        pass one or the other. ``edge_time`` is the 10 to 90 percent transition
        in seconds, and this instrument clamps it at 1 microsecond.
        """
        if width is not None and duty is not None:
            raise ValueError(
                "pass width (seconds) or duty (percent), not both: they are the "
                "same pulse parameter expressed two ways."
            )
        writes = []
        if edge_time is not None:
            writes.append(
                ("FUNC:PULS:TRAN", edge_time, self._get_pulse_edge_time(), _SIG_EDGE_TIME)
            )
        if width is not None:
            writes.append(("FUNC:PULS:WIDT", width, self._get_pulse_width(), _SIG_WIDTH))
        if duty is not None:
            writes.append(("FUNC:PULS:DCYC", duty, self._get_pulse_duty_cycle(), _SIG_PERCENT))
        return self._set_waveform(
            Function.PULSE, freq=freq, ampl=ampl, offset=offset, shape_writes=tuple(writes)
        )

    def set_waveform_noise(
        self, ampl: float, *, offset: float = 0.0, bandwidth: float | None = None
    ) -> WaveformState:
        """Output gaussian noise at ``ampl`` volts peak to peak.

        Noise has no frequency, so this takes none. ``bandwidth`` in Hz sets how
        much of the spectrum the noise occupies.
        """
        return self._set_waveform(
            Function.NOISE,
            ampl=ampl,
            offset=offset,
            shape_writes=(
                ("FUNC:NOIS:BAND", bandwidth, self._get_noise_bandwidth(), _SIG_FREQUENCY),
            )
            if bandwidth is not None
            else (),
        )

    def set_waveform_dc(self, offset: float) -> WaveformState:
        """Output a steady level of ``offset`` volts."""
        return self._set_waveform(Function.DC, offset=offset)

    def set_waveform_prbs(
        self,
        bit_rate: float,
        ampl: float,
        *,
        offset: float = 0.0,
        sequence: "PrbsSequence | str | None" = None,
        edge_time: float | None = None,
    ) -> WaveformState:
        """Output a pseudo-random bit stream at ``bit_rate`` bits per second.

        Enum arg: <awg>.PrbsSequence.<TAB>

        ``sequence`` is the shift-register length; a :class:`PrbsSequence`.
        """
        if self._get_function() is not Function.PRBS:
            self.write(f"{self._src}FUNC {Function.PRBS.value}")
        if sequence is not None:
            token = _token(sequence, PrbsSequence, "sequence")
            if self.query(f"{self._src}FUNC:PRBS:DATA?").strip().upper() != token:
                self.write(f"{self._src}FUNC:PRBS:DATA {token}")
        if edge_time is not None and not _same(
            self._get_prbs_edge_time(), edge_time, _SIG_EDGE_TIME
        ):
            self.write(f"{self._src}FUNC:PRBS:TRAN {_num(edge_time)}")
        if not _same(self.get_amplitude(), ampl, _SIG_VOLT):
            self.write(f"{self._src}VOLT {_num(ampl)}")
        if not _same(self.get_offset(), offset, _SIG_VOLT):
            self.write(f"{self._src}VOLT:OFFS {_num(offset)}")
        if not _same(self._get_prbs_bit_rate(), bit_rate, _SIG_FREQUENCY):
            self.write(f"{self._src}FUNC:PRBS:BRAT {_num(bit_rate)}")
        return self.get_waveform()

    def set_waveform_arb(
        self,
        samples: Sequence[float] | None,
        *,
        sample_rate: float,
        ampl: float,
        offset: float = 0.0,
        name: str = "VOLATILE",
        arb_filter: "ArbFilter | str | None" = None,
    ) -> WaveformState:
        """Play ``samples`` back at ``sample_rate`` samples per second.

        Enum arg: <awg>.ArbFilter.<TAB>

        ``samples`` is one period as a sequence of real numbers; it is scaled so
        the largest magnitude reaches ``ampl``/2 either side of the offset. Pass
        None to replay whatever waveform the channel already holds. The output
        frequency is ``sample_rate`` divided by the number of points.
        """
        if samples is not None:
            self.arb.set_data_floats(name, samples)
            self.arb.set_selected(name)
        if self._get_function() is not Function.ARBITRARY:
            self.write(f"{self._src}FUNC {Function.ARBITRARY.value}")
        if arb_filter is not None:
            self.arb.set_filter(arb_filter)
        if not _same(self.arb.get_sample_rate(), sample_rate, _SIG_FREQUENCY):
            self.write(f"{self._src}FUNC:ARB:SRAT {_num(sample_rate)}")
        if not _same(self.get_amplitude(), ampl, _SIG_VOLT):
            self.write(f"{self._src}VOLT {_num(ampl)}")
        if not _same(self.get_offset(), offset, _SIG_VOLT):
            self.write(f"{self._src}VOLT:OFFS {_num(offset)}")
        return self.get_waveform()

    def _get_apply(self) -> tuple:
        """Read shape, frequency, amplitude and offset in one round trip (APPL?).

        Returns ``(function_token, arg1, arg2, offset)``, where arg1 is the
        frequency, or the sample rate for an arb and the bit rate for a PRBS.

        Kept because it is genuinely cheaper than the four separate queries
        :meth:`get_waveform` makes: on this link a query costs about 4.8 ms
        against 1.5 ms for a write, so round trips are the thing worth saving.
        Underscored because it returns an untyped tuple, where get_waveform
        returns a WaveformState with the shape parameters filled in.

        The matching APPLy WRITE is deliberately absent from this driver. It
        closes the output relay as a side effect (bench-verified 2026-08-11),
        so a configuration call would energize the output. The set_waveform_*
        helpers reach the same state without that.
        """
        resp = self.query(f"{self._src}APPL?").strip().strip('"')
        func, _, rest = resp.partition(" ")
        parts = [p for p in rest.split(",") if p.strip()]

        def _f(i: int) -> float:
            try:
                return float(parts[i])
            except (IndexError, ValueError):
                return float("nan")

        return func, _f(0), _f(1), _f(2)

    def get_waveform(self) -> WaveformState:
        """Read the shape now being generated and every parameter that applies.

        Fields that do not apply to the active shape are None: a square wave
        reports a duty cycle and no symmetry, a ramp the reverse.
        """
        function = self._get_function()
        offset = self.get_offset()
        if function is Function.DC:
            return WaveformState(
                function=function.value,
                frequency=0.0,
                amplitude=0.0,
                offset=offset,
                output_state=self.get_output_state(),
            )
        if function is Function.PRBS:
            frequency = self._get_prbs_bit_rate()
        elif function is Function.NOISE:
            frequency = 0.0
        else:
            frequency = self.get_frequency()
        return WaveformState(
            function=function.value,
            frequency=frequency,
            amplitude=self.get_amplitude(),
            offset=offset,
            output_state=self.get_output_state(),
            duty=(
                self._get_square_duty_cycle()
                if function is Function.SQUARE
                else self._get_pulse_duty_cycle()
                if function is Function.PULSE
                else None
            ),
            symmetry=self._get_ramp_symmetry() if function is Function.RAMP else None,
            width=self._get_pulse_width() if function is Function.PULSE else None,
            edge_time=(
                self._get_pulse_edge_time()
                if function is Function.PULSE
                else self._get_prbs_edge_time()
                if function is Function.PRBS
                else None
            ),
            sample_rate=(self.arb.get_sample_rate() if function is Function.ARBITRARY else None),
            bandwidth=self._get_noise_bandwidth() if function is Function.NOISE else None,
        )

    # ---------- output voltage limits ----------

    def set_voltage_limits(self, low: float, high: float) -> "VoltageLimits":
        """Clamp the output between ``low`` and ``high`` volts, and turn the clamp on.

        Limits are referenced to the present load setting. Written low first,
        then high, then the enable, so the clamp is never live against a
        half-updated pair (design standard 6.2 rule 2).
        """
        if low >= high:
            raise ValueError(f"low ({low} V) must be below high ({high} V)")
        self.write(f"{self._src}VOLT:LIM:LOW {_num(low)}")
        self.write(f"{self._src}VOLT:LIM:HIGH {_num(high)}")
        self.write(f"{self._src}VOLT:LIM:STAT ON")
        return self.get_voltage_limits()

    def set_voltage_limits_off(self) -> "VoltageLimits":
        """Stop clamping the output, leaving the limit values where they are."""
        self.write(f"{self._src}VOLT:LIM:STAT OFF")
        return self.get_voltage_limits()

    def get_voltage_limits(self) -> "VoltageLimits":
        """The output voltage clamp, and whether it is in force."""
        return VoltageLimits(
            low=self._get_voltage_limit_low(),
            high=self._get_voltage_limit_high(),
            enabled=self._get_voltage_limits_state(),
        )

    # ---------- modulation, one call per mode (design standard 4.3) ----------
    #
    # This instrument runs one modulation at a time: bench-verified 2026-08-11,
    # enabling FM cleared AM by itself and queued -221 "AM turned off by
    # selection of other mode or modulation". Each helper therefore clears the
    # others explicitly first, so what ends up on is what was asked for rather
    # than whatever the instrument decided to drop.
    #
    # Parameters are written before the enable, so the modulation never runs for
    # an instant with another session's settings.

    def set_am(
        self,
        depth: float,
        *,
        rate: float | None = None,
        shape: "ModulationShape | str | None" = None,
        source: "ModulationSource | str | None" = None,
        dssc: bool | None = None,
    ) -> "ModulationState":
        """Amplitude-modulate the carrier to ``depth`` percent, and turn AM on.

        Enum arg: shape  = <awg>.ModulationShape.<TAB>
                  source = <awg>.ModulationSource.<TAB>

        ``rate`` is the modulating frequency in Hz and ``shape`` its waveform.
        ``dssc`` selects double-sideband suppressed carrier.
        """
        self.set_modulation_off()
        if shape is not None:
            self.am.set_internal_function(shape)
        if rate is not None:
            self.am.set_internal_frequency(rate)
        if source is not None:
            self.am.set_source(source)
        if dssc is not None:
            self.am.set_dssc(dssc)
        self.am.set_depth(depth)
        self.am.set_state(True)
        return self.get_modulation()

    def set_fm(
        self,
        deviation: float,
        *,
        rate: float | None = None,
        shape: "ModulationShape | str | None" = None,
        source: "ModulationSource | str | None" = None,
    ) -> "ModulationState":
        """Frequency-modulate the carrier by ``deviation`` Hz peak, and turn FM on.

        Enum arg: shape  = <awg>.ModulationShape.<TAB>
                  source = <awg>.ModulationSource.<TAB>

        The deviation cannot exceed the carrier frequency; the instrument clamps
        it and queues an error if it does.
        """
        self.set_modulation_off()
        if shape is not None:
            self.fm.set_internal_function(shape)
        if rate is not None:
            self.fm.set_internal_frequency(rate)
        if source is not None:
            self.fm.set_source(source)
        self.fm.set_deviation(deviation)
        self.fm.set_state(True)
        return self.get_modulation()

    def set_pm(
        self,
        deviation: float,
        *,
        rate: float | None = None,
        shape: "ModulationShape | str | None" = None,
        source: "ModulationSource | str | None" = None,
    ) -> "ModulationState":
        """Phase-modulate the carrier by ``deviation`` degrees peak, and turn PM on.

        Enum arg: shape  = <awg>.ModulationShape.<TAB>
                  source = <awg>.ModulationSource.<TAB>
        """
        self.set_modulation_off()
        if shape is not None:
            self.pm.set_internal_function(shape)
        if rate is not None:
            self.pm.set_internal_frequency(rate)
        if source is not None:
            self.pm.set_source(source)
        self.pm.set_deviation(deviation)
        self.pm.set_state(True)
        return self.get_modulation()

    def set_pwm(
        self,
        deviation: float,
        *,
        rate: float | None = None,
        shape: "ModulationShape | str | None" = None,
        source: "ModulationSource | str | None" = None,
    ) -> "ModulationState":
        """Pulse-width-modulate by ``deviation`` seconds peak, and turn PWM on.

        Enum arg: shape  = <awg>.ModulationShape.<TAB>
                  source = <awg>.ModulationSource.<TAB>

        Only a pulse carrier can be width-modulated, so this refuses on any
        other shape rather than letting the instrument queue a settings
        conflict.
        """
        function = self._get_function()
        if function is not Function.PULSE:
            raise ValueError(
                f"set_pwm() needs a pulse carrier; this channel is generating "
                f"{function.name}. Call set_waveform_pulse(...) first."
            )
        self.set_modulation_off()
        if shape is not None:
            self.pwm.set_internal_function(shape)
        if rate is not None:
            self.pwm.set_internal_frequency(rate)
        if source is not None:
            self.pwm.set_source(source)
        self.pwm.set_deviation(deviation)
        self.pwm.set_state(True)
        return self.get_modulation()

    def set_sum(
        self,
        percent: float,
        *,
        rate: float | None = None,
        shape: "ModulationShape | str | None" = None,
        source: "ModulationSource | str | None" = None,
    ) -> "ModulationState":
        """Add a second waveform at ``percent`` of the carrier, and turn SUM on.

        Enum arg: shape  = <awg>.ModulationShape.<TAB>
                  source = <awg>.ModulationSource.<TAB>
        """
        self.set_modulation_off()
        if shape is not None:
            self.sum.set_internal_function(shape)
        if rate is not None:
            self.sum.set_internal_frequency(rate)
        if source is not None:
            self.sum.set_source(source)
        self.sum.set_amplitude(percent)
        self.sum.set_state(True)
        return self.get_modulation()

    def set_fsk(
        self,
        hop_frequency: float,
        *,
        rate: float | None = None,
        source: "ModulationSource | str | None" = None,
    ) -> "ModulationState":
        """Shift between the carrier and ``hop_frequency`` Hz, and turn FSK on.

        Enum arg: <awg>.ModulationSource.<TAB>
        """
        self.set_modulation_off()
        if rate is not None:
            self.fsk.set_rate(rate)
        if source is not None:
            self.fsk.set_source(source)
        self.fsk.set_hop_frequency(hop_frequency)
        self.fsk.set_state(True)
        return self.get_modulation()

    def set_bpsk(
        self,
        phase: float,
        *,
        rate: float | None = None,
        source: "ModulationSource | str | None" = None,
    ) -> "ModulationState":
        """Shift the carrier phase by ``phase`` degrees in step, and turn BPSK on.

        Enum arg: <awg>.ModulationSource.<TAB>
        """
        self.set_modulation_off()
        if rate is not None:
            self.bpsk.set_rate(rate)
        if source is not None:
            self.bpsk.set_source(source)
        self.bpsk.set_phase(phase)
        self.bpsk.set_state(True)
        return self.get_modulation()

    def set_modulation_off(self) -> "ModulationState":
        """Turn every modulation, the sweep and the burst off, leaving a plain carrier."""
        for node in (*self._modulation_nodes().values(), self.sweep, self.burst):
            with contextlib.suppress(Exception):
                node.set_state(False)
        return self.get_modulation()

    def get_modulation(self) -> "ModulationState":
        """Which modulation is running on this channel, and its settings.

        ``mode`` is None when the carrier is unmodulated. The other fields carry
        whichever of depth, deviation or amplitude that mode uses, in that
        mode's own unit.
        """
        active = None
        for name, node in self._modulation_nodes().items():
            if node.get_state():
                active = (name, node)
                break
        if active is None:
            return ModulationState(mode=None)
        name, node = active
        level = {
            "am": lambda: node.get_depth(),
            "fm": lambda: node.get_deviation(),
            "pm": lambda: node.get_deviation(),
            "pwm": lambda: node.get_deviation(),
            "sum": lambda: node.get_amplitude(),
            "fsk": lambda: node.get_hop_frequency(),
            "bpsk": lambda: node.get_phase(),
        }[name]()
        rate = None
        shape = None
        if isinstance(node, _InternalMod):
            rate = node.get_internal_frequency()
            shape = node.get_internal_function()
        elif name in ("fsk", "bpsk"):
            rate = node.get_rate()
        return ModulationState(
            mode=name.upper(),
            level=level,
            rate=rate,
            shape=shape,
            source=node.get_source(),
        )

    # ---------- sweep, burst and frequency list ----------

    def set_sweep(
        self,
        start: float,
        stop: float,
        time: float,
        *,
        spacing: "Spacing | str" = Spacing.LINEAR,
        hold_time: float | None = None,
        return_time: float | None = None,
        marker: float | None = None,
        trigger: "TriggerSource | str | None" = None,
    ) -> "SweepState":
        """Sweep from ``start`` to ``stop`` Hz over ``time`` seconds, and turn sweep on.

        Enum arg: spacing = <awg>.Spacing.<TAB>
                  trigger = <awg>.TriggerSource.<TAB>

        ``hold_time`` dwells at the stop frequency and ``return_time`` sets the
        retrace, both in seconds. ``marker`` is the frequency at which the Sync
        connector drops, which gives the scope a mid-sweep reference.
        """
        self.set_modulation_off()
        self.sweep.set_spacing(spacing)
        self.sweep.set_start_frequency(start)
        self.sweep.set_stop_frequency(stop)
        self.sweep.set_time(time)
        if hold_time is not None:
            self.sweep.set_hold_time(hold_time)
        if return_time is not None:
            self.sweep.set_return_time(return_time)
        if marker is not None:
            self.sweep.set_marker_frequency(marker)
        if trigger is not None:
            self.trigger.set_source(trigger)
        self.sweep.set_state(True)
        return self.get_sweep()

    def get_sweep(self) -> "SweepState":
        """The frequency sweep configuration, and whether it is running."""
        return SweepState(
            enabled=self.sweep.get_state(),
            start=self.sweep.get_start_frequency(),
            stop=self.sweep.get_stop_frequency(),
            time=self.sweep.get_time(),
            spacing=self.sweep.get_spacing(),
            hold_time=self.sweep.get_hold_time(),
            return_time=self.sweep.get_return_time(),
        )

    def set_burst(
        self,
        cycles: int,
        *,
        period: float | None = None,
        phase: float | None = None,
        mode: "BurstMode | str" = BurstMode.TRIGGERED,
        trigger: "TriggerSource | str | None" = None,
    ) -> "BurstState":
        """Emit ``cycles`` cycles per trigger, and turn burst on.

        Enum arg: mode    = <awg>.BurstMode.<TAB>
                  trigger = <awg>.TriggerSource.<TAB>

        ``period`` is how often a burst starts, in seconds, and applies only
        with an immediate trigger source. ``phase`` is where in the waveform
        each burst begins. Pass ``cycles`` as ``float('inf')`` for a gated burst
        that runs while the gate is asserted.
        """
        self.set_modulation_off()
        self.burst.set_mode(mode)
        if phase is not None:
            self.burst.set_phase(phase)
        if period is not None:
            self.burst.set_period(period)
        if trigger is not None:
            self.trigger.set_source(trigger)
        self.burst.set_cycles("INF" if cycles == float("inf") else cycles)
        self.burst.set_state(True)
        return self.get_burst()

    def get_burst(self) -> "BurstState":
        """The burst configuration, and whether it is running."""
        return BurstState(
            enabled=self.burst.get_state(),
            cycles=self.burst.get_cycles(),
            mode=self.burst.get_mode(),
            period=self.burst.get_period(),
            phase=self.burst.get_phase(),
        )

    def set_frequency_list(self, frequencies: Sequence[float], *, dwell: float) -> list[float]:
        """Step through ``frequencies`` in Hz, holding each for ``dwell`` seconds.

        Takes 1 to 128 values. Returns the list the instrument holds. The list
        advances on the trigger source, so an immediate source free-runs it.
        """
        self.set_modulation_off()
        self.freq_list.set_frequencies(frequencies)
        self.freq_list.set_dwell(dwell)
        self._set_frequency_mode(FrequencyMode.LIST)
        return self.freq_list.get_frequencies()

    # ---------- modulation state, for snapshot and restore ----------

    def _modulation_nodes(self) -> "dict[str, Node]":
        """Every modulation subsystem on this channel, by attribute name."""
        return {
            "am": self.am,
            "fm": self.fm,
            "pm": self.pm,
            "pwm": self.pwm,
            "sum": self.sum,
            "fsk": self.fsk,
            "bpsk": self.bpsk,
        }

    def _get_modulation_states(self) -> "dict[str, bool]":
        """Which modulations are on, plus sweep and burst."""
        states = {name: node.get_state() for name, node in self._modulation_nodes().items()}
        states["sweep"] = self.sweep.get_state()
        states["burst"] = self.burst.get_state()
        return states

    def _restore_modulation_states(self, states: "dict[str, bool]") -> None:
        """Put the modulation, sweep and burst enables back as they were.

        Everything is turned off before anything is turned on, because the
        instrument runs one at a time and enabling one clears another.
        """
        nodes = dict(self._modulation_nodes())
        nodes["sweep"] = self.sweep
        nodes["burst"] = self.burst
        for name, node in nodes.items():
            if not states.get(name, False):
                with contextlib.suppress(Exception):
                    node.set_state(False)
        for name, node in nodes.items():
            if states.get(name, False):
                with contextlib.suppress(Exception):
                    node.set_state(True)

    # ---------- square-wave shape ----------

    def _set_square_duty_cycle(self, percent: float | str) -> float:
        """Set the square-wave duty cycle in percent, 0.01-99.99.

        Limited at high frequency by the 16 ns (33500) / 5-8 ns (33600) minimum
        pulse width; the instrument clamps and queues -221 if out of range.
        """
        self.write(f"{self._src}FUNC:SQU:DCYC {_num(percent)}")
        return self._get_square_duty_cycle()

    def _get_square_duty_cycle(self, limit: str | None = None) -> float:
        """Query the square-wave duty cycle in percent; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:SQU:DCYC?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def _set_square_period(self, seconds: float | str) -> float:
        """Set the square-wave period in seconds. Affects frequency."""
        self.write(f"{self._src}FUNC:SQU:PER {_num(seconds)}")
        return self._get_square_period()

    def _get_square_period(self, limit: str | None = None) -> float:
        """Query the square-wave period in seconds; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:SQU:PER?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    # ---------- ramp shape ----------

    def _set_ramp_symmetry(self, percent: float | str) -> float:
        """Set the ramp symmetry in percent, 0-100.

        Symmetry is the fraction of the cycle spent rising: 100% is a sawtooth
        rising, 0% falling, 50% a triangle. Returns the read-back.
        """
        self.write(f"{self._src}FUNC:RAMP:SYMM {_num(percent)}")
        return self._get_ramp_symmetry()

    def _get_ramp_symmetry(self, limit: str | None = None) -> float:
        """Query the ramp symmetry in percent; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:RAMP:SYMM?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    # ---------- pulse shape ----------

    def _set_pulse_width(self, seconds: float | str) -> float:
        """Set the pulse width in seconds, 50% to 50% thresholds."""
        self.write(f"{self._src}FUNC:PULS:WIDT {_num(seconds)}")
        return self._get_pulse_width()

    def _get_pulse_width(self, limit: str | None = None) -> float:
        """Query the pulse width in seconds; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:PULS:WIDT?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def _set_pulse_duty_cycle(self, percent: float | str) -> float:
        """Set the pulse duty cycle in percent. Same parameter as width."""
        self.write(f"{self._src}FUNC:PULS:DCYC {_num(percent)}")
        return self._get_pulse_duty_cycle()

    def _get_pulse_duty_cycle(self, limit: str | None = None) -> float:
        """Query the pulse duty cycle in percent; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:PULS:DCYC?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def _set_pulse_period(self, seconds: float | str) -> float:
        """Set the pulse period in seconds. Paired with FREQ."""
        self.write(f"{self._src}FUNC:PULS:PER {_num(seconds)}")
        return self._get_pulse_period()

    def _get_pulse_period(self, limit: str | None = None) -> float:
        """Query the pulse period in seconds; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:PULS:PER?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def _set_pulse_edge_time(self, seconds: float | str, edge: str = "BOTH") -> float:
        """Set the pulse edge (transition) time in seconds.

        ``edge`` selects which edge: ``"BOTH"`` (default), ``"LEADING"``, or
        ``"TRAILING"``. Edge time is the 10%-90% transition. Returns the
        read-back for the selected edge.
        """
        node = self._pulse_edge_node(edge)
        self.write(f"{self._src}FUNC:PULS:TRAN{node} {_num(seconds)}")
        return self._get_pulse_edge_time(edge)

    def _get_pulse_edge_time(self, edge: str = "BOTH", limit: str | None = None) -> float:
        """Query the pulse edge time in seconds for ``edge``; ``limit`` in {MIN, MAX}."""
        node = self._pulse_edge_node(edge)
        cmd = f"{self._src}FUNC:PULS:TRAN{node}?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    @staticmethod
    def _pulse_edge_node(edge: str) -> str:
        mapping = {
            "BOTH": "",
            "LEADING": ":LEAD",
            "LEAD": ":LEAD",
            "TRAILING": ":TRA",
            "TRAIL": ":TRA",
        }
        key = edge.upper()
        if key not in mapping:
            raise ValueError(f"edge must be BOTH/LEADING/TRAILING, got {edge!r}")
        return mapping[key]

    def _set_pulse_hold(self, which: str) -> str:
        """Set which pulse parameter is held as period varies.

        ``"WIDTH"`` holds pulse width (seconds); ``"DCYCLE"`` holds duty cycle
        (percent). Returns the read-back token (WIDT or DCYC).
        """
        self.write_choice(
            f"{self._src}FUNC:PULS:HOLD",
            which.upper(),
            {"WIDTH": "WIDTh", "WIDT": "WIDTh", "DCYCLE": "DCYCle", "DCYC": "DCYCle"},
        )
        return self._get_pulse_hold()

    def _get_pulse_hold(self) -> str:
        """Query the held pulse parameter token (WIDT or DCYC)."""
        return self.query(f"{self._src}FUNC:PULS:HOLD?").strip()

    # ---------- noise shape ----------

    def _set_noise_bandwidth(self, hertz: float | str) -> float:
        """Set the noise bandwidth in Hz, 1 mHz to max frequency."""
        self.write(f"{self._src}FUNC:NOIS:BAND {_num(hertz)}")
        return self._get_noise_bandwidth()

    def _get_noise_bandwidth(self, limit: str | None = None) -> float:
        """Query the noise bandwidth in Hz; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:NOIS:BAND?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    # ---------- PRBS shape ----------

    def _set_prbs_bit_rate(self, bits_per_s: float | str) -> float:
        """Set the PRBS bit rate in bits/s."""
        self.write(f"{self._src}FUNC:PRBS:BRAT {_num(bits_per_s)}")
        return self._get_prbs_bit_rate()

    def _get_prbs_bit_rate(self, limit: str | None = None) -> float:
        """Query the PRBS bit rate in bits/s; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:PRBS:BRAT?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def _set_prbs_data(self, sequence: str) -> str:
        """Set the PRBS sequence type: PN7/PN9/PN11/PN15/PN20/PN23.

        (33600 accepts PN3 through PN32.) The value after PN is the shift-register
        length in bits. Returns the read-back token.
        """
        self.write(f"{self._src}FUNC:PRBS:DATA {sequence.upper()}")
        return self._get_prbs_data()

    def _get_prbs_data(self) -> str:
        """Query the PRBS sequence type token (e.g. PN7)."""
        return self.query(f"{self._src}FUNC:PRBS:DATA?").strip()

    def _set_prbs_edge_time(self, seconds: float | str) -> float:
        """Set the PRBS transition (edge) time in seconds."""
        self.write(f"{self._src}FUNC:PRBS:TRAN {_num(seconds)}")
        return self._get_prbs_edge_time()

    def _get_prbs_edge_time(self, limit: str | None = None) -> float:
        """Query the PRBS edge time in seconds; ``limit`` in {MIN, MAX}."""
        cmd = f"{self._src}FUNC:PRBS:TRAN?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    # ---------- phase ----------

    def set_phase(self, angle: float | str) -> float:
        """Set the waveform phase offset in the current angle unit.

        Default unit is degrees (see :meth:`Keysight_33500.set_angle_unit`); range
        -360 to +360 deg. This adds an offset; use
        :meth:`Keysight_33500.phase_sync` to establish a common channel-to-channel
        zero first. Returns the read-back.
        """
        self.write(f"{self._src}PHAS {_num(angle)}")
        return self.get_phase()

    def get_phase(self, limit: str | None = None) -> float:
        """Query the waveform phase offset in the current angle unit."""
        cmd = f"{self._src}PHAS?" + (f" {limit}" if limit else "")
        return self.query_float(cmd)

    def cmd_phase_reference(self) -> None:
        """Set a new zero-phase reference: folds the current offset into
        the primary generator and resets the queried phase to 0 without changing
        the output."""
        self.write(f"{self._src}PHAS:REF")

    def set_phase_unlock_error_state(self, state: bool) -> bool:
        """Enable/disable a "phase-locked loop unlocked" error."""
        self.write(f"{self._src}PHAS:UNL:ERR:STAT {_onoff(state)}")
        return self.get_phase_unlock_error_state()

    def get_phase_unlock_error_state(self) -> bool:
        """Query whether the phase-unlock error is enabled."""
        return self.query_bool(f"{self._src}PHAS:UNL:ERR:STAT?")


def _ch_token(channel: "ChannelId | str | int") -> str:
    """Normalize a channel argument to the CH1 or CH2 token.

    Accepts a :class:`ChannelId`, the token, the member name, or a plain 1 or 2,
    because all four read naturally at a call site and a round trip through
    ``get_source`` hands back a member.
    """
    if isinstance(channel, int) and not isinstance(channel, bool):
        if channel in (1, 2):
            return f"CH{channel}"
        raise ValueError(f"channel must be 1 or 2, got {channel!r}")
    return _token(channel, ChannelId, "channel")


class _Sync(Node):
    """The shared front-panel Sync output (OUTPut:SYNC ...).

    One physical Sync connector; :meth:`set_source` chooses which channel drives
    it. ``mode`` and ``polarity`` are per-channel (they apply when that channel is
    the sync source).
    """

    def set_state(self, state: bool) -> bool:
        """Enable/disable the Sync connector. Returns the read-back."""
        self.write(f"OUTP:SYNC {_onoff(state)}")
        return self.get_state()

    def get_state(self) -> bool:
        """Query whether the Sync connector is enabled."""
        return self.query_bool("OUTP:SYNC?")

    def set_source(self, channel: "ChannelId | str | int") -> "ChannelId":
        """Choose which channel drives the one physical Sync connector.

        Enum arg: <awg>.ChannelId.<TAB>

        The mode and the polarity of the Sync pulse belong to the channel that
        drives it, so they live there: ``awg.ch1.set_sync_mode``.
        """
        self.write(f"OUTP:SYNC:SOUR {_ch_token(channel)}")
        return self.get_source()

    def get_source(self) -> "ChannelId":
        """Which channel drives the Sync connector."""
        return _member(self.query("OUTP:SYNC:SOUR?").strip(), ChannelId, "channel")


class _TriggerOut(Node):
    """The rear-panel Ext Trig connector as a trigger OUTPUT for sweep/burst."""

    def set_state(self, state: bool) -> bool:
        """Enable/disable the trigger-out pulse. Returns the read-back."""
        self.write(f"OUTP:TRIG {_onoff(state)}")
        return self.get_state()

    def get_state(self) -> bool:
        """Query whether trigger-out is enabled."""
        return self.query_bool("OUTP:TRIG?")

    def set_slope(self, slope: "Slope | str") -> "Slope":
        """Set which edge the outgoing trigger pulse presents; a :class:`Slope`.

        Enum arg: <awg>.Slope.<TAB>
        """
        self.write(f"OUTP:TRIG:SLOP {_token(slope, Slope, 'slope')}")
        return self.get_slope()

    def get_slope(self) -> "Slope":
        """Which edge the outgoing trigger pulse presents."""
        return _member(self.query("OUTP:TRIG:SLOP?").strip(), Slope, "slope")

    def set_source(self, channel: "ChannelId | str | int") -> "ChannelId":
        """Choose which channel's sweep or burst emits the trigger pulse.

        Enum arg: <awg>.ChannelId.<TAB>
        """
        self.write(f"OUTP:TRIG:SOUR {_ch_token(channel)}")
        return self.get_source()

    def get_source(self) -> "ChannelId":
        """Which channel's sweep or burst emits the trigger pulse."""
        return _member(self.query("OUTP:TRIG:SOUR?").strip(), ChannelId, "channel")


class Keysight_33500(Instrument):
    """Keysight/Agilent 33500 and 33600 series Trueform waveform generator.

    ``__init__`` does no I/O, so building one costs nothing and cannot hang on a
    powered-off instrument. ``connect()`` reads ``*IDN?`` and attaches the
    channels the model actually has: on a one-channel model ``ch2`` does not
    exist at all, rather than existing and refusing to work.
    """

    # Declared for the API-map generator, which cannot see attributes that
    # connect() attaches. No assignment, so nothing exists until it is real.
    ch1: _Channel
    ch2: _Channel  # attached by connect() on two-channel models only

    # Enums live on the class so the operator can reach them from the live
    # object: awg.Function.<TAB> resolves, where a bare name would not.
    Function = Function
    AmplitudeUnit = AmplitudeUnit
    Load = Load
    Polarity = Polarity
    FrequencyMode = FrequencyMode
    ModulationShape = ModulationShape
    ModulationSource = ModulationSource
    Spacing = Spacing
    BurstMode = BurstMode
    TriggerSource = TriggerSource
    Slope = Slope
    ArbFilter = ArbFilter
    ArbAdvance = ArbAdvance
    PrbsSequence = PrbsSequence
    PulseHold = PulseHold
    SyncMode = SyncMode
    Track = Track
    Combine = Combine
    AngleUnit = AngleUnit
    Reference = Reference
    ByteOrder = ByteOrder
    ChannelId = ChannelId

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
            pyvisa resource for the generator (default the bench VXI-11 string).
        connect_timeout_ms : int
            I/O timeout used for connect and normal operation, in milliseconds.
        """
        super().__init__(resource_name, name="AWG")
        self.link.default_timeout_ms = connect_timeout_ms
        self._connect_timeout_ms = connect_timeout_ms

        self.model: str | None = None
        self.firmware: str | None = None
        self.options: list[str] = []
        self.num_channels: int = 2
        self.series: str | None = None
        self.max_frequency: float | None = None
        self.max_sample_rate: float | None = None
        self.arb_points: int | None = None

        # Channel 1 always exists. Channel 2 is attached by connect() only when
        # the identified model has one (design standard 5.1 and 5.2).
        self.ch1 = _Channel(self.link, 1)

        # Instrument-level (shared) subsystems.
        self.sync = _Sync(self.link, name="SYNC")
        self.trigger_out = _TriggerOut(self.link, name="TRIGGER_OUT")
        self.display = _Display(self.link, name="DISPLAY")
        self.memory = _Memory(self.link, name="MEMORY")
        self.status = _Status(self.link, name="STATUS")
        self._reference = _ReferenceClock(self.link, name="REFERENCE")
        self.cal = _Calibration(self.link, name="CALIBRATION")

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the pinned resource, confirm the identity, and attach the channels.

        First contact is a VISA clear (base link) plus one ``*IDN?``. A VISA
        timeout is fatal (:class:`InstrumentUnresponsiveError`): the operator must
        power cycle; the link is never retried. An IDN from something other than a
        33500/33600 raises :class:`InstrumentError` - a wrong resource is a wiring
        fact to fix, not a condition to probe around (ADR-034).
        """
        to = self._connect_timeout_ms if timeout_ms is None else timeout_ms
        try:
            self.link.connect(timeout_ms=to)
            idn = self.link.query("*IDN?").strip()
        except visa.VisaIOError as exc:
            if exc.error_code == visa.constants.StatusCode.error_timeout:
                raise InstrumentUnresponsiveError(
                    f"{self.resource_name} did not answer *IDN? (VISA timeout). Stop "
                    "all work - the operator must power cycle the generator."
                ) from exc
            raise
        if not _IDN_RE.search(idn):
            self.link.disconnect()
            raise InstrumentError(
                f"{self.resource_name} did not answer with a 33500/33600 IDN: {idn!r}. "
                "Check the resource; this driver does not scan for the instrument."
            )
        self.idn = idn
        self._on_connect(idn)
        # Clear status + error queue so a session starts clean.
        self.link.write("*CLS")
        return idn

    def _on_connect(self, idn: str) -> None:
        """Parse IDN for model/firmware/options and gate channels + caps."""
        match = _IDN_RE.search(idn)
        if not match:
            raise InstrumentError(f"Not a recognized 33500/33600 IDN: {idn!r}")
        model = match.group(1).upper()
        self.model = model
        fields = [f.strip() for f in idn.split(",")]
        self.firmware = fields[3] if len(fields) >= 4 else None

        info = _MODEL_INFO.get(model)
        if info is None:
            # IDN matched the family pattern but the exact model is not tabled:
            # connect with conservative caps rather than refuse a valid instrument.
            info = {
                "channels": 2,
                "max_freq": 20e6,
                "max_srate": 250e6,
                "arb_points": 1_000_000,
                "series": "33500",
            }
        self.num_channels = info["channels"]
        self.series = info["series"]
        self.max_frequency = info["max_freq"]
        self.max_sample_rate = info["max_srate"]
        self.arb_points = info["arb_points"]

        # Attach or remove ch2 to match the hardware, so autocomplete tells the
        # truth about this instrument rather than about the series.
        if self.num_channels == 2:
            if not hasattr(self, "ch2"):
                self.ch2 = _Channel(self.link, 2)
        elif hasattr(self, "ch2"):
            del self.ch2

        with contextlib.suppress(Exception):
            self.options = self.get_options()

    def safe_state(self) -> None:
        """De-energize both outputs before the link closes.

        This AWG has a per-channel output relay, so unlike the 33120A it does NOT
        leave the signal running: every installed channel's relay is opened.
        Best-effort (never raises out of disconnect).

        NOTE: no ``SYST:LOCal`` is sent. On the LAN (VXI-11) interface the 33500/
        33600 reject ``SYSTem:LOCal/REMote/RWLock`` as an undefined header (-113,
        bench-verified 2026-07-29) - they are RS-232/GPIB remote-lockout concepts.
        There is no front-panel lockout to release over LAN; the panel is live
        and returns to local control when the session closes.
        """
        for ch in self._channels():
            with contextlib.suppress(Exception):
                self.link.write(f"OUTP{ch.n} OFF")

    def _channels(self) -> list[_Channel]:
        """Every channel this instrument actually has, in order."""
        found = [self.ch1]
        if hasattr(self, "ch2"):
            found.append(self.ch2)
        return found

    def channel(self, n: int) -> _Channel:
        """Return the channel node for ``n``."""
        return self._channels()[self._check_channel(n) - 1]

    def _check_channel(self, n: int) -> int:
        """Validate a channel number against what this model has."""
        count = len(self._channels())
        if n not in range(1, count + 1):
            raise ValueError(
                f"channel must be 1..{count} on a {self.model or 'this model'}, got {n!r}"
            )
        return n

    # ---------- both outputs at once ----------

    def set_output_state_all(self, state: bool) -> list[bool]:
        """Open or close every channel's output relay; returns each read-back.

        The relays are written one at a time, so on a two-channel instrument the
        outputs do not switch simultaneously.
        """
        for ch in self._channels():
            ch.set_output_state(state)
        return self.get_output_state_all()

    def get_output_state_all(self) -> list[bool]:
        """Whether each channel's output relay is closed, in channel order."""
        return [ch.get_output_state() for ch in self._channels()]

    # ---------- cross-channel (two-channel models) ----------

    def set_track(self, mode: Track | str) -> Track:
        """Make channel 2 copy channel 1; a :class:`Track`.

        Enum arg: <awg>.Track.<TAB>

        ON gives both channels the same signal, INVERTED flips channel 2 about
        its offset, OFF returns them to independent control. While tracking is
        on, writing to channel 2 is overridden by channel 1.
        """
        self.link.write(f"TRAC {_token(mode, Track, 'mode')}")
        return self.get_track()

    def get_track(self) -> Track:
        """Whether channel 2 copies channel 1."""
        return _member(self.link.query("TRAC?").strip(), Track, "mode")

    def set_combine_feed(self, source: Combine | str, *, base: int = 1) -> Combine:
        """Sum one channel onto the other channel's output connector.

        Enum arg: <awg>.Combine.<TAB>

        ``base`` is the connector the combined signal comes out of, and
        ``source`` is the channel summed into it. ``Combine.NONE`` separates
        them again.

        The two amplitudes add, and the sum must fit the output range: with
        both channels at 2 Vpp the instrument refuses, queues -221 "combine
        amplitude exceeds limit. Combine disabled", and leaves combining off
        (bench-observed 2026-08-11). Turn both amplitudes down first, then
        combine. The refusal is silent apart from the error queue, so check the
        returned value rather than assuming the call took.
        """
        member = _member(source, Combine, "source")
        self.link.write(f"SOUR{self._check_channel(base)}:COMB:FEED {member.value}")
        return self.get_combine_feed(base=base)

    def get_combine_feed(self, *, base: int = 1) -> Combine:
        """Which channel is summed onto the ``base`` channel's connector."""
        raw = self.link.query(f"SOUR{self._check_channel(base)}:COMB:FEED?").strip()
        return _member(raw, Combine, "source")

    # ---------- triggering (instrument-wide) ----------

    def cmd_trigger(self) -> None:
        """Issue a software (bus) trigger; requires a BUS trigger source."""
        self.link.write("*TRG")

    def cmd_abort(self) -> None:
        """Halt any running sequence/list/sweep/burst and idle the trigger system
        (ABORt). Always applies to both channels."""
        self.link.write("ABOR")

    def cmd_initiate_all(self) -> None:
        """Arm both channels so the next trigger runs their sweep, burst or list.

        Only has an effect when continuous re-arming is off.
        """
        self.link.write("INIT:IMM:ALL")

    def set_initiate_continuous_all(self, state: bool) -> list[bool]:
        """Set continuous re-arming on both channels; returns each read-back."""
        self.link.write(f"INIT:CONT:ALL {_onoff(state)}")
        return [ch.trigger.get_continuous_state() for ch in self._channels()]

    def cmd_phase_sync(self) -> None:
        """Reset all phase generators to a common zero reference.

        Establishes a known channel-to-channel phase difference equal to the sum
        of the two channels' PHASe offsets. Call after setting each channel's
        phase to make the inter-channel phase deterministic.
        """
        self.link.write("PHAS:SYNC")

    def set_angle_unit(self, unit: AngleUnit | str) -> AngleUnit:
        """Set the unit every phase argument and return uses; an :class:`AngleUnit`.

        Enum arg: <awg>.AngleUnit.<TAB>

        This is instrument-wide: it changes how ``ch.set_phase`` and the burst
        phase are read on both channels.
        """
        self.link.write(f"UNIT:ANGL {_token(unit, AngleUnit, 'unit')}")
        return self.get_angle_unit()

    def get_angle_unit(self) -> AngleUnit:
        """The unit every phase argument and return uses."""
        return _member(self.link.query("UNIT:ANGL?").strip(), AngleUnit, "unit")

    # ---------- mode ----------
    #
    # No remote()/local()/remote_lock() here. SYSTem:LOCal/REMote/RWLock are
    # RS-232/GPIB remote-lockout commands; the 33500/33600 reject them as an
    # undefined header (-113) over the LAN (VXI-11) interface this driver uses
    # (bench-verified 2026-07-29). LAN sessions are always "remote"; there is no
    # front-panel lockout to toggle.

    # ---------- system ----------

    # ---------- the 10 MHz reference timebase ----------

    def set_reference_clock_external(self) -> ReferenceState:
        """Lock the timebase to the 10 MHz reference on the rear panel input.

        Turns automatic selection off, so the choice is honoured rather than
        revisited by the instrument. Check ``in_use`` on the returned state: if
        no usable signal is present the instrument stays on its internal
        oscillator, and that is the only way to find out.
        """
        self._reference._set_auto_state(False)
        self._reference._set_source(Reference.EXTERNAL)
        return self.get_reference_clock()

    def set_reference_clock_internal(self) -> ReferenceState:
        """Lock the timebase to the internal oscillator, ignoring any input.

        Turns automatic selection off. Without that, a 10 MHz signal on the
        rear input takes over again and the request is quietly undone.
        """
        self._reference._set_auto_state(False)
        self._reference._set_source(Reference.INTERNAL)
        return self.get_reference_clock()

    def set_reference_clock_automatic(self) -> ReferenceState:
        """Let the instrument use an external 10 MHz reference whenever one appears.

        This is the power-on behaviour. Convenient, but it means the timebase
        source can change under a measurement without any indication, so the
        two explicit calls are better where the result has to be reproducible.
        """
        self._reference._set_auto_state(True)
        return self.get_reference_clock()

    def get_reference_clock(self) -> ReferenceState:
        """Which timebase is selected, which is in use, and whether it may switch."""
        return ReferenceState(
            selected=self._reference._get_source(),
            in_use=self._reference._get_in_use(),
            automatic=self._reference._get_auto_state(),
        )

    # ---------- errors ----------

    def get_errors(self) -> list[str]:
        """Drain and return the error queue; an empty list means clean.

        Reads until the queue reports no error, so one call leaves it empty.
        The queue holds up to 20 entries and then replaces the last with an
        overflow marker, so drain it often enough that nothing is lost.
        """
        errors: list[str] = []
        for _ in range(_ERROR_QUEUE_DEPTH + 1):
            raw = self.link.query("SYST:ERR?").strip()
            if not raw:
                break
            code = raw.split(",", 1)[0].strip().lstrip("+")
            if code in ("0", "-0"):
                break
            errors.append(raw)
        return errors

    def cmd_reset(self) -> None:
        """Reset to the power-on default state, then wait for completion."""
        self.link.write("*RST")
        self.link.query("*OPC?")

    def cmd_clear_status(self) -> None:
        """Clear the status byte and all event registers."""
        self.link.write("*CLS")

    def self_test(self) -> int:
        """Run the internal self-test; 0 = pass, non-zero = fail.

        Uses a long timeout - the self-test takes several seconds.
        """
        with self.link.temporary_timeout(30_000):
            return self.link.query_int("*TST?")

    def cmd_beep(self) -> None:
        """Issue a single beep."""
        self.link.write("SYST:BEEP")

    def get_scpi_version(self) -> str:
        """Query the SCPI version string."""
        return self.link.query("SYST:VERS?").strip()

    def get_options(self) -> list[str]:
        """Query the installed options as a list of tokens.

        A bare ``0`` (no options) returns an empty list.
        """
        resp = self.link.query("*OPT?").strip().strip('"')
        opts = [o.strip() for o in resp.split(",") if o.strip() and o.strip() != "0"]
        return opts

    # ---------- rear-panel Modulation In (INPut subsystem) ----------
    #
    # Underscored rather than public. This exists only on the 33600 Series: the
    # 33500 has a fixed +/-5 V modulation input with no attenuator, and answers
    # INPut:ATTenuation as an undefined header, which on VXI-11 shows up as a
    # query timeout (bench-verified 2026-07-29). A public method that raises on
    # the connected instrument would be a lie to the completion popup (design
    # standard 5.1), and there is no 33600 on this bench to verify it against,
    # so it stays in the granular tier with the model check kept.

    def _set_modulation_input_attenuation(self, enabled: bool) -> bool:
        """Set the rear Modulation In level range (INPut:ATTenuation). 33600 only.

        True accepts signals up to 5 V, False up to 1 V.
        """
        self._require_33600("modulation-input attenuation")
        self.link.write(f"INP:ATT {_onoff(enabled)}")
        return self._get_modulation_input_attenuation()

    def _get_modulation_input_attenuation(self) -> bool:
        """Query the Modulation In level range (INPut:ATTenuation?). 33600 only."""
        self._require_33600("modulation-input attenuation")
        return self.link.query_bool("INP:ATT?")

    def _require_33600(self, feature: str) -> None:
        if self.series != "33600":
            raise InstrumentError(
                f"{feature} needs a 33600-Series generator; this is a "
                f"{self.model}, which has a fixed modulation-input range."
            )

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Read back every setting this driver can change, for an exact restore.

        Queries only. It never uses ``*SAV``, which would overwrite one of the
        operator's own memory slots. Restore with :meth:`restore`, or wrap the
        work in :meth:`preserved_setup`.
        """
        snap: dict = {
            "track": self.get_track(),
            "sync_state": self.sync.get_state(),
            "sync_source": self.sync.get_source(),
            "angle_unit": self.get_angle_unit(),
            "channels": {},
        }
        for ch in self._channels():
            function = ch._get_function()
            entry = {
                "unit": ch.get_amplitude_unit(),
                "load": ch.get_load(),
                "function": function,
                "frequency": ch.get_frequency(),
                "amplitude": ch.get_amplitude(),
                "offset": ch.get_offset(),
                "phase": ch.get_phase(),
                "polarity": ch.get_polarity(),
                "gating": ch.get_gating_state(),
                "sync_mode": ch.get_sync_mode(),
                "sync_polarity": ch.get_sync_polarity(),
                "output": ch.get_output_state(),
                # The shape parameters live in per-shape registers that survive a
                # change of function, so they are restored whatever shape is up.
                "duty_square": ch._get_square_duty_cycle(),
                "symmetry": ch._get_ramp_symmetry(),
                "width": ch._get_pulse_width(),
                "edge_time": ch._get_pulse_edge_time(),
                "modulation": ch._get_modulation_states(),
            }
            snap["channels"][ch.n] = entry
        return snap

    def restore(self, snapshot: dict) -> None:
        """Restore the settings captured by :meth:`snapshot`, exactly as found.

        Ordered so no intermediate state is invalid. Tracking goes off first so
        the channels restore independently. Per channel: the unit and the load,
        which scale every voltage; then the shape; then the shape's own
        parameters; then amplitude before offset, because the valid offset range
        depends on the amplitude; then frequency; and the output relay last, so
        nothing is energized while it is being configured.
        """
        with contextlib.suppress(Exception):
            self.set_track(Track.OFF)
        for n, s in snapshot.get("channels", {}).items():
            ch = self.channel(n)
            ch.set_amplitude_unit(s["unit"])
            load = s["load"]
            ch.set_load(Load.HIGHZ if load == float("inf") else load)
            ch._set_function(s["function"])
            with contextlib.suppress(Exception):
                ch._set_square_duty_cycle(s["duty_square"])
                ch._set_ramp_symmetry(s["symmetry"])
                ch._set_pulse_width(s["width"])
                ch._set_pulse_edge_time(s["edge_time"])
            ch.set_amplitude(s["amplitude"])
            ch.set_offset(s["offset"])
            ch.set_frequency(s["frequency"])
            ch.set_phase(s["phase"])
            ch.set_polarity(s["polarity"])
            ch.set_gating_state(s["gating"])
            ch.set_sync_mode(s["sync_mode"])
            ch.set_sync_polarity(s["sync_polarity"])
            ch._restore_modulation_states(s["modulation"])
            ch.set_output_state(s["output"])
        self.set_angle_unit(snapshot["angle_unit"])
        self.set_track(snapshot["track"])
        self.sync.set_source(snapshot["sync_source"])
        self.sync.set_state(snapshot["sync_state"])

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator:
        """Snapshot the touched settings on enter and restore them on exit."""
        state = self.snapshot()
        try:
            yield self
        finally:
            self.restore(state)


def _main(argv: list[str] | None = None) -> int:
    """Live read-mostly self-test against the bench 33522A.

    Connects, prints identity/model gating/options, reads back both channels'
    output-relay states and SCPI version, drains the error queue, and returns to
    local on disconnect. Touches no output settings and toggles no relay.
    """
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE")
        return 2
    resource = argv[1]
    gen = Keysight_33500(resource)
    print(f"connecting to 33500/33600 on {gen.resource_name} ...")
    idn = gen.connect()
    try:
        print(f"  IDN: {idn}")
        print(f"  model={gen.model}  firmware={gen.firmware}  series={gen.series}")
        print(
            f"  channels={gen.num_channels}  max_freq={gen.max_frequency:g} Hz  "
            f"max_srate={gen.max_sample_rate:g} Sa/s  arb_points={gen.arb_points}"
        )
        print(f"  options={gen.options or '(none)'}")
        print(f"  SCPI version: {gen.get_scpi_version()}")
        for ch in gen._channels():
            print(f"  {ch.name} output relay: {'ON' if ch.get_output_state() else 'OFF'}")
        gen.check_errors()
        print("  error queue clean.")
    finally:
        gen.disconnect()
        print("disconnected (outputs off, local).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
