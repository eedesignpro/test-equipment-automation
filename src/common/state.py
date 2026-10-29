"""Frozen state objects returned by driver helper methods.

A helper that sets several coupled values returns one of these instead of a
bare tuple. The operator gets autocomplete on the fields. An agent that reads
the dataclass learns every field at once, with no need to read the method body.

See docs/driver-api-design.md section 4.4.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.scpi import ScpiEnum


@dataclass(frozen=True)
class TimebaseState:
    """The horizontal setting of an oscilloscope.

    Scale, record length and sample rate are locked together by
    ``div_count * scale = record_length / sample_rate``. Setting any two fixes
    the third, which is why they belong in one helper rather than three atomic
    setters.
    """

    scale: float
    """Seconds per division."""

    samplerate: float
    """Samples per second."""

    record_length: float
    """Record length in points."""

    mode: str
    """AUTO or MANUAL. AUTO discards sample-rate and record-length writes."""

    position: float
    """Trigger position as a percent of the record."""


@dataclass(frozen=True)
class WaveformState:
    """The complete output setting of a signal source.

    Every signal source on this bench returns this type: the scope's built-in
    AFG, the 33120A function generator, and the 33522A AWG. One read teaches an
    agent all three (design standard, sections 3.6 and 9.4).

    The first four fields are always populated. Every other field is None when
    it does not apply to the active function or to that instrument. A square
    wave has a duty cycle and no symmetry. A ramp has symmetry and no duty
    cycle. Additive noise exists only on the scope AFG.
    """

    function: str
    """Function token read back from the instrument, for example SINE.

    A token, not an enum, because the three instruments spell the same shapes
    differently: the 33120A calls an arbitrary waveform USER and the 33522A
    calls it ARB. Each driver's own enum governs what you may pass in.
    """

    frequency: float
    """Output frequency in Hz."""

    amplitude: float
    """Output amplitude in volts peak to peak."""

    offset: float
    """Output DC offset in volts."""

    output_state: bool | None = None
    """True when the output connector is driven.

    None on an instrument with no output relay, which is not the same as False.
    The 33120A output is always live and reports None here.
    """

    duty: float | None = None
    """Square-wave or pulse duty cycle in percent."""

    symmetry: float | None = None
    """Ramp symmetry in percent."""

    width: float | None = None
    """Pulse width in seconds."""

    edge_time: float | None = None
    """Pulse or PRBS edge transition time in seconds; the leading edge where the two differ."""

    trailing_edge_time: float | None = None
    """Pulse trailing edge in seconds, on a generator that sets the edges apart."""

    delay: float | None = None
    """Pulse lead delay in seconds, from the start of the period to the leading edge."""

    sample_rate: float | None = None
    """Arbitrary-waveform sample rate in samples per second."""

    bandwidth: float | None = None
    """Noise bandwidth in Hz."""

    noise_add: float | None = None
    """Additive noise in percent. None when additive noise is off.

    The scope AFG and the AFG31000 have it; the 33120A and the 33522A do not.
    """


@dataclass(frozen=True)
class VoltageLimits:
    """The clamp on how far a signal source's output may swing.

    The limits keep their values when the clamp is switched off, so ``low`` and
    ``high`` stay meaningful while ``enabled`` is False. A generator whose clamp
    has no off switch, such as the AFG31000, always reports True.
    """

    low: float
    """Lowest output voltage allowed, in volts."""

    high: float
    """Highest output voltage allowed, in volts."""

    enabled: bool
    """Whether the clamp is in force."""


@dataclass(frozen=True)
class ModulationState:
    """Which modulation is running on a channel, and how it is set up.

    Both bench generators run one modulation at a time, so this describes that
    one. ``mode`` is None when the carrier is unmodulated. A field is None when
    the mode does not use it, or when that generator does not have it.

    ``level`` is whichever quantity the mode uses, in that mode's own unit,
    because the modes do not share one: AM has a depth in percent, FM a
    deviation in Hz, PM a deviation in degrees, PWM a deviation in seconds, SUM
    an amplitude in percent, FSK a hop frequency in Hz, BPSK a phase in degrees.
    """

    mode: str | None
    """AM, FM, PM, PWM, SUM, FSK or BPSK; None when nothing is modulating."""

    level: float | None = None
    """The mode's own quantity, in the unit named in the class docstring."""

    rate: float | None = None
    """Modulating frequency in Hz, for an internally modulated mode."""

    shape: "ScpiEnum | None" = None
    """The internal modulating waveform, where the mode has one."""

    source: "ScpiEnum | None" = None
    """Where the modulating signal comes from."""


@dataclass(frozen=True)
class SweepState:
    """A frequency sweep, and whether it is running."""

    enabled: bool
    """Whether the sweep is running."""

    start: float
    """Start frequency in Hz."""

    stop: float
    """Stop frequency in Hz."""

    time: float
    """Time from start to stop, in seconds."""

    spacing: "ScpiEnum"
    """Whether the steps are even in frequency or in ratio."""

    hold_time: float | None = None
    """Dwell at the stop frequency, in seconds. None where unsupported.

    The 33120A has no hold or return time; the 33522A has both.
    """

    return_time: float | None = None
    """Retrace time back to the start frequency, in seconds. None where unsupported."""


@dataclass(frozen=True)
class BurstState:
    """A burst, and whether it is running."""

    enabled: bool
    """Whether burst mode is on."""

    cycles: float
    """Cycles per burst. Infinity reads back as 9.9e37 from the instrument."""

    period: float
    """Time between the start of one burst and the next, in seconds."""

    phase: float
    """Where in the waveform each burst begins, in the current angle unit."""

    mode: "ScpiEnum | None" = None
    """Whether each burst is started by a trigger or held on by a gate.

    None on a generator that does not separate the two, such as the 33120A,
    where the burst source doubles as the mode.
    """

    delay: float | None = None
    """Time from the trigger to the first cycle, in seconds. None where unsupported."""

    idle: ScpiEnum | None = None
    """What the output holds between bursts. None where the generator has no choice."""
