"""Driver for the HP/Agilent/Keysight 33120A function/arbitrary waveform generator.

Public import path::

    from drivers.keysight_33120a import Keysight_33120A

The 33120A is a single-channel 15 MHz function generator with a 12-bit / 16 k
arbitrary waveform engine. The bench unit is on a 9600-baud RS-232 UART
(``ASRL4::INSTR``) - the same HP/Agilent serial family as the E3646A supply, so
this driver reuses that module's discipline: paced I/O, a VISA timeout treated
as a fatal "power cycle me" condition (never retried), and a settle after mode
changes (see the deaf-window note below).

Structure mirrors the scope's built-in AFG (``Tektronix_MSO.afg``): the core
waveform configuration lives as flat ``set_/get_`` pairs on the instrument, with
modulation / burst / sweep / arbitrary / status grouped into subsystem nodes.
The 33120A has NO output on/off relay - the output is always live - so there is
deliberately no ``output_on/off``; ``safe_state`` only returns to local and
leaves the last waveform running (see ADR / plan iteration 5).

Command surface follows the Agilent 33120A User's Guide, chapter 4.

Common tasks:
  1. Put a 1 kHz 2 Vpp sine on the output:
         siggen.set_waveform_sine(1e3, 2.0)
  2. Step frequency without disturbing anything else:
         for hz in (1e3, 2e3, 5e3):
             siggen.set_frequency(hz)
  3. Amplitude-modulate, then go back to a plain carrier:
         siggen.set_am(50, rate=1e3)
         siggen.set_modulation_off()
  4. Send a burst of five cycles on each trigger:
         siggen.set_burst(5, rate=100)
  5. Tell it what the cable terminates into, so the amplitude it
     reports matches what the load sees:
         siggen.set_load(siggen.Load.HIGHZ)
"""

from __future__ import annotations

import contextlib

from collections.abc import Iterator, Sequence
import re
import time

import numpy as np
import pyvisa as visa

from common.errors import InstrumentError, InstrumentUnresponsiveError
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import ScpiEnum, quote
from common.state import BurstState, ModulationState, SweepState, WaveformState

# Inter-command spacing for the 9600-baud UART. Matches the evidenced-safe floor
# established for the E3646A on the same link family (ADR-022/023): 50 ms held
# through sustained write bursts; 10 ms overflowed and hung the instrument.
_DEFAULT_PACING_S = 0.05

# Generous connect/IO timeout - the UART is slow and must never be hammered.
_DEFAULT_TIMEOUT_MS = 5000

# Settle after a SYST:REM/LOC/RWL mode change. Bench-measured 2026-07-27 on the
# ASRL4 33120A (fw 8.0-5.0-1.0): unlike the E3646A, SYST:REM has NO desyncing
# deaf-window - a *IDN? issued with 0.0 s settle never times out, it just blocks
# ~124 ms until the instrument answers; by 0.05 s the read is the bare 72 ms
# 9600-baud round-trip. The legacy driver's 0.5 s sleep was overly conservative.
# 0.1 s gives comfortable margin over the 0.05 s ready point and protects a
# write issued immediately after a mode change.
_REMOTE_SETTLE_S = 0.1

# After a STATE-CHANGING write the 33120A is deaf to queries while it re-ranges
# the DAC/attenuator/output; a readback issued too soon is DROPPED (never
# answered) -> the read times out AND desyncs the serial stream. Bench-measured
# 2026-07-27 on the ASRL4 unit, per-transition first-OK settle: same-range and
# most changes clear at 0.15 s; a function change (SQU->TRI) needs 0.20 s; a
# large amplitude step that crosses an attenuator range DOWN (e.g. 5->0.05 Vpp)
# needs 0.30 s - the worst case observed. 0.40 s covers all of them with margin.
# The window only appears when the write actually CHANGES a setting (a no-op
# write has none), so OUTP:LOAD/SYNC and VOLT:UNIT settle too since they change
# output scaling. Queries with no preceding write do not wait.
_RECONFIG_SETTLE_S = 0.4

# Non-volatile arb operations (DATA:COPY/DEL/DEL:ALL) write flash and stay deaf
# well beyond the normal reconfig window; a query issued too soon times out.
_ARB_NVRAM_SETTLE_S = 2.0

_IDN_RE = re.compile(r"33120A", re.IGNORECASE)

# FUNC:SHAP accepts these; the query returns the short token (left column value).
_FUNCTION = {
    "SIN": "SIN",
    "SINE": "SIN",
    "SINUSOID": "SIN",
    "SQU": "SQU",
    "SQUARE": "SQU",
    "TRI": "TRI",
    "TRIANGLE": "TRI",
    "RAMP": "RAMP",
    "NOIS": "NOIS",
    "NOISE": "NOIS",
    "DC": "DC",
    "USER": "USER",
}

_UNIT = {"VPP": "VPP", "VRMS": "VRMS", "DBM": "DBM"}

# OUTP:LOAD takes 50 or INF; FIFTY/HIGHZ tokens mirror the scope AFG vocabulary.
_LOAD = {"FIFTY": "50", "50": "50", "HIGHZ": "INF", "INF": "INF", "INFINITY": "INF"}

# Internal modulating-waveform shapes (AM/FM); note DC is not valid as a modulator.
_MOD_FUNCTION = {
    "SIN": "SIN",
    "SINE": "SIN",
    "SQU": "SQU",
    "SQUARE": "SQU",
    "TRI": "TRI",
    "TRIANGLE": "TRI",
    "RAMP": "RAMP",
    "NOIS": "NOIS",
    "NOISE": "NOIS",
    "USER": "USER",
}
_AM_SOURCE = {"BOTH": "BOTH", "EXT": "EXT", "EXTERNAL": "EXT"}
_INT_EXT = {"INT": "INT", "INTERNAL": "INT", "EXT": "EXT", "EXTERNAL": "EXT"}
_TRIG_SOURCE = {"IMM": "IMM", "IMMEDIATE": "IMM", "EXT": "EXT", "EXTERNAL": "EXT", "BUS": "BUS"}
_TRIG_SLOPE = {"POS": "POS", "POSITIVE": "POS", "NEG": "NEG", "NEGATIVE": "NEG"}
_SWEEP_SPACING = {"LIN": "LIN", "LINEAR": "LIN", "LOG": "LOG", "LOGARITHMIC": "LOG"}


class Load(ScpiEnum):
    """What the far end of the cable terminates into.

    This does not change the 50 ohm source impedance. It tells the generator
    what to assume, so the amplitude it displays matches what the load sees.
    """

    FIFTY = "50"
    HIGHZ = "INF"


class Spacing(ScpiEnum):
    """How a sweep steps from its start frequency to its stop frequency."""

    LINEAR = "LIN"
    LOGARITHMIC = "LOG"


class ModulationShape(ScpiEnum):
    """The internal modulating waveform for AM and FM.

    DC is absent because this instrument rejects it as a modulator.
    """

    SINE = "SIN"
    SQUARE = "SQU"
    TRIANGLE = "TRI"
    RAMP = "RAMP"
    NOISE = "NOIS"
    ARBITRARY = "USER"


def _token(value, enum_cls: type[ScpiEnum], label: str) -> str:
    """Return the SCPI token for an enum member, a member name, or a token string."""
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    for member in enum_cls:
        if text in (member.name, str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")


def _member(value, enum_cls: type[ScpiEnum], label: str):
    """Return the enum member for a member, a member name, or a token."""
    return enum_cls(_token(value, enum_cls, label))


def _onoff(state: bool) -> str:
    return "ON" if state else "OFF"


def _num(value: float | str) -> str:
    """Format a numeric SCPI argument: a token (MIN/MAX/DEF) or a number."""
    if isinstance(value, str):
        return value
    return f"{value:.12g}"


class _SigGenSerialLink(InstrumentLink):
    """InstrumentLink configured for the 33120A RS-232 UART.

    Opens the serial resource with fixed line settings (9600/8/N/1, ``\\n``
    terminators) WITHOUT a VISA ``clear`` (gentle first contact). 8 data bits /
    no parity is also required for binary ``DATA:DAC`` transfers. Any query or
    write that times out is translated to :class:`InstrumentUnresponsiveError`
    so callers stop rather than retry a hung link.

    This link also owns the inter-command pacing. The delay is a property of
    THIS generator on THIS UART, not generic link behavior, so it lives here and
    not in :class:`InstrumentLink` (ADR-033 supersedes ADR-022).
    """

    baud_rate = 9600

    def __init__(
        self,
        resource_name: str,
        timeout_ms: int = 3000,
        min_command_interval: float = 0.0,
    ):
        """``min_command_interval`` is the minimum spacing, in seconds, held
        between successive writes/queries. ``0.0`` disables pacing."""
        super().__init__(resource_name, timeout_ms=timeout_ms)
        self.min_command_interval = min_command_interval
        self._last_io = 0.0

    def _pace(self) -> None:
        """Sleep so successive I/O is at least ``min_command_interval`` apart."""
        if self.min_command_interval > 0:
            wait = self.min_command_interval - (time.monotonic() - self._last_io)
            if wait > 0:
                time.sleep(wait)
        self._last_io = time.monotonic()

    def connect(self, timeout_ms: int | None = None) -> None:
        """Open the serial port with the generator's fixed line settings."""
        rm = visa.ResourceManager()
        self._instr = self._open_resource(rm, self.resource_name)
        self.configure_serial()
        # Device-clear on open so a link left desynced by a prior session (e.g. a
        # dropped readback during a reconfig deaf-window) self-heals. The 33120A
        # is robust to this (its manual equates the RS-232 clear to an IEEE-488
        # device clear); it flushes stale bytes before the first SYST:REM/IDN.
        with contextlib.suppress(Exception):
            self._instr.clear()
        self._instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms

    def configure_serial(self) -> None:
        """Apply the fixed serial line settings to the open resource."""
        instr = self._instr
        with contextlib.suppress(Exception):
            instr.baud_rate = self.baud_rate
            instr.data_bits = 8
            instr.parity = visa.constants.Parity.none
            instr.stop_bits = visa.constants.StopBits.one
        # DTR/DSR hardware flow control - the 33120A's documented RS-232
        # handshake. It holds DTR false when its small input buffer fills, which
        # is essential for long DATA/DATA:DAC arbitrary-waveform uploads: without
        # it the instrument drops bytes mid-command (corrupt upload / +521
        # overflow). Guarded so a backend without serial flow control still opens.
        with contextlib.suppress(Exception):
            instr.flow_control = visa.constants.ControlFlow.dtr_dsr
        instr.write_termination = "\n"
        instr.read_termination = "\n"

    def query(self, message: str, **kwargs) -> str:
        """Send a query, paced, translating a timeout to a stop."""
        self._pace()
        try:
            return super().query(message, **kwargs)
        except visa.VisaIOError as exc:
            raise self._translate_timeout(message, exc) from exc

    def write(self, message: str, **kwargs) -> int:
        """Send a command, paced for the 9600-baud UART."""
        self._pace()
        try:
            return super().write(message, **kwargs)
        except visa.VisaIOError as exc:
            raise self._translate_timeout(message, exc) from exc

    @staticmethod
    def _translate_timeout(message: str, exc: visa.VisaIOError):
        if exc.error_code == visa.constants.StatusCode.error_timeout:
            return InstrumentUnresponsiveError(
                f"33120A did not respond to {message!r} (VISA timeout). Stop all "
                "work - the operator must power cycle the function generator."
            )
        return exc


class Keysight_33120A(Instrument):
    """HP/Agilent/Keysight 33120A function / arbitrary waveform generator.

    The subsystem tree is built in ``__init__`` so attribute autocompletion
    works before connecting. ``connect()`` opens the pinned serial port, enters
    remote mode (required before any RS-232 traffic on this instrument), settles
    the deaf-window, then reads and validates ``*IDN?``.
    """

    Load = Load
    Spacing = Spacing
    ModulationShape = ModulationShape

    def __init__(
        self,
        resource_name: str,
        *,
        pacing_s: float = _DEFAULT_PACING_S,
        connect_timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        """
        Parameters
        ----------
        resource_name : str
            pyvisa resource for the generator (default ``ASRL4::INSTR``).
        pacing_s : float
            minimum inter-command spacing in seconds (UART overflow guard).
        connect_timeout_ms : int
            I/O timeout used for connect and normal operation.
        """
        super().__init__(resource_name, name="SIGGEN")
        # Replace the default link with a paced serial link.
        self.link = _SigGenSerialLink(
            resource_name,
            timeout_ms=connect_timeout_ms,
            min_command_interval=pacing_s,
        )
        Node.__init__(self, self.link, name="SIGGEN")
        self.resource_name = resource_name
        self._connect_timeout_ms = connect_timeout_ms

        self.model: str | None = None
        self.firmware: str | None = None

        # Subsystem nodes (modulation / burst / sweep / trigger / status / cal).
        self.am = _AM(self.link, name="AM")
        self.fm = _FM(self.link, name="FM")
        self.fsk = _FSK(self.link, name="FSK")
        self.burst = _Burst(self.link, name="BURST")
        self.sweep = _Sweep(self.link, name="SWEEP")
        self.trigger = _Trigger(self.link, name="TRIGGER")
        self.status = _Status(self.link, name="STATUS")
        self.arb = _Arb(self.link, name="ARB")
        self.cal = _Calibration(self.link, name="CALIBRATION")

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the port, enter remote mode, settle, then read and check the identity.

        The 33120A manual requires SYST:REM before RS-232 traffic ("unpredictable
        results" otherwise), so remote mode is entered BEFORE the first query. A
        VISA timeout on *IDN? is fatal (:class:`InstrumentUnresponsiveError`) -
        the operator must power cycle; the link is never retry-hammered.
        """
        to = self._connect_timeout_ms if timeout_ms is None else timeout_ms
        self.link.connect(timeout_ms=to)
        self.link.write("SYST:REM")
        time.sleep(_REMOTE_SETTLE_S)
        # Clear the status/error queue so a session starts clean - the SCPI error
        # queue survives a VISA device-clear and *RST (only *CLS or power-off
        # drains it), so stale entries from a prior desync would otherwise leak.
        self.link.write("*CLS")
        idn = self.link.query("*IDN?").strip()
        if not _IDN_RE.search(idn):
            self.link.disconnect()
            raise InstrumentError(f"Not a 33120A (IDN={idn!r}) on {self.resource_name}")
        self.idn = idn
        self._on_connect(idn)
        return idn

    def _on_connect(self, idn: str) -> None:
        """Parse model and firmware from the IDN string."""
        fields = [f.strip() for f in idn.split(",")]
        self.model = fields[1] if len(fields) >= 2 else None
        self.firmware = fields[3] if len(fields) >= 4 else None

    def safe_state(self) -> None:
        """Return to local mode before the link closes.

        The 33120A has no output relay, so this does NOT de-energize the output
        (the last waveform keeps running) - a deliberate, documented departure
        from the "de-energize on disconnect" default, justified for a <=10 Vpp
        function generator. Use :meth:`preserved_setup` to restore as-found
        configuration around a block of work.
        """
        with contextlib.suppress(Exception):
            self.link.write("SYST:LOC")

    # ---------- mode ----------

    def cmd_remote(self) -> None:
        """Enter remote mode, then settle the deaf-window."""
        self.link.write("SYST:REM")
        time.sleep(_REMOTE_SETTLE_S)

    def cmd_local(self) -> None:
        """Return to local mode, then settle the deaf-window."""
        self.link.write("SYST:LOC")
        time.sleep(_REMOTE_SETTLE_S)

    def cmd_remote_lock(self) -> None:
        """Enter remote with the front-panel Local key locked, then settle."""
        self.link.write("SYST:RWL")
        time.sleep(_REMOTE_SETTLE_S)

    # ---------- core waveform configuration (mirrors Tektronix_MSO.afg) ----------

    def _write_reconfig(self, cmd: str) -> None:
        """Write an output-reconfiguring command, then wait out the deaf-window.

        See :data:`_RECONFIG_SETTLE_S`: a readback issued too soon after such a
        write is dropped and desyncs the link, so every reconfiguring setter
        settles before its verifying query.
        """
        self.link.write(cmd)
        time.sleep(_RECONFIG_SETTLE_S)

    def _set_function(self, function: str) -> str:
        """Set the output function shape.

        ### Parameters
        1. function : str
            - ['SINE', 'SQUARE', 'TRIANGLE', 'RAMP', 'NOISE', 'DC', 'USER']
        ### Returns
        1. function : str
            - the function token read back (SIN/SQU/TRI/RAMP/NOIS/DC/USER)
        """
        self.link.write_choice("FUNC:SHAP", function.upper(), _FUNCTION)
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_function()

    def get_function(self) -> str:
        """Query the output function token (SIN/SQU/TRI/RAMP/NOIS/DC/USER)."""
        return self.link.query("FUNC:SHAP?").strip()

    def set_frequency(self, frequency: float | str) -> float:
        """Set the output frequency in Hz (accepts MIN/MAX). Returns the read-back."""
        self._write_reconfig(f"FREQ {_num(frequency)}")
        return self.get_frequency()

    def get_frequency(self, limit: str | None = None) -> float:
        """Query the output frequency in Hz; ``limit`` in {MIN, MAX}."""
        cmd = "FREQ?" if limit is None else f"FREQ? {limit}"
        return self.link.query_float(cmd)

    def set_amplitude(self, amplitude: float | str) -> float:
        """Set the output amplitude in the current unit (accepts MIN/MAX).

        Amplitude is interpreted in the units set by :meth:`set_amplitude_unit`
        (default Vpp) and depends on the load setting (see :meth:`set_load`).
        """
        self._write_reconfig(f"VOLT {_num(amplitude)}")
        return self.get_amplitude()

    def get_amplitude(self, limit: str | None = None) -> float:
        """Query the output amplitude in the current unit; ``limit`` in {MIN, MAX}."""
        cmd = "VOLT?" if limit is None else f"VOLT? {limit}"
        return self.link.query_float(cmd)

    def set_offset(self, offset: float | str) -> float:
        """Set the DC offset in volts (accepts MIN/MAX). Returns the read-back.

        NOTE: the valid offset range is coupled to the amplitude by the
        instrument's DAC+offset->attenuator architecture, so set amplitude
        FIRST; an out-of-range offset is auto-clamped by the instrument (a -221
        "settings conflict" is queued and surfaced by :meth:`check_errors`).
        """
        self._write_reconfig(f"VOLT:OFFS {_num(offset)}")
        return self.get_offset()

    def get_offset(self, limit: str | None = None) -> float:
        """Query the DC offset in volts; ``limit`` in {MIN, MAX}."""
        cmd = "VOLT:OFFS?" if limit is None else f"VOLT:OFFS? {limit}"
        return self.link.query_float(cmd)

    def set_amplitude_unit(self, unit: str) -> str:
        """Set the amplitude unit (VPP/VRMS/DBM). Returns the read-back token."""
        self.link.write_choice("VOLT:UNIT", unit.upper(), _UNIT)
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_amplitude_unit()

    def get_amplitude_unit(self) -> str:
        """Query the amplitude unit token (VPP/VRMS/DBM)."""
        return self.link.query("VOLT:UNIT?").strip()

    def set_duty_cycle(self, percent: float | str) -> float:
        """Set the square-wave duty cycle in percent (accepts MIN/MAX).

        Valid 20-80% below 5 MHz, 40-60% above; the instrument clamps and queues
        a -221 if out of range for the present frequency.
        """
        self._write_reconfig(f"PULS:DCYC {_num(percent)}")
        return self.get_duty_cycle()

    def get_duty_cycle(self, limit: str | None = None) -> float:
        """Query the square-wave duty cycle in percent; ``limit`` in {MIN, MAX}."""
        cmd = "PULS:DCYC?" if limit is None else f"PULS:DCYC? {limit}"
        return self.link.query_float(cmd)

    def set_load(self, load: str) -> str:
        """Set the assumed output termination (FIFTY/HIGHZ, or 50/INF).

        This does not change the 50 ohm source impedance; it tells the generator
        what the far end is terminated into so the displayed amplitude/offset are
        correct. Returns the normalized read-back token (FIFTY or HIGHZ).
        """
        self.link.write_choice("OUTP:LOAD", str(load).upper(), _LOAD)
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_load()

    def get_load(self) -> str:
        """Query the assumed load termination, normalized to FIFTY or HIGHZ."""
        return "HIGHZ" if self.link.query_float("OUTP:LOAD?") > 1e30 else "FIFTY"

    def set_sync_state(self, state: bool) -> bool:
        """Enable/disable the front/rear SYNC output.

        The SYNC signal drives scope-trigger setups; its level is independent of
        the main output amplitude. Returns the read-back state.
        """
        self._write_reconfig(f"OUTP:SYNC {_onoff(state)}")
        return self.get_sync_state()

    def get_sync_state(self) -> bool:
        """Query whether the SYNC output is enabled."""
        return self.link.query_bool("OUTP:SYNC?")

    # ---------- the waveform, as one call per shape (design standard 8) ----------

    def set_waveform_sine(self, freq: float, ampl: float, *, offset: float = 0.0) -> WaveformState:
        """Output a sine wave at ``freq`` Hz and ``ampl`` volts peak to peak."""
        self._write_reconfig(f"APPL:SIN {_num(freq)},{_num(ampl)},{_num(offset)}")
        return self.get_waveform()

    def set_waveform_square(
        self, freq: float, ampl: float, *, offset: float = 0.0, duty: float | None = None
    ) -> WaveformState:
        """Output a square wave; ``duty`` is the high time as a percent.

        Leaving ``duty`` at None gives 50 percent, because the one-shot command
        underneath resets it.
        """
        self._write_reconfig(f"APPL:SQU {_num(freq)},{_num(ampl)},{_num(offset)}")
        if duty is not None:
            self.set_duty_cycle(duty)
        return self.get_waveform()

    def set_waveform_triangle(
        self, freq: float, ampl: float, *, offset: float = 0.0
    ) -> WaveformState:
        """Output a triangle wave at ``freq`` Hz and ``ampl`` volts peak to peak."""
        self._write_reconfig(f"APPL:TRI {_num(freq)},{_num(ampl)},{_num(offset)}")
        return self.get_waveform()

    def set_waveform_ramp(self, freq: float, ampl: float, *, offset: float = 0.0) -> WaveformState:
        """Output a ramp at ``freq`` Hz and ``ampl`` volts peak to peak."""
        self._write_reconfig(f"APPL:RAMP {_num(freq)},{_num(ampl)},{_num(offset)}")
        return self.get_waveform()

    def set_waveform_noise(self, ampl: float, *, offset: float = 0.0) -> WaveformState:
        """Output noise at ``ampl`` volts peak to peak. Noise has no frequency."""
        self._write_reconfig(f"APPL:NOIS DEF,{_num(ampl)},{_num(offset)}")
        return self.get_waveform()

    def set_waveform_dc(self, offset: float) -> WaveformState:
        """Output a steady level of ``offset`` volts."""
        self._write_reconfig(f"APPL:DC DEF,DEF,{_num(offset)}")
        return self.get_waveform()

    def set_waveform_arb(
        self, samples: Sequence[float] | None, freq: float, ampl: float, *, offset: float = 0.0
    ) -> WaveformState:
        """Upload ``samples`` and output them at ``freq`` Hz and ``ampl`` volts.

        ``samples`` is one period, as a sequence of floats from -1 to +1. Pass
        None to output the arbitrary waveform the instrument already holds.
        """
        if samples is not None:
            self.arb.set_data_floats(samples)
        self._write_reconfig(f"APPL:USER {_num(freq)},{_num(ampl)},{_num(offset)}")
        return self.get_waveform()

    def get_waveform(self) -> WaveformState:
        """Read the shape, frequency, amplitude, offset and duty now being output."""
        function, frequency, amplitude, offset = self._get_apply()
        duty = self.get_duty_cycle() if function.upper().startswith("SQU") else None
        # output_state stays None: this generator has no output relay, which is
        # not the same as a relay that is open.
        return WaveformState(
            function=function,
            frequency=frequency,
            amplitude=amplitude,
            offset=offset,
            duty=duty,
        )

    # ---------- modulation, one call each (design standard 4.3) ----------

    def set_am(
        self,
        depth: float,
        *,
        rate: float | None = None,
        shape: ModulationShape | str | None = None,
    ) -> ModulationState:
        """Amplitude-modulate the carrier to ``depth`` percent and turn AM on.

        Enum arg: <siggen>.ModulationShape.<TAB>

        ``rate`` is the modulating frequency in Hz and ``shape`` its waveform.
        Any other modulation is turned off first, because this instrument runs
        one at a time.
        """
        self.set_modulation_off()
        if shape is not None:
            self.am.set_internal_function(_token(shape, ModulationShape, "shape"))
        if rate is not None:
            self.am.set_internal_frequency(rate)
        self.am.set_depth(depth)
        self.am.set_state(True)
        return self.get_modulation()

    def set_fm(
        self,
        deviation: float,
        *,
        rate: float | None = None,
        shape: ModulationShape | str | None = None,
    ) -> ModulationState:
        """Frequency-modulate the carrier by ``deviation`` Hz and turn FM on.

        Enum arg: <siggen>.ModulationShape.<TAB>
        """
        self.set_modulation_off()
        if shape is not None:
            self.fm.set_internal_function(_token(shape, ModulationShape, "shape"))
        if rate is not None:
            self.fm.set_internal_frequency(rate)
        self.fm.set_deviation(deviation)
        self.fm.set_state(True)
        return self.get_modulation()

    def set_burst(
        self, count: int, *, rate: float | None = None, phase: float | None = None
    ) -> BurstState:
        """Output bursts of ``count`` cycles and turn burst mode on.

        ``rate`` is how often a burst starts, in Hz, and ``phase`` the starting
        phase in degrees.
        """
        self.set_modulation_off()
        if rate is not None:
            self.burst.set_rate(rate)
        if phase is not None:
            self.burst.set_phase(phase)
        self.burst.set_count(count)
        self.burst.set_state(True)
        return self.get_burst()

    def set_sweep(
        self, start: float, stop: float, time: float, *, spacing: Spacing | str | None = None
    ) -> SweepState:
        """Sweep from ``start`` to ``stop`` Hz over ``time`` seconds and turn sweep on.

        Enum arg: <siggen>.Spacing.<TAB>
        """
        self.set_modulation_off()
        self.sweep.set_start_frequency(start)
        self.sweep.set_stop_frequency(stop)
        self.sweep.set_time(time)
        if spacing is not None:
            self.sweep.set_spacing(_token(spacing, Spacing, "spacing"))
        self.sweep.set_state(True)
        return self.get_sweep()

    def set_modulation_off(self) -> ModulationState:
        """Turn off every modulation mode, leaving a plain carrier."""
        for node in (self.am, self.fm, self.fsk, self.burst, self.sweep):
            node.set_state(False)
        return self.get_modulation()

    def get_modulation(self) -> ModulationState:
        """Which modulation is running, and its settings.

        ``mode`` is None when the carrier is unmodulated. ``level`` carries
        whichever quantity the mode uses: AM a depth in percent, FM a deviation
        in Hz, FSK a hop frequency in Hz.
        """
        if self.am.get_state():
            return ModulationState(
                mode="AM",
                level=self.am.get_depth(),
                rate=self.am.get_internal_frequency(),
                shape=_member(self.am.get_internal_function(), ModulationShape, "shape"),
            )
        if self.fm.get_state():
            return ModulationState(
                mode="FM",
                level=self.fm.get_deviation(),
                rate=self.fm.get_internal_frequency(),
                shape=_member(self.fm.get_internal_function(), ModulationShape, "shape"),
            )
        if self.fsk.get_state():
            return ModulationState(
                mode="FSK",
                level=self.fsk.get_hop_frequency(),
                rate=self.fsk.get_rate(),
            )
        return ModulationState(mode=None)

    def get_burst(self) -> BurstState:
        """The burst configuration, and whether it is running.

        ``period`` is derived from the burst rate, which is how this generator
        expresses it. ``mode`` is None: the 33120A does not separate triggered
        from gated, its burst source doubles as the mode.
        """
        rate = self.burst.get_rate()
        return BurstState(
            enabled=self.burst.get_state(),
            cycles=self.burst.get_count(),
            period=1.0 / rate if rate else float("inf"),
            phase=self.burst.get_phase(),
        )

    def get_sweep(self) -> SweepState:
        """The frequency sweep, and whether it is running.

        ``hold_time`` and ``return_time`` are None: this generator has neither.
        """
        return SweepState(
            enabled=self.sweep.get_state(),
            start=self.sweep.get_start_frequency(),
            stop=self.sweep.get_stop_frequency(),
            time=self.sweep.get_time(),
            spacing=_member(self.sweep.get_spacing(), Spacing, "spacing"),
        )

    def cmd_trigger(self) -> None:
        """Send one trigger, for burst and sweep modes that wait for one."""
        self.link.write("*TRG")

    def _get_apply(self) -> tuple:
        """Query the present configuration (APPL?).

        Returns ``(function, frequency, amplitude, offset)`` where function is a
        token (e.g. ``"SIN"``) and the rest are floats.
        """
        resp = self.link.query("APPL?").strip().strip('"')
        func, _, rest = resp.partition(" ")
        parts = [p for p in rest.split(",") if p.strip()]
        freq = float(parts[0]) if len(parts) >= 1 else float("nan")
        amp = float(parts[1]) if len(parts) >= 2 else float("nan")
        off = float(parts[2]) if len(parts) >= 3 else float("nan")
        return func, freq, amp, off

    # ---------- computed conveniences (no direct 33120A SCPI) ----------

    def set_period(self, period: float) -> float:
        """Set the output period in seconds (computed: writes FREQ = 1/period)."""
        self.set_frequency(1.0 / period)
        return self.get_period()

    def get_period(self) -> float:
        """Query the output period in seconds (computed: 1/frequency)."""
        return 1.0 / self.get_frequency()

    def get_high_level(self) -> float:
        """Query the waveform high level in volts (computed: offset + amplitude/2)."""
        return self.get_offset() + self.get_amplitude() / 2.0

    def get_low_level(self) -> float:
        """Query the waveform low level in volts (computed: offset - amplitude/2)."""
        return self.get_offset() - self.get_amplitude() / 2.0

    def set_high_level(self, level: float) -> float:
        """Set the waveform high level in volts (computed via amplitude+offset)."""
        self._set_high_low(level, self.get_low_level())
        return self.get_high_level()

    def set_low_level(self, level: float) -> float:
        """Set the waveform low level in volts (computed via amplitude+offset)."""
        self._set_high_low(self.get_high_level(), level)
        return self.get_low_level()

    def _set_high_low(self, high: float, low: float) -> None:
        # Amplitude before offset: the valid offset range depends on amplitude.
        self.set_amplitude(high - low)
        self.set_offset((high + low) / 2.0)

    # ---------- system ----------

    def cmd_reset(self) -> None:
        """Reset to the power-on default state."""
        self.link.write("*RST")
        time.sleep(_REMOTE_SETTLE_S)

    def self_test(self) -> int:
        """Run the self-test; 0 = pass, non-zero = fail."""
        return self.link.query_int("*TST?")

    def cmd_beep(self) -> None:
        """Issue a single beep."""
        self.link.write("SYST:BEEP")

    def get_scpi_version(self) -> str:
        """Query the SCPI version string."""
        return self.link.query("SYST:VERS?").strip()

    def set_display_state(self, state: bool) -> bool:
        """Turn the front-panel display on/off. Returns the read-back."""
        self.link.write(f"DISP {_onoff(state)}")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_display_state()

    def get_display_state(self) -> bool:
        """Whether the front-panel display is on."""
        return self.link.query_bool("DISP?")

    def set_display_text(self, text: str) -> str:
        """Show a message (<=11 chars shown) on the front panel."""
        self.link.write(f"DISP:TEXT {quote(text)}")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_display_text()

    def get_display_text(self) -> str:
        """The message currently shown on the front panel."""
        return self.link.query("DISP:TEXT?").strip().strip('"')

    def cmd_clear_display_text(self) -> None:
        """Clear the front-panel message."""
        self.link.write("DISP:TEXT:CLE")
        time.sleep(_RECONFIG_SETTLE_S)

    # ---------- state storage (memory locations 0-3) ----------

    def cmd_save_state(self, location: int) -> None:
        """Save the present state to memory location 0-3."""
        self._check_location(location)
        self.link.write(f"*SAV {location}")
        time.sleep(_RECONFIG_SETTLE_S)

    def cmd_recall_state(self, location: int) -> None:
        """Recall a stored state from location 0-3."""
        self._check_location(location)
        self.link.write(f"*RCL {location}")
        time.sleep(_RECONFIG_SETTLE_S)

    def cmd_delete_state(self, location: int) -> None:
        """Delete a stored state and clear the location."""
        self._check_location(location)
        self.link.write(f"MEM:STAT:DEL {location}")
        time.sleep(_RECONFIG_SETTLE_S)

    @staticmethod
    def _check_location(location: int) -> None:
        if location not in (0, 1, 2, 3):
            raise ValueError(f"state location must be 0-3, got {location!r}")

    # ---------- arbitrary waveform (legacy-compatible top-level wrapper) ----------

    def _upload_waveform(self, data) -> int:
        """Upload ``data`` to VOLATILE, select it, and output it (legacy helper).

        Auto-scales the samples, downloads them as floats, selects VOLATILE and
        switches the output to the arbitrary function. Returns the point count.
        Use the ``arb`` subsystem directly for binary upload or non-volatile
        storage.
        """
        n = self.arb.upload_floats(data)
        self.arb.select("VOLATILE")
        self.arb.output()
        return n

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Read back the touched output settings so they can be restored later.

        Captures the amplitude unit and load (which govern how the amplitude and
        offset values are interpreted), the function, and the frequency,
        amplitude, offset, duty cycle and SYNC state. Read-back queries only (no
        *SAV, which would clobber a user memory slot). Restore with
        :meth:`restore` or the :meth:`preserved_setup` context manager.
        """
        return {
            "unit": self.get_amplitude_unit(),
            "load": self.get_load(),
            "function": self.get_function(),
            "frequency": self.get_frequency(),
            "amplitude": self.get_amplitude(),
            "offset": self.get_offset(),
            "duty": self.get_duty_cycle(),
            "sync": self.get_sync_state(),
        }

    def restore(self, snapshot: dict) -> None:
        """Restore settings captured by :meth:`snapshot`.

        Order matters: unit and load first (they scale amplitude/offset), then
        function, then amplitude before offset (the valid offset range depends on
        amplitude), then frequency, duty and SYNC.
        """
        self.set_amplitude_unit(snapshot["unit"])
        self.set_load(snapshot["load"])
        self._set_function(snapshot["function"])
        self.set_amplitude(snapshot["amplitude"])
        self.set_offset(snapshot["offset"])
        self.set_frequency(snapshot["frequency"])
        self.set_duty_cycle(snapshot["duty"])
        self.set_sync_state(snapshot["sync"])

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator:
        """Snapshot the touched settings on enter and restore them on exit."""
        state = self.snapshot()
        try:
            yield self
        finally:
            self.restore(state)


class _SigGenNode(Node):
    """Base for 33120A subsystem nodes: adds reconfig-settle write+readback.

    Any state-changing write inherits the ~90-300 ms deaf-window (see
    :data:`_RECONFIG_SETTLE_S`), so each mutating setter settles before its
    verifying query.
    """

    def _set_num(self, cmd: str, value: float | str, query: str) -> float:
        self.link.write(f"{cmd} {_num(value)}")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.link.query_float(query)

    def _set_bool(self, cmd: str, state: bool, query: str) -> bool:
        self.link.write(f"{cmd} {_onoff(state)}")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.link.query_bool(query)

    def _set_choice(self, cmd: str, value: str, mapping: dict[str, str], query: str) -> str:
        self.link.write_choice(cmd, value.upper(), mapping)
        time.sleep(_RECONFIG_SETTLE_S)
        return self.link.query(query).strip()


class _AM(_SigGenNode):
    """Amplitude-modulation subsystem (AM group).

    SYNC NOTE (bench-verified 2026-07-27): enabling AM does NOT disable the SYNC
    output - OUTP:SYNC stays on - but the SYNC terminal RETARGETS from the
    carrier rate to the MODULATING rate (e.g. 10 kHz carrier -> 200 Hz sync at
    AM:INT:FREQ 200), and snaps back when AM is turned off. A scope triggered on
    SYNC for the carrier will look dead; trigger on the modulation rate, or on
    the main output. The 33120A offers only SOUR BOTH or EXT (no internal-only);
    BOTH sums the rear-panel external AM input, so a floating/undriven external
    input can suppress the internal envelope.
    """

    def set_state(self, state: bool) -> bool:
        """Enable/disable AM. Set up other AM params first."""
        return self._set_bool("AM:STAT", state, "AM:STAT?")

    def get_state(self) -> bool:
        """Whether amplitude modulation is running."""
        return self.query_bool("AM:STAT?")

    def set_depth(self, percent: float | str) -> float:
        """Set the modulation depth in percent, 0-120."""
        return self._set_num("AM:DEPT", percent, "AM:DEPT?")

    def get_depth(self, limit: str | None = None) -> float:
        """The modulation depth in percent; ``limit`` in {MIN, MAX}."""
        return self.query_float("AM:DEPT?" if limit is None else f"AM:DEPT? {limit}")

    def set_internal_frequency(self, frequency: float | str) -> float:
        """Set the internal modulating frequency in Hz, 10 mHz-20 kHz."""
        return self._set_num("AM:INT:FREQ", frequency, "AM:INT:FREQ?")

    def get_internal_frequency(self, limit: str | None = None) -> float:
        """The internal modulating frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float("AM:INT:FREQ?" if limit is None else f"AM:INT:FREQ? {limit}")

    def set_internal_function(self, function: str) -> str:
        """Set the modulating waveshape: SIN/SQU/TRI/RAMP/NOIS/USER."""
        return self._set_choice("AM:INT:FUNC", function, _MOD_FUNCTION, "AM:INT:FUNC?")

    def get_internal_function(self) -> str:
        """The internal modulating waveshape token."""
        return self.query("AM:INT:FUNC?").strip()

    def set_source(self, source: str) -> str:
        """Set the modulating source: BOTH (int+ext) or EXT."""
        return self._set_choice("AM:SOUR", source, _AM_SOURCE, "AM:SOUR?")

    def get_source(self) -> str:
        """Where the modulating signal comes from: BOTH or EXT."""
        return self.query("AM:SOUR?").strip()


class _FM(_SigGenNode):
    """Frequency-modulation subsystem (FM group)."""

    def set_state(self, state: bool) -> bool:
        """Enable/disable FM. Set up other FM params first."""
        return self._set_bool("FM:STAT", state, "FM:STAT?")

    def get_state(self) -> bool:
        """Whether frequency modulation is running."""
        return self.query_bool("FM:STAT?")

    def set_deviation(self, hertz: float | str) -> float:
        """Set the peak frequency deviation in Hz, 10 mHz-7.5 MHz."""
        return self._set_num("FM:DEV", hertz, "FM:DEV?")

    def get_deviation(self, limit: str | None = None) -> float:
        """The peak frequency deviation in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float("FM:DEV?" if limit is None else f"FM:DEV? {limit}")

    def set_internal_frequency(self, frequency: float | str) -> float:
        """Set the internal modulating frequency in Hz, 10 mHz-10 kHz."""
        return self._set_num("FM:INT:FREQ", frequency, "FM:INT:FREQ?")

    def get_internal_frequency(self, limit: str | None = None) -> float:
        """The internal modulating frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float("FM:INT:FREQ?" if limit is None else f"FM:INT:FREQ? {limit}")

    def set_internal_function(self, function: str) -> str:
        """Set the modulating waveshape: SIN/SQU/TRI/RAMP/NOIS/USER."""
        return self._set_choice("FM:INT:FUNC", function, _MOD_FUNCTION, "FM:INT:FUNC?")

    def get_internal_function(self) -> str:
        """The internal modulating waveshape token."""
        return self.query("FM:INT:FUNC?").strip()


class _FSK(_SigGenNode):
    """Frequency-shift-keying subsystem (FSK group)."""

    def set_state(self, state: bool) -> bool:
        """Enable/disable FSK. Set up other FSK params first."""
        return self._set_bool("FSK:STAT", state, "FSK:STAT?")

    def get_state(self) -> bool:
        """Whether frequency-shift keying is running."""
        return self.query_bool("FSK:STAT?")

    def set_hop_frequency(self, frequency: float | str) -> float:
        """Set the FSK 'hop' (alternate) frequency in Hz."""
        return self._set_num("FSK:FREQ", frequency, "FSK:FREQ?")

    def get_hop_frequency(self, limit: str | None = None) -> float:
        """The alternate hop frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float("FSK:FREQ?" if limit is None else f"FSK:FREQ? {limit}")

    def set_rate(self, rate: float | str) -> float:
        """Set the internal FSK shift rate in Hz, 10 mHz-50 kHz."""
        return self._set_num("FSK:INT:RATE", rate, "FSK:INT:RATE?")

    def get_rate(self, limit: str | None = None) -> float:
        """The internal shift rate in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float("FSK:INT:RATE?" if limit is None else f"FSK:INT:RATE? {limit}")

    def set_source(self, source: str) -> str:
        """Set the FSK source: INT or EXT."""
        return self._set_choice("FSK:SOUR", source, _INT_EXT, "FSK:SOUR?")

    def get_source(self) -> str:
        """What drives the shift: INT or EXT."""
        return self.query("FSK:SOUR?").strip()


class _Burst(_SigGenNode):
    """Burst-modulation subsystem (BM group)."""

    def set_state(self, state: bool) -> bool:
        """Enable/disable burst. Set up other burst params first."""
        return self._set_bool("BM:STAT", state, "BM:STAT?")

    def get_state(self) -> bool:
        """Whether burst mode is running."""
        return self.query_bool("BM:STAT?")

    def set_count(self, cycles: int | str) -> float:
        """Set cycles per burst, 1-50000; accepts INF/MIN/MAX."""
        value = cycles if isinstance(cycles, str) else int(cycles)
        return self._set_num("BM:NCYC", value, "BM:NCYC?")

    def get_count(self, limit: str | None = None) -> float:
        """Cycles per burst; ``limit`` in {MIN, MAX}."""
        return self.query_float("BM:NCYC?" if limit is None else f"BM:NCYC? {limit}")

    def set_rate(self, rate: float | str) -> float:
        """Set the internal burst rate in Hz, 10 mHz-50 kHz."""
        return self._set_num("BM:INT:RATE", rate, "BM:INT:RATE?")

    def get_rate(self, limit: str | None = None) -> float:
        """How often a burst starts, in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float("BM:INT:RATE?" if limit is None else f"BM:INT:RATE? {limit}")

    def set_phase(self, degrees: float | str) -> float:
        """Set the burst starting phase in degrees, -360 to +360."""
        return self._set_num("BM:PHAS", degrees, "BM:PHAS?")

    def get_phase(self, limit: str | None = None) -> float:
        """Where in the waveform each burst begins, in degrees; ``limit`` in {MIN, MAX}."""
        return self.query_float("BM:PHAS?" if limit is None else f"BM:PHAS? {limit}")

    def set_source(self, source: str) -> str:
        """Set the burst source: INT (triggered) or EXT (gated)."""
        return self._set_choice("BM:SOUR", source, _INT_EXT, "BM:SOUR?")

    def get_source(self) -> str:
        """What starts each burst: INT (triggered) or EXT (gated)."""
        return self.query("BM:SOUR?").strip()


class _Sweep(_SigGenNode):
    """Frequency-sweep subsystem (SWEep group + FREQ:STAR/STOP)."""

    def set_state(self, state: bool) -> bool:
        """Enable/disable sweep. Set up other sweep params first."""
        return self._set_bool("SWE:STAT", state, "SWE:STAT?")

    def get_state(self) -> bool:
        """Whether the frequency sweep is running."""
        return self.query_bool("SWE:STAT?")

    def set_start_frequency(self, frequency: float | str) -> float:
        """Set the sweep start frequency in Hz."""
        return self._set_num("FREQ:STAR", frequency, "FREQ:STAR?")

    def get_start_frequency(self, limit: str | None = None) -> float:
        """The sweep start frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float("FREQ:STAR?" if limit is None else f"FREQ:STAR? {limit}")

    def set_stop_frequency(self, frequency: float | str) -> float:
        """Set the sweep stop frequency in Hz."""
        return self._set_num("FREQ:STOP", frequency, "FREQ:STOP?")

    def get_stop_frequency(self, limit: str | None = None) -> float:
        """The sweep stop frequency in Hz; ``limit`` in {MIN, MAX}."""
        return self.query_float("FREQ:STOP?" if limit is None else f"FREQ:STOP? {limit}")

    def set_time(self, seconds: float | str) -> float:
        """Set the sweep time in seconds, 1 ms-500 s."""
        return self._set_num("SWE:TIME", seconds, "SWE:TIME?")

    def get_time(self, limit: str | None = None) -> float:
        """The time from start to stop, in seconds; ``limit`` in {MIN, MAX}."""
        return self.query_float("SWE:TIME?" if limit is None else f"SWE:TIME? {limit}")

    def set_spacing(self, spacing: str) -> str:
        """Set the sweep spacing: LIN or LOG."""
        return self._set_choice("SWE:SPAC", spacing, _SWEEP_SPACING, "SWE:SPAC?")

    def get_spacing(self) -> str:
        """How the sweep steps: LIN or LOG."""
        return self.query("SWE:SPAC?").strip()


class _Trigger(_SigGenNode):
    """Triggering subsystem (burst/sweep trigger source, slope, bus trigger)."""

    def set_source(self, source: str) -> str:
        """Set the trigger source: IMM, EXT, or BUS."""
        return self._set_choice("TRIG:SOUR", source, _TRIG_SOURCE, "TRIG:SOUR?")

    def get_source(self) -> str:
        """What triggers a burst or a sweep: IMM, EXT or BUS."""
        return self.query("TRIG:SOUR?").strip()

    def set_slope(self, slope: str) -> str:
        """Set the external-trigger slope: POS or NEG."""
        return self._set_choice("TRIG:SLOP", slope, _TRIG_SLOPE, "TRIG:SLOP?")

    def get_slope(self) -> str:
        """Which edge of the external trigger input fires: POS or NEG."""
        return self.query("TRIG:SLOP?").strip()

    def cmd_trigger(self) -> None:
        """Issue a bus trigger; needs the BUS trigger source and remote mode."""
        self.link.write("*TRG")

    def cmd_wait(self) -> None:
        """Wait for pending burst/sweep operations to complete."""
        self.link.write("*WAI")


class _Status(_SigGenNode):
    """Status-reporting registers (*CLS/*ESE/*ESR/*SRE/*STB/*PSC)."""

    def cmd_clear(self) -> None:
        """Clear the status byte and all event registers."""
        self.link.write("*CLS")

    def get_event_status(self) -> int:
        """Query the Standard Event register."""
        return self.query_int("*ESR?")

    def set_event_status_enable(self, value: int) -> int:
        """Set the Standard Event enable mask."""
        self.link.write(f"*ESE {int(value)}")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_event_status_enable()

    def get_event_status_enable(self) -> int:
        """The mask deciding which standard events reach the status byte."""
        return self.query_int("*ESE?")

    def get_status_byte(self) -> int:
        """Query the Status Byte summary register."""
        return self.query_int("*STB?")

    def set_service_request_enable(self, value: int) -> int:
        """Set the Status Byte enable mask."""
        self.link.write(f"*SRE {int(value)}")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_service_request_enable()

    def get_service_request_enable(self) -> int:
        """The mask deciding which status-byte bits request service."""
        return self.query_int("*SRE?")

    def set_power_on_status_clear(self, state: bool) -> bool:
        """Set power-on status clear."""
        self.link.write(f"*PSC {1 if state else 0}")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_power_on_status_clear()

    def get_power_on_status_clear(self) -> bool:
        """Whether the enable registers are cleared at power-on."""
        return self.query_bool("*PSC?")


class _Arb(_SigGenNode):
    """Arbitrary-waveform subsystem (DATA / DATA:DAC / FUNC:USER / DATA:* / FORM).

    Upload a waveform to VOLATILE memory (floats via :meth:`upload_floats`, or
    12-bit DAC codes via :meth:`upload_dac`), then :meth:`select` it and
    :meth:`output` it. Built-in names: SINC, NEG_RAMP, EXP_RISE, EXP_FALL,
    CARDIAC. Up to four user waveforms live in non-volatile memory.

    Transfers are chunked and paced (see :meth:`_write_chunked`): the 33120A's
    RS-232 input buffer is small and DTR/DSR flow control is not relied upon, so
    large writes are broken up to avoid overrunning it.
    """

    ARB_MIN_POINTS = 8
    ARB_MAX_POINTS = 16000
    DAC_MAX = 2047
    _CHUNK_BYTES = 32
    # 0.2 s/32 B matches the legacy driver's proven pacing: the 33120A parses a
    # long DATA/DATA:DAC command out of its small (~128 B) RS-232 input buffer
    # slower than the wire delivers it, so faster pacing (0.05 s) overruns the
    # buffer (+521) and corrupts the upload. Large arbs are therefore slow to
    # send (bench work is rare); flow control could speed this up later.
    _CHUNK_PAUSE_S = 0.2

    def _write_chunked(self, payload: bytes) -> None:
        """Write raw bytes to the instrument in small paced chunks."""
        instr = self.link.visa
        total = len(payload)
        sent = 0
        while sent < total:
            end = min(sent + self._CHUNK_BYTES, total)
            instr.write_raw(payload[sent:end])
            sent = end
            if sent < total:
                time.sleep(self._CHUNK_PAUSE_S)

    def set_data_floats(self, data: Sequence[float]) -> int:
        """Download a waveform to VOLATILE memory as floats (DATA VOLATILE).

        ``data`` is any real sequence; it is auto-scaled so the largest
        magnitude maps to full scale (+/-1). Returns the number of points. Select
        and output it with :meth:`select` + :meth:`output` (or the driver's
        ``upload_waveform``).
        """
        arr = np.asarray(data, dtype=float)
        n = int(arr.size)
        if n < self.ARB_MIN_POINTS or n > self.ARB_MAX_POINTS:
            raise ValueError(
                f"waveform length must be {self.ARB_MIN_POINTS}-{self.ARB_MAX_POINTS}, got {n}"
            )
        peak = float(np.max(np.abs(arr)))
        if peak == 0.0:
            raise ValueError("waveform is all zeros - cannot scale")
        payload = ",".join(f"{v:.5f}" for v in (arr / peak))
        self._write_chunked(f"DATA VOLATILE,{payload}\n".encode("ascii"))
        time.sleep(_RECONFIG_SETTLE_S)
        return n

    def set_data_dac(self, codes: Sequence[int]) -> int:
        """Download a waveform to VOLATILE memory as 12-bit DAC codes.

        ``codes`` are integers in [-2047, 2047] (auto-clipped/rounded); sent as a
        big-endian (NORM byte order) IEEE-488.2 binary block via DATA:DAC. Returns
        the number of points.
        """
        arr = np.asarray(codes)
        n = int(arr.size)
        if n < self.ARB_MIN_POINTS or n > self.ARB_MAX_POINTS:
            raise ValueError(
                f"waveform length must be {self.ARB_MIN_POINTS}-{self.ARB_MAX_POINTS}, got {n}"
            )
        clipped = np.clip(np.round(arr), -self.DAC_MAX, self.DAC_MAX).astype(">i2")
        raw = clipped.tobytes()
        header = f"#{len(str(len(raw)))}{len(raw)}".encode("ascii")
        self.set_byte_order("NORM")
        self._write_chunked(b"DATA:DAC VOLATILE," + header + raw + b"\n")
        time.sleep(_RECONFIG_SETTLE_S)
        return n

    def set_selected(self, name: str = "VOLATILE") -> str:
        """Select the active arbitrary waveform; returns the read-back."""
        self.link.write(f"FUNC:USER {name}")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.get_selected()

    def get_selected(self) -> str:
        """Query the selected arbitrary-waveform name."""
        return self.link.query("FUNC:USER?").strip().strip('"')

    def get_output_name(self) -> str:
        """Output the selected arbitrary waveform."""
        self.link.write("FUNC:SHAP USER")
        time.sleep(_RECONFIG_SETTLE_S)
        return self.link.query("FUNC:SHAP?").strip()

    def get_catalog(self) -> list[str]:
        """List all waveform names in memory."""
        return [
            s.strip().strip('"')
            for s in self.link.query("DATA:CAT?").strip().split(",")
            if s.strip()
        ]

    def get_nvram_catalog(self) -> list[str]:
        """List user waveform names in non-volatile memory."""
        resp = self.link.query("DATA:NVOL:CAT?").strip()
        return [s.strip().strip('"') for s in resp.split(",") if s.strip().strip('"')]

    def get_nvram_free(self) -> int:
        """Query the number of free non-volatile waveform slots."""
        return int(round(self.link.query_float("DATA:NVOL:FREE?")))

    def cmd_copy(self, name: str) -> None:
        """Copy VOLATILE to a non-volatile name; <=8 chars, A-Z/0-9/_."""
        self.link.write(f"DATA:COPY {name}")
        time.sleep(_ARB_NVRAM_SETTLE_S)

    def cmd_delete(self, name: str) -> None:
        """Delete a stored waveform; cannot delete the active or built-ins."""
        self.link.write(f"DATA:DEL {name}")
        time.sleep(_ARB_NVRAM_SETTLE_S)

    def cmd_delete_all(self) -> None:
        """Delete VOLATILE and all user non-volatile waveforms."""
        self.link.write("DATA:DEL:ALL")
        time.sleep(_ARB_NVRAM_SETTLE_S)

    def get_points(self, name: str = "") -> int:
        """Number of points in a waveform; default = active.

        The 33120A returns this count in NR3 scientific notation (e.g.
        ``+2.56000E+02``), so it is parsed as a float and rounded, not via
        ``query_int`` (whose integer regex would misread the mantissa).
        """
        return int(round(self.link.query_float(f"DATA:ATTR:POIN?{(' ' + name) if name else ''}")))

    def get_average(self, name: str = "") -> float:
        """Arithmetic mean of a waveform's points."""
        return self.link.query_float(f"DATA:ATTR:AVER?{(' ' + name) if name else ''}")

    def get_peak_to_peak(self, name: str = "") -> float:
        """Peak-to-peak of a waveform, 0..1."""
        return self.link.query_float(f"DATA:ATTR:PTP?{(' ' + name) if name else ''}")

    def get_crest_factor(self, name: str = "") -> float:
        """Crest factor (peak/RMS) of a waveform."""
        return self.link.query_float(f"DATA:ATTR:CFAC?{(' ' + name) if name else ''}")

    def set_byte_order(self, order: str) -> str:
        """Set binary-block byte order: NORM (MSB first) or SWAP."""
        self.link.write_choice(
            "FORM:BORD",
            order.upper(),
            {"NORM": "NORM", "NORMAL": "NORM", "SWAP": "SWAP", "SWAPPED": "SWAP"},
        )
        return self.link.query("FORM:BORD?").strip()

    def get_byte_order(self) -> str:
        """Query the binary-block byte order token."""
        return self.link.query("FORM:BORD?").strip()


class _Calibration(Node):
    """Calibration subsystem - QUERY ONLY (never performs or unsecures a cal)."""

    def get_count(self) -> int:
        """Query the calibration count."""
        return self.query_int("CAL:COUN?")

    def get_string(self) -> str:
        """Query the stored calibration message."""
        return self.query("CAL:STR?").strip().strip('"')

    def get_secure_state(self) -> bool:
        """Query the calibration secured state."""
        return self.query_bool("CAL:SEC:STAT?")


def _characterize_deaf_window(gen: Keysight_33120A) -> None:
    """Measure the SYST:REM deaf-window on hardware (increment-1 deliverable).

    For a series of candidate settle times, toggle SYST:LOC -> SYST:REM, wait,
    and time an *IDN? readback. A failure/timeout means the instrument was still
    deaf; the link is cleared and resynced before the next trial so one failure
    does not desync the rest. Benign - no output settings are touched.
    """
    print("  -- SYST:REM deaf-window characterization --")
    for settle in (0.0, 0.01, 0.02, 0.05, 0.1):
        gen.link.write("SYST:LOC")
        time.sleep(1.0)  # ensure fully out of remote before the timed trial
        gen.link.write("SYST:REM")
        time.sleep(settle)
        t0 = time.monotonic()
        try:
            with gen.link.temporary_timeout(2000):
                idn = gen.link.query("*IDN?").strip()
            dt = (time.monotonic() - t0) * 1e3
            ok = bool(_IDN_RE.search(idn))
            print(f"    settle={settle:.2f}s -> {'OK' if ok else 'BAD'} in {dt:6.1f} ms")
        except Exception as exc:  # noqa: BLE001 - a deaf window shows up as a timeout
            print(f"    settle={settle:.2f}s -> DEAF ({type(exc).__name__})")
            with contextlib.suppress(Exception):
                gen.link.visa.clear()
                gen.link.write("*CLS")
    # Leave the instrument in remote and settled for the caller.
    gen.link.write("SYST:REM")
    time.sleep(1.0)


def _resync(gen: Keysight_33120A) -> None:
    with contextlib.suppress(Exception):
        gen.link.visa.clear()
        gen.link.write("*CLS")


def _characterize_apply_window(gen: Keysight_33120A) -> None:
    """Measure the deaf-window after output-reconfiguring writes.

    A reconfiguring write (APPLy, and possibly FUNC/VOLT/FREQ/OFFSet/duty) leaves
    the instrument briefly deaf; a query issued too soon is DROPPED, so the read
    times out AND desyncs. This finds (a) the APPLy settle floor and (b) which
    per-command writes share the window - comparing the driver's natural pacing
    gap (~50 ms, no extra settle) against a +150 ms settle. A VISA clear + *CLS
    resyncs before each trial. Benign; resync-safe.
    """
    gen.link.connect(timeout_ms=gen._connect_timeout_ms)
    _resync(gen)
    gen.link.write("SYST:REM")
    time.sleep(_REMOTE_SETTLE_S)
    print(f"  reconnected, IDN: {gen.link.query('*IDN?').strip()}")

    gen.link.write("OUTP:LOAD INF")
    time.sleep(0.3)
    _resync(gen)

    print("  -- reconfiguration settle floor per transition (find first OK) --")
    # (label, state-A write, state-B write, readback) - B is the timed transition.
    transitions = [
        ("same-range amp", "APPL:SIN 1000,1,0", "APPL:SIN 1000,1.2,0", "APPL?"),
        ("func SIN->SQU", "APPL:SIN 1000,2,0", "APPL:SQU 1000,2,0", "APPL?"),
        ("func SQU->TRI", "APPL:SQU 1000,2,0", "APPL:TRI 2000,3,0", "APPL?"),
        ("func TRI->RAMP", "APPL:TRI 1000,2,0", "APPL:RAMP 1000,2,0", "APPL?"),
        ("amp up 0.1->5", "APPL:SIN 1000,0.1,0", "APPL:SIN 1000,5,0", "APPL?"),
        ("amp down 5->0.05", "APPL:SIN 1000,5,0", "APPL:SIN 1000,0.05,0", "APPL?"),
        ("VOLT 0.1->5", "VOLT 0.1", "VOLT 5", "VOLT?"),
        ("VOLT 5->0.05", "VOLT 5", "VOLT 0.05", "VOLT?"),
    ]
    sweep = (0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0)
    for label, set_a, set_b, qcmd in transitions:
        floor = None
        for settle in sweep:
            _resync(gen)
            gen.link.write(set_a)
            time.sleep(0.8)  # settle state A fully
            _resync(gen)
            gen.link.write(set_b)
            time.sleep(settle)
            try:
                with gen.link.temporary_timeout(2500):
                    gen.link.query(qcmd)
                floor = settle
                break
            except Exception:  # noqa: BLE001
                _resync(gen)
        print(
            f"    {label:20s} first-OK settle = "
            f"{'>2.0s (never)' if floor is None else f'{floor:.2f}s'}"
        )


def _main(argv: list[str] | None = None) -> int:
    """Live self-test against the bench 33120A.

    Default: connect, print identity, characterize the SYST:REM
    deaf-window, drain errors, return to local. ``apply``: characterize the
    post-APPLy deaf-window instead. Touches only a benign 1 Vpp sine.
    """
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE [apply]")
        return 2
    resource = argv[1]
    mode = (argv[2] if len(argv) > 2 else "").lower()
    gen = Keysight_33120A(resource)
    if mode == "apply":
        print(f"connecting to 33120A on {gen.resource_name} (apply-window mode) ...")
        try:
            _characterize_apply_window(gen)
        finally:
            gen.disconnect()
            print("disconnected (local).")
        return 0

    print(f"connecting to 33120A on {gen.resource_name} ...")
    idn = gen.connect()
    print(f"  IDN: {idn}")
    print(f"  model={gen.model}  firmware={gen.firmware}")
    try:
        print(f"  SCPI version: {gen.get_scpi_version()}")
        _characterize_deaf_window(gen)
        gen.check_errors()
        print("  error queue clean.")
    finally:
        gen.disconnect()
        print("disconnected (local).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
