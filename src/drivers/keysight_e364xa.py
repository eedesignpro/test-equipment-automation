"""Driver for the Keysight/Agilent E364xA family DC power supplies.

Public import path::

    from drivers.keysight_e364xa import Keysight_E364xA

The E364xA family (E3640A-E3649A) are single- and dual-output bench supplies.
The bench unit is an E3646A (dual output, 8 V/3 A LOW and 20 V/1.5 A HIGH
ranges) on COM6 (``ASRL6::INSTR``) at 9600 baud. That link overflows easily, so
all I/O is paced by :class:`_PsuSerialLink` (ADR-033) and a query timeout is
treated as a fatal "power cycle me" condition rather than retried. The port is
pinned - the driver does not scan for the instrument (ADR-034).

Shape follows ``docs/driver-api-design.md`` (section 8, E3646A). The output
enable is a single GLOBAL control (``set_output_state_all``) because the E3646A
has one enable for both outputs; the ``ch1``/``ch2`` channels carry no output
method. Per channel, ``set_vi`` sets voltage and current limit atomically (and
auto-ranges), ``meas_*`` reads, ``get_condition`` reports CC/CV. Units carry no
scalar prefix (volts, amperes, seconds).

Task recipes::

    psu.connect()
    psu.ch1.set_vi(5.0, 1.0, ovp=6.0)        # 5 V, 1 A limit, OVP 6 V
    psu.set_output_state_all(True)
    r = psu.ch1.meas_vi()                     # r.voltage, r.current
    psu.set_output_state_all(False)

Command surface follows the Agilent E364xA User's and Service Guide, chapter 3.

Common tasks:
  1. Source 5 V with a 1 A limit on CH1, then energize:
         psu1.ch1.set_vi(5.0, 1.0)
         psu1.set_output_state_all(True)      # one enable, BOTH channels
  2. Read what it is actually delivering:
         v, i = psu1.ch1.meas_vi()
  3. Check whether it is regulating voltage or current:
         psu1.ch1.get_condition()             # CV, CC or UNREG
  4. Source with over-voltage protection armed:
         psu1.ch1.set_vi(5.0, 1.0, ovp=6.0)
  5. Recover after a protection trip:
         if psu1.ch1.get_ovp_tripped():
             psu1.ch1.cmd_clear_ovp()
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pyvisa as visa

from common.errors import InstrumentError, InstrumentUnresponsiveError
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import ScpiEnum, quote

# Inter-command spacing for the bench E3646A on its 9600-baud UART. 50 ms was
# exercised without fault throughout the T3 session (connect, snapshot, system
# group); dropping to 10 ms overflowed the UART after a short write burst and
# hung the PSU (see ADR-022/ADR-023 and the captures README). 50 ms is the
# evidenced-safe floor; overridable via the constructor.
_DEFAULT_PACING_S = 0.05

# Generous connect/IO timeout - the UART is slow and must never be hammered.
_DEFAULT_TIMEOUT_MS = 5000

# Settle time after SYST:REM (the front-panel display updates slowly).
_SETTLE_S = 0.5

# After a VOLT:RANGe change that ACTUALLY moves the range the E3646A stops
# answering queries for ~0.5-1 s (relay/DAC settle) while still ACCEPTING
# writes. A readback query issued too soon times out AND desyncs the RS-232
# stream. Bench-measured 2026-07-24: 0.5 s still fails, 1.0 s is clean -> 1.5 s
# with margin. A same-range write causes NO deaf window (bench-measured
# 2026-08-10: same-range readback in 93 ms; a real change timed out), so the
# driver only settles when the range actually changes.
_RANGE_SETTLE_S = 1.5

# Writing a status ENABLE register (*ESE / *SRE / *PSC / STAT:QUES:ENAB) to a
# NEW value briefly stops the E3646A answering queries - the same deaf-window /
# desync failure as VOLT:RANG (bench-verified 2026-07-24: *ESE 16 then an
# immediate query times out and desyncs; a settle clears it). Writing the value
# it already holds is a no-op with no window. Status-enable writes are rare, so
# the readback simply waits this out.
_STATUS_SETTLE_S = 1.5

# VOLT:RANG accepts LOW/HIGH aliases and the model tokens; the query returns the
# model token. Canonicalize both sides to LOW/HIGH so the deaf-window settle
# fires only on a real range change.
_LOW_TOKENS = {"LOW", "P8V", "P35V"}
_HIGH_TOKENS = {"HIGH", "P20V", "P60V"}

_DISPLAY_MODE = {"VI": "VI", "VV": "VV", "II": "II"}

_TRIG_SOURCE = {"BUS": "BUS", "IMM": "IMMediate", "IMMEDIATE": "IMMediate"}

_CONDITION = {0: "OFF", 1: "CC", 2: "CV", 3: "FAIL"}


# model -> (output count, LOW range (V, A), HIGH range (V, A)). Ranges are
# informational (exposed via ``self.ranges``); the instrument itself enforces
# limits. Dual-output models are E3646A-E3649A.
_MODEL_INFO: dict[str, dict] = {
    "E3640A": {"outputs": 1, "low": (8.0, 3.0), "high": (20.0, 1.5)},
    "E3641A": {"outputs": 1, "low": (35.0, 0.8), "high": (60.0, 0.5)},
    "E3642A": {"outputs": 1, "low": (8.0, 5.0), "high": (20.0, 2.5)},
    "E3643A": {"outputs": 1, "low": (35.0, 1.4), "high": (60.0, 0.8)},
    "E3644A": {"outputs": 1, "low": (8.0, 8.0), "high": (20.0, 4.0)},
    "E3645A": {"outputs": 1, "low": (35.0, 2.2), "high": (60.0, 1.3)},
    "E3646A": {"outputs": 2, "low": (8.0, 3.0), "high": (20.0, 1.5)},
    "E3647A": {"outputs": 2, "low": (35.0, 0.8), "high": (60.0, 0.5)},
    "E3648A": {"outputs": 2, "low": (8.0, 5.0), "high": (20.0, 2.5)},
    "E3649A": {"outputs": 2, "low": (35.0, 1.4), "high": (60.0, 0.8)},
}

_IDN_RE = re.compile(r"(E364\d[A-Za-z])", re.IGNORECASE)


def _onoff(state: bool) -> str:
    return "ON" if state else "OFF"


def _num(value: float | str) -> str:
    """Format an APPLy/level argument: a token (MIN/MAX/DEF) or a number."""
    if isinstance(value, str):
        return value
    return f"{value:g}"


def _canonical_range(token: str) -> str:
    """Map any range token/alias to canonical ``LOW``/``HIGH`` for comparison."""
    t = token.strip().upper()
    if t in _LOW_TOKENS:
        return "LOW"
    if t in _HIGH_TOKENS:
        return "HIGH"
    return t


@dataclass(frozen=True)
class Reading:
    """A measured voltage/current pair.

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
class SupplyState:
    """A channel's programmed setpoints and OVP configuration."""

    voltage: float  # volts (programmed level)
    current: float  # amperes (programmed limit)
    voltage_range: str  # the model range token in force (e.g. "P8V")
    ovp: float  # volts (OVP trip level)
    ovp_enabled: bool


class _PsuSerialLink(InstrumentLink):
    """InstrumentLink configured for the E364xA RS-232 UART.

    Opens the serial resource with the fixed line settings (9600/8/N/1, ``\\n``
    terminators) without issuing a VISA ``clear`` (the fragility protocol wants
    first contact gentle). Any query/write that times out is translated into
    :class:`InstrumentUnresponsiveError` so callers stop rather than retry.

    This link also owns the inter-command pacing. The delay is a property of
    THIS supply on THIS UART, not generic link behavior, so it lives here and
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
        rm = visa.ResourceManager()
        self._instr = self._open_resource(rm, self.resource_name)
        self.configure_serial()
        self._instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms

    def configure_serial(self) -> None:
        """Apply the fixed serial line settings to the open resource."""
        instr = self._instr
        with contextlib.suppress(Exception):
            instr.baud_rate = self.baud_rate
            instr.data_bits = 8
            instr.parity = visa.constants.Parity.none
            instr.stop_bits = visa.constants.StopBits.one
        # Set the PUBLIC termination attributes (a legacy driver set the private
        # _read_termination, which pyvisa ignores).
        instr.write_termination = "\n"
        instr.read_termination = "\n"

    def query(self, message: str, **kwargs) -> str:
        self._pace()
        try:
            return super().query(message, **kwargs)
        except visa.VisaIOError as exc:
            raise self._translate_timeout(message, exc) from exc

    def write(self, message: str, **kwargs) -> int:
        self._pace()
        try:
            return super().write(message, **kwargs)
        except visa.VisaIOError as exc:
            raise self._translate_timeout(message, exc) from exc

    @staticmethod
    def _translate_timeout(message: str, exc: visa.VisaIOError):
        if exc.error_code == visa.constants.StatusCode.error_timeout:
            return InstrumentUnresponsiveError(
                f"PSU did not respond to {message!r} (VISA timeout). Stop all "
                "work - the operator must power cycle the PSU."
            )
        return exc


class Keysight_E364xA(Instrument):
    """Keysight/Agilent E364xA family DC power supply.

    ``connect()`` opens the pinned port, reads one ``*IDN?``, enters remote mode
    and gates the second output on dual-output models. Output enable is global
    (:meth:`set_output_state_all`); per-channel settings live on ``ch1``/``ch2``,
    which select their output internally.
    """

    class Range(ScpiEnum):
        """Voltage range selector (the model maps LOW/HIGH to its own tokens)."""

        LOW = "LOW"
        HIGH = "HIGH"

    class DisplayMode(ScpiEnum):
        """Front-panel readout mode."""

        VI = "VI"
        VV = "VV"
        II = "II"

    class TriggerSource(ScpiEnum):
        """Trigger source for the triggered levels."""

        BUS = "BUS"
        IMMEDIATE = "IMMediate"

    def __init__(
        self,
        resource_name: str,
        *,
        pacing_s: float = _DEFAULT_PACING_S,
        connect_timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        super().__init__(resource_name, name="PSU")
        self.link = _PsuSerialLink(
            resource_name,
            timeout_ms=connect_timeout_ms,
            min_command_interval=pacing_s,
        )
        Node.__init__(self, self.link, name="PSU")
        self.resource_name = resource_name
        self._connect_timeout_ms = connect_timeout_ms

        self.model: str | None = None
        self.firmware: str | None = None
        self.num_outputs: int | None = None
        self.port: str | None = None
        self.ranges: dict[str, tuple[float, float] | None] = {}
        self._dual = False
        self._selected: int | None = None

        self.ch1 = _Channel(self, 1)
        self.ch2 = _Channel(self, 2)
        self.trig = _Trigger(self.link, name="TRIGGER")
        self.status = _Status(self.link, name="STATUS")
        self.cal = _Calibration(self.link, name="CALIBRATION")

    @property
    def outputs(self) -> list[_Channel]:
        return [self.ch1, self.ch2]

    def channel(self, n: int) -> _Channel:
        """Return the channel node for ``n`` (1 or 2)."""
        if n == 1:
            return self.ch1
        if n == 2:
            return self.ch2
        raise ValueError(f"channel must be 1 or 2, got {n!r}")

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the pinned port, read the identity once, then enter remote mode.

        First contact is one ``*IDN?`` on the configured resource (gentle: no
        VISA clear, no port scan - see ADR-034). A VISA timeout is fatal
        (:class:`InstrumentUnresponsiveError`); the link is never retried. An IDN
        from something other than an E364xA raises :class:`InstrumentError`.
        """
        to = self._connect_timeout_ms if timeout_ms is None else timeout_ms
        self.link.resource_name = self.resource_name
        self.link.connect(timeout_ms=to)
        idn = self.link.query("*IDN?").strip()
        if not _IDN_RE.search(idn):
            self.link.disconnect()
            raise InstrumentError(
                f"{self.resource_name} did not answer with an E364xA IDN: {idn!r}. "
                "Check the port; this driver does not scan for the instrument."
            )
        self.port = self.resource_name
        self.idn = idn
        self.link.write("SYST:REM")
        time.sleep(_SETTLE_S)
        self._on_connect(idn)
        return idn

    def _on_connect(self, idn: str) -> None:
        """Parse the IDN for model/firmware and gate outputs and ranges."""
        match = _IDN_RE.search(idn)
        if not match:
            raise InstrumentError(f"Not a recognized E364xA IDN: {idn!r}")
        model = match.group(1).upper()
        self.model = model
        fields = [f.strip() for f in idn.split(",")]
        self.firmware = fields[3] if len(fields) >= 4 else None

        info = _MODEL_INFO.get(model, {"outputs": 1, "low": None, "high": None})
        self.num_outputs = info["outputs"]
        self._dual = info["outputs"] == 2
        self.ranges = {"LOW": info["low"], "HIGH": info["high"]}

        self.ch2.installed = self._dual
        self._selected = None

    def safe_state(self) -> None:
        """Turn the outputs off, then return to local mode, before the link closes."""
        with contextlib.suppress(Exception):
            self.link.write("OUTP OFF")
        with contextlib.suppress(Exception):
            self.link.write("SYST:LOC")

    def _select_output(self, out: int) -> None:
        """Select the given output (INST:NSEL) unless already selected.

        No-op on single-output models. Selection state is tracked to avoid
        redundant paced round trips; it is reset whenever it could drift.
        """
        if not self._dual:
            return
        if self._selected == out:
            return
        self.link.write(f"INST:NSEL {out}")
        self._selected = out

    def get_selected_output(self) -> int:
        """Query the currently selected output number."""
        return self.link.query_int("INST:NSEL?")

    def cmd_reset(self) -> None:
        """Reset to the power-on state; selection returns to output 1."""
        self.link.write("*RST")
        self.link.query("*OPC?")
        self._selected = None

    def cmd_clear_status(self) -> None:
        """Clear the status byte and event registers."""
        self.link.write("*CLS")

    # ---------- global output enable (one enable for BOTH outputs) ----------

    def set_output_state_all(self, state: bool) -> bool:
        """Enable/disable the supply output - GLOBAL for both channels.

        The E3646A has a single output enable, so this lives on the instrument
        (not the channels) and drops/raises both outputs together.
        """
        self.link.write(f"OUTP {_onoff(state)}")
        return self.get_output_state_all()

    def get_output_state_all(self) -> bool:
        """Query the (global) output enable state."""
        return self.link.query_bool("OUTP?")

    def set_output_relay_state(self, state: bool) -> bool:
        """Set the RS-232 external-relay TTL signals."""
        self.link.write(f"OUTP:REL {_onoff(state)}")
        return self.get_output_relay_state()

    def get_output_relay_state(self) -> bool:
        """Query the external-relay TTL state."""
        return self.link.query_bool("OUTP:REL?")

    def set_tracking_state(self, state: bool) -> bool:
        """Enable/disable output tracking; dual-output models only."""
        self.link.write(f"OUTP:TRAC {_onoff(state)}")
        return self.get_tracking_state()

    def get_tracking_state(self) -> bool:
        """Query the output-tracking state."""
        return self.link.query_bool("OUTP:TRAC?")

    def set_trigger_coupling_state(self, state: bool) -> bool:
        """Couple the two outputs' triggered levels."""
        self.link.write(f"INST:COUP:TRIG {_onoff(state)}")
        return self.get_trigger_coupling_state()

    def get_trigger_coupling_state(self) -> bool:
        """Query the triggered-level coupling state."""
        return self.link.query_bool("INST:COUP:TRIG?")

    # ---------- system ----------

    def cmd_beep(self) -> None:
        """Issue a single beep."""
        self.link.write("SYST:BEEP")

    def set_display_state(self, state: bool) -> bool:
        """Turn the front-panel display on/off."""
        self.link.write(f"DISP {_onoff(state)}")
        return self.get_display_state()

    def get_display_state(self) -> bool:
        """Query the front-panel display state."""
        return self.link.query_bool("DISP?")

    def set_display_mode(self, mode: str) -> str:
        """Set the display readout mode; a :class:`DisplayMode`."""
        self.link.write_choice("DISP:MODE", str(mode).upper(), _DISPLAY_MODE)
        return self.get_display_mode()

    def get_display_mode(self) -> str:
        """Query the display readout mode."""
        return self.link.query("DISP:MODE?").strip()

    def set_display_text(self, text: str) -> str:
        """Show up to 11 characters on the front panel."""
        self.link.write(f"DISP:TEXT {quote(text)}")
        return self.get_display_text()

    def get_display_text(self) -> str:
        """Query the front-panel message."""
        return self.link.query("DISP:TEXT?").strip().strip('"')

    def cmd_clear_display_text(self) -> None:
        """Clear the front-panel message."""
        self.link.write("DISP:TEXT:CLE")

    def get_scpi_version(self) -> str:
        """Query the SCPI version string."""
        return self.link.query("SYST:VERS?").strip()

    def self_test(self) -> int:
        """Run the self-test; 0 = pass, non-zero = fail."""
        return self.link.query_int("*TST?")

    def get_errors(self) -> list[str]:
        """Drain and return the error queue (empty list means clean)."""
        out: list[str] = []
        for _ in range(40):
            resp = self.link.query("SYST:ERR?").strip()
            if resp.split(",", 1)[0].strip().lstrip("+") == "0":
                break
            out.append(resp)
        return out

    def cmd_remote(self) -> None:
        """Enter remote mode, then settle.

        Waits ``_SETTLE_S`` after the mode change: the E3646A is briefly deaf to
        queries while the front panel updates (bench-verified).
        """
        self.link.write("SYST:REM")
        time.sleep(_SETTLE_S)

    def cmd_local(self) -> None:
        """Return to local mode, then settle."""
        self.link.write("SYST:LOC")
        time.sleep(_SETTLE_S)

    def cmd_remote_lock(self) -> None:
        """Enter remote with the front-panel Local key locked, then settle."""
        self.link.write("SYST:RWL")
        time.sleep(_SETTLE_S)

    # ---------- state storage ----------

    def cmd_save_state(self, location: int) -> None:
        """Save the present state to memory location 1-5."""
        self._check_location(location)
        self.link.write(f"*SAV {location}")

    def cmd_recall_state(self, location: int) -> None:
        """Recall a stored state from location 1-5."""
        self._check_location(location)
        self.link.write(f"*RCL {location}")
        self._selected = None

    def get_state_name(self, location: int) -> str:
        """Query the name assigned to a memory location."""
        self._check_location(location)
        return self.link.query(f"MEM:STAT:NAME? {location}").strip().strip('"')

    @staticmethod
    def _check_location(location: int) -> None:
        if location not in (1, 2, 3, 4, 5):
            raise ValueError(f"state location must be 1-5, got {location!r}")

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Read back the touched settings so they can be restored later.

        Records the global output state plus, per installed channel, the full
        :class:`SupplyState` (range, voltage, current, OVP level and enable).
        Read-back queries only (no *SAV, which would clobber a memory slot).
        """
        state: dict = {"output": self.get_output_state_all(), "channels": {}}
        for ch in self.outputs:
            if ch.installed:
                state["channels"][ch._out] = ch.get_vi()
        return state

    def restore(self, snapshot: dict) -> None:
        """Restore settings captured by :meth:`snapshot`.

        Range (and OVP) are restored before the level (range gates the valid
        setpoint), then the global output enable last.
        """
        for out_num, st in snapshot.get("channels", {}).items():
            ch = self.outputs[out_num - 1]
            ch.set_voltage_range(st.voltage_range)
            ch.set_vi(st.voltage, st.current, ovp=st.ovp)
            ch.set_ovp_state(st.ovp_enabled)
        self.set_output_state_all(snapshot.get("output", False))

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator:
        """Snapshot the touched settings on enter and restore them on exit."""
        state = self.snapshot()
        try:
            yield self
        finally:
            self.restore(state)


class _Channel(Node):
    """One logical output of the supply (``ch1``/``ch2``).

    Every method selects its output (INST:NSEL) before acting, so callers never
    track the active output; on single-output models the select is a no-op. The
    output ENABLE is global and lives on the instrument, so the channel has no
    output method.
    """

    def __init__(self, psu: Keysight_E364xA, out: int):
        super().__init__(psu.link, name=f"CH{out}")
        self._psu = psu
        self._out = out

    def _sel(self) -> None:
        self._require_installed()
        self._psu._select_output(self._out)

    # ---------- primary set/get (voltage + current limit together) ----------

    def set_vi(self, voltage: float, current: float, *, ovp: float | None = None) -> SupplyState:
        """Set the voltage level and current LIMIT in one call; returns the state.

        Auto-ranges: the range that holds ``voltage`` is selected first (LOW when
        it fits the low range, else HIGH). ``ovp`` sets the over-voltage trip
        level before the output level lands. Both ``voltage`` and ``current`` are
        required - the supply must never source without a current limit.
        """
        self._sel()
        self._autorange(voltage)
        if ovp is not None:
            self.set_and_verify("VOLT:PROT", float(ovp))
        self.write(f"APPL {_num(voltage)},{_num(current)}")
        return self.get_vi()

    def get_vi(self) -> SupplyState:
        """Read the programmed voltage, current limit, range and OVP config."""
        self._sel()
        resp = self.query("APPL?").strip().strip('"')
        v_str, i_str = resp.split(",")
        return SupplyState(
            voltage=float(v_str),
            current=float(i_str),
            voltage_range=self.query("VOLT:RANG?").strip(),
            ovp=self.query_float("VOLT:PROT?"),
            ovp_enabled=self.query_bool("VOLT:PROT:STAT?"),
        )

    def _autorange(self, voltage: float) -> None:
        low = self._psu.ranges.get("LOW")
        if low is None:
            return  # unknown model: leave the current range as-is
        self.set_voltage_range("LOW" if abs(float(voltage)) <= low[0] + 1e-9 else "HIGH")

    # ---------- range (settle only on an actual change) ----------

    def set_voltage_range(self, rng: str) -> str:
        """Select the voltage range; a :class:`Range` or model token.

        Reads the current range first and only writes+settles when the canonical
        LOW/HIGH range actually changes: a same-range write causes no deaf window,
        while a real change makes the E3646A deaf for ~1 s (bench-verified).
        """
        target = _canonical_range(str(rng))
        if target not in ("LOW", "HIGH"):
            raise ValueError(f"invalid range {rng!r}; use Range.LOW/HIGH or a model token")
        self._sel()
        if _canonical_range(self.query("VOLT:RANG?").strip()) == target:
            return self.query("VOLT:RANG?").strip()
        self.write(f"VOLT:RANG {target}")
        time.sleep(_RANGE_SETTLE_S)  # deaf window only on a real change
        return self.query("VOLT:RANG?").strip()

    def get_voltage_range(self) -> str:
        """Query the selected range token, e.g. ``P8V``/``P20V``."""
        self._sel()
        return self.query("VOLT:RANG?").strip()

    # ---------- measurement ----------

    def meas_voltage(self) -> float:
        """Measure the output voltage, volts."""
        self._sel()
        return self.query_float("MEAS:VOLT?")

    def meas_current(self) -> float:
        """Measure the output current, amperes."""
        self._sel()
        return self.query_float("MEAS:CURR?")

    def meas_vi(self) -> Reading:
        """Measure the output voltage and current together."""
        return Reading(self.meas_voltage(), self.meas_current())

    def get_condition(self) -> str:
        """Regulation mode: OFF / CC / CV / FAIL."""
        self._require_installed()
        code = self.link.query_int(f"STAT:QUES:INST:ISUM{self._out}:COND?")
        return _CONDITION.get(code, str(code))

    # ---------- OVP ----------

    def set_ovp(self, volts: float) -> float:
        """Set the OVP trip level, volts. Returns the read-back."""
        self._sel()
        return self.set_and_verify("VOLT:PROT", float(volts))

    def get_ovp(self) -> float:
        """Query the OVP trip level, volts."""
        self._sel()
        return self.query_float("VOLT:PROT?")

    def set_ovp_state(self, state: bool) -> bool:
        """Enable/disable the OVP function."""
        self._sel()
        self.write(f"VOLT:PROT:STAT {_onoff(state)}")
        return self.get_ovp_state()

    def get_ovp_state(self) -> bool:
        """Query whether OVP is enabled."""
        self._sel()
        return self.query_bool("VOLT:PROT:STAT?")

    def get_ovp_tripped(self) -> bool:
        """Whether the OVP circuit has tripped."""
        self._sel()
        return self.query_bool("VOLT:PROT:TRIP?")

    def cmd_clear_ovp(self) -> None:
        """Clear a tripped OVP condition.

        Lower the output below the trip point (or raise the trip level) first.
        """
        self._sel()
        self.write("VOLT:PROT:CLE")

    # ---------- voltage step + triggered levels ----------

    def set_voltage_step(self, step: float) -> float:
        """Set the VOLT UP/DOWN step size, volts."""
        self._sel()
        return self.set_and_verify("VOLT:STEP", float(step))

    def get_voltage_step(self) -> float:
        """Query the voltage step size."""
        self._sel()
        return self.query_float("VOLT:STEP?")

    def cmd_voltage_up(self) -> None:
        """Step the voltage up by the step size (VOLT UP)."""
        self._sel()
        self.write("VOLT UP")

    def cmd_voltage_down(self) -> None:
        """Step the voltage down by the step size (VOLT DOWN)."""
        self._sel()
        self.write("VOLT DOWN")

    def set_triggered_vi(self, voltage: float, current: float) -> tuple[float, float]:
        """Set the pending triggered voltage and current.

        Applied on the next trigger; returns the read-back (voltage, current).
        """
        self._sel()
        v = self.set_and_verify("VOLT:TRIG", float(voltage))
        i = self.set_and_verify("CURR:TRIG", float(current))
        return float(v), float(i)

    def get_triggered_vi(self) -> tuple[float, float]:
        """Query the pending triggered (voltage, current)."""
        self._sel()
        return self.query_float("VOLT:TRIG?"), self.query_float("CURR:TRIG?")


class _Trigger(Node):
    """Triggering subsystem (bus/immediate source, delay, initiate, *TRG)."""

    def set_source(self, source: str) -> str:
        """Select the trigger source; a :class:`Keysight_E364xA.TriggerSource`."""
        self.write_choice("TRIG:SOUR", str(source).upper(), _TRIG_SOURCE)
        return self.get_source()

    def get_source(self) -> str:
        """Query the trigger source."""
        return self.query("TRIG:SOUR?").strip()

    def set_delay(self, seconds: float) -> float:
        """Set the trigger delay 0-3600 s. Returns the read-back."""
        return self.set_and_verify("TRIG:DEL", float(seconds))

    def get_delay(self) -> float:
        """Query the trigger delay, seconds."""
        return self.query_float("TRIG:DEL?")

    def cmd_initiate(self) -> None:
        """Initiate the trigger subsystem."""
        self.write("INIT")

    def cmd_trigger(self) -> None:
        """Send a bus trigger."""
        self.write("*TRG")


class _Status(Node):
    """Status-reporting registers (*ESR/*ESE/*STB/*SRE/*PSC and STAT:QUES)."""

    def get_event_status(self) -> int:
        """Standard Event register."""
        return self.query_int("*ESR?")

    def set_event_status_enable(self, value: int) -> int:
        """Set the Standard Event enable mask; settles the deaf window."""
        self.write(f"*ESE {int(value)}")
        time.sleep(_STATUS_SETTLE_S)
        return self.get_event_status_enable()

    def get_event_status_enable(self) -> int:
        """Standard Event enable mask."""
        return self.query_int("*ESE?")

    def get_status_byte(self) -> int:
        """Status Byte summary register."""
        return self.query_int("*STB?")

    def set_service_request_enable(self, value: int) -> int:
        """Set the Status Byte enable mask; settles the deaf window."""
        self.write(f"*SRE {int(value)}")
        time.sleep(_STATUS_SETTLE_S)
        return self.get_service_request_enable()

    def get_service_request_enable(self) -> int:
        """Status Byte enable mask."""
        return self.query_int("*SRE?")

    def set_power_on_status_clear(self, state: bool) -> bool:
        """Set power-on status clear; settles the deaf window."""
        self.write(f"*PSC {1 if state else 0}")
        time.sleep(_STATUS_SETTLE_S)
        return self.get_power_on_status_clear()

    def get_power_on_status_clear(self) -> bool:
        """Power-on status-clear flag."""
        return self.query_bool("*PSC?")

    def get_questionable_event(self) -> int:
        """Questionable Status event register."""
        return self.query_int("STAT:QUES?")

    def set_questionable_enable(self, value: int) -> int:
        """Set the Questionable Status enable mask; settles."""
        self.write(f"STAT:QUES:ENAB {int(value)}")
        time.sleep(_STATUS_SETTLE_S)
        return self.get_questionable_enable()

    def get_questionable_enable(self) -> int:
        """Questionable Status enable mask."""
        return self.query_int("STAT:QUES:ENAB?")

    def get_questionable_instrument_event(self) -> int:
        """Questionable-instrument event register."""
        return self.query_int("STAT:QUES:INST?")

    def set_questionable_instrument_enable(self, value: int) -> int:
        """Set the questionable-instrument enable mask; settles."""
        self.write(f"STAT:QUES:INST:ENAB {int(value)}")
        time.sleep(_STATUS_SETTLE_S)
        return self.get_questionable_instrument_enable()

    def get_questionable_instrument_enable(self) -> int:
        """Questionable-instrument enable mask."""
        return self.query_int("STAT:QUES:INST:ENAB?")

    def cmd_preset(self) -> None:
        """Preset the status enable registers."""
        self.write("STAT:PRES")


class _Calibration(Node):
    """Calibration subsystem - QUERY ONLY (no calibration writes ever)."""

    def get_count(self) -> int:
        """Calibration count."""
        return self.query_int("CAL:COUN?")

    def get_string(self) -> str:
        """Stored calibration message."""
        return self.query("CAL:STR?").strip().strip('"')

    def get_secure_state(self) -> bool:
        """Calibration secured state."""
        return self.query_bool("CAL:SEC:STAT?")


def _main(argv: list[str] | None = None) -> int:
    """Live read-mostly self-test against the bench E3646A (output stays OFF)."""
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE")
        return 2
    resource = argv[1]
    psu = Keysight_E364xA(resource)
    print(f"connecting to PSU on {psu.resource_name} ...")
    idn = psu.connect()
    print(f"  IDN: {idn}")
    print(f"  model={psu.model} outputs={psu.num_outputs} port={psu.port} fw={psu.firmware}")
    try:
        print(f"  output (global): {psu.get_output_state_all()}")
        for ch in psu.outputs:
            if ch.installed:
                print(f"  {ch.name}: {ch.get_vi()}  cond={ch.get_condition()}")
        print(f"  SCPI version : {psu.get_scpi_version()}")
        with psu.preserved_setup():
            print("  -- benign round-trips on CH1 (output OFF) --")
            print(f"    set_vi(2.5, 0.5)      -> {psu.ch1.set_vi(2.5, 0.5)}")
            print(f"    range HIGH (change)   -> {psu.ch1.set_voltage_range('HIGH')}")
            print(f"    range LOW  (change)   -> {psu.ch1.set_voltage_range('LOW')}")
            print(f"    set_vi(3, 1, ovp=6)   -> {psu.ch1.set_vi(3.0, 1.0, ovp=6.0)}")
        print("  restored to as-found.")
        errs = psu.get_errors()
        print(f"  error queue  : {errs or 'clean'}")
    finally:
        psu.disconnect()
        print("disconnected (local).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
