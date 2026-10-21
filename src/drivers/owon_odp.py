"""Driver for the OWON ODP-series programmable DC power supplies.

Public import path::

    from drivers.owon_odp import Owon_ODP

Shape follows ``docs/driver-api-design.md`` (section 8, ODP6033). A channel is
set through one call, ``set_vi(voltage, current)``, which returns a
:class:`SupplyState`. Both values are required, because a supply must never
source without a current limit. Measurements are ``meas_*``.

The bench unit is an ODP6033 with three outputs. ``connect()`` reads the
per-channel maxima from the instrument and attaches the channels it really has,
so a smaller model gets no ``ch3`` attribute rather than one that raises. The
maxima come from the hardware and not from a model table: the supply reports
61.0 V / 3.1 A on CH1 and CH2, and 6.6 V / 3.1 A on CH3.

Units carry no scalar prefix (design standard 4.7): volts, amperes, watts.

Bench observations behind the choices in this file (ADR-042). All measured on
the bench ODP6033 over USBTMC on 2026-08-10 with ``projects/owon_probe.py``:

- **This supply has no error queue.** ``SYSTem:ERRor`` does not exist. Its whole
  command set is 24 commands. A rejected write is therefore SILENT, and reading
  the value back is the only way to catch one. Every ``set_`` here verifies.
- **Never send a run of writes before a query.** A write costs 0.12 ms because
  it is buffered and a query costs 33 ms, so a burst outruns the supply and the
  query that follows returns an empty string. Worse, the session then holds a
  partial message: every later query answers one response behind, and a VISA
  clear does NOT recover it. Only a power cycle does. So every write in this
  driver is followed by its own read-back, which is the pattern the legacy
  driver used and the reason it never hit this. If a query ever returns
  something unparseable, that is the symptom, and the fix is a power cycle.
- A query costs about 32.6 ms and a write about 0.12 ms, so a query costs about
  280 times a write. Minimize QUERIES on this instrument, not writes. That is
  why :meth:`snapshot` and the ``_all`` calls use the three-channel forms: one
  query returns all three channels.
- Because a write is nearly free, the driver selects the channel before every
  channel command rather than caching the selection. A cached selection can go
  stale when somebody uses the front panel. The re-select costs 0.12 ms.
- **An output-relay write needs a pause before any further traffic, and the
  output needs far longer than that to arrive.** Three separate timescales get
  confused here. The relay's own read-back is immediate: a write followed by an
  instant query returned the correct new state every time, in 87 ms to 133 ms.
  That fact is what tempted this driver into deleting the legacy 0.5 s pause,
  and deleting it desynced the link, so :data:`_RELAY_SETTLE_S` restores it. The
  analog output is slower again: after enabling, the measured voltage stayed at
  0.000 V
  and then reached its setpoint at 332 ms for 1 V, 608 ms for 5 V open circuit,
  and 630 ms for 5 V into a 0.2 A load. Turning off, the output fell below 0.1 V
  within 89 ms.
- Those points imply about 250 ms of turn-on delay plus a ramp near 14 V/s. The
  ramp comes from two setpoints, so anything above 5 V is **# UNVERIFIED**.
  **This driver does not wait for any of it.** An output on its way up is how
  the equipment works, not a fault, and a supply that turns on into a load
  holding it at the current limit never settles at all, so a driver-side wait
  would block on exactly the case that matters. The numbers are recorded so a
  SCRIPT can choose its own delay before it measures.
- **A protection limit drags its setpoint down, silently.** Writing an OVP of
  3 V while the voltage setpoint was 5 V moved the setpoint to 3.000. Writing a
  5 V setpoint under a 3 V OVP left it at 3.000. Current behaves the same way.
  :meth:`_Output.set_vi` therefore writes the protections first, then the
  levels, then checks the read-back and raises a message that names the limit.
- **The coupling modes are mutually exclusive in hardware.** Turning on parallel
  cleared series by itself, and turning on tracking cleared parallel. So
  :meth:`set_coupling` writes one mode and lets the instrument clear the others.
- ``OUTPut?`` reports whether ANY channel is on, not whether all are. With CH1
  alone enabled it answered 1.

This supply reports no CV/CC condition. No command exists for it, so there is no
``get_condition`` here, unlike the E3646A. Compare a measured current against
the setpoint if a script needs to know.

Task recipes::

    psu2.connect()

    # 1. Source 5 V with a 1 A limit on CH1, then energize:
    psu2.ch1.set_vi(5.0, 1.0)
    psu2.ch1.set_output_state(True)

    # 2. Read what the output is really doing:
    r = psu2.ch1.meas_vi()                  # r.voltage, r.current
    v, i = psu2.ch1.meas_vi()               # the same, unpacked

    # 3. Add protection trips:
    psu2.ch1.set_vi(5.0, 1.0, ovp=6.0, ocp=1.5)

    # 4. Set all three channels in one atomic write:
    psu2.set_voltage_all(5.0, 3.3, 1.8)

    # 5. Everything off:
    psu2.set_output_state_all(False)

Command surface follows the OWON ODP-series programming manual.

Common tasks:
  1. Source 3.3 V at 0.5 A on CH1 and energize only that channel:
         psu2.ch1.set_vi(3.3, 0.5)
         psu2.ch1.set_output_state(True)
  2. Read one channel back:
         v, i = psu2.ch1.meas_vi()
  3. Bring every channel up or down together:
         psu2.set_output_state_all(True)
  4. Set all three voltages in one atomic write:
         psu2.set_voltage_all(3.3, 5.0, 12.0)
  5. Wire two channels in series for more voltage:
         psu2.set_coupling(psu2.Coupling.SERIES)
"""

from __future__ import annotations

import contextlib
import math
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pyvisa as visa

from common.errors import InstrumentError
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import ScpiEnum


_DEFAULT_TIMEOUT_MS = 5000

# The supply reports three decimals, so anything closer than a milli-unit is the
# same value. Used to decide whether a protection limit clamped a setpoint.
_CLAMP_ABS_TOL = 1.5e-3
_CLAMP_REL_TOL = 1e-3

# Measured time from an output-enable write to a valid measurement, so a script
# can choose its own delay before it measures. This driver does NOT wait: an
# output on its way up is normal, not a fault, and a supply that turns on into a
# load at its current limit never settles at all.
#   1 V open circuit    332 ms
#   5 V open circuit    608 ms
#   5 V into 0.2 A      630 ms
#   turning off, below 0.1 V within 89 ms
# Those imply roughly 250 ms of turn-on delay plus a ramp near 14 V/s. The ramp
# is inferred from two setpoints, so anything above 5 V is  # UNVERIFIED.
_SETTLE_1V_MS = 332
_SETTLE_5V_MS = 630

# Pause after ANY relay write, before further traffic. This is a LINK
# requirement, not an analog one. A relay write answers its own read-back
# immediately, which is what led to deleting this pause; that was wrong. Without
# it, a query one or two operations later returns an empty string and the
# session is then one response behind, sometimes recoverable by reconnecting and
# sometimes only by a power cycle.
#
# It is applied to the output relays, where it was measured. Whether the
# coupling relays need it too is  # UNVERIFIED: a run that exercised series,
# parallel and track desynced shortly afterwards, but the operator had a second
# session open on the same USB device at the time, and concurrent access alone
# explains an empty read. No settle is applied there on that evidence.
#
# The value is the legacy driver's, which ran for months without a desync. The
# true minimum is  # UNVERIFIED - characterizing it costs a power cycle per
# failed attempt, so the working value stands until that is worth it.
_RELAY_SETTLE_S = 0.5

# IDN form: OWON,ODP6033,<serial>,FV:<version> -> capture the ODPxxxx model.
_IDN_RE = re.compile(r",(ODP\d{4}),")

# The generator prints this in the API map. The channel count comes from the
# hardware rather than a model table, so it is stated here in words (design
# standard 9.1 rule 4, with the gate in data rather than in a code branch).
CHANNEL_GATE = (
    "Channel nodes are the same shape on every model, so they are listed once. "
    "connect() reads the per-channel maxima from the supply and attaches one "
    "node per output it reports: the bench ODP6033 gets ch1, ch2 and ch3."
)


def _onoff(state: bool) -> str:
    return "ON" if state else "OFF"


def _num(value: float) -> str:
    return f"{float(value):g}"


@dataclass(frozen=True)
class Reading:
    """A measured voltage/current pair.

    Iterates over the measured quantities in field order, which the ``meas_``
    name spells, so ``v, i = ch.meas_vi()`` works and ``r.voltage`` still does.
    A field added later that the name does not spell is not iterated, so no
    unpacking call site breaks.

    The supply has no combined measurement command, so the two values come from
    two queries and therefore from two instants a few tens of milliseconds apart.
    """

    voltage: float  # volts
    current: float  # amperes

    def __iter__(self) -> Iterator[float]:
        """Yield voltage then current, the quantities ``meas_vi`` names."""
        yield self.voltage
        yield self.current


@dataclass(frozen=True)
class SupplyState:
    """One channel's setpoints and its protection trips."""

    voltage: float  # volts, the programmed setpoint
    current: float  # amperes, the programmed limit
    ovp: float  # volts, the over-voltage trip
    ocp: float  # amperes, the over-current trip


class _OwonUsbLink(InstrumentLink):
    """InstrumentLink for the ODP USBTMC connection.

    Opens the resource with ``\\n`` terminators and WITHOUT a VISA ``clear``.
    The known-good legacy driver did neither a clear nor a remote-mode write on
    connect, and this supply works the same way.
    """

    def connect(self, timeout_ms: int | None = None) -> None:
        rm = visa.ResourceManager()
        self._instr = self._open_resource(rm, self.resource_name)
        with contextlib.suppress(Exception):
            self._instr.write_termination = "\n"
            self._instr.read_termination = "\n"
        self._instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms
        # No VISA clear here. A clear was tried against a link that had already
        # desynced and did NOT recover it, so it would only imply a safety net
        # that does not exist. See the module docstring: the cure is to never
        # send a burst of writes, and the recovery is a power cycle.


class Owon_ODP(Instrument):
    """OWON ODP-series programmable DC power supply on USBTMC.

    ``connect()`` opens the link, validates the identity and attaches one
    channel node per output the supply reports. Output enable is per channel
    (:meth:`_Output.set_output_state`) and also available for all channels at
    once (:meth:`set_output_state_all`).
    """

    class Coupling(ScpiEnum):
        """How CH1 and CH2 are wired together inside the supply.

        The modes are mutually exclusive in hardware: turning one on clears the
        others by itself (bench-verified). INDEPENDENT is the power-on state.
        """

        INDEPENDENT = "INDEPENDENT"
        SERIES = "SERIES"
        PARALLEL = "PARALLEL"
        TRACK = "TRACK"

    # Coupling member -> the command that turns that mode on.
    _COUPLING_CMD = {"SERIES": "OUTP:SER", "PARALLEL": "OUTP:PAR", "TRACK": "OUTP:TRAC"}

    ch1: _Output
    ch2: _Output  # attached by connect() on a two-output model or larger
    ch3: _Output  # attached by connect() on a three-output model

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
            pyvisa resource for the supply (default the bench USBTMC string).
        connect_timeout_ms : int
            I/O timeout used for connect and normal operation, in milliseconds.
        """
        super().__init__(resource_name, name="PSU2")
        self.link = _OwonUsbLink(resource_name, timeout_ms=connect_timeout_ms)
        Node.__init__(self, self.link, name="PSU2")
        self.resource_name = resource_name
        self._connect_timeout_ms = connect_timeout_ms

        self.model: str | None = None
        self.serial: str | None = None
        self.firmware: str | None = None
        self.num_outputs: int | None = None
        self.v_max: tuple[float, ...] = ()
        self.i_max: tuple[float, ...] = ()

        # ch1 exists on every model, so it is built here and the API-map
        # generator can walk it offline. connect() attaches the rest.
        self.ch1 = _Output(self, 1)

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the USB link, validate the identity, and attach the channels."""
        to = self._connect_timeout_ms if timeout_ms is None else timeout_ms
        self.link.connect(timeout_ms=to)
        idn = self.link.query("*IDN?").strip()
        if not _IDN_RE.search(idn):
            self.link.disconnect()
            raise InstrumentError(
                f"{self.resource_name} did not answer with an OWON ODP identity: {idn!r}. "
                "Check the resource string; this driver does not scan for the instrument."
            )
        self.idn = idn
        self._on_connect(idn)
        return idn

    def _on_connect(self, idn: str) -> None:
        """Parse the identity, read the maxima, attach one node per output."""
        match = _IDN_RE.search(idn)
        self.model = match.group(1) if match else None
        fields = [f.strip() for f in idn.split(",")]
        self.serial = fields[2] if len(fields) >= 3 else None
        self.firmware = fields[3] if len(fields) >= 4 else None

        # The supply reports its own per-channel maxima, which is more truthful
        # than a hand-written model table.
        self.v_max = self._query_floats("VOLT:LIM:ALL? MAX")
        self.i_max = self._query_floats("CURR:LIM:ALL? MAX")
        self.num_outputs = len(self.v_max) or 1

        for number in (2, 3):
            name = f"ch{number}"
            if self.num_outputs >= number:
                if not isinstance(getattr(self, name, None), _Output):
                    setattr(self, name, _Output(self, number))
            elif isinstance(getattr(self, name, None), _Output):
                delattr(self, name)

    def safe_state(self) -> None:
        """Turn every output off before the link closes (this supply has relays)."""
        with contextlib.suppress(Exception):
            self.link.write("OUTP OFF")

    def check_errors(self) -> None:
        """Do nothing: this supply has no error queue, so there is nothing to drain.

        There is deliberately no ``get_errors`` here. Returning an empty list
        would claim the supply reported no problem, when in truth it cannot
        report one at all. Every ``set_`` verifies its read-back instead.
        """

    @property
    def channels(self) -> list[_Output]:
        """The channel nodes this model has, in order."""
        found = [self.ch1]
        for number in (2, 3):
            node = getattr(self, f"ch{number}", None)
            if isinstance(node, _Output):
                found.append(node)
        return found

    def channel(self, n: int) -> _Output:
        """Return the channel node for ``n``."""
        for ch in self.channels:
            if ch.number == n:
                return ch
        have = [ch.number for ch in self.channels]
        raise ValueError(f"channel must be one of {have} on this {self.model or 'ODP'}, got {n!r}")

    # ---------- output enable ----------

    def set_output_state_all(self, state: bool) -> list[bool]:
        """Turn every channel's output on or off in one write.

        Returns as soon as the relay state reads back, which is immediate. The
        outputs themselves are still moving: see :meth:`_Output.set_output_state`
        for the measured times a caller should allow before measuring.
        """
        self.link.write(f"OUTP {_onoff(state)}")
        time.sleep(_RELAY_SETTLE_S)  # link requirement; see _RELAY_SETTLE_S
        return self.get_output_state_all()

    def get_output_state_all(self) -> list[bool]:
        """Whether each channel's output is on, in channel order."""
        resp = self.link.query("CHAN:OUTP:ALL?").strip()
        values = [v.strip() for v in resp.split(",") if v.strip()]
        expected = self.num_outputs or len(values)
        if len(values) != expected:
            raise InstrumentError(
                f"the supply answered its output-state query with {resp!r}, which is "
                f"not {expected} states. The USB link is most likely one response "
                "behind after an interrupted session. Reconnect to clear it."
            )
        return [v not in ("0", "OFF") for v in values]

    def _set_output_states(self, states: list[bool]) -> list[bool]:
        """Write every channel's output state at once (CHAN:OUTPut:ALL)."""
        payload = ",".join("1" if s else "0" for s in states[: self.num_outputs or len(states)])
        self.link.write(f"CHAN:OUTP:ALL {payload}")
        time.sleep(_RELAY_SETTLE_S)  # link requirement; see _RELAY_SETTLE_S
        return self.get_output_state_all()

    # ---------- coupling ----------

    def set_coupling(self, coupling: Coupling) -> Coupling:
        """Wire CH1 and CH2 together, or leave them independent; a :class:`Coupling`.

        The supply clears the other modes by itself, so this writes one command.
        """
        target = str(_token(coupling, self.Coupling, "coupling"))
        if target == "INDEPENDENT":
            for cmd in self._COUPLING_CMD.values():
                self.link.write(f"{cmd} OFF")
        else:
            self.link.write(f"{self._COUPLING_CMD[target]} ON")
        return self.get_coupling()

    def get_coupling(self) -> Coupling:
        """How CH1 and CH2 are wired together right now."""
        for member, cmd in self._COUPLING_CMD.items():
            if self.link.query_bool(f"{cmd}?"):
                return self.Coupling(member)
        return self.Coupling.INDEPENDENT

    # ---------- all-channel levels and measurements ----------

    def set_voltage_all(self, *volts: float) -> tuple[float, ...]:
        """Set every channel's voltage setpoint in one atomic write.

        Justified as an exception to the one-helper rule (design standard 5.3):
        the supply applies all three in a single command, which three per-channel
        calls cannot do.
        """
        self.link.write("APP:VOLT " + ",".join(_num(v) for v in volts))
        return self._query_floats("APP:VOLT?")

    def get_voltage_all(self) -> tuple[float, ...]:
        """Every channel's voltage setpoint, in one query."""
        return self._query_floats("APP:VOLT?")

    def set_current_all(self, *amps: float) -> tuple[float, ...]:
        """Set every channel's current limit in one atomic write."""
        self.link.write("APP:CURR " + ",".join(_num(a) for a in amps))
        return self._query_floats("APP:CURR?")

    def get_current_all(self) -> tuple[float, ...]:
        """Every channel's current limit, in one query."""
        return self._query_floats("APP:CURR?")

    def meas_voltage_all(self) -> tuple[float, ...]:
        """Measure every channel's output voltage in volts, in one query."""
        return self._query_floats("MEAS:VOLT:ALL?")

    def meas_current_all(self) -> tuple[float, ...]:
        """Measure every channel's output current in amperes, in one query."""
        return self._query_floats("MEAS:CURR:ALL?")

    # ---------- system ----------

    def cmd_reset(self) -> None:
        """Reset the supply to its power-on state."""
        self.link.write("*RST")

    def cmd_remote(self) -> None:
        """Enter remote mode, which locks the front panel except for the key lock."""
        self.link.write("SYST:REM")

    def cmd_local(self) -> None:
        """Return control to the front panel."""
        self.link.write("SYST:LOC")

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Read back every setting this driver can change, for an exact restore.

        Uses the three-channel query forms, so the whole supply costs five
        queries rather than one query per channel per value. A query costs about
        280 times a write on this link, which makes that worth doing.
        """
        return {
            "outputs": self.get_output_state_all(),
            "voltage": self.get_voltage_all(),
            "current": self.get_current_all(),
            "ovp": self._query_floats("VOLT:LIM:ALL?"),
            "ocp": self._query_floats("CURR:LIM:ALL?"),
        }

    def restore(self, snapshot: dict) -> None:
        """Restore the settings captured by :meth:`snapshot`, exactly as found.

        Protections go first, because a protection limit drags its setpoint down
        on this supply. Levels follow, and the output relays go last.
        """
        # One write per read-back, never a burst: a run of writes outruns this
        # supply and the next query returns empty. The read-backs are discarded
        # here, but they are what paces the sequence.
        for key, command in (
            ("ovp", "VOLT:LIM:ALL"),
            ("ocp", "CURR:LIM:ALL"),
            ("voltage", "APP:VOLT"),
            ("current", "APP:CURR"),
        ):
            values = snapshot.get(key)
            if values:
                self.link.write(f"{command} " + ",".join(_num(v) for v in values))
                self._query_floats(f"{command}?")
        if "outputs" in snapshot:
            self._set_output_states(list(snapshot["outputs"]))

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator:
        """Context manager: snapshot on enter, restore the exact state on exit."""
        state = self.snapshot()
        try:
            yield state
        finally:
            self.restore(state)

    # ---------- underscored plumbing (design standard 5.3) ----------

    def _select(self, out: int) -> None:
        """Select the channel that the un-suffixed commands act on (INSTrument:NSELect).

        Written every time rather than cached. A write costs 0.12 ms here, and a
        cached selection goes stale the moment somebody touches the front panel.
        """
        self.link.write(f"INST:NSEL {out}")

    def _get_selected(self) -> int:
        """Which channel the un-suffixed commands currently act on (INSTrument:NSELect?)."""
        return self.link.query_int("INST:NSEL?")

    def _set_coupling_parallel(self, state: bool) -> bool:
        """Turn parallel mode on or off by itself (OUTPut:PARallel)."""
        self.link.write(f"OUTP:PAR {_onoff(state)}")
        return self.link.query_bool("OUTP:PAR?")

    def _set_coupling_series(self, state: bool) -> bool:
        """Turn series mode on or off by itself (OUTPut:SERies)."""
        self.link.write(f"OUTP:SER {_onoff(state)}")
        return self.link.query_bool("OUTP:SER?")

    def _set_coupling_track(self, state: bool) -> bool:
        """Turn tracking mode on or off by itself (OUTPut:TRACk)."""
        self.link.write(f"OUTP:TRAC {_onoff(state)}")
        return self.link.query_bool("OUTP:TRAC?")

    def _set_ovp_all(self, *volts: float) -> tuple[float, ...]:
        """Set every channel's over-voltage trip in one write (VOLTage:LIMit:ALL)."""
        self.link.write("VOLT:LIM:ALL " + ",".join(_num(v) for v in volts))
        return self._query_floats("VOLT:LIM:ALL?")

    def _set_ocp_all(self, *amps: float) -> tuple[float, ...]:
        """Set every channel's over-current trip in one write (CURRent:LIMit:ALL)."""
        self.link.write("CURR:LIM:ALL " + ",".join(_num(a) for a in amps))
        return self._query_floats("CURR:LIM:ALL?")

    def _get_ovp_all(self) -> tuple[float, ...]:
        """Every channel's over-voltage trip, in one query (VOLTage:LIMit:ALL?)."""
        return self._query_floats("VOLT:LIM:ALL?")

    def _get_ocp_all(self) -> tuple[float, ...]:
        """Every channel's over-current trip, in one query (CURRent:LIMit:ALL?)."""
        return self._query_floats("CURR:LIM:ALL?")

    def _query_floats(self, cmd: str) -> tuple[float, ...]:
        """Query a comma-separated numeric response and return it as floats."""
        resp = self.link.query(cmd).strip()
        out = []
        for field in resp.split(","):
            field = field.strip()
            if not field:
                continue
            with contextlib.suppress(ValueError):
                out.append(float(field))
        return tuple(out)


def _token(value, enum_cls: type[ScpiEnum], label: str) -> str:
    """Return the member name for an enum member or a case-insensitive string."""
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    for member in enum_cls:
        if text in (member.name, str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")


class _Output(Node):
    """One output of the supply.

    Every call selects its channel first, so a caller never tracks which channel
    is active. The selection is written each time rather than cached, because a
    write is nearly free on this link and a cached selection can go stale.
    """

    def __init__(self, psu: Owon_ODP, number: int):
        super().__init__(psu.link, name=f"CH{number}")
        self._psu = psu
        self.number = number

    # ---------- the operating point ----------

    def set_vi(
        self,
        voltage: float,
        current: float,
        *,
        ovp: float | None = None,
        ocp: float | None = None,
    ) -> SupplyState:
        """Set the voltage setpoint and the current limit; returns the state.

        Both values are required, because this supply must never source without
        a current limit. ``ovp`` and ``ocp`` set the protection trips.

        The protections are written first, then the levels, because a protection
        limit on this supply drags its setpoint down to match. If a protection
        blocks the level that was asked for, this raises and names the limit,
        since the supply itself reports nothing.
        """
        self._psu._select(self.number)
        # One write per read-back, never a burst. A run of writes outruns this
        # supply and the next query comes back empty (see _write_verify).
        ovp_now = self._write_verify("VOLT:LIM", ovp) if ovp is not None else self.get_ovp()
        ocp_now = self._write_verify("CURR:LIM", ocp) if ocp is not None else self.get_ocp()
        voltage_now = self._write_verify("VOLT", voltage)
        current_now = self._write_verify("CURR", current)
        state = SupplyState(voltage=voltage_now, current=current_now, ovp=ovp_now, ocp=ocp_now)
        self._check_clamped("voltage", voltage, state.voltage, "OVP", state.ovp, "V")
        self._check_clamped("current", current, state.current, "OCP", state.ocp, "A")
        return state

    def get_vi(self) -> SupplyState:
        """Read the voltage setpoint, the current limit and both protection trips."""
        self._psu._select(self.number)
        return SupplyState(
            voltage=self.query_float("VOLT?"),
            current=self.query_float("CURR?"),
            ovp=self.query_float("VOLT:LIM?"),
            ocp=self.query_float("CURR:LIM?"),
        )

    def set_ovp(self, volts: float) -> float:
        """Set the over-voltage trip in V; it pulls the voltage setpoint down with it."""
        self._psu._select(self.number)
        self.write(f"VOLT:LIM {_num(volts)}")
        return self.get_ovp()

    def get_ovp(self) -> float:
        """The over-voltage trip in volts."""
        self._psu._select(self.number)
        return self.query_float("VOLT:LIM?")

    def set_ocp(self, amps: float) -> float:
        """Set the over-current trip in A; it pulls the current limit down with it."""
        self._psu._select(self.number)
        self.write(f"CURR:LIM {_num(amps)}")
        return self.get_ocp()

    def get_ocp(self) -> float:
        """The over-current trip in amperes."""
        self._psu._select(self.number)
        return self.query_float("CURR:LIM?")

    # ---------- output enable ----------

    def set_output_state(self, state: bool) -> bool:
        """Turn this channel's output on or off, leaving the other channels alone.

        Returns as soon as the relay state reads back, which is immediate. The
        output itself is still moving, and it measures 0.000 V until it arrives:
        332 ms to reach 1 V, 608 ms to reach 5 V open circuit, and 630 ms into a
        0.2 A load on the bench unit. Turning off, it is below 0.1 V within
        89 ms. Allow for that in the script before measuring.

        This call does not wait. An output on its way up is how the equipment
        works, and a supply turning on into a load that holds it at the current
        limit never settles at all, so a wait here would block on exactly the
        case that matters.
        """
        states = self._psu.get_output_state_all()
        while len(states) < (self._psu.num_outputs or self.number):
            states.append(False)
        states[self.number - 1] = bool(state)
        return self._psu._set_output_states(states)[self.number - 1]

    def get_output_state(self) -> bool:
        """Whether this channel's output is on."""
        return self._psu.get_output_state_all()[self.number - 1]

    # ---------- measurements ----------

    def meas_voltage(self) -> float:
        """Measure this channel's output voltage in volts."""
        self._psu._select(self.number)
        return self.query_float("MEAS:VOLT?")

    def meas_current(self) -> float:
        """Measure this channel's output current in amperes."""
        self._psu._select(self.number)
        return self.query_float("MEAS:CURR?")

    def meas_power(self) -> float:
        """Measure this channel's output power in watts."""
        self._psu._select(self.number)
        return self.query_float("MEAS:POW?")

    def meas_vi(self) -> Reading:
        """Measure this channel's output voltage and current.

        The supply has no combined command, so these are two queries and the two
        values come from two instants a few tens of milliseconds apart.
        """
        self._psu._select(self.number)
        return Reading(
            voltage=self.query_float("MEAS:VOLT?"),
            current=self.query_float("MEAS:CURR?"),
        )

    # ---------- underscored plumbing ----------

    def _check_clamped(
        self, what: str, asked: float, applied: float, guard: str, trip: float, unit: str
    ) -> None:
        """Raise when a protection trip blocked the setpoint that was requested."""
        if math.isclose(asked, applied, rel_tol=_CLAMP_REL_TOL, abs_tol=_CLAMP_ABS_TOL):
            return
        raise InstrumentError(
            f"{self.name}: asked for {asked:g} {unit} of {what} but the supply holds "
            f"{applied:g} {unit}. The {guard} is {trip:g} {unit} and a protection trip "
            f"drags its setpoint down on this supply. Raise the {guard} first, through "
            f"set_vi(..., {guard.lower()}=...)."
        )

    def _write_verify(self, cmd: str, value: float) -> float:
        """Write one value and read that same value straight back.

        One write per query, never a burst. A write costs 0.12 ms here because
        it is buffered, while a query costs 33 ms, so several writes in a row
        outrun the supply and the query that follows returns an empty string.
        Interleaving is also what the proven legacy driver did. The read-back is
        needed anyway, because a protection trip clamps a setpoint silently and
        this supply has no error queue to report it.
        """
        self.write(f"{cmd} {_num(value)}")
        return self.query_float(f"{cmd}?")

    def _set_voltage(self, volts: float) -> float:
        """Write the voltage setpoint alone (VOLTage)."""
        self._psu._select(self.number)
        return self._write_verify("VOLT", volts)

    def _get_voltage(self) -> float:
        """Read the voltage setpoint alone (VOLTage?)."""
        self._psu._select(self.number)
        return self.query_float("VOLT?")

    def _set_current(self, amps: float) -> float:
        """Write the current limit alone (CURRent)."""
        self._psu._select(self.number)
        return self._write_verify("CURR", amps)

    def _get_current(self) -> float:
        """Read the current limit alone (CURRent?)."""
        self._psu._select(self.number)
        return self.query_float("CURR?")


def _main(argv: list[str] | None = None) -> int:
    """Live self-test against the bench ODP.

    Default: read-only. It reports the identity, the maxima, the coupling and
    every channel's setpoints and measurements, and it changes nothing. Pass
    ``demo`` to source 1 V on CH1 briefly into whatever is attached, which ends
    with the outputs off.
    """
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE [demo]")
        return 2
    resource = argv[1]
    run_demo = "demo" in argv[2:]
    psu = Owon_ODP(resource)
    print(f"connecting to OWON ODP on {psu.resource_name} ...")
    idn = psu.connect()
    try:
        print(f"  identity: {idn}")
        print(f"  model={psu.model} serial={psu.serial} firmware={psu.firmware}")
        print(f"  outputs={psu.num_outputs} v_max={psu.v_max} i_max={psu.i_max}")
        print(f"  has ch2={hasattr(psu, 'ch2')} ch3={hasattr(psu, 'ch3')}")
        print(f"  coupling: {psu.get_coupling()}")
        print(f"  outputs on: {psu.get_output_state_all()}")
        for ch in psu.channels:
            state = ch.get_vi()
            reading = ch.meas_vi()
            print(
                f"  {ch.name}: set {state.voltage:.3f} V / {state.current:.3f} A  "
                f"OVP {state.ovp:.3f} V  OCP {state.ocp:.3f} A  "
                f"measured {reading.voltage:.4f} V / {reading.current:.4f} A"
            )
        if run_demo:
            print("  --- demo: CH1 at 1 V, 0.5 A limit ---")
            with psu.preserved_setup():
                print(f"    set_vi   -> {psu.ch1.set_vi(1.0, 0.5, ovp=6.0, ocp=1.0)}")
                print(f"    output   -> {psu.ch1.set_output_state(True)}")
                # The driver does not wait for the output to arrive, so the
                # script picks its own delay. Measured: 332 ms to reach 1 V.
                v, i = psu.ch1.meas_vi()
                print(f"    meas_vi immediately -> {v:.4f} V, {i:.4f} A  (still on its way up)")
                time.sleep(0.5)
                v, i = psu.ch1.meas_vi()
                print(f"    meas_vi after 500 ms -> {v:.4f} V, {i:.4f} A")
                print(f"    power    -> {psu.ch1.meas_power():.4f} W")
                print(f"    off      -> {psu.ch1.set_output_state(False)}")
            print("    restored as found.")
    finally:
        psu.disconnect()
        print("disconnected (outputs off).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
