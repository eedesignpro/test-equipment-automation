"""Driver for the HP / Agilent / Keysight 34401A 6.5 digit digital multimeter.

Public import path::

    from drivers.keysight_34401a import Keysight_34401A

The 34401A has RS-232 and GPIB. Over RS-232 it needs the line settings of its
manual (34401A User's Guide, chapter 4, "RS-232 Interface Configuration"):

- 9600 baud (factory setting; 300 to 9600 from the front panel only).
- 8 data bits, no parity (factory setting), and **2 stop bits, fixed**.
- **DTR/DSR handshake.** The meter drops DTR when its input buffer (about
  100 characters) is full, and while it has a query reply to send. So the
  controller must read each reply before it sends more data.
- ``SYST:REM`` before other RS-232 commands; ``SYST:LOC`` at the end.

A link with 1 stop bit and no handshake works for single commands, but a burst
of writes followed by a query loses the reply: the query times out and the
meter queues -410 "Query INTERRUPTED". This driver opens the port with the
settings above and also holds a short gap between commands.

Measurement is the whole instrument, so the ``meas_*`` methods sit at the top
level, as in the Siglent SDM driver. A ``meas_*`` call with no keyword
arguments reads with the present setup (``READ?``) if the function is already
active; with any keyword argument it configures first (``CONF``). An overload
(9.9E37) reads as NaN.

Common tasks:
  1. Read a DC voltage:
         v = dmm.meas_vdc()
  2. Read DC current on one fixed range for a whole test (the burden
     resistance changes with the range):
         i = dmm.meas_idc(range=3)          # then dmm.meas_idc() in the loop
  3. Trade speed for noise:
         v = dmm.meas_vdc(range=10, nplc=100)
  4. Take a block of readings in the meter (512 readings of memory):
         values = dmm.meas_buffer(50)
  5. Read the statistics the math block accumulated:
         s = dmm.meas_stats()
"""

from __future__ import annotations

import contextlib
import math
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pyvisa as visa

from common.errors import InstrumentError, InstrumentUnresponsiveError
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import ScpiEnum

_DEFAULT_TIMEOUT_MS = 5000
_DEFAULT_PACING_S = 0.02
_REMOTE_SETTLE_S = 0.1
_OVERLOAD = 1e37
_IDN_RE = re.compile(r"34401A", re.IGNORECASE)

# Integration time in seconds for one power-line cycle at 50 Hz (worst case).
_PLC_S = 0.02
# Readings the meter holds in memory.
MEMORY_READINGS = 512


def _reading(resp: str) -> float:
    """Parse one NR3 reading; overload (9.9E37) or text reads as NaN."""
    try:
        value = float(resp.strip())
    except ValueError:
        return math.nan
    return math.nan if abs(value) >= _OVERLOAD else value


def _num(value: float | str) -> str:
    """A numeric argument: a token (MIN, MAX, DEF) or a number."""
    return value.upper() if isinstance(value, str) else f"{value:.12g}"


class Function(ScpiEnum):
    """A measurement function. The ``meas_*`` method name picks it."""

    VOLTAGE_DC = "VOLT:DC"
    VOLTAGE_DC_RATIO = "VOLT:DC:RAT"
    VOLTAGE_AC = "VOLT:AC"
    CURRENT_DC = "CURR:DC"
    CURRENT_AC = "CURR:AC"
    RESISTANCE_2W = "RES"
    RESISTANCE_4W = "FRES"
    FREQUENCY = "FREQ"
    PERIOD = "PER"
    CONTINUITY = "CONT"
    DIODE = "DIOD"


class TriggerSource(ScpiEnum):
    """What starts a measurement."""

    IMMEDIATE = "IMM"
    BUS = "BUS"
    EXTERNAL = "EXT"


class AutoZero(ScpiEnum):
    """Auto-zero: a zero reading after each reading (ON), once (ONCE), or never."""

    ON = "ON"
    OFF = "OFF"
    ONCE = "ONCE"


# FUNC? answers without the :DC suffix (manual, FUNCtion? query; bench 2026-11-07).
_FUNCTION_READBACK = {"VOLT": "VOLT:DC", "CURR": "CURR:DC", "VOLT:RAT": "VOLT:DC:RAT"}

# Functions with a range, a resolution and (some) an integration time.
_RANGED = {"VOLT:DC", "VOLT:AC", "CURR:DC", "CURR:AC", "RES", "FRES"}
_NPLC = {"VOLT:DC", "CURR:DC", "RES", "FRES"}
# Frequency and period take an input voltage range, not a range of their own.
_RANGE_HEADER = {f: f for f in _RANGED} | {"FREQ": "FREQ:VOLT", "PER": "PER:VOLT",
                                           "VOLT:DC:RAT": "VOLT:DC"}

RANGES = {
    "VOLT:DC": (0.1, 1, 10, 100, 1000),
    "VOLT:AC": (0.1, 1, 10, 100, 750),
    "CURR:DC": (0.01, 0.1, 1, 3),
    "CURR:AC": (1, 3),
    "RES": (100, 1e3, 10e3, 100e3, 1e6, 10e6, 100e6),
    "FRES": (100, 1e3, 10e3, 100e3, 1e6, 10e6, 100e6),
}


def _token(value, enum_cls: type[ScpiEnum], label: str) -> str:
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    for member in enum_cls:
        if text in (member.name, str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")


@dataclass(frozen=True)
class Stats:
    """Statistics of the math block (CALC:FUNC AVER) since it was turned on."""

    average: float
    minimum: float
    maximum: float
    count: int


class _DmmLink(InstrumentLink):
    """Link for the 34401A: RS-232 per the manual (8N2, DTR/DSR), or GPIB as is.

    A VISA timeout becomes :class:`InstrumentUnresponsiveError`, so a caller
    stops instead of retrying into a desynced serial stream.
    """

    baud_rate = 9600

    def __init__(self, resource_name: str, timeout_ms: int, min_command_interval: float):
        super().__init__(resource_name, timeout_ms=timeout_ms)
        self.min_command_interval = min_command_interval
        self._last_io = 0.0

    @property
    def is_serial(self) -> bool:
        return self.resource_name.upper().startswith("ASRL")

    def connect(self, timeout_ms: int | None = None) -> None:
        rm = visa.ResourceManager()
        self._instr = self._open_resource(rm, self.resource_name)
        instr = self._instr
        if self.is_serial:
            instr.baud_rate = self.baud_rate
            instr.data_bits = 8
            instr.parity = visa.constants.Parity.none
            instr.stop_bits = visa.constants.StopBits.two
            with contextlib.suppress(Exception):
                instr.flow_control = visa.constants.ControlFlow.dtr_dsr
        instr.write_termination = "\n"
        instr.read_termination = "\n"
        instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms
        if self.is_serial:
            # Drop any stale reply from an earlier session before the first command.
            with contextlib.suppress(Exception):
                instr.flush(visa.constants.BufferOperation.discard_read_buffer)

    def _pace(self) -> None:
        wait = self.min_command_interval - (time.monotonic() - self._last_io)
        if wait > 0:
            time.sleep(wait)

    def _io(self, fn, message, **kwargs):
        self._pace()
        try:
            return fn(message, **kwargs)
        except visa.VisaIOError as exc:
            if exc.error_code == visa.constants.StatusCode.error_timeout:
                raise InstrumentUnresponsiveError(
                    f"34401A did not answer {message!r} (VISA timeout). Stop; read "
                    "SYST:ERR? on a fresh link before more commands."
                ) from exc
            raise
        finally:
            self._last_io = time.monotonic()

    def write(self, message: str, **kwargs) -> int:
        return self._io(super().write, message, **kwargs)

    def query(self, message: str, **kwargs) -> str:
        return self._io(super().query, message, **kwargs)


class Keysight_34401A(Instrument):
    """HP / Agilent / Keysight 34401A digital multimeter on RS-232 or GPIB."""

    Function = Function
    TriggerSource = TriggerSource
    AutoZero = AutoZero

    def __init__(
        self,
        resource_name: str,
        *,
        pacing_s: float = _DEFAULT_PACING_S,
        connect_timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        super().__init__(resource_name, name="DMM")
        self.link = _DmmLink(resource_name, connect_timeout_ms, pacing_s)
        Node.__init__(self, self.link, name="DMM")
        self.resource_name = resource_name
        self.model: str | None = None
        self.firmware: str | None = None
        self._function: str | None = None
        self._nplc: float | None = None  # of the active function; None = unknown

    def _set_active(self, function: str | None) -> None:
        if function != self._function:
            self._nplc = None
        self._function = function

    # ---------- connection ----------

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the link, enter remote (RS-232), clear status, check the identity."""
        self.link.connect(timeout_ms=timeout_ms)
        if self.link.is_serial:
            self.link.write("SYST:REM")
            time.sleep(_REMOTE_SETTLE_S)
        self.link.write("*CLS")
        idn = self.link.query("*IDN?").strip()
        if not _IDN_RE.search(idn):
            self.link.disconnect()
            raise InstrumentError(f"Not a 34401A (IDN={idn!r}) on {self.resource_name}")
        self.idn = idn
        fields = [f.strip() for f in idn.split(",")]
        self.model = fields[1] if len(fields) > 1 else None
        self.firmware = fields[3] if len(fields) > 3 else None
        self._set_active(None)
        return idn

    def safe_state(self) -> None:
        """Return the front panel (RS-232). A meter has no output to turn off."""
        if self.link.is_serial:
            with contextlib.suppress(Exception):
                self.link.write("SYST:LOC")

    def cmd_remote(self) -> None:
        """Enter remote (RS-232), then wait for the meter to accept commands."""
        self.link.write("SYST:REM")
        time.sleep(_REMOTE_SETTLE_S)

    def cmd_local(self) -> None:
        """Return the front panel to the operator (RS-232)."""
        self.link.write("SYST:LOC")

    def cmd_remote_lock(self) -> None:
        """Remote with the front-panel LOCAL key locked out (RS-232)."""
        self.link.write("SYST:RWL")
        time.sleep(_REMOTE_SETTLE_S)

    def reset(self) -> None:
        """Factory state: DC volts, autorange, 10 PLC, trigger immediate, one sample."""
        self.link.write("*RST")
        self.link.query("*OPC?")
        self._set_active(None)

    # ---------- measurements ----------

    def meas_vdc(self, *, range: float | str | None = None, nplc: float | None = None) -> float:
        """DC voltage in V. ``range``: 0.1 to 1000 V, or "AUTO"."""
        return self._measure("VOLT:DC", range=range, nplc=nplc)

    def meas_vdc_ratio(self, *, range: float | str | None = None, nplc: float | None = None) -> float:
        """Ratio of the input to the 4-wire sense reference voltage."""
        return self._measure("VOLT:DC:RAT", range=range, nplc=nplc)

    def meas_vac(self, *, range: float | str | None = None, bandwidth: float | None = None) -> float:
        """AC voltage in V RMS. ``bandwidth``: AC filter of 3, 20 or 200 Hz."""
        return self._measure("VOLT:AC", range=range, bandwidth=bandwidth)

    def meas_idc(self, *, range: float | str | None = None, nplc: float | None = None) -> float:
        """DC current in A. ``range``: 0.01, 0.1, 1 or 3 A, or "AUTO".

        The shunt (burden) resistance changes with the range. Give a fixed
        range for a test that must keep one burden throughout.
        """
        return self._measure("CURR:DC", range=range, nplc=nplc)

    def meas_iac(self, *, range: float | str | None = None, bandwidth: float | None = None) -> float:
        """AC current in A RMS. ``range``: 1 or 3 A, or "AUTO"."""
        return self._measure("CURR:AC", range=range, bandwidth=bandwidth)

    def meas_resistance_2w(self, *, range: float | str | None = None, nplc: float | None = None) -> float:
        """2-wire resistance in ohm."""
        return self._measure("RES", range=range, nplc=nplc)

    def meas_resistance_4w(self, *, range: float | str | None = None, nplc: float | None = None) -> float:
        """4-wire resistance in ohm (sense leads on the HI/LO sense terminals)."""
        return self._measure("FRES", range=range, nplc=nplc)

    def meas_frequency(self, *, range: float | str | None = None, aperture: float | None = None) -> float:
        """Frequency in Hz, 3 Hz to 300 kHz. ``range``: the input voltage range."""
        return self._measure("FREQ", range=range, aperture=aperture)

    def meas_period(self, *, range: float | str | None = None, aperture: float | None = None) -> float:
        """Period in s. ``range``: the input voltage range."""
        return self._measure("PER", range=range, aperture=aperture)

    def meas_continuity(self) -> float:
        """Continuity: resistance in ohm on the fixed 1 kohm range."""
        return self._measure("CONT")

    def meas_diode(self) -> float:
        """Diode forward voltage in V at 1 mA."""
        return self._measure("DIOD")

    def meas_reading(self) -> float:
        """One reading with the present setup."""
        return _reading(self.link.query("READ?"))

    def meas_buffer(self, count: int, *, function: Function | str | None = None) -> list[float]:
        """Take ``count`` readings into the meter memory, then fetch them all.

        Uses the present setup, or configures ``function`` first. The trigger
        source stays as it is; with BUS or EXTERNAL the readings wait for those.
        """
        if not 1 <= int(count) <= MEMORY_READINGS:
            raise ValueError(f"count {count}: 1 to {MEMORY_READINGS}")
        if function is not None:
            self.set_function(function)
        self.link.write(f"SAMP:COUN {int(count)}")
        timeout = self._reading_timeout_ms(int(count))
        try:
            self.link.write("INIT")
            with self.link.temporary_timeout(timeout):
                resp = self.link.query("FETC?")
        finally:
            self.link.write("SAMP:COUN 1")
        return [_reading(v) for v in resp.split(",") if v.strip()]

    def meas_stats(self) -> Stats:
        """Average, minimum, maximum and count from the average math function."""
        return Stats(
            average=_reading(self.link.query("CALC:AVER:AVER?")),
            minimum=_reading(self.link.query("CALC:AVER:MIN?")),
            maximum=_reading(self.link.query("CALC:AVER:MAX?")),
            count=int(float(self.link.query("CALC:AVER:COUN?"))),
        )

    # ---------- configuration ----------

    def set_function(self, function: Function | str) -> str:
        """Select the function with its present range and resolution; read back."""
        token = _token(function, Function, "function")
        self.link.write(f'FUNC "{token}"')
        self._set_active(self.get_function())
        if self._function != token:
            raise InstrumentError(f"function {token} sent, reads {self._function}")
        return self._function

    def get_function(self) -> str:
        """The active function as the canonical token, with the DC suffix."""
        raw = self.link.query("FUNC?").strip().strip('"').upper()
        return _FUNCTION_READBACK.get(raw, raw)

    def get_configuration(self) -> str:
        """Function, range and resolution, as the meter reports them."""
        return self.link.query("CONF?").strip().strip('"')

    def set_range(self, range: float | str, function: Function | str | None = None) -> float:
        """Set a fixed range (autorange goes off) or "AUTO"; return the range read back."""
        f = self._func(function)
        head = _RANGE_HEADER[f]
        if isinstance(range, str) and range.strip().upper() == "AUTO":
            self.link.write(f"{head}:RANG:AUTO ON")
        else:
            self.link.write(f"{head}:RANG {_num(range)}")
        return self.get_range(f)

    def get_range(self, function: Function | str | None = None) -> float:
        """The range of the function, in its unit."""
        return float(self.link.query(f"{_RANGE_HEADER[self._func(function)]}:RANG?"))

    def set_autorange(self, state: bool, function: Function | str | None = None) -> bool:
        """Turn autorange on or off for the function; return the read-back."""
        f = self._func(function)
        self.link.write(f"{_RANGE_HEADER[f]}:RANG:AUTO {'ON' if state else 'OFF'}")
        return self.get_autorange(f)

    def get_autorange(self, function: Function | str | None = None) -> bool:
        """True if autorange is on for the function."""
        return self.link.query(f"{_RANGE_HEADER[self._func(function)]}:RANG:AUTO?").strip() == "1"

    def set_resolution(self, resolution: float | str, function: Function | str | None = None) -> float:
        """Resolution in the function's unit; the meter picks the integration time from it."""
        f = self._func(function)
        self.link.write(f"{_RANGE_HEADER[f]}:RES {_num(resolution)}")
        return self.get_resolution(f)

    def get_resolution(self, function: Function | str | None = None) -> float:
        """The resolution of the function, in its unit."""
        return float(self.link.query(f"{_RANGE_HEADER[self._func(function)]}:RES?"))

    def set_nplc(self, nplc: float | str, function: Function | str | None = None) -> float:
        """Integration time in power-line cycles: 0.02, 0.2, 1, 10 or 100."""
        f = self._func(function)
        if f not in _NPLC:
            raise ValueError(f"{f} has no integration time; DC V, DC I and resistance have")
        self.link.write(f"{f}:NPLC {_num(nplc)}")
        return self.get_nplc(f)

    def get_nplc(self, function: Function | str | None = None) -> float:
        """The integration time of the function, in power-line cycles."""
        f = self._func(function)
        value = float(self.link.query(f"{f}:NPLC?"))
        if f == self._function:
            self._nplc = value
        return value

    def set_bandwidth(self, hz: float | str) -> float:
        """AC filter corner for AC volts and AC current: 3, 20 or 200 Hz."""
        self.link.write(f"DET:BAND {_num(hz)}")
        return self.get_bandwidth()

    def get_bandwidth(self) -> float:
        """The AC filter corner in Hz."""
        return float(self.link.query("DET:BAND?"))

    def set_aperture(self, seconds: float | str, function: Function | str | None = None) -> float:
        """Gate time for frequency and period: 0.01, 0.1 or 1 s."""
        f = self._func(function)
        if f not in ("FREQ", "PER"):
            raise ValueError(f"{f} has no aperture; frequency and period have")
        self.link.write(f"{f}:APER {_num(seconds)}")
        return float(self.link.query(f"{f}:APER?"))

    def set_autozero(self, mode: AutoZero | str | bool) -> str:
        """Auto-zero ON, OFF or ONCE (DC volts, DC current, resistance)."""
        token = ("ON" if mode else "OFF") if isinstance(mode, bool) else _token(mode, AutoZero, "auto-zero")
        self.link.write(f"ZERO:AUTO {token}")
        return self.get_autozero()

    def get_autozero(self) -> str:
        """ON or OFF. ONCE reads back as OFF after its one zero reading."""
        return "ON" if self.link.query("ZERO:AUTO?").strip() == "1" else "OFF"

    def set_input_impedance_auto(self, state: bool) -> bool:
        """DC volts input: True gives over 10 Gohm on the 0.1 to 10 V ranges; False 10 Mohm."""
        self.link.write(f"INP:IMP:AUTO {'ON' if state else 'OFF'}")
        return self.get_input_impedance_auto()

    def get_input_impedance_auto(self) -> bool:
        """True: over 10 Gohm on the low DC volt ranges. False: 10 Mohm."""
        return self.link.query("INP:IMP:AUTO?").strip() == "1"

    def get_terminals(self) -> str:
        """Front or rear: the position of the front-panel terminal switch."""
        return self.link.query("ROUT:TERM?").strip()

    # ---------- trigger ----------

    def set_trigger_source(self, source: TriggerSource | str) -> str:
        """Set what starts a measurement; return the read-back."""
        self.link.write(f"TRIG:SOUR {_token(source, TriggerSource, 'trigger source')}")
        return self.get_trigger_source()

    def get_trigger_source(self) -> str:
        """The trigger source: immediate, bus or external."""
        return self.link.query("TRIG:SOUR?").strip()

    def set_trigger_delay(self, seconds: float | str) -> float:
        """Delay from trigger to reading in s, or "AUTO" for the meter's own delay."""
        if isinstance(seconds, str) and seconds.strip().upper() == "AUTO":
            self.link.write("TRIG:DEL:AUTO ON")
        else:
            self.link.write(f"TRIG:DEL {_num(seconds)}")
        return self.get_trigger_delay()

    def get_trigger_delay(self) -> float:
        """Delay from trigger to reading, in s."""
        return float(self.link.query("TRIG:DEL?"))

    def get_trigger_delay_auto(self) -> bool:
        """True if the meter picks the trigger delay itself."""
        return self.link.query("TRIG:DEL:AUTO?").strip() == "1"

    def set_sample_count(self, count: int | str) -> int:
        """Readings per trigger, 1 to 50000; return the read-back."""
        self.link.write(f"SAMP:COUN {_num(count)}")
        return self.get_sample_count()

    def get_sample_count(self) -> int:
        """Readings per trigger."""
        return int(float(self.link.query("SAMP:COUN?")))

    def set_trigger_count(self, count: int | str) -> float:
        """Triggers per INIT; "INF" runs until a device clear."""
        self.link.write(f"TRIG:COUN {_num(count)}")
        return self.get_trigger_count()

    def get_trigger_count(self) -> float:
        """Triggers per measurement start; 9.9E37 means infinite."""
        return float(self.link.query("TRIG:COUN?"))

    def cmd_trigger(self) -> None:
        """Software trigger, for trigger source BUS."""
        self.link.write("*TRG")

    # ---------- math ----------

    def set_math(self, function: str | None) -> str | None:
        """Math NULL, DB, DBM, AVER or LIM; None turns math off."""
        if function is None:
            self.link.write("CALC:STAT OFF")
            return None
        token = function.strip().upper()[:4]
        if token not in ("NULL", "DB", "DBM", "AVER", "LIM"):
            raise ValueError(f"math {function!r}: NULL, DB, DBM, AVERAGE or LIMIT")
        self.link.write(f"CALC:FUNC {token}")
        self.link.write("CALC:STAT ON")
        return self.get_math()

    def get_math(self) -> str | None:
        """The active math function, or None when math is off."""
        if self.link.query("CALC:STAT?").strip() != "1":
            return None
        return self.link.query("CALC:FUNC?").strip()

    def set_null_offset(self, value: float | str) -> float:
        """Set the value that NULL math subtracts; return the read-back."""
        self.link.write(f"CALC:NULL:OFFS {_num(value)}")
        return float(self.link.query("CALC:NULL:OFFS?"))

    # ---------- system ----------

    def set_display(self, state: bool) -> bool:
        """Turn the display on or off; return the read-back."""
        self.link.write(f"DISP {'ON' if state else 'OFF'}")
        return self.link.query("DISP?").strip() == "1"

    def set_display_text(self, text: str | None) -> None:
        """Show up to 12 characters on the display; None clears it."""
        if text is None:
            self.link.write("DISP:TEXT:CLE")
        else:
            self.link.write(f'DISP:TEXT "{text[:12]}"')

    def cmd_beep(self) -> None:
        """Sound the beeper once."""
        self.link.write("SYST:BEEP")

    def get_version(self) -> str:
        """The SCPI version the meter conforms to."""
        return self.link.query("SYST:VERS?").strip()

    def get_errors(self) -> list[str]:
        """Drain and return the error queue (up to 20 entries); empty means clean."""
        found: list[str] = []
        for _ in range(25):
            resp = self.link.query("SYST:ERR?").strip()
            if resp.split(",", 1)[0].strip().lstrip("+") == "0":
                break
            found.append(resp)
        return found

    def check_errors(self) -> None:
        """Raise if the error queue holds entries; the queue is empty after."""
        errors = self.get_errors()
        if errors:
            raise InstrumentError("34401A reported errors:\n  " + "\n  ".join(errors))

    # ---------- preserved setup ----------

    def snapshot(self) -> dict:
        """Read back the settings this driver changes, for a restore."""
        f = self.get_function()
        snap = {"function": f, "trigger_source": self.get_trigger_source(),
                "sample_count": self.get_sample_count(), "trigger_count": self.get_trigger_count(),
                "trigger_delay_auto": self.get_trigger_delay_auto(),
                "trigger_delay": self.get_trigger_delay()}
        if f in _RANGE_HEADER:
            snap["autorange"] = self.get_autorange(f)
            snap["range"] = self.get_range(f)
        if f in _NPLC:
            snap["nplc"] = self.get_nplc(f)
        return snap

    def restore(self, snap: dict) -> None:
        """Write back the settings that ``snapshot`` read."""
        f = snap["function"]
        self.link.write(f'FUNC "{f}"')
        self._set_active(f)
        if "nplc" in snap:
            self.set_nplc(snap["nplc"], f)
        if "autorange" in snap:
            self.set_range("AUTO" if snap["autorange"] else snap["range"], f)
        self.set_trigger_source(snap["trigger_source"])
        self.set_sample_count(snap["sample_count"])
        self.set_trigger_count(snap["trigger_count"])
        if snap["trigger_delay_auto"]:
            self.set_trigger_delay("AUTO")
        else:
            self.set_trigger_delay(snap["trigger_delay"])

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator[dict]:
        """Snapshot on enter, restore on exit."""
        state = self.snapshot()
        try:
            yield state
        finally:
            self.restore(state)

    # ---------- plumbing ----------

    def _func(self, function) -> str:
        if function is not None:
            return _token(function, Function, "function")
        if self._function is None:
            self._set_active(self.get_function())
        return self._function

    def _reading_timeout_ms(self, count: int = 1) -> int:
        """Worst-case time for ``count`` readings at 100 PLC with auto-zero, plus margin."""
        # The cached integration time, so a read costs no extra serial query.
        nplc = self._nplc if self._nplc is not None else 100.0
        per =2 * nplc * _PLC_S + 0.05
        return int(max(self.link.default_timeout_ms, 2000 + 1000 * count * per * 1.5))

    def _measure(self, func: str, *, range=None, nplc=None, bandwidth=None, aperture=None) -> float:
        """Configure only if asked or if another function is active, then read."""
        if self._function is None:
            self._set_active(self.get_function())
        configure = any(v is not None for v in (range, nplc, bandwidth, aperture))
        if configure or self._function != func:
            if func in ("CONT", "DIOD"):
                self.link.write(f"CONF:{func}")
            elif range is None or (isinstance(range, str) and range.strip().upper() == "AUTO"):
                self.link.write(f"CONF:{func} DEF")
            else:
                # CONF with a range sets that range and turns autorange off.
                self.link.write(f"CONF:{func} {_num(range)}")
            self._set_active(func)
            self._nplc = None  # CONF sets the integration time from the resolution
            if nplc is not None:
                self.set_nplc(nplc, func)
            if bandwidth is not None:
                self.set_bandwidth(bandwidth)
            if aperture is not None:
                self.set_aperture(aperture, func)
            if range is not None and not isinstance(range, str):
                got = self.get_range(func)
                if not math.isclose(got, float(range), rel_tol=1e-6):
                    raise InstrumentError(f"{func} range {range} sent, reads {got}")
        with self.link.temporary_timeout(self._reading_timeout_ms()):
            return _reading(self.link.query("READ?"))
