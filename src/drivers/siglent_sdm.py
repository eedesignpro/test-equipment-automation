"""Driver for the Siglent SDM4000A-series bench digital multimeters.

Public import path::

    from drivers.siglent_sdm import Siglent_SDM

Shape follows ``docs/driver-api-design.md`` (section 8, SDM4065A). Measurement
is the whole instrument here, so the ``meas_*`` methods sit at the top level and
there is no measure node (standard 5.4). The method name picks the function, and
every configuration argument is keyword-only, because the name names only the
function.

The bench unit is an SDM4065A (6.5 digit) on a raw socket at port 5025. An
over-range reads back as 9.9E37 and is returned as ``float('nan')``, never
raised, so a sweep never stops on one.

Bench observations behind the choices in this file (ADR-042). All measured on
the bench SDM4065A on 2026-08-10 with ``projects/dmm_probe.py``:

- **The set token and the read-back token differ for the DC functions.** Writing
  ``VOLT:DC`` reads back as ``VOLT``, and ``CURR:DC`` reads back as ``CURR``.
  The AC forms echo verbatim. :meth:`get_function` normalizes the read-back, so
  a caller always sees the canonical token it wrote.
- **This DMM has an error queue**, unlike the ODP supply. ``SYST:ERR?`` answers,
  so :meth:`get_errors` drains it and a rejected command can be caught.
- **Auto-zero does not exist on this instrument.** ``VOLT:DC:AZ?`` and the same
  header on CURR:DC, RES and FRES are all undefined. An earlier driver exposed
  ``set_autozero`` on four nodes and sent the header inside a suppressed block,
  so it silently did nothing and queued an error each time. There is no
  auto-zero method here, because there is no auto-zero (standard 5.1).
- Per-function support, measured rather than assumed: NPLC exists on the DC
  functions and on both resistance functions, and NOT on AC, capacitance or
  temperature. Bandwidth exists on the AC functions only. Input impedance is DC
  voltage only. Frequency and period take an aperture and an input VOLTage
  range, and have no range of their own.
- A query costs about 1.4 ms and a write about 0.06 ms.
- ``READ?`` costs what the integration time costs: 8 ms at 0.1 NPLC, 64 ms at
  1 NPLC, 404 ms at 10 NPLC. ``MEAS:<f>?`` always costs about 410 ms, because
  it reconfigures the function to its defaults first, which includes 10 NPLC.
  So a ``meas_*`` call with no keyword arguments does NOT reconfigure. It
  selects the function if needed and reads, which keeps a measurement loop at
  the speed the operator configured. Passing any keyword argument reconfigures.

Bench topology: SMU CH1 -> DMM voltage input, 2-wire on both instruments. The
4-wire resistance function is implemented, because the instrument has it, but it
cannot be exercised in that wiring.

Task recipes::

    dmm.connect()

    # 1. One DC voltage reading:
    v = dmm.meas_vdc()

    # 2. A fast quiet loop: configure once, then read at that speed:
    dmm.dcv.set_nplc(0.1)
    for _ in range(100):
        print(dmm.meas_vdc())

    # 3. One reading at an explicit range and integration time:
    v = dmm.meas_vdc(range=2.0, nplc=10)

    # 4. Capture a block of readings into instrument memory and fetch them:
    block = dmm.meas_buffer(1000, function=dmm.Function.VOLTAGE_DC, nplc=0.001)

    # 5. Statistics over what the instrument has accumulated:
    s = dmm.meas_stats()                      # s.average, s.sdev, s.minimum

Command surface follows the Siglent SDM4000A Remote Manual.

Common tasks:
  1. Read a DC voltage:
         v = dmm.meas_vdc()
  2. Read a resistance through 4-wire sense:
         r = dmm.meas_resistance_4w()
  3. Trade speed for resolution on one reading:
         v = dmm.meas_vdc(range=10.0, nplc=100)
  4. Take a burst of readings without a Python loop:
         values = dmm.meas_buffer(200, nplc=1)
  5. Read the statistics the instrument accumulated over that burst:
         stats = dmm.meas_stats()
"""

from __future__ import annotations

import contextlib

from collections.abc import Iterator
import re
import time
from dataclasses import dataclass

import pyvisa as visa

from common.errors import InstrumentError
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import ScpiEnum

_DEFAULT_TIMEOUT_MS = 5000

# Capacitance and temperature read slowly enough to outlive the normal timeout.
_SLOW_MEASURE_TIMEOUT_MS = 20000

# A deep buffer capture can run for minutes at a high sample count.
_BUFFER_TIMEOUT_MS = 300000

# IDN form: Siglent Technologies,SDM4065A,<serial>,<fw>
_IDN_RE = re.compile(r",(SDM40\d{2}[A-Z]),")

_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

# Readings at or above this magnitude are the instrument's overload sentinel.
_OVERLOAD = 1e37


def _reading(resp: str) -> float:
    """Parse one NR3 reading; return NaN for overload (9.9E37) or unparseable."""
    first = resp.strip().split(",")[0].strip()
    try:
        value = float(first)
    except ValueError:
        match = _FLOAT_RE.search(first)
        if not match:
            return float("nan")
        value = float(match.group())
    return float("nan") if abs(value) >= _OVERLOAD else value


def _readings(resp: str) -> list[float]:
    """Parse a comma-separated list of NR3 readings (overload -> NaN)."""
    return [_reading(f) for f in resp.strip().split(",") if f.strip()]


def _onoff(state: bool) -> str:
    return "ON" if state else "OFF"


def _num(value: float | str) -> str:
    """Format a range or level argument: a token such as AUTO, or a number."""
    return value if isinstance(value, str) else f"{value:g}"


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


@dataclass(frozen=True)
class Stats:
    """Statistics the instrument accumulated over its readings."""

    average: float
    sdev: float
    minimum: float
    maximum: float
    peak_to_peak: float
    count: int


class _SdmSocketLink(InstrumentLink):
    """InstrumentLink for the SDM raw-socket connection.

    Opens with ``\\n`` terminators and WITHOUT a VISA ``clear``, because a raw
    socket does not carry IEEE-488 device clear.
    """

    def connect(self, timeout_ms: int | None = None) -> None:
        rm = visa.ResourceManager()
        self._instr = self._open_resource(rm, self.resource_name)
        with contextlib.suppress(Exception):
            self._instr.write_termination = "\n"
            self._instr.read_termination = "\n"
        self._instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms


class Siglent_SDM(Instrument):
    """Siglent SDM4000A-series digital multimeter on a LAN raw socket."""

    class Function(ScpiEnum):
        """A measurement function. The ``meas_*`` method name usually picks it."""

        VOLTAGE_DC = "VOLT:DC"
        VOLTAGE_AC = "VOLT:AC"
        CURRENT_DC = "CURR:DC"
        CURRENT_AC = "CURR:AC"
        RESISTANCE_2W = "RES"
        RESISTANCE_4W = "FRES"
        FREQUENCY = "FREQ"
        PERIOD = "PER"
        CAPACITANCE = "CAP"
        CONTINUITY = "CONT"
        DIODE = "DIOD"
        TEMPERATURE = "TEMP"

    class TriggerSource(ScpiEnum):
        """What starts a measurement."""

        IMMEDIATE = "IMM"
        EXTERNAL = "EXT"
        BUS = "BUS"
        INTERNAL = "INT"

    class Slope(ScpiEnum):
        """Which edge an external trigger fires on."""

        POSITIVE = "POS"
        NEGATIVE = "NEG"

    class InputImpedance(ScpiEnum):
        """DC voltage input impedance, on the low ranges only."""

        TEN_MEGAOHM = "10M"
        TEN_GIGAOHM = "10G"

    class TemperatureUnit(ScpiEnum):
        """Unit for every temperature reading."""

        CELSIUS = "C"
        FAHRENHEIT = "F"
        KELVIN = "K"

    class ScaleFunction(ScpiEnum):
        """Scaling applied to a reading by the math block."""

        DB = "DB"
        DBM = "DBM"

    # The DC functions read back without their :DC suffix (bench-verified), so a
    # read-back token maps through this to the canonical token that was written.
    _FUNCTION_READBACK = {"VOLT": "VOLT:DC", "CURR": "CURR:DC"}

    dcv: _DcVoltage
    acv: _AcFunction
    dci: _DcFunction
    aci: _AcFunction
    res: _DcFunction
    fres: _DcFunction
    cap: _BasicFunction
    freq: _FreqPeriod
    period: _FreqPeriod
    temp: _Temperature
    cont: _Continuity
    trigger: _Trigger
    calc: _Calc
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
            pyvisa resource for the DMM (default the bench raw-socket string).
        connect_timeout_ms : int
            I/O timeout used for connect and normal operation, in milliseconds.
        """
        super().__init__(resource_name, name="DMM")
        self.link = _SdmSocketLink(resource_name, timeout_ms=connect_timeout_ms)
        Node.__init__(self, self.link, name="DMM")
        self.resource_name = resource_name
        self.model: str | None = None
        self.serial: str | None = None
        self.firmware: str | None = None

        # Per-function configuration. Each node carries only the controls that
        # this instrument really has for that function, so nothing here raises
        # to say "not supported" (standard 5.1).
        self.dcv = _DcVoltage(self.link, "DCV", "VOLT:DC")
        self.acv = _AcFunction(self.link, "ACV", "VOLT:AC")
        self.dci = _DcFunction(self.link, "DCI", "CURR:DC")
        self.aci = _AcFunction(self.link, "ACI", "CURR:AC")
        self.res = _DcFunction(self.link, "RES", "RES")
        self.fres = _DcFunction(self.link, "FRES", "FRES")
        self.cap = _BasicFunction(self.link, "CAP", "CAP")
        self.freq = _FreqPeriod(self.link, "FREQ", "FREQ")
        self.period = _FreqPeriod(self.link, "PERIOD", "PER")
        self.temp = _Temperature(self.link, "TEMP")
        self.cont = _Continuity(self.link, "CONT")

        self.trigger = _Trigger(self.link, name="TRIGGER")
        self.calc = _Calc(self.link, name="CALC")
        self.status = _Status(self.link, name="STATUS")

    # ---------- connection lifecycle ----------

    def _on_connect(self, idn: str) -> None:
        match = _IDN_RE.search(idn)
        if not match:
            raise InstrumentError(
                f"{self.resource_name} did not answer with a Siglent SDM40xx identity: {idn!r}. "
                "Check the resource string; this driver does not scan for the instrument."
            )
        self.model = match.group(1)
        fields = [f.strip() for f in idn.split(",")]
        self.serial = fields[2] if len(fields) >= 3 else None
        self.firmware = fields[3] if len(fields) >= 4 else None

    def safe_state(self) -> None:
        """Return the front panel to local control. A DMM has nothing to de-energize."""
        with contextlib.suppress(Exception):
            self.link.write("SYST:LOC")

    # ---------- measurements ----------

    def meas_vdc(self, *, range: float | str | None = None, nplc: float | None = None) -> float:
        """Measure DC voltage in volts; an overload reads as NaN.

        ``range`` is in volts. ``nplc`` is the integration time in power-line
        cycles, where more cycles means slower and quieter. Passing either one
        reconfigures the function; passing neither keeps the present setup and
        just reads, which is what makes a measurement loop fast.
        """
        return self._measure("VOLT:DC", range=range, nplc=nplc)

    def meas_vac(
        self, *, range: float | str | None = None, bandwidth: float | None = None
    ) -> float:
        """Measure AC voltage in volts RMS; an overload reads as NaN.

        ``bandwidth`` is the AC filter corner in Hz: 3, 20 or 200. This function
        has no NPLC on this instrument.
        """
        return self._measure("VOLT:AC", range=range, bandwidth=bandwidth)

    def meas_idc(self, *, range: float | str | None = None, nplc: float | None = None) -> float:
        """Measure DC current in amperes; an overload reads as NaN."""
        return self._measure("CURR:DC", range=range, nplc=nplc)

    def meas_iac(
        self, *, range: float | str | None = None, bandwidth: float | None = None
    ) -> float:
        """Measure AC current in amperes RMS; an overload reads as NaN."""
        return self._measure("CURR:AC", range=range, bandwidth=bandwidth)

    def meas_resistance_2w(
        self, *, range: float | str | None = None, nplc: float | None = None
    ) -> float:
        """Measure resistance in ohms with two wires; an overload reads as NaN."""
        return self._measure("RES", range=range, nplc=nplc)

    def meas_resistance_4w(
        self, *, range: float | str | None = None, nplc: float | None = None
    ) -> float:
        """Measure resistance in ohms with four wires; an overload reads as NaN.

        This needs sense leads on the DUT. The bench wiring has none, so a
        reading here is meaningless until somebody wires it.
        """
        return self._measure("FRES", range=range, nplc=nplc)

    def meas_frequency(
        self, *, range: float | str | None = None, aperture: float | None = None
    ) -> float:
        """Measure frequency in Hz; an overload reads as NaN.

        ``range`` is the INPUT VOLTAGE range in volts, not a frequency range.
        ``aperture`` is the gate time in seconds: 0.001, 0.01, 0.1 or 1.
        """
        return self._measure("FREQ", voltage_range=range, aperture=aperture)

    def meas_period(
        self, *, range: float | str | None = None, aperture: float | None = None
    ) -> float:
        """Measure period in seconds; an overload reads as NaN."""
        return self._measure("PER", voltage_range=range, aperture=aperture)

    def meas_capacitance(self, *, range: float | str | None = None) -> float:
        """Measure capacitance in farads; an overload reads as NaN.

        Slow to settle, so this call carries a longer timeout than the others.
        """
        return self._measure("CAP", range=range, timeout_ms=_SLOW_MEASURE_TIMEOUT_MS)

    def meas_diode(self) -> float:
        """Measure a diode's forward voltage in volts; an open reads as NaN."""
        return self._measure("DIOD")

    def meas_continuity(self) -> float:
        """Measure continuity resistance in ohms; an open reads as NaN."""
        return self._measure("CONT")

    def meas_temperature(self, *, probe: str | None = None, sensor: str | None = None) -> float:
        """Measure temperature in the configured unit; an overload reads as NaN.

        ``probe`` is RTD or THER, and ``sensor`` the type within it, such as
        PT100. Both left at None keep the configured transducer.
        """
        if probe is not None and sensor is None:
            raise ValueError(
                "meas_temperature() needs both probe and sensor to select a transducer, "
                "for example probe='RTD', sensor='PT100'. Pass neither to keep the "
                "transducer the instrument already has."
            )
        if probe is not None:
            self.temp.set_transducer(probe, sensor)
        return self._measure("TEMP", timeout_ms=_SLOW_MEASURE_TIMEOUT_MS)

    def meas_buffer(
        self,
        count: int,
        *,
        function: Function | str = Function.VOLTAGE_DC,
        nplc: float | None = None,
        interval: float | None = None,
        timeout_ms: int = _BUFFER_TIMEOUT_MS,
    ) -> list[float]:
        """Capture ``count`` readings into instrument memory and return them.

        This is the only deep-capture path over the wire. The front panel's Data
        Log and Digitize features have no SCPI on this firmware, so this is built
        from the documented sample count, trigger count and fetch commands.

        Leave ``interval`` at None for a continuous burst, which runs as fast as
        ``nplc`` allows: 0.001 gives a 20 us aperture and about 50 kSa/s. Give
        ``interval`` in seconds to space the readings out for a slow log.
        """
        func = _token(function, self.Function, "function")
        self._configure(func)
        if nplc is not None:
            self.link.write(f"{func}:NPLC {_num(nplc)}")
        self.link.write("TRIG:SOUR IMM")
        if interval is None:
            self.link.write("TRIG:DEL:AUTO ON")
            self.link.write("TRIG:COUN 1")
            self.link.write(f"SAMP:COUN {int(count)}")
        else:
            # One sample per trigger, so the trigger delay spaces the readings.
            self.link.write("TRIG:DEL:AUTO OFF")
            self.link.write(f"TRIG:DEL {_num(interval)}")
            self.link.write(f"TRIG:COUN {int(count)}")
            self.link.write("SAMP:COUN 1")
        return self._acquire(int(count), timeout_ms)

    def meas_stats(self) -> Stats:
        """Read the statistics the instrument accumulated over its readings.

        Enable accumulation with ``dmm.calc.set_stats_state(True)`` and reset it
        with ``dmm.calc.cmd_clear_stats()`` first, or these describe whatever the
        instrument has been collecting.
        """
        average, sdev, minimum, maximum = (
            _readings(self.link.query("CALC:AVER:ALL?")) + [float("nan")] * 4
        )[:4]
        return Stats(
            average=average,
            sdev=sdev,
            minimum=minimum,
            maximum=maximum,
            peak_to_peak=_reading(self.link.query("CALC:AVER:PTP?")),
            count=int(round(_reading(self.link.query("CALC:AVER:COUN?")))),
        )

    # ---------- function and configuration ----------

    def get_function(self) -> str:
        """The active measurement function, as the canonical token.

        The instrument answers ``VOLT`` for DC voltage and ``CURR`` for DC
        current, dropping the suffix it was given. This normalizes both back, so
        the value compares equal to what was written.
        """
        raw = self.link.query("FUNC?").strip().strip('"').upper()
        return self._FUNCTION_READBACK.get(raw, raw)

    def get_configuration(self) -> str:
        """The active function, range and resolution, as the instrument reports them."""
        return self.link.query("CONF?").strip().strip('"')

    def set_temperature_unit(self, unit: TemperatureUnit | str) -> str:
        """Set the unit for every temperature reading; a :class:`TemperatureUnit`."""
        self.link.write(f"UNIT:TEMP {_token(unit, self.TemperatureUnit, 'temperature unit')}")
        return self.get_temperature_unit()

    def get_temperature_unit(self) -> str:
        """The unit every temperature reading uses."""
        return self.link.query("UNIT:TEMP?").strip()

    # ---------- reading memory ----------

    def cmd_initiate(self) -> None:
        """Arm the instrument, so readings collect into memory on each trigger."""
        self.link.write("INIT")

    def cmd_abort(self) -> None:
        """Stop a measurement in progress and return the instrument to idle."""
        self.link.write("ABOR")

    def get_reading_count(self) -> int:
        """How many readings are in memory. Safe to poll while a capture runs."""
        return int(round(_reading(self.link.query("DATA:POIN?"))))

    def get_last_reading(self) -> float:
        """The most recent reading, valid even while a capture runs."""
        return _reading(self.link.query("DATA:LAST?"))

    def get_data_available(self) -> bool:
        """Whether reading memory holds data.

        This reports data present, not a run in progress. It stays true after a
        capture finishes while the readings are still there. Poll
        :meth:`get_reading_count` to watch a run, because that number grows.
        """
        return bool(self.link.query_int("*STB?") & 0x04)

    def get_memory_limits(self) -> dict:
        """The instrument's own maximum sample count and trigger count."""
        limits = {}
        for key, cmd in (("sample_max", "SAMP:COUN? MAX"), ("trigger_max", "TRIG:COUN? MAX")):
            with contextlib.suppress(Exception):
                limits[key] = _reading(self.link.query(cmd))
        return limits

    # ---------- system ----------

    def cmd_reset(self) -> None:
        """Reset the instrument to its power-on state and wait for it to finish."""
        self.link.write("*RST")
        self.link.query("*OPC?")

    def cmd_clear_status(self) -> None:
        """Clear the status byte and every event register."""
        self.link.write("*CLS")

    def cmd_beep(self) -> None:
        """Sound one beep."""
        self.link.write("SYST:BEEP")

    def cmd_local(self) -> None:
        """Return control to the front panel."""
        self.link.write("SYST:LOC")

    def set_beeper_state(self, state: bool) -> bool:
        """Enable or disable the beeper."""
        self.link.write(f"SYST:BEEP:STAT {_onoff(state)}")
        return self.get_beeper_state()

    def get_beeper_state(self) -> bool:
        """Whether the beeper is enabled."""
        return self.link.query_bool("SYST:BEEP:STAT?")

    def get_internal_temperature(self) -> str:
        """The instrument's own internal temperatures, as it reports them."""
        return self.link.query("SYST:TEMP?").strip()

    def get_errors(self) -> list[str]:
        """Drain and return the error queue; an empty list means clean.

        This instrument does have an error queue, unlike the ODP supply, so a
        rejected command can be caught here rather than passing unnoticed.
        """
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
        with self.link.temporary_timeout(30000):
            return self.link.query_int("*TST?")

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Read back the settings this driver changes, for an exact restore."""
        return {
            "function": self.get_function(),
            "dcv_range": self.dcv.get_range(),
            "dcv_autorange": self.dcv.get_autorange(),
            "dcv_nplc": self.dcv.get_nplc(),
            "trigger_source": self.trigger.get_source(),
            "trigger_count": self.trigger.get_count(),
            "sample_count": self.trigger.get_sample_count(),
        }

    def restore(self, snapshot: dict) -> None:
        """Restore the settings captured by :meth:`snapshot`, exactly as found."""
        self.link.write(f'FUNC "{snapshot["function"]}"')
        self.dcv.set_nplc(snapshot["dcv_nplc"])
        if snapshot["dcv_autorange"]:
            self.dcv.set_autorange(True)
        else:
            self.dcv.set_range(snapshot["dcv_range"])
        self.trigger.set_source(snapshot["trigger_source"])
        self.trigger.set_count(snapshot["trigger_count"])
        self.trigger.set_sample_count(snapshot["sample_count"])

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator:
        """Context manager: snapshot on enter, restore the exact state on exit."""
        state = self.snapshot()
        try:
            yield state
        finally:
            self.restore(state)

    # ---------- underscored plumbing (design standard 5.3) ----------

    def _measure(
        self,
        func: str,
        *,
        range: float | str | None = None,
        voltage_range: float | str | None = None,
        nplc: float | None = None,
        bandwidth: float | None = None,
        aperture: float | None = None,
        timeout_ms: int | None = None,
    ) -> float:
        """Select the function if needed, apply any given setting, then read.

        With no setting given this does NOT reconfigure. It selects the function
        only if the instrument is on a different one, then reads at whatever
        speed the operator configured. ``MEASure?`` would reconfigure to
        defaults and cost 410 ms every time (bench-measured).
        """
        settings = (range, voltage_range, nplc, bandwidth, aperture)
        if any(s is not None for s in settings):
            self._configure(func, range)
            if voltage_range is not None:
                self.link.write(f"{func}:VOLT:RANG {_num(voltage_range)}")
            if nplc is not None:
                self.link.write(f"{func}:NPLC {_num(nplc)}")
            if bandwidth is not None:
                self.link.write(f"{func}:BAND {_num(bandwidth)}")
            if aperture is not None:
                self.link.write(f"{func}:APER {_num(aperture)}")
        elif self.get_function() != func:
            self.link.write(f'FUNC "{func}"')
        return self._read_one(timeout_ms)

    def _configure(self, func: str, range: float | str | None = None) -> str:
        """Select a function and its defaults (CONFigure); does not trigger."""
        self.link.write(f"CONF:{func}" + (f" {_num(range)}" if range is not None else ""))
        return self.get_configuration()

    def _read_one(self, timeout_ms: int | None = None) -> float:
        """Trigger one measurement and return it (READ?)."""
        if timeout_ms is None:
            return _reading(self.link.query("READ?"))
        with self.link.temporary_timeout(timeout_ms):
            return _reading(self.link.query("READ?"))

    def _read(self, timeout_ms: int | None = None) -> list[float]:
        """Trigger and return every reading taken (READ?)."""
        if timeout_ms is None:
            return _readings(self.link.query("READ?"))
        with self.link.temporary_timeout(timeout_ms):
            return _readings(self.link.query("READ?"))

    def _fetch(self, timeout_ms: int | None = None) -> list[float]:
        """Copy readings out of memory without erasing them (FETCh?)."""
        if timeout_ms is None:
            return _readings(self.link.query("FETC?"))
        with self.link.temporary_timeout(timeout_ms):
            return _readings(self.link.query("FETC?"))

    def _remove_readings(self, count: int, wait: bool = False) -> list[float]:
        """Read and erase the oldest ``count`` readings (DATA:REMove?)."""
        cmd = f"DATA:REM? {int(count)}" + (",WAIT" if wait else "")
        return _readings(self.link.query(cmd))

    def _set_function(self, func: Function | str) -> str:
        """Select the measurement function directly (FUNCtion)."""
        self.link.write(f'FUNC "{_token(func, self.Function, "function")}"')
        return self.get_function()

    def _acquire(self, expected: int, timeout_ms: int) -> list[float]:
        """Start a capture, wait for the readings to land, then fetch them.

        Progress is tracked with the reading count, because that query is valid
        mid-run and does not stop the acquisition the way a destructive read
        does. INITiate is overlapped, so waiting on operation-complete would not
        work here.
        """
        self.link.write("INIT")
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if self.get_reading_count() >= expected:
                break
            time.sleep(0.02)
        return self._fetch(timeout_ms)


class _BasicFunction(Node):
    """Range, autorange and null: the controls every measurement function has."""

    def __init__(self, link, label: str, func: str):
        super().__init__(link, name=label)
        self._func = func

    def set_range(self, value: float | str) -> float:
        """Fix the measurement range in the function's unit; turns autorange off."""
        self.write(f"{self._func}:RANG {_num(value)}")
        return self.get_range()

    def get_range(self) -> float:
        """The measurement range in force, in the function's unit."""
        return self.query_float(f"{self._func}:RANG?")

    def set_autorange(self, state: bool) -> bool:
        """Turn autoranging on or off."""
        self.write(f"{self._func}:RANG:AUTO {_onoff(state)}")
        return self.get_autorange()

    def get_autorange(self) -> bool:
        """Whether autoranging is on."""
        return self.query_bool(f"{self._func}:RANG:AUTO?")

    def set_null_state(self, state: bool) -> bool:
        """Turn the null (relative) offset on or off."""
        self.write(f"{self._func}:NULL {_onoff(state)}")
        return self.get_null_state()

    def get_null_state(self) -> bool:
        """Whether the null offset is applied."""
        return self.query_bool(f"{self._func}:NULL?")

    def set_null_value(self, value: float) -> float:
        """Set the null offset in the function's unit; turns automatic null off."""
        self.write(f"{self._func}:NULL:VAL {_num(value)}")
        return self.get_null_value()

    def get_null_value(self) -> float:
        """The null offset in the function's unit."""
        return self.query_float(f"{self._func}:NULL:VAL?")


class _DcFunction(_BasicFunction):
    """A function that integrates: it adds the integration time.

    Auto-zero is absent on purpose. This instrument has no such header on any
    function, so a method for it would be a lie (standard 5.1).
    """

    def set_nplc(self, nplc: float) -> float:
        """Set the integration time in power-line cycles; more is slower and quieter."""
        self.write(f"{self._func}:NPLC {_num(nplc)}")
        return self.get_nplc()

    def get_nplc(self) -> float:
        """The integration time in power-line cycles."""
        return self.query_float(f"{self._func}:NPLC?")


class _DcVoltage(_DcFunction):
    """DC voltage, which alone can also switch its input impedance."""

    def set_input_impedance(self, impedance: Siglent_SDM.InputImpedance | str) -> str:
        """Set the input impedance on the low ranges; an :class:`InputImpedance`."""
        token = _token(impedance, Siglent_SDM.InputImpedance, "input impedance")
        self.write(f"{self._func}:IMP {token}")
        return self.get_input_impedance()

    def get_input_impedance(self) -> str:
        """The input impedance in force."""
        return self.query(f"{self._func}:IMP?").strip()


class _AcFunction(_BasicFunction):
    """An AC function, which filters rather than integrates.

    NPLC is absent on purpose: this instrument rejects it on the AC functions.
    """

    def set_bandwidth(self, hz: float) -> float:
        """Set the AC filter corner in Hz: 3, 20 or 200."""
        self.write(f"{self._func}:BAND {_num(hz)}")
        return self.get_bandwidth()

    def get_bandwidth(self) -> float:
        """The AC filter corner in Hz."""
        return _reading(self.query(f"{self._func}:BAND?"))


class _FreqPeriod(Node):
    """Frequency and period, which gate an input rather than range on it.

    These have no range of their own on this instrument. They range on the INPUT
    VOLTAGE, and that is what ``set_voltage_range`` sets.
    """

    def __init__(self, link, label: str, func: str):
        super().__init__(link, name=label)
        self._func = func

    def set_aperture(self, seconds: float) -> float:
        """Set the gate time in seconds: 0.001, 0.01, 0.1 or 1."""
        self.write(f"{self._func}:APER {_num(seconds)}")
        return self.get_aperture()

    def get_aperture(self) -> float:
        """The gate time in seconds."""
        return self.query_float(f"{self._func}:APER?")

    def set_voltage_range(self, volts: float | str) -> float:
        """Fix the input voltage range in volts; turns its autorange off."""
        self.write(f"{self._func}:VOLT:RANG {_num(volts)}")
        return self.get_voltage_range()

    def get_voltage_range(self) -> float:
        """The input voltage range in volts."""
        return self.query_float(f"{self._func}:VOLT:RANG?")

    def set_voltage_autorange(self, state: bool) -> bool:
        """Turn input-voltage autoranging on or off."""
        self.write(f"{self._func}:VOLT:RANG:AUTO {_onoff(state)}")
        return self.get_voltage_autorange()

    def get_voltage_autorange(self) -> bool:
        """Whether input-voltage autoranging is on."""
        return self.query_bool(f"{self._func}:VOLT:RANG:AUTO?")

    def set_null_state(self, state: bool) -> bool:
        """Turn the null (relative) offset on or off."""
        self.write(f"{self._func}:NULL {_onoff(state)}")
        return self.get_null_state()

    def get_null_state(self) -> bool:
        """Whether the null offset is applied."""
        return self.query_bool(f"{self._func}:NULL?")

    def set_null_value(self, value: float) -> float:
        """Set the null offset in the function's unit."""
        self.write(f"{self._func}:NULL:VAL {_num(value)}")
        return self.get_null_value()

    def get_null_value(self) -> float:
        """The null offset in the function's unit."""
        return self.query_float(f"{self._func}:NULL:VAL?")


class _Temperature(Node):
    """Temperature, which selects a transducer rather than a range."""

    def get_transducer(self) -> str:
        """The active transducer, as the instrument reports it."""
        return self.query("TEMP:TRAN?").strip()

    def set_transducer(self, probe: str, sensor: str) -> str:
        """Select the transducer: ``probe`` is RTD or THER, ``sensor`` its type."""
        self.write(f"TEMP:MDEF:{probe}:TRAN {sensor}")
        return self.get_transducer()

    def get_transducer_list(self, probe: str) -> str:
        """The sensor types available for a probe."""
        return self.query(f"TEMP:MDEF:{probe}:TRAN:LIST?").strip()

    def set_null_state(self, state: bool) -> bool:
        """Turn the null (relative) offset on or off."""
        self.write(f"TEMP:NULL {_onoff(state)}")
        return self.get_null_state()

    def get_null_state(self) -> bool:
        """Whether the null offset is applied."""
        return self.query_bool("TEMP:NULL?")

    def set_null_value(self, value: float) -> float:
        """Set the null offset in the configured temperature unit."""
        self.write(f"TEMP:NULL:VAL {_num(value)}")
        return self.get_null_value()

    def get_null_value(self) -> float:
        """The null offset in the configured temperature unit."""
        return self.query_float("TEMP:NULL:VAL?")


class _Continuity(Node):
    """Continuity, which only has a threshold."""

    def set_threshold(self, ohms: float) -> float:
        """Set the resistance in ohms below which continuity beeps, 0 to 2000."""
        self.write(f"CONT:THR:VAL {_num(ohms)}")
        return self.get_threshold()

    def get_threshold(self) -> float:
        """The continuity threshold in ohms."""
        return self.query_float("CONT:THR:VAL?")


class _Trigger(Node):
    """What starts a measurement, and how many it takes."""

    def set_source(self, source: Siglent_SDM.TriggerSource | str) -> str:
        """Select what starts a measurement; a :class:`TriggerSource`."""
        self.write(f"TRIG:SOUR {_token(source, Siglent_SDM.TriggerSource, 'trigger source')}")
        return self.get_source()

    def get_source(self) -> str:
        """What starts a measurement."""
        return self.query("TRIG:SOUR?").strip()

    def set_count(self, count: float) -> float:
        """Set how many triggers to accept before returning to idle."""
        self.write(f"TRIG:COUN {_num(count)}")
        return self.get_count()

    def get_count(self) -> float:
        """How many triggers the instrument accepts before returning to idle."""
        return self.query_float("TRIG:COUN?")

    def set_sample_count(self, count: float) -> int:
        """Set how many readings each trigger takes."""
        self.write(f"SAMP:COUN {_num(count)}")
        return self.get_sample_count()

    def get_sample_count(self) -> int:
        """How many readings each trigger takes."""
        return int(round(_reading(self.query("SAMP:COUN?"))))

    def set_delay(self, seconds: float) -> float:
        """Set the wait between the trigger and the first reading; turns auto delay off."""
        self.write(f"TRIG:DEL {_num(seconds)}")
        return self.get_delay()

    def get_delay(self) -> float:
        """The wait between the trigger and the first reading, in seconds."""
        return self.query_float("TRIG:DEL?")

    def set_delay_auto(self, state: bool) -> bool:
        """Let the instrument choose the trigger delay."""
        self.write(f"TRIG:DEL:AUTO {_onoff(state)}")
        return self.get_delay_auto()

    def get_delay_auto(self) -> bool:
        """Whether the instrument chooses the trigger delay."""
        return self.query_bool("TRIG:DEL:AUTO?")

    def set_slope(self, slope: Siglent_SDM.Slope | str) -> str:
        """Select which edge an external trigger fires on; a :class:`Slope`."""
        self.write(f"TRIG:SLOP {_token(slope, Siglent_SDM.Slope, 'slope')}")
        return self.get_slope()

    def get_slope(self) -> str:
        """Which edge an external trigger fires on."""
        return self.query("TRIG:SLOP?").strip()

    def set_level(self, level: float) -> float:
        """Set the level an internal trigger fires at; needs the INTERNAL source."""
        self.write(f"TRIG:LEV {_num(level)}")
        return self.get_level()

    def get_level(self) -> float:
        """The level an internal trigger fires at."""
        return self.query_float("TRIG:LEV?")

    def cmd_trigger(self) -> None:
        """Send one bus trigger; needs the BUS source and an armed instrument."""
        self.write("*TRG")


class _Calc(Node):
    """The math block: statistics, limit testing and dB scaling."""

    def set_stats_state(self, state: bool) -> bool:
        """Turn statistics accumulation on or off."""
        self.write(f"CALC:AVER {_onoff(state)}")
        return self.get_stats_state()

    def get_stats_state(self) -> bool:
        """Whether statistics are accumulating."""
        return self.query_bool("CALC:AVER?")

    def cmd_clear_stats(self) -> None:
        """Throw away the statistics accumulated so far."""
        self.write("CALC:AVER:CLE")

    def set_limits(
        self, *, lower: float | None = None, upper: float | None = None
    ) -> tuple[float, float]:
        """Set the limit-test bounds in the function's unit."""
        if lower is not None:
            self.write(f"CALC:LIM:LOW {_num(lower)}")
        if upper is not None:
            self.write(f"CALC:LIM:UPP {_num(upper)}")
        return self.get_limits()

    def get_limits(self) -> tuple[float, float]:
        """The limit-test bounds as (lower, upper)."""
        return (self.query_float("CALC:LIM:LOW?"), self.query_float("CALC:LIM:UPP?"))

    def set_limit_state(self, state: bool) -> bool:
        """Turn limit testing on or off."""
        self.write(f"CALC:LIM {_onoff(state)}")
        return self.get_limit_state()

    def get_limit_state(self) -> bool:
        """Whether limit testing is on."""
        return self.query_bool("CALC:LIM?")

    def cmd_clear_limits(self) -> None:
        """Clear the limit-test result."""
        self.write("CALC:LIM:CLE")

    def set_scale_function(self, func: Siglent_SDM.ScaleFunction | str) -> str:
        """Select the scaling applied to a reading; a :class:`ScaleFunction`."""
        self.write(f"CALC:SCAL:FUNC {_token(func, Siglent_SDM.ScaleFunction, 'scale function')}")
        return self.get_scale_function()

    def get_scale_function(self) -> str:
        """The scaling applied to a reading."""
        return self.query("CALC:SCAL:FUNC?").strip()

    def set_scale_state(self, state: bool) -> bool:
        """Turn scaling on or off."""
        self.write(f"CALC:SCAL {_onoff(state)}")
        return self.get_scale_state()

    def get_scale_state(self) -> bool:
        """Whether scaling is on."""
        return self.query_bool("CALC:SCAL?")

    def set_db_reference(self, volts: float) -> float:
        """Set the reference the dB scaling subtracts, in volts."""
        self.write(f"CALC:SCAL:DB:REF {_num(volts)}")
        return self.query_float("CALC:SCAL:DB:REF?")

    def set_dbm_reference(self, ohms: float) -> float:
        """Set the reference impedance the dBm scaling assumes, in ohms."""
        self.write(f"CALC:SCAL:DBM:REF {_num(ohms)}")
        return self.query_float("CALC:SCAL:DBM:REF?")

    def set_scale_reference_auto(self, state: bool) -> bool:
        """Let the first reading become the scaling reference."""
        self.write(f"CALC:SCAL:REF:AUTO {_onoff(state)}")
        return self.query_bool("CALC:SCAL:REF:AUTO?")

    def cmd_clear_all(self) -> None:
        """Clear the limits, the histogram, the statistics and the measurements."""
        self.write("CALC:CLE")


class _Status(Node):
    """The IEEE-488.2 status registers."""

    def get_event_status(self) -> int:
        """Read and clear the standard event status register."""
        return self.query_int("*ESR?")

    def set_event_status_enable(self, mask: int) -> int:
        """Set the standard event status enable mask."""
        self.write(f"*ESE {int(mask)}")
        return self.get_event_status_enable()

    def get_event_status_enable(self) -> int:
        """The standard event status enable mask."""
        return self.query_int("*ESE?")

    def get_status_byte(self) -> int:
        """Read the status byte."""
        return self.query_int("*STB?")

    def set_service_request_enable(self, mask: int) -> int:
        """Set the service-request enable mask."""
        self.write(f"*SRE {int(mask)}")
        return self.get_service_request_enable()

    def get_service_request_enable(self) -> int:
        """The service-request enable mask."""
        return self.query_int("*SRE?")

    def set_power_on_status_clear(self, state: bool) -> bool:
        """Set whether the status registers clear at power-on."""
        self.write(f"*PSC {1 if state else 0}")
        return self.get_power_on_status_clear()

    def get_power_on_status_clear(self) -> bool:
        """Whether the status registers clear at power-on."""
        return self.query_bool("*PSC?")

    def cmd_wait(self) -> None:
        """Make the instrument finish its pending work before the next command."""
        self.write("*WAI")

    def get_operation_complete(self) -> int:
        """Block until pending work finishes, then report completion."""
        return self.query_int("*OPC?")


def _main(argv: list[str] | None = None) -> int:
    """Live self-test against the bench SDM.

    Read-only: identity, function, configuration, a few DC voltage readings and
    the internal temperature. It changes no setting the operator would miss.
    """
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE")
        return 2
    resource = argv[1]
    dmm = Siglent_SDM(resource)
    print(f"connecting to Siglent SDM on {dmm.resource_name} ...")
    idn = dmm.connect()
    try:
        print(f"  identity: {idn}")
        print(f"  model={dmm.model}  serial={dmm.serial}  firmware={dmm.firmware}")
        print(f"  function: {dmm.get_function()}   configuration: {dmm.get_configuration()}")
        print(
            f"  DCV range {dmm.dcv.get_range():g} V, autorange {dmm.dcv.get_autorange()}, "
            f"NPLC {dmm.dcv.get_nplc():g}, input impedance {dmm.dcv.get_input_impedance()}"
        )
        print(f"  meas_vdc(): {dmm.meas_vdc():.6f} V")
        print(f"  meas_vdc(nplc=0.1): {dmm.meas_vdc(nplc=0.1):.6f} V")
        print(f"  readings in memory: {dmm.get_reading_count()}")
        print(f"  internal temperature: {dmm.get_internal_temperature()}")
        print(f"  error queue: {dmm.get_errors() or 'clean'}")
    finally:
        dmm.disconnect()
        print("disconnected (local).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
