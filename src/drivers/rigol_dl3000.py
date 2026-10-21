"""Driver for the Rigol DL3000 family programmable DC electronic load.

Public import path::

    from drivers.rigol_dl3000 import Rigol_DL3000

Shape follows ``docs/driver-api-design.md`` (section 8, DL3021A). The load mode
is chosen by the method name - ``set_load_current`` / ``set_load_voltage`` /
``set_load_resistance`` / ``set_load_power`` - each of which sets the level and
its required limit in one call and returns a :class:`LoadState`. Measurements
are ``meas_*`` methods; the transient, list and battery features are single
configuration calls; ``cmd_trigger`` fires the bus trigger.

Units carry no scalar prefix (design standard 4.7): amperes, volts, ohms, watts,
seconds, hertz, amperes/second, amp-hours, watt-hours. The DL3000's prefixed
native units (ms, kHz, A/us, mAh) are converted at this boundary.

Task recipes::

    load.connect()
    load.set_load_current(0.5, v_limit=8, slew=1e5)   # 0.5 A CC, 8 V limit
    load.set_output_state(True)
    r = load.meas_vip()                                # r.voltage, r.current, r.power
    load.set_output_state(False)

    # transient CC toggling between 0.2 A and 0.4 A at 1 Hz:
    load.set_transient(0.4, 0.2, width_a=0.5, width_b=0.5)

    # battery discharge at 0.5 A until 2 V (cutoff required):
    load.set_battery_test(0.5, stop_voltage=2.0)

Command surface follows the Rigol DL3000 Programming Guide.

Common tasks:
  1. Discharge at 1 A with a 2.5 V cutoff:
         eload.set_load_current(1.0, v_limit=2.5)
         eload.set_output_state(True)
  2. Read voltage, current and power together:
         v, i, p = eload.meas_vip()
  3. Read the accumulated capacity of a discharge:
         ah = eload.meas_capacity()
  4. Present a fixed resistance instead of a fixed current:
         eload.set_load_resistance(50.0, i_limit=2.0)
  5. Run a battery test that stops itself:
         eload.set_battery_test(0.5, stop_voltage=3.0)
         eload.set_output_state(True)
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

from common.errors import InstrumentError
from common.instrument import Instrument
from common.scpi import ScpiEnum

_DEFAULT_TIMEOUT_MS = 5000

# Instrument-unit-per-API-unit scale factors (API value * scale -> instrument
# value). The public API carries no scalar prefix (docs/driver-api-design.md
# 4.7); convert here where the DL3000 speaks a prefixed unit.
_MS_PER_S = 1.0e3  # DL3000 times are milliseconds; API is seconds
_AUS_PER_AS = 1.0e-6  # DL3000 slew rate is A/us; API is amperes/second
_MAH_PER_AH = 1.0e3  # DL3000 battery capacity is milliamp-hours; API is amp-hours

_IDN_RE = re.compile(r"(DL30[0-9]{2}A?)", re.IGNORECASE)


def _onoff(state: bool) -> str:
    return "ON" if state else "OFF"


@dataclass(frozen=True)
class Reading:
    """A voltage/current/power measurement taken together.

    Iterates over the measured quantities in field order, which the ``meas_``
    name spells, so ``v, i, p = load.meas_vip()`` works and ``r.voltage`` still
    does. A field added later that the name does not spell is not iterated, so
    no unpacking call site breaks.
    """

    voltage: float  # volts
    current: float  # amperes
    power: float  # watts

    def __iter__(self) -> Iterator[float]:
        """Yield voltage, current then power, the quantities ``meas_vip`` names."""
        yield self.voltage
        yield self.current
        yield self.power


@dataclass(frozen=True)
class LoadState:
    """The active load mode, its level, and the limits/options in force.

    ``level`` is in the mode's unit (A / V / ohm / W). Only the fields that
    apply to the active mode are populated; the rest are ``None``.
    """

    mode: str  # one of LoadMode ("CURR"/"VOLT"/"RES"/"POW") or "NONE"
    level: float
    v_limit: float | None = None  # volts
    i_limit: float | None = None  # amperes
    von: float | None = None  # volts (turn-on threshold)
    slew: float | None = None  # amperes/second (CC only)


@dataclass(frozen=True)
class BatteryState:
    """Battery-discharge test configuration read back from the instrument."""

    current: float  # amperes
    stop_voltage: float | None  # volts (None if the voltage cutoff is disabled)
    stop_capacity: float | None  # amp-hours (None if disabled)
    stop_time: float | None  # seconds (None if disabled)


@dataclass(frozen=True)
class TransientState:
    """CC transient (dynamic) configuration read back from the instrument."""

    mode: str  # one of Transient
    level_a: float  # amperes
    level_b: float  # amperes
    width_a: float  # seconds
    width_b: float  # seconds


class Rigol_DL3000(Instrument):
    """Rigol DL3000 family programmable DC electronic load.

    ``connect()`` reads ``*IDN?`` and gates the model; the input is turned off on
    disconnect for safety. The mode-selecting ``set_load_*`` helpers refuse to
    run on a live input (they would apply a stale level through the mode switch);
    pass ``allow_live=True`` to override (design standard 6.3).
    """

    class Mode(ScpiEnum):
        """Static load regulation mode; the ``set_load_*`` method name picks it."""

        CURRENT = "CURR"
        VOLTAGE = "VOLT"
        RESISTANCE = "RES"
        POWER = "POW"

    class Transient(ScpiEnum):
        """CC transient (dynamic) sub-mode."""

        CONTINUOUS = "CONTinuous"
        PULSE = "PULSe"
        TOGGLE = "TOGGle"

    class TriggerSource(ScpiEnum):
        """Source that advances transient/list steps."""

        BUS = "BUS"
        EXTERNAL = "EXTernal"
        MANUAL = "MANual"

    # Mode key -> the SCPI set token (:SOUR:FUNC takes CURRent/VOLTage/...).
    _FUNC_TOKEN = {"CURR": "CURRent", "VOLT": "VOLTage", "RES": "RESistance", "POW": "POWer"}
    # :SOUR:FUNC? reports the panel token (CC/CV/CR/CP), not the set token.
    _FUNC_READBACK = {"CC": "CURR", "CV": "VOLT", "CR": "RES", "CP": "POW"}
    # :SOUR:LIST:MODE takes the panel tokens (CC/CV/CR/CP), not CURR/VOLT/...
    _LIST_TOKEN = {"CURR": "CC", "VOLT": "CV", "RES": "CR", "POW": "CP"}

    def __init__(
        self,
        resource_name: str,
        *,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        super().__init__(resource_name, name="ELOAD")
        self.link.default_timeout_ms = timeout_ms
        self.model: str | None = None
        self.serial: str | None = None
        self.firmware: str | None = None

    # ---------- connection lifecycle ----------

    def _on_connect(self, idn: str) -> None:
        match = _IDN_RE.search(idn)
        if not match:
            raise InstrumentError(f"Not a recognized DL3000 IDN: {idn!r}")
        fields = [f.strip() for f in idn.split(",")]
        self.model = match.group(1).upper()
        self.serial = fields[2] if len(fields) >= 3 else None
        self.firmware = fields[3] if len(fields) >= 4 else None

    def safe_state(self) -> None:
        """Turn the input off before the link closes (never sink unattended)."""
        with contextlib.suppress(Exception):
            self.link.write(":SOUR:INP:STAT OFF")

    # ---------- underscored plumbing (design standard 5.3) ----------

    def _setq(self, cmd: str, value: float, scale: float = 1.0) -> float:
        """Write ``cmd`` with an API value (scaled to instrument units), verify
        the read-back, and return it in API units."""
        applied = self.link.set_and_verify(cmd, float(value) * scale)
        return float(applied) / scale

    def _getq(self, cmd: str, scale: float = 1.0) -> float:
        """Query ``cmd?`` and return the result in API units."""
        return self.link.query_float(f"{cmd}?") / scale

    def _require_off(self, method: str, allow_live: bool) -> None:
        if not allow_live and self.get_output_state():
            raise InstrumentError(
                f"{method}() refuses a live input: a mode switch applies a stale "
                "level. Call set_output_state(False) first, or pass allow_live=True."
            )

    def _set_function_mode(self, token: str) -> None:
        """Select the regulation source (:SOUR:FUNC:MODE FIX/LIST/WAVe/BATT/OCP/OPP)."""
        self.link.write(f":SOUR:FUNC:MODE {token}")

    def _set_static_mode(self, mode: str) -> None:
        """Select fixed (non-list/wave/battery) operation in the given load mode."""
        self._set_function_mode("FIXed")
        self.link.write(f":SOUR:FUNC {self._FUNC_TOKEN[mode]}")

    def _meas(self, cmd: str, scale: float = 1.0) -> float:
        return self.link.query_float(cmd) / scale

    # ---------- input / remote sense ----------

    def set_output_state(self, state: bool) -> bool:
        """Enable/disable the load input. Returns the read-back."""
        self.link.write(f":SOUR:INP:STAT {_onoff(state)}")
        return self.get_output_state()

    def get_output_state(self) -> bool:
        """Whether the load input is drawing."""
        return self.link.query_bool(":SOUR:INP:STAT?")

    def set_remote_sense_state(self, state: bool) -> bool:
        """Enable/disable 4-wire remote sense. Returns the read-back."""
        self.link.write(f":SOUR:SENS {_onoff(state)}")
        return self.get_remote_sense_state()

    def get_remote_sense_state(self) -> bool:
        """Whether 4-wire remote sense is on."""
        return self.link.query_bool(":SOUR:SENS?")

    # ---------- load mode (the method name is the mode select) ----------

    def set_load_current(
        self,
        amps: float,
        *,
        v_limit: float,
        von: float | None = None,
        slew: float | None = None,
        allow_live: bool = False,
    ) -> LoadState:
        """Constant-current load: sink ``amps`` A, voltage limit ``v_limit`` V.

        ``von`` is the turn-on voltage threshold; ``slew`` the CC slew in A/s.
        Mode and limit are written before the level (design standard 6.2); the
        call refuses a live input unless ``allow_live=True``.
        """
        self._require_off("set_load_current", allow_live)
        self._set_static_mode("CURR")
        self._setq(":SOUR:CURR:VLIM", v_limit)
        if von is not None:
            self._setq(":SOUR:CURR:VON", von)
        if slew is not None:
            self._setq(":SOUR:CURR:SLEW", slew, _AUS_PER_AS)
        self._setq(":SOUR:CURR", amps)
        return self.get_load()

    def set_load_voltage(
        self, volts: float, *, i_limit: float, allow_live: bool = False
    ) -> LoadState:
        """Constant-voltage load: hold ``volts`` V, current limit ``i_limit`` A.

        The DL3021A has no CV turn-on threshold (``:VON`` is CC/battery only).
        """
        self._require_off("set_load_voltage", allow_live)
        self._set_static_mode("VOLT")
        self._setq(":SOUR:VOLT:ILIM", i_limit)
        self._setq(":SOUR:VOLT", volts)
        return self.get_load()

    def set_load_resistance(
        self, ohms: float, *, i_limit: float, allow_live: bool = False
    ) -> LoadState:
        """Constant-resistance load: present ``ohms``, current limit ``i_limit`` A."""
        self._require_off("set_load_resistance", allow_live)
        self._set_static_mode("RES")
        self._setq(":SOUR:RES:ILIM", i_limit)
        self._setq(":SOUR:RES", ohms)
        return self.get_load()

    def set_load_power(
        self, watts: float, *, i_limit: float, allow_live: bool = False
    ) -> LoadState:
        """Constant-power load: sink ``watts`` W, current limit ``i_limit`` A."""
        self._require_off("set_load_power", allow_live)
        self._set_static_mode("POW")
        self._setq(":SOUR:POW:ILIM", i_limit)
        self._setq(":SOUR:POW", watts)
        return self.get_load()

    def get_load(self) -> LoadState:
        """Read the active mode, its level, and the limits/options in force."""
        mode = self._FUNC_READBACK.get(self.link.query(":SOUR:FUNC?").strip().upper(), "NONE")
        if mode == "CURR":
            return LoadState(
                "CURR",
                self._getq(":SOUR:CURR"),
                v_limit=self._getq(":SOUR:CURR:VLIM"),
                von=self._getq(":SOUR:CURR:VON"),
                slew=self._getq(":SOUR:CURR:SLEW", _AUS_PER_AS),
            )
        if mode == "VOLT":
            return LoadState(
                "VOLT", self._getq(":SOUR:VOLT"), i_limit=self._getq(":SOUR:VOLT:ILIM")
            )
        if mode == "RES":
            return LoadState("RES", self._getq(":SOUR:RES"), i_limit=self._getq(":SOUR:RES:ILIM"))
        if mode == "POW":
            return LoadState("POW", self._getq(":SOUR:POW"), i_limit=self._getq(":SOUR:POW:ILIM"))
        return LoadState("NONE", float("nan"))

    def get_load_mode(self) -> str:
        """Active static mode CURR/VOLT/RES/POW (or NONE).

        The instrument reports CC/CV/CR/CP; this normalizes to the set tokens.
        """
        return self._FUNC_READBACK.get(self.link.query(":SOUR:FUNC?").strip().upper(), "NONE")

    def set_range(self, mode: str, value: float) -> float:
        """Set a mode's measurement/source range in that mode's unit.

        ``mode`` is a :class:`Mode`. A value picks the range that contains it.
        """
        m = str(mode).upper()[:4]
        if m not in self._FUNC_TOKEN:
            raise ValueError(f"mode must be one of {list(self._FUNC_TOKEN)}, got {mode!r}")
        return self._setq(f":SOUR:{m}:RANG", value)

    # ---------- measurements ----------

    def meas_voltage(self) -> float:
        """Input voltage, volts."""
        return self._meas(":MEAS:VOLT?")

    def meas_current(self) -> float:
        """Input current, amperes."""
        return self._meas(":MEAS:CURR?")

    def meas_power(self) -> float:
        """Input power, watts."""
        return self._meas(":MEAS:POW?")

    def meas_resistance(self) -> float:
        """Input resistance, ohms."""
        return self._meas(":MEAS:RES?")

    def meas_vip(self) -> Reading:
        """Voltage, current and power in one call (design standard: no meas_all)."""
        return Reading(self.meas_voltage(), self.meas_current(), self.meas_power())

    def meas_voltage_max(self) -> float:
        """Peak input voltage, volts."""
        return self._meas(":MEAS:VOLT:MAX?")

    def meas_voltage_min(self) -> float:
        """Minimum input voltage, volts."""
        return self._meas(":MEAS:VOLT:MIN?")

    def meas_current_max(self) -> float:
        """Peak input current, amperes."""
        return self._meas(":MEAS:CURR:MAX?")

    def meas_current_min(self) -> float:
        """Minimum input current, amperes."""
        return self._meas(":MEAS:CURR:MIN?")

    def meas_integration_time(self) -> float:
        """Measurement integration time, seconds."""
        return self._meas(":MEAS:TIME?", _MS_PER_S)

    def meas_capacity(self) -> float:
        """Accumulated battery capacity, amp-hours (native mAh)."""
        return self._meas(":MEAS:CAP?", _MAH_PER_AH)

    def meas_energy(self) -> float:
        """Accumulated energy, watt-hours; Wh has no scalar prefix."""
        return self._meas(":MEAS:WATT?")

    def meas_discharge_time(self) -> float:
        """Battery discharge elapsed time, seconds.

        The documented short form ``:MEAS:DISC?`` draws -113 on the bench
        DL3021A firmware; the long form replies (bench-verified).
        """
        return self.link.query_float(":MEASure:DISChargingTime?")

    def meas_waveform(self) -> list[float]:
        """The 400-point waveform-display buffer, oldest first.

        Enter the waveform recorder first (its function mode is the DL3000 WAVE
        screen). Values track the displayed trace; the instrument tags no unit.
        """
        raw = self.link.query(":MEAS:WAV?").strip()
        return [float(t) for t in raw.split(",") if t.strip()]

    # ---------- transient (CC dynamic) ----------

    def set_transient(
        self,
        level_a: float,
        level_b: float,
        *,
        width_a: float,
        width_b: float,
        mode: str = Transient.CONTINUOUS,
        allow_live: bool = False,
    ) -> TransientState:
        """Configure the CC transient generator (Level A/B, widths, mode).

        Levels are amperes, widths seconds. In CONTINUOUS mode the two widths set
        the period; the DL3000 bounds each width by the period, so they are
        written together. Enable it with :meth:`set_transient_state`. Refuses a
        live input unless ``allow_live=True``.
        """
        self._require_off("set_transient", allow_live)
        self._set_static_mode("CURR")
        self.link.write(f":SOUR:CURR:TRAN:MODE {str(mode)}")
        self._setq(":SOUR:CURR:TRAN:ALEV", level_a)
        self._setq(":SOUR:CURR:TRAN:BLEV", level_b)
        # AWID/BWID are bounded by the transient period: a small leftover period
        # snaps a large width toward 0 (and queues -220). Widen the period to fit
        # both widths first, then set them (bench-verified 2026-08-10).
        self.link.write(f":SOUR:CURR:TRAN:PER {(float(width_a) + float(width_b)) * _MS_PER_S:.12g}")
        self._setq(":SOUR:CURR:TRAN:AWID", width_a, _MS_PER_S)
        self._setq(":SOUR:CURR:TRAN:BWID", width_b, _MS_PER_S)
        return self.get_transient()

    def get_transient(self) -> TransientState:
        """Read back the CC transient configuration."""
        return TransientState(
            self.link.query(":SOUR:CURR:TRAN:MODE?").strip(),
            self._getq(":SOUR:CURR:TRAN:ALEV"),
            self._getq(":SOUR:CURR:TRAN:BLEV"),
            self._getq(":SOUR:CURR:TRAN:AWID", _MS_PER_S),
            self._getq(":SOUR:CURR:TRAN:BWID", _MS_PER_S),
        )

    def set_transient_state(self, state: bool) -> bool:
        """Start/stop the transient generator. Returns read-back."""
        self.link.write(f":SOUR:TRAN:STAT {_onoff(state)}")
        return self.get_transient_state()

    def get_transient_state(self) -> bool:
        """Whether the transient generator is running."""
        return self.link.query_bool(":SOUR:TRAN:STAT?")

    # ---------- list (sequenced) ----------

    def set_list(
        self,
        levels: list[float],
        widths: list[float],
        *,
        slews: list[float] | None = None,
        count: int = 1,
        mode: str = Mode.CURRENT,
        allow_live: bool = False,
    ) -> int:
        """Program a list sweep from Python lists; returns the step count.

        ``levels`` are in the list mode's unit (A/V/ohm/W), ``widths`` seconds,
        optional ``slews`` A/s (CC). ``count`` is the number of cycles (0 =
        infinite). Replaces the per-step indexed accessors. Refuses a live input
        unless ``allow_live=True``.
        """
        self._require_off("set_list", allow_live)
        if len(levels) != len(widths):
            raise ValueError(f"levels ({len(levels)}) and widths ({len(widths)}) differ in length")
        if slews is not None and len(slews) != len(levels):
            raise ValueError("slews must match levels in length")
        m = str(mode).upper()[:4]
        if m not in self._FUNC_TOKEN:
            raise ValueError(f"list mode must be one of {list(self._FUNC_TOKEN)}, got {mode!r}")
        self._set_function_mode("LIST")
        self.link.write(f":SOUR:LIST:MODE {self._LIST_TOKEN[m]}")
        self.link.write(f":SOUR:LIST:COUN {int(count)}")
        self.link.write(f":SOUR:LIST:STEP {len(levels)}")
        for i, (lvl, wid) in enumerate(zip(levels, widths)):
            self.link.write(f":SOUR:LIST:LEV {i},{float(lvl):.12g}")
            self.link.write(f":SOUR:LIST:WID {i},{float(wid):.12g}")
            if slews is not None:
                self.link.write(f":SOUR:LIST:SLEW {i},{float(slews[i]) * _AUS_PER_AS:.12g}")
        return int(self.link.query_float(":SOUR:LIST:STEP?"))

    def get_list_step(self, index: int) -> tuple[float, float]:
        """Return (level, width_seconds) for list step ``index`` (0-based)."""
        lvl = self.link.query_float(f":SOUR:LIST:LEV? {int(index)}")
        wid = self.link.query_float(f":SOUR:LIST:WID? {int(index)}")
        return lvl, wid

    def set_list_end_state(self, keep_last: bool) -> str:
        """Post-list end state: hold the last step (True) or turn off."""
        token = "LAST" if keep_last else "OFF"
        self.link.write(f":SOUR:LIST:END {token}")
        return self.link.query(":SOUR:LIST:END?").strip()

    # ---------- battery discharge ----------

    def set_battery_test(
        self,
        amps: float,
        *,
        stop_voltage: float,
        stop_capacity: float | None = None,
        stop_time: float | None = None,
        allow_live: bool = False,
    ) -> BatteryState:
        """Configure a battery-discharge test at ``amps`` with a required cutoff.

        ``stop_voltage`` (V) is mandatory - a discharge with no cutoff damages the
        cell. ``stop_capacity`` is amp-hours, ``stop_time`` seconds; each optional
        cutoff is enabled only when given. Refuses a live input unless
        ``allow_live=True``.
        """
        self._require_off("set_battery_test", allow_live)
        self._set_function_mode("BATTery")
        self._setq(":SOUR:BATT", amps)
        self._setq(":SOUR:BATT:VST", stop_voltage)
        self.link.write(":SOUR:BATT:VEN ON")
        if stop_capacity is not None:
            self._setq(":SOUR:BATT:CST", stop_capacity, _MAH_PER_AH)
        self.link.write(f":SOUR:BATT:CEN {_onoff(stop_capacity is not None)}")
        if stop_time is not None:
            self._setq(":SOUR:BATT:TIM", stop_time)
        self.link.write(f":SOUR:BATT:TEN {_onoff(stop_time is not None)}")
        return self.get_battery_test()

    def get_battery_test(self) -> BatteryState:
        """Read back the battery-discharge configuration (disabled cutoffs -> None)."""
        v = self._getq(":SOUR:BATT:VST") if self.link.query_bool(":SOUR:BATT:VEN?") else None
        cap = (
            self._getq(":SOUR:BATT:CST", _MAH_PER_AH)
            if self.link.query_bool(":SOUR:BATT:CEN?")
            else None
        )
        t = self._getq(":SOUR:BATT:TIM") if self.link.query_bool(":SOUR:BATT:TEN?") else None
        return BatteryState(self._getq(":SOUR:BATT"), v, cap, t)

    # ---------- OCP / OPP protection sweeps ----------

    def set_ocp_test(
        self,
        start_current: float,
        step_current: float,
        *,
        von: float,
        step_delay: float | None = None,
        allow_live: bool = False,
    ) -> None:
        """Configure the over-current-protection search (ramp from start by step).

        Currents are amperes, ``von`` volts, ``step_delay`` seconds. Refuses a
        live input unless ``allow_live=True``.
        """
        self._require_off("set_ocp_test", allow_live)
        self._set_function_mode("OCP")
        self._setq(":SOUR:OCP:ISET", start_current)
        self._setq(":SOUR:OCP:ISTep", step_current)
        self._setq(":SOUR:OCP:VON", von)
        if step_delay is not None:
            self._setq(":SOUR:OCP:IDELaystep", step_delay, _MS_PER_S)

    def get_ocp_result(self) -> tuple[float, float]:
        """Return (trip_current A, trip_time s) from the last OCP run."""
        return self.link.query_float(":SOUR:OCP:VOCP?"), self.link.query_float(
            ":SOUR:OCP:TOCP?"
        ) / _MS_PER_S

    def set_opp_test(
        self,
        start_power: float,
        step_power: float,
        *,
        von: float,
        step_delay: float | None = None,
        allow_live: bool = False,
    ) -> None:
        """Configure the over-power-protection search (ramp from start by step).

        Powers are watts, ``von`` volts, ``step_delay`` seconds.
        """
        self._require_off("set_opp_test", allow_live)
        self._set_function_mode("OPP")
        self._setq(":SOUR:OPP:PSET", start_power)
        self._setq(":SOUR:OPP:PSTep", step_power)
        self._setq(":SOUR:OPP:VON", von)
        if step_delay is not None:
            self._setq(":SOUR:OPP:PDELaystep", step_delay, _MS_PER_S)

    def get_opp_result(self) -> tuple[float, float]:
        """Return (trip_power W, trip_time s) from the last OPP run."""
        return self.link.query_float(":SOUR:OPP:VOPP?"), self.link.query_float(
            ":SOUR:OPP:TOPP?"
        ) / _MS_PER_S

    # ---------- waveform recorder (WAVE screen) ----------

    def get_wave_window_time(self) -> float:
        """Window (span) time of the waveform display, seconds."""
        return self.link.query_float(":SOUR:WAV:TIM?")

    def cmd_wave_window_increase(self) -> None:
        """Step the waveform-display window time up one step."""
        self.link.write(":SOUR:WAV:TIM ADD")

    def cmd_wave_window_decrease(self) -> None:
        """Step the waveform-display window time down one step."""
        self.link.write(":SOUR:WAV:TIM SUB")

    def set_wave_time_step_scale(self, scale: int) -> int:
        """Waveform window-time step scale, 1 or 10."""
        if int(scale) not in (1, 10):
            raise ValueError(f"time step scale must be 1 or 10, got {scale!r}")
        self.link.write(f":SOUR:WAV:TST {int(scale)}")
        return self.get_wave_time_step_scale()

    def get_wave_time_step_scale(self) -> int:
        """Waveform window-time step scale, 1 or 10."""
        return self.link.query_int(":SOUR:WAV:TST?")

    # ---------- trigger ----------

    def set_trigger_source(self, source: str) -> str:
        """Set what advances transient/list steps; a :class:`TriggerSource`."""
        self.link.write(f":TRIG:SOUR {str(source)}")
        return self.get_trigger_source()

    def get_trigger_source(self) -> str:
        """Present trigger source."""
        return self.link.query(":TRIG:SOUR?").strip()

    def cmd_trigger(self) -> None:
        """Fire an immediate bus trigger; requires TriggerSource.BUS."""
        self.link.write(":TRIG")

    # ---------- status ----------

    def get_questionable_condition(self) -> int:
        """Questionable status condition register."""
        return self.link.query_int(":STAT:QUES:COND?")

    def get_questionable_event(self) -> int:
        """Questionable status event register, cleared on read."""
        return self.link.query_int(":STAT:QUES?")

    def set_questionable_enable(self, mask: int) -> int:
        """Questionable status enable mask."""
        self.link.write(f":STAT:QUES:ENAB {int(mask)}")
        return self.link.query_int(":STAT:QUES:ENAB?")

    def get_operation_condition(self) -> int:
        """Operation status condition register."""
        return self.link.query_int(":STAT:OPER:COND?")

    def get_operation_event(self) -> int:
        """Operation status event register, cleared on read."""
        return self.link.query_int(":STAT:OPER?")

    def set_operation_enable(self, mask: int) -> int:
        """Operation status enable mask."""
        self.link.write(f":STAT:OPER:ENAB {int(mask)}")
        return self.link.query_int(":STAT:OPER:ENAB?")

    def get_status_byte(self) -> int:
        """Status byte."""
        return self.link.query_int("*STB?")

    def cmd_status_preset(self) -> None:
        """Reset the status enable registers to defaults."""
        self.link.write(":STAT:PRES")

    # ---------- system ----------

    def cmd_reset(self) -> None:
        """Reset to the power-on default state."""
        self.link.write("*RST")

    def cmd_clear_status(self) -> None:
        """Clear the status byte and event registers."""
        self.link.write("*CLS")

    def cmd_trigger_bus(self) -> None:
        """Send a bus trigger in the common-command form."""
        self.link.write("*TRG")

    def get_scpi_version(self) -> str:
        """SCPI version string."""
        return self.link.query(":SYST:VERS?").strip()

    def get_options(self) -> str:
        """Installed options."""
        return self.link.query("*OPT?").strip()

    def self_test(self) -> str:
        """Run the self-test; returns the per-rail PASS/FAIL string."""
        return self.link.query("*TST?").strip()

    def get_errors(self) -> list[str]:
        """Drain and return the error queue (empty list means clean)."""
        out: list[str] = []
        for _ in range(50):
            resp = self.link.query(":SYST:ERR?").strip()
            if resp.split(",", 1)[0].strip().lstrip("+") == "0":
                break
            out.append(resp)
        return out

    # ---------- snapshot / restore ----------

    def snapshot(self) -> dict:
        """Capture the touched settings for exact restore (read-back only)."""
        return {
            "output": self.get_output_state(),
            "remote_sense": self.get_remote_sense_state(),
            "function_mode": self.link.query(":SOUR:FUNC:MODE?").strip(),
            "load": self.get_load(),
        }

    def restore(self, snapshot: dict) -> None:
        """Restore settings captured by :meth:`snapshot` (input restored last).

        Re-applies the captured static load mode/level/limits, the remote-sense
        state and the function mode, then the input state. A captured ``NONE``
        load (the instrument was in a waveform/list/battery state) leaves the
        mode to the ``function_mode`` restore.
        """
        self.set_output_state(False)
        state: LoadState = snapshot["load"]
        with contextlib.suppress(Exception):
            if state.mode == "CURR":
                self.set_load_current(
                    state.level,
                    v_limit=state.v_limit,
                    von=state.von,
                    slew=state.slew,
                    allow_live=True,
                )
            elif state.mode == "VOLT":
                self.set_load_voltage(state.level, i_limit=state.i_limit, allow_live=True)
            elif state.mode == "RES":
                self.set_load_resistance(state.level, i_limit=state.i_limit, allow_live=True)
            elif state.mode == "POW":
                self.set_load_power(state.level, i_limit=state.i_limit, allow_live=True)
        with contextlib.suppress(Exception):
            self.link.write(f":SOUR:FUNC:MODE {snapshot['function_mode']}")
        self.set_remote_sense_state(snapshot["remote_sense"])
        self.set_output_state(snapshot["output"])

    @contextlib.contextmanager
    def preserved_setup(self) -> Iterator:
        """Context manager: snapshot on enter, restore the exact state on exit."""
        state = self.snapshot()
        try:
            yield self
        finally:
            self.restore(state)


def _main(argv: list[str] | None = None) -> int:
    """Live read-mostly self-test against the bench DL3021A (input stays OFF)."""
    argv = argv or []
    if len(argv) < 2:
        print(f"usage: python {__file__} RESOURCE")
        return 2
    resource = argv[1]
    load = Rigol_DL3000(resource)
    print(f"connecting to {load.resource_name} ...")
    idn = load.connect()
    print(f"  IDN: {idn}")
    print(f"  model={load.model} serial={load.serial} fw={load.firmware}")
    try:
        print(f"  SCPI version : {load.get_scpi_version()}")
        print(f"  options      : {load.get_options()}")
        print(f"  output state : {load.get_output_state()}")
        print(f"  remote sense : {load.get_remote_sense_state()}")
        print(f"  load         : {load.get_load()}")
        r = load.meas_vip()
        print(f"  meas         : V={r.voltage:.4f}  I={r.current:.4f}  P={r.power:.4f}")
        with load.preserved_setup():
            st = load.set_load_current(0.1, v_limit=8.0, allow_live=True)
            print(f"  set_load_current(0.1, v_limit=8) -> {st}")
        errs = load.get_errors()
        print(f"  error queue  : {errs or 'clean'}")
    finally:
        load.disconnect()
        print("disconnected (input off).")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
