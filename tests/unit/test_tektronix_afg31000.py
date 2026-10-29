"""Hardware-free tests for the Tektronix AFG31000 driver.

The fake below reproduces the behaviour measured on the bench AFG31102 on
2026-10-28 (projects/afg31000_probe.py, ADR-054 and ADR-055), so the tests
check the driver against how the instrument acts, not against canned strings:

- an amplitude or offset outside the voltage limit is refused with -222, and a
  limit written inside the present swing pulls the amplitude down to fit;
- a bare amplitude is read in the panel unit, a VPP suffix always in Vpp;
- the load setting rescales the levels and limits, which are source voltages;
- channel n plays only edit memory n, and FUNC? answers EMEM for both;
- only one run mode runs at a time: AM, FM, PM, FSK, PWM, burst and sweep each
  end the others silently, and a shape that cannot carry a mode ends it;
- AM, FM, PM and FSK are refused on a pulse and PWM on anything else (-221);
- angles are written in degrees (DEG) or radians and always read in radians;
- each value is held at its measured precision, and pulse edges are truncated;
- while the pulse is selected its period and the frequency are one register;
  entering the pulse takes the previous shape's frequency and clamps a width
  that no longer fits, and the clamp stays;
- burst idle is taken only while the burst is on;
- the CH1=CH2 couplings are one shared state that either channel follows;
- a binary block over a VXI-11 session fails the first read after it with
  INV_PROT, and lands only once a later read succeeds there;
- each interface has its own error queue; a fresh VXI-11 queue reads -500 first.

Where the bench did not show what the instrument does, the fake refuses, the
strict choice, so a test fails if the driver depends on a guess. Those places
say UNVERIFIED.
"""

from __future__ import annotations

import copy
import math
import re
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

import pytest
import pyvisa as visa

import drivers.tektronix_afg31000 as afg_module
from common.errors import InstrumentError
from common.state import BurstState, ModulationState, SweepState, VoltageLimits, WaveformState
from drivers.tektronix_afg31000 import (
    BurstIdle,
    BurstMode,
    FileEntry,
    ModulationShape,
    ModulationSource,
    Polarity,
    Spacing,
    Tektronix_AFG31000,
    _AddInChannel,
    _Channel,
)

_RMS_FACTOR = {"SIN": 2 * math.sqrt(2), "SQU": 2.0}
_SHAPES = {
    "SIN", "SQU", "PULS", "RAMP", "PRN", "DC", "SINC", "GAUS", "LOR", "ERIS", "EDEC", "HAV",
    "EMEM", "EFIL",
}  # fmt: skip
# Highest frequency of each shape on the AFG31102 (stage shapes2; the sine from the spec).
_MAX_FREQ = {"SIN": 100e6, "PULS": 80e6, "EMEM": 50e6} | dict.fromkeys(
    ("RAMP", "SINC", "GAUS", "LOR", "ERIS", "EDEC", "HAV"), 1e6
)
# The shapes each run mode cannot run on (stages mod2 and burst).
_REFUSES = {
    "AM": {"PULS", "PRN", "DC"},
    "FM": {"PULS", "PRN", "DC"},
    "PM": {"PULS", "PRN", "DC"},
    "FSK": {"PULS", "PRN", "DC"},
    "BURS": {"PRN", "DC"},
    "SWE": {"PULS", "PRN", "DC"},
}
_EDGE_MIN = 4e-9
_INV_PROT = visa.constants.StatusCode.error_invalid_protocol
_OUT_OF_RANGE = '-222,"Data out of range"'
_CONFLICT = '-221,"Settings conflict"'
_NOT_ALLOWED = '-108,"Parameter not allowed"'
_NOT_FOUND = '-256,"File name not found"'


def _q(x: float, digits: int, floor_exp: int, rounding: str = ROUND_HALF_UP) -> float:
    """``x`` as the instrument holds it: ``digits`` figures, never finer than 10**floor_exp."""
    if x == 0 or not math.isfinite(x):
        return float(x)
    exp = max(math.floor(math.log10(abs(x))) - digits + 1, floor_exp)
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(exp), rounding=rounding))


def _number(arg: str) -> tuple[float, str]:
    """Split ``12.3DEG`` into (12.3, "DEG")."""
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)", arg.strip())
    assert m, arg
    return float(m.group(1)), m.group(2).upper()


def _on(arg: str) -> bool:
    return arg.strip().upper() in ("ON", "1")


def _path(arg: str) -> str:
    p = arg.strip().strip('"').replace("\\", "/").rstrip("/")
    return p[0].upper() + p[1:] if len(p) >= 2 and p[1] == ":" else p


def _parent(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _pulse_fits(width: float, period: float, lead: float, trail: float) -> bool:
    """The manual's rule, which the bench enforced: 0.625(lead+trail) <= width <= T - 0.8(...)."""
    edges = lead + trail
    slack = 1e-9 * period
    return 0.625 * edges <= width + slack and width <= period - 0.8 * edges + slack


class _ChannelState:
    def __init__(self) -> None:
        self.output = False
        self.load = 50.0
        self.polarity = "NORM"
        self.func = "SIN"
        self.efile = ""
        self.freq = 1e6
        self.unit = "VPP"
        self.vpp = 1.0
        self.offset = 0.0
        self.lim_low = -5.0
        self.lim_high = 5.0
        self.phase = 0.0  # degrees
        self.symmetry = 50.0
        self.comb = ""
        self.noise = 10.0
        self.run: str | None = None  # None (continuous), AM, FM, PM, FSK, PWM, BURS or SWE
        self.pulse = {
            "per": 1e-6,
            "width": 500e-9,
            "lead": 4e-9,
            "trail": 4e-9,
            "delay": 0.0,
            "hold": "WIDT",
        }
        self.fm = {"dev": 1e3, "rate": 10e3, "shape": "SIN", "source": "INT"}
        self.am = {"level": 50.0, "rate": 10e3, "shape": "SIN", "source": "INT"}
        self.pm = {"level": 90.0, "rate": 10e3, "shape": "SIN", "source": "INT"}  # degrees
        self.fsk = {"level": 1e6, "rate": 50.0, "source": "INT"}
        self.pwm = {"level": 5.0, "rate": 10e3, "shape": "SIN", "source": "INT"}  # percent
        self.sweep = {"STAR": 1e5, "STOP": 1e6, "TIME": 0.01, "HTIM": 0.0, "RTIM": 0.001}
        self.spacing = "LIN"
        self.sweep_mode = "AUTO"
        self.burst = {"mode": "TRIG", "ncyc": 5.0, "tdel": 0.0, "idle": "START"}
        self.emem = bytes.fromhex("1fff3ffe")
        self._per_written = False

    def mod(self, kind: str) -> dict:
        return {"AM": self.am, "FM": self.fm, "PM": self.pm, "FSK": self.fsk, "PWM": self.pwm}[kind]


class _Session:
    """One pyvisa resource open on the fake instrument, with its interface's error queue."""

    def __init__(self, inst: FakeAfg, name: str, errors: list[str]) -> None:
        self.inst = inst
        self.name = name
        self.errors = errors
        self.timeout = 2000
        self.cleared = 0
        self.closed = False
        self.read_termination = None
        self.write_termination = None
        self.fail_next_query: list[visa.VisaIOError] = []
        self.busy_reads = 0
        self._block: tuple[int, bytes] | None = None
        self._block_busy = False

    @property
    def is_socket(self) -> bool:
        return self.name.upper().endswith("::SOCKET")

    def _check_open(self) -> None:
        assert not self.closed, f"{self.name} used after close()"

    def write(self, message: str, **kwargs) -> int:
        self._check_open()
        self.inst.log.append(message)
        self.inst._run(self, message)
        return len(message)

    def write_raw(self, data: bytes) -> int:
        self._check_open()
        match = re.match(rb"DATA:DATA EMEM(\d),\s*#(\d)", data)
        assert match, data[:40]
        n, ndigits = int(match.group(1)), int(match.group(2))
        start = match.end()
        count = int(data[start : start + ndigits])
        payload = data[start + ndigits : start + ndigits + count]
        assert data[start + ndigits + count :] == b"\n"
        self.inst.log.append(f"DATA:DATA EMEM{n},<{count} bytes>")
        self.inst.blocks.append((self.name, n, count))
        if self.is_socket:
            if b"\n" in payload:
                # The socket parser ends the message at the newline (ADR-055).
                self.errors.append('-102,"Syntax error; unexpected EOI"')
            else:
                self.inst.ch[n].emem = payload
        else:
            self._block = (n, payload)
            self._block_busy = True
        return len(data)

    def query(self, message: str, **kwargs) -> str:
        self._check_open()
        if self.fail_next_query:
            raise self.fail_next_query.pop(0)
        if self._block_busy:
            self._block_busy = False
            self.busy_reads += 1
            raise visa.VisaIOError(_INV_PROT)
        if self._block is not None:
            n, payload = self._block
            self.inst.ch[n].emem = payload
            self._block = None
        self.inst.queries.append(message)
        return self.inst._ask(self, message)

    def query_binary_values(self, message: str, datatype: str = "B", container=bytes):
        self._check_open()
        self.inst.queries.append(message)
        n = int(re.fullmatch(r"DATA\? EMEM(\d)", message).group(1))
        data = self.inst.ch[n].emem
        if not self.is_socket and b"\n" in data:
            raise visa.VisaIOError(_INV_PROT)
        assert len(data) <= 32_000, "a read this large never answered and left the socket deaf"
        return container(data)

    def clear(self) -> None:
        self.cleared += 1

    def close(self) -> None:
        if self._block is not None:
            # Closing a VXI-11 session at once dropped the block (ADR-055).
            self.inst.discarded.append(self._block)
            self._block = None
        self.closed = True


class FakeAfg(_Session):
    """A pyvisa-resource stand-in that behaves like the bench AFG31102.

    It is the main session itself; ``open_resource`` hands out side sessions.
    """

    def __init__(self, model: str = "AFG31102", resource: str = "TCPIP0::x::5025::SOCKET"):
        super().__init__(self, resource, [])
        self.model = model
        self.ch = {1: _ChannelState(), 2: _ChannelState()}
        self.reference = "EXT"
        self.trigger_source = "EXT"
        self.trigger_timer = 1e-3
        self.slope = "POS"
        self.trigger_out_mode = "TRIG"
        self.panel_lock = False
        self.conc = {"FREQ": False, "VOLT": False, "PHAS": False}
        self.ques = 0
        self.memories: dict[int, dict | None] = {0: self._setup(), 1: None, 2: None, 3: None}
        self.memories[4] = None
        self.locks = {1: False, 2: False, 3: False, 4: False}
        self.auto_recall = True
        self.dirs = {"M:", "M:/lost+found"}
        self.files: dict[str, tuple[int, object]] = {"M:/.setup0.tfs": (2048, None)}
        self.cwd = "M:"
        self.vxi_errors = ['-500,"Power on"']
        self.log: list[str] = []
        self.queries: list[str] = []
        self.blocks: list[tuple[str, int, int]] = []
        self.sessions: list[_Session] = []
        self.discarded: list[tuple[int, bytes]] = []
        self.events: list[str] = []
        self.beeps = 0
        self.triggers = 0
        self.rearms: list[int] = []
        self.phase_syncs = 0
        self.self_test_timeout: int | None = None
        self._queue = self.errors

    def open_resource(self, rm, name: str) -> _Session:
        if name == self.name:
            self.closed = False
            return self
        session = _Session(self, name, self.vxi_errors)
        self.sessions.append(session)
        return session

    # ---- dispatch ----

    def _run(self, session: _Session, message: str) -> None:
        self._queue = session.errors
        self._execute(message)

    def _ask(self, session: _Session, message: str) -> str:
        self._queue = session.errors
        self._session = session
        return self._answer(message)

    def _error(self, text: str) -> None:
        self._queue.append(text)

    def _execute(self, message: str) -> None:
        head, _, arg = message.partition(" ")
        if head == "*CLS":
            self._queue.clear()
        elif head == "*RST":
            for n in self.ch:
                emem = self.ch[n].emem
                self.ch[n] = _ChannelState()
                self.ch[n].emem = emem
            self.conc = dict.fromkeys(self.conc, False)
        elif head == "*TRG":
            self.triggers += 1
        elif head == "*SAV":
            if self.locks.get(int(arg)):
                self._error('-258,"Media protected"')
            else:
                self.memories[int(arg)] = self._setup()
        elif head == "*RCL":
            setup = self.memories[int(arg)]
            if setup is None:
                self._error(_NOT_FOUND)
            else:
                self._recall(setup)
        elif head == "ROSC:SOUR":
            self.reference = arg
        elif head == "TRIG:SOUR":
            self.trigger_source = arg
        elif head == "TRIG:TIM":
            seconds = _q(float(arg), 4, -9)
            if not 1e-6 <= seconds <= 500:
                self._error(_OUT_OF_RANGE)
            else:
                self.trigger_timer = seconds
        elif head == "TRIG:SLOP":
            self.slope = arg
        elif head == "OUTP:TRIG:MODE":
            self.trigger_out_mode = arg
        elif head == "SYST:KLOC":
            self.panel_lock = _on(arg)
        elif head == "SYST:BEEP":
            self.beeps += 1
        elif head == "MEM:STAT:DEL":
            self.memories[int(arg)] = None
        elif head == "MEM:STAT:LOCK":
            location, state = arg.split(",")
            self.locks[int(location)] = _on(state)
        elif head == "MEM:STAT:REC:AUTO":
            self.auto_recall = _on(arg)
        elif head.startswith("MMEM:") or head in ("SAV:SETU", "RECALL:SETU"):
            self._file_command(head, arg)
        elif head == "AFGC:CSC":
            a, b = (int(x) for x in re.fullmatch(r"CH(\d),CH(\d)", arg).groups())
            copied = copy.deepcopy(self.ch[a])
            copied.emem = self.ch[b].emem
            self.ch[b] = copied
            # Both outputs read off after a copy, either way round (bench, 2026-10-28).
            self.ch[a].output = self.ch[b].output = False
        elif m := re.fullmatch(r"OUTP(\d)(:IMP|:POL)?", head):
            c = self.ch[int(m.group(1))]
            if m.group(2) == ":IMP":
                self._set_load(c, math.inf if arg == "INF" else float(arg))
            elif m.group(2) == ":POL":
                c.polarity = arg
            else:
                c.output = _on(arg)
        elif m := re.fullmatch(r"SOUR([34]):POW", head):
            level = _q(float(arg), 6, 0)  # whole percent, 0.5 -> 1 (stage noiseadd)
            if not 0 <= level <= 50:
                self._error(_OUT_OF_RANGE)
            else:
                self.ch[int(m.group(1)) - 2].noise = level
        elif m := re.fullmatch(r"SOUR(\d):(\S+)", head):
            self._channel_write(int(m.group(1)), m.group(2), arg)
        else:
            raise AssertionError(f"fake does not know {message!r}")

    # ---- one channel ----

    def _channel_write(self, n: int, header: str, arg: str) -> None:
        c = self.ch[n]
        if header in ("FREQ:CONC", "VOLT:CONC", "PHAS:CONC"):
            self._couple(header[:4], _on(arg))
        elif header == "PHAS:INIT":
            self.phase_syncs += 1
        elif header == "BURS:INFI:REAR":
            self.rearms.append(n)
        elif header == "FUNC":
            self._select(n, arg)
        elif header == "FUNC:EFIL":
            if not arg.strip('"'):
                self._error('-232,"Invalid format"')  # bench, 2026-10-28
                return
            c.efile = arg.strip('"')
        elif header == "FUNC:RAMP:SYMM":
            symmetry = _q(float(arg), 6, -1)
            if not 0 <= symmetry <= 100:
                self._error(_OUT_OF_RANGE)
            else:
                c.symmetry = symmetry
        elif header == "FREQ":
            self._write_frequency(n, _q(float(arg), 12, -6))
        elif header == "VOLT":
            self._write_levels(n, vpp=self._vpp_from(c, arg))
        elif header == "VOLT:OFFS":
            self._write_levels(n, offset=_q(float(arg), 4, -3))
        elif header in ("VOLT:LIM:HIGH", "VOLT:LIM:LOW"):
            value = _q(float(arg), 4, -3)
            if header.endswith("HIGH"):
                c.lim_high = value
            else:
                c.lim_low = value
            room = 2 * min(c.lim_high - c.offset, c.offset - c.lim_low)
            c.vpp = _q(min(c.vpp, max(room, 0.0)), 4, -4)
        elif header == "VOLT:UNIT":
            c.unit = arg
        elif header == "PHAS":
            self._write_phase(n, arg)
        elif header == "COMB:FEED":
            token = arg.strip('"').upper()
            if token not in ("", "NOIS", "EXT", "BOTH") or (n == 2 and token in ("EXT", "BOTH")):
                self._error('-150,"String data error"')
            else:
                c.comb = token
        elif header == "FREQ:MODE":
            self._run_mode(n, "SWE", arg.startswith("SWE"))
        elif header.endswith(":STAT"):
            self._run_mode(n, header[: -len(":STAT")], _on(arg))
        elif header in ("AM", "FM", "PM", "FSK", "PWM:DCYC"):
            self._write_mod_level(n, header.split(":")[0], arg)
        elif m := re.fullmatch(r"(AM|FM|PM|PWM):INT:FREQ|(FSK):INT:RATE", header):
            c.mod(m.group(1) or m.group(2))["rate"] = _q(float(arg), 4, -3)
        elif m := re.fullmatch(r"(AM|FM|PM|PWM):INT:FUNC", header):
            shapes = ("SIN", "SQU", "TRI", "RAMP", "NRAM", "PRN", "EMEM", f"EMEM{n}", "EFIL")
            if arg not in shapes:
                self._error(_NOT_ALLOWED)
            else:
                c.mod(m.group(1))["shape"] = "EMEM" if arg.startswith("EMEM") else arg
        elif m := re.fullmatch(r"(AM|FM|PM|FSK|PWM):SOUR", header):
            c.mod(m.group(1))["source"] = arg
        elif header in ("FREQ:STAR", "FREQ:STOP"):
            c.sweep[header.split(":")[1]] = _q(float(arg), 9, -6)
        elif header in ("SWE:TIME", "SWE:HTIM", "SWE:RTIM"):
            c.sweep[header.split(":")[1]] = _q(float(arg), 4, -3)
        elif header == "SWE:SPAC":
            c.spacing = arg
        elif header == "SWE:MODE":
            c.sweep_mode = arg
        elif header.startswith("PULS:"):
            self._pulse_write(n, header[len("PULS:") :], arg)
        elif header.startswith("BURS:"):
            self._burst_write(n, header[len("BURS:") :], arg)
        else:
            raise AssertionError(f"fake does not know SOUR{n}:{header} {arg}")

    def _select(self, n: int, token: str) -> None:
        c = self.ch[n]
        if token not in _SHAPES and token != f"EMEM{n}":
            self._error(_NOT_ALLOWED)
            return
        new = "EMEM" if token.startswith("EMEM") else token
        if c.unit == "VRMS":  # the panel's Vrms value is kept, so the peak moves
            vrms = c.vpp / _RMS_FACTOR.get(c.func, 2 * math.sqrt(2))
            c.vpp = _q(vrms * _RMS_FACTOR.get(new, 2 * math.sqrt(2)), 4, -4)
        entering = new == "PULS" and c.func != "PULS"
        c.func = new
        if c.run is not None and not self._carries(c.run, new):
            c.run = None
        if entering:
            self._enter_pulse(n)

    @staticmethod
    def _carries(kind: str, func: str) -> bool:
        return func == "PULS" if kind == "PWM" else func not in _REFUSES[kind]

    def _enter_pulse(self, n: int) -> None:
        """Model the stale hazard of stages shapes2 and followup.

        A period written while another shape was selected is kept; otherwise
        the previous shape's frequency is taken. A width that no longer fits is
        clamped (99.995 percent duty seen on the bench), and it stays.
        """
        c, p = self.ch[n], self.ch[n].pulse
        if c._per_written:
            c.freq = 1.0 / p["per"]
        else:
            old = p["per"]
            p["per"] = 1.0 / c.freq
            if p["hold"] == "DUTY":
                p["width"] = _q(p["width"] * p["per"] / old, 5, -12)
        c._per_written = False
        if not _pulse_fits(p["width"], p["per"], p["lead"], p["trail"]):
            edges = p["lead"] + p["trail"]
            top = min(0.99995 * p["per"], p["per"] - 0.8 * edges)
            p["width"] = _q(max(top, 0.625 * edges), 5, -12, ROUND_DOWN)
            self.events.append(f"CH{n} pulse width clamped to {p['width']:g} s")

    def _write_frequency(self, n: int, hz: float) -> None:
        targets = [n, 3 - n] if self.conc["FREQ"] else [n]
        plans = {}
        for t in targets:
            c, p = self.ch[t], self.ch[t].pulse
            if not 0 < hz <= _MAX_FREQ.get(c.func, math.inf):
                self._error(_OUT_OF_RANGE)
                return
            if c.func == "PULS":
                per = 1.0 / hz
                width = p["width"]
                if p["hold"] == "DUTY":
                    width = _q(width * per / p["per"], 5, -12)
                # UNVERIFIED: a period change that leaves the pulse no room was
                # not tried on the bench; the fake refuses it, the strict choice.
                if not _pulse_fits(width, per, p["lead"], p["trail"]):
                    self._error(_OUT_OF_RANGE)
                    return
                plans[t] = (per, width)
        for t in targets:
            self.ch[t].freq = hz
            if t in plans:
                self.ch[t].pulse["per"], self.ch[t].pulse["width"] = plans[t]

    def _within(self, c: _ChannelState, vpp: float, offset: float) -> bool:
        if c.func == "DC":  # the amplitude plays no part in DC (stage followup)
            vpp = 0.0
        return offset + vpp / 2 <= c.lim_high + 1e-9 and offset - vpp / 2 >= c.lim_low - 1e-9

    def _write_levels(self, n: int, *, vpp: float | None = None, offset: float | None = None):
        targets = [n, 3 - n] if self.conc["VOLT"] else [n]
        for t in targets:
            c = self.ch[t]
            new_vpp = c.vpp if vpp is None else vpp
            new_offset = c.offset if offset is None else offset
            if not self._within(c, new_vpp, new_offset):
                self._error(_OUT_OF_RANGE)
                return
        for t in targets:
            if vpp is not None:
                self.ch[t].vpp = vpp
            if offset is not None:
                self.ch[t].offset = offset

    @staticmethod
    def _vpp_from(c: _ChannelState, arg: str) -> float:
        value, unit = _number(arg)
        factor = _RMS_FACTOR.get(c.func, 2 * math.sqrt(2))
        unit = unit or c.unit
        if unit == "VRMS":
            value *= factor
        elif unit == "DBM":
            value = math.sqrt(10 ** (value / 10) / 1000 * 50) * factor
        return _q(value, 4, -4)

    @staticmethod
    def _volt_in_unit(c: _ChannelState) -> float:
        factor = _RMS_FACTOR.get(c.func, 2 * math.sqrt(2))
        if c.unit == "VRMS":
            return c.vpp / factor
        if c.unit == "DBM":
            return 10 * math.log10((c.vpp / factor) ** 2 / 50 * 1000)
        return c.vpp

    @staticmethod
    def _set_load(c: _ChannelState, ohms: float) -> None:
        """The levels and limits are held as source voltages, so a load change rescales them."""

        def k(r: float) -> float:
            return 1.0 if r == math.inf else r / (r + 50)

        ratio = k(ohms) / k(c.load)
        c.load = ohms
        c.vpp = _q(c.vpp * ratio, 4, -4)
        c.offset = _q(c.offset * ratio, 4, -3)
        c.lim_low = _q(c.lim_low * ratio, 4, -3)
        c.lim_high = _q(c.lim_high * ratio, 4, -3)

    def _write_phase(self, n: int, arg: str) -> None:
        value, unit = _number(arg)
        degrees = value if unit == "DEG" else math.degrees(value)
        if abs(degrees) > 180 + 1e-9:
            self._error(_OUT_OF_RANGE)
            return
        for t in [n, 3 - n] if self.conc["PHAS"] else [n]:
            self.ch[t].phase = _q(degrees, 12, -2)

    def _run_mode(self, n: int, kind: str, on: bool) -> None:
        c = self.ch[n]
        if on:
            if not self._carries(kind, c.func):
                self._error(_CONFLICT)
                return
            if c.run != kind and kind != "BURS":
                self._clear_noise_add(c)
            c.run = kind
        elif c.run == kind:
            c.run = None
            if kind != "BURS":
                self._clear_noise_add(c)

    @staticmethod
    def _clear_noise_add(c) -> None:
        """AM, FM, PM, FSK, PWM and the sweep, turning on or off, clear noise add silently.

        A burst keeps it, and so does AM:STAT ON with AM already on (stages
        noiseadd, followup2). Seen on channel 2 only, so with Add In on as well
        (BOTH) keeping EXT is a guess.  # UNVERIFIED for BOTH
        """
        c.comb = {"NOIS": "", "BOTH": "EXT"}.get(c.comb, c.comb)

    def _write_mod_level(self, n: int, kind: str, arg: str) -> None:
        c = self.ch[n]
        value, unit = _number(arg)
        if kind == "AM":
            level = _q(value, 6, -1)
            ok = 0 <= level <= 120
        elif kind == "FM":
            level = _q(value, 6, -6)
            ok = 0 <= level <= c.freq
        elif kind == "PM":
            degrees = value if unit == "DEG" else math.degrees(value)
            level = _q(degrees, 6, -1)
            ok = 0 <= degrees <= 180 + 1e-9
        elif kind == "FSK":
            level = _q(value, 7, -6)
            ok = 0 < level <= 100e6
        else:
            level = _q(value, 6, -1)
            duty = c.pulse["width"] / c.pulse["per"] * 100
            ok = 0 <= level <= min(duty, 100 - duty) + 1e-9
        if not ok:
            self._error(_OUT_OF_RANGE)
            return
        c.mod(kind)["dev" if kind == "FM" else "level"] = level

    def _pulse_write(self, n: int, sub: str, arg: str) -> None:
        c, p = self.ch[n], self.ch[n].pulse
        # A width is checked against the selected shape's period (stage followup).
        period = p["per"] if c.func == "PULS" else 1.0 / c.freq
        if sub == "HOLD":
            assert arg in ("WIDT", "DUTY"), arg
            p["hold"] = arg
        elif sub == "PER":
            if c.func == "PULS":
                self._write_frequency(n, 1.0 / float(arg))
            else:
                p["per"] = float(arg)
                c._per_written = True
        elif sub in ("WIDT", "DCYC"):
            if sub == "WIDT":
                width = _q(float(arg), 5, -12)
            else:
                width = _q(_q(float(arg), 6, -3) / 100 * period, 5, -12)
            if not _pulse_fits(width, period, p["lead"], p["trail"]):
                self._error(_OUT_OF_RANGE)
                return
            p["width"] = width
            if sub == "DCYC":
                # A duty with no room for the PWM deviation clamps it, silently (stage followup2).
                duty = width / period * 100
                c.pwm["level"] = min(c.pwm["level"], duty, 100 - duty)
        elif sub in ("TRAN", "TRAN:TRA"):
            edge = _EDGE_MIN if arg == "MIN" else _q(float(arg), 4, -12, ROUND_DOWN)
            lead, trail = (edge, p["trail"]) if sub == "TRAN" else (p["lead"], edge)
            # An edge that breaks the width rule is refused, not clamped (stage shapes2).
            if edge < _EDGE_MIN * (1 - 1e-9) or not _pulse_fits(p["width"], period, lead, trail):
                self._error(_OUT_OF_RANGE)
                return
            p["lead"], p["trail"] = lead, trail
        elif sub == "DEL":
            delay = _q(float(arg), 7, -12)
            if delay < 0:
                self._error(_OUT_OF_RANGE)
                return
            p["delay"] = delay
        else:
            raise AssertionError(f"fake does not know SOUR{n}:PULS:{sub} {arg}")

    def _burst_write(self, n: int, sub: str, arg: str) -> None:
        b = self.ch[n].burst
        if sub == "MODE":
            assert arg in ("TRIG", "GAT"), arg
            b["mode"] = arg
        elif sub == "NCYC":
            cycles = {"INF": math.inf, "MIN": 1.0, "MAX": 1e6}.get(arg)
            cycles = _q(float(arg), 12, 0) if cycles is None else cycles  # 3.7 -> 4
            if not (cycles == math.inf or 1 <= cycles <= 1e6):
                self._error(_OUT_OF_RANGE)
                return
            b["ncyc"] = cycles
        elif sub == "TDEL":
            b["tdel"] = _q(float(arg), 5, -12)
        elif sub == "IDLE":
            assert arg in ("START", "DC", "END"), arg
            if self.ch[n].run != "BURS":
                self._error(_CONFLICT)  # taken only while the burst is on (stage followup)
                return
            b["idle"] = arg
        elif sub == "INF:REAR":
            self._error('-113,"Undefined header"')  # the manual's form; INFI:REAR works
        else:
            raise AssertionError(f"fake does not know SOUR{n}:BURS:{sub} {arg}")

    def _couple(self, what: str, on: bool) -> None:
        if on and not self.conc[what]:
            # Turning a coupling on copies channel 1 to channel 2 (stages chan2, followup).
            a, b = self.ch[1], self.ch[2]
            if what == "FREQ":
                self._write_frequency(2, a.freq)
            elif what == "VOLT":
                b.vpp, b.offset = a.vpp, a.offset
            else:
                b.phase = a.phase
        self.conc[what] = on

    # ---- setups and files ----

    def _setup(self) -> dict:
        return {"channels": copy.deepcopy(self.ch)}

    def _recall(self, setup: dict) -> None:
        for n, saved in setup["channels"].items():
            c = copy.deepcopy(saved)
            c.emem = self.ch[n].emem
            c.output = False  # a recall never turns an output on (stage system2)
            self.ch[n] = c

    def _file_command(self, head: str, arg: str) -> None:
        if head == "MMEM:CDIR":
            if _path(arg) not in self.dirs:
                self._error(_NOT_FOUND)
            else:
                self.cwd = _path(arg)
        elif head == "MMEM:MDIR":
            path = _path(arg)
            if _parent(path) not in self.dirs:
                self._error(_NOT_FOUND)
            else:
                self.dirs.add(path)  # a folder already there queues no error (bench)
        elif head == "MMEM:DEL":
            path = _path(arg)
            if path in self.files:
                del self.files[path]
            elif path in self.dirs:
                assert not any(_parent(p) == path for p in [*self.files, *self.dirs]), path
                self.dirs.discard(path)
            else:
                self._error(_NOT_FOUND)
        elif head in ("MMEM:STOR:TRAC", "MMEM:LOAD:TRAC"):
            n, raw = re.fullmatch(r"EMEM(\d),(.+)", arg).groups()
            path, c = _path(raw), self.ch[int(n)]
            if head == "MMEM:STOR:TRAC":
                if _parent(path) not in self.dirs:
                    self._error(_NOT_FOUND)
                else:
                    self.files[path] = (512 + len(c.emem), c.emem)  # stage mmem_path
            elif not isinstance(self.files.get(path, (0, None))[1], bytes):
                self._error(_NOT_FOUND)
            else:
                c.emem = self.files[path][1]
        elif head == "SAV:SETU":
            path = _path(arg)
            if _parent(path) not in self.dirs:
                self._error(_NOT_FOUND)
            else:
                self.files[path] = (4096, self._setup())
        elif head == "RECALL:SETU":
            content = self.files.get(_path(arg), (0, None))[1]
            if not isinstance(content, dict):
                self._error(_NOT_FOUND)
            else:
                self._recall(content)
        else:
            raise AssertionError(f"fake does not know {head} {arg}")

    def _catalog(self) -> str:
        entries = ['"..,DIR,4096"'] if "/" in self.cwd else []
        for d in sorted(self.dirs):
            if _parent(d) == self.cwd:
                entries.append(f'"{d.rsplit("/", 1)[1]},DIR,4096"')
        for f, (size, _) in sorted(self.files.items()):
            if _parent(f) == self.cwd:
                entries.append(f'"{f.rsplit("/", 1)[1]},,{size}"')
        used = sum(size for size, _ in self.files.values())
        return ",".join([str(used), str(1_000_000_000 - used), *entries])

    # ---- queries ----

    def _answer(self, message: str) -> str:
        fixed = {
            "*IDN?": f"TEKTRONIX,{self.model},C000000,SCPI:99.0 FV:1.6.1",
            "*OPT?": '"AFG3BW100"',
            "*OPC?": "1",
            "ROSC:SOUR?": self.reference,
            "TRIG:SOUR?": self.trigger_source,
            "TRIG:TIM?": f"{self.trigger_timer:.12g}",
            "TRIG:SLOP?": self.slope,
            "OUTP:TRIG:MODE?": self.trigger_out_mode,
            "SYST:KLOC?": "1" if self.panel_lock else "0",
            "STAT:QUES:COND?": str(self.ques),
            "MEM:STAT:REC:AUTO?": "1" if self.auto_recall else "0",
            "MMEM:CDIR?": f'"{self.cwd}"',
        }
        if message in fixed:
            return fixed[message]
        if message == "*TST?":
            self.self_test_timeout = self._session.timeout
            return "0"
        if message == "SYST:ERR?":
            return self._queue.pop(0) if self._queue else '0,"No error"'
        if message == "MMEM:CAT?":
            return self._catalog()
        if m := re.fullmatch(r"MEM:STAT:(VAL|LOCK)\? (\d)", message):
            location = int(m.group(2))
            on = (
                self.memories[location] is not None if m.group(1) == "VAL" else self.locks[location]
            )
            return "1" if on else "0"
        if m := re.fullmatch(r"OUTP(\d)(:IMP|:POL)?\?", message):
            c = self.ch[int(m.group(1))]
            if m.group(2) == ":IMP":
                return "99.0e36" if c.load == math.inf else f"{c.load:g}"
            if m.group(2) == ":POL":
                return c.polarity
            return "1" if c.output else "0"
        if m := re.fullmatch(r"SOUR([34]):POW\?", message):
            return f"{self.ch[int(m.group(1)) - 2].noise:g}"
        m = re.fullmatch(r"SOUR(\d):(\S+)\?", message)
        assert m, message
        return self._channel_answer(int(m.group(1)), m.group(2))

    def _channel_answer(self, n: int, header: str) -> str:
        c, p, b = self.ch[n], self.ch[n].pulse, self.ch[n].burst
        if header in ("FREQ:CONC", "VOLT:CONC", "PHAS:CONC"):
            return "1" if self.conc[header[:4]] else "0"
        if m := re.fullmatch(r"(AM|FM|PM|FSK|PWM|BURS):STAT", header):
            return "1" if c.run == m.group(1) else "0"

        def g(x: float) -> str:
            return f"{x:.12g}"

        values = {
            "FUNC": c.func,
            "FUNC:EFIL": f'"{c.efile}"',
            "FUNC:RAMP:SYMM": g(c.symmetry),
            "FREQ": g(c.freq),
            "VOLT:UNIT": c.unit,
            "VOLT": g(self._volt_in_unit(c)),
            "VOLT:OFFS": g(c.offset),
            "VOLT:HIGH": g(round(c.offset + c.vpp / 2, 3)),
            "VOLT:LOW": g(round(c.offset - c.vpp / 2, 3)),
            "VOLT:LIM:HIGH": g(c.lim_high),
            "VOLT:LIM:LOW": g(c.lim_low),
            "PHAS": f"{math.radians(c.phase):.5f}",  # 12.35 deg read 0.21555 (stage chan2)
            "COMB:FEED": f'"{c.comb}"',
            "RMOD": {None: "CONT", "SWE": "SWE", "BURS": "BURS"}.get(c.run, "MOD"),
            "FREQ:MODE": "SWE" if c.run == "SWE" else "CW",
            "FM": g(c.fm["dev"]),
            "AM": g(c.am["level"]),
            "PM": f"{math.radians(c.pm['level']):.6g}",  # 90 deg read 1.57080 (stage mod2)
            "FSK": g(c.fsk["level"]),
            "PWM:DCYC": g(c.pwm["level"]),
            "FSK:INT:RATE": g(c.fsk["rate"]),
            "FSK:SOUR": c.fsk["source"],
            "FREQ:STAR": g(c.sweep["STAR"]),
            "FREQ:STOP": g(c.sweep["STOP"]),
            "SWE:TIME": g(c.sweep["TIME"]),
            "SWE:HTIM": g(c.sweep["HTIM"]),
            "SWE:RTIM": g(c.sweep["RTIM"]),
            "SWE:SPAC": c.spacing,
            "SWE:MODE": c.sweep_mode,
            "PULS:PER": g(p["per"]),
            "PULS:WIDT": g(p["width"]),
            "PULS:DCYC": g(_q(p["width"] / p["per"] * 100, 6, -3)),
            "PULS:TRAN": g(p["lead"]),
            "PULS:TRAN:TRA": g(p["trail"]),
            "PULS:DEL": g(p["delay"]),
            "PULS:HOLD": p["hold"],
            "BURS:MODE": b["mode"],
            "BURS:NCYC": "9.9e+37" if b["ncyc"] == math.inf else g(b["ncyc"]),
            "BURS:TDEL": g(b["tdel"]),
            "BURS:IDLE": b["idle"],
        }
        for kind in ("AM", "FM", "PM", "PWM"):
            mod = c.mod(kind)
            values[f"{kind}:INT:FREQ"] = g(mod["rate"])
            values[f"{kind}:INT:FUNC"] = mod["shape"]
            values[f"{kind}:SOUR"] = mod["source"]
        return values[header]


@pytest.fixture
def make_afg(monkeypatch):
    """Connect a driver to a FakeAfg. Returns (driver, fake)."""
    monkeypatch.setattr(afg_module.visa, "ResourceManager", lambda: None)
    monkeypatch.setattr(afg_module, "_BUSY_SETTLE_S", 0.0)
    monkeypatch.setattr(afg_module, "_UPLOAD_RETRY_S", 0.0)

    def factory(model: str = "AFG31102", resource: str = "TCPIP0::x::5025::SOCKET"):
        fake = FakeAfg(model, resource)
        afg = Tektronix_AFG31000(resource)
        monkeypatch.setattr(afg.link, "_open_resource", fake.open_resource)
        afg.connect()
        fake.log.clear()
        fake.queries.clear()
        return afg, fake

    return factory


def _writes(fake: FakeAfg) -> list[str]:
    return list(fake.log)


def _in_order(writes: list[str], *expected: str) -> bool:
    """Whether every ``expected`` write is present, in this order."""
    found = [writes.index(w) for w in expected if w in writes]
    return len(found) == len(expected) and found == sorted(found)


_INSTRUMENT_SETTINGS = (
    "reference",
    "trigger_source",
    "trigger_timer",
    "slope",
    "trigger_out_mode",
    "panel_lock",
    "conc",
)


def _fake_state(fake: FakeAfg) -> dict:
    """Every setting the fake holds, copied, for a before and after comparison."""
    return copy.deepcopy(
        {
            "channels": {
                n: {k: v for k, v in vars(c).items() if not k.startswith("_")}
                for n, c in fake.ch.items()
            },
            "instrument": {k: getattr(fake, k) for k in _INSTRUMENT_SETTINGS},
        }
    )


def _assert_same(before, after, where: str = "") -> None:
    if isinstance(before, dict):
        assert before.keys() == after.keys(), where
        for key in before:
            _assert_same(before[key], after[key], f"{where}/{key}")
    elif isinstance(before, float):
        assert after == pytest.approx(before, rel=1e-9, abs=1e-15), where
    else:
        assert after == before, where


# ---------------------------------------------------------------- model gating


def test_two_channel_100mhz_model_has_ch2_and_add_in(make_afg):
    afg, _ = make_afg("AFG31102")
    assert isinstance(afg.ch1, _AddInChannel)
    assert isinstance(afg.ch2, _Channel)
    assert not hasattr(afg.ch2, "set_external_add_state")
    assert afg.options == ["AFG3BW100"]


def test_25mhz_model_has_no_add_in(make_afg):
    afg, _ = make_afg("AFG31022")
    assert type(afg.ch1) is _Channel
    assert not hasattr(afg.ch1, "set_external_add_state")
    assert hasattr(afg, "ch2")


def test_single_channel_model_has_no_ch2(make_afg):
    afg, _ = make_afg("AFG31101")
    assert not hasattr(afg, "ch2")
    with pytest.raises(ValueError, match="channel must be 1..1"):
        afg.channel(2)


def test_wrong_instrument_is_refused(monkeypatch):
    monkeypatch.setattr(afg_module.visa, "ResourceManager", lambda: None)
    fake = FakeAfg("MSO68B")
    afg = Tektronix_AFG31000("TCPIP0::x::5025::SOCKET")
    monkeypatch.setattr(afg.link, "_open_resource", lambda rm, name: fake)
    with pytest.raises(InstrumentError, match="does not scan"):
        afg.connect()


def test_socket_gets_terminators_and_no_device_clear(make_afg):
    _, fake = make_afg(resource="TCPIP0::x::5025::SOCKET")
    assert fake.read_termination == "\n" and fake.write_termination == "\n"
    assert fake.cleared == 0


def test_vxi11_gets_a_device_clear(make_afg):
    _, fake = make_afg(resource="TCPIP0::x::inst0::INSTR")
    assert fake.cleared == 1


def test_interchannel_exists_only_on_two_channel_models(make_afg):
    afg, _ = make_afg("AFG31102")
    assert hasattr(afg, "interchannel")
    afg, _ = make_afg("AFG31101")
    assert not hasattr(afg, "interchannel")


# ---------------------------------------------------------------- waveform helpers


def test_sine_writes_limit_then_shape_then_levels_then_frequency(make_afg):
    afg, fake = make_afg()
    fake.ch[2].func = "SQU"
    state = afg.ch2.set_waveform_sine(1e3, 2.0, v_limit=1.5)
    writes = _writes(fake)
    order = [
        next(i for i, w in enumerate(writes) if w.startswith(prefix))
        for prefix in ("SOUR2:VOLT:LIM:", "SOUR2:FUNC SIN", "SOUR2:VOLT 2VPP", "SOUR2:FREQ 1000")
    ]
    assert order == sorted(order)
    assert isinstance(state, WaveformState)
    assert (state.function, state.frequency, state.amplitude) == ("SIN", 1e3, 2.0)
    assert afg.ch2.get_voltage_limits() == VoltageLimits(-1.5, 1.5, True)
    assert not any(w.startswith("OUTP") for w in writes)


def test_repeat_call_writes_nothing_and_one_change_writes_one_value(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_sine(44_416.0, 2.0, v_limit=1.5)
    fake.log.clear()
    afg.ch1.set_waveform_sine(44_416.0, 2.0, v_limit=1.5)
    assert _writes(fake) == []
    afg.ch1.set_waveform_sine(44_416.000001, 2.0, v_limit=1.5)
    assert _writes(fake) == ["SOUR1:FREQ 44416.000001"]


def test_swing_outside_the_limit_is_refused_before_any_write(make_afg):
    afg, fake = make_afg()
    with pytest.raises(ValueError, match="outside the limit"):
        afg.ch1.set_waveform_sine(1e3, 2.0, v_limit=0.9)
    with pytest.raises(ValueError, match="outside the limit"):
        afg.ch1.set_waveform_square(1e3, 1.0, v_limit=(-0.2, 1.0), offset=-0.1)
    assert _writes(fake) == []


@pytest.mark.parametrize(
    ("start", "end"),
    [((1.8, 0.0), (0.2, 0.85)), ((0.2, 0.85), (1.8, 0.0)), ((0.2, -0.85), (0.4, 0.75))],
)
def test_level_order_never_crosses_a_tight_limit(make_afg, start, end):
    afg, fake = make_afg()
    afg.ch1.set_waveform_sine(1e3, start[0], v_limit=1.0, offset=start[1])
    state = afg.ch1.set_waveform_sine(1e3, end[0], v_limit=1.0, offset=end[1])
    assert (state.amplitude, state.offset) == pytest.approx(end)
    assert fake.errors == []


def test_vrms_panel_unit_still_speaks_vpp(make_afg):
    afg, fake = make_afg()
    fake.ch[1].unit = "VRMS"
    state = afg.ch1.set_waveform_sine(1e3, 2.0, v_limit=1.5)
    assert "SOUR1:VOLT 2VPP" in _writes(fake)
    assert state.amplitude == pytest.approx(2.0, abs=2e-3)
    fake.log.clear()
    afg.ch1.set_waveform_sine(1e3, 2.0, v_limit=1.5)
    assert _writes(fake) == []


def test_set_amplitude_outside_the_limit_raises_with_the_instrument_error(make_afg):
    afg, _ = make_afg()
    afg.ch1.set_voltage_limits(-1.0, 1.0)
    with pytest.raises(InstrumentError, match="-222"):
        afg.ch1.set_amplitude(3.0)


def test_lower_limit_first_when_moving_the_window_down(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_voltage_limits(2.0, 4.0)
    fake.log.clear()
    afg.ch1.set_voltage_limits(-4.0, -2.0)
    assert _writes(fake) == ["SOUR1:VOLT:LIM:LOW -4", "SOUR1:VOLT:LIM:HIGH -2"]
    afg.ch1.set_voltage_limits(2.0, 4.0)
    assert _writes(fake)[-2:] == ["SOUR1:VOLT:LIM:HIGH 4", "SOUR1:VOLT:LIM:LOW 2"]


def test_ramp_writes_its_symmetry_and_the_triangle_is_fifty_percent(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_waveform_ramp(1e3, 1.0, v_limit=1.0, symmetry=12.3456)
    assert (state.function, state.symmetry, state.duty) == ("RAMP", 12.3, None)
    fake.log.clear()
    afg.ch1.set_waveform_ramp(1e3, 1.0, v_limit=1.0)  # None keeps the symmetry
    assert _writes(fake) == []
    state = afg.ch1.set_waveform_triangle(1e3, 1.0, v_limit=1.0)
    assert _writes(fake) == ["SOUR1:FUNC:RAMP:SYMM 50"]
    assert state.symmetry == 50.0


def test_noise_has_no_frequency_and_writes_none(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_waveform_noise(2.0, v_limit=1.5)
    assert (state.function, state.frequency, state.amplitude) == ("PRN", 0.0, 2.0)
    assert not any(w.startswith("SOUR1:FREQ") for w in _writes(fake))


def test_dc_level_may_reach_the_limit_and_writes_no_amplitude(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_waveform_dc(1.5, v_limit=1.5)
    assert (state.function, state.frequency, state.amplitude, state.offset) == ("DC", 0, 0, 1.5)
    assert not any(w.startswith("SOUR1:VOLT ") for w in _writes(fake))
    assert fake.errors == []
    fake.log.clear()
    with pytest.raises(ValueError, match="outside the limit"):
        afg.ch1.set_waveform_dc(1.6, v_limit=1.5)
    assert _writes(fake) == []


@pytest.mark.parametrize(
    ("method", "token"),
    [
        ("set_waveform_sinc", "SINC"),
        ("set_waveform_gaussian", "GAUS"),
        ("set_waveform_lorentz", "LOR"),
        ("set_waveform_exp_rise", "ERIS"),
        ("set_waveform_exp_decay", "EDEC"),
        ("set_waveform_haversine", "HAV"),
    ],
)
def test_built_in_shapes(make_afg, method, token):
    afg, fake = make_afg()
    state = getattr(afg.ch2, method)(10e3, 1.0, v_limit=1.0, offset=0.2)
    assert (state.function, state.frequency, state.amplitude, state.offset) == (
        token,
        10e3,
        1.0,
        0.2,
    )
    assert f"SOUR2:FUNC {token}" in _writes(fake)
    fake.log.clear()
    getattr(afg.ch2, method)(10e3, 1.0, v_limit=1.0, offset=0.2)
    assert _writes(fake) == []


def test_arb_file_names_the_file_then_selects_the_shape(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_waveform_arb_file("M:/waves/a.tfwx", 1e3, 1.0, v_limit=1.0)
    assert _in_order(_writes(fake), 'SOUR1:FUNC:EFIL "M:/waves/a.tfwx"', "SOUR1:FUNC EFIL")
    assert (state.function, state.frequency) == ("EFIL", 1e3)
    fake.log.clear()
    afg.ch1.set_waveform_arb_file("M:/waves/a.tfwx", 1e3, 1.0, v_limit=1.0)
    assert _writes(fake) == []
    afg.ch1.set_waveform_arb_file("M:/waves/b.tfwx", 1e3, 1.0, v_limit=1.0)
    assert _writes(fake) == ['SOUR1:FUNC:EFIL "M:/waves/b.tfwx"', "SOUR1:FUNC EFIL"]


# ---------------------------------------------------------------- pulse


def test_pulse_from_a_sine_writes_every_setting(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_waveform_pulse(
        1e3, 2.0, v_limit=1.5, width=10e-6, edge_time=20e-9, delay=1e-6
    )
    assert fake.errors == [] and fake.events == []
    assert (state.function, state.frequency, state.width, state.duty) == ("PULS", 1e3, 10e-6, 1.0)
    assert (state.edge_time, state.trailing_edge_time, state.delay) == (20e-9, 20e-9, 1e-6)
    assert _in_order(
        _writes(fake),
        "SOUR1:FUNC PULS",
        "SOUR1:FREQ 1000",
        "SOUR1:PULS:WIDT 1e-05",
        "SOUR1:PULS:TRAN 2e-08",
        "SOUR1:PULS:TRAN:TRA 2e-08",
        "SOUR1:PULS:DEL 1e-06",
    )
    assert not any(w.startswith("OUTP") for w in _writes(fake))


def test_pulse_to_a_shorter_period_writes_the_width_first(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=400e-6)
    fake.log.clear()
    # The fake refuses a period that leaves the held 400 us no room, so only
    # the width-first order gets through.
    state = afg.ch1.set_waveform_pulse(10e3, 1.0, v_limit=1.0, width=50e-6)
    assert _writes(fake) == ["SOUR1:PULS:WIDT 5e-05", "SOUR1:FREQ 10000"]
    assert (state.frequency, state.width) == (10e3, 50e-6)
    assert fake.errors == []


def test_pulse_to_a_longer_period_writes_the_frequency_first(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_pulse(10e3, 1.0, v_limit=1.0, width=50e-6)
    fake.log.clear()
    state = afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=400e-6)
    assert _writes(fake) == ["SOUR1:FREQ 1000", "SOUR1:PULS:WIDT 0.0004"]
    assert (state.frequency, state.width) == (1e3, 400e-6)


def test_pulse_with_the_duty_held_to_a_shorter_period_stays_valid(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, duty=99.99)
    fake.log.clear()
    # With the duty held, the period change scales the width: 99.99 percent of
    # 100 ns breaks the edge rule, so the duty has to change first.
    state = afg.ch1.set_waveform_pulse(10e6, 1.0, v_limit=1.0, duty=50.0)
    assert (state.frequency, state.duty) == (10e6, 50.0)
    assert _writes(fake) == ["SOUR1:PULS:DCYC 50", "SOUR1:FREQ 10000000"]
    assert fake.errors == []


def test_pulse_recovers_a_width_clamped_on_entering_the_shape(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=400e-6)
    afg.ch1.set_waveform_sine(10e3, 1.0, v_limit=1.0)
    state = afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0)
    assert fake.events == ["CH1 pulse width clamped to 9.9993e-05 s"]
    assert (state.frequency, state.width) == (1e3, 400e-6)
    assert fake.errors == []


def test_pulse_width_and_duty_together_are_refused_before_any_write(make_afg):
    afg, fake = make_afg()
    with pytest.raises(ValueError, match="not both"):
        afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=1e-4, duty=10.0)
    with pytest.raises(ValueError, match="edge_time sets both edges"):
        afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, edge_time=1e-8, leading_edge_time=1e-8)
    assert _writes(fake) == []


@pytest.mark.parametrize(
    ("freq", "width", "edge"),
    [(1e3, 10e-9, 20e-9), (1e6, 990e-9, 20e-9), (1e3, 2e-3, 4e-9)],
)
def test_pulse_breaking_the_edge_rule_is_refused_before_any_write(make_afg, freq, width, edge):
    afg, fake = make_afg()
    with pytest.raises(ValueError, match="refused"):
        afg.ch1.set_waveform_pulse(freq, 1.0, v_limit=1.0, width=width, edge_time=edge)
    assert _writes(fake) == []


def test_pulse_hold_follows_width_or_duty(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, duty=25.0)
    assert fake.ch[1].pulse["hold"] == "DUTY"
    assert "SOUR1:PULS:HOLD DUTY" in _writes(fake)
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=100e-6)
    assert fake.ch[1].pulse["hold"] == "WIDT"
    fake.log.clear()
    state = afg.ch1.set_waveform_pulse(2e3, 1.0, v_limit=1.0)  # neither: the width is kept
    assert not any("HOLD" in w for w in _writes(fake))
    assert (state.width, state.duty) == (100e-6, 20.0)


def test_pulse_edges_are_truncated_and_still_count_as_written(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_waveform_pulse(
        1e3,
        1.0,
        v_limit=1.0,
        width=400e-6,
        leading_edge_time=1.23456e-6,
        trailing_edge_time=5.5555e-9,
    )
    assert (state.edge_time, state.trailing_edge_time) == (1.234e-6, 5.555e-9)
    fake.log.clear()
    afg.ch1.set_waveform_pulse(
        1e3,
        1.0,
        v_limit=1.0,
        width=400e-6,
        leading_edge_time=1.23456e-6,
        trailing_edge_time=5.5555e-9,
    )
    assert _writes(fake) == []


# ---------------------------------------------------------------- noise add


def test_noise_add_on_left_alone_and_off(make_afg):
    afg, fake = make_afg()
    state = afg.ch2.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=12.3456)
    assert (state.noise_add, fake.ch[2].comb) == (12.0, "NOIS")
    assert _in_order(_writes(fake), "SOUR4:POW 12.3456", 'SOUR2:COMB:FEED "NOIS"')
    fake.log.clear()
    afg.ch2.set_waveform_sine(1e3, 1.0, v_limit=1.0)
    assert _writes(fake) == []
    state = afg.ch2.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=False)
    assert state.noise_add is None
    assert _writes(fake) == ['SOUR2:COMB:FEED ""']


def test_a_modulation_turned_on_clears_noise_add(make_afg):
    afg, fake = make_afg()
    afg.ch2.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=10)
    afg.ch2.set_fm(100.0)
    assert fake.ch[2].comb == ""
    state = afg.ch2.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=10)
    assert (state.noise_add, fake.ch[2].run) == (10.0, "FM")


def test_noise_add_keeps_add_in_on_channel_one(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_external_add_state(True)
    afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=5)
    assert fake.ch[1].comb == "BOTH"
    afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=False)
    assert fake.ch[1].comb == "EXT"
    assert afg.ch1.get_external_add_state() is True


def test_noise_add_half_a_percent_rounds_up_and_counts_as_written(make_afg):
    afg, _ = make_afg()
    state = afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=0.5)
    assert state.noise_add == 1.0  # 0.5 read back 1 (stage noiseadd)


@pytest.mark.parametrize("noise_add", [51, -1, True])
def test_noise_add_out_of_range_is_refused_before_any_write(make_afg, noise_add):
    afg, fake = make_afg()
    with pytest.raises(ValueError, match="noise_add"):
        afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=noise_add)
    assert _writes(fake) == []


# ---------------------------------------------------------------- phase and polarity


def test_phase_is_written_and_read_in_degrees(make_afg):
    afg, fake = make_afg()
    assert afg.ch1.set_phase(12.3456789) == pytest.approx(12.35, abs=1e-3)
    assert _writes(fake) == ["SOUR1:PHAS 12.3456789DEG"]
    assert fake.ch[1].phase == 12.35
    assert afg.ch1.get_phase() == pytest.approx(12.35, abs=1e-3)
    fake.log.clear()
    afg.ch1.set_phase(12.3456789)
    assert _writes(fake) == []
    assert afg.ch1.set_phase(-180.0) == pytest.approx(-180.0, abs=1e-3)


@pytest.mark.parametrize("func", ["PULS", "PRN", "DC"])
def test_phase_is_refused_on_a_shape_without_one(make_afg, func):
    afg, fake = make_afg()
    fake.ch[1].func = func
    with pytest.raises(ValueError, match="no phase"):
        afg.ch1.set_phase(10.0)
    assert _writes(fake) == []


def test_phase_outside_half_a_turn_is_refused(make_afg):
    afg, fake = make_afg()
    with pytest.raises(ValueError, match="-180 to \\+180"):
        afg.ch1.set_phase(181.0)
    assert _writes(fake) == []


def test_polarity_writes_only_a_change(make_afg):
    afg, fake = make_afg()
    assert afg.ch2.set_polarity(Polarity.INVERTED) is Polarity.INVERTED
    assert _writes(fake) == ["OUTP2:POL INV"]
    fake.log.clear()
    assert afg.ch2.set_polarity("inv") is Polarity.INVERTED
    assert _writes(fake) == []
    assert afg.ch2.set_polarity("NORMAL") is Polarity.NORMAL


# ---------------------------------------------------------------- modulation and sweep


def test_fm_writes_settings_before_the_enable_and_ends_a_sweep(make_afg):
    afg, fake = make_afg()
    afg.ch2.set_sweep(1e3, 2e3, 1.0)
    fake.log.clear()
    state = afg.ch2.set_fm(100.0, rate=10.0, shape=ModulationShape.SQUARE)
    writes = _writes(fake)
    assert writes[-1] == "SOUR2:FM:STAT ON"
    assert {"SOUR2:FM:INT:FUNC SQU", "SOUR2:FM:INT:FREQ 10", "SOUR2:FM 100"} <= set(writes)
    assert state == ModulationState(
        mode="FM",
        level=100.0,
        rate=10.0,
        shape=ModulationShape.SQUARE,
        source=ModulationSource.INTERNAL,
    )
    assert afg.ch2.get_sweep().enabled is False


def test_fm_from_the_external_input_reports_no_rate(make_afg):
    afg, _ = make_afg()
    state = afg.ch1.set_fm(500.0, source=ModulationSource.EXTERNAL)
    assert state.source is ModulationSource.EXTERNAL
    assert state.rate is None


def test_fm_arbitrary_shape_names_the_channels_own_memory(make_afg):
    afg, fake = make_afg()
    afg.ch2.set_fm(100.0, shape=ModulationShape.ARBITRARY)
    assert "SOUR2:FM:INT:FUNC EMEM2" in _writes(fake)
    assert fake.errors == []


def test_fm_refuses_a_dc_carrier(make_afg):
    afg, fake = make_afg()
    fake.ch[1].func = "DC"
    with pytest.raises(ValueError, match="Set the carrier shape first"):
        afg.ch1.set_fm(100.0)
    assert _writes(fake) == []


def test_am_depth_ends_the_running_fm(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_fm(100.0)
    state = afg.ch1.set_am(80.04, rate=1e3, shape=ModulationShape.TRIANGLE)
    assert state == ModulationState(
        mode="AM",
        level=80.0,
        rate=1e3,
        shape=ModulationShape.TRIANGLE,
        source=ModulationSource.INTERNAL,
    )
    assert _writes(fake)[-1] == "SOUR1:AM:STAT ON"
    assert afg.ch1.get_run_mode() is afg.RunMode.MODULATION
    assert fake.ch[1].run == "AM"
    fake.log.clear()
    afg.ch1.set_am(80.04, rate=1e3, shape=ModulationShape.TRIANGLE)
    assert _writes(fake) == []


def test_am_depth_past_the_range_is_refused_by_the_instrument(make_afg):
    afg, _ = make_afg()
    with pytest.raises(InstrumentError, match="-222"):
        afg.ch1.set_am(121.0)


def test_pm_speaks_degrees(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_pm(12.3456, rate=50.0)
    assert "SOUR1:PM 12.3456DEG" in _writes(fake)
    assert fake.ch[1].pm["level"] == 12.3
    assert (state.mode, state.rate) == ("PM", 50.0)
    assert state.level == pytest.approx(12.3, abs=1e-3)
    fake.log.clear()
    afg.ch1.set_pm(12.3456, rate=50.0)
    assert _writes(fake) == []
    with pytest.raises(InstrumentError, match="-222"):
        afg.ch1.set_pm(181.0)


def test_fsk_hops_to_a_frequency_and_has_no_shape(make_afg):
    afg, _ = make_afg()
    state = afg.ch2.set_fsk(1234.567891234, rate=10.0)
    assert state == ModulationState(
        mode="FSK", level=1234.568, rate=10.0, shape=None, source=ModulationSource.INTERNAL
    )
    assert afg.ch2.get_modulation() == state


def test_pwm_speaks_seconds_at_the_present_frequency(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=400e-6)
    state = afg.ch1.set_pwm(100e-6, rate=100.0)
    assert "SOUR1:PWM:DCYC 10" in _writes(fake)
    assert (state.mode, state.rate, state.shape) == ("PWM", 100.0, ModulationShape.SINE)
    assert state.level == pytest.approx(100e-6)
    assert afg.ch1.get_run_mode() is afg.RunMode.MODULATION


def test_pwm_deviation_wider_than_the_pulse_is_refused_by_the_instrument(make_afg):
    afg, _ = make_afg()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, duty=20.0)
    with pytest.raises(InstrumentError, match="-222"):
        afg.ch1.set_pwm(250e-6)  # 25 percent against a 20 percent pulse (stage mod2)


@pytest.mark.parametrize(
    ("func", "call"),
    [
        ("PULS", lambda ch: ch.set_am(50.0)),
        ("PULS", lambda ch: ch.set_fm(10.0)),
        ("PULS", lambda ch: ch.set_pm(10.0)),
        ("PULS", lambda ch: ch.set_fsk(1e3)),
        ("SIN", lambda ch: ch.set_pwm(1e-6)),
        ("PRN", lambda ch: ch.set_am(50.0)),
    ],
)
def test_modulation_on_the_wrong_carrier_is_refused_before_any_write(make_afg, func, call):
    afg, fake = make_afg()
    fake.ch[1].func = func
    with pytest.raises(ValueError, match="Set the carrier shape first"):
        call(afg.ch1)
    assert _writes(fake) == []


def test_sweep_writes_settings_then_turns_on(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_sweep(1e3, 2e3, 1.5, spacing=Spacing.LINEAR, hold_time=0.1)
    assert _writes(fake)[-1] == "SOUR1:FREQ:MODE SWE"
    assert state == SweepState(
        enabled=True,
        start=1e3,
        stop=2e3,
        time=1.5,
        spacing=Spacing.LINEAR,
        hold_time=0.1,
        return_time=0.001,
    )
    assert afg.ch1.get_run_mode() is afg.RunMode.SWEEP


def test_modulation_off_clears_fm_and_sweep(make_afg):
    afg, _ = make_afg()
    afg.ch1.set_fm(100.0)
    assert afg.ch1.set_modulation_off() == ModulationState(mode=None)
    afg.ch1.set_sweep(1e3, 2e3, 1.0)
    afg.ch1.set_modulation_off()
    assert afg.ch1.get_run_mode() is afg.RunMode.CONTINUOUS


# ---------------------------------------------------------------- burst


def test_burst_infinite_writes_the_idle_state_after_the_burst_is_on(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_burst(math.inf, idle=BurstIdle.DC, delay=12.3456789e-6)
    assert _in_order(
        _writes(fake), "SOUR1:BURS:NCYC INF", "SOUR1:BURS:STAT ON", "SOUR1:BURS:IDLE DC"
    )
    assert state == BurstState(
        enabled=True,
        cycles=9.9e37,
        period=1e-3,
        phase=0.0,
        mode=BurstMode.TRIGGERED,
        delay=pytest.approx(12.346e-6),
        idle=BurstIdle.DC,
    )
    assert fake.errors == []
    assert afg.ch1.get_run_mode() is afg.RunMode.BURST


def test_burst_count_mode_and_repeat(make_afg):
    afg, fake = make_afg()
    afg.ch2.set_sweep(1e3, 2e3, 1.0)
    state = afg.ch2.set_burst(3, mode=BurstMode.GATED)
    assert (state.enabled, state.cycles, state.mode) == (True, 3.0, BurstMode.GATED)
    assert afg.ch2.get_sweep().enabled is False  # a burst ends the sweep
    fake.log.clear()
    afg.ch2.set_burst(3, mode=BurstMode.GATED)
    assert _writes(fake) == []


@pytest.mark.parametrize("cycles", [0, 3.5, 1_000_001])
def test_burst_count_out_of_range_is_refused(make_afg, cycles):
    afg, fake = make_afg()
    with pytest.raises(ValueError, match="cycles"):
        afg.ch1.set_burst(cycles)
    assert _writes(fake) == []


@pytest.mark.parametrize("func", ["PRN", "DC"])
def test_burst_is_refused_on_noise_and_dc(make_afg, func):
    afg, fake = make_afg()
    fake.ch[1].func = func
    with pytest.raises(ValueError, match="Set the carrier shape first"):
        afg.ch1.set_burst(3)
    assert _writes(fake) == []


def test_burst_runs_on_a_pulse(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=10e-6)
    state = afg.ch1.set_burst(2)
    assert state.enabled and state.phase == 0.0
    assert fake.errors == []


def test_burst_start_phase_is_in_degrees(make_afg):
    afg, fake = make_afg()
    state = afg.ch1.set_burst(3, phase=90.0)
    assert "SOUR1:PHAS 90DEG" in _writes(fake)
    assert state.phase == pytest.approx(90.0, abs=1e-3)


def test_burst_rearm(make_afg):
    afg, fake = make_afg()
    afg.ch2.cmd_burst_rearm()
    assert _writes(fake) == ["SOUR2:BURS:INFI:REAR"]
    assert fake.rearms == [2] and fake.errors == []


# ---------------------------------------------------------------- arbitrary waveforms


def test_arb_uploads_big_endian_codes_into_the_channels_own_memory(make_afg):
    afg, fake = make_afg()
    state = afg.ch2.set_waveform_arb([0.0, 2.0, -2.0, 1.0], 60.0, 1.0, v_limit=1.0)
    assert fake.ch[2].emem == bytes.fromhex("1fff 3ffe 0000 2ffe")
    writes = _writes(fake)
    assert writes.index("DATA:DATA EMEM2,<8 bytes>") < writes.index("SOUR2:FUNC EMEM")
    assert (state.function, state.frequency) == ("EMEM", 60.0)


def test_arb_reupload_reselects_the_memory(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_arb([1.0, -1.0], 60.0, 1.0, v_limit=1.0)
    fake.log.clear()
    afg.ch1.set_waveform_arb([-1.0, 1.0], 60.0, 1.0, v_limit=1.0)
    assert _writes(fake) == ["DATA:DATA EMEM1,<4 bytes>", "SOUR1:FUNC EMEM"]


@pytest.mark.parametrize("samples", [[1.0], [0.0, 0.0, 0.0], [0.0, float("nan")], [0.0] * 131_073])
def test_arb_rejects_bad_samples(make_afg, samples):
    afg, fake = make_afg()
    with pytest.raises(ValueError):
        afg.ch1.set_waveform_arb(samples, 60.0, 1.0, v_limit=1.0)
    assert _writes(fake) == []


def test_arb_block_goes_over_its_own_vxi11_session_never_the_socket(make_afg):
    afg, fake = make_afg()
    # Code 0x0A0A puts newline bytes in the block, which the socket would cut.
    samples = [1.0, (0x0A0A - 8191) / 8191]
    afg.ch1.set_waveform_arb(samples, 1e3, 1.0, v_limit=1.0)
    assert [name for name, _, _ in fake.blocks] == ["TCPIP0::x::inst0::INSTR"]
    assert fake.ch[1].emem == bytes.fromhex("3ffe 0a0a")
    side = fake.sessions[-1]
    assert side.closed and not fake.closed
    assert fake.discarded == [] and fake.errors == []
    afg.ch1.set_waveform_arb([0.0, 1.0], 1e3, 1.0, v_limit=1.0)  # a second side session
    assert len(fake.sessions) == 2 and fake.ch[1].emem == bytes.fromhex("1fff 3ffe")


def test_upload_waits_out_inv_prot_without_a_device_clear(make_afg, monkeypatch):
    afg, fake = make_afg()
    sleeps: list[float] = []
    monkeypatch.setattr(afg_module.time, "sleep", sleeps.append)
    afg.ch2.set_waveform_arb([0.0, 1.0, -1.0], 1e3, 1.0, v_limit=1.0)
    side = fake.sessions[-1]
    assert side.busy_reads == 1
    assert sleeps == [afg_module._UPLOAD_RETRY_S]
    assert side.cleared == 0 and fake.cleared == 0
    assert fake.discarded == []


def test_upload_over_vxi11_goes_straight_and_still_skips_the_device_clear(make_afg, monkeypatch):
    afg, fake = make_afg(resource="TCPIP0::x::inst0::INSTR")
    monkeypatch.setattr(afg_module.time, "sleep", lambda s: None)
    afg.ch1.set_waveform_arb([0.0, 1.0], 1e3, 1.0, v_limit=1.0)
    assert fake.blocks == [("TCPIP0::x::inst0::INSTR", 1, 4)]
    assert fake.sessions == [] and fake.busy_reads == 1
    assert fake.cleared == 1  # the one at connect
    assert fake.ch[1].emem == bytes.fromhex("1fff 3ffe")


def test_save_arb_file_copies_the_channels_own_memory(make_afg):
    afg, fake = make_afg()
    fake.ch[2].emem = bytes(2000)
    assert afg.ch2.cmd_save_arb_file("M:/w.tfwx") == "M:/w.tfwx"
    assert _writes(fake) == ['MMEM:STOR:TRAC EMEM2,"M:/w.tfwx"']
    assert fake.files["M:/w.tfwx"] == (2512, bytes(2000))  # 512 + 2 x 1000 points
    with pytest.raises(InstrumentError, match="-256"):
        afg.ch2.cmd_save_arb_file("M:/no_such_folder/w.tfwx")


def test_upload_refused_by_the_instrument_raises(make_afg):
    afg, fake = make_afg()
    fake.vxi_errors.append('-222,"Data out of range"')
    with pytest.raises(InstrumentError, match="-222"):
        afg.ch1.set_waveform_arb([0.0, 1.0], 1e3, 1.0, v_limit=1.0)
    assert fake.sessions[-1].closed


# ---------------------------------------------------------------- interchannel


def test_frequency_coupling_is_one_state_both_channels_follow(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_frequency(2e3)
    assert afg.interchannel.set_frequency_coupling_state(True) is True
    assert fake.ch[2].freq == 2e3
    afg.ch2.set_frequency(5e3)
    assert afg.ch1.get_frequency() == 5e3
    assert afg.interchannel.get_frequency_coupling_state() is True
    fake.log.clear()
    afg.interchannel.set_frequency_coupling_state(True)
    assert _writes(fake) == []
    assert afg.interchannel.set_frequency_coupling_state(False) is False


def test_level_coupling_refuses_mixed_panel_units(make_afg):
    afg, fake = make_afg()
    fake.ch[2].unit = "VRMS"
    with pytest.raises(ValueError, match="different units"):
        afg.interchannel.set_level_coupling_state(True)
    assert _writes(fake) == []


def test_level_coupling_copies_and_refuses_outside_either_limit(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_sine(1e3, 0.8, v_limit=1.0, offset=0.1)
    afg.ch2.set_voltage_limits(-0.5, 0.5)
    assert afg.interchannel.set_level_coupling_state(True) is True
    assert (fake.ch[2].vpp, fake.ch[2].offset) == (0.8, 0.1)
    with pytest.raises(InstrumentError, match="-222"):
        afg.ch1.set_amplitude(1.0)  # inside CH1's limit, outside CH2's
    assert afg.interchannel.get_level_coupling_state() is True


def test_phase_coupling_and_phase_sync(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_phase(30.0)
    assert afg.interchannel.set_phase_coupling_state(True) is True
    afg.ch2.set_phase(-90.0)
    assert afg.ch1.get_phase() == pytest.approx(-90.0, abs=1e-3)
    assert afg.interchannel.get_phase_coupling_state() is True
    fake.log.clear()
    afg.interchannel.cmd_phase_sync()
    assert _writes(fake) == ["SOUR1:PHAS:INIT"] and fake.phase_syncs == 1


def test_copy_carries_the_setup_and_turns_both_outputs_off(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_square(2e3, 1.0, v_limit=1.0)
    afg.set_output_state_all(True)
    fake.log.clear()
    afg.interchannel.cmd_copy_ch1_to_ch2()
    assert _writes(fake) == ["AFGC:CSC CH1,CH2"]
    assert (fake.ch[2].func, fake.ch[2].freq, fake.ch[2].lim_high) == ("SQU", 2e3, 1.0)
    assert afg.get_output_state_all() == [False, False]
    afg.set_output_state_all(True)
    afg.interchannel.cmd_copy_ch2_to_ch1()
    assert _writes(fake)[-1] == "AFGC:CSC CH2,CH1"
    assert afg.get_output_state_all() == [False, False]


def test_burst_phase_on_a_pulse_is_refused_before_any_write(make_afg):
    afg, fake = make_afg()
    afg.ch2.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=100e-6)
    fake.log.clear()
    with pytest.raises(ValueError, match="no phase"):
        afg.ch2.set_burst(3, phase=10.0)
    assert _writes(fake) == []


@pytest.mark.parametrize(
    "call",
    [
        lambda afg: afg.set_trigger_slope(afg.Slope.POSITIVE),
        lambda afg: afg.set_trigger_out_mode(afg.TriggerOutMode.TRIGGER),
        lambda afg: afg.set_front_panel_lock_state(False),
        lambda afg: afg.ch1.set_amplitude_unit(afg.AmplitudeUnit.VPP),
        lambda afg: afg.ch1.set_load(50),
    ],
)
def test_a_setting_already_held_is_not_written_again(make_afg, call):
    afg, fake = make_afg()
    call(afg)
    fake.log.clear()
    call(afg)
    assert _writes(fake) == []


def test_restore_puts_back_the_phase_of_a_phaseless_shape_and_no_empty_file(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0)
    afg.ch1.set_phase(30.0)
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=100e-6)
    snap = afg.snapshot()
    afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0)
    afg.ch1.set_phase(-45.0)
    afg.ch1.set_waveform_arb_file("M:/w.tfwx", 1e3, 1.0, v_limit=1.0)
    afg.restore(snap)
    assert afg.ch1.get_waveform().function == "PULS"
    assert fake.ch[1].phase == pytest.approx(30.0, abs=0.01)  # the fake holds degrees
    assert not any(w.endswith('EFIL ""') for w in fake.log)


# ---------------------------------------------------------------- setup memories and files


def test_setup_memory_save_recall_and_delete(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_square(2e3, 1.0, v_limit=1.0)
    afg.ch1.set_output_state(True)
    afg.memory.cmd_save_state(4)
    assert afg.memory.get_state_valid(4) is True
    afg.ch1.set_waveform_sine(5e3, 0.5, v_limit=1.0)
    afg.memory.cmd_recall_state(4)
    assert (fake.ch[1].func, fake.ch[1].freq) == ("SQU", 2e3)
    assert fake.ch[1].output is False  # a recall never turns an output on
    afg.memory.cmd_delete_state(4)
    assert afg.memory.get_state_valid(4) is False
    with pytest.raises(InstrumentError, match="-256"):
        afg.memory.cmd_recall_state(4)


def test_locked_setup_memory_refuses_a_save(make_afg):
    afg, _ = make_afg()
    assert afg.memory.set_lock_state(4, True) is True
    with pytest.raises(InstrumentError, match="-258"):
        afg.memory.cmd_save_state(4)
    assert afg.memory.set_lock_state(4, False) is False
    afg.memory.cmd_save_state(4)
    with pytest.raises(ValueError, match="1 to 4"):
        afg.memory.set_lock_state(0, True)
    with pytest.raises(ValueError, match="0 to 4"):
        afg.memory.cmd_save_state(5)


def test_auto_recall_state(make_afg):
    afg, fake = make_afg()
    assert afg.memory.set_auto_recall_state(False) is False
    assert fake.auto_recall is False
    assert afg.memory.get_auto_recall_state() is False


def test_file_node_folders_setups_and_the_catalog(make_afg):
    afg, fake = make_afg()
    afg.file.cmd_make_directory("M:/tests")
    assert FileEntry("tests", True, 4096) in afg.file.get_catalog("M:")
    assert afg.file.cmd_save_setup("M:/tests/a.tfs") == "M:/tests/a.tfs"
    assert afg.file.get_catalog("M:/tests") == [FileEntry("a.tfs", False, 4096)]
    afg.ch1.set_waveform_square(3e3, 1.0, v_limit=1.0)
    afg.file.cmd_recall_setup("M:/tests/a.tfs")
    assert (fake.ch[1].func, fake.ch[1].freq) == ("SIN", 1e6)
    afg.file.cmd_delete("M:/tests/a.tfs")
    assert afg.file.get_catalog("M:/tests") == []
    with pytest.raises(InstrumentError, match="-256"):
        afg.file.cmd_delete("M:/tests/a.tfs")
    afg.file.cmd_delete("M:/tests")
    assert "M:/tests" not in fake.dirs
    assert "M:/.setup0.tfs" in fake.files


def test_catalog_leaves_the_working_folder_as_found(make_afg):
    afg, fake = make_afg()
    fake.cwd = "M:/lost+found"
    entries = afg.file.get_catalog("M:")
    assert FileEntry(".setup0.tfs", False, 2048) in entries
    assert fake.cwd == "M:/lost+found"


def test_recalling_a_missing_setup_file_raises(make_afg):
    afg, _ = make_afg()
    with pytest.raises(InstrumentError, match="-256"):
        afg.file.cmd_recall_setup("M:/nothing.tfs")


# ---------------------------------------------------------------- trigger and system


def test_trigger_timer_is_held_to_four_digits(make_afg):
    afg, fake = make_afg()
    assert afg.set_trigger_timer(12.3456789e-3) == pytest.approx(12.35e-3)
    fake.log.clear()
    afg.set_trigger_timer(12.3456789e-3)
    assert _writes(fake) == []
    for seconds in (0.5e-6, 501.0):
        with pytest.raises(ValueError, match="1 us to 500 s"):
            afg.set_trigger_timer(seconds)
    assert _writes(fake) == []


def test_trigger_slope_and_trigger_out_mode(make_afg):
    afg, fake = make_afg()
    assert afg.set_trigger_slope(afg.Slope.NEGATIVE) is afg.Slope.NEGATIVE
    assert afg.set_trigger_out_mode("sync") is afg.TriggerOutMode.SYNC
    assert (fake.slope, fake.trigger_out_mode) == ("NEG", "SYNC")
    afg.cmd_trigger()
    assert fake.triggers == 1


def test_front_panel_lock_beep_and_reset(make_afg):
    afg, fake = make_afg()
    assert afg.set_front_panel_lock_state(True) is True
    assert afg.get_front_panel_lock_state() is True
    afg.cmd_beep()
    assert fake.beeps == 1
    fake.log.clear()
    afg.cmd_reset()
    assert _writes(fake) == ["*RST"]
    assert fake.queries[-1] == "*OPC?"


def test_self_test_waits_with_a_long_timeout(make_afg):
    afg, fake = make_afg()
    timeout = fake.timeout
    assert afg.self_test() == 0
    assert fake.self_test_timeout == afg_module._SELF_TEST_TIMEOUT_MS
    assert fake.timeout == timeout


# ---------------------------------------------------------------- add in, load, reference


def test_external_add_on_channel_one(make_afg):
    afg, fake = make_afg()
    assert afg.ch1.set_external_add_state(True) is True
    assert fake.ch[1].comb == "EXT"
    assert afg.ch1.set_external_add_state(False) is False


def test_load_reads_back_high_impedance_as_infinity(make_afg):
    afg, _ = make_afg()
    assert afg.ch1.set_load(afg.Load.HIGHZ) == math.inf
    assert afg.ch1.set_load(1000) == 1000.0


def test_load_change_rescales_the_limit(make_afg):
    afg, _ = make_afg()
    afg.ch1.set_voltage_limits(-2.5, 2.5)
    afg.ch1.set_load(afg.Load.HIGHZ)
    assert afg.ch1.get_voltage_limits() == VoltageLimits(-5.0, 5.0, True)


def test_reference_reports_the_questionable_frequency_bit(make_afg):
    afg, fake = make_afg()
    assert afg.set_reference_clock_internal().selected is afg.Reference.INTERNAL
    fake.ques = 1 << 5
    state = afg.set_reference_clock_external()
    assert state.selected is afg.Reference.EXTERNAL
    assert state.frequency_questionable is True


# ---------------------------------------------------------------- lifecycle


def test_disconnect_turns_both_outputs_off(make_afg):
    afg, fake = make_afg()
    afg.set_output_state_all(True)
    afg.disconnect()
    assert fake.log[-2:] == ["OUTP1 OFF", "OUTP2 OFF"]
    assert not fake.ch[1].output and not fake.ch[2].output


def test_vxi11_busy_window_is_waited_out(make_afg, monkeypatch):
    afg, fake = make_afg(resource="TCPIP0::x::inst0::INSTR")
    sleeps: list[float] = []
    monkeypatch.setattr(afg_module, "_BUSY_SETTLE_S", 0.15)
    monkeypatch.setattr(afg_module.time, "sleep", sleeps.append)
    fake.ch[1].func = "SQU"
    busy = visa.VisaIOError(visa.constants.StatusCode.error_invalid_protocol)
    original_write = fake.write

    def write_then_busy(message: str, **kwargs) -> int:
        if message == "SOUR1:FUNC SIN":
            fake.fail_next_query.append(busy)
        return original_write(message, **kwargs)

    fake.write = write_then_busy
    clears_before = fake.cleared
    afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0)
    assert sleeps == [0.15, 0.15]
    assert fake.cleared == clears_before + 1


def test_socket_waits_on_opc_without_sleeping(make_afg, monkeypatch):
    afg, fake = make_afg()
    sleeps: list[float] = []
    monkeypatch.setattr(afg_module.time, "sleep", sleeps.append)
    fake.ch[1].func = "SQU"
    afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0)
    assert sleeps == []


# ---------------------------------------------------------------- snapshot and restore


def test_restore_puts_back_what_snapshot_found(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_amplitude_unit(afg.AmplitudeUnit.VRMS)
    afg.ch1.set_fm(1181.0, rate=100e3, shape=ModulationShape.SQUARE)
    afg.ch1.set_modulation_off()
    afg.ch1.set_output_state(True)
    before = _fake_state(fake)
    with afg.preserved_setup():
        afg.ch1.set_output_state(False)
        afg.ch1.set_waveform_arb([1.0, -1.0, 0.5], 60.0, 0.5, v_limit=(-0.5, 0.5))
        afg.ch2.set_sweep(1e3, 2e3, 1.0)
        afg.ch2.set_fm(10.0, source=ModulationSource.EXTERNAL)
    _assert_same(before, _fake_state(fake))


def test_restore_puts_back_every_setting_the_driver_can_change(make_afg):
    afg, fake = make_afg()
    ch1, ch2 = afg.ch1, afg.ch2
    ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=400e-6, edge_time=20e-9, delay=1e-6)
    ch1.set_pwm(50e-6, rate=100.0)
    ch1.set_polarity(Polarity.INVERTED)
    ch1.set_external_add_state(True)
    ch1.set_output_state(True)
    ch2.set_waveform_ramp(5e3, 0.5, v_limit=(-0.5, 1.0), offset=0.1, symmetry=25.0, noise_add=7)
    ch2.set_phase(-45.0)
    ch2.set_pm(30.0, rate=50.0)
    ch2.set_am(70.0, shape=ModulationShape.TRIANGLE)
    ch2.set_fsk(2e3, rate=10.0)
    ch2.set_burst(math.inf, idle=BurstIdle.END, delay=2e-6)
    afg.set_trigger_timer(5e-3)
    afg.set_trigger_slope(afg.Slope.NEGATIVE)
    afg.set_trigger_out_mode(afg.TriggerOutMode.SYNC)
    afg.set_front_panel_lock_state(True)
    before = _fake_state(fake)
    with afg.preserved_setup():
        ch1.set_output_state(False)
        ch1.set_modulation_off()
        ch1.set_polarity(Polarity.NORMAL)
        ch1.set_external_add_state(False)
        ch1.set_waveform_arb([1.0, -1.0, 0.5], 60.0, 0.4, v_limit=0.5, noise_add=20)
        ch2.set_modulation_off()
        ch2.set_waveform_square(1e3, 0.2, v_limit=1.0, noise_add=False)
        ch2.set_pm(170.0)
        ch2.set_am(10.0, rate=1.0)
        ch2.set_waveform_pulse(10.0, 0.2, v_limit=1.0, width=50e-3, edge_time=1e-6)
        afg.set_trigger_timer(1.0)
        afg.set_trigger_slope(afg.Slope.POSITIVE)
        afg.set_trigger_out_mode(afg.TriggerOutMode.TRIGGER)
        afg.set_trigger_source(afg.TriggerSource.TIMER)
        afg.set_front_panel_lock_state(False)
    _assert_same(before, _fake_state(fake))
    assert fake.errors == []


def test_restore_puts_back_noise_add_over_a_running_modulation(make_afg):
    afg, fake = make_afg()
    afg.ch2.set_waveform_sine(1e3, 1.0, v_limit=1.0)
    afg.ch2.set_am(50.0, rate=10.0)
    afg.ch2.set_waveform_sine(1e3, 1.0, v_limit=1.0, noise_add=10)
    assert (fake.ch[2].run, fake.ch[2].comb) == ("AM", "NOIS")
    before = _fake_state(fake)
    with afg.preserved_setup():
        afg.ch2.set_modulation_off()
        afg.ch2.set_waveform_square(2e3, 0.5, v_limit=1.0, noise_add=False)
    _assert_same(before, _fake_state(fake))
    assert fake.errors == []


def test_restore_puts_back_the_couplings(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_sine(2e3, 0.5, v_limit=1.0)
    afg.ch2.set_waveform_sine(2e3, 0.5, v_limit=1.0)
    afg.interchannel.set_frequency_coupling_state(True)
    afg.interchannel.set_level_coupling_state(True)
    before = _fake_state(fake)
    with afg.preserved_setup():
        afg.interchannel.set_frequency_coupling_state(False)
        afg.interchannel.set_level_coupling_state(False)
        afg.ch2.set_waveform_sine(3e3, 0.25, v_limit=1.0)
    _assert_same(before, _fake_state(fake))


def test_restore_puts_back_levels_the_present_offset_would_block(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_sine(1e3, 1.8, v_limit=1.0)
    snap = afg.snapshot()
    afg.ch1.set_waveform_sine(1e3, 0.2, v_limit=1.0, offset=0.85)
    afg.restore(snap)
    assert (fake.ch[1].vpp, fake.ch[1].offset) == (1.8, 0.0)
    assert fake.errors == []


def test_restore_puts_back_a_pulse_period_shorter_than_the_present_width(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=400e-6)
    snap = afg.snapshot()
    afg.ch1.set_waveform_pulse(10.0, 1.0, v_limit=1.0, width=50e-3)
    afg.restore(snap)
    assert (fake.ch[1].pulse["per"], fake.ch[1].pulse["width"]) == (1e-3, 400e-6)
    assert fake.errors == []


def test_restore_puts_back_a_pulse_below_a_leftover_fm_deviation(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_waveform_sine(10e3, 1.0, v_limit=1.0)
    afg.ch1.set_fm(5e3)
    afg.ch1.set_modulation_off()
    afg.ch1.set_waveform_pulse(1e3, 1.0, v_limit=1.0, width=400e-6)
    snap = afg.snapshot()
    afg.ch1.set_waveform_sine(2e3, 1.0, v_limit=1.0)
    afg.restore(snap)
    assert (fake.ch[1].func, fake.ch[1].freq, fake.ch[1].pulse["width"]) == ("PULS", 1e3, 400e-6)
    assert fake.ch[1].fm["dev"] == 5e3
    assert fake.errors == []


def test_snapshot_keeps_each_edit_memory_as_a_file_and_never_reads_it(make_afg):
    afg, fake = make_afg()
    fake.ch[2].emem = bytes(2000)
    snap = afg.snapshot()
    assert 'MMEM:MDIR "M:/pyvisa_tmp"' in _writes(fake)
    assert set(snap["files"]) == {1, 2}
    for n, path in snap["files"].items():
        assert path.startswith("M:/pyvisa_tmp/snapshot_") and path.endswith(f"_ch{n}.tfwx")
        assert fake.files[path] == (512 + len(fake.ch[n].emem), fake.ch[n].emem)
    assert not any(q.startswith("DATA?") for q in fake.queries)
    fake.ch[2].emem = bytes.fromhex("0001")
    afg.restore(snap)
    assert fake.ch[2].emem == bytes(2000)
    assert all(path in fake.files for path in snap["files"].values())  # a bare snapshot keeps them


def test_snapshot_twice_does_not_trip_over_its_folder(make_afg):
    afg, fake = make_afg()
    afg.snapshot()
    afg.snapshot()
    assert fake.errors == []


def test_preserved_setup_deletes_its_snapshot_files(make_afg):
    afg, fake = make_afg()
    with afg.preserved_setup():
        assert sum("snapshot_" in path for path in fake.files) == 2
    assert not any("snapshot_" in path for path in fake.files)
    assert "M:/.setup0.tfs" in fake.files
    assert fake.errors == []


def test_preserved_setup_deletes_its_files_when_restore_fails(make_afg):
    afg, fake = make_afg()

    def spoil_ch2_offset() -> None:
        _spoil(fake, "SOUR2:VOLT:OFFS", lambda: setattr(fake.ch[2], "offset", 0.1), once=False)

    with pytest.raises(InstrumentError, match="could not put back"), afg.preserved_setup():
        spoil_ch2_offset()
    assert not any("snapshot_" in path for path in fake.files)


# ---------------------------------------------------------------- stale load setting (ADR-054)


def _spoil(fake: FakeAfg, prefix: str, action, *, once: bool) -> list[str]:
    """Run ``action`` after the fake executes a write starting with ``prefix``."""
    original = fake._execute
    hits: list[str] = []

    def spoiled(message: str) -> None:
        original(message)
        if message.startswith(prefix) and not (once and hits):
            hits.append(message)
            action()

    fake._execute = spoiled
    return hits


def test_a_load_change_during_a_helper_raises(make_afg):
    afg, fake = make_afg()
    afg.ch1.set_load(afg.Load.HIGHZ)
    _spoil(fake, "SOUR1:VOLT:OFFS", lambda: setattr(fake.ch[1], "load", 50.0), once=True)
    with pytest.raises(InstrumentError, match="load setting read inf ohm before"):
        afg.ch1.set_waveform_sine(1e3, 1.0, v_limit=1.0, offset=0.1)


def test_restore_applies_again_when_the_first_pass_does_not_stick(make_afg):
    afg, fake = make_afg()
    snap = afg.snapshot()
    afg.ch2.set_waveform_sine(2e3, 0.5, v_limit=1.0)
    hits = _spoil(fake, "SOUR2:VOLT:OFFS", lambda: setattr(fake.ch[2], "offset", 0.123), once=True)
    afg.restore(snap)
    assert hits
    assert fake.ch[2].offset == 0.0


def test_restore_raises_when_a_setting_will_not_stick(make_afg):
    afg, fake = make_afg()
    snap = afg.snapshot()
    _spoil(fake, "SOUR2:VOLT:OFFS", lambda: setattr(fake.ch[2], "offset", 0.123), once=False)
    with pytest.raises(InstrumentError, match="could not put back.*CH2 offset"):
        afg.restore(snap)


def test_trigger_source_is_set_and_restored(make_afg):
    afg, fake = make_afg()
    snap = afg.snapshot()
    assert afg.set_trigger_source(afg.TriggerSource.TIMER) is afg.TriggerSource.TIMER
    afg.restore(snap)
    assert fake.trigger_source == "EXT"


def test_restore_puts_back_a_deviation_larger_than_the_carrier(make_afg):
    afg, fake = make_afg()
    fake.ch[1].freq, fake.ch[1].fm["dev"] = 1e3, 1181.0  # left by a higher carrier
    snap = afg.snapshot()
    afg.ch1.set_frequency(2e3)
    afg.restore(snap)
    assert (fake.ch[1].freq, fake.ch[1].fm["dev"]) == (1e3, 1181.0)
    assert fake.errors == []
