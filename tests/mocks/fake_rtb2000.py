# SPDX-License-Identifier: Apache-2.0
# Copyright: see the NOTICE file.
"""A simulated RTB2004 for the hardware-free RTB2000 tests.

Ported from the original repository (``tests/fake_rtb.py``). It echoes
settings, synthesizes a 10 kHz sine trace and measurement results, and models
an interrupted wait: when the trigger does not fire, the completion query
times out and leaves -410 in the error queue.

This fake is lenient, as the original one was: a query it does not know answers
the last value written to that header, or "0". It does not prove that the
scope accepts a command. A test against it proves the driver's command
sequence and its parsing, not the instrument's behaviour.

Changes from the original fake, for the port:

- it opens through ``open_resource(rm, name)`` and has the pyvisa terminator
  attributes, so the driver's own link opens it;
- the completion timeout raises ``pyvisa.errors.VisaIOError`` as pyvisa does;
- the generator and pattern outputs keep their written state;
- logic channels answer a data block, and the logic pod reports a probe;
- ``refuse`` maps a header to an error the next write of it queues;
- the identity and the option list are constructor arguments.
"""

from __future__ import annotations

import numpy as np
import pyvisa as visa

IDN = "Rohde&Schwarz,RTB2004,1333.1005k04/100000,03.000"
IDN_2CH = "Rohde&Schwarz,RTB2002,1333.1005k02/100001,03.000"
OPTIONS = "K1,K2,K3,K15,B1,B6,B243,K36"

DEFAULTS = {
    "ACQ:STAT": "RUN",
    "TRIG:A:SOUR": "CH1",
    "TRIG:A:MODE": "AUTO",
    "TRIG:A:TYPE": "EDGE",
    "TRIG:A:EDGE:SLOP": "POS",
    "TRIG:A:EDGE:COUP": "DC",
    "TRIG:A:HOLD:MODE": "OFF",
    "TRIG:A:HOLD:TIME": "5.12E-08",
    "ACQ:TYPE": "REFR",
    "ACQ:PEAK": "OFF",
    "ACQ:HRES": "AUTO",
    "ACQ:INT": "SINX",
    "ACQ:POIN": "1000000",
    "ACQ:MEM": "DMEM",
    "ACQ:SRAT": "1.04E+07",
    "ACQ:POIN:ARAT": "6.25E+07",
    "ACQ:AVER:COUN": "2",
    "TIM:SCAL": "1E-04",
    "TIM:POS": "0",
    "TIM:REF": "5.00E+01",
    "TIM:ROLL:AUT": "1",
    "TIM:RANG": "1.2E-03",
    "TIM:RAT": "1.6E-03",
    "ACQ:SEGM:STAT": "OFF",
    "TRIG:OUT:MODE": "REF",
    "DISP:PERS:TYPE": "OFF",
    "DISP:GRID:STYL": "LIN",
    "DISP:STYL": "VECT",
    "DISP:INT:WAV": "85",
    "DISP:INT:GRID": "30",
    "DISP:DTIM": "1",
    "TIM:ZOOM:STAT": "OFF",
    "TIM:ZOOM:SCAL": "5E-05",
    "TIM:ZOOM:TIME": "0",
    "TIM:ZOOM:POS": "50",
    "TIM:ROLL:MTIM": "0.05",
    "CHAN1:THR": "0.26",
    "CHAN1:THR:HYST": "SMAL",
    "ACQ:COUN": "100",
    "ACQ:AVA": "120",
    "CHAN1:HIST:TSR:ALL": "-2E-03,-1E-03,-0.0",
    "WGEN:OUTP": "OFF",
    "WGEN:OUTP:LOAD": "HIGHZ",
    "WGEN:FUNC": "SIN",
    "WGEN:FREQ": "1000",
    "WGEN:VOLT": "0.5",
    "WGEN:VOLT:OFFS": "0",
    "WGEN:FUNC:PULS:DCYC": "50",
    "WGEN:BURS": "OFF",
    "WGEN:BURS:PHAS": "0",
    "PGEN:PATT:STAT": "OFF",
    "PGEN:FUNC": "SQU",
    "LOG1:PROB": "1",
    "LOG2:PROB": "0",
}
for _n in range(1, 5):
    DEFAULTS |= {
        f"CHAN{_n}:STAT": "1" if _n == 1 else "0",
        f"CHAN{_n}:SCAL": "0.2",
        f"CHAN{_n}:OFFS": "0",
        f"CHAN{_n}:POS": "0",
        f"CHAN{_n}:COUP": "DCL",
        f"CHAN{_n}:BAND": "FULL",
        f"CHAN{_n}:POL": "NORM",
        f"CHAN{_n}:TYPE": "HRES",
        f"PROB{_n}:SET:ATT:MAN": "1",
        f"PROB{_n}:SET:ATT:UNIT": "V",
        f"TRIG:A:LEV{_n}": "0",
    }
DEFAULTS["TRIG:A:LEV5"] = "0"

MEAS_VALUES = {
    "FREQ": 1e4,
    "PER": 1e-4,
    "PEAK": 1.0,
    "AMPL": 1.0,
    "MEAN": 0.0,
    "RMS": 0.3536,
    "RTIM": 3e-9,
    "PHAS": 90.0,
    "DEL": 2.5e-5,
}

# Headers whose OFF is one word of a mode list, not a boolean, so the fake
# answers OFF there rather than 0. The original fake answered 0 for every OFF.
_MODE_HEADERS = {"TRIG:OUT:MODE", "TRIG:A:HOLD:MODE", "DISP:PERS:TYPE", "ACQ:PEAK", "ACQ:HRES"}

_PASS_THROUGH = ("RUN", "STOP", "AUT", "*RST", "FORM:DATA", "FORM:BORD", "HCOP:LANG")


class FakeRTB:
    """One open pyvisa resource on a simulated RTB2004."""

    def __init__(self, idn: str = IDN, options: str = OPTIONS):
        self.idn = idn
        self.timeout = 3000
        self.write_termination: str | None = None
        self.read_termination: str | None = None
        self.s = dict(DEFAULTS)
        self.s["*OPT"] = options
        self.errors: list[str] = []
        self.log: list[str] = []
        self.refuse: dict[str, str] = {}
        self.trigger_fires = True
        self.pending = False
        self.files: dict[str, bytes] = {}
        self.pending_read: bytes | None = None
        self.clip = False
        self.cleared = 0
        self.closed = False
        self._rb = b""

    # ---- pyvisa resource surface ----

    def open_resource(self, rm, name: str) -> FakeRTB:
        return self

    def close(self) -> None:
        self.closed = True

    def clear(self) -> None:
        self.cleared += 1
        self.log.append("<clear>")

    def write(self, cmd: str, **kwargs) -> int:
        self.log.append(cmd)
        head, _, val = cmd.partition(" ")
        if head in self.refuse:
            self.errors.append(self.refuse[head])
            return len(cmd)
        if head == "MMEM:STOR:STAT":
            self.files[val.split(",")[1].strip('"')] = b"setup" * 100
        elif head == "EXP:WAV:SAVE":
            stem = self.s["EXP:WAV:NAME"].strip('"').rsplit("/", 1)[1]
            self.files[stem + ".CSV"] = b"t,ch1\n0,0.5\n"
        elif head == "MMEM:DEL":
            self.files.pop(val.strip('"'), None)
        elif head == "MMEM:DATA?":
            self.pending_read = self.files[val.strip('"')]
        elif head in ("MASK:CHC", "MASK:RES:COUN", "DISP:CLE", "WGEN:BURS:TRIG:SING"):
            pass
        elif head.startswith("REFC") and head.endswith(":UPD"):
            pass
        elif head == "SING":
            self.s["ACQ:STAT"] = "COMP" if self.trigger_fires else "RUN"
            self.pending = not self.trigger_fires
        elif head in _PASS_THROUGH:
            if head == "*RST":
                options = self.s["*OPT"]
                self.s = dict(DEFAULTS)
                self.s["*OPT"] = options
            if head in ("RUN", "STOP"):
                self.s["ACQ:STAT"] = "RUN" if head == "RUN" else "STOP"
        elif head == "TRIG:A:FIND":
            self.s[f"TRIG:A:LEV{self.s['TRIG:A:SOUR'][-1]}"] = "0.01"
        elif head.startswith("MEAS") and head[4:].isdigit():
            self.s[head + ":ENAB"] = val
        elif head.startswith("MEAS") and head.endswith(":STAT:RES"):
            pass
        else:
            if head == "TIM:SCAL":
                self.s["TIM:RANG"] = repr(float(val) * 12)
            if head == "ACQ:TYPE":
                val = {"AVER": "AVER", "ENV": "ENV"}.get(val, "REFR")
            self.s[head] = val
        return len(cmd)

    def query(self, cmd: str, **kwargs) -> str:
        self.log.append(cmd)
        if cmd == "*IDN?":
            return self.idn + "\n"
        if cmd == "*OPC?":
            if self.pending:
                self.pending = False
                self.errors.append('-410,"Query INTERRUPTED"')
                raise visa.VisaIOError(visa.constants.StatusCode.error_timeout)
            return "1\n"
        if cmd == "SYST:ERR?":
            return (self.errors.pop(0) if self.errors else '0,"No error"') + "\n"
        head = cmd[:-1]
        if head.startswith("MEAS") and head.endswith(":RES"):
            main = self.s.get(head[:-4] + ":MAIN", "NONE")
            return f"{MEAS_VALUES.get(main, 9.91e37):.6E}\n"
        if head.startswith("MEAS") and ":RES:" in head:
            n = head.split(":")[0]
            v = MEAS_VALUES.get(self.s.get(n + ":MAIN", "NONE"), 0.0)
            stats = {
                "AVG": f"{v}",
                "STDD": "1E-06",
                "NPE": f"{v * 0.999}",
                "PPE": f"{v * 1.001}",
                "WFMC": "150",
            }
            return stats[head.split(":")[-1]] + "\n"
        if head.startswith("DVM") and head.endswith(":RES:STAT"):
            kind = self.s.get(head[:4] + ":TYPE", "DC")
            return {"DC": "0.25,1", "ACDC": "0.434,1", "ACRM": "0.354,1"}.get(kind, "0,0") + "\n"
        if head == "SEAR:RCO":
            return "3\n"
        if head == "SEAR:RES:ALL":
            return (
                "1,-1E-04,0,WIDTH,POSITIVE,2E-05,2,0,0,WIDTH,POSITIVE,2E-05,"
                "3,1E-04,0,WIDTH,POSITIVE,2E-05\n"
            )
        if head in ("MASK:COUN", "MASK:VCO"):
            return ("26" if head == "MASK:COUN" else "0") + "\n"
        if head == "MMEM:DRIV":
            return '"/INT"\n'
        if head.startswith("MMEM:CAT"):
            listing = ",".join(f'"{k},,{len(v)}"' for k, v in self.files.items())
            return "100,1000," + listing + "\n"
        if head == "TCO:RES:FREQ":
            return "1.0E+04\n"
        if head == "TCO:RES:PER":
            return "1.0E-04\n"
        if head == "MEAS1:ARES":
            return "1.0,0.75,-0.25,0.42,0.25,1e-4,1e4,2.9e-5,2.9e-5\n"
        if head == "CURS1:XDEL":
            return f"{float(self.s['CURS1:X2P']) - float(self.s['CURS1:X1P'])}\n"
        if head == "CURS1:XDEL:INV":
            return f"{1 / (float(self.s['CURS1:X2P']) - float(self.s['CURS1:X1P']))}\n"
        if head == "CURS1:YDEL":
            return f"{abs(float(self.s['CURS1:Y1P']) - float(self.s['CURS1:Y2P']))}\n"
        if head.endswith(":DATA:HEAD"):
            return "-6E-04,6E-04,1200,1\n"
        if head.startswith("DIG") and head.endswith(":DATA"):
            return ",".join("1" if i % 100 < 50 else "0" for i in range(1200)) + "\n"
        val = self.s.get(head, "0")
        if val in ("ON", "OFF") and head not in _MODE_HEADERS:
            val = "1" if val == "ON" else "0"
        return val + "\n"

    def read_bytes(self, n: int, **kwargs) -> bytes:
        if not self._rb:
            d = self.pending_read or b""
            self.pending_read = None
            ln = str(len(d))
            self._rb = f"#{len(ln)}{ln}".encode() + d + b"\n"
        out, self._rb = self._rb[:n], self._rb[n:]
        return out

    def read_raw(self, size: int | None = None) -> bytes:
        d = self.pending_read or b""
        self.pending_read = None
        n = str(len(d))
        return f"#{len(n)}{n}".encode() + d + b"\n"

    def query_binary_values(self, cmd: str, datatype="f", is_big_endian=False, container=list):
        self.log.append(cmd)
        if cmd.startswith("HCOP"):
            return b"\x89PNG fake"
        if not cmd.startswith("CHAN"):
            return list(np.sin(np.linspace(0, 12 * np.pi, 1200)))
        n = int(cmd[4])
        t = np.linspace(-6e-4, 6e-4, 1200, endpoint=False)
        amp = 0.5 if not self.clip else 5.0
        return list(amp * np.sin(2 * np.pi * 1e4 * t) + float(self.s.get(f"CHAN{n}:OFFS", 0)))
