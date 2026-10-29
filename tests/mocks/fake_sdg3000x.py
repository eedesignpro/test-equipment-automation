# SPDX-License-Identifier: Apache-2.0
# Copyright: see the NOTICE file.
"""A simulated SDG3202X for hardware-free tests of the SDG3000X driver.

Ported from the original repository's fake, which was built from the reply
shapes and quirks measured there on 2026-10-15 (SDG3202X, firmware 1.1.1.7R10).

Quirks reproduced: out-of-range values clamp without an error (SDG-01); a shape
change is ignored while modulation, sweep or burst is on; a sweep on a pulse
carrier is ignored; -113 for an unknown header; errors stay in the queue; the
hopping tables have no query; multitone and multipulse switch the channel to
sequence mode; multitone ignores sample-rate writes; a list clear leaves default
items; segment insert and scale fail with -200 (SDG-06); ``G1:COUP?`` has a stray
0 after TRACE,ON; ``G1:MODE?`` answers PHASE-LOCKED; sync on is refused during a
burst or sweep (SDG-12); copy, move and the file read queue a false -108
(SDG-19); a setup recall leaves the outputs off (SDG-20).

Changed in the port: a setup save and recall now keep and put back the settings,
which the original fake ignored; the original record says a recall restored every
setting its snapshot read (SDG-25). ``*RST`` now keeps the file system, the user
arbs and the saved setups, where the original fake wiped them  # UNVERIFIED

Fault injection: ``override`` (forced waveform read-back values), ``fail_on``
(commands that queue an error), ``stuck_output`` (the output stays off).
"""

from __future__ import annotations

import copy
import re

# A placeholder serial: the bench serial is not for this repository.
IDN = "Siglent Technologies,SDG3202X,SDG3XA0000000,1.1.1.7R10"
UNITS = {
    "FRQ": "HZ", "PERI": "S", "AMP": "V", "AMPVRMS": "Vrms", "OFST": "V", "HLEV": "V",
    "LLEV": "V", "DLY": "S", "WIDTH": "S", "RISE": "S", "FALL": "S", "STDEV": "V",
    "MEAN": "V", "BANDWIDTH": "HZ", "MAX_OUTPUT_AMP": "V", "PRD": "S", "TIME": "S",
    "START": "HZ", "STOP": "HZ", "CENTER": "HZ", "SPAN": "HZ", "DEVI": "HZ", "KFRQ": "HZ",
    "HFRQ": "HZ", "DLAY": "S", "ENDTIME": "S",
}  # fmt: skip
SHAPE_KEYS = {
    "SINE": ["FRQ", "AMP", "MAX_OUTPUT_AMP", "OFST", "DLY", "PHSE"],
    "SQUARE": ["FRQ", "AMP", "MAX_OUTPUT_AMP", "OFST", "DLY", "PHSE", "DUTY"],
    "RAMP": ["FRQ", "AMP", "MAX_OUTPUT_AMP", "OFST", "DLY", "PHSE", "SYM"],
    "PULSE": ["FRQ", "AMP", "MAX_OUTPUT_AMP", "OFST", "DUTY", "WIDTH", "RISE", "FALL", "DLY"],
    "NOISE": ["MAX_OUTPUT_AMP", "OFST", "STDEV", "MEAN", "BANDSTATE", "BANDWIDTH"],
    "DC": ["MAX_OUTPUT_AMP", "OFST"],
    "PRBS": ["BITRATE", "AMP", "OFST", "HLEV", "LLEV", "LENGTH", "EDGE", "LOGICLEVEL", "DIFFSTATE"],
    "ARB": ["FRQ", "AMP", "MAX_OUTPUT_AMP", "OFST", "DLY", "PHSE"],
}


def fmt(key, v):
    if isinstance(v, str):
        return v
    return f"{v:.10g}" + UNITS.get(key, "")


def lst(items):
    return ";".join(f"{i},{','.join(f'{x:.6f}' for x in it)}" for i, it in enumerate(items, 1))


class Chan:
    def __init__(self):
        self.p = {
            "WVTP": "SINE", "FRQ": 1000.0, "AMP": 4.0, "OFST": 0.0, "DLY": 0.0, "PHSE": 0.0,
            "DUTY": 50.0, "SYM": 50.0, "WIDTH": 2e-4, "RISE": 4e-9, "FALL": 4e-9,
            "STDEV": 0.2168, "MEAN": 0.0, "MAX_OUTPUT_AMP": 20.0, "BANDSTATE": "OFF",
            "BANDWIDTH": 2e8, "BITRATE": 2000.0, "LENGTH": 3.0, "EDGE": 1e-8,
            "LOGICLEVEL": "CUSTOM", "DIFFSTATE": "OFF", "HLEV": 2.0, "LLEV": -2.0,
        }  # fmt: skip
        self.prbs = {"FSTAte": "OFF", "FORMula": "X3+X2+1", "SEED": "1"}
        self.out, self.load, self.plrt, self.mode, self.cmbn = "OFF", "HZ", "NOR", "AFG", "OFF"
        self.mod = {
            "STATE": "OFF", "TYPE": "AM", "MDSP": "SINE", "SRC": "INT", "FRQ": 100.0,
            "DEPTH": 100.0, "DEVI": 100.0, "KFRQ": 100.0, "HFRQ": 1e6, "PLRT": "POS",
        }  # fmt: skip
        self.sweep = {
            "STATE": "OFF", "TIME": 1.0, "ENDTIME": 0.0, "STOP": 1500.0, "START": 500.0,
            "CENTER": 1000.0, "SPAN": 1000.0, "TRSR": "INT", "TRMD": "OFF",
            "SWMD": "LINE", "DIR": "UP", "EDGE": "RISE",
        }  # fmt: skip
        self.burst = {
            "STATE": "OFF", "PRD": 0.01, "STPS": 0.0, "TRSR": "INT", "TRMD": "OFF",
            "TIME": "1", "DLAY": 4.01e-7, "GATE_NCYC": "NCYC", "PLRT": "POS",
            "EDGE": "RISE", "COUNTER": "1", "HOLD": "START",
        }  # fmt: skip
        self.sync = "OFF"
        self.swsub = {
            "TYPE": "FREQ", "SHTime": "0", "RTime": "0", "SAMPlitude": "1", "EAMPlitude": "2",
            "AMODe": "LINE", "ADIRection": "UP", "ASYMmetry": "50", "ASNumber": "3",
        }  # fmt: skip
        self.harm = {"HARMSTATE": "OFF", "HARMTYPE": "ALL", "HARMORDER": "2"}
        self.harm_levels = {}
        self.noise_add = ("OFF", 10.0)
        self.filt = ("OFF", 2e8)
        self.arwv = (2, "StairUp")
        self.fhop = {
            "SWITch": "OFF", "TYPE": "MANUAL", "TIME": "0.001", "SFREquency": "1000000",
            "EFREquency": "10000000", "FSTep": "1000000", "RPATtern": "7",
            "RLPAttern": "7", "ALSTate": "OFF",
        }  # fmt: skip
        self.mt = {"STAte": "OFF", "RSTAte": "OFF", "NSTAte": "OFF", "SRATe": "7400",
                   "AMPlitude": "4"}  # fmt: skip
        self.tones = [[1000.0], [2000.0]]
        self.notches = [[2000.0, 3000.0]]
        self.mp = {
            "STAte": "OFF", "RSTAte": "OFF", "HLEVel": "2", "LLEVel": "0",
            "TRIGger:SOURce": "MAN", "TRIGger:TIMer": "5e-05", "TRIGger:DELAy": "1e-05",
            "SRate": "100000000", "SRate:TYPe": "AUTO",
        }  # fmt: skip
        self.pulses = [[1e-5, 1e-6], [1e-6, 1e-5]]
        self.awg = {
            "STATE": "STOP", "SRATE": "32768000", "INTPtype": "SINC", "HOLDtype": "MID",
            "USRHOLD": "0", "SCALe": "1", "OFFset": "0", "INCReasing": "INT",
            "DECReasing": "DECi", "RMODe": "CONT", "TRIGger:SOURce": "MAN",
            "TRIGger:TIMer": "0.001", "TRIGger:DELAy": "1.28e-06", "BURSt:COUNt": "1",
            "BURSt:PERiod": "0.001", "SEGMent:COUNt": "1", "TRIGAger:SLOPe": "RISE",
        }  # fmt: skip
        self.seg1 = {"WAVeform": "stairup", "LENGth": "16384", "LOOP": "1", "AMPlitude": "2",
                     "OFFset": "0", "GOTO": "1", "WAITEvent": "AUTO"}  # fmt: skip

    def busy(self):
        return "ON" in (self.mod["STATE"], self.sweep["STATE"], self.burst["STATE"])


class FakeSDG:
    """The pyvisa surface the driver uses, over a simulated two-channel instrument."""

    # The settings a setup file holds, as the fake models it.
    _SETUP_FIELDS = ("ch", "rosc", "phase_mode", "coup", "coup_vals", "counter", "iq")

    def __init__(self, idn: str = IDN):
        self.idn = idn
        self.timeout = 3000
        self.write_termination = "\n"
        self.read_termination = "\n"
        self.closed = False
        self.fs: dict[str, bytes] = {}  # "Local/x.bin" -> bytes
        self.dirs: set[str] = set()
        self.user: set[str] = set()
        self.setups: dict[str, dict] = {}
        self.log: list[str] = []
        self._reset_settings()

    def _reset_settings(self):
        self.ch = {"C1": Chan(), "C2": Chan()}
        self.errors: list[str] = []
        self.override: dict[str, object] = {}
        self.fail_on: set[str] = set()
        self.stuck_output = False
        self.phase_mode = "INDEPENDENT"
        # counter keys as the firmware takes them (SDG-26: edge and gate under TOTalizer)
        self.counter = {
            "CONFig:STATE": "OFF", "CONFig:MODE": "FREQuency", "CONFig:COUPLing": "AC",
            "CONFig:HFREJect": "OFF", "CONFig:TLEVel": "0", "CONFig:PAUSe": "ON",
            "FREQuency:RFREQuency": "10000000", "TOTalizer:EDGE": "RISE",
            "TOTalizer:GATE:STATe": "OFF", "TOTalizer:GATE:MODE": "LEVEL",
            "TOTalizer:GATE:POLarity": "POSitive", "TOTalizer:GATE:EDGE": "RISE",
        }  # fmt: skip
        self.counter_reply = {"FREQUENCY": "FREQuency", "TOTALIZER": "TOTALizer",
                              "POSITIVE": "POSitive", "NEGATIVE": "NEGative"}  # fmt: skip
        self.coup = {"TRACE": "OFF", "FCOUP": "OFF", "PCOUP": "OFF", "ACOUP": "OFF"}
        self.coup_vals: dict[str, str] = {}
        self.iq = {
            "CENTerfreq": "0", "SAMPlerate": "4000000", "SYMBolrate": "1000000",
            "AMPLitude": "0.5", "IQADjustment:GAIN": "0", "IQADjustment:IOFFset": "0",
            "IQADjustment:QOFFset": "0", "IQADjustment:QSKew": "0",
            "TRIGger:SOURce": "INT", "WAVEload": "BUILTIN 2ASK",
            "FrequencySamplingLimit": "MAX,200000000,MIN,1000",
        }  # fmt: skip
        self.rosc = ("INT", "OFF")
        self.pending_read: bytes | None = None

    # ---- pyvisa surface ----
    def close(self):
        self.closed = True

    def read_raw(self, size=None) -> bytes:
        out, self.pending_read = self.pending_read or b"", None
        return out

    def write_raw(self, data: bytes):
        self.log.append(data[:80].decode(errors="replace"))
        if data.startswith(b"MMEMory:TRANsfer"):
            path = data.split(b'"')[1].decode()
            rest = data.split(b'",#', 1)[1]
            nd = int(rest[:1])
            n = int(rest[1 : 1 + nd])
            self.fs[path] = rest[1 + nd : 1 + nd + n]
            return len(data)
        if b"WVDT WVNM," in data:
            self.user.add(data.split(b"WVNM,")[1].split(b",")[0].decode())
        return len(data)

    def write(self, cmd: str):
        self.log.append(cmd)
        if cmd in self.fail_on:
            self.errors.append(f'-222,"Data out of range;{cmd}"')
        try:
            self._do(cmd)
        except KeyError:
            self.errors.append(f'-113,"Undefined header;{cmd}"')
        return len(cmd)

    def query(self, cmd: str) -> str:
        self.log.append(cmd)
        return self._ask(cmd) + "\n"

    # ---- behaviour ----
    def _clamp(self, c: Chan, key, v):
        if key == "FRQ":
            return min(v, 200e6)
        if key == "AMP":
            return min(v, 20.0 if c.load == "HZ" else 10.0)
        return v

    def _e200(self):
        self.errors.append('-200,"Execution error"')

    def _mmem(self, cmd: str):
        args = re.findall(r'"([^"]*)"', cmd)
        op = cmd.split()[0]
        if op == "MMEMory:DELete":
            self.fs.pop(args[0], None)
        elif op in ("MMEMory:COPY", "MMEMory:MOVE"):
            self.fs[args[1]] = self.fs[args[0]]
            if op == "MMEMory:MOVE":
                del self.fs[args[0]]
            self.errors.append('-108,"Parameter not allowed"')  # false error (SDG-19)
        elif op == "MMEMory:MDIRectory":
            self.dirs.add(args[0])
        elif op == "MMEMory:RDIRectory":
            if not any(k.startswith(args[0] + "/") for k in self.fs):
                self.dirs.discard(args[0])
        elif op == "MMEMory:SAVE:XML":
            self.fs[args[0]] = b"<setup/>" * 100
            self.setups[args[0]] = {f: copy.deepcopy(getattr(self, f)) for f in self._SETUP_FIELDS}
        elif op == "MMEMory:LOAD:XML":
            saved = self.setups.get(args[0])
            if saved is not None:
                for f, value in saved.items():
                    setattr(self, f, copy.deepcopy(value))
            for c in self.ch.values():
                c.out = "OFF"  # a recall leaves the outputs off (SDG-20)
        else:
            raise KeyError(cmd)

    def _do(self, cmd: str):
        if cmd.startswith("SENSe:COUNTer:"):
            key, _, val = cmd[len("SENSe:COUNTer:") :].partition(" ")
            if key == "CLEar":
                return
            if key not in self.counter:
                raise KeyError(cmd)
            self.counter[key] = self.counter_reply.get(val.upper(), val)
            return
        if cmd.startswith("MMEMory:"):
            return self._mmem(cmd)
        if cmd.startswith("WVDT? USER,"):
            name = cmd.split(",", 1)[1]
            data = self.fs.get(f"Local/{name}.bin", b"")
            data = data[: len(data) // 2 * 2]
            self.pending_read = (
                f"WVDT POS,Local,WVNM, {name}, LENGTH, {len(data)}B, TYPE, 10,WAVEDATA,".encode()
                + data
                + b"\n"
            )
            self.errors.append('-108,"Parameter not allowed"')
            return
        if cmd == "*RST":
            self._reset_settings()
            return
        if cmd == "IQ:TRIGger:IMMediate":
            return
        if cmd.startswith("IQ:"):
            m = re.match(r"^IQ:(\S+) (.*)$", cmd)
            if not m:
                raise KeyError(cmd)
            k, v = m.groups()
            if k == "AMPLitude" and v.endswith("dBm"):
                v = repr((10 ** (float(v[:-3]) / 10) * 1e-3 * 50) ** 0.5)
            if k == "WAVEload:BUILtin":
                self.iq["WAVEload"] = "BUILTIN " + v.strip('"')
            elif k == "WAVEload:USERstored":
                self.iq["WAVEload"] = "USERSTORED " + v.strip('"')
            elif k in self.iq:
                self.iq[k] = v
            else:
                raise KeyError(cmd)
            return
        if cmd.startswith("G1:MODE "):
            self.phase_mode = cmd.split()[1]
            return
        if cmd.startswith("G1:COUP "):
            k, v = cmd.split()[1].split(",")
            if k in self.coup:
                self.coup[k] = v
                if v == "OFF":
                    for kk in [x for x in self.coup_vals if x[0] == k[0]]:
                        del self.coup_vals[kk]
            else:
                for kk in [x for x in self.coup_vals if x[0] == k[0]]:
                    del self.coup_vals[kk]
                self.coup_vals[k] = v
            return
        if cmd.startswith("ROSC "):
            a = cmd.split()[1].split(",")
            self.rosc = (a[0], self.rosc[1]) if len(a) == 1 else (self.rosc[0], a[1])
            return
        if cmd.startswith("OUT_BOTHCH "):
            for c in self.ch.values():
                c.out = "OFF" if self.stuck_output else cmd.split()[1]
            return
        if cmd.startswith(("G1:EQPHASE", "BUZZ", "KEY")):
            return
        if cmd.startswith("PACP "):
            target, source = cmd.split()[1].split(",")
            self.ch[target].p = copy.deepcopy(self.ch[source].p)
            return
        m = re.match(r"^(C[12]):(\w+)(:\S+)? ?(.*)$", cmd)
        if not m:
            raise KeyError(cmd)
        name, op, sub, rest = m.groups()
        c = self.ch[name]
        args = rest.split(",") if rest else []
        if op == "HARM":
            for k, v in zip(args[::2], args[1::2]):
                if k in ("HARMDBC", "HARMPHASE", "HARMAMP"):
                    c.harm_levels[(c.harm["HARMORDER"], k)] = float(v)
                else:
                    c.harm[k] = v
            return
        if op == "NOISE_ADD":
            d = dict(zip(args[::2], args[1::2]))
            c.noise_add = (d["STATE"], float(d.get("RATIO_DB", c.noise_add[1])))
            return
        if op == "FILT":
            k, v = args
            c.filt = (v, c.filt[1]) if k == "STATE" else (c.filt[0], max(1e7, float(v)))
            return
        if op == "PRBS":
            c.prbs[sub[1:]] = rest.strip().strip('"')
            return
        if op == "SYNC":
            if args[0] == "ON":
                if c.burst["STATE"] == "ON" or c.sweep["STATE"] == "ON":
                    return  # refused, no error (SDG-12)
                for other in self.ch.values():
                    other.sync = "OFF"
                c.sync = "ON"
            elif args[0] == "OFF":
                c.sync = "OFF"
            return  # TYPE writes ignored (MEAS)
        if op == "OUTP":
            if args[0] in ("ON", "OFF"):
                c.out = "OFF" if self.stuck_output else args[0]
            elif args[0] == "LOAD":
                c.load = args[1]
            elif args[0] == "PLRT":
                c.plrt = args[1]
            return
        if op == "CMBN":
            c.cmbn = args[0]
            return
        if op == "CPARam":
            c.mode = args[1]
            if args[1] == "IQ" and name == "C1":
                self.ch["C2"].mode = "IQ"
            if args[1] == "AFG" and name == "C1" and self.ch["C2"].mode == "IQ":
                self.ch["C2"].mode = "AFG"
            return
        if op == "BSWV":
            for k, v in zip(args[::2], args[1::2]):
                if k == "WVTP":
                    if not c.busy():
                        c.p["WVTP"] = v
                elif k in ("BANDSTATE", "LOGICLEVEL", "DIFFSTATE"):
                    c.p[k] = v
                else:
                    c.p[k] = self._clamp(c, k, float(v))
            return
        if op == "MDWV":
            if args[0] == "STATE":
                c.mod["STATE"] = args[1]
                return
            if len(args) == 1:
                return  # bare type: ignored (SDG-02, later withdrawn)
            c.mod["TYPE"] = args[0]
            for k, v in zip(args[1::2], args[2::2]):
                c.mod[k] = v if k in ("MDSP", "SRC", "PLRT") else float(v)
            return
        if op == "SWWV":
            if args[0] == "STATE":
                if args[1] == "ON" and c.p["WVTP"] not in ("SINE", "SQUARE", "RAMP", "ARB"):
                    return
                c.sweep["STATE"] = args[1]
                return
            if args[0] == "MTRIG":
                return
            for k, v in zip(args[::2], args[1::2]):
                c.sweep[k] = v if k in ("TRSR", "TRMD", "SWMD", "DIR", "EDGE") else float(v)
            return
        if op == "SWEep":
            if sub == ":EHTime":
                c.sweep["ENDTIME"] = float(rest)
            else:
                c.swsub[sub[1:]] = rest.strip()
            return
        if op == "BTWV":
            if args[0] == "STATE":
                c.burst["STATE"] = args[1]
                return
            if args[0] == "MTRIG":
                return
            for k, v in zip(args[::2], args[1::2]):
                if k == "COUNT":
                    if c.burst["TRSR"] != "INT":
                        c.burst["COUNTER"] = v
                    continue
                if k == "HOLD" and v not in ("START", "END"):
                    continue  # other words ignored (SDG-13)
                text_keys = ("TRSR", "TRMD", "GATE_NCYC", "PLRT", "TIME", "EDGE", "HOLD")
                c.burst[k] = v if k in text_keys else float(v)
            return
        if op == "ARWV":
            if args[0] == "INDEX":
                c.arwv = (int(args[1]), "StairUp")
            elif args[1].startswith('"Local/'):
                n = args[1].strip('"')[6:]
                if n[:-4] in self.user:
                    c.arwv = (None, n)
            elif args[1] in ("StairUp", "Sine", "Cardiac"):
                c.arwv = (99, args[1])
            return
        if op == "FHOP":
            k = sub[1:].split()[0] if sub else ""
            if k in c.fhop:
                c.fhop[k] = rest
                if k == "SWITch" and rest == "ON":
                    c.p["WVTP"] = "SINE"
            elif k in ("CFLIst", "COLIst", "CALIst", "MFLIst", "AFLIst", "MOLIst", "AOLIst",
                       "AALIst", "SFLIst", "SOLIst", "SALIst", "LFLIst", "LOLIst", "LALIst"):  # fmt: skip
                pass
            else:
                raise KeyError(cmd)
            return
        if op == "MTONE":
            k = sub[1:]
            if k == "STAte":
                c.mt["STAte"] = rest
                if rest == "ON":
                    c.mode = "AWG"
            elif k in ("RSTAte", "NSTAte", "AMPlitude"):
                c.mt[k] = rest
            elif k == "SRATe":
                pass  # ignored (MEAS)
            elif k == "CTLIst":
                c.tones = [[1000.0]]
            elif k == "ATLIst":
                c.tones.append([float(rest)])
            elif k == "DTLIst":
                del c.tones[int(rest) - 1]
            elif k == "CNLIst":
                c.notches = [[2000.0, 2000.0]]
            elif k == "ANLIst":
                if c.mt["NSTAte"] != "ON":
                    self._e200()
                else:
                    c.notches.append([float(x) for x in args])
            elif k == "DNLIst":
                del c.notches[int(rest) - 1]
            else:
                raise KeyError(cmd)
            return
        if op == "MPULse":
            k = sub[1:]
            if k == "STAte":
                c.mp["STAte"] = rest
                if rest == "ON":
                    c.mode = "AWG"
            elif k == "PNUMber":
                n = int(rest)
                c.pulses = (c.pulses + [[1e-6, 1e-5]] * 30)[:n]
            elif k == "MPULse":
                i, wd, gp = args
                c.pulses[int(i) - 1] = [float(wd), float(gp)]
            elif k in c.mp:
                c.mp[k] = rest
            else:
                raise KeyError(cmd)
            return
        if op == "AWG":
            k = sub[1:]
            if k in ("SEGMent:INSErt", "SEGMent:DELEte", "SEGMent:CLEAR", "SCALe", "DEFAult",
                     "SEGMent:STARTNumb"):  # fmt: skip
                self._e200()  # SDG-06
                return
            if k.startswith("SEGMent1:"):
                c.seg1[k.split(":")[1]] = rest.lower() if k.endswith("WAVeform") else rest
                return
            if k in ("TRIGger:IMMediate", "SAVE", "LOAD"):
                return
            if k in c.awg:
                c.awg[k] = rest
                return
            raise KeyError(cmd)
        raise KeyError(cmd)

    def _ask(self, cmd: str) -> str:
        if cmd.startswith("MMEMory:CATalog? "):
            folder = re.findall(r'"([^"]*)"', cmd)[0]
            if folder in ("U-disk0", "net_storage"):
                return "0,0"
            files = [
                f'"{k.rsplit("/", 1)[1]},{k.rsplit(".", 1)[-1].upper()},{len(v)}"'
                for k, v in self.fs.items()
                if k.rsplit("/", 1)[0] == folder
            ]
            dirs = [f'"{d.rsplit("/", 1)[1]}"' for d in self.dirs if d.rsplit("/", 1)[0] == folder]
            used = str(sum(len(v) for v in self.fs.values()))
            return ",".join(["7000000", used] + files + dirs)
        if cmd == "*IDN?":
            return self.idn
        if cmd == "*OPC?":
            return "1"
        if cmd == "SYST:ERR?":
            return self.errors.pop(0) if self.errors else '0,"No error"'
        if cmd == "SYST:COMM:LAN:IPAD?":
            return '"192.0.2.10"'
        if cmd == "G1:MODE?":
            if self.phase_mode == "PHASELOCKED":
                return "MODE PHASE-LOCKED"
            return f"MODE {self.phase_mode}"
        if cmd == "G1:COUP?":
            parts = []
            for k in ("TRACE", "FCOUP", "PCOUP", "ACOUP"):
                stray = ["0"] if k == "TRACE" and self.coup[k] == "ON" else []
                parts += [k, self.coup[k]] + stray
            for k, v in self.coup_vals.items():
                parts += [k, v + ("HZ" if k == "FDEV" else "")]
            return ",".join(parts)
        if cmd.startswith("SENSe:COUNTer:"):
            key = cmd[len("SENSe:COUNTer:") : -1]
            if key == "TOTalizer:GATE:EDGE" and self.counter["TOTalizer:GATE:MODE"] == "LEVEL":
                self.errors.append('-200,"Execution error"')
                return ""
            if key in self.counter:
                return self.counter[key]
            return {
                "FREQuency": "1000.0", "FREQuency:PERiod": "0.001", "FREQuency:DUTY": "25.0",
                "FREQuency:SNUMber": "3", "TOTalizer": "1234", "FREQuency:FDEViation": "1.5",
                "FREQuency:MEAN": "1000.0", "FREQuency:MAX": "1000.2", "FREQuency:MIN": "999.8",
                "FREQuency:SDEV": "0.1", "FREQuency:DUTY:MEAN": "25.0",
                "FREQuency:MEAN:FDEViation": "1.4",
            }[key]  # fmt: skip
        if cmd == "ROSC?":
            return f"ROSC {self.rosc[0]},10MOUT,{self.rosc[1]},AMPL,31dBm"
        if cmd in ("BUZZ?", "KEY?"):
            return f"{cmd[:-1]} ON"
        if cmd.startswith("IQ:"):
            k = cmd[3:-1]
            if k == "WAVeinfo":
                mod = self.iq["WAVEload"].split()[-1]
                return (
                    f"WAVE_INFO,SYMBOL_LENTH,1024,OVER_SAMPLING,4,MODULATION,{mod},"
                    "FILTER_TYPE,RootCosine,FILTER_ALPHA,0.35"
                )
            return self.iq[k]
        if cmd == "STL? USER":
            return "STL WVNM," + ",".join(sorted(self.user))
        if cmd == "STL? BUILDIN":
            return "STL M2, StairUp, M0, Sine, M26, Cardiac"
        m = re.match(r"^(C[12]):(\w+)(:\S+)?\?$", cmd)
        if not m:
            self.errors.append(f'-113,"Undefined header;{cmd}"')
            return ""
        name, op, sub = m.groups()
        c = self.ch[name]
        if op == "SYNC":
            return f"{name}:SYNC {c.sync},TYPE,CH{name[1]}"
        if op == "HARM":
            if c.harm["HARMSTATE"] == "OFF":
                return f"{name}:HARM HARMSTATE,OFF"
            o = c.harm["HARMORDER"]
            dbc = c.harm_levels.get((o, "HARMDBC"), -80.0)
            phase = c.harm_levels.get((o, "HARMPHASE"), 0.0)
            return (
                f"{name}:HARM HARMSTATE,ON,HARMTYPE,{c.harm['HARMTYPE']},HARMORDER,{o},"
                f"HARMAMP,0.001V,HARMDBC,{dbc:g}dBc,HARMPHASE,{phase:g}"
            )
        if op == "NOISE_ADD":
            state, db = c.noise_add
            return f"{name}:NOISE_ADD STATE,{state},RATIO,{10 ** (db / 10):g},RATIO_DB,{db:g}dB"
        if op == "FILT":
            return f"STATE,{c.filt[0]},COFF_FRQ,{c.filt[1]:.0f}HZ"
        if op == "PRBS":
            return c.prbs[sub[1:]]
        if op == "OUTP":
            return f"{name}:OUTP {c.out},LOAD,{c.load},PLRT,{c.plrt}"
        if op == "CMBN":
            return f"{name}:CMBN {c.cmbn}"
        if op == "CPARam":
            return f"MODE,{c.mode}"
        if op == "BSWV":
            vals = dict(c.p)
            vals.update(self.override)
            keys = [
                k
                for k in SHAPE_KEYS[c.p["WVTP"]]
                if k != "BANDWIDTH" or vals.get("BANDSTATE") == "ON"
            ]
            body = ",".join(f"{k},{fmt(k, vals[k])}" for k in ["WVTP"] + keys)
            return f"{name}:BSWV {body}"
        if op == "MDWV":
            if c.mod["STATE"] == "OFF":
                return f"{name}:MDWV STATE,OFF"
            keys = {
                "AM": ["MDSP", "SRC", "FRQ", "DEPTH"], "FM": ["MDSP", "SRC", "FRQ", "DEVI"],
                "PM": ["MDSP", "SRC", "FRQ", "DEVI"], "PWM": ["MDSP", "SRC", "FRQ", "DEVI"],
                "DSBSC": ["MDSP", "SRC", "FRQ"], "ASK": ["SRC", "KFRQ"],
                "FSK": ["SRC", "KFRQ", "HFRQ"], "PSK": ["SRC", "KFRQ", "PLRT"],
            }[c.mod["TYPE"]]  # fmt: skip
            if c.mod["SRC"] == "EXT":  # MEAS: external drops the internal keys
                keys = [k for k in keys if k not in ("MDSP", "FRQ", "DEPTH")]
            body = ",".join(f"{k},{fmt(k, c.mod[k])}" for k in keys)
            if c.mod["TYPE"] in ("ASK", "FSK", "PSK") and c.mod["SRC"] == "EXT":
                body = body.replace("SRC,EXT", "TRSR,EXT_A")
            return f"{name}:MDWV STATE,ON,{c.mod['TYPE']},{body},CARR,WVTP,{c.p['WVTP']}"
        if op == "SWWV":
            if c.sweep["STATE"] == "OFF":
                return f"{name}SWWV STATE,OFF"
            body = ",".join(f"{k},{fmt(k, v)}" for k, v in c.sweep.items() if k != "STATE")
            body = body.replace("TRSR,MAN", "TRSR,MAN_A").replace("TRSR,EXT", "TRSR,EXT_A")
            return f"{name}SWWV STATE,ON,{body},CARR,WVTP,{c.p['WVTP']}"
        if op == "SWEep" and sub == ":EHTime":
            return f"{c.sweep['ENDTIME']:g}"
        if op == "SWEep":
            return c.swsub.get(sub[1:], "0")
        if op == "BTWV":
            if c.burst["STATE"] == "OFF":
                return f"{name}:BTWV STATE,OFF"
            skip = {"STATE"}
            if c.burst["TRSR"] == "INT":
                skip |= {"COUNTER", "EDGE"}
            if c.burst["TRSR"] == "MAN":
                skip |= {"EDGE"}
            body = ",".join(f"{k},{fmt(k, v)}" for k, v in c.burst.items() if k not in skip)
            body = body.replace("TRSR,EXT", "TRSR,EXT_A").replace("TRSR,MAN", "TRSR,MAN_A")
            return f"{name}:BTWV STATE,ON,{body},CARR,WVTP,{c.p['WVTP']}"
        if op == "ARWV":
            if c.arwv[0] is None:
                return f"{name}:ARWV NAME,{c.arwv[1]}"
            return f"{name}:ARWV INDEX,{c.arwv[0]},NAME,{c.arwv[1]}"
        if op == "FHOP":
            k = sub[1:]
            if k in c.fhop:
                return c.fhop[k]
        if op == "MTONE":
            k = sub[1:]
            if k == "TLISt":
                return lst(c.tones)
            if k == "NLISt":
                return lst(c.notches)
            if k == "SRATe":
                return "37000"
            if k in c.mt:
                return c.mt[k]
        if op == "MPULse":
            k = sub[1:]
            if k == "PULse":
                return ";".join(f"{i},{w:.9f},{g:.9f}" for i, (w, g) in enumerate(c.pulses, 1))
            if k == "PNUMber":
                return str(len(c.pulses))
            if k in c.mp:
                return c.mp[k]
        if op == "AWG":
            k = sub[1:]
            if k.startswith("SEGMent1:"):
                return c.seg1[k.split(":")[1]]
            if k in c.awg:
                return c.awg[k]
        self.errors.append(f'-113,"Undefined header;{cmd}"')
        return ""
