# test-equipment-automation

Python control of bench test instruments over VISA and SCPI. Each instrument
gets one class. You type an address, connect, and drive the instrument through
methods that follow its front panel, not its command manual.

The main user types in a Jupyter notebook or a short script, and
autocomplete is the way to find a method. So method names follow a strict
pattern. The second user is a coding agent that drafts test scripts. A
compact map of every public method in `docs/api-map/` helps that agent.

**Status: early.** The code is in daily use on one bench. It is not yet on
PyPI. The Python package name and the import names will change before the
first release.

## Safety

This code switches real outputs on and off. A wrong limit or a script bug can
damage the device under test or the instrument, and it can hurt a person.

- Every method that sources or sinks power requires its limits as arguments.
- On the power supplies, the SMU, the electronic load, and the 33500
  generator, `disconnect()` turns the outputs or the input off before it
  closes the link.
- The 33120A generator has no output relay, so its signal keeps running after
  `disconnect()`. The oscilloscope class leaves the scope's internal generator
  as it is.
- Check your wiring and your limits before you energize anything.

The software comes with no warranty. See `LICENSE`.

## Supported instruments

"Bench-tested" means the class was run against that model on real hardware.
Other models in a family are in the class's model table but are not tested.

| Instrument family | Class | Import from | Models in the class | Bench-tested |
| --- | --- | --- | --- | --- |
| Tektronix MSO 4, 5 and 6 series oscilloscopes | `Tektronix_MSO` | `drivers.tektronix_mso` | MSO 4/5/6 series | MSO58 |
| Keysight B2900 source-measure units | `Keysight_B2900` | `drivers.keysight_b2900` | B2901, B2902, B2911, B2912 | B2902B |
| Agilent or Keysight E364xA DC power supplies | `Keysight_E364xA` | `drivers.keysight_e364xa` | E3640A to E3649A | E3646A |
| Keysight 33500 and 33600 waveform generators | `Keysight_33500` | `drivers.keysight_33522a` | 33509B to 33522B, 33611A to 33622A | 33522A |
| HP or Agilent 33120A function generator | `Keysight_33120A` | `drivers.keysight_33120a` | 33120A | 33120A |
| Rigol DL3000 electronic loads | `Rigol_DL3000` | `drivers.rigol_dl3000` | DL30xx | DL3021A |
| Siglent SDM4000A digital multimeters | `Siglent_SDM` | `drivers.siglent_sdm` | SDM4055A, SDM4065A | SDM4065A |
| OWON ODP DC power supplies | `Owon_ODP` | `drivers.owon_odp` | ODP series | ODP6033 |
| Tektronix AFG31000 function generators | `Tektronix_AFG31000` | `drivers.tektronix_afg31000` | AFG31000 series | AFG31102 |
| Siglent SDG3000X waveform generators | `Siglent_SDG3000X` | `drivers.siglent_sdg3000x` | SDG3082X, SDG3162X, SDG3202X | SDG3202X |
| Rohde & Schwarz RTB2000 oscilloscopes | `RohdeSchwarz_RTB2000` | `drivers.rohde_schwarz_rtb2000` | RTB2002, RTB2004 | RTB2004 |

Tested platform: Windows 11, Python 3.14. Other platforms and Python versions
are not tested.

## Install

Until the first release, install from a clone:

```bash
git clone <this repository>
cd test-equipment-automation
python -m venv .venv
.venv/Scripts/python.exe -m pip install pyvisa-py pyserial
.venv/Scripts/python.exe -m pip install -e .
```

On Linux or macOS, use `.venv/bin/python` in place of `.venv/Scripts/python.exe`.

The install pulls in pyvisa, numpy, scipy and matplotlib.
`pyvisa-py` is a VISA backend written in Python. It needs `pyserial` for `ASRL` addresses. If your computer already has
a vendor VISA library, such as Keysight IO Libraries or NI-VISA, pyvisa uses it
and `pyvisa-py` is optional.

## Quick start

Read the address off the instrument and type it in. The code never scans for
instruments. A scan misses many instruments, returns unstable serial port
numbers, and can upset a serial link that is in the middle of a conversation.

```python
from drivers.keysight_e364xa import Keysight_E364xA

psu = Keysight_E364xA("ASRL1::INSTR")  # the address of your supply

with psu:                               # connect on entry, disconnect on exit
    psu.ch1.set_vi(5.0, 1.0, ovp=6.0)   # 5 V, 1 A limit, 6 V overvoltage trip
    psu.set_output_state_all(True)
    print(psu.ch1.meas_vi())            # Reading(voltage=..., current=...)
# On exit the outputs turn off and the front panel returns to local control.
```

Making an instrument object does not touch the instrument. The link opens only
on `connect()`, so a script that imports a list of instruments does not hang
when one of them is off.

Each module docstring has short recipes for common tasks. The method list for
each instrument is in `docs/api-map/`.

## Frequency sweeps

`src/sweep` runs Bode-style and drive-level sweeps from any source and any
measuring instrument. A source is anything that sets a frequency; a
measurement is anything that measures. Ready-made parts cover each driver
above. Every run writes a folder with the data, the waveforms, plots, a
condensed table and a short review, so the run can be analyzed again with no
instrument attached. `docs/frequency-sweep.md` is the guide. It also gives
the best use and the blind spots of each analysis method.

```python
from sweep.run import run_sweep
from sweep.plan import frequencies
from sweep.simulate import SimulatedBench, rc_lowpass

bench = SimulatedBench({"in": lambda f: 1.0, "out": rc_lowpass(1e3, 159e-9)})
run_sweep(frequencies(100, 1e5, points_per_decade=10), bench.source(),
          [bench.receiver()], amplitude=1.0,
          expressions={"H": lambda v: v["out"] / v["in"]}, folder="runs")
```

The simulated bench needs no hardware. On a real bench, replace it with a
source and a receiver from `sweep.instruments`.

## How the methods are named

| Prefix | Meaning | Example |
| --- | --- | --- |
| `set_` | Changes a setting and returns what the instrument read back | `set_vi(5.0, 1.0)` |
| `get_` | Reads a setting or a status | `get_output_state_all()` |
| `meas_` | Takes a new measurement | `meas_vi()` |
| `cmd_` | Sends a one-time command, such as a trigger or a reset | `cmd_trigger()` |

Channels and subsystems are attributes, so autocomplete narrows the list as you
type: `psu.ch1.`, `scope.horz.`, `smu.ch2.`. On the SMU, the 33500 generator,
and the OWON supply, a channel the model lacks does not appear. A
single-channel SMU has no `ch2`.

## Documentation

| File | What it holds |
| --- | --- |
| `docs/hardware-not-software.md` | Why instrument control code differs from ordinary code. Read it first. |
| `docs/driver-api-design.md` | The standard every instrument class follows. Read it before you change a class. |
| `docs/api-map.md` | An index to one method list per instrument. |
| `docs/frequency-sweep.md` | The frequency sweep guide: running, stopping, capture rules, methods and their blind spots. |

Some comments and documents cite design records by number, such as ADR-042.
Those records are kept in the original internal repository and are not part of
this copy. The text around each citation states the rule that applies.

## Vendor manuals

The command sets follow the vendor programming manuals. This repository does
not include them. Get them from each vendor's website.

| Instrument | Manual |
| --- | --- |
| Tektronix MSO 4/5/6 | 4/5/6 Series MSO Programmer Manual, 077-1305-26 |
| Tektronix waveform files | Reference Waveform File Format manual, 077-0220-11 |
| Keysight B2900 | B2900B/BL Series SCPI Command Reference |
| Agilent E364xA | E364xA User's and Service Guide, E3646-90001 |
| Keysight 33500/33600 | Trueform Series Operating and Service Guide |
| Agilent 33120A | 33120A User's Guide |
| Rigol DL3000 | DL3000 Series Programming Guide |
| Siglent SDM4000A | SDM4000A Remote Manual |
| OWON ODP | ODP Series Power Supply Programming Manual |
| Tektronix AFG31000 | AFG31000 Series Programmer Manual |
| Siglent SDG3000X | SDG3000X Series Programming Guide |
| Rohde & Schwarz RTB2000 | R&S RTB2000 User Manual, remote control chapter |

This project is not affiliated with Tektronix, Keysight, Agilent, Rigol, Rohde & Schwarz,
Siglent, or OWON. Their names appear only to say which instruments the code
controls.

## Tests

The test suite runs with no instrument attached. It replays responses that
were recorded from real instruments.

```bash
.venv/Scripts/python.exe -m pip install pytest ruff
.venv/Scripts/python.exe -m pytest
```

Tests marked `hardware` are skipped by default. To run them, connect a
Tektronix MSO with its internal generator cabled to CH5. Then set
`MSO_RESOURCE` to the scope's address and select the marker:

```bash
MSO_RESOURCE="TCPIP::192.0.2.10::INSTR" .venv/Scripts/python.exe -m pytest -m hardware
```

## Contributing

Read `docs/hardware-not-software.md` and `docs/driver-api-design.md` first.
Then follow these rules:

- Each timing constant, settle delay, or ordering rule cites a bench
  observation. Mark any behavior you did not observe with `# UNVERIFIED`.
- Never add code that scans for instruments.
- Run `ruff check` and `pytest` before you open a pull request.
- Run `tools/gen_api_map.py` after any change to a public method.

## License

Copyright 2026 SeeScan, Inc. Licensed under the Apache License, Version 2.0.
See `LICENSE` and `NOTICE`.

SeeScan, Inc. publishes this code as-is. SeeScan, Inc. does not support or maintain it.
