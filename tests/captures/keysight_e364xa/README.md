# Keysight E3646A PSU captures (iteration 4, T3/T4)

Raw replayable traffic in `{"op","cmd","resp"}` records (queries carry the
stripped response; writes carry `resp: null`). Consumable by
`tests/mocks/capture_replay.py`.

> **This file is a historical session record. Two behaviors it describes have
> since changed.** The driver no longer probes candidate COM ports: the port is
> pinned to `ASRL6::INSTR` (ADR-034). Pacing no longer lives in
> `InstrumentLink`: `_PsuSerialLink` owns it (ADR-033). The "PACING FINDING"
> section below is also superseded - ADR-026 traced that hang to the E3646A
> deaf-window, not to a 10 ms write burst. The 50 ms default still stands.

## Session (2026-07-24) - PARTIAL, ended by PSU unresponsive event

- **Instrument:** Agilent E3646A (dual output), firmware `2.5-6.1-2.1`,
  SCPI version `1997.0`.
- **Port:** `ASRL6::INSTR` (COM6). COM4 (`ASRL4::INSTR`) was probed first with a
  single `*IDN?` and did not answer as an E364xA; COM6 answered on the second
  candidate.
- **Line settings:** 9600/8/N/1, `\n` read+write termination, 5000 ms timeout.
- **Ranges (E3646A):** LOW = 8 V / 3 A (`P8V`), HIGH = 20 V / 1.5 A (`P20V`).

### What completed cleanly (captured, bench-verified)

- `connect_trace.json` - discovery `*IDN?`, `SYST:REM`.
- `snapshot_trace.json` - as-found state read-back: output OFF; both outputs
  `P8V` range, V=0, I=3.0, OVP=22.0 V, OVP state ON.
- `pacing_experiment_trace.json` - summary of the escalating-delay experiment.
- `system_trace.json` - `SYST:VERS?` -> `1997.0`; `DISP?` -> `1`; `DISP:MODE?`
  -> `VI`; `DISP:TEXT "BENCH"` round-trips (`DISP:TEXT?` -> `"BENCH"`);
  `DISP:TEXT:CLE`; `SYST:BEEP`. All clean.
- `full_session_trace.json` - the complete ordered traffic log up to the hang.

Observed response formats (reality, for the QA replay tests):
- `VOLT?`/`CURR?` return scientific notation, e.g. `+3.00000000E+00`.
- `VOLT:RANG?` returns the range token `P8V` (not `LOW`).
- `DISP:TEXT?` returns a quoted string, e.g. `"BENCH"`.
- Boolean queries (`OUTP?`, `DISP?`, `VOLT:PROT:STAT?`) return `0`/`1`.

### PACING FINDING (drove ADR-022 / ADR-023)

The escalating experiment (50 -> 25 -> 10 ms, ten alternating `VOLT?`/`CURR?`
each) reported "all pass" - but this was misleading. Pure query traffic
self-throttles: the host blocks ~50 ms per 9600-baud round trip, so a
`min_command_interval` *below* the natural round-trip time adds no real spacing.
The experiment therefore never stressed the write path.

When the driver then adopted 10 ms and ran real work, a short burst of writes
(`DISP:TEXT:CLE`, `SYST:BEEP`, `VOLT:RANG LOW`, `VOLT 5`) at 10 ms spacing
overflowed the PSU's UART input buffer. The next query (`VOLT?`) timed out and
the PSU stopped responding. **Last good exchange: `VOLT:RANG?` -> `P8V`.**

Per the binding fragility protocol the run was STOPPED at that point (no retry,
no reconnect). The port was closed cleanly (`SYST:LOC` attempted, best-effort).

**Evidenced-safe pacing: 50 ms** (used without fault for connect + snapshot +
system group). 10 ms is unsafe for sustained write traffic. The shipped driver
default is 50 ms (`_DEFAULT_PACING_S`).

### NOT verified this session (deferred to the post-power-cycle run)

`APPLy`/`APPL?`, per-range apply+measure, OVP trip/clear, `OUTP` on/off with
measurement, `*SAV`/`*RCL`, the trigger group (`TRIG:SOUR`/`DEL`/`INIT`/`*TRG`),
status registers, calibration queries, and `SYST:ERR?` error-queue behavior
(the driver's `check_errors` SYST:ERR? drain is thus still unverified live for
this family). The bench script re-runs all of these; the next run must use the
50 ms pacing and a write-stressing pacing experiment.

## Operator action required

The PSU must be power cycled before any further bench work (it is hung from the
UART overflow). After the power cycle, re-run the bench session at 50 ms pacing.
