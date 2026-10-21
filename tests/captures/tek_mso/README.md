# Tektronix MSO 4/5/6 bench captures

Real request/response traffic and artifacts captured from the live bench
oscilloscope (MSO58). These are the fixtures for the MSO unit tests (mock-VISA
playback) and parser validation. Per the testing policy, nothing here is
synthetic - every byte came off the instrument.

Moved under `tek_mso/` in iteration 4 (driver self-containment refactor); the
capture content is unchanged from iterations 2-3. See the top-level
`tests/captures/README.md` for the folder layout and the format standard that
NEW instrument captures must follow.

## Instrument / session

- **Model:** Tektronix MSO58 (8 analog channels)
- **`*IDN?`:** `TEKTRONIX,MSO58,C000000,CF:91.1CT FV:2.6.38.1348` (serial number masked)
- **Firmware:** FV:2.6.38.1348
- **Resource:** `TCPIP::<address>::INSTR` (address removed)
- **Date captured:** 2026-07-22
- **Signal source:** scope internal AFG cabled to CH5
  - `AFG:FUNCTION SINE`, `AFG:AMPLITUDE 1.0`, `AFG:OUTPUT:STATE ON`
  - `AFG:FREQUENCY 1E6` (1 MHz - chosen so a full cycle fits in a small,
    committable record; see "wfm notes" below)
  - CH5: 1 MOhm termination, 0.3 V/div (AFG 1.0 V into 50 Ohm source, 1 MOhm
    load reads ~2 Vpp)

## Files

### `meas_dispatcher_trace.json`
Full write/query trace of `meas.frequency` / `meas.amplitude` /
`meas.peak_to_peak` (the `generic_scpi` dispatcher) plus the measured values
and the parsed-waveform summary. Key captured facts for unit-test playback:
- `MEASUrement:LIST?` returns `NONE` when empty.
- A freshly added slot (`MEASUrement:ADDNew "MEAS<n>"`) defaults to type
  `PERIOD`; the dispatcher must set `TYPe` and then wait for a *fresh*
  acquisition before reading, or it reads the stale PERIOD value.
- `MEASUrement:MEAS<n>:RESUlts:CURRentacq:POPUlation?` goes 0 -> non-zero
  within ~0.1 s once a fresh acquisition lands.
- Measured: frequency 999999.87 Hz, amplitude 1.99 V, pk2pk 2.06 V.

### `ch5_sine_1mhz.wfm`
Real Tektronix internal `.wfm` binary of CH5 (1 MHz sine), record length
10000. `wfm_reader` parses it to 10064 samples (record + pre/post-charge),
dt = 0.16 ns (6.25 GS/s), y in [-1.032, 1.008] V.

### `ch5_sine_1mhz.csv`
CSV export of the *same* acquisition (10005 numeric rows) for cross-checking
the parser's sample count and scaling against the vendor exporter.

### `screenshot_mso58.png`
Full-resolution screenshot retrieved via `SAVe:IMAGe` + `FILESystem:READFile`.
Valid PNG (magic bytes `89 50 4E 47`).

## wfm notes / deviations (see context/decisions.md)

- This scope pins the sample rate at max (6.25 GS/s) and derives the time base
  from record length; a small record of a *low*-frequency signal is therefore
  near-DC. A 1 MHz AFG signal was used so a small (~11 kB) record still shows a
  full sine cycle.
- Record length and sample rate are quantized: `HOR:RECO 12500` and
  `HOR:MOD:SAMPLER 2.5E6` were rejected by the instrument (empty query
  response); 10000 is a valid record length.
- `FILESystem:READFile` streams the file bytes followed by the interface
  message terminator; reading exactly N bytes leaves that terminator in the
  buffer and desyncs the next query. The driver drains it (`_drain_terminator`).
- `*OPT?` returns verbose per-license text with no discrete AFG/DVM/DIG option
  code, so it is not a usable gate for those subsystems.

## T19 scope-extension verification (2026-07-23, MSO58 FV:2.6.38.1348, main session)

- `meas_wrapper_enumeration.json` - all 40 new measurement wrappers run live
  against AFG sine 1 MHz 1 Vpp on CH5: 39/40 returned physically correct
  values (freq 1.000 MHz, period 1 us, duty ~50%, Vpp 2.04). BURSTWIDTH never
  populates on a continuous waveform (verified twice, incl. non-clipping
  scale) - requires a bursted signal; documented in the wrapper docstring.
  Note: run captured with 0.2 V/div (signal clipped) - explains the benign
  549 "Clipping negative" measurement-warning events in the traces.
- `trig_sv_verification.json` - config_timeout/runt/window/transition accepted
  with correct readbacks; config_video accepted for NTSC/PAL/SECAM (command
  acceptance only - no video signal available from the AFG); SV peak markers:
  threshold/count settable, get_peaks() returned the 1 MHz carrier at
  +6.4 dBm as the reference peak (side entries are window-skirt artifacts).
- SV marker addendum (same session): excursion settable (10 dB readback);
  DELTA/ABSOLUTE marker types accepted - in DELTA mode with excursion raised,
  get_peaks() returns [] (only the reference peak survived filtering; deltas
  exclude the reference itself - genuine behavior, ESR=0); reference
  amplitude +6.41 dBm; marker_to_center() snapped a deliberately mis-tuned
  1.2 MHz center back to exactly 1.000000 MHz. Full SV:MARKER group (10
  commands) now live-verified.
- Horizontal semantics probe (2026-07-23, main session): AUTO mode - scale is
  the only writable knob (setting it re-picks SR, RL held); SR/RL writes are
  SILENTLY IGNORED (ESR stays 0). MANUAL mode - invariant 10*scale = RL/SR;
  write scale -> RL adjusts (SR held); write SR -> scale adjusts (RL held);
  write RL -> scale adjusts (SR held). Driver redesigned accordingly: setters
  never change mode, AUTO-mode SR/RL setters raise, set_manual() and
  snapshot()/restore() added; restore verified byte-exact twice on the bench.
- SV per-channel vertical probe (2026-07-23): CH<x>:SV:POSition is a FALSE
  FRIEND - it is Spectrum Time, not vertical position. The real per-channel
  vertical controls are DISPLAY:SPECVIEW<v>:CH<x>:VERTICAL:SCALE (dB/div,
  continuous - no snapping, 7 dB/div accepted) and :VERTICAL:POSITION
  (divisions, fractional/negative OK). No interaction with units or each
  other. Wrapped as ch<x>.sv.set_scale/set_position/get_* (view kwarg).

## Iteration 3 captures (2026-07-23, MSO58 FV:2.6.38.1348)

All iteration-3 bench scripts wrapped their work in `preserved_setup()` (the
new save/recall context manager) and left the scope as found. The scope was
left sane at session end: AFG output OFF, acquisition running, USB drive
(E:) contents unchanged.

### `preserved_setup_trace.json` (T1/R3, R5)
Round-trip proof of `preserved_setup()`: CH5 scale changed inside the context
and restored exactly on both normal exit and an exception; scope-side scratch
`.set` file cleaned up. Also `clear_status`/`wait_for_complete` sanity.

### `afg_subsystem_trace.json` (T2/R1)
All 18 AFG-group commands on CH5 loopback: every one of the 14 function tokens
accepted (read back as the full uppercase word, e.g. SINE), plus frequency/
period/amplitude/offset/high+low level/square duty/ramp symmetry/pulse width/
additive noise/output state+mode/load impedance/burst count+trigger. All with
clean *ESR?. AFG:ARBitrary:SOUrce accepts a non-existent path silently
(ADR-011).

### `capabilities_data_sv_trace.json` (T3/R2, T4/R4, T5/R7, R8)
- *OPT? parsed to options [ULTIMATE, BW5-2000, SREMBD, VID, PWR], bandwidth
  2000 MHz, feature set expands ULTIMATE -> AFG/SRCOMP/SREMBD/SVRFVT/PWR.
- Runtime probes: afg present (True), dvm absent (False - not
  registered/enabled on this unit), CH1/CH4/CH5 PROBETYPE all ANALOG (the
  TCP202 current probe on CH4 reports ANALOG, not digital).
- Data node: CURVe? scaled numpy array matches file.get_waveform on the same
  CH5 signal (min/max/mean agree within tolerance). NOTE: WFMOutpre? returns an
  UNLABELED positional list when HEADer is OFF, so the driver reads the labeled
  WFMOutpre:<field>? subqueries instead.
- SV CH5: per-channel center (SV:LOCKC OFF + CH<x>:SV:CENTERF), spectrum time
  (CH<x>:SV:POSition), squelch state+threshold - all accepted.
- HORizontal:MODe:MANual:CONFIGure default HORIZONTALSCALE; settable to
  RECORDLENGTH and back (ADR-014).

### `filesystem_usb_trace.json` (T5/R6)
USB drive detected as E:. MKDir + save_setup staged a scratch file, then
FILESystem copy/rename/rmdir all succeeded with clean ESR; drive root left
byte-for-byte as found (user file id_ed25519.pub untouched).

## Iteration 3 Run 2 captures (2026-07-23, MSO58 FV:2.6.38.1348, R10-R20)

All Run-2 bench scripts wrapped their work in `preserved_setup()` and also
snapshotted/restored acquisition state; the scope was left sane (AFG output
OFF, acquisition running). AFG sine 1 MHz / 1 Vpp on CH5. Every command was
error-attributed with a `*CLS` -> command -> `*ESR?`/`ALLEV?` probe; all
commands are clean (ESR 0) except the documented deviations below.

### `meas_config_trace.json` (T6/R10)
Reference levels (global + per-channel + per-measurement), gating (global +
per-meas), current-acq stats (MAX/MIN/PK2PK/STDDev), all-acqs PK2PK,
`DELETEALL`, and slot hygiene (LABel/STATUS?/XUNIT?/YUNIT?). Ref levels
demonstrably change results: rise time on the CH5 sine measured 291.8 ns at
10-90 vs 204.2 ns at 20-80. TIME gating (-2us..+2us) restricted the region
(gated frequency 1.0005 MHz). The benign 540 "Low amplitude" measurement
warnings in the trace (bit 4 of *ESR?) are the R5 warning-class events
`check_errors` surfaces as warnings, not errors.

### `trig_acq_trace.json` (T7/R11+R13)
B-event basics (STATE/BY/EVENTS:COUNt/TIMe + `config_b_edge`), holdoff BY
(TIME/RANDOM -> reads back RANDOM), user hysteresis STATe/VALue, `get_state()`
(returned TRIGGER). `TRIGger:STATUs?` hangs to a VISA timeout and sets CME - not
implemented on this firmware (ADR-015), so it is intentionally unwrapped.
`config_b_edge` uses the undocumented-but-real `TRIGger:B:EDGE` branch
(ADR-016). Acquisition: SEQuence:CURrent?/MODe, FASTAVerage state/limit/
stopafter - all clean.

### `history_fastframe_trace.json` (T8/R12)
FastFrame is this family's acquisition-history mechanism (ADR-017): STATE,
COUNt, MAXFRames? (8000), SELECTED, SUMFrame(+STATE), MULtipleframes:MODe,
REF frame/include, timestamp + XZEro queries - captured 10 frames of the CH5
sine and read them back via the data-node segmented readout
(`DATa:FRAMESTARt/FRAMESTOP`, MAX echoes the literal token). Deviations:
`SUMFrame <mode>` set is context-restricted (CME "Command not allowed" here);
`TSTamptable:ADDNew` needs the UI Acquisition-History feature enabled (EXE
"disabled") - both documented-correct SCPI, verified via their error paths.

### `display_vertical_ref_trace.json` (T9/R14+R15)
Display SELect (SOUrce/VIEW/MATH/BUS/REFerence + WAVEView<x>:SOUrce). Vertical
queries on CH1 (TPP1000), CH4 (TCP202 - reports PROBETYPE ANALOG), CH5 (BNC):
`CH<x>?` (59 fields) and `CH<x>:PRObe?` (21 fields) parsed to ordered field
lists via the new quote-aware `parse_compound` (ADR-018), PROBECal?/
SELFCal:State?, `CONFIGuration:ANALOg:BANDWidth?` (2.0 GHz), SELect:CH<x>,
PROBEFunc:EXTDBatten. Reference create/delete from CH5 via `REF:ADDNew` (since
`SAVe:WAVEform`-to-REF is not implemented, ADR-020); REF:LIST? confirmed REF1
then NONE after delete; label/deskew set clean.

### `exports_sv_cursor_zoom_dvm_trace.json` + `measurement_table.csv` (T10/R16-R20)
Save/Recall exports: `SAVe:EVENTtable:MEASUrement` round-tripped a real
408-byte CSV (`measurement_table.csv`) that contains the CH5 frequency
measurement; SAVe:IMAGe:COMPosition NORMAL/INVERTED. SV completion:
per-channel SPANABovebw?/SPANBELowdc?, RF_NORMal/RF_PHASe/SPECtrogram selects,
LOCKSpectrum, spectrogram CSCale MIN/MAX + slice CURSor A/B, RF_PHASe reference
master, PEAKSTABle ADDNew/LIST?/DELete (name must be `TABLE<n>`, ADR-020) - all
clean. Cursors (view-scoped form, ADR-019) on the CH5 sine: V-bar delta read
500 ns for a -200/+300 ns pair. Zoom on/off + H/V scale/position. DVM is NOT
enabled on this unit (`dvm.installed` False from the R2 probe); `DVM:MODe?`
hangs (CME), so `_Dvm` methods are gated off and implemented-from-docs only.
