# Changelog

All notable changes to this project are documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] - 2026-07-29

Iterations 4, 5 and 6. This release turns a single-instrument project into a
bench: five instruments now have drivers, and the drivers cross-check each other
on real hardware instead of only being exercised against captures. It also
carries two operator directives from 2026-07-29 that change shipped behavior.

### Added
- **Keysight/Agilent E364xA DC power supply** (`src/drivers/keysight_e364xa.py`):
  APPLy, per-output V and I with `INSTrument` select, P8V and P20V ranges, OVP,
  output relay control, system and status groups, calibration queries,
  `snapshot()` / `restore()` / `preserved_setup()`, plus `set_vi()` and
  `set_state()` conveniences. Bench unit is an E3646A on `ASRL6::INSTR`.
- **Rigol DL3000 electronic load** (`src/drivers/rigol_dl3000.py`): full surface
  on subsystem nodes (cc, cv, cr, cp, transient, meas, trig, status, ocp, opp,
  battery, list). The public API is SI-unit and float-only. `safe_state()` turns
  the input off on disconnect.
- **HP/Agilent 33120A function and arbitrary waveform generator**
  (`src/drivers/keysight_33120a.py`): APPLy, function and level control, output
  load and SYNC, AM, FM, FSK, burst and sweep nodes, trigger, status,
  calibration queries, and arbitrary-waveform upload by floats or 12-bit DAC
  codes with chunked paced transfers. On `ASRL4::INSTR`.
- **Owon ODP6033 triple-output power supply** (`src/drivers/owon_odp.py`):
  ch1, ch2 and ch3 output nodes with select-per-method, V and I set and measure,
  OVP through `VOLT:LIM` and OCP through `CURR:LIM` (per channel and all
  channels, including `? MAX`), per-channel and global output enable, APPLy,
  and parallel, series and track coupling. About 95 percent of the 24-command
  surface.
- **Siglent SDM4065A 6.5-digit multimeter** (`src/drivers/siglent_sdm.py`), a
  new instrument type: measurement primitives (CONF, MEAS, READ, FETCh, INIT,
  ABOR, FUNC, DATA), per-function configuration nodes (dcv, acv, dci, aci, res,
  fres, freq, period, cap, temp), and trigger, calc, status and system nodes.
  About 90 percent of the base instrument. The ROUTe scanner is excluded because
  this model has none.
- **DMM deep capture without a Data Log API**: `digitize(n)` reaches a
  bench-measured 49.7 kSa/s (200 000 points in 4.02 s) and `capture(n,
  interval=T)` holds an exact sample interval. Both are built from documented
  primitives after probing proved the vendor Data Log and Digitize headers do
  not exist on firmware 0.0.0.22R1 (ADR-035). `data_available()` and
  `DATA:POINts?` polling give a non-destructive progress signal (ADR-036).
- **Bench harnesses and tooling**: `src/fixture/bench_validation.py` (PSU
  against e-load cross-check: CC, CV, CR, CP, OVP trip and clear, ranges),
  `src/fixture/siggen_verify.py` (33120A settings verified on the MSO58),
  `src/fixture/bench_smu_verify.py` (3-way source-measure loop: PSU setpoint
  against DMM voltage against e-load current and voltage, inside a 30 V and 1 A
  envelope), `src/fixture/equipment.py` (pinned handles `scope`, `psu1`, `psu2`,
  `eload`, `siggen`, `dmm`), and `projects/scpi_helper.py` (raw-SCPI CLI,
  scenario runner, link clear, driver self-test).
- `InstrumentUnresponsiveError` in `common/errors.py`: a VISA query timeout is
  fatal and is never retried. The operator must power cycle (ADR-024).
- Overload handling for the DMM: a `9.9E37` or `9.91E37` reading is a sentinel
  and becomes `float('nan')`. It is never returned as a value and never raises,
  so an unattended run does not stop on an over-range (ADR-038).
- `src/drivers/tektronix_common/` and `src/drivers/tektronix_mso/` packages: the
  MSO driver is now self-contained, with `wfm_reader` shared at vendor level.

### Changed
- **BREAKING: `InstrumentLink` no longer paces I/O** (ADR-033, supersedes
  ADR-022). The `min_command_interval` constructor parameter and the internal
  `_pace()` are gone from `src/common/link.py`. Inter-command timing is a
  property of a specific instrument on a specific transport, so each driver that
  needs it owns it in its own link subclass: `_PsuSerialLink` in
  `keysight_e364xa.py` and `_SigGenSerialLink` in `keysight_33120a.py`. Driver
  constructors keep their `pacing_s` parameter and the 50 ms default, so normal
  callers see no change. Code that constructed an `InstrumentLink` with
  `min_command_interval` directly must move the delay into its own subclass.
- **BREAKING: the E364xA driver no longer scans COM ports** (ADR-034, partly
  supersedes ADR-025). `Keysight_E364xA` takes one pinned resource string
  (default `ASRL6::INSTR`) and the `candidate_ports` parameter and attribute are
  removed. `connect()` opens that port, reads one `*IDN?`, then enters remote
  mode. A non-E364xA answer now raises `InstrumentError` naming the port. There
  is no requirement to auto-discover hardware: the bench is fixed and known, and
  probing ports pokes other instruments.
- The E3646A driver settles around its deaf-window (ADR-026 and ADR-027). After
  `VOLT:RANG`, `SYST:REM`, `SYST:LOC`, `SYST:RWL`, or a status-enable write to a
  new value, the supply stops answering queries for roughly 0.5 to 1.5 seconds
  while it still accepts writes. An immediate read-back both times out and
  desyncs the serial stream.
- E3646A output enable is a single global relay, so `set_state()` lives on the
  instrument and not on the channel (ADR-029).
- `Keysight_33120A.safe_state()` deliberately does NOT de-energize on
  disconnect: this generator has no output relay, so the last waveform keeps
  running. `Owon_ODP.safe_state()` does de-energize (`OUTP OFF`), because that
  supply does have one.
- `.claude` permission allowlist narrowed to a hybrid: read-anything and
  arbitrary-python auto-approval removed, specific dev grants kept (ADR-032).
- CLAUDE.md gained the "Working with Live Equipment" procedure: only the main
  agent talks to instruments, the per-command error protocol, and bench-state
  discipline.

### Fixed
- E3646A `set_voltage_range` validates before it selects, so a bad token cannot
  leave the wrong output selected.
- `Rigol_DL3000.restore()` restores `function_mode` and guards the NONE-function
  case.
- The "fragile UART" and "PSU on COM4" record was a misdiagnosis from trusting
  stale bench notes. The link is robust and the hang was the deaf-window desync.
  The E3646A is on `ASRL6` and `ASRL4` is the 33120A (ADR-026 and ADR-028).

### Known limitations
- The ADR-033 and ADR-034 serial changes are lint-clean and unit-test-clean but
  are NOT yet re-run on the E3646A or the 33120A. One read-mostly self-test per
  instrument closes this. See note 1 in `context/qa-report.md`.
- DMM accuracy is unconfirmed for ACV, DCI, ACI, frequency, period, capacitance
  and temperature. Each needs a source or a known component that the fixed
  wiring does not have. `context/dmm-verify-gaps.md` lists them.
- The DMM 2 M reading depth is extrapolated from 200 000 verified points.
- The DMM has no remote path for logging to file (the 360 M path). That feature
  is front-panel only.
- Driver unit tests are an explicit operator exception: capture replay cannot
  express a fixed interconnect, a timing fact, or an absent SCPI header. The 429
  unit tests cover `common/` and the MSO driver.

## [0.3.0] - 2026-07-23

Iteration 3: closes out the full R1-R20 coverage pool (operator directive -
no iteration-4 deferrals) on top of the v0.2.0 node skeleton, plus a
foundational setup-preservation fix used to dogfood the rest of the
iteration's bench work.

### Added
- AFG subsystem: full 18-command wrap on `_Afg` (function shapes including
  SINE/SQUare/PULSe/RAMP/NOISe/DC/SINC/GAUSsian/LORENtz/ERISe/EDECAy/
  HAVERSINe/CARDIac/ARBitrary, frequency/period/amplitude/offset/high-low
  level, square duty, ramp symmetry, pulse width, noise add state+percent,
  output state/mode, load impedance, burst count/trigger, arb source load);
  setters return read-back actuals; enables closed-loop AFG->CH5 self-test
  without raw `link.write` in hardware tests.
- Capabilities module (`src/drivers/capabilities.py`): `parse_opt()` +
  bundle expansion (ULTIMATE/STARTER) plus authoritative runtime probes for
  `afg`/`dvm`/per-channel digital install flags (no `*OPT?` substring
  gating, per ADR-004); DVM probed present-but-unactivated on the bench
  unit.
- `preserved_setup()` context manager (`Instrument`-level): save-on-enter,
  exact-state restore-on-exit even under exception; setup/session
  save-recall commands on the file node; used throughout this iteration's
  own bench scripts (dogfooded first, per plan).
- `data` node: waveform transfer core path - `DATa:SOUrce/STARt/STOP/
  ENCdg/WIDth` setters, `WFMOutpre?` parsing, `CURVe?` binary block decoded
  directly to a scaled numpy array (bypasses the scope-disk round trip);
  bench-cross-checked against `file.get_waveform` on the same signal.
- Framework sync/status helpers: `link.clear_status()` (`*CLS`),
  `link.wait_for_complete()` (`*OPC?`/`*WAI`), `instrument.reset()`
  (`*RST`); measurement-warning-aware error filter treats event class
  540-549 as a warning rather than a raised `InstrumentError`.
- `common/scpi.py`: shared `quote()`/`parse_compound()` helpers (promoted
  out of ad hoc driver quoting) for compound-response parsing without a
  naive quote strip.
- Measurement configuration: reference levels at all three scopes (global,
  per-measurement, per-channel; METHod PERCent/ABSolute, rise/fall
  high/mid/low, hysteresis, BASETop - bench-proven the 10/90 vs 20/80
  rise-time difference); gating (global + per-measurement time window and
  active state); statistics completion (CURRentacq MAX/MIN/PK2PK/STDDev,
  ALLAcqs PK2PK); `delete_all()`; measurement slot hygiene (label, status,
  x/y units).
- Trigger additions: B-event sequence basics (state, event count, time,
  edge source/slope/level/coupling), holdoff BY (auto/random), user
  hysteresis (state/value), state/status query wrappers.
- Horizontal History + FastFrame: history group (state/selected/overlay/
  stats/reference-acquisition/include + timestamp queries) and FastFrame
  group (state/count/max-frames/selected/sum-frame/multiple-frames-mode/
  reference-frame/timestamp/XZero queries), paired with the data node's
  segmented readout.
- Acquisition additions: `SEQuence:CURrent?` progress query, `SEQuence:
  MODe`, FASTAVerage trio (state/limit/stop-after).
- Display `SELect` commands: source/view/math/bus/reference plus
  per-waveview source selection.
- Vertical query completion + reference-waveform basics: all remaining
  query-only vertical commands with correct compound-response parsing,
  `SELect:CH<x>`, `PROBEFunc:EXTDBatten`, ref add/delete/list/label/deskew.
- Save/Recall exports: event-table measurement + peaks CSV export/
  retrieval, legacy plot-data port, image-composition save (CSV round-trip
  bench-proven).
- Spectrum View completion: peak table add/delete/list, lock-spectrum,
  span-above-bandwidth/span-below-DC queries, spectrogram/RF-vs-time batch
  (RF-normal/RF-phase/spectrogram select, spectrogram color-scale +
  cursors, RF-phase reference master).
- Spectrum View per-channel additions: per-channel center frequency
  (distinct from the existing global center), `set_spectrum_time()`
  (reconciled against the pre-existing `_Spectrum.set_position` naming so
  the two are not confused), squelch state/threshold.
- Cursor basics: waveform cursor state, source, A/B positions, and
  readouts (~10 core commands).
- Zoom completion: full 12-command zoom group (on/off, H/V scale/
  position).
- DVM node body: full 12-command wrap, gated behind the capabilities
  probe's install flag (documented as unit-unactivated on the bench, not
  bench-verified per ADR-020).
- FILESystem `copy`/`rename`/`rmdir` on the file node, routed through a new
  shared quote helper; verified against a USB drive (network mount/
  unmount explicitly out of scope).
- `HORizontal:MODe:MANual:CONFIGure` probed and wrapped
  (`set_manual_config`/`get_manual_config`) once semantics were confirmed
  live (ADR-014).
- 10 new bench-verified ADRs (ADR-011 through ADR-020), all operator-
  accepted.
- Hardware milestone test (`test_mso58_milestone.py`) is now self-
  contained: drives its own AFG state via the new subsystem inside
  `preserved_setup()` instead of relying on ambient bench state.
- 101 additional unit tests (8 new files) closing out coverage gaps on the
  iteration's earliest, structural subsystems (AFG, capabilities,
  `preserved_setup()`, data node, filesystem trio, SV per-channel
  additions, manual-config probe, and the 540-warning classifier) that
  had shipped ahead of dedicated tests; unit suite now at 422 tests (422/
  422 passing) plus 7/7 live hardware tests on the bench MSO58.

### Notes
- QA verdict: PASS_WITH_NOTES. All R1-R20 acceptance criteria met; the note
  is a process finding, not a defect - 4 of 17 committed bench-capture
  traces were recorded in a non-replayable summary format rather than raw
  op/cmd/resp pairs, which left 8 of 20 requirements without dedicated unit
  coverage until this audit's 101 added tests closed the gap. Recommended
  for iteration 4: standardize the bench-capture format on the raw
  op/cmd/resp shape.
- Security review: CLEAN (0 Critical/High/Medium/Low, 3 Info - all
  accepted, no action required).

## [0.2.0] - 2026-07-23

First release of the rebuilt `Vendor_Model`-style driver framework. Combines
iteration 1 (framework core, closed without a release) and iteration 2
(full MSO 4/5/6 node surface) into a single tagged release.

### Added
- Framework core: `Link`/`Node`/`Instrument` architecture in `src/common/`
  (`link.py`, `node.py`, `instrument.py`, `errors.py`), replacing the legacy
  `scpi_node.py`. Includes a mock VISA resource for hardware-free unit
  testing.
- `Tektronix_MSO` driver ported onto Node/Link with explicit `ch1`-`ch8`
  channels, IDN-gated install, and enum-validated setters.
- Full MSO 4/5/6 node skeleton: every subsystem in the front-panel/manual
  command tree exists as a `Node` (stub or fully implemented), including
  `horz`, `acq`, `trig`, per-channel `sv`/`probe`, global `sv`, `meas`,
  `file`, `data`, `afg`, dynamic `math(n)`, `ref1`-`ref4`, `bus1`-`bus4`,
  `dig`, `zoom`, `cursors`, `disp`, `search`, `dvm`.
- `meas` measurement subsystem: `generic_scpi`/`custom` dispatcher, 11
  legacy wrappers plus 40+ new named wrappers organized as `meas.vert`
  (amplitude-domain), `meas.time` (timing), and `meas.dual` (two-source:
  delay/phase/skew/setup/hold/common-mode), full statistics exposure, and a
  `custom()`/`custom_dual()` escape hatch for option-gated measurement
  types.
- `file` subsystem: `get_screenshot`, `get_waveform`, `get_waveform_data`,
  `get_spectrum`, `write_file`, local/remote path validation.
- Standalone `.wfm` parser (`src/utils/wfm_reader.py`) with
  `read_wfm_file`/`read_wfm_data` entry points and driver-side
  `parse_wfm_to_numpy`/`parse_wfm_data_to_numpy` delegators.
- Trigger configuration coverage: `config_video` (NTSC/PAL/SECAM),
  `config_timeout`, `config_runt`, `config_window`, `config_transition`, in
  addition to existing edge/pulse configs.
- Vertical flesh-out: label, deskew, offset/position ranges,
  termination, bandwidth-optimization, and a full `_Probe` node
  (id/type/serial/gain/units/resistance/status/input-mode/forced-range/
  autozero/degauss).
- Horizontal flesh-out: mode-aware API (`set_manual`, `snapshot`/`restore`,
  `configure()` for exact scale/sample-rate/record-length pairs), div count,
  roll-mode query, acquisition duration, actual sample rate, delay mode.
- Spectrum View peak markers: threshold/excursion settings, peak count,
  per-peak frequency/amplitude readout, marker type, `marker_to_center`, and
  per-channel vertical scale/position.
- Value-handling layer 1: `ValueSnappedWarning` (warn-not-raise by default,
  optional `strict`) on `set_and_verify`; `float | Literal[...]` token
  support on setters where the manual documents discrete tokens (e.g.
  `set_bandwidth("FULL")`).
- Two-tier test suite: 241 unit tests (mock-VISA, replaying real captured
  instrument traffic per the project's hardware-first testing policy) plus
  7 hardware tests (marker `hardware`, skipped by default; run live against
  the bench MSO58).
- `tests/captures/`: documented real-bench artifacts (`.wfm`, CSV export,
  PNG screenshot, dispatcher traces, `*OPT?` response, SV peak list) - no
  synthetic instrument fixtures anywhere in the suite.

### Fixed
- Legacy `CH<x>:COUPling` command was incorrectly sent as `CH<x>:BANdwidth`.
- Legacy `SV:WINDOW` argument typo (`WINOW`) corrected.
- Pulse trigger `WITHIN` qualifier check corrected.
- Edge trigger slope token corrected from `EITH` to `EITHER`.
- `temporary_timeout` context manager was missing its `@contextmanager`
  decorator.
- `query_float`/`query_int` silently returned `None` on a bad response
  instead of raising.
- MSO 4/5/6 does not implement standard `SYST:ERR?` (the query hangs to VISA
  timeout); added an overridable `Instrument.check_errors()` hook, with the
  Tek override draining `*ESR?`/`ALLEV?` instead.
- Measurement dispatcher returned a stale value: `MEASUrement:ADDNew`
  defaults a fresh slot to type `PERIOD`, and population alone did not
  reflect the just-set type/source; the dispatcher now waits for a fresh
  acquisition (`ACQ:NUMAC?` advance + non-zero population) before reading
  the result.
- `peak_to_peak` used an undocumented `PK2P` type token; corrected to the
  manual-documented `PK2PK`.
- Horizontal setters (`set_samplerate`/`set_recordlength`) previously forced
  `HOR:MODE MAN` as an undocumented side effect; in `AUTO` mode the scope
  silently ignores sample-rate/record-length writes with no SCPI error.
  Setters are now mode-aware and no longer force a mode change; added
  `set_manual(sr, rl)` and `snapshot()`/`restore()` for exact state
  restoration.
- `_Channel.set_units` (security, Low): rejected embedded double-quotes that
  could break out of the quoted SCPI argument; same guard applied to
  `set_label`.
- `InstrumentLink._parse_binary_block` (security, Low): raises
  `InstrumentQueryError` on a truncated payload (actual bytes shorter than
  the declared count) instead of silently returning a short slice.
- `_File.get_spectrum` copy-paste bug: trace selection previously wrote the
  wrong SCPI command.
- File-transfer terminator leak: reading exactly `nbytes` after
  `FILESystem:READFile` left the interface terminator in the input buffer,
  corrupting the next query; now drained explicitly.

### Security
- Both iteration-1 Low findings (`set_units` quote handling,
  `_parse_binary_block` truncation) fixed and independently re-verified.
- Iteration 2 security review: CLEAN (0 Critical/High/Medium/Low, 3 Info,
  all accepted with no action required).
