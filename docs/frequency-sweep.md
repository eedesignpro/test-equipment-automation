# Frequency sweeps with `src/sweep`

A Bode-style sweep from any source and any measuring instrument. The result
of every sweep is a run folder holding normalized data, plots, and short
views that a person or an AI agent can read quickly.

## The shortest sweep

```python
from bench import awg, scope                      # your own user folder
from sweep.instruments import Keysight33500Source, TektronixMsoReceiver
from sweep.plan import frequencies
from sweep.run import run_sweep

src = Keysight33500Source(awg, 1, max_vpp=0.5, load=5.0)
rx = TektronixMsoReceiver(scope, {"V_R": "CH8", "I_R": "CH3"}, trigger="CH5", autoscale=True)
scope.connect(); awg.connect()
run_sweep(frequencies(100, 5e6, points_per_decade=8), src, [rx], amplitude=0.5,
          expressions={"Z": (lambda v: v["V_R"] / v["I_R"], "ohm")}, title="5ohm")
```

A full worked example with a command-line option for each feature is
`sweep_5ohm_demo.py`, in a user folder of the internal repository. Add `dry_run=True` to see
the plan, the highest drive and the record time without touching anything.

## Running a sweep: background, check-ins, stopping

A sweep can take minutes. Start it as a background task and check in on it.
An agent runs every sweep this way by default.

1. Start the script in the background. Its first output line is
   `RUN_FOLDER <path>`, then the stop and emergency-stop commands.
2. Check in at any time. `progress.json` is rewritten after every point and
   `plot.png` at most every 30 s (`live_plot=`), from the points so far.

   ```bash
   .venv/Scripts/python.exe -m sweep status <your folder>/data/sweeps --review
   ```

   A parent folder means its newest run. `--plot` redraws the plot now.
3. Stop cleanly after the current point, or stop at once:

   ```bash
   .venv/Scripts/python.exe -m sweep stop <your folder>/data/sweeps
   ```

   ```bash
   .venv/Scripts/python.exe -m sweep estop <your folder>/data/sweeps
   ```

On any stop, an error, Ctrl+C, or a terminate signal, the sweep turns the
source output off first, then runs every `teardown` function, then restores
each instrument's found setup. A second signal during that sequence is
ignored. A hard kill of the process skips all of this, and on Windows a
terminate sent by another process cannot be caught: use `stop` or `estop`
instead. Given a parent folder, `stop` and `estop` write to the one sweep
that is running there, and refuse when none or several are running.
A `stop_if` limit is checked before the next point is set, so a sweep with
`stop_if` does not overlap processing with the next capture. The emergency stop is checked before every
capture, repeats and retries included; a capture already in progress
finishes first, and so do the autoscale retries inside the MSO receiver.

**Your own equipment:** put its safe-off in `teardown=` (one function or a
list), and give a plain-function source its own off with
`FunctionSource(fn, on=..., off=...)`. These run on every stop.

## Impedance from two voltage probes and a shunt

With both probes on the generator's ground, `sweep.expressions` gives the
DUT voltage, the current and Z, ready for `expressions=`:

- `low_side_shunt("V_total", "V_shunt", r_shunt)`: DUT on top, shunt to ground.
- `high_side_shunt("V_source", "V_dut", r_shunt)`: shunt between the source and the DUT.
- `current_probe("V", "I")`: a voltage probe and a current probe.

Any difference of measured signals works the same way in your own lambda. The
noise of each expression comes from perturbing its inputs, so a small
difference of two large voltages shows its real, larger noise on the plots.

## Other equipment: a few lines of glue

A source is anything with `set(freq, amplitude)`, or a plain function. A
measurement is anything with `measure(freq, plan)`, or a plain function that
returns a number, a dict of numbers, a `Readings`, or a `Capture` of sampled
records. Nothing in the library changes.

```python
def my_source(freq, amplitude):
    box.write(f"FREQ {freq}")
    return {"freq": float(box.query("FREQ?"))}   # the read-back, for the sanity check

def my_scope(freq, plan):                        # plan.sample_rate, plan.record_length
    t, y1, y2 = grab(plan)
    return Capture({"in": y1, "out": y2}, t=t,
                   settings={"sample_rate": actual_sr, "record_length": actual_n})

run_sweep(freqs, my_source, [my_scope], expressions={"H": lambda v: v["out"] / v["in"]})
```

Ready-made pieces, in `sweep.instruments`: sources for the 33500, 33120A,
AFG31000, SDG3000X, MSO built-in generator and RTB2000 generator; time-domain
receivers for the MSO and RTB2000; scalar readings for the SDM, E364xA, OWON,
B2900 and DL3000.

## Capture control

`CaptureRules` sets the record at each frequency. Each rule is a number, a
function of frequency, or a table of (frequency, value) points:

| Rule | Effect |
| --- | --- |
| `min_cycles` | cycles the record must hold |
| `rbw`, `rbw_fraction` | resolution bandwidth in Hz, or as a fraction of the frequency |
| `min_samples_per_cycle`, `harmonics` | sample rate floor |
| `min_record_length` | more points at no cost in time: averages noise down |
| `max_record_length`, `max_sample_rate`, `max_duration` | limits; the plan records which one cut the record |
| `window` | `blackman` (default), `blackmanharris`, `kaiser`, `flattop`, `hann`, `rect`, or a function of frequency; see the window table below |

On the 5 ohm bench, longer records at high frequency raised the estimate SNR
by 8 to 13 dB and took the run from 27 s to 44 s.

## Repeats, noise floor and calibration

- `repeats=3` captures each point three times, averages them, and records the
  spread; a spread above 1 % or 1 degree flags the point `unstable`.
- `drive_off=True` runs the same plan with the output never on: the noise and
  pickup each receiver sees at each frequency.
- `calibration={"Z": Calibration.load(path)}` applies a saved correction live.
  `sweep_5ohm_make_calibration.py`, next to the worked example, builds one
  from a run against a known standard.

## Several tones at once

`sweep.multitone.run_multitone_sweep(freqs, source, [rx], tones_per_capture=8, ...)`
plays a group of neighbouring frequencies as one waveform and measures them
from one capture. Each tone is snapped to a multiple of the group's base
frequency (under 0.25 %) and saved as its own point, so the plots and review
are the same. The source needs `set_multitone`; the 33500 source has one. The
drive amplitude is the peak to peak of the sum, so each tone gets less of it. The drift, jitter and amplitude-change check cuts the record into segments.
A segment resolves less, so with close tones it uses fewer segments, and with
tones closer than 12 bins of the whole record it reports nothing for that
tone. Before this limit, every tone of a group had false `amplitude_varies`
and `phase_jitter` flags (bench, 2026-11-03: 367 and 270 of 396 rows; 26 and
1 after).

## Drive-level sweep

`run_level_sweep(10e3, [0.05, 0.1, 0.2, 0.5], source, [rx], ...)` holds one
frequency and steps the drive, low to high. Plots and the review use the drive
as the x axis, so THD, harmonics and gain compression read against level.
Distortion that falls as the drive rises is a measurement floor, not the
device.

## Adaptive points and settling

- `refine_on="Z"` adds points at the log midpoint of every step where Z
  changes by more than `refine_db` or `refine_deg`, for a few rounds. A sharp
  resonance is found from a coarse plan.
- `settle_captures=3` captures again until two captures in a row agree within
  the larger of a limit and three times their own noise, and keeps the last.

## What every capture is checked for

Every capture reads back what the instrument actually used and checks it
against the plan: sample rate, record length, and for a scope the time scale,
division count and horizontal mode, plus the points transferred. A capture
that cannot give a trustworthy value (for example AUTO mode, a sample rate at
or below twice the frequency, under one cycle, or scale times divisions that
disagrees with record length over sample rate) is measured once more and then
flagged `capture_invalid`. `on_invalid="stop"` ends the sweep instead. Every
point also compares the source's frequency read-back with the command.

## What is measured at each point

For each signal: the complex value at the frequency actually received, the
harmonics, THD, the extra tones you name, SNR of the whole record and of the
estimate itself, phase uncertainty, spurs, clipping, and drift, jitter and
amplitude change across the record. Flags name anything doubtful:
`clip`, `low_snr`, `nonlinear`, `sideband`, `amplitude_varies`,
`phase_jitter`, `peak_off_target`, `peak_at_search_edge`, `strongest_elsewhere`,
`not_coherent`, `leakage`, `capture_invalid`.

Phase is reported only for ratios of signals from one capture.

## How a value is read: methods and blind spots

Every method below gives the same thing, an RMS phasor at time zero, so runs
made with different methods compare directly. Choose them with
`run_sweep(..., estimator=, search=, coherent=, capture=CaptureRules(window=))`.
`meta.json` records the choice. The default (`estimator="dft"`,
`search="fft"`, Blackman window) works for locked and unlocked benches.
Change it for a reason from the tables.

**Each method looks only where it is told to look.** No estimator reads a
tone, spur or interharmonic that nobody named. That signal can still bias a
value, as each "Blind spot" cell says. The checks that look at the whole
record are the safety net: `strongest_elsewhere`, `spurs.csv`, and the saved
spectra (`spectra/`, the waterfall plot). Look at them on a new device
before you trust a narrow method. No method sees a signal that aliases (above
half the sample rate, or above the scope bandwidth), a signal under the noise
floor, or an event between captures.

### Estimators: how the value at a known frequency is read

| `estimator` | Reads the value by | Best use | Blind spot |
| --- | --- | --- | --- |
| `dft` (default) | the windowed DFT at the exact frequency; no bins | any bench, locked or not; any frequency, no snapping | another tone inside the window main lobe (3 bins either side for Blackman; 1 bin = 1 / record duration) adds into the value with no flag of its own. A tone outside the main lobe is cut to the sidelobe level, so a weaker one is not seen at all. If the frequency is wrong (unlocked with `refine=False`), the value reads low: 1.1 dB at half a bin with Blackman |
| `coherent` | one bin of an FFT with no window | locked clocks with `coherent=True`: harmonics and multitone with the lowest noise (noise bandwidth 1 bin against 1.73 for Blackman, 2.4 dB better) and no leakage between tones on bins | needs the tone exactly on a bin. Off a bin it reads low (3.9 dB at half a bin) and leaks; `not_coherent` and `leakage` flag only the tone being read (`coherent_off_bins` and `coherent_leak_db` in `points.csv` hold the two numbers). `leakage` needs the neighbour bins both above -40 dB of the tone and 10 dB above the local noise floor, so noise alone does not set it; leakage under the noise floor is not seen. With no window, an unlocked spur or a tone off the bin grid spreads across the spectrum with sidelobes of only -13 dB and lands under other tones with no flag. It does no frequency search: a lost lock shows as a level change plus the two flags |
| `fit` | a least-squares sine fit at the known frequency (IEEE 1057, 3 parameters) | records of only a few cycles, where a window main lobe is wider than the tone spacing | no window, so every tone not in the fit (harmonics, the other tones of a group, spurs) biases the value when the record is not a whole number of its periods. Measured in the tests: a 2nd harmonic 28 dB under a strong fundamental read 16 % wrong (complex error) from a single-tone fit |
| `fit4` | the same fit, with the frequency fitted too (4 parameters, starting from the search result) | an unlocked single tone on a short record; a frequency estimate to a small fraction of a bin | the same as `fit`. It also moves to the nearest peak of whatever it starts on. It is the slowest method |
| `multifit` | one joint fit of the tone, the extra `tones` and the harmonics | tones closer than a main lobe; weak harmonics beside a strong fundamental; multitone groups with tight spacing | separates only the frequencies it is given; an unlisted tone biases every value. It does not fit the frequencies, so an unlocked bench needs `refine`. The fit leftover, `fit_residual_rms` in `points.csv`, is the noise plus everything unlisted: a leftover well above the noise floor means a signal is present that nobody named |

Measured cost, one capture of two 200 000-point signals with 5 harmonics, on
this PC: `coherent` 16 ms, `dft` 25 ms, `multifit` 56 ms, `fit` 67 ms, `fit4`
79 ms. `search="czt"` adds about 20 ms. Source: a timing loop over
`analysis.analyze`, best of 4, each on a new record. The pipeline runs this
behind the next capture, so it costs sweep time only when it takes longer
than the capture.

### Searches: how an unlocked tone is found

`refine=True` (the default) searches for the received frequency inside the
tolerance (`freq_tolerance`, `freq_tolerance_hz`; default 2 bins either
side) and analyzes there. `coherent` never searches. `fit4` refines the
search result once more.

| `search` | Finds the frequency by | Best use | Blind spot |
| --- | --- | --- | --- |
| `fft` (default) | the strongest FFT bin in the tolerance, a parabola through its neighbours, then a 3-point fit on exact DFT values | one dominant tone, any tolerance | the strongest peak inside the tolerance wins, also when it is not the drive (`peak_off_target` flags it). Two peaks closer than a main lobe merge into one |
| `czt` | a chirp-Z zoom of 512 points across the tolerance, then the same fit | a wide tolerance on a short record, where only a few FFT bins span it | the same strongest-peak rule. It is a finer grid, not a finer resolution |

### Windows

Measured from the window functions in `analysis.py` (FFT of the window,
zero-padded 64 times):

| `window` | Main lobe, bins either side | Highest sidelobe | Noise bandwidth, bins | Loss at half a bin | Use |
| --- | --- | --- | --- | --- | --- |
| `rect` | 1 | -13 dB | 1.00 | 3.9 dB | coherent records only |
| `hann` | 2 | -32 dB | 1.50 | 1.4 dB | close tones of similar level |
| `blackman` (default) | 3 | -58 dB | 1.73 | 1.1 dB | general use |
| `blackmanharris` | 4 | -92 dB | 2.00 | 0.8 dB | weak harmonics or spurs beside a strong tone, unlocked |
| `kaiser` (beta 14) | 4.6 | -106 dB | 2.16 | 0.7 dB | the most dynamic range; past about -96 dB the 16-bit capture is the floor |
| `flattop` | 5 | -93 dB | 3.77 | 0.01 dB | the level of a tone whose frequency is not known; the most noise |

A wider main lobe needs tones further apart. A multitone group keeps its
tones two main lobes apart, so a wider window gives longer records. The loss
at half a bin applies only when the frequency is wrong. At the right
frequency, the exact-frequency DFT has no loss.

### Coherent capture

`coherent=True` needs the source and receiver clocks locked (an external
reference: a setup task, not software). At each point the receiver sets its
horizontal first and reports the sample rate and record length it took. The
drive frequency then moves to the nearest bin of that record. The move is at
most half a bin; `freq_planned` keeps the frequency that was asked for. For
multitone, the group base becomes the largest multiple of the bin at or under
its own base, so the record holds a whole number of waveform periods and
every tone sits on a bin. A receiver takes part when it has
`prepare_horizontal`; the MSO receiver and the simulated bench have one.
UNVERIFIED on the bench: that the source sets the moved frequency exactly (the
source read-back check flags only a difference that the source reports), and
that the scope sample rate is exactly its read-back value.

### Zoom spectrum

`sweep.estimators.zoom_spectrum(x, dt, center, half_width)` gives the RMS
spectrum on a grid as fine as asked around one frequency, from a saved
waveform. It shows the shape of a peak and its near neighbours. It does not
narrow the resolution: that stays one main lobe over the record duration.

## The run folder

| File | Content |
| --- | --- |
| `meta.json` | setup, instruments, probes, rules, status, reason |
| `points.csv` | the normalized long table: one row per point, signal and tone |
| `spurs.csv` | largest spurs per point |
| `waveforms/` | raw records, saved by default; the MSO saves its own 16-bit codes and scale factors, lossless |
| `plot.png`, `plot.svg` | the standard figure |
| `condensed.csv` | about 100 points, evenly in log frequency, extremes kept |
| `wide.csv` | one row per point with every quantity, for a spreadsheet |
| `impedance_<name>.png` | for any expression in ohm: \|Z\|, phase, R, X, series L, series C and Q against frequency; L and C only where X is resolved above noise |
| `spectra/`, `waterfall_<signal>.png` | a small spectrum per point, and the waterfall of the whole sweep |
| `instrument_measurements.csv`, `.png`, `screenshots/` | the scope's own measurements and screens at each point, when asked for |
| `review.md` | summary per quantity, corners, peaks, delay, group delay, automatic fit, flags, a short table |

## After the run, no hardware needed

```bash
.venv/Scripts/python.exe -m sweep review <run_folder>
.venv/Scripts/python.exe -m sweep plot <run_folder> --harmonics V_R
.venv/Scripts/python.exe -m sweep compare Z <run_a> <run_b>
.venv/Scripts/python.exe -m sweep fit <run_folder> Z --delay
.venv/Scripts/python.exe -m sweep list <folder>
```

`sweep.combine.combine_runs` evaluates an expression across several runs, for
example coupled minus alone, and writes a new run folder. `python -m sweep
spurs <run> --known switcher=2e6` lists recurring spurs and which known tone
each one aliases from.

In Python: `sweep.store.Run(folder).series("Z")`, `sweep.interp.interpolate`,
`sweep.calibration.Calibration`, `sweep.fit.fit_rational`, `auto_fit`,
`fit_delay`, and `fit_model` for a known circuit, for example a resistor with
lead inductance and probe skew.
