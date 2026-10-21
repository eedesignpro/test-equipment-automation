# Driver API Design Standard

This standard applies to every driver under `src/drivers/`. Read it before you
write or change a driver. `CLAUDE.md` holds the short version. This file is the
authority.

Related records: ADR-041 (task-shaped APIs), ADR-042 (unverified hardware
behavior), ADR-043 (naming and signature rules).

## Terms

This document uses one name for each thing. Use the same names in driver code
and docstrings.

- **operator** - the human who uses the driver from a notebook or a script.
- **agent** - an AI that reads or writes driver code.
- **node** - an object that holds part of an instrument, such as a channel.
- **helper** - one method that sets several coupled values in one call.
- **probe** - a short bench test that measures real instrument behavior.
- **front panel** - the physical controls and menus on the instrument.

---

## 1. Why this document exists

Agents built the drivers in this repo command-first. The method was mechanical.
Read the vendor SCPI reference. Walk the command tree. Write one public method
per command. This gives high command coverage. It also gave 1362 public methods
across eight drivers.

The result has four names for "turn the output on": `get_output`,
`get_output_state`, `get_input`, and `get_state`. It has three conventions for a
measurement. On some instruments the most common bench action needs five calls
across two objects.

The root cause is a category error. **A SCPI reference is organized for lookup,
not for use.** It answers the question "what does `OUTP:LOW` do?". It does not
answer the question "how do I bias this DUT at 200 mV and read the current?". A
driver that mirrors the reference gets the structure of a filing system. It does
not get the structure of an interface.

SCPI trees are also not clean designs to copy. They carry decades of forced
backwards compatibility. Related settings sit in unrelated subsystems for
historical reasons. The B2900 puts source compliance under `SENSe`. The DL3000
writes a list sweep as indexed per-step commands. A driver that copies these
choices imports another vendor's compatibility debt.

Two more causes made the result worse. Both will happen again without this
document.

1. No API standard existed. `CLAUDE.md` gave `snake_case` and the
   `Vendor_Model` class pattern and nothing more. Agents wrote seven drivers in
   seven separate contexts. Each driver was reasonable alone. The set was not.
2. Convenience aliases look free from inside one file. Examples are
   `output_on()`, `set_state()`, and `input_on()`. Autocomplete is a shared and
   finite resource. Every alias takes from it. The cost lands on the operator
   who types `set_o<TAB>` and gets six results.

## 2. The design target

Every instrument on the bench already has a good interface. Its front panel
groups controls the way an operator thinks about the instrument. Common actions
take one or two button presses.

**An operator who knows how to do an action on the front panel must find the
method by autocomplete alone.** This is the acceptance test. Command coverage is
not.

This library has two consumers. Design for both.

1. **The operator, typing in a Jupyter notebook.** This is the primary case.
   Autocomplete is the discovery tool, so the namespace is part of the
   interface.
2. **An agent, drafting tests and scripts.** An agent cannot autocomplete. It
   pays context tokens to read. Section 9 covers what this needs.

The two consumers want the same things for different reasons. A predictable name
saves the operator a lookup. It saves the agent a file read.

---

## 3. Naming

### 3.1 Four verb prefixes

Choose the prefix from the **primary intent** of the call. Do not choose it from
the SCPI that the method sends.

| Prefix | Meaning |
|---|---|
| `set_` | Changes a setting or a state. Returns the read-back. |
| `get_` | Reads a setting or a status. Takes no new measurement. |
| `meas_` | Takes a new physical measurement. |
| `cmd_` | Sends a transient command. Examples: `*TRG`, `INIT`, `ABOR`, `*RST`, save, recall, beep, clear. |

Intent decides. `dmm.meas_vdc(range=100, nplc=100)` keeps the `meas_` prefix. It
writes two settings first, but its purpose is to return a measurement. Range and
NPLC change the reading, so they belong to the measurement.

Any method whose purpose is to change instrument state gets `set_`.

These methods need no prefix: `connect`, `disconnect`, `snapshot`, `restore`,
`preserved_setup`, `channel`, `__enter__`, and `__exit__`.

### 3.2 Names are paths

A method name is a path. The operator types it to walk a tree. Put the general
segment first, so each keystroke narrows the result set.

```
set_source_voltage        set_waveform_sine        meas_resistance_4w
set_source_current        set_waveform_square      meas_resistance_2w
```

**The node tree carries the subsystem name or the method carries it. Never
both. Never neither.** `scope.meas.frequency()` is correct, because the node
already names the subsystem. `eload.meas_voltage()` is correct, because no
measure node exists.

Long names cost nothing. Autocomplete removes the typing penalty. Prefer a long
clear name over a short unclear one. This matters most for rare methods.

### 3.3 Use `_all` for anything that affects several outputs

A method that acts on more than one output must say so in its name. Use `_all`.
Do not add a plural `s`.

```python
psu1.set_output_state_all(bool)      # one global enable, both outputs
psu2.ch1.set_output_state(bool)      # this channel only
psu2.set_voltage_all(v1, v2, v3)
```

Apply this test. **Could a reader take the same name for a per-channel call?**
A reader could take `set_output_state` that way, so it needs `_all`. A reader
could not take `set_coupling(SERIES)` that way, because coupling between
channels always spans channels. So `set_coupling` needs no suffix.

`_all` means "all channels". It never means "all quantities". Name a
multi-quantity read after what it returns: `meas_vi()` or `meas_vip()`.

`_all` does not promise a collection return. `psu1.get_output_state_all()`
returns `bool`, because the E3646A has one physical enable.
`psu2.get_output_state_all()` returns `list[bool]`, because the ODP channels can
differ. State the return type in each docstring.

### 3.4 Terminology follows the instrument class

Use the words printed on that class of instrument. Do not invent one word for
all classes.

| Instrument class | Name for the 4-wire connection | Method |
|---|---|---|
| PSU, e-load | remote sense | `set_remote_sense_state(bool)` |
| SMU | 4-wire | `set_4w_sense_state(bool)` |
| DMM | 4-wire, per measurement | `meas_resistance_4w()` and `meas_resistance_2w()` |

Do not use `set_sense_state`. It reads as though the instrument senses nothing
when the setting is off.

The DMM has no plain `meas_resistance`. Both wire counts are explicit, so the
operator guesses nothing.

### 3.5 Abbreviations

An abbreviation earns its place when it meets two tests. It is standard in the
domain. It also labels the argument order.

`set_vi(voltage, current)` passes both tests. `set_vdc_ilimit` fails.
"ilimit" says that current is a limit. A bench supply in CC mode regulates
current as the setpoint. The load decides which value limits, not the call.

### 3.6 One name has one meaning across every driver

A method name must mean the same thing on every instrument. Two drivers that
share a name and not a behavior create a trap. Nothing warns the reader.

The repo already holds one. `set_output_mode` on the 33522A selects NORMAL or
GATED gating. `set_output_mode` on the MSO AFG selects OFF, CONTINUOUS, or BURST
run mode. Same name. Same bench. Different behavior.

Rename both. The 33522A value gates the output, so call it `set_gating_state`.
The MSO value is a run mode, so fold BURST into `set_burst(...)` and fold OFF
into `set_output_state(False)`.

Check every new name against the other seven drivers before you use it.

---

## 4. Signatures

### 4.1 The keyword boundary

**Put the `*` marker directly after the arguments that the method name names.**
The operator must pass every other argument by keyword. An argument is required
when it has no default.

```python
def set_vi(self, voltage, current, *, ovp=None): ...
# the name names both values, so both stay positional

def set_source_voltage(self, volts, *, i_limit, v_range=None, i_range=None): ...
# the name names one value, so one stays positional
# i_limit is required, and the caller must spell it
```

Python enforces this rule. No decorator and no validation code are needed. The
`TypeError` names the missing argument.

Any reader can derive this rule from a method name. It needs no judgment.

### 4.2 Limits are required on anything that sources or sinks power

A PSU operating point without a current limit is not valid. An optional limit
creates a call that is wrong by omission. A required limit removes that error
class.

A required limit also makes every energizing call readable at the call site. The
operator never scrolls back to find the compliance value.

```python
psu.ch1.set_vi(voltage, current, *, ovp=None)          # both required
smu.ch1.set_source_voltage(volts, *, i_limit, ...)     # i_limit required
eload.set_load_current(amps, *, v_limit, von=None)     # v_limit required
```

Keep protections optional. The operator leaves OVP, OCP, Von, and ranges at
default almost always.

Compliance takes two forms. A scalar means a symmetric limit of plus and minus
the value. A `(min, max)` tuple means an asymmetric limit, and `min` is
negative. Example: `(-0.5, 2.0)`. Both forms use one code path. The scalar case
is the tuple with a sign flip.

### 4.3 Helpers track coupling, not instrument size

A helper earns its place when the settings are **coupled**. Coupled settings
define one physical operating point together. That operating point must be valid
at every moment. Instruments with outputs have coupled settings, because their
settings *are* an operating point.

| Coupled, so use a helper | Independent, so keep atomic |
|---|---|
| SMU source: level, compliance, range | Scope vertical: scale, offset, coupling, bandwidth |
| E-load mode, level, limits | Scope acquisition, display, cursors |
| PSU voltage, current, OVP, range | DMM per-function config |
| Waveform shape, frequency, amplitude, offset | Any single-value setting |
| Scope trigger. The type sets which arguments apply. | |
| Scope horizontal. Scale, record length, and sample rate lock together. | |

Scope vertical settings are independent views of an incoming signal. A bad write
snaps or limits the value. It does not create an invalid state. Do not bundle
these settings.

### 4.4 Every `set_` returns the read-back

A `set_` method returns the value that the instrument holds after the write.
When the instrument has no query for that value, return the value written, and
say so in the docstring. A `set_` method never returns `None`.

A helper returns a frozen dataclass. Examples: `SupplyState`, `SourceState`,
`LoadState`, and `WaveformState`. The operator gets autocomplete on the fields
of the return value. The matching `get_` returns the same type.

### 4.5 Minimal-write re-callability

**A second call with one changed argument must change only that parameter.**
Call `set_waveform_square(1e3, 2.0, duty=30)`. Call it again with `duty=70`.
Only the duty cycle changes. The frequency write must not reset output phase.
The amplitude must not glitch.

A helper therefore compares its arguments against the present instrument state.
It writes only the values that differ.

**Assume no cached state.** The front panel and other sessions can change any
setting at any time, so the helper queries the instrument. A query may cost more
than a redundant write, or less. That cost is instrument-specific. Measure it.
See section 6.

First measurement, for scale: on the MSO58 a query costs 0.74 ms and a
synchronized write costs about 135 ms, so a redundant write is about 180 times
more expensive than the check that avoids it. A same-value write costs the same
as a new-value write, so the instrument does no short circuit of its own. On
that instrument this rule is a large speed win as well as a correctness rule.
Do not carry those numbers to another instrument. Measure each one (ADR-044).

Two changes are discontinuous by nature. A change of waveform shape and a change
of source mode both break the output. Everything else must be clean. "Clean"
means verified on a scope, not assumed.

### 4.6 Enums

Multi-option arguments use `common.scpi.ScpiEnum`, which is a plain `enum.Enum`
whose `__str__` returns the SCPI token.

**Do not use `enum.StrEnum`.** A `StrEnum` member inherits all 47 string
methods, so `psu2.Coupling.<TAB>` offered 51 completions of which 4 were the
enum and the rest were `capitalize`, `casefold`, `encode` and friends.
Autocomplete is the discovery tool here, so the namespace is part of the
interface (Section 2), and 4-in-51 defeats it. With `ScpiEnum` the completion
list is exactly the members.

`str(member)` and `f"{member}"` still give the token, so formatting call sites
are unaffected. The one thing given up is `member == "TOKEN"`, which is now
False. Compare `.value`, or pass the member to the driver: every token helper
accepts a member, a member name, or a token string, case-insensitively.

Watch for `isinstance(value, str)` when converting an existing driver. That test
used to catch enum members and no longer does. It is the one silent breakage,
and it bit the scope's `set_load`.

Meet two requirements.

1. **Make the enum total.** Include the default member and the off member. Add
   `Coupling.INDEPENDENT` next to `SERIES`, `PARALLEL`, and `TRACK`. An enum
   that cannot state the power-on condition forces a string, and the string
   defeats the enum.
2. **Attach the enum to the class.** Jupyter offers no members inside
   `set_coupling(<TAB>)`. Attach the enum so `psu2.Coupling.<TAB>` resolves from
   the live object.

Never pass a raw string for a two-way choice. Put the choice in the method name.
Use `set_source_voltage` and `set_source_current`. Do not use
`set_source_function("VOLT")`. A string argument never autocompletes.

### 4.7 Units carry no scalar prefix

Every public argument and return is in the unprefixed unit. No milli, micro,
kilo, or mega. Convert at the driver boundary when the instrument speaks a
prefixed unit: the DL3000 takes milliseconds, kilohertz, and amperes per
microsecond, and the driver presents seconds, hertz, and amperes per second.
Volts, amperes, ohms, watts, seconds, hertz.

The rule is "no scalar prefix," not "SI base unit." Where an unprefixed
compound unit is the industry standard for a quantity, use it rather than
forcing a base unit: battery capacity is **amp-hours (Ah)**, not milliamp-hours
and not coulombs; energy is **watt-hours (Wh)**, not kilowatt-hours and not
joules. `mAh` is banned; `Ah` is correct. State the unit in the docstring
(Section 9.1, rule 5).

---

## 5. Structure

### 5.1 Build capability into the object, and never raise instead

**A method that exists and then raises is a lie to the completion popup.** Build
the object graph from what the instrument is. Autocomplete then documents the
capability by itself.

- The 33120A has no output relay. It gets **no** `set_output_state`. Not one
  that raises. The method is absent.
- A B2901 gets no `ch2`.
- The E3646A enable is global. Its channels get **no** output method. The
  instrument gets `set_output_state_all`.

A `HardwareNotPresentError` stub frustrates the operator. It also fills the
namespace with methods that no call can ever use.

### 5.2 Build the node tree in `connect()`

`__init__` sets the resource name and the link. It does nothing more.
`connect()` reads `*IDN?` and attaches the model-dependent nodes.

**`__init__` must do no I/O.** A user's bench file, `bench.py`,
builds every bench instrument at import time. Construction I/O would hang on any
powered-off instrument.

There is no `model=` constructor argument. `*IDN?` is authoritative. A hand-typed
model string can never beat it. A script that must assert a model does that in
its own logic.

Jupyter reads attributes from the live object, so nodes attached in `connect()`
autocomplete correctly. Pylance cannot see them in a `.py` file. This standard
accepts that trade for runtime truth.

### 5.3 Underscore the granular layer

**The underscore is autocomplete tiering. It is not access control, and it is
not privacy.** These drivers present a typed user interface to hardware the
operator owns. There is no security boundary here and nothing to hide: an API
that locks the operator out of their own bench is a failed API. The underscore
says only "this is niche enough that it should not crowd the top of the
completion list". Every underscored method stays callable, stays documented, and
stays supported.

It follows that a reshape **never removes a capability**. Folding six atomic
setters into one helper is correct, but the six must survive with an underscore.
Someone will eventually need to set one value without disturbing the other five,
or to reach a mode the helper does not compose, and the driver must let them do
it without editing the driver. Renaming is fine. Deleting is not.

A public command needs **justification**. Do not make a command public by
default. Where a helper covers the action, give the commands it uses a `_set_`
prefix. They stay available for bench work. They keep the helper readable at
review time. They stay out of the first tier of autocomplete.

This is a strong guideline, not a law. State the reason for each exception in the
docstring. These exceptions stand:

- `set_frequency()` and `set_amplitude()` on the signal generators. A loop that
  steps frequency must not rewrite amplitude on every pass.
- `psu2.set_voltage_all()` on the ODP. `APP:VOLT` is atomic in hardware. Three
  per-channel calls are not. Simultaneity is the reason to call it.
- Waveform helpers take `freq, ampl` positionally. The name names neither value.
  Frequency then amplitude is the order on every signal generator front panel
  and in SCPI `APPLy`.

`Node` holds eleven raw I/O helpers: `write`, `query`, `query_float`,
`query_int`, `query_bool`, `write_choice`, `set_and_verify`, `read_bytes`,
`read_raw`, `read_binary_block`, and `write_binary_block`. Every node in every
driver inherits them. They add eleven noise entries to every autocomplete list.
Underscore them.

### 5.4 Add a measure node only when measurement is one subsystem

The scope earns `scope.meas`. Measurement is one menu among many, and a large
method count there is expected. The operator already holds that map.

The DMM does not earn one. Measurement is the whole instrument. Its measure
methods sit at the top level.

### 5.5 Method count follows the mental model

This is not a strict one-to-one law. Keep the method count at a node as close as
reasonable to the operator's mental model of that region of the instrument.
Where the panel has one button, prefer one method. A node with far more methods
than the panel has controls is a design smell. Investigate it.

---

## 6. Hardware behavior needs the bench

**An agent will break this section first.**

No agent can know these facts:

- how long a relay takes to settle
- whether a same-value write costs anything
- whether a mode change clears a level register
- whether a query inside a settle window returns late or not at all
- whether a frequency rewrite resets output phase

This is a gap in training data, not a gap in reasoning. Inference cannot close
it. Text written about these facts without an observation is generated
plausibility.

This repo already paid for that error once. The "fragile UART / PSU on COM4"
record was a misdiagnosis. An agent treated a plausible story as a fact. See
ADR-026 and ADR-028.

### 6.1 Rules

1. Cite the bench observation for every timing constant, settle delay, ordering
   rule, and cost claim. Put the citation at the constant or in the docstring.
2. Mark every unobserved behavior `# UNVERIFIED`. No helper may depend on it.
   `UNVERIFIED` is the grep target for open probe work.
3. Probe one command at a time. Check its own `SYST:ERR?`. Record a wall-clock
   delta. Never hide a behavior question inside a long chained sequence.
4. Sub-agents cannot reach the bench. Probing is a main-agent task. A sub-agent
   that needs a hardware fact writes the probe and stops.

### 6.2 Helper sequence rules

Rules 1 to 4 are safe by construction. Apply them in every helper. Rule 5 is
instrument-specific and needs measurement.

1. Write range and mode first. They clamp or snap dependent values.
2. Write limits before levels, always. No moment may exist where a new level
   runs under an old limit. This order is safe in both directions, so the helper
   needs no conditional logic.
3. Write levels last.
4. Never energize inside a config helper. Closing an output relay is always a
   separate `set_output_state` call.
5. Measure redundant-write cost and settle cost. Take three observations per
   instrument. Does a same-value write cost anything? Does the settle depend on
   a real change? Is a read-before-write cheaper than the settle?

### 6.3 The live mode-switch hazard

A mode change on an energized output applies the new mode's **stale** level from
an earlier session. The new level lands after that. On the SMU, a switch from
voltage force to current force with the relay closed can apply a stale current
setpoint. On the e-load, `set_load_current` on a live input can apply a stale CC
level.

A mode-changing helper **refuses by default** when the output is live. The
message names the fix. The helper accepts `allow_live=True` as an explicit
escape. A helper that opens a relay by itself creates its own surprise.

**Scope the refusal to the transition, not to the call.** The hazard is the
switch, so refuse only when the call would actually change something: the mode,
the function, a range, or the output shape. A call that moves only a level or a
limit changes no operating mode and must go through untouched. A helper that
refuses whenever it is called teaches the operator to pass `allow_live=True`
out of habit, and the flag is then already set on the day a real switch happens.
Compare each argument against the present state and refuse on the difference.

**Give every sourcing driver a level-only setter**, such as `set_source_level`.
Stepping a level on a live output is ordinary bench work: an IV curve, a
threshold search, a soak. That call writes the level and nothing else, so it
cannot switch anything and needs no flag. Without it the operator has only the
full helper, and the full helper drags mode, range and limit writes along with
it. This is the same justified exception as `set_frequency` in section 5.3, and
it is measurably cheaper: on the B2902B a level-only write is 2 round trips
against about 15 for the full helper.

---

## 7. Banned patterns

Agents add more often than they remove. These additions damaged this repo.

| Banned | Use instead |
|---|---|
| `output_on()`, `output_off()`, `input_on()` | `set_output_state(bool)` |
| `set_state()` and `get_state()` as aliases of another pair | one name |
| Any alias pair. Two names, one behavior. | one name |
| `is_*()` and bare-noun getters such as `voltage()`, `free()`, `condition()` | `get_*` |
| `X_tripped()` | `get_X_tripped()` |
| A method that raises `HardwareNotPresentError` in place of absence | remove the method |
| `set_*` that returns `None` | return the read-back |
| A raw string for a two-way choice | put the choice in the method name |
| An enum without its default member | make the enum total |
| `set_sense_state` | `set_remote_sense_state` or `set_4w_sense_state` |
| A method that polls one error queue where the instrument needs two commands | check the vendor sequence. The MSO hides events until `*ESR?` is read (ADR-044). |
| `meas_all()` for all quantities | `meas_vi()` or `meas_vip()`. `_all` means channels. |
| A timing constant with no cited observation | cite it or mark `# UNVERIFIED` |

---

## 8. Reference API shapes

The operator agreed these shapes. `...` bridges obvious gaps. Every driver also
gets `connect`, `disconnect`, `snapshot`, `restore`, `preserved_setup`,
`cmd_reset`, `get_errors`, and the underscored plumbing.

### Keysight E3646A, dual-output PSU, global enable

```python
psu1.set_output_state_all(bool) -> bool     # hardware has ONE enable for both
psu1.get_output_state_all() -> bool
psu1.ch1, psu1.ch2                          # ch2 on dual-output models only
                                            # the channels get NO output method

ch.set_vi(voltage, current, *, ovp=None) -> SupplyState
ch.get_vi() -> SupplyState
ch.meas_vi() -> Reading
ch.meas_voltage() -> float
ch.meas_current() -> float
ch.get_condition() -> str                   # CV, CC, or UNREG
ch.get_ovp_tripped() -> bool
ch.cmd_clear_ovp()
```

`set_vi` auto-ranges. The range must satisfy voltage **and** current together.
The LOW range is 8 V and 3 A. The HIGH range is 20 V and 1.5 A. When no range
fits, raise and name both limits.

### Owon ODP6033, 3-channel PSU, per-channel enable

```python
psu2.set_output_state_all(bool) -> bool
psu2.set_coupling(Coupling.SERIES) -> Coupling
psu2.set_voltage_all(v1, v2, v3)            # justified: atomic in hardware
psu2.ch1, psu2.ch2, psu2.ch3

ch.set_vi(voltage, current, *, ovp=None, ocp=None) -> SupplyState
ch.set_output_state(bool) -> bool
ch.meas_vi() -> Reading
ch.meas_voltage() / meas_current() / meas_power() -> float
```

### Rigol DL3021A, e-load

```python
eload.set_output_state(bool) -> bool
eload.set_remote_sense_state(bool) -> bool

eload.set_load_current(amps, *, v_limit, von=None, slew=None) -> LoadState
eload.set_load_voltage(volts, *, i_limit, von=None) -> LoadState
eload.set_load_resistance(ohms, *, i_limit, von=None) -> LoadState
eload.set_load_power(watts, *, i_limit, von=None) -> LoadState
eload.get_load() -> LoadState

eload.meas_voltage() / meas_current() / meas_power() / meas_resistance() -> float
eload.meas_vip() -> Reading
eload.meas_capacity() -> float              # accumulated Ah
eload.meas_energy() -> float                # accumulated Wh
eload.meas_time() -> float

eload.set_battery_test(amps, *, stop_voltage, stop_capacity=None, stop_time=None)
eload.set_transient(level_a, level_b, *, width_a, width_b, mode=Transient.CONTINUOUS)
eload.set_list(levels, widths, *, slews=None, count=1)
eload.cmd_trigger()
```

`set_load_<TAB>` gives the four mode keys. The method **is** the mode select.
`set_list` takes Python lists. It replaces 15 indexed per-step accessors.
`stop_voltage` is required. A discharge without a cutoff damages the cell.

### Keysight B2902B, SMU

```python
smu.set_output_state_all(bool) -> list[bool]
smu.ch1, smu.ch2                            # ch2 on B2902 and B2912 only

ch.set_source_voltage(volts, *, i_limit, v_range=None, i_range=None,
                      allow_live=False) -> SourceState
ch.set_source_current(amps, *, v_limit, ...) -> SourceState
ch.get_source() -> SourceState
ch.set_output_state(bool) -> bool
ch.set_4w_sense_state(bool) -> bool
ch.set_nplc(nplc) -> float

ch.meas_voltage() / meas_current() / meas_resistance() -> float
ch.meas_vi() -> Reading
ch.get_compliance_tripped() -> bool

ch.set_sweep_voltage(start, stop, points, *, i_limit, spacing=Spacing.LINEAR, ...)
ch.set_sweep_current(start, stop, points, *, v_limit, ...)
ch.set_list_voltage(values, *, i_limit)
ch.set_list_current(values, *, v_limit)
ch.meas_sweep() -> Reading                  # atomic READ:ARRay
```

### Keysight 33120A, function generator, NO output relay

```python
siggen.set_waveform_sine(freq, ampl, *, offset=0.0) -> WaveformState
siggen.set_waveform_square(freq, ampl, *, offset=0.0, duty=None)
siggen.set_waveform_triangle(freq, ampl, *, offset=0.0)
siggen.set_waveform_ramp(freq, ampl, *, offset=0.0)
siggen.set_waveform_noise(ampl, *, offset=0.0)
siggen.set_waveform_dc(offset)
siggen.set_waveform_arb(samples, freq, ampl, *, offset=0.0)
siggen.get_waveform() -> WaveformState

siggen.set_frequency(hz) -> float           # justified: loop stepping
siggen.set_amplitude(volts) -> float
siggen.set_load(ohms | Load.HIGHZ) -> float
siggen.set_sync_state(bool) -> bool         # SYNC connector, NOT a signal relay

siggen.set_am(depth, *, rate=None, shape=None)
siggen.set_fm(deviation, *, rate=None, shape=None)
siggen.set_burst(count, *, rate=None, phase=None)
siggen.set_sweep(start, stop, time, *, spacing=Spacing.LINEAR)
siggen.set_modulation_off()
siggen.cmd_trigger()
```

`set_waveform_<TAB>` gives the FUNCTION menu. `set_<TAB>` also shows the atomic
parameter setters. The `set_waveform_` prefix groups the full-spec calls without
a fifth verb that only two drivers would use.

This instrument gets **no `set_output_state`**. It has no output relay.
`set_sync_state` controls the SYNC connector. Its name prevents the mistake.

### Keysight 33522A, 2-channel AWG

```python
awg.set_output_state_all(bool) -> list[bool]
awg.ch1, awg.ch2

ch.set_waveform_sine / _square / _triangle / _ramp / _pulse / _noise / _dc
ch.set_waveform_arb(samples, *, sample_rate, ampl, offset=0.0)
ch.get_waveform() -> WaveformState
ch.set_output_state(bool) -> bool           # a real relay, unlike the 33120A
ch.set_load(ohms | Load.HIGHZ) -> float
ch.set_frequency(hz) / ch.set_amplitude(volts)
ch.set_am / set_fm / set_pm / set_burst / set_sweep(...)

awg.sync.set_state(bool) -> bool
awg.set_track(Track.OFF) -> Track
awg.cmd_phase_sync() / awg.cmd_trigger()
```

The waveform vocabulary matches the 33120A, so one mental model covers both.

### Siglent SDM4065A, DMM, no measure node

```python
dmm.meas_vdc(*, range=None, nplc=None) -> float
dmm.meas_vac(*, range=None) -> float
dmm.meas_idc(*, range=None, nplc=None) -> float
dmm.meas_iac(*, range=None) -> float
dmm.meas_resistance_2w(*, range=None, nplc=None) -> float
dmm.meas_resistance_4w(*, range=None, nplc=None) -> float
dmm.meas_frequency(*, range=None, aperture=None) -> float
dmm.meas_period(*, aperture=None) -> float
dmm.meas_capacitance(*, range=None) -> float
dmm.meas_diode() / meas_continuity() -> float
dmm.meas_temperature(*, probe=None, sensor=None) -> float
dmm.meas_buffer(count, *, function, interval=None) -> list[float]
dmm.meas_stats() -> Stats
```

The method name and the return units imply the units of `range`. Every config
argument is keyword-only, because the name names only the function.

### Tektronix MSO58, scope, atomic by default

```python
scope.ch1 ... scope.ch8                     # attached per model at connect
ch.set_scale(volts_per_div) -> float        # independent settings stay atomic
ch.set_offset(volts) / ch.set_position(divisions)
ch.set_coupling(Coupling.DC) / ch.set_bandwidth(hz) / ch.set_termination(ohms)
ch.set_invert(bool) / ch.set_label(str) / ch.set_display_state(bool)
ch.set_deskew(seconds) / ch.set_attenuation(ratio)
ch.get_clipping() -> bool

scope.horz.set_timebase(*, scale=None, samplerate=None,
                        record_length=None, position=None) -> TimebaseState
scope.horz.get_timebase() -> TimebaseState

scope.trig.set_edge(source, *, level, slope=Slope.RISE, coupling=None)
scope.trig.set_pulse_width(source, *, level, low, high, polarity=...)
scope.trig.set_timeout(source, *, level, time)
scope.trig.set_mode(TrigMode.NORMAL) / scope.trig.get_state() -> str

scope.cmd_run() / cmd_stop() / cmd_single()
scope.acq.set_mode(AcqMode.SAMPLE) / acq.set_count(n)

scope.meas.frequency(source) / amplitude(source) / rms(source) / ...
scope.meas.get_stats(num) -> MeasStats
scope.data.get_waveform(source, *, start=None, stop=None)
scope.file.get_screenshot(title, folder_path=None) / save_waveform(source, path)

scope.afg.set_waveform_sine(freq, ampl, *, offset=0.0, noise_add=None)
scope.afg.set_waveform_square(freq, ampl, *, offset=0.0, duty=None, noise_add=None)
scope.afg.set_waveform_ramp(freq, ampl, *, offset=0.0, symmetry=None, noise_add=None)
scope.afg.set_waveform_pulse(freq, ampl, *, offset=0.0, width=None, noise_add=None)
scope.afg.set_waveform_arb_file(path, freq, ampl, *, offset=0.0, noise_add=None)
scope.afg.set_waveform_noise(ampl, *, offset=0.0)      # no noise_add
scope.afg.set_waveform_dc(offset)                      # no noise_add
scope.afg.get_waveform() -> WaveformState
scope.afg.set_output_state(bool) -> bool
scope.afg.set_load(ohms | Load.HIGHZ) -> float
scope.afg.set_burst(count, *, ...) / scope.afg.cmd_burst_trigger()
```

The AFG adds noise to every output type except DC and noise. The signature
carries that limit. `noise_add` appears on the five waveform methods that accept
it. It is absent from `set_waveform_dc` and `set_waveform_noise`. Autocomplete
then shows the argument only where the hardware accepts it. No runtime check is
needed, and no method raises.

The atomic setters `_set_noise_add_state` and `_set_noise_add_percent` go
private. A loop that steps noise level is rare, unlike a loop that steps
frequency.

The AFG uses the same waveform vocabulary as the 33120A and the 33522A. One
mental model then covers all three signal sources on this bench. Three
differences stay, because they are real hardware. The AFG loads an arb from a
scope file path, so it gets `set_waveform_arb_file(path, ...)` where the
generators get `set_waveform_arb(samples, ...)`. The AFG has no modulation, so
it gets no `set_am` and no `set_fm`. Noise-add exists only on the AFG.

`set_timebase` is the strongest helper case on this instrument. Scale, record
length, and sample rate lock together as `10 x scale = RL/SR`. AUTO mode also
**drops SR and RL writes without an error**. One call can force MANUAL and then
apply a consistent set of three values. Three atomic setters cannot.

The trigger gets one helper per trigger type, because the required arguments
change with the type. `source` stays positional. Every other argument is
keyword-only.

`scope.meas` keeps a large method count. That is correct. It is the measurement
menu, and the operator already holds that map.

### One asymmetry is deliberate

The PSU gets `set_vi(v, i)`. The SMU gets `set_source_voltage(v, *, i_limit)`
and `set_source_current(i, *, v_limit)`. The supply has no priority to state.
The SMU does. The names differ **because the instruments differ**. A consistency
pass would flatten this and make both worse. Do not "fix" it.

---

## 9. Support for agents

An agent cannot autocomplete. It spends context tokens to learn the API. Eight
driver files hold about 8700 lines, so a full read is expensive and wasteful.
The items below cut that cost. They are ranked by leverage.

### 9.1 Generate one API map file

Write `docs/api-map.md` from introspection. Generate it with a script, so it
cannot drift from the code. An agent reads one file of a few hundred lines
instead of eight files of 8700 lines.

Use one line per method. Include the full call path, the signature, the return
type, and one short purpose clause.

```
psu1.ch1.set_vi(voltage, current, *, ovp=None) -> SupplyState   set both setpoints, auto-range
psu1.ch1.meas_vi() -> Reading                                   read voltage and current in one pass
psu1.set_output_state_all(state) -> bool                        one global enable for both outputs
```

Group the lines by instrument, then by node. Regenerate the file as a test step,
and fail the build when it changes without a commit. This is the single highest
return item in this section.

#### What the generator needs from the source

The generator reads the source. It never touches the bench. Write driver code so
that the generator can extract every field. These six rules are a contract.

1. **The first docstring line is one complete sentence.** Keep it under 90
   characters. State the purpose and the units. The generator takes that line
   and nothing more.

   **Do not name SCPI commands in a public docstring.** A helper sends several
   commands, and the set changes with the arguments, so any single name
   misleads. A command name also pulls the reader back to the command tree,
   which this standard exists to escape. The commands sit on the write lines a
   few lines below. Anyone tracing SCPI is already in the source.

   ```python
   def set_source_voltage(self, volts, *, i_limit, v_range=None):
       """Force voltage in V with a current compliance in A.

       <detail paragraph, which the generator drops>

       Range: model-dependent. See the SOA table in the module docstring.
       """
   ```

   Name the SCPI command in the **private** layer only. There the mapping is one
   to one, and a driver maintainer needs it. Private methods never reach the
   map, so this costs no context.

   ```python
   def _set_volt_level(self, volts: float) -> float:
       """Write the immediate voltage level (SOUR<c>:VOLT)."""
   ```

2. **Annotate every parameter and every return.** The generator reads the
   signature with `inspect.signature`. An unannotated return prints as bare, and
   the agent must then read the body.

3. **Declare the node tree with class-level annotations.** `connect()` attaches
   the nodes, so no instance exists offline. A class-level annotation without an
   assignment gives the generator the tree shape. It creates no attribute at
   runtime, so autocomplete stays truthful.

   ```python
   class Keysight_B2900(Instrument):
       ch1: _SmuChannel
       ch2: _SmuChannel      # B2902 and B2912 only
   ```

4. **Keep model gating in data, not in code branches.** The generator reads a
   table such as `_MODEL_INFO` and prints the gate in the map. It cannot read an
   `if` statement inside `_on_connect`.

5. **State units in the first line or in a `Units:` field.** An agent that writes
   a test needs the unit before it needs anything else.

6. **Give every public method a docstring.** A method with no docstring prints an
   empty purpose clause. That wastes the map line.

The map is generated, so it can never drift. These six rules decide whether it
is worth reading.

### 9.2 Keep names predictable

Section 3 helps the agent more than it helps the operator. The operator
autocompletes. The agent must guess.

An agent that has read `meas_vdc`, `meas_idc`, and `meas_resistance_4w` can
write `meas_vac` correctly without another read. Consistency across the eight
drivers means that one driver teaches all eight. Each rule you break costs the
agent a file read.

### 9.3 Put task recipes in the module docstring

Section 10 requires the five most common bench tasks in each module docstring.
The operator uses that list to find methods. An agent uses it as starting code.

Write each task as runnable lines. An agent that reads a ten-line recipe writes
a correct test without reading the class body.

```python
"""...
Common tasks:
  1. Constant-current discharge at 1 A with a 2.5 V cutoff:
         eload.set_load_current(1.0, v_limit=2.5)
         eload.set_output_state(True)
  2. Read the accumulated capacity:
         ah = eload.meas_capacity()
  ...
"""
```

### 9.4 Return typed dataclasses, not tuples

An agent that reads `-> SupplyState` and the dataclass definition learns every
field at once. An agent that reads `-> tuple` must read the method body to learn
the order and the meaning. Typed returns document themselves at no context cost.

Put the shared dataclasses in one module under `src/common/`. One read then
covers every driver.

**Give every `Reading` an `__iter__`, and no other return type one.** A
measurement result is the one place the operator wants two names on one line, so
`v, i = ch.meas_vi()` must work. Iterate the measured quantities in field order,
which is the order the `meas_` name spells: `meas_vi` yields two, `meas_vip`
yields three. The method name is then the unpacking contract, and it is visible
at the call site.

Do **not** reach for a `NamedTuple` to get the same thing. A `NamedTuple` fixes
the arity forever: the day a reading gains a field, every `v, i = ...` site
breaks. That is not hypothetical, because these types already differ in width
across drivers. A frozen dataclass with an explicit `__iter__` keeps the two
decisions separate: fields may grow, while what unpacking yields changes only
when someone edits `__iter__` on purpose. Pin the arity in a unit test.

The state objects get no `__iter__`. `SupplyState` and `SourceState` carry
nullable and mutually exclusive fields, so a positional read of one is a bug
waiting to happen.

### 9.5 Use enums so wrong values fail early

An agent guessing a string will write `"LOG"` or `"LOGARITHMIC"` or `"log"`. Two
of the three fail on the bench. `Spacing.LOG` fails at author time instead, or
it does not fail at all.

An enum turns a runtime bench error into an import error or an attribute error.
The agent finds it before the operator energizes anything.

### 9.6 Write error messages that name the fix

The exception message is the only feedback an agent gets. A message that names
the fix saves a debug cycle. A generic message costs one.

Good: `set_source_voltage() refused: CH1 output is live. Turn it off, or pass
allow_live=True.`

Bad: `RuntimeError: invalid state`.

The required keyword-only rule already gives good messages for free. Python
writes `missing 1 required keyword-only argument: 'i_limit'`.

### 9.7 Offer an offline link for test drafts

Sub-agents cannot reach the bench, so nobody checks an agent-drafted test until
the operator runs it. A stub link that replays recorded responses lets the agent
run its own draft and find its own mistakes.

This item gives real value and needs real work. Rank it after the six above.

### 9.8 Point agents at the map

An agent must learn that `docs/api-map.md` exists. Name it in `CLAUDE.md` and in
`.claude/agents/implementer.md`. Serena `find_symbol` with `include_body` is the
cheap second step for one method. Design docstrings so that one symbol read
answers the whole question. State units and valid ranges. Do not state SCPI
commands. See rule 1 in section 9.1.

---

## 10. Acceptance criteria for a driver

Replace command coverage with these six tests.

1. **The task list.** Put the five most common bench tasks in the module
   docstring. Each task takes two calls or fewer.
2. **Front-panel findability.** An operator who knows the panel action finds the
   method by autocomplete, without reading source.
3. **No stubs that raise.** Every method and node present works on the connected
   hardware.
4. **Every `set_` returns its read-back.**
5. **Minimal-write re-callability.** Verify it on a scope where output
   continuity matters.
6. **No uncited timing constants.** Every delay and ordering rule cites an
   observation or carries `# UNVERIFIED`.
7. **The verification harness uses the driver, never raw SCPI.** Raw SCPI
   belongs to capability probes, where the point is to learn what the instrument
   does. It never belongs to finding an instrument, which this repo does not do
   (ADR-051). A harness that reaches past the driver to `link.write` is not testing
   the driver, and it hides the gap it is working around. When a harness needs
   raw SCPI, that is the finding: the driver is missing a method, or an
   underscored one is not discoverable from the helper that needs it. Calling an
   underscored method from a harness is fine, because the underscore is only a
   completion tier (Section 5.3). Reaching past the driver entirely is not.
8. **No capability is lost in a reshape.** Diff the commands the old driver
   could send against the new one. Anything the new driver cannot reach at all
   is a regression, however tidy the new surface looks.

## 11. Enforcement

This standard has **no prefix linter** on purpose. These are guidelines that
need case-by-case judgment. A mechanical check would flag the justified
exceptions in section 5.3 as errors.

Three things enforce the standard instead.

1. This document. It states the reasons, so the reasoning transfers to a new
   instrument. A rule list alone gets pattern-matched and then worked around.
2. The reference driver. Something to copy beats something to comply with. Port
   one driver in full. Then measure the others against it.
3. Code review against section 10.
