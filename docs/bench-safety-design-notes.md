# Bench safety: design notes

**Status: ideas, not built.** Nothing in this file is implemented or decided.
It records a design direction for hardware protection, so later work starts
from it. Ratings named here are examples of the kind of limit. None of them is
a bench observation.

## 1. The problem

A script that drives a source or a load can damage the device under test, the
connection hardware, or another instrument. Scripts are now often written by AI
agents, which write fast and can be confidently wrong. Protection has to work
at many levels of effort: from a quick check by hand with no setup, to a fixed
test station where every connection is defined and checked.

## 2. Where the limit really is

The limit is not a property of the instrument or of the script. It is the
weakest point of the physical chain between a source and everything it reaches:

| Part of the chain | Examples of a limit |
| --- | --- |
| Device under test node | Pin maximum voltage, battery minimum voltage and maximum current |
| Connection hardware | Clip lead current, terminator power, probe voltage at frequency, relay switching rating |
| Receivers on the same node | Scope input at 50 ohm, active probe input, DMM range, electronic load rating |
| Ground links | A BNC shell or a probe ground clip that ties a node to earth |
| Instrument settings | Scope termination, probe attenuation, generator load setting, remote sense |
| Test intent | This step needs only 0 V to 1 V |

### Connection hardware

A bench connection is a chain of parts:

- **Cables:** coax, twisted pair, ribbon.
- **Leads:** banana, clip, mini-grabber, jumper, Kelvin.
- **Connectors and adapters:** BNC, SMA, triax, terminals, pogo pins, tees, BNC to banana.
- **Inline passive parts:** feedthrough terminators, attenuator pads, DC blocks, bias tees, dividers, shunts, filters.
- **Probes:** passive, high voltage, differential, active, current clamp.
- **Inline active parts:** amplifiers, low-noise amplifiers, isolation transformers.
- **Switching:** relay matrices, multiplexers, hand patching.
- **Fixtures and protection parts:** jigs, sockets, fuses, series resistors, clamps.

Each part has ratings, changes the signal (gain, loss, conversion, load), and
may tie the node to earth.

### Dimensions of a limit

Voltage between conductors, voltage to earth, current, power, frequency, time
and energy (pulses, inrush, inductive kick, stored charge), polarity, switching
state, and the match between an instrument setting and the hardware connected.

### Rules that follow

1. The chain limit is its weakest part, converted to the source's terms.
2. A source on a node must stay inside every receiver's rating at its present
   setting.
3. Each earth-referenced part fixes its node to earth. Two earth links on
   different nodes short them through ground.
4. A setting that disagrees with the connected hardware is a hazard on its own.
5. A relay state or a hand patch is a named wiring configuration with its own
   limits.
6. A call, a test, or a restore may narrow a limit. Only a change to the bench
   record may widen it.
7. A request outside a limit is refused with an error that names the node, the
   value, and the record. It is never clipped in silence.

## 3. Recording the bench, at any level of detail

| Level | Record | Gives |
| --- | --- | --- |
| 0 | Prose wiring comments | A person or an AI agent can read it. Nothing enforces it. |
| 1 | One cap line per output channel | Software refusal and the instrument's own protection |
| 2 | Named nodes with ratings, and channels bound to nodes | One limit shared by every instrument on a node |
| 3 | Chains of parts from a shared parts library, or a KiCad bench schematic that produces them | Weakest-part limits, conversions, earth links, receivers |
| 4 | Wiring checks: a small, safe signal from one named instrument must reach another named instrument | Proof that the record matches the cables today |

The parts library is like a KiCad symbol library. Each part type has a name,
its ratings per dimension, how it changes the signal, and an earth flag. A bench
record lists chains of library parts from each instrument terminal to a device
node, plus the receivers on that node. Every structured record carries the date
it was confirmed and who confirmed it.

## 4. One global safety level

A single setting decides how much of the above is enforced. It runs from off,
for a quick check by hand, to maximum, where every layer must be defined.

| Level | Name | Behavior |
| --- | --- | --- |
| 0 | `off` | No bench checks, no report, no limit programming. The instruments behave as the bare drivers do. |
| 1 | `report` | Read whatever records exist. At connect, report what protects each channel and what is missing. Block nothing. |
| 2 | `enforce` | Enforce every limit that is recorded. Program the instruments' own protection and read it back. Refuse a setting that disagrees with a recorded part. Allow what is not recorded, with a warning. |
| 3 | `strict` | Level 2, and a channel with no record cannot energize. A raw instrument command needs an explicit escape with a reason, and the run log keeps it. |
| 4 | `maximum` | Level 3, and every layer must be fully defined before anything energizes. |

### What "fully defined" means at `maximum`

A check of the records, run before any connection, in the same spirit as a
KiCad electrical rules check:

1. Every output channel has a chain to a named node.
2. Every part in every chain comes from the parts library, with a rating in
   each enforced dimension.
3. Every receiver on every energized node is recorded, with its rating at its
   expected setting.
4. Every part says whether it ties to earth. No two earth links short two
   different nodes.
5. The active wiring configuration is named, and relay or patch states match it.
6. Every record was confirmed within a set number of days.
7. The wiring checks pass in this session.
8. A dry run against simulated instruments produced an energy plan, the highest
   level each channel will see, and a person approved it.
9. A watchdog runs during the live run and turns outputs off on an excursion.

The check reports every gap at once, so the person can close them in one pass.

### Where the level is set, and who may lower it

1. **The agent environment sets a floor.** For example, an environment setting
   that an AI agent's configuration sets to `strict`. A script run by the agent
   cannot go below the floor.
2. **The bench record sets a default.** A fixed test station can default to
   `maximum`.
3. **The script may raise the level.** It may lower the level only down to the
   floor.
4. **A person at a notebook may choose `off`** when no floor is set.
5. **A single channel may get an exception** with a written reason. The run log
   records it.

What stays on at every level, because it is part of the driver interface, not
a check: outputs off at disconnect, and the limit arguments that the driver
standard makes required on power calls.

## 5. Safety layers, outside in

| Layer | Example | Holds when |
| --- | --- | --- |
| Physical | Fuses, series resistors, clamps, interlocks | All software fails |
| Instrument | Overvoltage, overcurrent, compliance, output clamps | Python stops, a knob turns |
| Driver | Required limits, refusal before the write, outputs off at disconnect | A call asks for too much |
| Bench | Recorded limits, narrow-only rule, setting match | A script and the wiring disagree |
| Run | Dry run, energy plan, watchdog | A script is wrong across many calls |
| Agent | Rules for AI agents, a permission prompt before an energizing script | An AI agent writes or runs the script |

## 6. Python shape, in outline

- `common/limits.py`: a limit record per dimension, with intersect, contains,
  and convert-through-gain.
- A parts library file, and chain records in each bench file. A KiCad netlist
  reader can produce the same chains.
- A `Bench` object: the instrument handles, the records, the safety level, the
  check of the records, the connect report, and the run log.
- Each output channel: bind to its chain, program its own protection and read
  it back at connect, check every setter before the write, refuse a restore
  that would widen a limit.
- Each input channel: report its rating at its present setting, so a source
  can respect it.
- `bench_check`: runs the check of the records with no instrument I/O, then the
  address checks, then the wiring checks.

## 7. First steps, when work starts

1. A level 1 cap line per channel, and the connect report.
2. The global level with `off`, `report` and `enforce`.
3. Setting match for scope termination, probe attenuation and generator load.
4. `strict` items: no energize without a record, and the raw-command escape.
5. Then the parts library, chains, and `maximum`.
