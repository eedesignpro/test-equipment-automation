# These drivers control hardware. Read this before you design anything.

Written for the AI assistants that work in this repo, because the same wrong
turns keep happening several times a session. A human maintainer may find it
useful too.

Everything here has one root: **a driver object is a handle on a physical
instrument that exists whether your program is running or not.** Habits that
are correct in ordinary software are wrong here, not because they are bad
habits, but because they solve problems this domain does not have and create
problems it does.

There are no blanket rules below. The instruction is always the same shape:
**find out what this instrument actually is, then act on that.** A rule that
fits the eight instruments on this bench may be wrong for the ninth.

---

## 1. Learn the specific instrument before you generalize

Two instruments in the same category are not the same hardware. The E3646A is a
dual-output supply with ONE physical output enable for both channels. The
ODP6033 is a triple-output supply where every channel switches independently.
A driver that assumes "dual-output PSU" gets one of them wrong.

Do not carry a shape from one instrument to another because the category
matches. Carry it when the hardware matches.

## 2. Match the object graph to the hardware's real independence

The 33522A has two channels that share a chassis, a timebase, a trigger system
and a Sync connector. They are not independent instruments, and the driver
should not pretend they are.

An N-channel electronic load rack with genuinely independent slots is a
different machine, and there a much more self-contained per-slot object would
be right. Neither answer is universal. Look at what is actually shared.

The corollary for API design: an operator working with an instrument has the
whole instrument in scope. Do not design channel objects to be self-sufficient
against the possibility of being separated from their parent. That is a
generic-software concern about objects being passed between modules; it does
not describe how a bench is driven, and paying for it in verbosity and
autocomplete noise is a real cost for an imaginary benefit.

## 3. The instrument's state is shared, persistent, and not yours

The front panel is live. Another session may hold the same instrument. Settings
survive your program exiting, and the next run inherits them.

So: do not cache instrument state in Python. Query it. A cached value is a
correctness bug waiting for someone to touch the front panel, not an
optimization.

And: assert whatever your measurement depends on rather than inheriting it. A
previous session left the scope in 10 s/div roll mode, which made every
measurement fail to populate. Another left a coarse vertical, which made a
frequency measurement report 37 kHz for a 1 kHz sine. Neither raised an error.

## 4. Every state change is a physical act, including the ones on exit

There is no blanket right answer for what to leave behind. Restoring what you
found can be exactly wrong: it can re-energize an output after the DUT has been
disconnected, or put back a voltage that is only safe under conditions that no
longer hold. Leaving things where the test ended can be equally wrong.

The script and the bench setup decide. What is required is that the decision is
**deliberate and stated**, not incidental. If a fixture leaves an output live,
say so and say why. If it de-energizes, same. Never let exit behavior be an
accident of whatever the last line of code happened to do.

Note which instruments cannot be de-energized at all. The 33120A has no output
relay: its output is always live, and "turning it off" is not available.

## 5. You cannot infer hardware behavior. Measure it.

No amount of reasoning tells you how long a relay settles, whether a same-value
write costs anything, whether a mode change clears a level register, or whether
a command has a side effect the manual does not mention. Text written about
these without an observation is generated plausibility, and this repo has paid
for that error more than once.

Cite an observation, or mark it `# UNVERIFIED` and let no helper depend on it.

Three real examples of things that could not have been reasoned out:

- `APPLy` on the 33522A closes the output relay as a side effect. Nine driver
  methods were built on it before anyone checked.
- A query costs about 4.8 ms on the 33522A against a 1.5 ms write. On the
  MSO58 a query is 0.74 ms against a 135 ms write. The ratio is inverted
  between two instruments on the same bench, so no intuition transfers.
- Deciding whether a value needs rewriting needed the instrument's own storage
  precision, per quantity. A guessed tolerance was 30x too tight and would have
  rewritten everything on every call.

## 6. Absence is information

If the hardware cannot do something, the method should not exist. A method that
exists and raises tells the completion popup a lie, and the operator finds out
at the bench. The 33120A gets no `set_output_state` because it has no output
relay. A single-channel model gets no `ch2`.

## 7. Errors are physical events, and recovery is instrument-specific

A timeout can mean several different things: a genuinely hung link, a deaf
window while the instrument re-ranges, an undefined header the instrument is
ignoring, or an operation that is simply slower than the timeout. These have
different recoveries and only observation distinguishes them.

Do not retry-hammer a silent link. Equally, do not treat one timeout as proof
the instrument is dead and stop to ask for help that may not be needed. Work
out what this instrument does in this situation; a VISA clear plus `*CLS` plus
an error drain recovers several of these cases.

## 8. A test that passes may be blind

Verification code has the same failure mode as driver code: it can be
confidently wrong. Before trusting that a detector did not fire, prove that it
fires when it should.

Two glitch detectors written for the 33522A passed every check while being
completely blind. A timeout trigger cannot distinguish a sine from a square of
the same frequency and amplitude. Period statistics never record a gap, because
while the output is absent the measurement simply stops populating. Both
reported clean runs across a deliberately interrupted output.

## 9. Units are physical

Every public argument and return is in the unprefixed unit: volts, amperes,
ohms, watts, seconds, hertz. Convert at the driver boundary when the instrument
speaks something else. An instrument that takes milliseconds gets converted, not
exposed.

## 10. Hardware work is main-agent work here

Sub-agents can technically reach the bench. They are not allowed to, for an
operational reason: if an instrument hangs mid-task, a sub-agent cannot be
interrupted without losing all of its progress. So a sub-agent that needs a
hardware fact writes the probe and stops; the main agent runs it.

---

## The short version

Ask what the hardware actually does before deciding what the code should look
like. When you cannot answer from observation, say so in the code rather than
writing something plausible.
