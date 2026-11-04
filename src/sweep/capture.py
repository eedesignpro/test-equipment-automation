"""Capture control: how long and how fast to record at each frequency.

A rule is a number, a function of frequency, or a table of (frequency, value)
pairs that is interpolated on log axes. :class:`CaptureRules` holds the rules,
and :func:`plan_capture` turns them into a sample rate and a record length for
one frequency. The receiver then snaps those to what the instrument allows and
reports the values it actually used.

Typical trade-offs, all adjustable:

- At low frequency a few cycles already take a long time, so a minimum cycle
  count or a resolution bandwidth sets the record length.
- At high frequency many cycles fit in a short record, so a narrower
  resolution bandwidth costs little and rejects more noise and spurs.
- Memory and sample rate limits cap the record; the plan reports when a limit
  cut the record short.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence, Union

import numpy as np

Rule = Union[float, int, None, Callable[[float], float], Sequence[tuple[float, float]]]


def rule_value(rule: Rule, freq: float) -> float | None:
    """Evaluate a rule at ``freq``. A table interpolates log-log and holds its ends."""
    if rule is None:
        return None
    if callable(rule):
        return float(rule(freq))
    if isinstance(rule, (int, float)):
        return float(rule)
    pts = sorted((float(f), float(v)) for f, v in rule)
    fs = np.log10([p[0] for p in pts])
    vs = [p[1] for p in pts]
    if all(v > 0 for v in vs):
        return float(10 ** np.interp(math.log10(freq), fs, np.log10(vs)))
    return float(np.interp(math.log10(freq), fs, vs))


@dataclass
class CaptureRules:
    """Rules for the record at each frequency. Every field is a :data:`Rule`.

    ``min_cycles``: cycles of the drive frequency the record must hold.
    ``rbw``: resolution bandwidth target in Hz (record length = 1/rbw).
    ``rbw_fraction``: resolution bandwidth as a fraction of the frequency.
    ``min_samples_per_cycle``: oversampling of the drive frequency.
    ``harmonics``: highest harmonic that must stay below Nyquist.
    ``min_record_length``: raise the sample rate until the record holds at
    least this many points; this averages noise down at no cost in time.
    ``max_record_length``, ``max_sample_rate``, ``min_sample_rate``,
    ``max_duration``: instrument or patience limits.
    ``window``: the analysis window name, or a function of frequency that
    returns one.
    """

    min_cycles: Rule = 20
    rbw: Rule = None
    rbw_fraction: Rule = 0.01
    min_samples_per_cycle: Rule = 20
    harmonics: Rule = 5
    min_record_length: Rule = 20_000
    max_record_length: Rule = 1_000_000
    max_sample_rate: Rule = None
    min_sample_rate: Rule = None
    max_duration: Rule = None
    window: Union[str, Callable[[float], str]] = "blackman"

    def window_at(self, freq: float) -> str:
        return self.window(freq) if callable(self.window) else self.window


@dataclass
class CapturePlan:
    """What the rules ask for at one frequency, and why."""

    freq: float
    sample_rate: float
    record_length: int
    duration: float
    cycles: float
    rbw: float
    window: str
    limited_by: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def plan_capture(freq: float, rules: CaptureRules | None = None) -> CapturePlan:
    """Sample rate and record length for ``freq`` under ``rules``."""
    r = rules or CaptureRules()
    v = lambda rule: rule_value(rule, freq)  # noqa: E731
    limited = []

    duration = 0.0
    if v(r.min_cycles):
        duration = max(duration, v(r.min_cycles) / freq)
    if v(r.rbw):
        duration = max(duration, 1.0 / v(r.rbw))
    if v(r.rbw_fraction):
        duration = max(duration, 1.0 / (v(r.rbw_fraction) * freq))
    if duration <= 0:
        duration = 10.0 / freq
    if v(r.max_duration) and duration > v(r.max_duration):
        duration = v(r.max_duration)
        limited.append("max_duration")

    sr = freq * (v(r.min_samples_per_cycle) or 10.0)
    h = v(r.harmonics)
    if h:
        sr = max(sr, 2.5 * freq * (h + 1))
    if v(r.min_sample_rate):
        sr = max(sr, v(r.min_sample_rate))
    if v(r.max_sample_rate) and sr > v(r.max_sample_rate):
        sr = v(r.max_sample_rate)
        limited.append("max_sample_rate")

    min_n = v(r.min_record_length)
    if min_n and sr * duration < min_n:
        want = min_n / duration
        cap = v(r.max_sample_rate)
        sr = min(want, cap) if cap else want
    n = int(math.ceil(sr * duration))
    max_n = v(r.max_record_length)
    spc = v(r.min_samples_per_cycle) or 10.0
    cap = v(r.max_sample_rate)
    if max_n and n > max_n:
        floor_sr = max(2.5 * freq * ((h or 0) + 1), freq * spc)
        if cap:
            floor_sr = min(floor_sr, cap)
        sr = max(floor_sr, max_n / duration)
        if cap:
            sr = min(sr, cap)
        n = int(math.ceil(sr * duration))
        if n > max_n:
            duration = max_n / sr
            n = int(max_n)
            limited.append("max_record_length")
    if sr < freq * spc * (1 - 1e-9) and "min_samples_per_cycle" not in limited:
        limited.append("min_samples_per_cycle")
    return CapturePlan(
        freq=freq,
        sample_rate=sr,
        record_length=n,
        duration=n / sr,
        cycles=freq * n / sr,
        rbw=sr / n,
        window=r.window_at(freq),
        limited_by=limited,
    )


# Read-back keys a time-domain receiver reports in ``Capture.settings``. The
# first two are required; the scope ones are checked when present.
READBACK_KEYS = (
    "sample_rate",
    "record_length",
    "time_scale",
    "div_count",
    "horizontal_mode",
    "points_transferred",
)


def validate_capture(
    plan: CapturePlan, settings: dict, *, harmonics: int = 0, dt: float | None = None
) -> tuple[list[str], bool]:
    """Check the acquisition the instrument actually made against the plan.

    Returns (flags, invalid). ``invalid`` means the record cannot give a
    trustworthy value at this frequency: missing read-back, a sample rate
    at or below twice the drive frequency, less than one cycle, a horizontal
    setting that does not hold together, or fewer points than the instrument
    reported. The other flags mark a capture that differs from the plan but
    still holds the tone.
    """
    flags: list[str] = []
    invalid = False
    f = plan.freq
    sr = settings.get("sample_rate")
    rl = settings.get("record_length")
    for key in ("sample_rate", "record_length"):
        if settings.get(key) in (None, "") or not np.isfinite(float(settings.get(key) or "nan")):
            flags.append(f"readback_missing_{key}")
            invalid = True
    if invalid:
        return flags, True
    sr, rl = float(sr), float(rl)
    if sr <= 2.0 * f:
        flags.append("sample_rate_invalid")
        invalid = True
    elif harmonics and sr < 2.0 * f * (harmonics + 1):
        flags.append("harmonics_above_nyquist")
    cycles = f * rl / sr
    if cycles < 1.0:
        flags.append("less_than_one_cycle")
        invalid = True
    elif cycles < 0.5 * plan.cycles:
        flags.append("fewer_cycles_than_plan")
    if rl < 0.5 * plan.record_length:
        flags.append("record_shorter_than_plan")
    if sr < 0.5 * plan.sample_rate:
        flags.append("sample_rate_below_plan")
    scale, divs = settings.get("time_scale"), settings.get("div_count")
    if scale and divs:
        span = float(scale) * float(divs)
        if abs(span - rl / sr) > 0.02 * (rl / sr):
            flags.append("horizontal_inconsistent")
            invalid = True
    mode = settings.get("horizontal_mode")
    if mode is not None and "MAN" not in str(mode).upper():
        flags.append("horizontal_not_manual")
        invalid = True
    pts = settings.get("points_transferred")
    if pts is not None and int(pts) < int(rl):
        flags.append("transfer_short")
        invalid = True
    if dt:
        if abs(1.0 / dt - sr) > 1e-3 * sr:
            flags.append("time_axis_disagrees")
            invalid = True
    return flags, invalid
