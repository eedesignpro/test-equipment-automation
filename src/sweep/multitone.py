"""Several tones at once: one capture measures a whole group of frequencies.

The frequency list is split into groups of neighbouring frequencies. Each
group's tones are moved to exact multiples of one base frequency, so a source
can play them as one periodic waveform (an arbitrary waveform with Schroeder
phases, which keeps the crest factor low). One capture then holds every tone
of the group, and the ordinary sweep loop analyzes each tone as its own point,
with the other tones of the group blanked from its noise estimate. Plots,
reviews and fits are the same as for a one-tone sweep.

A source takes part when it has ``set_multitone(freqs, amplitude, base)``;
the 33500 source and the simulated bench have one. ``amplitude`` is the peak
to peak of the summed waveform, so the source's own limit still applies.
"""

from __future__ import annotations

import math

import numpy as np

from sweep.capture import CapturePlan, CaptureRules, plan_capture
from sweep.run import SweepResult, run_sweep


def schroeder_phases(n: int) -> np.ndarray:
    """Phases that keep the peak of a sum of ``n`` equal tones low."""
    k = np.arange(1, n + 1)
    return -np.pi * k * (k - 1) / n


def multitone_samples(harmonics: list[int], points: int) -> np.ndarray:
    """One period of a sum of cosines at the given multiples of the base, scaled to +-1."""
    t = np.arange(points) / points
    ph = schroeder_phases(len(harmonics))
    x = sum(np.cos(2 * np.pi * h * t + p) for h, p in zip(harmonics, ph))
    return x / np.max(np.abs(x))


def make_groups(freqs, tones_per_capture: int, base_divisor: int = 200):
    """Split ``freqs`` into groups and snap each group onto multiples of a base.

    Returns a list of (base frequency, snapped tone frequencies). The base is
    the group's lowest frequency over ``base_divisor``, so the snap moves a
    tone by at most half a base step, under 0.25 % for the default.
    """
    fs = sorted(float(f) for f in freqs)
    groups = []
    for i in range(0, len(fs), tones_per_capture):
        g = fs[i : i + tones_per_capture]
        base = g[0] / base_divisor
        snapped = sorted({max(1, round(f / base)) * base for f in g})
        groups.append((base, snapped))
    return groups


class _GroupSource:
    """Plays a whole group when its first tone is asked for; later tones cost nothing."""

    def __init__(self, inner, groups):
        self.inner = inner
        self.name = f"{getattr(inner, 'name', 'source')}+multitone"
        self._group_of = {f: (base, tones) for base, tones in groups for f in tones}
        self._key_of = {f: tuple(tones) for _, tones in groups for f in tones}
        self._coherent: dict = {}
        self._playing = None
        self._last: dict = {}

    def open(self):
        if hasattr(self.inner, "open"):
            self.inner.open()

    def close(self):
        if hasattr(self.inner, "close"):
            self.inner.close()

    def output(self, state):
        if hasattr(self.inner, "output"):
            return self.inner.output(state)
        return None

    def coherent_freq(self, freq, sample_rate, record_length):
        """Move ``freq``'s whole group onto bins of the record the receiver will take.

        The new base is the largest multiple of the bin (sample rate / record
        length) at or under the group's own base, so the record holds a whole
        number of waveform periods and every tone sits exactly on a bin.
        Returns ``freq``'s new frequency.
        """
        if freq in self._coherent:
            return freq
        base, tones = self._group_of[freq]
        bin_hz = float(sample_rate) / int(record_length)
        base_c = bin_hz * max(1, math.floor(base / bin_hz + 1e-9))
        moved = [max(1, round(t / base_c)) * base_c for t in tones]
        if len(set(moved)) < len(moved):
            raise ValueError(
                f"two tones of one group land on one bin ({bin_hz:g} Hz): the record is too "
                "short for coherent multitone; lengthen the capture"
            )
        key = tuple(tones)
        for t, m in zip(tones, moved):
            self._coherent[m] = (base_c, moved, key)
        self._key_of.update({m: key for m in moved})
        return moved[tones.index(freq)]

    def group_key(self, freq):
        return self._key_of[freq]

    def other_tones(self, freq):
        if freq in self._coherent:
            return [t for t in self._coherent[freq][1] if t != freq]
        return [t for t in self._group_of[freq][1] if t != freq]

    def set(self, freq, amplitude):
        if freq in self._coherent:
            base, tones, _ = self._coherent[freq]
        else:
            base, tones = self._group_of[freq]
        key = tuple(tones)
        if self._playing != key:
            self._last = self.inner.set_multitone(tones, amplitude, base) or {}
            self._playing = key
        return {"freq": freq, "amplitude": self._last.get("amplitude", amplitude)}

    def describe(self):
        d = dict(self.inner.describe()) if hasattr(self.inner, "describe") else {}
        d["multitone_groups"] = len({tuple(t) for _, t in self._group_of.values()})
        return d


class _GroupReceiver:
    """Captures once per group and hands the same capture to every tone of it."""

    def __init__(self, inner, groups, plan_for_group, group_key=None):
        self.inner = inner
        self.name = getattr(inner, "name", "measurement")
        self._group_of = {f: tuple(tones) for _, tones in groups for f in tones}
        self._group_key = group_key or self._group_of.__getitem__
        self._plan_for_group = plan_for_group
        self._cached = None
        self._last_freq = None
        self._prepared = None

    def open(self):
        if hasattr(self.inner, "open"):
            self.inner.open()

    def close(self):
        if hasattr(self.inner, "close"):
            self.inner.close()

    def measure(self, freq, plan):
        key = self._group_key(freq)
        # The same tone asked for again is a retry of an invalid capture:
        # capture the group again instead of handing back the same record.
        again = self._last_freq == freq
        if again or self._cached is None or self._cached[0] != key:
            self._cached = (key, self.inner.measure(freq, self._plan_for_group(key)))
        self._last_freq = freq
        return self._cached[1]

    def _prepare_horizontal(self, plan):
        # Once per group: later tones reuse the group's capture, so preparing
        # again would leave a one-shot setting behind for the next group.
        if self._prepared is None or self._prepared[0] != plan:
            self._prepared = (plan, self.inner.prepare_horizontal(plan))
        return self._prepared[1]

    def describe(self):
        return self.inner.describe() if hasattr(self.inner, "describe") else {"name": self.name}

    def __getattr__(self, name):
        if name == "prepare_horizontal" and hasattr(self.inner, "prepare_horizontal"):
            return self._prepare_horizontal
        if name == "after_point":
            hook = getattr(self.inner, "after_point", None)
            if hook is None:
                raise AttributeError(name)
            return hook
        raise AttributeError(name)


def group_plan(tones, rules: CaptureRules | None = None) -> CapturePlan:
    """A capture that resolves the group: duration from the lowest tone, rate from the highest."""
    lo = plan_capture(min(tones), rules)
    hi = plan_capture(max(tones), rules)
    spacing = min(b / a for a, b in zip(tones, tones[1:])) - 1 if len(tones) > 1 else 1.0
    # Neighbouring tones need to sit outside each other's window main lobe:
    # 2 x the main-lobe half width in bins (6 bins for Blackman).
    from sweep.analysis import _MAINLOBE_BINS

    sep_bins = 2 * _MAINLOBE_BINS.get(lo.window, 3)
    duration = max(lo.duration, sep_bins / (spacing * min(tones)))
    sr = max(lo.sample_rate, hi.sample_rate)
    n = int(math.ceil(sr * duration))
    limited = list(lo.limited_by)
    cap = (rules or CaptureRules()).max_record_length
    if isinstance(cap, (int, float)) and n > cap:
        n = int(cap)
        limited.append("max_record_length")
    return CapturePlan(min(tones), sr, n, n / sr, min(tones) * n / sr, sr / n, lo.window, limited)


def run_multitone_sweep(
    freqs,
    source,
    measurements,
    *,
    tones_per_capture: int = 8,
    base_divisor: int = 200,
    capture: CaptureRules | None = None,
    **kw,
) -> SweepResult:
    """A sweep that plays ``tones_per_capture`` tones at once.

    Takes the same keywords as :func:`sweep.run.run_sweep`. Frequencies are
    snapped onto each group's base (under 0.25 % by default) and the snapped
    values are what the run records. Harmonics default to 0, because the
    harmonics of one tone can land on another tone of the group.

    With ``coherent=True`` (locked clocks, a receiver with
    ``prepare_horizontal``) each group is moved once more, onto bins of the
    record the receiver takes, and the waveform period divides the record.
    """
    for opt in ("repeats", "settle_captures"):
        if (kw.get(opt) or 0) > (1 if opt == "repeats" else 0):
            raise ValueError(
                f"{opt} is not supported by a multitone sweep: every tone of a group "
                "reads one shared capture, so the extra captures would not happen"
            )
    groups = make_groups(freqs, tones_per_capture, base_divisor)
    snapped = [f for _, tones in groups for f in tones]
    src = _GroupSource(source, groups)
    meas = [
        _GroupReceiver(m, groups, lambda key: group_plan(list(key), capture), src.group_key)
        for m in measurements
    ]
    kw.setdefault("harmonics", 0)
    kw.setdefault("title", "multitone_sweep")
    return run_sweep(
        snapped,
        src,
        meas,
        tones=src.other_tones,
        capture=lambda f: group_plan(list(src.group_key(f)), capture),
        _extra_meta={
            "multitone": {
                "tones_per_capture": tones_per_capture,
                "base_divisor": base_divisor,
                "groups": len(groups),
            }
        },
        **kw,
    )
