"""A simulated bench: a source and a receiver that need no hardware.

Use it to rehearse a sweep, to test capture rules, and in unit tests. The
device is a set of named responses ``H(f)`` from the drive to each signal.
The receiver can add noise, a clock offset and drift between source and
receiver, phase jitter, harmonics, amplitude modulation, a fixed spur, and
clipping, so every flag and estimate in the toolkit can be exercised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from sweep.capture import CapturePlan
from sweep.interfaces import Capture


@dataclass
class SimulatedBench:
    """Holds the drive state shared by :meth:`source` and :meth:`receiver`."""

    responses: dict[str, Callable[[float], complex]]
    units: dict[str, str] = field(default_factory=dict)
    noise_rms: float | dict = 1e-4
    clock_ppm: float = 0.0
    drift_ppm_per_s: float = 0.0
    jitter_rad: float = 0.0
    harmonic_levels: dict[int, float] = field(default_factory=dict)
    am_depth: float = 0.0
    am_rate_hz: float = 0.0
    spur: tuple[float, float] | None = None
    clip_at: float | None = None
    sample_clock_hz: float | None = None
    seed: int = 1
    freq: float = 0.0
    amplitude: float = 1.0
    output_on: bool = False
    log: list = field(default_factory=list)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def source(self):
        bench = self

        class _Src:
            name = "simulated_source"

            def set(self, freq, amplitude):
                bench.tones = None
                bench.freq = float(freq)
                bench.amplitude = 1.0 if amplitude is None else float(amplitude)
                bench.log.append(("set", freq, amplitude))
                return {"freq": bench.freq, "amplitude": bench.amplitude}

            def set_multitone(self, freqs, amplitude, base):
                bench.tones = [float(f) for f in freqs]
                bench.amplitude = float(amplitude)
                bench.log.append(("multitone", tuple(bench.tones), amplitude))
                return {"amplitude": bench.amplitude}

            def output(self, state):
                bench.output_on = bool(state)
                bench.log.append(("output", bool(state)))

            def describe(self):
                return {"class": "SimulatedSource"}

        return _Src()

    def receiver(self, name: str = "sim_scope"):
        bench = self

        class _Rx:
            def __init__(self):
                self.name = name
                self._prepared = None

            def prepare_horizontal(self, plan: CapturePlan):
                sr = plan.sample_rate
                if bench.sample_clock_hz:
                    # A scope divides one sample clock: the rate snaps up to clock / k.
                    sr = bench.sample_clock_hz / max(1, math.floor(bench.sample_clock_hz / sr))
                self._prepared = (float(sr), int(plan.record_length))
                return self._prepared

            def measure(self, freq, plan: CapturePlan):
                if self._prepared is not None:
                    sr, rl = self._prepared
                    self._prepared = None
                    plan = replace(plan, sample_rate=sr, record_length=rl)
                return bench.capture(plan)

            def describe(self):
                return {"class": "SimulatedReceiver", "signals": list(bench.responses)}

        return _Rx()

    def capture(self, plan: CapturePlan) -> Capture:
        n = int(plan.record_length)
        dt = 1.0 / plan.sample_rate
        t = np.arange(n) * dt
        f_true = self.freq * (1 + self.clock_ppm * 1e-6)
        inst = f_true * (1 + self.drift_ppm_per_s * 1e-6 * t)
        phase = 2 * np.pi * np.cumsum(inst) * dt
        if self.jitter_rad:
            phase = phase + self._rng.normal(0, self.jitter_rad, n)
        env = 1 + self.am_depth * np.sin(2 * np.pi * self.am_rate_hz * t)
        a = self.amplitude / 2 if self.output_on else 0.0
        if getattr(self, "tones", None):
            from sweep.multitone import schroeder_phases

            tones = self.tones
            ph0 = schroeder_phases(len(tones))
            base = min(tones) / 200
            tt = np.arange(4096) / 4096
            # Scale like the source does: the summed waveform peaks at the set amplitude.
            peak = np.max(
                np.abs(
                    sum(np.cos(2 * np.pi * round(f / base) * tt + p) for f, p in zip(tones, ph0))
                )
            )
            sigs = {}
            for name, h in self.responses.items():
                x = np.zeros(n)
                for f, p in zip(tones, ph0):
                    hv = complex(h(f * (1 + self.clock_ppm * 1e-6)))
                    x = x + a / peak * abs(hv) * np.cos(
                        2 * np.pi * f * (1 + self.clock_ppm * 1e-6) * t + p + np.angle(hv)
                    )
                nr = (
                    self.noise_rms.get(name, 0.0)
                    if isinstance(self.noise_rms, dict)
                    else self.noise_rms
                )
                sigs[name] = x + self._rng.normal(0, nr, n)
            return Capture(
                signals=sigs,
                t=t,
                units={k: self.units.get(k, "V") for k in sigs},
                group="sim",
                settings={"sample_rate": plan.sample_rate, "record_length": n},
            )
        sigs = {}
        for name, h in self.responses.items():
            hv = complex(h(f_true))
            x = a * abs(hv) * env * np.cos(phase + np.angle(hv))
            for k, lev in self.harmonic_levels.items():
                hk = complex(h(k * f_true))
                x = x + a * lev * abs(hk) * np.cos(k * phase + np.angle(hk))
            if self.spur:
                fs, amp = self.spur
                x = x + amp * np.cos(2 * np.pi * fs * t)
            nr = (
                self.noise_rms.get(name, 0.0)
                if isinstance(self.noise_rms, dict)
                else self.noise_rms
            )
            x = x + self._rng.normal(0, nr, n)
            if self.clip_at is not None:
                x = np.clip(x, -self.clip_at, self.clip_at)
            sigs[name] = x
        return Capture(
            signals=sigs,
            t=t,
            units={k: self.units.get(k, "V") for k in sigs},
            group="sim",
            settings={"sample_rate": plan.sample_rate, "record_length": n},
        )


def rc_lowpass(r: float, c: float) -> Callable[[float], complex]:
    return lambda f: 1 / (1 + 2j * math.pi * f * r * c)


def series_rl(r: float, inductance: float) -> Callable[[float], complex]:
    return lambda f: r + 2j * math.pi * f * inductance


def resonance(f0: float, q: float, gain: float = 1.0) -> Callable[[float], complex]:
    def h(f):
        s = 1j * f / f0
        return gain / (s**2 + s / q + 1)

    return h


def delay(seconds: float, h: Callable[[float], complex] | None = None):
    base = h or (lambda f: 1.0)
    return lambda f: complex(base(f)) * np.exp(-2j * math.pi * f * seconds)
