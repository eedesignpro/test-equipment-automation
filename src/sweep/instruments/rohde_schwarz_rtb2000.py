"""Rohde & Schwarz RTB2000 as a time-domain receiver, and its built-in generator as a source."""

from __future__ import annotations

import bisect
import dataclasses

import numpy as np

from sweep.capture import CapturePlan
from sweep.instruments.base import Preserved, check_amplitude, identity
from sweep.interfaces import Capture

# The record covers the 12 screen divisions (driver quirk RTB-04).
_HORIZONTAL_DIVS = 12.0
# UNVERIFIED: the ACQ:POIN values the scope accepts. The driver states only
# "10 k to 20 M predefined values"; this 1-2-5 list is not read off the bench.
RECORD_LENGTHS = (
    10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000,
)  # fmt: skip
# UNVERIFIED: the screen spans 5 divisions either side of the offset and the
# ADC limit sits at the screen edge. The driver's clip heuristic (4.9 div from
# the offset) implies the 5; no bench observation confirms the ADC limit.
_HALF_DIVS = 5.0


def nearest_record_length(points: int, choices=RECORD_LENGTHS) -> int:
    """The smallest choice at or above ``points``, or the largest choice."""
    ordered = sorted(int(c) for c in choices)
    i = bisect.bisect_left(ordered, int(points))
    return ordered[min(i, len(ordered) - 1)]


class RohdeSchwarzRtb2000Receiver(Preserved):
    """Capture named scope channels in one acquisition at each frequency.

    ``channels`` maps a signal name to a channel, for example
    ``{"V_in": "ch1", "V_out": "ch2"}``. At each frequency the time scale is
    the plan duration over the 12 screen divisions, and the record length is
    the smallest of ``record_lengths`` at or above the plan's. The values the
    scope actually used are recorded. ``trigger`` names a channel or ``"ext"``
    for an edge trigger; the trigger mode is AUTO so an acquisition always
    completes. Each signal gets a full-scale pair from the channel's scale and
    offset for the toolkit's clip check; the driver has no non-raising clip
    flag, so ``clipped`` stays empty. The scope settings found on open are
    restored on close.
    """

    def __init__(
        self,
        scope,
        channels: dict[str, str],
        *,
        trigger: str | None = None,
        trigger_level: float | None = None,
        record_lengths=RECORD_LENGTHS,
        timeout_s: float = 20.0,
        preserve: bool = True,
        name: str | None = None,
    ):
        super().__init__(scope, preserve=preserve)
        self.scope = scope
        self.channels = {k: v.lower() for k, v in channels.items()}
        self.trigger = trigger.lower() if trigger else None
        self.trigger_level = trigger_level
        self.record_lengths = tuple(record_lengths)
        self.timeout_s = timeout_s
        self.name = name or "scope"

    def _ch(self, source: str):
        return getattr(self.scope, source)

    def open(self) -> None:
        super().open()
        if self.trigger:
            self.scope.trig.set_edge(self.trigger, level=self.trigger_level, mode="AUTO")
        else:
            self.scope.trig.set_mode("AUTO")
        for src in self.channels.values():
            self._ch(src).set_display_state(True)

    def measure(self, freq: float, plan: CapturePlan) -> Capture:
        horz = self.scope.horz.set_timebase(plan.duration / _HORIZONTAL_DIVS)
        want = nearest_record_length(plan.record_length, self.record_lengths)
        rl = self.scope.acq.set_record_length(want)
        self.scope.cmd_single()
        self.scope.cmd_wait_acquisition(timeout=self.timeout_s + 3.0 * horz.range)
        sources = list(self.channels.values())
        cap = self.scope.data.get_waveform(sources, check_clip=False)
        t = np.asarray(cap.t, dtype=float)
        n = min(len(t), *(len(cap.channels[s]) for s in sources))
        signals = {
            k: np.asarray(cap.channels[s][:n], dtype=float) for k, s in self.channels.items()
        }
        full_scale, scales = {}, {}
        for k, s in self.channels.items():
            c = cap.settings.channels[s]
            scales[k] = c.scale
            # UNVERIFIED best guess: a position of +p divisions moves the trace up p
            # divisions, so the screen spans offset - (5 + p) to offset + (5 - p) div.
            pos = float(getattr(c, "position", 0.0) or 0.0)
            full_scale[k] = (
                c.offset - (_HALF_DIVS + pos) * c.scale,
                c.offset + (_HALF_DIVS - pos) * c.scale,
            )
        sr_axis = float(1.0 / (t[1] - t[0])) if n > 1 else float("nan")
        settings = {
            "sample_rate_time_axis": sr_axis,
            "record_length_setting": int(rl),
            "record_length_requested": int(want),
            "time_scale": horz.scale,
            "div_count": _HORIZONTAL_DIVS,
            "time_range": horz.range,
            "points_transferred": int(n),
            "vertical_scale": scales,
        }
        # Read back what the scope used, so the sweep loop can check it against
        # the plan: the ADC rate of the transferred record, the record length,
        # and the acquisition and trigger modes.
        try:
            settings["sample_rate"] = float(self.scope.acq.get_adc_rate())
        except Exception as exc:
            settings["sample_rate"] = sr_axis
            settings["readback_error_sample_rate"] = str(exc)
        try:
            settings["record_length"] = int(self.scope.acq.get_record_length())
        except Exception as exc:
            settings["record_length"] = int(n)
            settings["readback_error_record_length"] = str(exc)
        for key, fn in (
            ("acquisition_mode", self.scope.acq.get_mode),
            ("trigger_mode", self.scope.trig.get_mode),
            ("sample_rate_display", self.scope.acq.get_sample_rate),
        ):
            try:
                settings[key] = str(fn())
            except Exception as exc:
                settings[f"readback_error_{key}"] = str(exc)
        return Capture(
            signals=signals,
            t=t[:n],
            units={k: "V" for k in signals},
            group=self.name,
            full_scale=full_scale,
            settings=settings,
        )

    def describe(self) -> dict:
        d = identity(self.scope)
        d.update(
            channels=self.channels,
            trigger=self.trigger,
            record_lengths=list(self.record_lengths),
            clip_flag="none from the driver; full_scale is offset +-5 div (UNVERIFIED)",
        )
        setup = {}
        for name, src in self.channels.items():
            try:
                setup[name] = dataclasses.asdict(self._ch(src).get_settings())
            except Exception:
                pass
        d["channel_setup"] = setup
        return d


class RohdeSchwarzRtb2000AfgSource(Preserved):
    """The RTB2000's built-in generator driving a sine at [Gen Out].

    ``max_vpp`` is required: the sweep refuses any amplitude above it. The
    driver accepts 0.06 to 6 Vpp. ``load`` is 50 or ``"HIGHZ"``; None keeps the
    instrument's setting. Settings found on open are restored on close, and
    the output stops on close and on error.
    """

    def __init__(
        self,
        scope,
        *,
        max_vpp: float,
        load: float | str | None = None,
        offset: float = 0.0,
        preserve: bool = True,
    ):
        super().__init__(scope, preserve=preserve)
        self.scope = scope
        self.max_vpp = float(max_vpp)
        self.load = load
        self.offset = offset
        self.name = "scope.afg"
        self._shaped = False

    def open(self) -> None:
        super().open()
        self.scope.afg.set_output_state(False)
        # A burst found on would give bursts, not a continuous sine. The
        # preserved setup brings the found burst state back on close.
        self.scope.afg.set_burst_state(False)
        if self.load is not None:
            self.scope.afg.set_load(self.load)
        self._shaped = False

    def set(self, freq: float, amplitude: float | None) -> dict:
        a = check_amplitude(amplitude, self.max_vpp, self.name)
        if not self._shaped:
            s = self.scope.afg.set_waveform_sine(freq, a, offset=self.offset)
            self._shaped = True
            return {"freq": s.frequency, "amplitude": s.amplitude}
        return {
            "freq": self.scope.afg.set_frequency(freq),
            "amplitude": self.scope.afg.set_amplitude(a),
        }

    def output(self, state: bool) -> None:
        self.scope.afg.set_output_state(bool(state))

    def close(self) -> None:
        try:
            self.scope.afg.set_output_state(False)
        finally:
            super().close()

    def describe(self) -> dict:
        d = identity(self.scope)
        d.update(source="afg", max_vpp=self.max_vpp, load_setting=self.load, offset=self.offset)
        try:
            d["load_readback"] = self.scope.afg.get_load()
        except Exception:
            pass
        return d
