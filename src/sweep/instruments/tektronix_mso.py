"""Tektronix MSO 4/5/6 as a time-domain receiver, and its built-in AFG as a source."""

from __future__ import annotations

import time

import numpy as np

from common.errors import InstrumentTimeoutError

from sweep.capture import CapturePlan
from sweep.instruments.base import Preserved, check_amplitude, identity
from sweep.interfaces import Capture

# Vertical divisions either side of center on the MSO 4/5/6 screen.
_HALF_DIVS = 5.0


class TektronixMsoReceiver(Preserved):
    """Capture named scope channels in one acquisition at each frequency.

    ``channels`` maps a signal name to a channel, for example
    ``{"V_R": "CH8", "I_R": "CH3"}``. The capture plan's sample rate and
    record length go to the scope in MANUAL horizontal mode, and the values the
    scope actually used are recorded. ``trigger`` names a channel, such as a
    generator sync, for an edge trigger; the trigger mode stays AUTO so an
    acquisition always completes. ``autoscale`` sets each channel's vertical
    scale from the previous capture and captures again when a channel clips or
    uses little of the screen. ``screenshots`` saves the screen at each point.
    ``scope_measurements`` reads every measurement set up on the scope at each
    point, from the same capture; ``add_measurements`` sets up more for the
    sweep, as (type, source) pairs such as ``("FREQUENCY", "CH5")``.
    ``bandwidth`` sets every measured channel's bandwidth limit for the sweep,
    in Hz or ``"FULL"``; None keeps what the scope has. The whole
    scope setup found on open, measurements included, is restored on close.
    """

    def __init__(
        self,
        scope,
        channels: dict[str, str],
        *,
        trigger: str | None = None,
        trigger_level: float | None = None,
        width: int = 2,
        autoscale: bool = False,
        fill: float = 0.7,
        retries: int = 3,
        timeout_s: float = 20.0,
        preserve: bool = True,
        name: str | None = None,
        screenshots: bool = False,
        scope_measurements: bool = False,
        add_measurements: list[tuple[str, str]] | None = None,
        bandwidth: float | str | None = None,
    ):
        super().__init__(scope, preserve=preserve)
        self.scope = scope
        self.channels = dict(channels)
        self.screenshots = screenshots
        self.scope_measurements = scope_measurements or bool(add_measurements)
        self.add_measurements = list(add_measurements or [])
        self.bandwidth = bandwidth
        self._slots: list[dict] = []
        self._prepared: tuple[float, int] | None = None
        self.trigger = trigger
        self.trigger_level = trigger_level
        self.width = width
        self.autoscale = autoscale
        self.fill = fill
        self.retries = retries
        self.timeout_s = timeout_s
        self.name = name or "scope"
        self._units: dict[str, str] = {}

    def _ch(self, source: str):
        return getattr(self.scope, source.lower())

    def open(self) -> None:
        super().open()
        self.scope.trig.set_mode("AUTO")
        if self.trigger:
            self.scope.trig.set_edge(self.trigger, self.trigger_level)
        for name, src in self.channels.items():
            ch = self._ch(src)
            ch.set_selected(True)
            if self.bandwidth is not None:
                ch.set_bandwidth(self.bandwidth)
            try:
                self._units[name] = ch.probe.get_units() or "V"
            except Exception:
                self._units[name] = "V"
        if self.add_measurements:
            used = {s["slot"] for s in self._find_slots()}
            n = 1
            for token, source in self.add_measurements:
                while n in used:
                    n += 1
                try:
                    self.scope.meas.custom(token, source, num=n, persist=True, timeout=0.5)
                except Exception:
                    pass  # no acquisition yet; the slot is set up and reads at each point
                used.add(n)
        if self.scope_measurements:
            self._slots = self._find_slots()

    def prepare_horizontal(self, plan: CapturePlan) -> tuple[float, int]:
        """Set the horizontal for ``plan`` now and return what the scope took.

        Returns (sample rate, record length) as read back. The next capture
        uses exactly this setting, so a coherent sweep can place the drive on
        a bin of that record before the capture is armed.
        """
        sr, rl = self.scope.horz.set_manual(plan.sample_rate, plan.record_length)
        self._prepared = (float(sr), int(rl))
        return self._prepared

    def _find_slots(self) -> list[dict]:
        """Every measurement slot set up on the scope, with what it measures."""
        listing = self.scope.meas.query("MEASUrement:LIST?").strip().strip('"')
        slots = []
        for tok in listing.split(","):
            tok = tok.strip().upper()
            if not tok.startswith("MEAS") or not tok[4:].isdigit():
                continue
            n = int(tok[4:])
            info = {"slot": n}
            for key, cmd in (("type", "TYPe"), ("source", "SOUrce1")):
                try:
                    info[key] = self.scope.meas.query(f"MEASUrement:MEAS{n}:{cmd}?").strip()
                except Exception:
                    info[key] = ""
            for key, fn in (
                ("label", self.scope.meas.get_label),
                ("unit", self.scope.meas.get_yunit),
            ):
                try:
                    info[key] = fn(n)
                except Exception:
                    info[key] = ""
            slots.append(info)
        return slots

    def after_point(self, point: int, freq: float, folder) -> list[dict]:
        """Save a screenshot and read every scope measurement, from the capture just made.

        The scope stops after each single acquisition, so the screen and the
        measurement badges show the same record the sweep analyzed.
        """
        rows = []
        if self.screenshots:
            d = folder / "screenshots"
            d.mkdir(exist_ok=True)
            self.scope.file.get_screenshot(f"point_{point:04d}_{freq:.6g}Hz", folder_path=str(d))
        if self.scope_measurements:
            m = self.scope.meas
            for slot in self._slots:
                n = slot["slot"]
                row = {k: v for k, v in slot.items()}
                for key, fn in (
                    ("value", m.get_value),
                    ("curr_min", m.get_curr_min),
                    ("curr_max", m.get_curr_max),
                    ("curr_std", m.get_curr_std),
                ):
                    try:
                        val = fn(n)
                        # The scope reports 9.91e37 for a value it cannot compute.
                        row[key] = float("nan") if abs(val) >= 9.9e37 else val
                    except Exception:
                        row[key] = float("nan")
                rows.append(row)
        return rows

    def _acquire(self, plan: CapturePlan) -> tuple[dict, np.ndarray, dict]:
        # MANUAL mode with sample rate and record length written together: the
        # scale follows from them. Every value is read back after the capture.
        t0 = time.perf_counter()
        if self._prepared is not None:
            sr, rl = self._prepared
            self._prepared = None
        else:
            sr, rl = self.scope.horz.set_manual(plan.sample_rate, plan.record_length)
        t1 = time.perf_counter()
        self.scope.acq.cmd_single()
        deadline = time.monotonic() + self.timeout_s + 3.0 * rl / sr
        time.sleep(min(0.05, rl / sr))
        while self.scope.acq.get_state() != 0:
            if time.monotonic() > deadline:
                raise InstrumentTimeoutError(f"{self.name}: acquisition did not complete")
            time.sleep(0.02)
        t2 = time.perf_counter()
        signals, clipped, raw = {}, {}, {}
        t = None
        for name, src in self.channels.items():
            codes, pre = self.scope.data.get_waveform_codes(src, width=self.width)
            scale = {
                k: float(pre.get(k, d)) for k, d in (("ymult", 1.0), ("yoff", 0.0), ("yzero", 0.0))
            }
            raw[name] = {"codes": codes, **scale}
            signals[name] = scale["yzero"] + scale["ymult"] * (
                codes.astype(np.float64) - scale["yoff"]
            )
            if t is None:
                timing = {
                    k: float(pre.get(k, d))
                    for k, d in (("xincr", 1.0), ("xzero", 0.0), ("pt_off", 0.0))
                }
                raw["_time"] = timing
                n = np.arange(codes.size, dtype=np.float64)
                t = timing["xzero"] + timing["xincr"] * (n - timing["pt_off"])
            try:
                clipped[name] = bool(self._ch(src).get_clipping())
            except Exception:
                clipped[name] = False
        t3 = time.perf_counter()
        horz = self.scope.horz
        settings = {
            "sample_rate": horz.get_samplerate(),
            "record_length": int(horz.get_record_length()),
            "time_scale": horz.get_scale(),
            "div_count": horz.get_div_count(),
            "horizontal_mode": str(horz.get_mode()),
            "points_transferred": min(len(v) for v in signals.values()),
            "sample_rate_set": sr,
            "record_length_set": int(rl),
        }
        t4 = time.perf_counter()
        settings["time_s"] = {
            "horizontal_setup": round(t1 - t0, 4),
            "acquire": round(t2 - t1, 4),
            "transfer": round(t3 - t2, 4),
            "readback": round(t4 - t3, 4),
        }
        return signals, t, {"clipped": clipped, "raw": raw, **settings}

    def _rescale(self, signals: dict, clipped: dict) -> bool:
        changed = False
        for name, src in self.channels.items():
            ch = self._ch(src)
            scale = ch.get_scale()
            offset = ch.get_offset()
            peak = float(np.max(np.abs(signals[name] - offset)))
            used = peak / (scale * _HALF_DIVS)
            if clipped.get(name):
                ch.set_scale(scale * 2.5)
                changed = True
            elif used < self.fill / 3 or used > 0.95:
                want = max(peak, 1e-12) / (self.fill * _HALF_DIVS)
                got = ch.set_scale(want)
                changed = changed or abs(got - scale) / scale > 0.05
        return changed

    def measure(self, freq: float, plan: CapturePlan) -> Capture:
        tries = 0
        while True:
            signals, t, info = self._acquire(plan)
            clipped = info.pop("clipped")
            raw = info.pop("raw")
            tries += 1
            if not self.autoscale or tries > self.retries or not self._rescale(signals, clipped):
                break
        n = min(len(t), *(len(v) for v in signals.values()))
        signals = {k: np.asarray(v[:n], dtype=float) for k, v in signals.items()}
        for k, v in raw.items():
            if k != "_time":
                v["codes"] = v["codes"][:n]
        scales = {}
        for name, src in self.channels.items():
            try:
                scales[name] = self._ch(src).get_scale()
            except Exception:
                pass
        info.update(tries=tries, vertical_scale=scales)
        return Capture(
            signals=signals,
            t=np.asarray(t[:n], dtype=float),
            units={k: self._units.get(k, "V") for k in signals},
            group=self.name,
            clipped=clipped,
            settings=info,
            raw=raw,
        )

    def describe(self) -> dict:
        d = identity(self.scope)
        d.update(
            channels=self.channels,
            trigger=self.trigger,
            autoscale=self.autoscale,
            screenshots=self.screenshots,
            scope_measurement_slots=self._slots,
        )
        probes = {}
        for name, src in self.channels.items():
            ch = self._ch(src)
            entry = {}
            for key, fn in (
                ("probe_type", lambda: ch.probe.get_type()),
                ("units", lambda: ch.probe.get_units()),
                ("gain", lambda: ch.probe.get_gain()),
                ("termination", ch.get_termination),
                ("deskew", ch.get_deskew),
                ("scale", ch.get_scale),
                ("bandwidth", ch.get_bandwidth),
            ):
                try:
                    entry[key] = fn()
                except Exception:
                    pass
            probes[name] = entry
        d["channel_setup"] = probes
        return d


class TektronixMsoAfgSource(Preserved):
    """The MSO's built-in generator driving a sine. ``max_vpp`` is required."""

    def __init__(self, scope, *, max_vpp: float, offset: float = 0.0, preserve: bool = True):
        super().__init__(scope, preserve=preserve)
        self.scope = scope
        self.max_vpp = float(max_vpp)
        self.offset = offset
        self.name = "scope.afg"
        self._shaped = False

    def open(self) -> None:
        super().open()
        self.scope.afg.set_output_state(False)
        # A burst found on would give bursts, not a continuous sine; the found
        # state comes back on close through the preserved setup.
        self.scope.afg.set_continuous()
        self._shaped = False

    def set(self, freq: float, amplitude: float | None) -> dict:
        a = check_amplitude(amplitude, self.max_vpp, self.name)
        if not self._shaped:
            s = self.scope.afg.set_waveform_sine(freq, a, offset=self.offset, noise_add=False)
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
        d.update(source="afg", max_vpp=self.max_vpp, offset=self.offset)
        try:
            d["load_readback"] = self.scope.afg.get_load()
        except Exception:
            pass
        return d
