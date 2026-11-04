"""Tektronix AFG31000 channel as a sweep source."""

from __future__ import annotations

import dataclasses

from sweep.instruments.base import Preserved, check_amplitude, identity


class TektronixAfg31000Source(Preserved):
    """One AFG31000 channel driving a sine.

    ``max_vpp`` is required: the sweep refuses any amplitude above it.
    ``v_limit`` is required and goes to the driver's ``set_waveform_sine``: it
    sets the instrument's output voltage clamp, in volts at the load, as
    +-volts or a ``(low, high)`` pair, and the driver refuses a swing outside
    it before anything is written. ``load`` is ohms, or ``"HIGHZ"``; None keeps
    the instrument's setting. On open, modulation, sweep, burst and noise add
    are turned off, so the output is a plain sine. The driver itself guards
    the stale-load window after a load change (ADR-054). Settings found on open
    are restored on close, and the output opens on close and on error.
    """

    def __init__(
        self,
        afg,
        channel: int = 1,
        *,
        max_vpp: float,
        v_limit: float | tuple[float, float],
        load: float | str | None = None,
        offset: float = 0.0,
        preserve: bool = True,
    ):
        super().__init__(afg, preserve=preserve)
        self.afg = afg
        self.n = channel
        self.max_vpp = float(max_vpp)
        self.v_limit = v_limit
        self.load = load
        self.offset = offset
        self.name = f"{type(afg).__name__}.ch{channel}"
        self._shaped = False

    @property
    def ch(self):
        return self.afg.channel(self.n)

    def open(self) -> None:
        super().open()
        self.ch.set_output_state(False)
        self.ch.set_modulation_off()
        if self.load is not None:
            self.ch.set_load(self.load)
        self._shaped = False

    def set(self, freq: float, amplitude: float | None) -> dict:
        a = check_amplitude(amplitude, self.max_vpp, self.name)
        if not self._shaped:
            s = self.ch.set_waveform_sine(
                freq, a, v_limit=self.v_limit, offset=self.offset, noise_add=False
            )
            self._shaped = True
            return {"freq": s.frequency, "amplitude": s.amplitude}
        return {"freq": self.ch.set_frequency(freq), "amplitude": self.ch.set_amplitude(a)}

    def output(self, state: bool) -> None:
        self.ch.set_output_state(bool(state))

    def close(self) -> None:
        try:
            self.ch.set_output_state(False)
        finally:
            super().close()

    def describe(self) -> dict:
        d = identity(self.afg)
        d.update(
            channel=self.n,
            max_vpp=self.max_vpp,
            v_limit=self.v_limit,
            load_setting=self.load,
            offset=self.offset,
        )
        for key, fn in (("load_readback", "get_load"), ("voltage_limits", "get_voltage_limits")):
            try:
                v = getattr(self.ch, fn)()
                d[key] = dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
            except Exception:
                pass
        return d
