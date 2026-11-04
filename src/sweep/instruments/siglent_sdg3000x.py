"""Siglent SDG3000X channel as a sweep source."""

from __future__ import annotations

import math

from sweep.instruments.base import Preserved, check_amplitude, identity


class SiglentSdg3000xSource(Preserved):
    """One SDG3000X channel driving a sine.

    ``ampl_limit`` is required and goes to the driver's ``set_waveform_sine``:
    it sets the instrument's amplitude limit in Vpp into high impedance, so at
    a 50 ohm load setting it allows half its value (driver note SDG-34). The
    sweep refuses any amplitude above that load-aware allowance, and above
    ``max_vpp`` when given. After the first point the amplitude moves with
    ``set_amplitude``, which the instrument clamps silently at the limit, so
    the check here runs before every write. ``load`` is 50 or ``"HIGHZ"``;
    None keeps the instrument's setting. Settings found on open are restored
    on close, and the output opens on close and on error.
    """

    def __init__(
        self,
        sdg,
        channel: int = 1,
        *,
        ampl_limit: float,
        max_vpp: float | None = None,
        load: float | str | None = None,
        offset: float = 0.0,
        preserve: bool = True,
    ):
        super().__init__(sdg, preserve=preserve)
        self.sdg = sdg
        self.n = channel
        self.ampl_limit = float(ampl_limit)
        self.max_vpp = float(max_vpp) if max_vpp is not None else self.ampl_limit
        self.load = load
        self.offset = offset
        self.name = f"{type(sdg).__name__}.ch{channel}"
        self._shaped = False
        self._allowed = min(self.max_vpp, self.ampl_limit)

    @property
    def ch(self):
        return self.sdg.channel(self.n)

    def open(self) -> None:
        super().open()
        self.ch.set_output_state(False)
        ohms = self.ch.set_load(self.load) if self.load is not None else self.ch.get_load()
        at_load = self.ampl_limit if ohms == math.inf else self.ampl_limit / 2
        self._allowed = min(self.max_vpp, at_load)
        self._shaped = False

    def set(self, freq: float, amplitude: float | None) -> dict:
        a = check_amplitude(amplitude, self._allowed, self.name)
        if not self._shaped:
            s = self.ch.set_waveform_sine(freq, a, ampl_limit=self.ampl_limit, offset=self.offset)
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
        d = identity(self.sdg)
        d.update(
            channel=self.n,
            ampl_limit_highz=self.ampl_limit,
            max_vpp=self.max_vpp,
            allowed_vpp_at_load=self._allowed,
            load_setting=self.load,
            offset=self.offset,
        )
        try:
            d["load_readback"] = self.ch.get_load()
        except Exception:
            pass
        return d
