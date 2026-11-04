"""Keysight 33120A as a sweep source."""

from __future__ import annotations

from sweep.instruments.base import Preserved, check_amplitude, identity


class Keysight33120ASource(Preserved):
    """The 33120A driving a sine. It has NO output relay.

    The front-panel output is always connected, so nothing here can disconnect
    it. ``output(False)`` sets the minimum amplitude instead (the driver's
    ``set_amplitude("MIN")``); the output still drives that amplitude and the
    offset. ``set()`` writes the frequency and keeps the amplitude at the
    minimum until ``output(True)``, so ``output`` is the only path to the drive
    level; a ``drive_off`` sweep therefore stays at the minimum amplitude.
    ``output(True)`` writes the requested amplitude and returns its read-back.

    ``max_vpp`` is required: the sweep refuses any amplitude above it. ``load``
    is ``"FIFTY"`` or ``"HIGHZ"``, what the generator assumes it drives, so the
    amplitude means volts at that load; None keeps the instrument's setting.
    ``sync`` turns the SYNC connector on. The amplitude unit is set to Vpp.

    With ``preserve`` (the default) the settings found on open come back on
    close, including the found amplitude. ``run_sweep`` then turns the output
    "off" once more by default (``leave_output_off=True``), so the sweep ends
    at the minimum amplitude, not the found one.
    """

    def __init__(
        self,
        gen,
        *,
        max_vpp: float,
        load: str | None = None,
        offset: float = 0.0,
        sync: bool = True,
        preserve: bool = True,
    ):
        super().__init__(gen, preserve=preserve)
        self.gen = gen
        self.max_vpp = float(max_vpp)
        self.load = load
        self.offset = offset
        self.sync = sync
        self.name = type(gen).__name__
        self._shaped = False
        self._amplitude: float | None = None
        self._at_minimum = False
        self._on = False

    def _minimum(self) -> float:
        got = self.gen.set_amplitude("MIN")
        self._at_minimum = True
        return got

    def open(self) -> None:
        super().open()
        self._minimum()
        self.gen.set_amplitude_unit("VPP")
        if self.load is not None:
            self.gen.set_load(self.load)
        if self.sync:
            self.gen.set_sync_state(True)
        self._shaped = False
        self._on = False

    def set(self, freq: float, amplitude: float | None) -> dict:
        a = check_amplitude(amplitude, self.max_vpp, self.name)
        self._amplitude = a
        if not self._shaped:
            lowest = self.gen.get_amplitude("MIN")
            s = self.gen.set_waveform_sine(freq, a if self._on else lowest, offset=self.offset)
            self._shaped = True
            self._at_minimum = not self._on
            return {"freq": s.frequency, "amplitude": s.amplitude}
        f = self.gen.set_frequency(freq)
        if not self._on:
            return {"freq": f, "amplitude": self.gen.get_amplitude()}
        self._at_minimum = False
        return {"freq": f, "amplitude": self.gen.set_amplitude(a)}

    def output(self, state: bool) -> float | None:
        self._on = bool(state)
        if not state:
            return self._minimum()
        if self._amplitude is not None:
            self._at_minimum = False
            return self.gen.set_amplitude(self._amplitude)
        return None

    def close(self) -> None:
        try:
            self._minimum()
        finally:
            super().close()

    def describe(self) -> dict:
        d = identity(self.gen)
        d.update(
            max_vpp=self.max_vpp,
            load_setting=self.load,
            offset=self.offset,
            output_relay=False,
            output_off_means="minimum amplitude; the output stays connected",
            restores_found_amplitude_on_close=self.preserve,
            ends_at="minimum amplitude when run_sweep leaves the output off (its default)",
        )
        try:
            d["load_readback"] = self.gen.get_load()
        except Exception:
            pass
        try:
            d["minimum_amplitude"] = self.gen.get_amplitude("MIN")
        except Exception:
            pass
        return d
