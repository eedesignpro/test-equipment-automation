"""Keysight 33500/33600 channel as a sweep source."""

from __future__ import annotations

import math

from sweep.instruments.base import Preserved, check_amplitude, identity


class Keysight33500Source(Preserved):
    """One 33500 channel driving a sine.

    ``max_vpp`` is required: the sweep refuses any amplitude above it. ``load``
    sets what the generator assumes it drives, in ohms, so the amplitude means
    volts at that load; leave it None to keep the instrument's setting.
    ``sync`` routes the Sync connector to this channel. Settings found on open
    are restored on close, and the output relay opens on close and on error.
    """

    def __init__(
        self,
        awg,
        channel: int = 1,
        *,
        max_vpp: float,
        load: float | None = None,
        offset: float = 0.0,
        sync: bool = True,
        preserve: bool = True,
    ):
        super().__init__(awg, preserve=preserve)
        self.awg = awg
        self.n = channel
        self.max_vpp = float(max_vpp)
        self.load = load
        self.offset = offset
        self.sync = sync
        self.name = f"{type(awg).__name__}.ch{channel}"
        self._shaped = False

    @property
    def ch(self):
        return getattr(self.awg, f"ch{self.n}")

    def open(self) -> None:
        super().open()
        self.ch.set_output_state(False)
        # The limit is in Vpp, so the channel must read amplitudes as Vpp.
        self.ch.set_amplitude_unit("VPP")
        if self.load is not None:
            self.ch.set_load(self.load)
        if self.sync:
            self.awg.sync.set_source(self.n)
            self.awg.sync.set_state(True)
        self._shaped = False

    def set(self, freq: float, amplitude: float | None) -> dict:
        a = check_amplitude(amplitude, self.max_vpp, self.name)
        if not self._shaped:
            state = self.ch.set_waveform_sine(freq, a, offset=self.offset)
            self._shaped = True
            return {"freq": state.frequency, "amplitude": state.amplitude}
        f = self.ch.set_frequency(freq)
        got = self.ch.set_amplitude(a)
        return {"freq": f, "amplitude": got}

    # Samples per period of the highest tone in a multi-tone waveform. At 4 the
    # bench 5 ohm |Z| at 5 MHz read 5 % high against a one-tone sweep.
    MULTITONE_SAMPLES_PER_CYCLE = 10

    def set_multitone(self, freqs, amplitude: float, base: float) -> dict:
        """Play the given tones, all multiples of ``base``, as one arbitrary waveform.

        ``amplitude`` is the peak to peak of the sum. The waveform holds one
        period of ``base`` at MULTITONE_SAMPLES_PER_CYCLE (10) samples per period of the highest tone.
        UNVERIFIED: the arbitrary-waveform rate and length limits of this model
        are taken from the driver's own checks, not measured here.
        """
        from sweep.multitone import multitone_samples

        a = check_amplitude(amplitude, self.max_vpp, self.name)
        harmonics = [int(round(f / base)) for f in freqs]
        points = max(32, int(math.ceil(self.MULTITONE_SAMPLES_PER_CYCLE * max(harmonics))))
        state = self.ch.set_waveform_arb(
            multitone_samples(harmonics, points).tolist(),
            sample_rate=points * base,
            ampl=a,
            offset=self.offset,
            name="SWEEPMT",
        )
        self._shaped = False
        return {"amplitude": state.amplitude, "points": points, "sample_rate": points * base}

    def output(self, state: bool) -> None:
        self.ch.set_output_state(bool(state))

    def close(self) -> None:
        try:
            self.ch.set_output_state(False)
        finally:
            super().close()

    def describe(self) -> dict:
        d = identity(self.awg)
        d.update(channel=self.n, max_vpp=self.max_vpp, load_setting=self.load, offset=self.offset)
        try:
            d["load_readback"] = self.ch.get_load()
        except Exception:
            pass
        return d
