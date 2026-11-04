"""Shared pieces for the ready-made instrument sources and receivers."""

from __future__ import annotations

import contextlib
import math


class Preserved:
    """Enter an instrument's ``preserved_setup()`` on open and leave it on close.

    The settings found before the sweep come back exactly after it, through
    the driver's own snapshot and restore. When two sweep parts share one
    instrument, for example the MSO receiver and the MSO generator source, only
    the first to open saves the setup and the last to close restores it, so the
    two never save over each other.
    """

    _holders: dict[int, list] = {}

    def __init__(self, instrument, *, preserve: bool = True):
        self.instrument = instrument
        self.preserve = preserve
        self._stack: contextlib.ExitStack | None = None
        self._held = False

    def open(self) -> None:
        if not (self.preserve and hasattr(self.instrument, "preserved_setup")):
            return
        key = id(self.instrument)
        entry = Preserved._holders.get(key)
        if entry is None or entry[2] is not self.instrument:
            stack = contextlib.ExitStack()
            stack.enter_context(self.instrument.preserved_setup())
            Preserved._holders[key] = [stack, 1, self.instrument]
        else:
            entry[1] += 1
        self._held = True

    def close(self) -> None:
        if not self._held:
            return
        self._held = False
        key = id(self.instrument)
        entry = Preserved._holders.get(key)
        if entry is None or entry[2] is not self.instrument:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            del Preserved._holders[key]
            entry[0].close()


def identity(instrument) -> dict:
    """Identity fields an instrument driver exposes, without new I/O where possible."""
    out = {"class": type(instrument).__name__}
    for attr in ("resource_name", "model", "idn"):
        v = getattr(instrument, attr, None)
        if v is not None and not callable(v):
            out[attr] = str(v)
    return out


def check_amplitude(amplitude: float, max_vpp: float, name: str) -> float:
    if amplitude is None:
        raise ValueError(f"{name}: an amplitude is required")
    if not (math.isfinite(max_vpp) and max_vpp > 0):
        raise ValueError(f"{name}: the limit must be a positive number, not {max_vpp}")
    if not (math.isfinite(amplitude) and amplitude >= 0):
        raise ValueError(f"{name}: amplitude {amplitude} is not a valid Vpp")
    if amplitude > max_vpp:
        raise ValueError(f"{name}: {amplitude} Vpp is above the limit of {max_vpp} Vpp")
    return float(amplitude)
