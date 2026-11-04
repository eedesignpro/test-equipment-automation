"""Scalar measurements: one number per point from a meter, a supply, an SMU or a load.

:class:`ScalarMeasurement` wraps any function that returns one number. The
helpers below build one for a common reading on each bench instrument::

    dmm_v = siglent_sdm_ac_voltage(dmm, "V_out", bandwidth=20)
    run_sweep(freqs, source, [scope_rx, dmm_v], ...)

A reading has no phase, so each is its own group in the results.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from sweep.capture import CapturePlan
from sweep.instruments.base import Preserved, identity
from sweep.interfaces import Readings


class ScalarMeasurement(Preserved):
    """Call ``fn()`` at each point and return its number as :class:`Readings`.

    ``settle_s`` waits before the first reading. With ``n`` above 1 the
    readings are averaged; their spread goes into the settings. An overload
    that a driver reports as NaN stays NaN in the average. ``instrument`` is
    used for ``describe()`` and, with ``preserve``, its ``preserved_setup()``
    runs from open to close, so a meter that the reading reconfigures comes
    back as found.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[], float],
        unit: str,
        settle_s: float = 0.0,
        *,
        n: int = 1,
        instrument=None,
        preserve: bool = False,
        settings: dict | None = None,
    ):
        if int(n) < 1:
            raise ValueError(f"{name}: n must be at least 1, got {n}")
        super().__init__(instrument, preserve=preserve)
        self.name = name
        self.fn = fn
        self.unit = unit
        self.settle_s = float(settle_s)
        self.n = int(n)
        self.settings = dict(settings or {})

    def measure(self, freq: float, plan: CapturePlan | None = None) -> Readings:
        if self.settle_s:
            time.sleep(self.settle_s)
        values = [float(self.fn()) for _ in range(self.n)]
        mean = math.fsum(values) / len(values)
        settings = {"n": self.n, "settle_s": self.settle_s, **self.settings}
        if self.n > 1:
            spread = [v - mean for v in values]
            settings["std"] = math.sqrt(math.fsum(d * d for d in spread) / (self.n - 1))
            settings["readings"] = values
        return Readings({self.name: mean}, units={self.name: self.unit}, settings=settings)

    def describe(self) -> dict:
        d = identity(self.instrument) if self.instrument is not None else {}
        d.update(
            kind="scalar",
            name=self.name,
            unit=self.unit,
            n=self.n,
            settle_s=self.settle_s,
            preserved=bool(self.preserve and hasattr(self.instrument, "preserved_setup")),
            **self.settings,
        )
        return d


def _scalar(name, fn, unit, instrument, reading, *, preserve=False, **kw) -> ScalarMeasurement:
    settings = {"reading": reading}
    return ScalarMeasurement(
        name, fn, unit, instrument=instrument, preserve=preserve, settings=settings, **kw
    )


# ---- Siglent SDM: each reading reconfigures the meter, so the setup is preserved.


def siglent_sdm_ac_voltage(
    dmm, name: str = "V_ac", *, range=None, bandwidth=None, preserve: bool = True, **kw
) -> ScalarMeasurement:
    """AC voltage in V RMS (``meas_vac``). ``bandwidth`` is the AC filter corner: 3, 20, 200 Hz."""
    return _scalar(
        name,
        lambda: dmm.meas_vac(range=range, bandwidth=bandwidth),
        "V",
        dmm,
        "meas_vac",
        preserve=preserve,
        **kw,
    )


def siglent_sdm_ac_current(
    dmm, name: str = "I_ac", *, range=None, bandwidth=None, preserve: bool = True, **kw
) -> ScalarMeasurement:
    """AC current in A RMS (``meas_iac``)."""
    return _scalar(
        name,
        lambda: dmm.meas_iac(range=range, bandwidth=bandwidth),
        "A",
        dmm,
        "meas_iac",
        preserve=preserve,
        **kw,
    )


def siglent_sdm_frequency(
    dmm, name: str = "f_meter", *, range=None, aperture=None, preserve: bool = True, **kw
) -> ScalarMeasurement:
    """Frequency in Hz (``meas_frequency``). ``range`` is the input voltage range."""
    return _scalar(
        name,
        lambda: dmm.meas_frequency(range=range, aperture=aperture),
        "Hz",
        dmm,
        "meas_frequency",
        preserve=preserve,
        **kw,
    )


# The channel is looked up at each reading, not when the helper is made, because
# some drivers (B2900, OWON) attach channels 2 and 3 only in connect().

# ---- Keysight E364xA: readings only, nothing to preserve.


def keysight_e364xa_voltage(psu, channel: int = 1, name: str | None = None, **kw):
    """Output voltage in V of ``channel`` (``meas_voltage``)."""
    return _scalar(
        name or f"V_ch{channel}",
        lambda: psu.channel(channel).meas_voltage(),
        "V",
        psu,
        "meas_voltage",
        **kw,
    )


def keysight_e364xa_current(psu, channel: int = 1, name: str | None = None, **kw):
    """Output current in A of ``channel`` (``meas_current``)."""
    return _scalar(
        name or f"I_ch{channel}",
        lambda: psu.channel(channel).meas_current(),
        "A",
        psu,
        "meas_current",
        **kw,
    )


def keysight_e364xa_power(psu, channel: int = 1, name: str | None = None, **kw):
    """Output power in W of ``channel``: voltage times current from one ``meas_vi``.

    The supply has no power reading; this is computed on the PC.
    """

    def power() -> float:
        r = psu.channel(channel).meas_vi()
        return r.voltage * r.current

    return _scalar(name or f"P_ch{channel}", power, "W", psu, "meas_vi (V*I on the PC)", **kw)


# ---- OWON ODP: readings only, nothing to preserve.


def owon_odp_voltage(psu, channel: int = 1, name: str | None = None, **kw):
    """Output voltage in V of ``channel`` (``meas_voltage``)."""
    return _scalar(
        name or f"V_ch{channel}",
        lambda: psu.channel(channel).meas_voltage(),
        "V",
        psu,
        "meas_voltage",
        **kw,
    )


def owon_odp_current(psu, channel: int = 1, name: str | None = None, **kw):
    """Output current in A of ``channel`` (``meas_current``)."""
    return _scalar(
        name or f"I_ch{channel}",
        lambda: psu.channel(channel).meas_current(),
        "A",
        psu,
        "meas_current",
        **kw,
    )


def owon_odp_power(psu, channel: int = 1, name: str | None = None, **kw):
    """Output power in W of ``channel`` (``meas_power``, the supply's own reading)."""
    return _scalar(
        name or f"P_ch{channel}",
        lambda: psu.channel(channel).meas_power(),
        "W",
        psu,
        "meas_power",
        **kw,
    )


# ---- Keysight B2900: readings only, nothing to preserve.


def keysight_b2900_voltage(smu, channel: int = 1, name: str | None = None, **kw):
    """Measured voltage in V of ``channel`` (``meas_voltage``); an overload reads NaN."""
    return _scalar(
        name or f"V_smu{channel}",
        lambda: smu.channel(channel).meas_voltage(),
        "V",
        smu,
        "meas_voltage",
        **kw,
    )


def keysight_b2900_current(smu, channel: int = 1, name: str | None = None, **kw):
    """Measured current in A of ``channel`` (``meas_current``); an overload reads NaN."""
    return _scalar(
        name or f"I_smu{channel}",
        lambda: smu.channel(channel).meas_current(),
        "A",
        smu,
        "meas_current",
        **kw,
    )


# ---- Rigol DL3000: readings only, nothing to preserve.


def rigol_dl3000_voltage(load, name: str = "V_load", **kw):
    """Input voltage in V (``meas_voltage``)."""
    return _scalar(name, load.meas_voltage, "V", load, "meas_voltage", **kw)


def rigol_dl3000_current(load, name: str = "I_load", **kw):
    """Input current in A (``meas_current``)."""
    return _scalar(name, load.meas_current, "A", load, "meas_current", **kw)


def rigol_dl3000_power(load, name: str = "P_load", **kw):
    """Input power in W (``meas_power``)."""
    return _scalar(name, load.meas_power, "W", load, "meas_power", **kw)
