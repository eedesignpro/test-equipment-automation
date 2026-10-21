"""Every Reading unpacks in the order its ``meas_`` name spells.

The reading types are frozen dataclasses rather than tuples so they can gain a
field without breaking call sites (design standard 9.4). ``__iter__`` buys back
``v, i = ch.meas_vi()`` for notebook work. These tests pin the contract: the
arity and order of iteration are part of the API, so a field added later must
either be spelled by the method name and added here on purpose, or stay out of
``__iter__``.

No hardware: the dataclasses are constructed directly.
"""

from __future__ import annotations

import pytest

from drivers.keysight_b2900 import Reading as SmuReading
from drivers.keysight_b2900 import SweepReading
from drivers.keysight_e364xa import Reading as PsuReading
from drivers.rigol_dl3000 import Reading as LoadReading


@pytest.mark.parametrize("cls", [SmuReading, PsuReading])
def test_vi_reading_unpacks_voltage_then_current(cls):
    reading = cls(voltage=1.25, current=0.5)
    voltage, current = reading
    assert (voltage, current) == (1.25, 0.5)
    assert list(reading) == [1.25, 0.5]


@pytest.mark.parametrize("cls", [SmuReading, PsuReading])
def test_vi_reading_keeps_attribute_access(cls):
    reading = cls(voltage=1.25, current=0.5)
    assert reading.voltage == 1.25
    assert reading.current == 0.5


def test_vip_reading_unpacks_voltage_current_then_power():
    reading = LoadReading(voltage=12.0, current=2.0, power=24.0)
    voltage, current, power = reading
    assert (voltage, current, power) == (12.0, 2.0, 24.0)
    assert reading.power == 24.0


def test_sweep_reading_unpacks_into_columns():
    pts = SweepReading(voltage=[0.0, 1.0, 2.0], current=[0.0, 0.1, 0.2])
    voltage, current = pts
    assert voltage == [0.0, 1.0, 2.0]
    assert current == [0.0, 0.1, 0.2]


@pytest.mark.parametrize(
    ("cls", "kwargs", "expected"),
    [
        (SmuReading, {"voltage": 1.0, "current": 2.0}, 2),
        (PsuReading, {"voltage": 1.0, "current": 2.0}, 2),
        (LoadReading, {"voltage": 1.0, "current": 2.0, "power": 3.0}, 3),
        (SweepReading, {"voltage": [1.0], "current": [2.0]}, 2),
    ],
)
def test_iteration_arity_is_pinned(cls, kwargs, expected):
    """A new field must not silently change what unpacking yields."""
    assert len(list(cls(**kwargs))) == expected


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (SmuReading, {"voltage": 1.0, "current": 2.0}),
        (PsuReading, {"voltage": 1.0, "current": 2.0}),
        (LoadReading, {"voltage": 1.0, "current": 2.0, "power": 3.0}),
    ],
)
def test_readings_stay_frozen(cls, kwargs):
    reading = cls(**kwargs)
    with pytest.raises(AttributeError):
        reading.voltage = 99.0
