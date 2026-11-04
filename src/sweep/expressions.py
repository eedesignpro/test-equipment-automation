"""Ready-made expressions for common bench measurements.

Each function returns a dict for ``run_sweep(expressions=...)``: name to
``(function, unit)``. The functions read the complex value of each named
signal at the drive frequency, so a difference of two channels is taken as
phasors, with phase. Pass the result straight in, or merge several::

    expressions = low_side_shunt("V_total", "V_shunt", r_shunt=1.0)
    expressions |= {"gain": lambda v: v["V_out"] / v["V_in"]}

Both shunt layouts assume the two voltage probes share the generator's ground.
Pick the shunt so its voltage sits well above the probe noise but stays small
next to the DUT voltage; the uncertainty band on the plots shows when it does
not.
"""

from __future__ import annotations


def difference(a: str, b: str, unit: str = "V") -> tuple:
    """``a - b`` as an expression, for example the voltage across a part."""
    return (lambda v: v[a] - v[b], unit)


def low_side_shunt(v_total: str, v_shunt: str, r_shunt: float, *, prefix: str = "") -> dict:
    """DUT on top, shunt resistor to ground below it.

    ``v_total`` is the probe at the top of the DUT (the drive), ``v_shunt`` the
    probe across the shunt. The DUT voltage is their difference and the current
    is the shunt voltage over ``r_shunt``.
    """
    r = float(r_shunt)
    return {
        f"{prefix}V_dut": (lambda v: v[v_total] - v[v_shunt], "V"),
        f"{prefix}I": (lambda v: v[v_shunt] / r, "A"),
        f"{prefix}Z": (lambda v: (v[v_total] - v[v_shunt]) * r / v[v_shunt], "ohm"),
    }


def high_side_shunt(v_source: str, v_dut: str, r_shunt: float, *, prefix: str = "") -> dict:
    """Shunt resistor between the source and the DUT, DUT to ground.

    ``v_source`` is the probe at the source side of the shunt, ``v_dut`` the
    probe at the top of the DUT. The current is the shunt's voltage, their
    difference, over ``r_shunt``.
    """
    r = float(r_shunt)
    return {
        f"{prefix}V_dut": (lambda v: v[v_dut], "V"),
        f"{prefix}I": (lambda v: (v[v_source] - v[v_dut]) / r, "A"),
        f"{prefix}Z": (lambda v: v[v_dut] * r / (v[v_source] - v[v_dut]), "ohm"),
    }


def current_probe(v: str, i: str, *, prefix: str = "") -> dict:
    """A voltage probe across the DUT and a current probe through it."""
    return {f"{prefix}Z": (lambda x: x[v] / x[i], "ohm")}
