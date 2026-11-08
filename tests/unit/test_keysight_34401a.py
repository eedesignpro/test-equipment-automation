# SPDX-License-Identifier: Apache-2.0
# Copyright: see the NOTICE file.
"""Hardware-free tests for the 34401A driver, against a small fake meter.

The reply shapes come from the bench 34401A (firmware 4-1-1) on 2026-11-07.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import pytest

from common.errors import InstrumentError
from drivers.keysight_34401a import Keysight_34401A, _reading


class FakeLink:
    """Answers the queries the driver sends; records every write."""

    resource_name = "ASRL4::INSTR"
    is_serial = True
    default_timeout_ms = 5000

    def __init__(self):
        self.writes: list[str] = []
        self.func = "VOLT"
        self.rng = {"VOLT:DC": 10.0, "CURR:DC": 3.0}
        self.auto = {"VOLT:DC": True, "CURR:DC": True}
        self.nplc = {"VOLT:DC": 10.0, "CURR:DC": 10.0}
        self.reading = "+5.00155000E+00"
        self.errors: list[str] = []

    def connect(self, timeout_ms=None):
        pass

    def disconnect(self):
        pass

    @property
    def is_connected(self):
        return True

    @contextmanager
    def temporary_timeout(self, ms):
        yield

    def _f(self):
        return {"VOLT": "VOLT:DC", "CURR": "CURR:DC"}.get(self.func, self.func)

    def write(self, cmd):
        self.writes.append(cmd)
        head, _, arg = cmd.partition(" ")
        if head.startswith("CONF:"):
            f = head[5:]
            self.func = {"VOLT:DC": "VOLT", "CURR:DC": "CURR"}.get(f, f)
            if arg and arg != "DEF":
                self.rng[f], self.auto[f] = float(arg), False
            else:
                self.auto[f] = True
        elif head.endswith(":RANG:AUTO"):
            self.auto[head[:-10]] = arg == "ON"
        elif head.endswith(":RANG"):
            self.rng[head[:-5]], self.auto[head[:-5]] = float(arg), False
        elif head.endswith(":NPLC"):
            self.nplc[head[:-5]] = float(arg)
        elif head == "FUNC":
            f = arg.strip('"')
            self.func = {"VOLT:DC": "VOLT", "CURR:DC": "CURR"}.get(f, f)

    def query(self, cmd):
        self.writes.append(cmd)
        if cmd == "*IDN?":
            return "HEWLETT-PACKARD,34401A,0,4-1-1"
        if cmd == "FUNC?":
            return f'"{self.func}"'
        if cmd in ("READ?",):
            return self.reading
        if cmd == "SYST:ERR?":
            return self.errors.pop(0) if self.errors else '+0,"No error"'
        if cmd.endswith(":RANG:AUTO?"):
            return "1" if self.auto[cmd[:-11]] else "0"
        if cmd.endswith(":RANG?"):
            return f"{self.rng[cmd[:-6]]:+.8E}"
        if cmd.endswith(":NPLC?"):
            return f"{self.nplc[cmd[:-6]]:+.8E}"
        raise AssertionError(f"unexpected query {cmd}")


@pytest.fixture
def dmm():
    d = Keysight_34401A("ASRL4::INSTR")
    d.link = FakeLink()
    d.connect()
    d.link.writes.clear()
    return d


def test_connect_enters_remote_and_checks_identity(dmm):
    assert dmm.model == "34401A" and dmm.firmware == "4-1-1"


def test_wrong_identity_is_refused():
    d = Keysight_34401A("ASRL4::INSTR")
    d.link = FakeLink()
    d.link.query = lambda cmd: "maker,OTHER,0,1"
    with pytest.raises(InstrumentError, match="Not a 34401A"):
        d.connect()


def test_dc_suffix_read_back_is_normalized(dmm):
    assert dmm.get_function() == "VOLT:DC"


def test_meas_without_arguments_reads_only(dmm):
    assert dmm.meas_vdc() == pytest.approx(5.00155)
    assert not any(w.startswith("CONF") for w in dmm.link.writes)


def test_fixed_current_range_turns_autorange_off_and_is_checked(dmm):
    dmm.meas_idc(range=3)
    assert "CONF:CURR:DC 3" in dmm.link.writes
    assert dmm.get_range() == 3 and dmm.get_autorange() is False
    dmm.link.writes.clear()
    dmm.meas_idc()
    assert dmm.link.writes == ["READ?"]


def test_range_that_does_not_read_back_raises(dmm):
    real = dmm.link.write

    def snapping(cmd):
        real(cmd.replace("CONF:VOLT:DC 7", "CONF:VOLT:DC 10"))

    dmm.link.write = snapping
    with pytest.raises(InstrumentError, match="range 7 sent, reads 10"):
        dmm.meas_vdc(range=7)


def test_overload_reads_nan():
    assert math.isnan(_reading("+9.90000000E+37"))
    assert math.isnan(_reading("-9.90000000E+37"))
    assert _reading("-1.5E-03") == -0.0015


def test_nplc_refused_on_ac(dmm):
    with pytest.raises(ValueError, match="no integration time"):
        dmm.set_nplc(1, "VOLT:AC")


def test_errors_are_drained_and_raised(dmm):
    dmm.link.errors = ['-410,"Query INTERRUPTED"']
    with pytest.raises(InstrumentError, match="-410"):
        dmm.check_errors()
    assert dmm.get_errors() == []


def test_buffer_count_limit(dmm):
    with pytest.raises(ValueError, match="1 to 512"):
        dmm.meas_buffer(513)
