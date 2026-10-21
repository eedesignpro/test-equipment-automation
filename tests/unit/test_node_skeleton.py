"""Unit tests for the R15 full node skeleton on Tektronix_MSO."""

import pytest

from common.node import Node
from drivers.tektronix_mso import Tektronix_MSO

# _Math is private, so it is imported from the implementation module rather than
# re-exported by the package __init__ (ADR-021).
from drivers.tektronix_mso.driver import _Math

from tests.mocks.mock_visa import MockVisaResource


def _connect(mock):
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    mock.responses.setdefault("*IDN?", lambda _c: "TEKTRONIX,MSO58,C0,CF:91")
    scope.connect()
    mock.writes.clear()
    return scope


_TOP_LEVEL_NODES = [
    "horz",
    "acq",
    "trig",
    "sv",
    "meas",
    "file",
    "data",
    "afg",
    "zoom",
    "cursors",
    "disp",
    "search",
    "dvm",
    "dig",
]

_REF_NODES = ["ref1", "ref2", "ref3", "ref4"]
_BUS_NODES = ["bus1", "bus2", "bus3", "bus4"]
_CHANNELS = [f"ch{n}" for n in range(1, 9)]


@pytest.mark.parametrize("attr", _TOP_LEVEL_NODES)
def test_top_level_node_exists_and_is_a_node(attr):
    scope = _connect(MockVisaResource())
    node = getattr(scope, attr)
    assert isinstance(node, Node)


@pytest.mark.parametrize("attr", _REF_NODES)
def test_reference_nodes_declared_line_by_line(attr):
    scope = _connect(MockVisaResource())
    node = getattr(scope, attr)
    assert isinstance(node, Node)
    assert node.name == f"REF{attr[-1]}"


@pytest.mark.parametrize("attr", _BUS_NODES)
def test_bus_nodes_declared_line_by_line(attr):
    scope = _connect(MockVisaResource())
    node = getattr(scope, attr)
    assert isinstance(node, Node)
    assert node.name == f"BUS{attr[-1]}"


@pytest.mark.parametrize("attr", _CHANNELS)
def test_channel_nodes_have_sv_and_probe_children(attr):
    scope = _connect(MockVisaResource())
    ch = getattr(scope, attr)
    assert isinstance(ch.sv, Node)
    assert isinstance(ch.probe, Node)


def test_math_accessor_creates_and_caches_nodes():
    scope = _connect(MockVisaResource())
    m6 = scope.math(6)
    assert isinstance(m6, _Math)
    assert m6.name == "MATH6"
    assert scope.math(6) is m6  # cached, not re-created
    m1 = scope.math(1)
    assert m1 is not m6


def test_math_accessor_supports_unlimited_index():
    scope = _connect(MockVisaResource())
    m99 = scope.math(99)
    assert m99.name == "MATH99"


def test_math_accessor_rejects_non_positive_index():
    scope = _connect(MockVisaResource())
    with pytest.raises(ValueError):
        scope.math(0)
    with pytest.raises(ValueError):
        scope.math(-1)


def test_afg_dvm_gated_from_runtime_probes():
    """R2 (supersedes ADR-004): afg/dvm install flags come from runtime probes.

    A clean subsystem probe (query succeeds, *ESR? has no CME/EXE bit) marks
    the subsystem installed.
    """
    mock = MockVisaResource(
        responses={
            "*IDN?": "TEKTRONIX,MSO58,C0,CF:91",
            "AFG:FUNCTION?": "SINE",
            "DVM:MODE?": "OFF",
            "*ESR?": "0",
            "ALLEV?": '0,"No events to report - queue empty"',
        }
    )
    scope = _connect(mock)
    assert scope.afg.installed is True
    assert scope.dvm.installed is True


def test_afg_dvm_absent_when_probe_unclean():
    """A probe that leaves a CME/EXE ESR bit set gates the install flag off."""
    mock = MockVisaResource(
        responses={
            "*IDN?": "TEKTRONIX,MSO58,C0,CF:91",
            "AFG:FUNCTION?": "",
            "DVM:MODE?": "",
            "*ESR?": "32",  # CME (command error) bit set
            "ALLEV?": '113,"Undefined header"',
        }
    )
    scope = _connect(mock)
    assert scope.afg.installed is False
    assert scope.dvm.installed is False


def test_dig_not_frozen_at_connect():
    """Digital is hot-pluggable: dig node is not probe-frozen at connect."""
    scope = _connect(MockVisaResource())
    assert scope.dig.installed is True


def test_options_string_recorded_from_opt_query():
    opt_text = "5-ULTIMATE-PER:License; Software Bundle"
    mock = MockVisaResource(responses={"*OPT?": opt_text})
    scope = _connect(mock)
    assert scope.options == opt_text


def test_options_gating_failure_does_not_break_connect():
    class BrokenOptResource(MockVisaResource):
        def query(self, message, **kwargs):
            if message == "*OPT?":
                raise RuntimeError("simulated *OPT? failure")
            return super().query(message, **kwargs)

    mock = BrokenOptResource(responses={"*IDN?": "TEKTRONIX,MSO58,C0,CF:91"})
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    scope.connect()  # must not raise
    assert scope.options == ""
