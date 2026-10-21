"""Horizontal mode semantics (bench-verified 2026-07-23 on the MSO58).

AUTO: scale is the only writable knob; SR/RL writes are silently ignored by
the scope (no SCPI error), so the driver must refuse them itself.
MANUAL: 10 * scale = record_length / sample_rate; setters never change mode.
"""

import pytest

from common.errors import InstrumentError
from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource


def _scope(monkeypatch, responses=None):
    base = {"*IDN?": "TEKTRONIX,MSO58,C0,CF:91", "*OPC?": "1"}
    base.update(responses or {})
    mock = MockVisaResource(responses=base)
    scope = Tektronix_MSO("MOCK::INSTR")
    monkeypatch.setattr(scope.link, "_open_resource", lambda rm, name: mock)
    scope.connect()
    return scope, mock


def test_set_samplerate_raises_in_auto_mode(monkeypatch):
    scope, mock = _scope(monkeypatch, {"HOR:MODE?": "AUTO"})
    with pytest.raises(InstrumentError, match="silently ignores"):
        scope.horz.set_samplerate(6.25e6)
    assert not any(w.startswith("HOR:MOD:SAMPLER") for w in mock.writes)


def test_set_record_length_raises_in_auto_mode(monkeypatch):
    scope, mock = _scope(monkeypatch, {"HOR:MODE?": "AUTO"})
    with pytest.raises(InstrumentError, match="silently ignores"):
        scope.horz.set_record_length(1000)
    assert not any(w.startswith("HOR:RECO") for w in mock.writes)


def test_setters_allowed_in_manual_mode(monkeypatch):
    scope, mock = _scope(
        monkeypatch,
        {"HOR:MODE?": "MANUAL", "HOR:MOD:SAMPLER?": "6.25E+6", "HOR:RECO?": "62500"},
    )
    assert scope.horz.set_samplerate(6.25e6) == 6.25e6
    assert scope.horz.set_record_length(62500) == 62500
    assert not any(w.startswith("HOR:MODE ") for w in mock.writes)


def test_set_scale_never_touches_mode(monkeypatch):
    scope, mock = _scope(monkeypatch, {"HOR:MOD:SCA?": "0.001"})
    assert scope.horz.set_scale(1e-3) == 1e-3
    assert not any(w.startswith("HOR:MODE ") for w in mock.writes)


def test_set_manual_sets_mode_then_both_values(monkeypatch):
    scope, mock = _scope(monkeypatch, {"HOR:MOD:SAMPLER?": "6.25E+6", "HOR:RECO?": "62500"})
    sr, rl = scope.horz.set_manual(6.25e6, 62500)
    assert (sr, rl) == (6.25e6, 62500)
    i_mode = mock.writes.index("HOR:MODE MAN")
    i_sr = next(i for i, w in enumerate(mock.writes) if w.startswith("HOR:MOD:SAMPLER"))
    i_rl = next(i for i, w in enumerate(mock.writes) if w.startswith("HOR:RECO"))
    assert i_mode < i_sr < i_rl


def test_restore_auto_orders_mode_before_scale(monkeypatch):
    scope, mock = _scope(monkeypatch)
    snap = {
        "mode": "AUTO",
        "scale": 4e-4,
        "samplerate": 312.5e6,
        "record_length": 1.25e6,
        "position": 50.0,
        "delay_mode": False,
        "delay_time": 0.0,
    }
    scope.horz.restore(snap)
    i_mode = mock.writes.index("HOR:MODE AUTO")
    i_scale = next(i for i, w in enumerate(mock.writes) if w.startswith("HOR:MOD:SCA "))
    assert i_mode < i_scale
    assert not any(w.startswith("HOR:MOD:SAMPLER") for w in mock.writes)
    assert "HOR:DEL:MOD OFF" in mock.writes


def test_restore_manual_restores_sr_and_rl(monkeypatch):
    scope, mock = _scope(monkeypatch)
    snap = {
        "mode": "MANUAL",
        "scale": 5e-3,
        "samplerate": 6.25e6,
        "record_length": 312500,
        "position": 10.0,
        "delay_mode": True,
        "delay_time": 1e-3,
    }
    scope.horz.restore(snap)
    assert "HOR:MODE MAN" in mock.writes
    assert any(w.startswith("HOR:MOD:SAMPLER") for w in mock.writes)
    assert any(w.startswith("HOR:RECO") for w in mock.writes)
    assert "HOR:DEL:MOD ON" in mock.writes


def _write_index(mock, prefix):
    return next(i for i, w in enumerate(mock.writes) if w.startswith(prefix))


def test_configure_sr_scale_writes_scale_last(monkeypatch):
    scope, mock = _scope(
        monkeypatch,
        {
            "HOR:MOD:SCA?": "0.002",
            "HOR:MOD:SAMPLER?": "6.25E+6",
            "HOR:RECO?": "125000",
            "HOR:MODE?": "MANUAL",
            "HOR:POS?": "10.0",
        },
    )
    scope.horz.set_timebase(samplerate=6.25e6, scale=2e-3)
    assert _write_index(mock, "HOR:MOD:SAMPLER") < _write_index(mock, "HOR:MOD:SCA ")


def test_configure_scale_rl_derives_sr(monkeypatch):
    scope, mock = _scope(
        monkeypatch,
        {
            "HOR:DIV?": "10",
            "HOR:MOD:SCA?": "0.002",
            "HOR:MOD:SAMPLER?": "3.125E+6",
            "HOR:RECO?": "62500",
            "HOR:MODE?": "MANUAL",
            "HOR:POS?": "10.0",
        },
    )
    scope.horz.set_timebase(scale=2e-3, record_length=62500)
    sr_write = next(w for w in mock.writes if w.startswith("HOR:MOD:SAMPLER"))
    assert float(sr_write.split()[1]) == 62500 / (10 * 2e-3)
    assert _write_index(mock, "HOR:MOD:SAMPLER") < _write_index(mock, "HOR:RECO")
    assert not any(w.startswith("HOR:MOD:SCA ") for w in mock.writes)


def test_configure_requires_exactly_two(monkeypatch):
    scope, _ = _scope(monkeypatch)
    with pytest.raises(ValueError, match="exactly two"):
        scope.horz.set_timebase(scale=1e-3)
    with pytest.raises(ValueError, match="exactly two"):
        scope.horz.set_timebase(scale=1e-3, samplerate=1e6, record_length=1000)
