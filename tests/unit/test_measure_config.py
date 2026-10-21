"""Unit tests for R10 measurement configuration (reference levels, gating,
stats completion, delete_all, slot hygiene).

Query responses replay real MSO58 traffic captured in
tests/captures/meas_config_trace.json (bench session 2026-07-23).
"""

import pytest

from tests.mocks.capture_replay import connect_replay

TRACE = "tek_mso/meas_config_trace.json"


# --- reference levels ------------------------------------------------------


def test_ref_levels_set_method_command_and_readback():
    scope, mock = connect_replay(TRACE)
    result = scope.meas.ref_levels.set_method("PERCENT")
    assert "MEASUrement:REFLevels:METHod PERCent" in mock.writes
    assert result == "PERCENT"


def test_ref_levels_invalid_method_raises():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.meas.ref_levels.set_method("BOGUS")


def test_ref_levels_percent_type_maps_10_90():
    scope, mock = connect_replay(TRACE)
    scope.meas.ref_levels.set_percent_type("10-90")
    assert "MEASUrement:REFLevels:PERCent:TYPE TENNinety" in mock.writes


def test_ref_levels_percent_type_maps_20_80():
    scope, mock = connect_replay(TRACE)
    scope.meas.ref_levels.set_percent_type("20-80")
    assert "MEASUrement:REFLevels:PERCent:TYPE TWENtyeighty" in mock.writes


def test_ref_levels_set_percent_levels_writes_only_given():
    scope, mock = connect_replay(TRACE)
    scope.meas.ref_levels.set_percent_levels(rise_high=85, fall_low=15)
    assert "MEASUrement:REFLevels:PERCent:RISEHigh 85" in mock.writes
    assert "MEASUrement:REFLevels:PERCent:FALLLow 15" in mock.writes
    # no SET of an unspecified level was issued (a readback query is not a set)
    assert not any(w.startswith("MEASUrement:REFLevels:PERCent:RISEMid ") for w in mock.writes)


def test_ref_levels_set_absolute_levels_command():
    scope, mock = connect_replay(TRACE)
    scope.meas.ref_levels.set_absolute_levels(rise_mid=0.0)
    assert "MEASUrement:REFLevels:ABSolute:RISEMid 0" in mock.writes


def test_ref_levels_hysteresis_percent_branch():
    scope, mock = connect_replay(TRACE)
    result = scope.meas.ref_levels.set_hysteresis(5.0, method="PERCENT")
    assert "MEASUrement:REFLevels:PERCent:HYSTeresis 5" in mock.writes
    assert result == pytest.approx(5.0)


def test_ref_levels_hysteresis_absolute_branch_builds_command():
    scope, mock = connect_replay(TRACE)
    # ABSolute hysteresis readback was not part of the captured session; assert
    # the correct SET command is built (the readback path is exercised by the
    # PERCENT branch above).
    with pytest.raises(Exception):  # noqa: B017 - empty readback raises on this mock
        scope.meas.ref_levels.set_hysteresis(0.1, method="ABSOLUTE")
    assert "MEASUrement:REFLevels:ABSolute:HYSTeresis 0.1" in mock.writes


def test_ref_levels_basetop_and_source_type():
    scope, mock = connect_replay(TRACE)
    scope.meas.ref_levels.set_basetop("AUTO")
    scope.meas.ref_levels.set_source_type("GLOBAL")
    assert "MEASUrement:REFLevels:BASETop AUTO" in mock.writes
    assert "MEASUrement:REFLevels:TYPE GLOBal" in mock.writes


def test_source_type_only_on_global_node():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.meas.ref_levels_ch(5).set_source_type("GLOBAL")


def test_per_channel_ref_levels_prefix():
    scope, mock = connect_replay(TRACE)
    scope.meas.ref_levels_ch(5).set_method("PERCENT")
    assert "MEASUrement:CH5:REFLevels:METHod PERCent" in mock.writes


def test_per_measurement_ref_levels_indexed_prefix():
    scope, mock = connect_replay(TRACE)
    scope.meas.ref_levels_meas(1, 1).set_method("ABSOLUTE")
    assert "MEASUrement:MEAS1:REFLevels1:METHod ABSolute" in mock.writes


# --- gating ----------------------------------------------------------------


def test_gating_mode_command_and_readback():
    scope, mock = connect_replay(TRACE)
    result = scope.meas.gating.set_mode("TIME")
    assert "MEASUrement:GATing TIMe" in mock.writes
    assert result == "NONE"  # final captured state (mode reset to NONE)


def test_gating_time_window():
    scope, mock = connect_replay(TRACE)
    start, end = scope.meas.gating.set_time_window(-2e-6, 2e-6)
    assert "MEASUrement:GATing:STARTtime -2e-06" in mock.writes
    assert "MEASUrement:GATing:ENDtime 2e-06" in mock.writes
    assert (start, end) == (pytest.approx(-2e-6), pytest.approx(2e-6))


def test_gating_active_maps():
    scope, mock = connect_replay(TRACE)
    scope.meas.gating.set_active("HIGH")
    assert "MEASUrement:GATing:ACTive HIGH" in mock.writes


def test_per_meas_gating_global_command():
    scope, mock = connect_replay(TRACE)
    scope.meas.gating_meas(1).set_global(True)
    assert "MEASUrement:MEAS1:GATing:GLOBal ON" in mock.writes


def test_global_gating_has_no_global_method():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.meas.gating.set_global(True)


# --- stats completion / hygiene / delete_all -------------------------------


def test_delete_all_command():
    scope, mock = connect_replay(TRACE)
    scope.meas.cmd_delete_all()
    assert "MEASUrement:DELETEALL" in mock.writes


def test_current_acq_stat_queries():
    scope, mock = connect_replay(TRACE)
    scope.meas.get_curr_max(2)
    scope.meas.get_curr_pk2pk(2)
    scope.meas.get_curr_std(2)
    scope.meas.get_allacqs_pk2pk(2)
    assert "MEASUrement:MEAS2:RESUlts:CURRentacq:MAXimum?" in mock.writes
    assert "MEASUrement:MEAS2:RESUlts:CURRentacq:PK2PK?" in mock.writes
    assert "MEASUrement:MEAS2:RESUlts:CURRentacq:STDDev?" in mock.writes
    assert "MEASUrement:MEAS2:RESUlts:ALLAcqs:PK2PK?" in mock.writes


def test_set_and_get_label_strips_quotes():
    scope, mock = connect_replay(TRACE)
    assert scope.meas.set_label(2, "amp5") == "amp5"
    assert 'MEASUrement:MEAS2:LABel "amp5"' in mock.writes
    assert scope.meas.get_label(2) == "amp5"


def test_label_rejects_embedded_quote():
    scope, _ = connect_replay(TRACE)
    with pytest.raises(ValueError):
        scope.meas.set_label(2, 'a"b')


def test_status_and_unit_queries():
    scope, _ = connect_replay(TRACE)
    assert scope.meas.get_status(2) == "FAIL"
    assert scope.meas.get_xunit(2) == "s"
    assert scope.meas.get_yunit(2) == "V"
