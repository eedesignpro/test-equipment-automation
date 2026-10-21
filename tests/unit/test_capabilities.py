"""Unit tests for src/drivers/capabilities.py (R2).

``parse_opt`` is pure-Python logic tested directly against the real ``*OPT?``
text captured off the bench (tests/captures/capabilities_data_sv_trace.json /
meas_dispatcher_trace.json), per the QA policy allowing direct testing of pure
parsing logic. The runtime probes (``probe_afg``/``probe_dvm``/
``probe_channel_digital``) are exercised through a real ``InstrumentLink``
wired to ``MockVisaResource`` (the ``make_link`` fixture) - no capture trace
exists in raw op/cmd/resp form for these (afg_subsystem_trace.json and
capabilities_data_sv_trace.json are summary-only), so the exact-value
sequences below are reconstructed from the bench-derived facts recorded in
that summary (ADR-012, capabilities_data_sv_trace.json).
"""

import pytest

from drivers.tektronix_mso.capabilities import (
    BUNDLE_CONTENTS,
    Capabilities,
    parse_opt,
    probe_afg,
    probe_channel_digital,
    probe_dvm,
)

# Real *OPT? text captured off the bench MSO58 (meas_dispatcher_trace.json /
# capabilities_data_sv_trace.json "opt"/"opt_raw" field).
_BENCH_OPT_TEXT = (
    "5-ULTIMATE-PER:License; Software Bundle; Ultimate Bundle for 5 Series MSO; "
    "Requires 5-WIN; Perpetual License; Node Locked,"
    "BW5-2000:License; 2 GHz bandwidth on 5 Series Oscilloscopes; Node locked,"
    "LIC5-SREMBD:License (installed option); Embedded serial triggering and "
    "analysis (I2C, SPI); Node locked,"
    "LIC5-VID:License; Analog Video Trigger(NTSC, PAL, SECAM); Node Locked,"
    "LIC5-PWR:License (installed option); Advanced power analysis; Node locked"
)


def test_parse_opt_bench_capture_options_and_bandwidth():
    """Matches capabilities_data_sv_trace.json caps_options/caps_bandwidth_mhz."""
    caps = parse_opt(_BENCH_OPT_TEXT)
    assert caps.options == ["ULTIMATE", "BW5-2000", "SREMBD", "VID", "PWR"]
    assert caps.bandwidth_mhz == 2000
    assert caps.raw == _BENCH_OPT_TEXT


def test_parse_opt_bench_capture_features_expand_from_ultimate():
    """Matches capabilities_data_sv_trace.json caps_features (ULTIMATE bundle)."""
    caps = parse_opt(_BENCH_OPT_TEXT)
    assert caps.features == frozenset(
        {"AFG", "BW5-2000", "PWR", "SRCOMP", "SREMBD", "SVRFVT", "ULTIMATE", "VID"}
    )
    assert caps.has("afg")
    assert caps.has("AFG")
    assert not caps.has("dvm")


def test_parse_opt_empty_and_none_are_safe():
    assert parse_opt("").options == []
    assert parse_opt(None).options == []
    assert parse_opt(None).features == frozenset()


@pytest.mark.parametrize(
    "prefix,suffix",
    [("5-", "-PER"), ("SUP5-", "-1Y"), ("LIC5-", "-FL"), ("", "")],
)
def test_parse_opt_normalizes_prefixes_and_suffixes(prefix, suffix):
    caps = parse_opt(f"{prefix}STARTER{suffix}:License")
    assert caps.options == ["STARTER"]
    assert caps.features & BUNDLE_CONTENTS["STARTER"] == BUNDLE_CONTENTS["STARTER"]


def test_parse_opt_does_not_split_prose_containing_commas():
    """The split regex must not break on commas inside a license description."""
    text = "LIC5-VID:License; Analog Video Trigger(NTSC, PAL, SECAM); Node Locked"
    caps = parse_opt(text)
    assert caps.options == ["VID"]


def test_starter_bundle_is_subset_of_ultimate():
    assert BUNDLE_CONTENTS["STARTER"] <= BUNDLE_CONTENTS["ULTIMATE"]


def test_capabilities_has_is_case_insensitive():
    caps = Capabilities(raw="", options=[], bandwidth_mhz=None, features=frozenset({"AFG"}))
    assert caps.has("afg")
    assert caps.has("AFG")
    assert not caps.has("dvm")


# ---------------------------------------------------------------------------
# Runtime probes (authoritative gate per ADR-012, supersedes ADR-004)
# ---------------------------------------------------------------------------


def test_probe_afg_true_on_clean_query(make_link):
    """Bench fact: AFG present -> AFG:FUNCTION? answers cleanly, ESR 0."""
    link, _ = make_link(
        {"AFG:FUNCTION?": "SINE", "*ESR?": "0", "ALLEV?": '0,"No events to report - queue empty"'}
    )
    assert probe_afg(link) is True


def test_probe_afg_false_when_esr_reports_cme(make_link):
    link, _ = make_link({"AFG:FUNCTION?": "", "*ESR?": "32", "ALLEV?": '113,"Undefined header"'})
    assert probe_afg(link) is False


def test_probe_dvm_false_on_bench_when_not_registered(make_link):
    """Bench fact: DVM not enabled on this unit -> probe False (dvm_installed_probe: false)."""
    link, _ = make_link({"DVM:MODE?": "", "*ESR?": "32", "ALLEV?": '113,"Undefined header"'})
    assert probe_dvm(link) is False


def test_probe_afg_false_when_query_raises():
    class RaisingLink:
        def write(self, *_a, **_k):
            raise RuntimeError("simulated I/O failure")

        def query(self, *_a, **_k):
            raise RuntimeError("simulated I/O failure")

        def query_int(self, *_a, **_k):
            raise RuntimeError("simulated I/O failure")

    assert probe_afg(RaisingLink()) is False
    assert probe_dvm(RaisingLink()) is False


def test_probe_channel_digital_false_for_analog_probetype(make_link):
    """Bench fact: CH1/CH4/CH5 all report PROBETYPE ANALOG -> not digital."""
    link, _ = make_link(
        {"CH5:PROBETYPE?": "ANALOG", "ALLEV?": '0,"No events to report - queue empty"'}
    )
    assert probe_channel_digital(link, 5) is False


def test_probe_channel_digital_true_when_response_contains_dig(make_link):
    link, _ = make_link({"CH3:PROBETYPE?": "DIGITAL", "ALLEV?": '0,"No events"'})
    assert probe_channel_digital(link, 3) is True


def test_probe_channel_digital_never_raises_on_query_failure():
    class RaisingLink:
        def write(self, *_a, **_k):
            raise RuntimeError("boom")

        def query(self, *_a, **_k):
            raise RuntimeError("boom")

    assert probe_channel_digital(RaisingLink(), 1) is False
