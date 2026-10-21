"""The MSO's enums must place what the instrument actually answers.

This scope writes an abbreviation and answers with the full keyword: it is set
with ``SAM`` and answers ``SAMPLE``, set with ``RISe`` and answers ``RISING``.
The long form is not recoverable from the abbreviation by inspection, so the
read-backs below were observed on the bench with projects/mso_probe.py rather
than derived.

If a token table is edited and a member's name or write token stops matching
what the instrument says, ``from_response`` starts raising at the bench. These
tests move that failure to the build.
"""

from __future__ import annotations

import pytest

from drivers.tektronix_mso import enums as E

# Observed 2026-08-12, MSO58 fw 2.6.38.1348, one member per enum: whatever the
# scope happened to be set to. Extend this as more members are observed.
OBSERVED_READBACKS = {
    "AcqType": ("SAMPLE", "SAMPLE"),
    "AcqSequenceMode": ("NUMACQS", "NUMACQS"),
    "TriggerMode": ("AUTO", "AUTO"),
    "Slope": ("RISE", "RISING"),
    "TriggerCoupling": ("DC", "DC"),
    "PulsePolarity": ("POSITIVE", "POSITIVE"),
    "PulseWhen": ("LESSTHAN", "LESSTHAN"),
    "VideoStandard": ("NTSC", "NTSC"),
    "VideoField": ("ALLLINES", "ALL_LINES"),
    "VideoPolarity": ("POSITIVE", "POSITIVE"),
    "TimeoutPolarity": ("STAYSHIGH", "STAYS_HIGH"),
    "RuntWhen": ("OCCURS", "OCCURS"),
    "RuntPolarity": ("POSITIVE", "POSITIVE"),
    "WindowWhen": ("ENTERSWINDOW", "ENTERS"),
    "WindowCrossing": ("UPPER", "UPPER"),
    "TransitionWhen": ("FASTER", "FASTER"),
    "TriggerBBy": ("TIME", "TIME"),
    "HoldoffBy": ("RANDOM", "RANDOM"),
    "Coupling": ("DC", "DC"),
    "HorizontalMode": ("MANUAL", "MANUAL"),
    "HorizontalManualConfig": ("HORIZONTALSCALE", "SCALE"),
    "FastFrameSummary": ("NONE", "NONE"),
    "FastFrameMultiMode": ("OFF", "OFF"),
    "SpectrumMarkerType": ("ABSOLUTE", "ABSOLUTE"),
    "Function": ("PULSE", "PULSE"),
    "AfgOutputMode": ("OFF", "OFF"),
    "Load": ("HIGHZ", "HIGHZ"),
    "RefLevelMethod": ("PERCENT", "PERCENT"),
    "RefLevelPercentType": ("TENNINETY", "TEN_NINETY"),
    "RefLevelAbsoluteType": ("SAME", "SAME"),
    "RefLevelBaseTop": ("AUTO", "AUTO"),
    "RefLevelType": ("GLOBAL", "GLOBAL"),
    "GatingMode": ("SCREEN", "SCREEN"),
    "CursorFunction": ("WAVEFORM", "WAVEFORM"),
    "CursorMode": ("INDEPENDENT", "INDEPENDENT"),
    "ImageComposition": ("NORMAL", "NORMAL"),
    "DataEncoding": ("RIBINARY", "RIBINARY"),
    "AuxOutSource": ("REFOUT", "REFOUT"),
    "AuxOutEdge": ("RISING", "RISING"),
}


def _all_enums():
    for name in dir(E):
        obj = getattr(E, name)
        if isinstance(obj, type) and issubclass(obj, E.ScpiEnum) and obj is not E.ScpiEnum:
            yield name, obj


@pytest.mark.parametrize("cls_name,pair", sorted(OBSERVED_READBACKS.items()))
def test_observed_readback_resolves_to_the_right_member(cls_name, pair):
    """What the scope said maps onto the member the operator would have picked."""
    response, expected_member = pair
    cls = getattr(E, cls_name)
    assert E.from_response(response, cls, cls_name).name == expected_member


@pytest.mark.parametrize("cls_name,cls", sorted(_all_enums()))
def test_every_member_round_trips_from_its_write_token(cls_name, cls):
    """A member's own write token resolves back to that member.

    Weaker than the observed read-backs above, but it holds for every member
    rather than the one each enum happened to be set to.
    """
    for member in cls:
        assert E.from_response(str(member.value), cls, cls_name) is member


@pytest.mark.parametrize("cls_name,cls", sorted(_all_enums()))
def test_token_accepts_member_name_and_raw_token(cls_name, cls):
    """``token()`` takes a member, its name, or the raw SCPI token."""
    for member in cls:
        assert E.token(member, cls, cls_name) == member.value
        assert E.token(member.name, cls, cls_name) == member.value
        assert E.token(str(member.value).lower(), cls, cls_name) == member.value


def test_an_unknown_response_raises_and_names_the_options():
    """A response the driver cannot place is a finding, not a silent default."""
    with pytest.raises(ValueError, match="SAMPLE"):
        E.from_response("NOT_A_REAL_TOKEN", E.AcqType, "acquisition type")
