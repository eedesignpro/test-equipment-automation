"""Generate src/drivers/tektronix_mso/enums.py from the driver's token dicts.

Token sets come from the dicts the driver already sends, so they cannot drift
from known-good SCPI. Only the class names and member names are hand-chosen.
"""

import ast
import pathlib

DRIVER = pathlib.Path("src/drivers/tektronix_mso/driver.py")
OUT = pathlib.Path("src/drivers/tektronix_mso/enums.py")

# dict -> (ClassName, one-line purpose)
NAMES = {
    "_ACQ_TYPE": ("AcqType", "How the acquisition combines samples into a record."),
    "_ACQ_SEQUENCE_MODE": ("AcqSequenceMode", "What bounds a single acquisition sequence."),
    "_TRIG_MODE": ("TriggerMode", "Whether the scope free-runs when no trigger arrives."),
    "_EDGE_SLOPE": ("Slope", "Which edge an edge trigger fires on."),
    "_EDGE_COUPLING": ("TriggerCoupling", "The filter applied to the trigger path only."),
    "_CH_COUPLING": ("Coupling", "How an input channel is coupled."),
    "_PULSE_POLARITY": ("PulsePolarity", "Which pulse sense a width trigger looks for."),
    "_PULSE_WHEN": ("PulseWhen", "How a measured pulse width compares to the limits."),
    "_RUNT_WHEN": ("RuntWhen", "How a measured runt width compares to the limit."),
    "_WINDOW_WHEN": ("WindowWhen", "What the signal must do relative to the window."),
    "_WINDOW_CROSSING": ("WindowCrossing", "Which window threshold the crossing is measured on."),
    "_TRANSITION_WHEN": ("TransitionWhen", "How a measured transition time compares to the limit."),
    "_EDGE_POLARITY": ("RuntPolarity", "Which runt sense the trigger looks for."),
    "_TIMEOUT_POLARITY": ("TimeoutPolarity", "Which level the signal must hold for a timeout."),
    "_VIDEO_STANDARD": ("VideoStandard", "The analog video standard being triggered on."),
    "_VIDEO_FIELD": ("VideoField", "Which video field or line the trigger selects."),
    "_VIDEO_POLARITY": ("VideoPolarity", "The sync polarity of the video signal."),
    "_TRIG_B_BY": ("TriggerBBy", "Whether the B trigger waits for events or for time."),
    "_TRIG_HOLDOFF_BY": ("HoldoffBy", "Whether trigger holdoff is a set time or randomized."),
    "_BW_OPTIMIZATION": ("BandwidthOptimization", "What the bandwidth filter is optimized for."),
    "_PROBE_CONTROL": ("ProbeControl", "Whether probe attenuation is read or set by hand."),
    "_PROBE_INPUT_MODE": ("ProbeInputMode", "Which input a differential probe presents."),
    "_HORIZ_MODE": ("HorizontalMode", "Whether the scope or the operator owns the timebase."),
    "_HORIZ_MANUAL_CONFIG": ("HorizontalManualConfig", "Which value manual mode holds constant."),
    "_FASTFRAME_SUMFRAME": ("FastFrameSummary", "How FastFrame summarizes the captured frames."),
    "_FASTFRAME_MULTI_MODE": ("FastFrameMultiMode", "How multiple FastFrame frames are drawn."),
    "_SV_WINDOW": ("SpectrumWindow", "The FFT window applied to the spectrum."),
    "_SV_UNITS": ("SpectrumUnits", "The vertical unit of the spectrum trace."),
    "_SV_TRACE": ("SpectrumTrace", "A spectrum trace, as named in the display commands."),
    "_SV_TRACE_SELECT": ("SpectrumTraceSelect", "Which spectrum trace subsequent commands act on."),
    "_SV_MARKER_TYPE": ("SpectrumMarkerType", "Whether a marker reads absolute or relative."),
    "_AFG_FUNCTION": ("Function", "The shape the built-in generator puts out."),
    "_AFG_OUTPUT_MODE": ("AfgOutputMode", "Whether the generator runs continuously or in bursts."),
    "_AFG_LOAD_IMPEDANCE": ("Load", "What the generator assumes its output drives."),
    "_REFLEVEL_METHOD": ("RefLevelMethod", "Whether reference levels are percentages or volts."),
    "_REFLEVEL_PERCENT_TYPE": (
        "RefLevelPercentType",
        "Which percentage pair the edges are measured between.",
    ),
    "_REFLEVEL_ABSOLUTE_TYPE": (
        "RefLevelAbsoluteType",
        "Whether both edges share one absolute level set.",
    ),
    "_REFLEVEL_BASETOP": ("RefLevelBaseTop", "How the top and base of a waveform are found."),
    "_REFLEVEL_TYPE": ("RefLevelType", "Whether reference levels are global or per source."),
    "_GATING_MODE": ("GatingMode", "What restricts the region a measurement runs over."),
    "_GATING_ACTIVE": ("GatingActive", "Which logic level opens a logic gate."),
    "_CURSOR_FUNCTION": ("CursorFunction", "Which cursor type is displayed."),
    "_CURSOR_MODE": ("CursorMode", "Whether the two cursors move together."),
    "_DVM_MODE": ("DvmMode", "What the built-in voltmeter measures."),
    "_IMAGE_COMPOSITION": ("ImageComposition", "Whether a saved screen image is inverted."),
    "_DATA_ENCODING": ("DataEncoding", "The wire format of a waveform transfer."),
    "_AUXOUT_SOURCE": ("AuxOutSource", "What drives the rear Aux Out connector."),
    "_AUXOUT_EDGE": ("AuxOutEdge", "Which edge the Aux Out pulse presents."),
}

# member name overrides, where the dict's friendliest key is a poor name
MEMBERS = {
    "_EDGE_SLOPE": {"EITHER": "EITHER", "BOTH": "EITHER"},
    "_AUXOUT_SOURCE": {"ATRIGger": "ATRIGGER", "REFOut": "REFOUT", "AFG": "AFG"},
    "_REFLEVEL_PERCENT_TYPE": {
        "TENNinety": "TEN_NINETY",
        "TWENtyeighty": "TWENTY_EIGHTY",
        "CUSTom": "CUSTOM",
    },
    "_SV_TRACE": {
        "SV_NORMAL": "NORMAL",
        "SV_MAXHOLD": "MAX_HOLD",
        "SV_MINHOLD": "MIN_HOLD",
        "SV_AVERAGE": "AVERAGE",
    },
}

# read-backs observed on the bench 2026-08-12 with projects/mso_probe.py
OBSERVED = {
    "_ACQ_TYPE": "SAMPLE",
    "_ACQ_SEQUENCE_MODE": "NUMACQS",
    "_TRIG_MODE": "AUTO",
    "_EDGE_SLOPE": "RISE",
    "_EDGE_COUPLING": "DC",
    "_PULSE_POLARITY": "POSITIVE",
    "_PULSE_WHEN": "LESSTHAN",
    "_VIDEO_STANDARD": "NTSC",
    "_VIDEO_FIELD": "ALLLINES",
    "_VIDEO_POLARITY": "POSITIVE",
    "_TIMEOUT_POLARITY": "STAYSHIGH",
    "_RUNT_WHEN": "OCCURS",
    "_EDGE_POLARITY": "POSITIVE",
    "_WINDOW_WHEN": "ENTERSWINDOW",
    "_WINDOW_CROSSING": "UPPER",
    "_TRANSITION_WHEN": "FASTER",
    "_TRIG_B_BY": "TIME",
    "_TRIG_HOLDOFF_BY": "RANDOM",
    "_CH_COUPLING": "DC",
    "_HORIZ_MODE": "MANUAL",
    "_HORIZ_MANUAL_CONFIG": "HORIZONTALSCALE",
    "_FASTFRAME_SUMFRAME": "NONE",
    "_FASTFRAME_MULTI_MODE": "OFF",
    "_SV_MARKER_TYPE": "ABSOLUTE",
    "_AFG_FUNCTION": "PULSE",
    "_AFG_OUTPUT_MODE": "OFF",
    "_AFG_LOAD_IMPEDANCE": "HIGHZ",
    "_REFLEVEL_METHOD": "PERCENT",
    "_REFLEVEL_PERCENT_TYPE": "TENNINETY",
    "_REFLEVEL_ABSOLUTE_TYPE": "SAME",
    "_REFLEVEL_BASETOP": "AUTO",
    "_REFLEVEL_TYPE": "GLOBAL",
    "_GATING_MODE": "SCREEN",
    "_CURSOR_FUNCTION": "WAVEFORM",
    "_CURSOR_MODE": "INDEPENDENT",
    "_IMAGE_COMPOSITION": "NORMAL",
    "_DATA_ENCODING": "RIBINARY",
    "_AUXOUT_SOURCE": "REFOUT",
    "_AUXOUT_EDGE": "RISING",
}

HEADER = '''"""Enums for the Tektronix MSO 4/5/6 driver.

Multi-option arguments take one of these rather than a bare string, so the
options are discoverable by autocomplete instead of by reading the manual.
Every method that takes one names the path in its docstring, as
``Enum arg: <scope>.AcqType.<TAB>``.

**A member's value is the token the driver WRITES, not the one the instrument
answers with.** Those differ here, unlike on the 33522A. ``ACQuire:MODe`` is set
with ``SAM`` and answers ``SAMPLE``; ``AUXout:EDGE`` is set with ``RISe`` and
answers ``RISING``. Read-backs were observed on the bench 2026-08-12 with
projects/mso_probe.py rather than derived from the write tokens, because the
long form of an abbreviation is not recoverable by inspection.

:func:`from_response` resolves an answer back to a member by matching the
uppercased write token or the member name with underscores removed. That rule
covers all 39 read-backs observed. Anything else raises, because a response the
driver cannot place is a finding, not something to guess at.

Generated from the driver's own token tables; see the scratchpad generator
recorded in ADR-051. Hand-edit freely from here on.
"""

from __future__ import annotations

from common.scpi import ScpiEnum


def from_response(response: str, enum_cls: type[ScpiEnum], label: str) -> ScpiEnum:
    """Resolve an instrument answer to a member of ``enum_cls``.

    The instrument answers with the full keyword in upper case, while the
    driver writes an abbreviation, so this matches on either the uppercased
    write token or the member name with underscores stripped.
    """
    text = response.strip().upper().strip('"')
    for member in enum_cls:
        if text == str(member.value).upper():
            return member
    for member in enum_cls:
        if text == member.name.replace("_", ""):
            return member
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(
        f"the instrument answered {response!r} for {label}, which is not a known "
        f"{enum_cls.__name__}; expected one of: {valid}"
    )


def token(value, enum_cls: type[ScpiEnum], label: str) -> str:
    """Return the SCPI token to write for a member, a member name, or a token."""
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    for member in enum_cls:
        if text in (member.name, member.name.replace("_", " "), str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")
'''


def member_name(dict_name, key, value):
    override = MEMBERS.get(dict_name, {})
    if value in override:
        return override[value]
    if key in override:
        return override[key]
    name = key.upper().replace(" ", "_").replace("-", "_")
    if name and name[0].isdigit():
        name = "N" + name
    return name


def main():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    dicts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Dict):
                try:
                    dicts[node.targets[0].id] = ast.literal_eval(node.value)
                except Exception:
                    pass

    out = [HEADER]
    for dict_name, (cls_name, purpose) in NAMES.items():
        d = dicts.get(dict_name)
        if d is None:
            print(f"  MISSING dict {dict_name}")
            continue
        by_val = {}
        for k, v in d.items():
            by_val.setdefault(v, []).append(k)
        lines = [f"\n\nclass {cls_name}(ScpiEnum):", f'    """{purpose}']
        obs = OBSERVED.get(dict_name)
        if obs:
            lines.append("")
            lines.append(f"    Observed read-back for one member: {obs!r} (bench, 2026-08-12).")
        lines.append('    """')
        lines.append("")
        seen = set()
        for value, keys in by_val.items():
            name = member_name(dict_name, sorted(keys, key=len, reverse=True)[0], value)
            if name in seen:
                continue
            seen.add(name)
            lines.append(f'    {name} = "{value}"')
        out.append("\n".join(lines))
    OUT.write_text("".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(NAMES)} enums")


if __name__ == "__main__":
    main()
