"""Enums for the Tektronix MSO 4/5/6 driver.

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

from common.scpi import ScpiEnum, member_from_response


# The resolver is shared: three drivers now need it, and the two answer
# conventions it handles are described there.
from_response = member_from_response


# Extra input spellings the driver accepted before these enums existed. Keeping
# them means the conversion takes no capability away from a script that already
# works; they are not member names because they are not valid identifiers, or
# because they read worse than the name chosen.
ALIASES: dict[str, dict[str, str]] = {
    "Slope": {"BOTH": "EITHER"},
    "RefLevelPercentType": {"10-90": "TEN_NINETY", "20-80": "TWENTY_EIGHTY"},
    "AuxOutSource": {
        "TRIGGER": "ATRIGGER",
        "TRIG": "ATRIGGER",
        "REFERENCE": "REFOUT",
        "REF": "REFOUT",
        "10MHZ": "REFOUT",
        "SYNC": "AFG",
    },
}


def token(value, enum_cls: type[ScpiEnum], label: str) -> str:
    """Return the SCPI token to write for a member, a member name, or a token.

    Accepts the member itself, its name with underscores or spaces, the raw
    SCPI token, and any older spelling listed in :data:`ALIASES`.
    """
    if isinstance(value, enum_cls):
        return str(value.value)
    text = str(value).strip().upper()
    alias = ALIASES.get(enum_cls.__name__, {}).get(text)
    if alias is not None:
        return str(enum_cls[alias].value)
    for name, member in enum_cls.__members__.items():
        if text in (name, name.replace("_", " "), str(member.value).upper()):
            return str(member.value)
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"invalid {label} {value!r}; use one of: {valid}")


class AcqType(ScpiEnum):
    """How the acquisition combines samples into a record.

    Observed read-back for one member: 'SAMPLE' (bench, 2026-08-12).
    """

    SAMPLE = "SAM"
    AVERAGE = "AVER"
    PEAK = "PEAK"
    HIRES = "HIR"
    ENVELOPE = "ENV"


class AcqSequenceMode(ScpiEnum):
    """What bounds a single acquisition sequence.

    Observed read-back for one member: 'NUMACQS' (bench, 2026-08-12).
    """

    NUMACQS = "NUMACQs"


class TriggerMode(ScpiEnum):
    """Whether the scope free-runs when no trigger arrives.

    Observed read-back for one member: 'AUTO' (bench, 2026-08-12).
    """

    AUTO = "AUTO"
    NORMAL = "NORM"


class Slope(ScpiEnum):
    """Which edge an edge trigger fires on.

    Observed read-back for one member: 'RISE' (bench, 2026-08-12).
    """

    RISING = "RISE"
    FALLING = "FALL"
    EITHER = "EITHER"


class TriggerCoupling(ScpiEnum):
    """The filter applied to the trigger path only.

    Observed read-back for one member: 'DC' (bench, 2026-08-12).
    """

    DC = "DC"
    HF_REJECT = "HFR"
    LF_REJECT = "LFR"
    NOISE_REJECT = "NOISE"


class Coupling(ScpiEnum):
    """How an input channel is coupled.

    Observed read-back for one member: 'DC' (bench, 2026-08-12).
    """

    AC = "AC"
    DC = "DC"
    DC_REJECT = "DCR"


class PulsePolarity(ScpiEnum):
    """Which pulse sense a width trigger looks for.

    Observed read-back for one member: 'POSITIVE' (bench, 2026-08-12).
    """

    POSITIVE = "POS"
    NEGATIVE = "NEG"


class PulseWhen(ScpiEnum):
    """How a measured pulse width compares to the limits.

    Observed read-back for one member: 'LESSTHAN' (bench, 2026-08-12).
    """

    LESSTHAN = "LESS"
    MORETHAN = "MORE"
    EQUAL = "EQ"
    UNEQUAL = "UNEQ"
    WITHIN = "WIT"
    OUTSIDE = "OUT"


class RuntWhen(ScpiEnum):
    """How a measured runt width compares to the limit.

    Observed read-back for one member: 'OCCURS' (bench, 2026-08-12).
    """

    LESSTHAN = "LESSthan"
    MORETHAN = "MOREthan"
    EQUAL = "EQual"
    UNEQUAL = "UNEQual"
    OCCURS = "OCCURS"


class WindowWhen(ScpiEnum):
    """What the signal must do relative to the window.

    Observed read-back for one member: 'ENTERSWINDOW' (bench, 2026-08-12).
    """

    ENTERS = "ENTERSWindow"
    EXITS = "EXITSWindow"
    INSIDE = "INSIDEGreater"
    OUTSIDE = "OUTSIDEGreater"


class WindowCrossing(ScpiEnum):
    """Which window threshold the crossing is measured on.

    Observed read-back for one member: 'UPPER' (bench, 2026-08-12).
    """

    UPPER = "UPPer"
    LOWER = "LOWer"
    EITHER = "EITher"
    NONE = "NONe"


class TransitionWhen(ScpiEnum):
    """How a measured transition time compares to the limit.

    Observed read-back for one member: 'FASTER' (bench, 2026-08-12).
    """

    SLOWER = "SLOWer"
    FASTER = "FASTer"
    EQUAL = "EQual"
    UNEQUAL = "UNEQual"


class RuntPolarity(ScpiEnum):
    """Which runt sense the trigger looks for.

    Observed read-back for one member: 'POSITIVE' (bench, 2026-08-12).
    """

    POSITIVE = "POSitive"
    NEGATIVE = "NEGative"
    EITHER = "EITher"


class TimeoutPolarity(ScpiEnum):
    """Which level the signal must hold for a timeout.

    Observed read-back for one member: 'STAYSHIGH' (bench, 2026-08-12).
    """

    STAYS_HIGH = "STAYSHigh"
    STAYS_LOW = "STAYSLow"
    EITHER = "EITher"


class VideoStandard(ScpiEnum):
    """The analog video standard being triggered on.

    Observed read-back for one member: 'NTSC' (bench, 2026-08-12).
    """

    NTSC = "NTSc"
    PAL = "PAL"
    SECAM = "SECAM"


class VideoField(ScpiEnum):
    """Which video field or line the trigger selects.

    Observed read-back for one member: 'ALLLINES' (bench, 2026-08-12).
    """

    ALL_LINES = "ALLLines"
    NUMERIC = "NUMERic"
    ODD = "ODD"
    EVEN = "EVEN"
    ALL_FIELDS = "ALLFields"


class VideoPolarity(ScpiEnum):
    """The sync polarity of the video signal.

    Observed read-back for one member: 'POSITIVE' (bench, 2026-08-12).
    """

    POSITIVE = "POSitive"
    NEGATIVE = "NEGative"


class TriggerBBy(ScpiEnum):
    """Whether the B trigger waits for events or for time.

    Observed read-back for one member: 'TIME' (bench, 2026-08-12).
    """

    EVENTS = "EVENTS"
    TIME = "TIMe"


class HoldoffBy(ScpiEnum):
    """Whether trigger holdoff is a set time or randomized.

    Observed read-back for one member: 'RANDOM' (bench, 2026-08-12).
    """

    TIME = "TIMe"
    RANDOM = "RANDom"


class BandwidthOptimization(ScpiEnum):
    """What the bandwidth filter is optimized for."""

    STEP_RESPONSE = "STEPRESPONSE"
    FLATNESS = "FLATNESS"


class ProbeControl(ScpiEnum):
    """Whether probe attenuation is read or set by hand."""

    AUTO = "AUTO"
    MANUAL = "MANual"


class ProbeInputMode(ScpiEnum):
    """Which input a differential probe presents."""

    A = "A"
    B = "B"
    COMMON_MODE = "COMMONMODE"
    DIFFERENTIAL = "DIFFERENTIAL"


class HorizontalMode(ScpiEnum):
    """Whether the scope or the operator owns the timebase.

    Observed read-back for one member: 'MANUAL' (bench, 2026-08-12).
    """

    AUTO = "AUTO"
    MANUAL = "MANual"


class HorizontalManualConfig(ScpiEnum):
    """Which value manual mode holds constant.

    Observed read-back for one member: 'HORIZONTALSCALE' (bench, 2026-08-12).
    """

    SCALE = "HORIZontalscale"
    RECORD_LENGTH = "RECORDLength"


class FastFrameSummary(ScpiEnum):
    """How FastFrame summarizes the captured frames.

    Observed read-back for one member: 'NONE' (bench, 2026-08-12).
    """

    NONE = "NONe"
    AVERAGE = "AVErage"
    ENVELOPE = "ENVelope"


class FastFrameMultiMode(ScpiEnum):
    """How multiple FastFrame frames are drawn.

    Observed read-back for one member: 'OFF' (bench, 2026-08-12).
    """

    OFF = "OFF"
    OVERLAY = "OVERlay"


class SpectrumWindow(ScpiEnum):
    """The FFT window applied to the spectrum."""

    KAISER = "KAIS"
    RECT = "RECT"
    HAMMING = "HAMM"
    HANNING = "HANN"
    BLACKMAN = "BLAC"
    FLATTOP = "FLAT"


class SpectrumUnits(ScpiEnum):
    """The vertical unit of the spectrum trace."""

    DBM = "DBM"
    DBUW = "DBUW"
    DBMV = "DBMV"
    DBUV = "DBUV"
    DBMA = "DBMA"
    DBUA = "DBUA"


class SpectrumTrace(ScpiEnum):
    """A spectrum trace, as named in the display commands."""

    NORMAL = "SV_NORMAL"
    MAX_HOLD = "SV_MAXHOLD"
    MIN_HOLD = "SV_MINHOLD"
    AVERAGE = "SV_AVERAGE"


class SpectrumTraceSelect(ScpiEnum):
    """Which spectrum trace subsequent commands act on."""

    NORMAL = "NORMal"
    MAX_HOLD = "MAXHold"
    MIN_HOLD = "MINHold"
    AVERAGE = "AVErage"


class SpectrumMarkerType(ScpiEnum):
    """Whether a marker reads absolute or relative.

    Observed read-back for one member: 'ABSOLUTE' (bench, 2026-08-12).
    """

    DELTA = "DELta"
    ABSOLUTE = "ABSolute"


class Function(ScpiEnum):
    """The shape the built-in generator puts out.

    Observed read-back for one member: 'PULSE' (bench, 2026-08-12).
    """

    SINE = "SINE"
    SQUARE = "SQUare"
    PULSE = "PULSe"
    RAMP = "RAMP"
    NOISE = "NOISe"
    DC = "DC"
    SINC = "SINC"
    GAUSSIAN = "GAUSsian"
    LORENTZ = "LORENtz"
    ERISE = "ERISe"
    EDECAY = "EDECAy"
    HAVERSINE = "HAVERSINe"
    CARDIAC = "CARDIac"
    ARBITRARY = "ARBitrary"


class AfgOutputMode(ScpiEnum):
    """Whether the generator runs continuously or in bursts.

    Observed read-back for one member: 'OFF' (bench, 2026-08-12).
    """

    OFF = "OFF"
    CONTINUOUS = "CONTinuous"
    BURST = "BURSt"


class Load(ScpiEnum):
    """What the generator assumes its output drives.

    Observed read-back for one member: 'HIGHZ' (bench, 2026-08-12).
    """

    FIFTY = "FIFty"
    HIGHZ = "HIGHZ"


class RefLevelMethod(ScpiEnum):
    """Whether reference levels are percentages or volts.

    Observed read-back for one member: 'PERCENT' (bench, 2026-08-12).
    """

    PERCENT = "PERCent"
    ABSOLUTE = "ABSolute"


class RefLevelPercentType(ScpiEnum):
    """Which percentage pair the edges are measured between.

    Observed read-back for one member: 'TENNINETY' (bench, 2026-08-12).
    """

    TEN_NINETY = "TENNinety"
    TWENTY_EIGHTY = "TWENtyeighty"
    CUSTOM = "CUSTom"


class RefLevelAbsoluteType(ScpiEnum):
    """Whether both edges share one absolute level set.

    Observed read-back for one member: 'SAME' (bench, 2026-08-12).
    """

    SAME = "SAME"
    UNIQUE = "UNIQue"


class RefLevelBaseTop(ScpiEnum):
    """How the top and base of a waveform are found.

    Observed read-back for one member: 'AUTO' (bench, 2026-08-12).
    """

    AUTO = "AUTO"
    MINMAX = "MINMax"
    MEAN_HISTOGRAM = "MEANhistogram"
    MODE_HISTOGRAM = "MODEhistogram"
    EYE_HISTOGRAM = "EYEhistogram"


class RefLevelType(ScpiEnum):
    """Whether reference levels are global or per source.

    Observed read-back for one member: 'GLOBAL' (bench, 2026-08-12).
    """

    GLOBAL = "GLOBal"
    PER_SOURCE = "PERSource"


class GatingMode(ScpiEnum):
    """What restricts the region a measurement runs over.

    Observed read-back for one member: 'SCREEN' (bench, 2026-08-12).
    """

    NONE = "NONE"
    SCREEN = "SCREEN"
    CURSOR = "CURSor"
    LOGIC = "LOGic"
    SEARCH = "SEARCH"
    TIME = "TIMe"


class GatingActive(ScpiEnum):
    """Which logic level opens a logic gate."""

    HIGH = "HIGH"
    LOW = "LOW"


class CursorFunction(ScpiEnum):
    """Which cursor type is displayed.

    Observed read-back for one member: 'WAVEFORM' (bench, 2026-08-12).
    """

    HBARS = "HBArs"
    VBARS = "VBArs"
    SCREEN = "SCREEN"
    WAVEFORM = "WAVEFORM"


class CursorMode(ScpiEnum):
    """Whether the two cursors move together.

    Observed read-back for one member: 'INDEPENDENT' (bench, 2026-08-12).
    """

    TRACK = "TRACK"
    INDEPENDENT = "INDEPENDENT"


class DvmMode(ScpiEnum):
    """What the built-in voltmeter measures."""

    ACRMS = "ACRMS"
    ACDCRMS = "ACDCRMS"
    DC = "DC"
    OFF = "OFF"


class ImageComposition(ScpiEnum):
    """Whether a saved screen image is inverted.

    Observed read-back for one member: 'NORMAL' (bench, 2026-08-12).
    """

    NORMAL = "NORMal"
    INVERTED = "INVErted"


class DataEncoding(ScpiEnum):
    """The wire format of a waveform transfer.

    Observed read-back for one member: 'RIBINARY' (bench, 2026-08-12).
    """

    ASCII = "ASCIi"
    RIBINARY = "RIBinary"
    RPBINARY = "RPBinary"
    FPBINARY = "FPBinary"
    SRIBINARY = "SRIbinary"
    SRPBINARY = "SRPbinary"
    SFPBINARY = "SFPbinary"


class AuxOutSource(ScpiEnum):
    """What drives the rear Aux Out connector.

    Observed read-back for one member: 'REFOUT' (bench, 2026-08-12).
    """

    ATRIGGER = "ATRIGger"
    REFOUT = "REFOut"
    AFG = "AFG"


class AuxOutEdge(ScpiEnum):
    """Which edge the Aux Out pulse presents.

    Observed read-back for one member: 'RISING' (bench, 2026-08-12).
    """

    RISING = "RISe"
    FALLING = "FALl"
