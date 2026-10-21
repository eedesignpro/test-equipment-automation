"""Driver for Tektronix MSO 4/5/6 Series Mixed Signal Oscilloscopes.

Common tasks:
  1. Set up to look at a 1 kHz signal on CH1:
         scope.ch1.set_scale(0.5)
         scope.horz.set_timebase(scale=1e-3, record_length=100_000)
  2. Trigger on a rising edge:
         scope.trig.set_edge("CH1", level=0.0, slope=scope.Slope.RISING)
  3. Measure it:
         scope.meas.frequency("CH1")
         scope.meas.amplitude("CH1")     # High minus Low, ignores overshoot
  4. Capture one acquisition and pull the samples back:
         scope.acq.cmd_single()
         samples = scope.data.get_waveform("CH1")
         t = scope.data.get_time_axis()
  5. Save a screenshot of what is on the display:
         scope.file.get_screenshot("check")
"""

import contextlib
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

from common.errors import (
    InstrumentError,
    InstrumentQueryError,
    InstrumentTimeoutError,
    InstrumentWarning,
)
from common.instrument import Instrument
from common.link import InstrumentLink
from common.node import Node
from common.scpi import parse_compound, quote

from .enums import (
    AcqSequenceMode,
    AcqType,
    AfgOutputMode,
    AuxOutEdge,
    AuxOutSource,
    BandwidthOptimization,
    Coupling,
    CursorFunction,
    CursorMode,
    DataEncoding,
    DvmMode,
    FastFrameMultiMode,
    FastFrameSummary,
    Function,
    GatingActive,
    GatingMode,
    HoldoffBy,
    HorizontalManualConfig,
    HorizontalMode,
    ImageComposition,
    Load,
    ProbeControl,
    ProbeInputMode,
    PulsePolarity,
    PulseWhen,
    RefLevelAbsoluteType,
    RefLevelBaseTop,
    RefLevelMethod,
    RefLevelPercentType,
    RefLevelType,
    RuntPolarity,
    RuntWhen,
    Slope,
    SpectrumMarkerType,
    SpectrumTrace,
    SpectrumTraceSelect,
    SpectrumUnits,
    SpectrumWindow,
    TimeoutPolarity,
    TransitionWhen,
    TriggerBBy,
    TriggerCoupling,
    TriggerMode,
    VideoField,
    VideoPolarity,
    VideoStandard,
    WindowCrossing,
    WindowWhen,
    from_response,
    token,
)
from common.state import TimebaseState, WaveformState
from drivers.tektronix_common import wfm_reader
from drivers.tektronix_mso.capabilities import (
    parse_opt,
    probe_afg,
    probe_channel_digital,
    probe_dvm,
)

_ACQ_TYPE = {
    "SAMPLE": "SAM",
    "AVERAGE": "AVER",
    "PEAK": "PEAK",
    "HIRES": "HIR",
    "ENVELOPE": "ENV",
}

_TRIG_MODE = {"AUTO": "AUTO", "NORMAL": "NORM"}

_EDGE_SLOPE = {"RISING": "RISE", "FALLING": "FALL", "BOTH": "EITHER"}

_EDGE_COUPLING = {
    "DC": "DC",
    "HF REJECT": "HFR",
    "LF REJECT": "LFR",
    "NOISE REJECT": "NOISE",
}

_PULSE_POLARITY = {"POSITIVE": "POS", "NEGATIVE": "NEG"}

_PULSE_WHEN = {
    "LESSTHAN": "LESS",
    "MORETHAN": "MORE",
    "EQUAL": "EQ",
    "UNEQUAL": "UNEQ",
    "WITHIN": "WIT",
    "OUTSIDE": "OUT",
}

_SV_WINDOW = {
    "KAISER": "KAIS",
    "RECT": "RECT",
    "HAMMING": "HAMM",
    "HANNING": "HANN",
    "BLACKMAN": "BLAC",
    "FLATTOP": "FLAT",
}

_CH_COUPLING = {"AC": "AC", "DC": "DC", "DC REJECT": "DCR"}

# Spectrum View trace source suffixes for SAVe:WAVEform (verified against
# Save_and_Recall.md: CH<x>_SV_NORMal / _SV_MAXHold / _SV_MINHold / _SV_AVErage).
# Video trigger choice maps (Trigger.md: TRIGger:{A|B}:VIDeo:*).
_VIDEO_STANDARD = {"NTSC": "NTSc", "PAL": "PAL", "SECAM": "SECAM"}
_VIDEO_FIELD = {
    "ALL LINES": "ALLLines",
    "NUMERIC": "NUMERic",
    "ODD": "ODD",
    "EVEN": "EVEN",
    "ALL FIELDS": "ALLFields",
}
_VIDEO_POLARITY = {"POSITIVE": "POSitive", "NEGATIVE": "NEGative"}

# Timeout trigger (TRIGger:{A|B}:TIMEOut:POLarity).
_TIMEOUT_POLARITY = {"STAYS HIGH": "STAYSHigh", "STAYS LOW": "STAYSLow", "EITHER": "EITher"}

# Runt / transition polarity (TRIGger:{A|B}:{RUNT|TRANsition}:POLarity).
_EDGE_POLARITY = {"POSITIVE": "POSitive", "NEGATIVE": "NEGative", "EITHER": "EITher"}

# Runt qualifier (TRIGger:{A|B}:RUNT:WHEn).
_RUNT_WHEN = {
    "LESSTHAN": "LESSthan",
    "MORETHAN": "MOREthan",
    "EQUAL": "EQual",
    "UNEQUAL": "UNEQual",
    "OCCURS": "OCCURS",
}

# Window qualifier / crossing (TRIGger:{A|B}:WINdow:{WHEn,CROSSIng}).
_WINDOW_WHEN = {
    "ENTERS": "ENTERSWindow",
    "EXITS": "EXITSWindow",
    "INSIDE": "INSIDEGreater",
    "OUTSIDE": "OUTSIDEGreater",
}
_WINDOW_CROSSING = {"UPPER": "UPPer", "LOWER": "LOWer", "EITHER": "EITher", "NONE": "NONe"}

# Transition qualifier (TRIGger:{A|B}:TRANsition:WHEn).
_TRANSITION_WHEN = {
    "SLOWER": "SLOWer",
    "FASTER": "FASTer",
    "EQUAL": "EQual",
    "UNEQUAL": "UNEQual",
}

# Vertical / probe choice maps (Vertical.md).
_BW_OPTIMIZATION = {"STEP RESPONSE": "STEPRESPONSE", "FLATNESS": "FLATNESS"}
_PROBE_CONTROL = {"AUTO": "AUTO", "MANUAL": "MANual"}
_PROBE_INPUT_MODE = {
    "A": "A",
    "B": "B",
    "COMMON MODE": "COMMONMODE",
    "DIFFERENTIAL": "DIFFERENTIAL",
}

# Horizontal mode (HORizontal:MODe).
_HORIZ_MODE = {"AUTO": "AUTO", "MANUAL": "MANual"}

# Which manual-mode control follows a sample-rate change
# (HORizontal:MODe:MANual:CONFIGure).
_HORIZ_MANUAL_CONFIG = {"SCALE": "HORIZontalscale", "RECORD LENGTH": "RECORDLength"}

# Spectrum View per-channel units / trace / marker type.
_SV_UNITS = {
    "DBM": "DBM",
    "DBUW": "DBUW",
    "DBMV": "DBMV",
    "DBUV": "DBUV",
    "DBMA": "DBMA",
    "DBUA": "DBUA",
}
_SV_TRACE_SELECT = {
    "NORMAL": "NORMal",
    "MAX HOLD": "MAXHold",
    "MIN HOLD": "MINHold",
    "AVERAGE": "AVErage",
}
_SV_MARKER_TYPE = {"DELTA": "DELta", "ABSOLUTE": "ABSolute"}

# AFG function generator choice maps (AFG.md).
_AFG_OUTPUT_MODE = {"OFF": "OFF", "CONTINUOUS": "CONTinuous", "BURST": "BURSt"}
_REFLEVEL_METHOD = {"PERCENT": "PERCent", "ABSOLUTE": "ABSolute"}
_REFLEVEL_PERCENT_TYPE = {
    "10-90": "TENNinety",
    "20-80": "TWENtyeighty",
    "CUSTOM": "CUSTom",
}
_REFLEVEL_ABSOLUTE_TYPE = {"SAME": "SAME", "UNIQUE": "UNIQue"}
_REFLEVEL_BASETOP = {
    "AUTO": "AUTO",
    "MINMAX": "MINMax",
    "MEAN HISTOGRAM": "MEANhistogram",
    "MODE HISTOGRAM": "MODEhistogram",
    "EYE HISTOGRAM": "EYEhistogram",
}
_REFLEVEL_TYPE = {"GLOBAL": "GLOBal", "PER SOURCE": "PERSource"}

# Measurement gating choice maps (Measurement.md).
_GATING_MODE = {
    "NONE": "NONE",
    "SCREEN": "SCREEN",
    "CURSOR": "CURSor",
    "LOGIC": "LOGic",
    "SEARCH": "SEARCH",
    "TIME": "TIMe",
}
_GATING_ACTIVE = {"HIGH": "HIGH", "LOW": "LOW"}

# Trigger B-event choice maps (Trigger.md + bench probe: the B edge event uses
# the undocumented-but-real TRIGger:B:EDGE branch, see ADR-016).
_TRIG_B_BY = {"EVENTS": "EVENTS", "TIME": "TIMe"}
_TRIG_HOLDOFF_BY = {"TIME": "TIMe", "RANDOM": "RANDom"}

# Acquisition sequence mode (Acquisition.md: ACQuire:SEQuence:MODe).
_ACQ_SEQUENCE_MODE = {"NUMACQS": "NUMACQs"}

# FastFrame summary-frame mode (Horizontal.md).
_FASTFRAME_SUMFRAME = {"NONE": "NONe", "AVERAGE": "AVErage", "ENVELOPE": "ENVelope"}
_FASTFRAME_MULTI_MODE = {"OFF": "OFF", "OVERLAY": "OVERlay"}

# Cursor choice maps (Cursor.md, DISplay:WAVEView<x>:CURSor form).
_CURSOR_FUNCTION = {
    "HBARS": "HBArs",
    "VBARS": "VBArs",
    "SCREEN": "SCREEN",
    "WAVEFORM": "WAVEFORM",
}
_CURSOR_MODE = {"TRACK": "TRACK", "INDEPENDENT": "INDEPENDENT"}

# DVM choice maps (DVM.md).
_DVM_MODE = {"ACRMS": "ACRMS", "ACDCRMS": "ACDCRMS", "DC": "DC", "OFF": "OFF"}

# Image save composition (Save_and_Recall.md: SAVe:IMAGe:COMPosition).
_IMAGE_COMPOSITION = {"NORMAL": "NORMal", "INVERTED": "INVErted"}

# Waveform Transfer data encoding (Waveform_Transfer.md: DATa:ENCdg).
_DATA_ENCODING = {
    "ASCII": "ASCIi",
    "RIBINARY": "RIBinary",
    "RPBINARY": "RPBinary",
    "FPBINARY": "FPBinary",
    "SRIBINARY": "SRIbinary",
    "SRPBINARY": "SRPbinary",
    "SFPBINARY": "SFPbinary",
}


@dataclass
class MeasStats:
    """Accumulated statistics for a measurement slot.

    ``value`` is the current-acquisition badge value; the remaining fields are
    accumulated over all acquisitions since the last statistics reset.
    """

    value: float
    mean: float
    minimum: float
    maximum: float
    stddev: float
    population: int


class Tektronix_MSO(Instrument):
    """Tektronix MSO 4/5/6 Series oscilloscope.

    The subsystem tree is built in ``__init__`` (before connecting). Channels
    beyond the model's channel count are marked ``installed=False`` on connect.
    """

    # Enums live on the class so the operator can reach them from the live
    # object: scope.Coupling.<TAB> resolves, where a bare name would not.
    # An operator working with an instrument has the whole instrument in
    # scope, so they are NOT repeated on the nodes.
    AcqSequenceMode = AcqSequenceMode
    AcqType = AcqType
    AfgOutputMode = AfgOutputMode
    BandwidthOptimization = BandwidthOptimization
    Coupling = Coupling
    CursorFunction = CursorFunction
    CursorMode = CursorMode
    DataEncoding = DataEncoding
    DvmMode = DvmMode
    FastFrameMultiMode = FastFrameMultiMode
    FastFrameSummary = FastFrameSummary
    GatingActive = GatingActive
    GatingMode = GatingMode
    HoldoffBy = HoldoffBy
    HorizontalManualConfig = HorizontalManualConfig
    HorizontalMode = HorizontalMode
    ImageComposition = ImageComposition
    ProbeControl = ProbeControl
    ProbeInputMode = ProbeInputMode
    PulsePolarity = PulsePolarity
    PulseWhen = PulseWhen
    RefLevelAbsoluteType = RefLevelAbsoluteType
    RefLevelBaseTop = RefLevelBaseTop
    RefLevelMethod = RefLevelMethod
    RefLevelPercentType = RefLevelPercentType
    RefLevelType = RefLevelType
    RuntPolarity = RuntPolarity
    RuntWhen = RuntWhen
    Slope = Slope
    SpectrumMarkerType = SpectrumMarkerType
    SpectrumTraceSelect = SpectrumTraceSelect
    SpectrumUnits = SpectrumUnits
    SpectrumWindow = SpectrumWindow
    TimeoutPolarity = TimeoutPolarity
    TransitionWhen = TransitionWhen
    TriggerBBy = TriggerBBy
    TriggerCoupling = TriggerCoupling
    TriggerMode = TriggerMode
    VideoField = VideoField
    VideoPolarity = VideoPolarity
    VideoStandard = VideoStandard
    WindowCrossing = WindowCrossing
    WindowWhen = WindowWhen
    AuxOutEdge = AuxOutEdge
    AuxOutSource = AuxOutSource
    Function = Function
    Load = Load
    SpectrumTrace = SpectrumTrace

    def __init__(self, resource_name: str) -> None:
        """
        Parameters
        ----------
        resource_name : str
            full pyvisa instrument connection string
        """
        super().__init__(resource_name, name="MSO")

        self.model: str | None = None
        self.num_channels: int | None = None

        self.horz = _Horizontal(self.link, name="HORIZONTAL")
        self.acq = _Acquisition(self.link, name="ACQUISITION")
        self.trig = _Trigger(self.link, name="TRIGGER")
        self.sv = _Spectrum(self.link, name="SV")

        self.ch1 = _Channel(self.link, 1)
        self.ch2 = _Channel(self.link, 2)
        self.ch3 = _Channel(self.link, 3)
        self.ch4 = _Channel(self.link, 4)
        self.ch5 = _Channel(self.link, 5)
        self.ch6 = _Channel(self.link, 6)
        self.ch7 = _Channel(self.link, 7)
        self.ch8 = _Channel(self.link, 8)

        self.meas = _Measure(self.link, name="MEASUREMENT")
        self.file = _File(self.link, name="FILE")

        # Waveform Transfer (CURVe / WFMOutpre, direct to memory).
        self.data = _Data(self.link, name="DATA")

        # Reference and bus decode subsystems.
        self.ref1 = _Reference(self.link, 1)
        self.ref2 = _Reference(self.link, 2)
        self.ref3 = _Reference(self.link, 3)
        self.ref4 = _Reference(self.link, 4)
        self.bus1 = _Bus(self.link, 1)
        self.bus2 = _Bus(self.link, 2)
        self.bus3 = _Bus(self.link, 3)
        self.bus4 = _Bus(self.link, 4)

        # Display / navigation / analysis subsystems.
        self.zoom = _Zoom(self.link, name="ZOOM")
        self.cursors = _Cursors(self.link, name="CURSOR")
        self.disp = _Display(self.link, name="DISPLAY")
        self.search = _Search(self.link, name="SEARCH")

        # Rear-panel Auxiliary Output BNC (trigger out / 10 MHz ref out / AFG out).
        self.aux_out = _AuxOut(self.link, name="AUXOUT")

        # Option-gated subsystems (installed state set from *OPT? on connect).
        self.afg = _Afg(self.link, name="AFG")
        self.dvm = _Dvm(self.link, name="DVM")
        self.dig = _Digital(self.link, name="DIGITAL")

        # Math waveforms - unlimited count on the instrument; created on demand.
        self._math_cache: dict[int, _Math] = {}

    @property
    def channels(self) -> list["_Channel"]:
        return [
            self.ch1,
            self.ch2,
            self.ch3,
            self.ch4,
            self.ch5,
            self.ch6,
            self.ch7,
            self.ch8,
        ]

    def math(self, n: int) -> "_Math":
        """Return the math waveform node ``MATH<n>``.

        Math count is unlimited on the instrument, so math nodes are created
        and cached on first access rather than declared up front.
        """
        if n < 1:
            raise ValueError(f"math index must be >= 1, got {n}")
        node = self._math_cache.get(n)
        if node is None:
            node = _Math(self.link, n)
            self._math_cache[n] = node
        return node

    def _on_connect(self, idn: str) -> None:
        """Parse IDN for model + channel count; gate channel/option install state."""
        match = re.search(r"TEKTRONIX,\s*(MSO(\d+)[A-Za-z]*)", idn, re.IGNORECASE)
        if not match:
            raise RuntimeError(f"Error parsing IDN: {idn!r}")
        self.model = match.group(1)
        self.num_channels = int(match.group(2)[-1])
        self.file.model = self.model

        for ch in self.channels:
            installed = ch._ch <= self.num_channels
            ch.installed = installed
            ch.sv.installed = installed
            ch.probe.installed = installed

        self._gate_options()

    def _gate_options(self) -> None:
        """Gate afg/dvm install flags from RUNTIME PROBES (ADR-004).

        ``*OPT?`` on this family is verbose per-license bundle text with no
        discrete AFG/DVM/DIG code, so it is parsed for reference only
        (``self.capabilities``) and NOT used as a presence gate. The
        authoritative check is a defensive runtime probe per subsystem (a probe
        failure flags the subsystem absent and never breaks connect). Digital is
        hot-pluggable and detected per channel on demand via
        ``get_channel_digital`` rather than frozen here.
        """
        try:
            self.options = self.link.query("*OPT?").strip()
        except Exception:  # noqa: BLE001 - option gating must not break connect
            self.options = ""
        self.capabilities = parse_opt(self.options)
        try:
            self.afg.installed = probe_afg(self.link)
            self.dvm.installed = probe_dvm(self.link)
        except Exception:  # noqa: BLE001 - probing must not break connect
            pass
        self._drain_probe_events()

    def _drain_probe_events(self) -> None:
        """Clear the events the capability probes leave behind.

        A probe for an absent option raises event 113 "Undefined header". On
        this bench the DVM probe leaves 113,"...DVM:MODE?" queued, which then
        surfaces later and looks like a fault in unrelated code
        (bench-verified 2026-08-06).

        The MSO hides events until *ESR? is read. A bare ALLEV? answers "new
        events pending *ESR?", so both commands are required. Best effort. A
        failure here must never break connect.
        """
        with contextlib.suppress(Exception):
            for _ in range(4):
                self.link.query("*ESR?")
                resp = self.link.query("ALLEV?").strip()
                if resp.startswith("0,") or "queue empty" in resp:
                    return

    def get_channel_digital(self, channel: int) -> bool:
        """Probe whether a channel currently has a digital probe (hot-pluggable).

        Refreshable: safe to call any time, since digital probes can be
        attached or removed while connected.

        ### Parameters
        1. channel : int
            - analog channel number
        ### Returns
        1. bool
            - True if the channel reports a digital probe type
        """
        return probe_channel_digital(self.link, channel)

    def get_analog_bandwidth(self) -> float:
        """Query the instrument's maximum analog bandwidth in Hz.

        Reads ``CONFIGuration:ANALOg:BANDWidth?`` (the hardware bandwidth of the
        model, independent of any per-channel bandwidth limit).
        """
        return self.link.query_float("CONFIGuration:ANALOg:BANDWidth?")

    # Event codes 540-549 are the "Measurement warning" class (e.g. clipping);
    # they are informational and must not abort a script (see R5).
    _EVENT_RE = re.compile(r'(-?\d+)\s*,\s*"([^"]*)"')
    _MEAS_WARNING_RANGE = range(540, 550)

    @classmethod
    def _classify_events(cls, allev_text: str) -> tuple[list[str], list[str]]:
        """Split an ALLEV? response into (measurement-warning, error) events.

        ### Parameters
        1. allev_text : str
            - raw ALLEV? response
        ### Returns
        1. (warnings, errors) : tuple[list[str], list[str]]
            - each item is a ``code,"message"`` string; code 0 is dropped
        """
        warnings_out: list[str] = []
        errors_out: list[str] = []
        for code_str, message in cls._EVENT_RE.findall(allev_text):
            code = int(code_str)
            if code == 0:
                continue
            entry = f'{code},"{message}"'
            if code in cls._MEAS_WARNING_RANGE:
                warnings_out.append(entry)
            else:
                errors_out.append(entry)
        return warnings_out, errors_out

    def check_errors(self) -> None:
        """Raise on queued command/execution errors; warn on measurement warnings.

        MSO 4/5/6 has no SYST:ERR? — it reports via *ESR? and the event
        queue (ALLEV?). Event codes in the 540-549 "Measurement warning" class
        (e.g. signal clipping) are surfaced via ``warnings.warn`` rather than
        raised, so a benign clipping warning does not abort a script (R5).
        """
        esr = self.link.query_int("*ESR?")
        if not (esr & 0b00111100):
            return
        events = self.link.query("ALLEV?").strip()
        meas_warnings, errors = self._classify_events(events)
        if meas_warnings:
            warnings.warn(
                InstrumentWarning("Instrument measurement warning(s): " + "; ".join(meas_warnings)),
                stacklevel=2,
            )
        if errors:
            raise InstrumentError(f"Instrument reported errors (ESR={esr}): " + "; ".join(errors))

    @contextlib.contextmanager
    def preserved_setup(self, timeout_ms: int | None = None) -> None:
        """Snapshot the full instrument setup, restoring it exactly on exit.

        Saves the current setup to a scope-side scratch ``.set`` file on enter
        and recalls it on exit - even if the body raises - then deletes the
        scratch file. This is the structural fix for the horizontal-mode
        incident: bench scripts wrap their work in this so they never leave the
        scope in a changed state.

        ### Parameters
        1. timeout_ms : int (default None)
            - I/O timeout for the save/recall round trip
        ### Yields
        1. Tektronix_MSO
            - this driver instance
        """
        scratch = self.file._scratch_path(f"preserved_{self.file._timestamp()}.set")
        if timeout_ms is not None:
            with self.link.temporary_timeout(timeout_ms):
                self.file.cmd_save_setup(scratch)
        else:
            self.file.cmd_save_setup(scratch)
        try:
            yield self
        finally:
            try:
                self.file.cmd_recall_setup(scratch)
            finally:
                with contextlib.suppress(Exception):
                    self.file.cmd_delete(scratch)


class _Horizontal(Node):
    """Horizontal configuration for time-domain views."""

    def set_scale(self, scale: float) -> float:
        """Set the time per horizontal division. Does NOT change the mode.

        Bench-verified behavior per mode:
        - AUTO: the scope re-picks the sample rate (record length held)
        - MANUAL: record length adjusts (sample rate held), keeping the
          invariant 10 * scale = record_length / sample_rate

        ### Parameters
        1. scale : float
            - seconds per division
        ### Returns
        1. scale : float
            - actual scale configured by the scope
        """
        self.write(f"HOR:MOD:SCA {scale:.12g}")
        self.query("*OPC?")
        return self.query_float("HOR:MOD:SCA?")

    def set_position(self, percent: float) -> None:
        """Set the horizontal trigger position in % of screen
        ### Parameters
        1. position : float
            - 0.0 is left edge, 100.0 is right edge
        """
        self.write(f"HOR:POS {percent:.12g}")

    def get_div_count(self) -> int:
        """Query the number of horizontal divisions"""
        return self.query_int("HOR:DIV?")

    def set_delay(self, delay: float) -> None:
        """Set the horizontal delay time from trigger
        ### Parameters
        1. delay : float
            - Delay time in seconds
        """
        if delay > 0:
            self.write("HOR:DEL:MOD ON")
            self.write(f"HOR:DEL:TIM {delay:.12g}")
        else:
            self.write("HOR:DEL:MOD OFF")

    def set_record_length(self, length: float) -> float:
        """Set the memory record length in points. MANUAL mode only.

        Raises InstrumentError in AUTO mode: the scope silently ignores the
        write there (no SCPI error - bench-verified). Does NOT change the
        mode; call set_mode('MANUAL') or set_manual() first.
        In MANUAL the scale adjusts (sample rate held).

        ### Parameters
        1. length : float (for exponents only)
            - record length in points.  Any whole number.
        ### Returns
        1. length : float
            - actual record length configured by the scope
        """
        self._require_manual("record length")
        self.write(f"HOR:RECO {int(length)}")
        return self.query_float("HOR:RECO?")

    def get_record_length(self) -> float:
        """Get the memory record length in points"""
        return self.query_float("HOR:RECO?")

    def _require_manual(self, what: str) -> None:
        mode = self.query("HOR:MODE?").strip()
        if not mode.upper().startswith("MAN"):
            raise InstrumentError(
                f"Cannot set {what} in horizontal {mode} mode - the scope "
                "silently ignores the write. Call set_mode('MANUAL') or "
                "set_manual(samplerate, record_length) first."
            )

    def set_samplerate(self, rate: float) -> float:
        """Set the ADC sample rate (snaps to the scope's legal steps).
        MANUAL mode only.

        Raises InstrumentError in AUTO mode: the scope silently ignores the
        write there (no SCPI error - bench-verified). Does NOT change the
        mode. In MANUAL the scale adjusts (record length held) - your
        time/div WILL change.

        ### Parameters
        1. rate : float
            - samples per second
        ### Returns
        1. rate : float
            - actual sample rate configured on the scope
        """
        self._require_manual("sample rate")
        self.write(f"HOR:MOD:SAMPLER {rate:.12g}")
        return self.query_float("HOR:MOD:SAMPLER?")

    def set_manual(self, samplerate: float, record_length: float) -> float:
        """Switch to MANUAL mode and set sample rate + record length together.

        The explicit way to say "give me exactly this SR and this many
        points"; the resulting time/div follows from
        10 * scale = record_length / sample_rate.

        ### Parameters
        1. samplerate : float
            - samples per second
        2. record_length : float
            - record length in points
        ### Returns
        1. (samplerate, record_length) : tuple[float, float]
            - actual values configured by the scope
        """
        self.write("HOR:MODE MAN")
        self.write(f"HOR:MOD:SAMPLER {samplerate:.12g}")
        self.write(f"HOR:RECO {int(record_length)}")
        return (
            self.query_float("HOR:MOD:SAMPLER?"),
            self.query_float("HOR:RECO?"),
        )

    def set_timebase(
        self,
        *,
        scale: float | None = None,
        samplerate: float | None = None,
        record_length: float | None = None,
    ) -> TimebaseState:
        """Set exactly two of scale, samplerate and record_length in one call.

        The three values are locked by ``div_count * scale = record_length /
        samplerate``, so the third follows from the two you give. Units are
        seconds per division, samples per second, and points. All three snap to
        the legal steps of the instrument.

        This helper switches to MANUAL mode first. AUTO mode discards
        sample-rate and record-length writes and reports no error, so three
        atomic setters cannot express this safely (bench-verified).

        The write order per pair is bench-verified, because the instrument
        recomputes scale whenever sample rate or record length is written:
        samplerate then record_length; samplerate then scale, so the scale
        write lands last and holds; for scale with record_length the sample
        rate cannot be left alone, so it is derived and written first.
        """
        given = sum(v is not None for v in (scale, samplerate, record_length))
        if given != 2:
            raise ValueError(f"specify exactly two of scale/samplerate/record_length (got {given})")
        self.write("HOR:MODE MAN")
        if scale is None:
            self.write(f"HOR:MOD:SAMPLER {samplerate:.12g}")
            self.write(f"HOR:RECO {int(record_length)}")
        elif record_length is None:
            self.write(f"HOR:MOD:SAMPLER {samplerate:.12g}")
            self.write(f"HOR:MOD:SCA {scale:.12g}")
        else:
            derived_sr = record_length / (self.get_div_count() * scale)
            self.write(f"HOR:MOD:SAMPLER {derived_sr:.12g}")
            self.write(f"HOR:RECO {int(record_length)}")
        self.query("*OPC?")
        return self.get_timebase()

    def get_timebase(self) -> TimebaseState:
        """Read the complete horizontal setting."""
        return TimebaseState(
            scale=self.get_scale(),
            samplerate=self.get_samplerate(),
            record_length=self.get_record_length(),
            mode=self.get_mode(),
            position=self.get_position(),
        )

    def snapshot(self) -> dict:
        """Record the current horizontal configuration for later restore().

        ### Returns
        1. snapshot : dict
            - mode, scale, samplerate, record_length, position, delay state
        """
        return {
            "mode": self.get_mode(),
            "scale": self.get_scale(),
            "samplerate": self.get_samplerate(),
            "record_length": self.get_record_length(),
            "position": self.get_position(),
            "delay_mode": self.get_delay_mode(),
            "delay_time": self.get_delay(),
        }

    def restore(self, snap: dict) -> None:
        """Restore a configuration captured by snapshot(), exactly.

        Mode-aware ordering: MANUAL restores sample rate and record length
        (scale follows); AUTO restores scale (SR/RL follow).

        ### Parameters
        1. snap : dict
            - value returned by snapshot()
        """
        if snap["mode"].upper().startswith("MAN"):
            self.write("HOR:MODE MAN")
            self.write(f"HOR:MOD:SAMPLER {snap['samplerate']:.12g}")
            self.write(f"HOR:RECO {int(snap['record_length'])}")
            self.write(f"HOR:MOD:SCA {snap['scale']:.12g}")
        else:
            self.write("HOR:MODE AUTO")
            self.write(f"HOR:MOD:SCA {snap['scale']:.12g}")
        self.write(f"HOR:POS {snap['position']:.12g}")
        if snap["delay_mode"]:
            self.write("HOR:DEL:MOD ON")
            self.write(f"HOR:DEL:TIM {snap['delay_time']:.12g}")
        else:
            self.write("HOR:DEL:MOD OFF")
        self.query("*OPC?")

    def get_samplerate(self) -> float:
        """Query the ADC samplerate"""
        return self.query_float("HOR:MOD:SAMPLER?")

    def cmd_increase_samplerate(self) -> float:
        """Step the sample rate up one legal step. MANUAL mode only."""
        current = self.get_samplerate()
        increment = 1.1
        setpoint = current * increment
        while current == self.set_samplerate(setpoint):
            setpoint *= increment
        return self.get_samplerate()

    def cmd_decrease_samplerate(self) -> float:
        """Step the sample rate down one setting, and return the new rate."""
        current = self.get_samplerate()
        increment = 0.9
        setpoint = current * increment
        while current == self.set_samplerate(setpoint):
            setpoint *= increment
        return self.get_samplerate()

    def set_mode(self, mode: HorizontalMode | str) -> str:
        """Set the horizontal mode.

        Enum arg: <scope>.HorizontalMode.<TAB>

        In AUTO mode the scope picks record length and sample rate from the
        scale; in MANUAL mode they are set explicitly.

        ### Parameters
        1. mode : str
            - ['AUTO', 'MANUAL']
        ### Returns
        1. mode : str
            - actual horizontal mode read back from the scope
        """
        self.write(f"HOR:MODE {token(mode, HorizontalMode, 'mode')}")
        return self.query("HOR:MODE?").strip()

    def get_mode(self) -> HorizontalMode:
        """Query the horizontal mode (AUTO or MANUAL)."""
        return from_response(self.query("HOR:MODE?").strip(), HorizontalMode, "mode")

    def set_manual_config(self, target: HorizontalManualConfig | str) -> str:
        """Select which control follows a MANUAL-mode sample-rate change.

        Enum arg: <scope>.HorizontalManualConfig.<TAB>

        Wraps ``HORizontal:MODe:MANual:CONFIGure``. This is a PREFERENCE, not
        an action: it does not change the mode or any value. It picks which of
        scale or record length primarily moves when the sample rate is adjusted
        in MANUAL mode (the other may still move if the selected one hits a
        limit). Distinct from :meth:`configure`, which actively applies two of
        the three horizontal values. Bench default is ``SCALE`` (verified: the
        MSO58 reports ``CONFIGURE HORIZONTALSCALE``), which is why
        :meth:`set_samplerate` holds record length and moves scale.

        ### Parameters
        1. target : str
            - ['SCALE', 'RECORD LENGTH']
        ### Returns
        1. target : str
            - actual configure target read back from the scope
        """
        self.write(f"HOR:MOD:MAN:CONFIG {token(target, HorizontalManualConfig, 'target')}")
        return self.query("HOR:MOD:MAN:CONFIG?").strip()

    def get_manual_config(self) -> HorizontalManualConfig:
        """Query which control follows a MANUAL-mode sample-rate change."""
        return from_response(
            self.query("HOR:MOD:MAN:CONFIG?").strip(), HorizontalManualConfig, "manual_config"
        )

    def get_position(self) -> float:
        """Query the horizontal trigger position in % of screen."""
        return self.query_float("HOR:POS?")

    def get_scale(self) -> float:
        """Query the time per horizontal division in seconds."""
        return self.query_float("HOR:MOD:SCA?")

    def get_delay(self) -> float:
        """Query the horizontal delay time from trigger in seconds."""
        return self.query_float("HOR:DEL:TIM?")

    def get_delay_mode(self) -> bool:
        """Query whether horizontal delay mode is enabled."""
        return self.query_bool("HOR:DEL:MOD?")

    def get_roll(self) -> str:
        """Query the roll-mode state (AUTO/ON/OFF) reported by the scope."""
        return self.query("HOR:ROLL?").strip()

    def get_acq_duration(self) -> float:
        """Query the acquisition duration (record length / sample rate) in seconds."""
        return self.query_float("HOR:ACQDURATION?")

    def get_sample_rate_actual(self) -> float:
        """Query the actual (realized) horizontal sample rate in S/s."""
        return self.query_float("HOR:SAMPLERate?")

    # ------------------------------------------------------------------
    # FastFrame (segmented acquisition). On the MSO 4/5/6 FastFrame IS the
    # acquisition-history mechanism: the "History" the plan describes (state /
    # overlay / cumulative stats / reference frame / timestamps) maps onto the
    # HORizontal:FASTframe:* group, not a separate History command set. The
    # actual History command group on this family is only the timestamp-table
    # trio wrapped below (see ADR-017). Probe-first: FastFrame interacts with
    # acquisition; wrap it in preserved_setup()/acquisition snapshotting.
    # ------------------------------------------------------------------
    def set_fastframe_state(self, state: bool) -> bool:
        """Enable or disable FastFrame segmented acquisition; returns readback."""
        self.write(f"HORizontal:FASTframe:STATE {'ON' if state else 'OFF'}")
        return self.query_bool("HORizontal:FASTframe:STATE?")

    def get_fastframe_state(self) -> bool:
        """Query whether FastFrame is enabled."""
        return self.query_bool("HORizontal:FASTframe:STATE?")

    def set_fastframe_count(self, count: int) -> int:
        """Set the number of frames to capture; returns the read-back count."""
        self.write(f"HORizontal:FASTframe:COUNt {int(count)}")
        return self.query_int("HORizontal:FASTframe:COUNt?")

    def get_fastframe_count(self) -> int:
        """Query the configured FastFrame frame count."""
        return self.query_int("HORizontal:FASTframe:COUNt?")

    def get_fastframe_max_frames(self) -> int:
        """Query the maximum number of frames the current record allows."""
        return self.query_int("HORizontal:FASTframe:MAXFRames?")

    def set_fastframe_selected(self, frame: int) -> int:
        """Select the displayed/queried frame; returns the read-back index."""
        self.write(f"HORizontal:FASTframe:SELECTED {int(frame)}")
        return self.query_int("HORizontal:FASTframe:SELECTED?")

    def get_fastframe_selected(self) -> int:
        """Query the selected FastFrame index."""
        return self.query_int("HORizontal:FASTframe:SELECTED?")

    def set_fastframe_summary(self, mode: FastFrameSummary | str) -> str:
        """Set the FastFrame summary-frame calculation.

        Enum arg: <scope>.FastFrameSummary.<TAB>
        ### Parameters
        1. mode : str
            - ['NONE', 'AVERAGE', 'ENVELOPE']
        ### Returns
        1. mode : str
            - actual summary mode read back from the scope
        """
        self.write(f"HORizontal:FASTframe:SUMFrame {token(mode, FastFrameSummary, 'mode')}")
        return self.query("HORizontal:FASTframe:SUMFrame?").strip()

    def set_fastframe_summary_state(self, state: bool) -> bool:
        """Enable or disable the FastFrame summary frame; returns readback."""
        self.write(f"HORizontal:FASTframe:SUMFrame:STATE {'ON' if state else 'OFF'}")
        return self.query_bool("HORizontal:FASTframe:SUMFrame:STATE?")

    def set_fastframe_multi_mode(self, mode: FastFrameMultiMode | str) -> str:
        """Set the multiple-frames display mode.

        Enum arg: <scope>.FastFrameMultiMode.<TAB>
        ### Parameters
        1. mode : str
            - ['OFF', 'OVERLAY']
        ### Returns
        1. mode : str
            - actual multiple-frames mode read back from the scope
        """
        self.write(
            f"HORizontal:FASTframe:MULtipleframes:MODe {token(mode, FastFrameMultiMode, 'mode')}"
        )
        return self.query("HORizontal:FASTframe:MULtipleframes:MODe?").strip()

    def set_fastframe_ref_frame(self, frame: int) -> int:
        """Set the FastFrame reference frame index; returns the read-back index."""
        self.write(f"HORizontal:FASTframe:REF:FRAme {int(frame)}")
        return self.query_int("HORizontal:FASTframe:REF:FRAme?")

    def set_fastframe_ref_include(self, state: bool) -> bool:
        """Include/exclude the reference frame in the display; returns readback."""
        self.write(f"HORizontal:FASTframe:REF:INCLUde {'ON' if state else 'OFF'}")
        return self.query_bool("HORizontal:FASTframe:REF:INCLUde?")

    def get_fastframe_timestamp(self, which: str = "SELECTED") -> str:
        """Query a FastFrame timestamp.
        ### Parameters
        1. which : str (default 'SELECTED')
            - ['SELECTED', 'REFERENCE', 'DELTA', 'ALL']
        ### Returns
        1. timestamp : str
            - the raw timestamp response for the requested frame(s)
        """
        key = which.strip().upper()
        suffix = {
            "SELECTED": "SELECTED?",
            "REFERENCE": "REFerence?",
            "DELTA": "DELTa?",
            "ALL": "ALL?",
        }
        if key not in suffix:
            raise ValueError(f"invalid timestamp selector {which!r}; valid: {list(suffix)}")
        return self.query(f"HORizontal:FASTframe:TIMEStamp:{suffix[key]}").strip().strip('"')

    def get_fastframe_xzero(self, which: str = "SELECTED") -> float:
        """Query a FastFrame XZEro (frame time origin) in seconds.
        ### Parameters
        1. which : str (default 'SELECTED')
            - ['SELECTED', 'REF']
        ### Returns
        1. xzero : float
            - the XZEro of the requested frame in seconds
        """
        key = which.strip().upper()
        suffix = {"SELECTED": "SELECTED?", "REF": "REF?"}
        if key not in suffix:
            raise ValueError(f"invalid xzero selector {which!r}; valid: {list(suffix)}")
        return self.query_float(f"HORizontal:FASTframe:XZEro:{suffix[key]}")

    # ------------------------------------------------------------------
    # History timestamp-result tables (the actual "History" command group on
    # this family - see ADR-017).
    # ------------------------------------------------------------------
    def cmd_add_timestamp_table(self, name: str) -> None:
        """Add a History time-stamp results table."""
        self.write(f"TSTamptable:ADDNew {quote(name)}")

    def cmd_delete_timestamp_table(self, name: str) -> None:
        """Delete a History time-stamp results table."""
        self.write(f"TSTamptable:DELETE {quote(name)}")

    def get_timestamp_tables(self) -> list[str]:
        """List the defined History time-stamp results tables."""
        listing = self.query("TSTamptable:LIST?").strip()
        if not listing or listing.upper() in ("NONE", '""'):
            return []
        return [t.strip().strip('"') for t in listing.split(",") if t.strip()]


class _Acquisition(Node):
    def cmd_run(self) -> None:
        """Start acquiring continuously. This is the front-panel Run button."""
        self.write("ACQ:STOPA RUNST")
        self.write("ACQ:STATE RUN")

    def cmd_stop(self) -> None:
        """Stop acquiring. This is the front-panel Stop button."""
        self.write("ACQ:STATE STOP")

    def get_state(self) -> int:
        """Read the acquisition state. 1 while acquiring, 0 when stopped."""
        return self.query_int("ACQ:STATE?")

    def cmd_sequence(self, n_single: int) -> None:
        """Arm for n_single acquisitions, then stop."""
        self.write("ACQ:STOPA SEQ")
        self.write("ACQ:SEQ:NUMSEQ %d" % int(n_single))
        self.write("ACQ:STATE RUN")

    def cmd_single(self) -> None:
        """Arm for one acquisition. This is the front-panel Single button."""
        self.write("ACQ:STOPA SEQ")
        self.write("ACQ:SEQ:NUMSEQ 1")
        self.write("ACQ:STATE RUN")

    def set_type(self, type: AcqType | str) -> None:
        """Change the acquisition type/mode

        Enum arg: <scope>.AcqType.<TAB>
        ### Parameters
        1. type : str
            - ['SAMPLE', 'AVERAGE', 'PEAK', 'HIRES', 'ENVELOPE']
        """
        self.write(f"ACQ:MODE {token(type, AcqType, 'type')}")

    def set_avg_count(self, n_avg: int) -> int:
        """set the number of acquisitions to average
        ### Parameters
        1. n_avg : int
            - number of acquisitions to average
        """
        self.write("ACQ:NUMAV %d" % n_avg)
        return self.query_int("ACQ:NUMAV?")

    def get_acq_count(self) -> int:
        """Query how many acquisitions have occured since last start"""
        return self.query_int("ACQ:NUMAC?")

    def get_max_samplerate(self) -> float:
        """Query the maximum samplerate"""
        return self.query_float("ACQ:MAXS?")

    def get_sequence_current(self) -> int:
        """Query how many acquisitions of the current sequence have completed.

        Reads ``ACQuire:SEQuence:CURrent?`` - a live progress counter for a
        single-sequence (single/N-acquisition) run.
        """
        return self.query_int("ACQ:SEQ:CURRENT?")

    def set_sequence_mode(self, mode: AcqSequenceMode | str) -> str:
        """Set the acquisition-sequence mode.

        Enum arg: <scope>.AcqSequenceMode.<TAB>
        ### Parameters
        1. mode : str
            - ['NUMACQS']
        ### Returns
        1. mode : str
            - actual sequence mode read back from the scope
        """
        self.write(f"ACQ:SEQ:MOD {token(mode, AcqSequenceMode, 'mode')}")
        return self.query("ACQ:SEQ:MOD?").strip()

    def get_sequence_mode(self) -> AcqSequenceMode:
        """Query the acquisition-sequence mode."""
        return from_response(self.query("ACQ:SEQ:MOD?").strip(), AcqSequenceMode, "sequence_mode")

    def set_fast_average_state(self, state: bool) -> bool:
        """Enable or disable FastAcq averaging; returns the read-back state."""
        self.write(f"ACQ:FASTAV:STATE {'ON' if state else 'OFF'}")
        return self.query_bool("ACQ:FASTAV:STATE?")

    def get_fast_average_state(self) -> bool:
        """Query whether FastAcq averaging is enabled."""
        return self.query_bool("ACQ:FASTAV:STATE?")

    def set_fast_average_limit(self, count: int) -> int:
        """Set the FastAcq-average number of waveforms; returns the read-back count."""
        self.write(f"ACQ:FASTAV:LIM {int(count)}")
        return self.query_int("ACQ:FASTAV:LIM?")

    def set_fast_average_stopafter(self, count: int) -> int:
        """Set the FastAcq-average stop-after count; returns the read-back count."""
        self.write(f"ACQ:FASTAV:STOPA {int(count)}")
        return self.query_int("ACQ:FASTAV:STOPA?")


class _Trigger(Node):
    def set_mode(self, mode: TriggerMode | str) -> None:
        """Set the trigger mode to AUTO or NORMAL

        Enum arg: <scope>.TriggerMode.<TAB>
        ### Parameters
        1. mode : str
            - 'AUTO' or 'NORMAL'
        """
        self.write(f"TRIG:A:MOD {token(mode, TriggerMode, 'mode')}")

    def set_holdoff(self, holdoff: float) -> float:
        """Set the trigger holdoff time in seconds
        ### Parameters
        1. holdoff : float
            - holdoff time in seconds
        ### Returns
        1. holdoff : float
            - actual holdoff time in seconds
        """
        self.write("TRIG:A:HOLD:TIM %g" % holdoff)
        return self.query_float("TRIG:A:HOLD:TIM?")

    def set_autolevel(self) -> None:
        """Set the trigger level to automatic 50%"""
        self.write("TRIG:A SETL")
        self.query("*OPC?")

    def set_edge(
        self,
        source: str,
        level: float | None = None,
        slope: Slope | str = "RISING",
        coupling: TriggerCoupling | str = "DC",
    ) -> float:
        """Enable and configure trigger type as: Edge

        Enum arg: slope    = <scope>.Slope.<TAB>
                  coupling = <scope>.TriggerCoupling.<TAB>
        ### Parameters
        1. source : str
            - ['CHx', 'CHx_Dy', 'LINE', 'AUX']
        2. *level : float (default None)
            - threshold voltage or None for auomatic 50%
        3. *slope : str (default 'RISING')
            - ['RISING', 'FALLING', 'BOTH']
        4. * coupling :str (default='DC')
            - ['DC', 'HF REJECT', 'LF REJECT', 'NOISE REJECT']
        ### Returns
        1. level : float
            - actual trigger level
        """
        self.write("TRIG:A:TYP EDGE")
        self.write("TRIG:A:EDGE:SOU %s" % source)

        if level is not None:
            self.write("TRIG:A:LEV:%s %g" % (source, level))
        else:
            self.write("TRIG:A SETLEVEL")
            self.query("*OPC?")

        self.write(f"TRIG:A:EDGE:SLO {token(slope, Slope, 'slope')}")
        self.write(f"TRIG:A:EDGE:COUP {token(coupling, TriggerCoupling, 'coupling')}")

        return self.query_float("TRIG:A:LEV:%s?" % (source))

    def set_pulse_width(
        self,
        source: str,
        limit: float,
        limit_high: float | None = None,
        level: float | None = None,
        polarity: PulsePolarity | str = "POSITIVE",
        when: PulseWhen | str = "LESSTHAN",
    ) -> float:
        """Enable and configure trigger type as: Pulse

        Enum arg: polarity = <scope>.PulsePolarity.<TAB>
                  when     = <scope>.PulseWhen.<TAB>
        ### Parameters
        1. source : str
            - ['CHx', 'CHx_Dy']
        2. limit : float
            single limit or lower limit in seconds
        3. *limit_high : float (default None)
            - high limit in seconds
        4. *level : float (default None)
            - threshold voltage or None for auomatic 50%
        3. *polarity : str (default 'POSITIVE')
            - ['POSITIVE', 'NEGATIVE']
        4. *when :str (default='LESSTHAN')
            - ['LESSTHAN', 'MORETHAN', 'EQUAL', 'UNEQUAL', 'WITHIN', 'OUTSIDE']
        ### Returns
        1. level : float
            - actual trigger level
        """
        self.write("TRIG:A:TYP WID")
        self.write("TRIG:A:PULSEW:SOU %s" % source)

        if level is not None:
            self.write("TRIG:A:LEV:%s %g" % (source, level))
        else:
            self.write("TRIG:A SETLEVEL")
            self.query("*OPC?")

        self.write(f"TRIG:A:PULSEW:POL {token(polarity, PulsePolarity, 'polarity')}")
        self.write(f"TRIG:A:PULSEW:WHE {token(when, PulseWhen, 'when')}")

        if (when == "OUTSIDE" or when == "WITHIN") and (limit_high is None):
            raise ValueError('"high" limit required when using OUTSIDE or WITHIN "when" operator')

        if limit_high is not None:
            self.write("TRIG:A:PULSEW:HIGHL %g" % limit_high)
        self.write("TRIG:A:PULSEW:LOWL %g" % limit)

        return self.query_float("TRIG:A:LEV:%s?" % (source))

    def set_video(
        self,
        source: str,
        standard: VideoStandard | str = "NTSC",
        field: VideoField | str = "ALL LINES",
        line: int | None = None,
        polarity: VideoPolarity | str = "POSITIVE",
    ) -> str:
        """Enable and configure trigger type as: Video.

        Enum arg: standard = <scope>.VideoStandard.<TAB>
                  field    = <scope>.VideoField.<TAB>
                  polarity = <scope>.VideoPolarity.<TAB>

        ### Parameters
        1. source : str
            - ['CHx']
        2. *standard : str (default 'NTSC')
            - ['NTSC', 'PAL', 'SECAM']
        3. *field : str (default 'ALL LINES')
            - ['ALL LINES', 'NUMERIC', 'ODD', 'EVEN', 'ALL FIELDS']
            - use 'NUMERIC' together with ``line`` to trigger on a specific line
        4. *line : int (default None)
            - line number to trigger on (sets ``field`` to NUMERIC when given)
        5. *polarity : str (default 'POSITIVE')
            - ['POSITIVE', 'NEGATIVE']
        ### Returns
        1. standard : str
            - actual video standard read back from the scope
        """
        self.write("TRIG:A:TYP VIDEO")
        self.write(f"TRIG:A:VID:SOU {source}")
        self.write(f"TRIG:A:VID:STAN {token(standard, VideoStandard, 'standard')}")
        if line is not None:
            self.write_choice("TRIG:A:VID:FIELD", "NUMERIC", _VIDEO_FIELD)
            self.write(f"TRIG:A:VID:LINE {int(line)}")
        else:
            self.write(f"TRIG:A:VID:FIELD {token(field, VideoField, 'field')}")
        self.write(f"TRIG:A:VID:POL {token(polarity, VideoPolarity, 'polarity')}")
        return self.query("TRIG:A:VID:STAN?").strip()

    def set_timeout(
        self,
        source: str,
        time: float,
        polarity: TimeoutPolarity | str = "STAYS HIGH",
        level: float | None = None,
    ) -> float:
        """Enable and configure trigger type as: Timeout.

        Enum arg: <scope>.TimeoutPolarity.<TAB>

        Triggers when the signal stays high/low/either longer than ``time``.

        ### Parameters
        1. source : str
            - ['CHx', 'CHx_Dy']
        2. time : float
            - timeout duration in seconds
        3. *polarity : str (default 'STAYS HIGH')
            - ['STAYS HIGH', 'STAYS LOW', 'EITHER']
        4. *level : float (default None)
            - threshold voltage or None for automatic 50%
        ### Returns
        1. time : float
            - actual timeout duration in seconds
        """
        self.write("TRIG:A:TYP TIMEOUT")
        self.write(f"TRIG:A:TIMEO:SOU {source}")
        self.write(f"TRIG:A:TIMEO:POL {token(polarity, TimeoutPolarity, 'polarity')}")
        if level is not None:
            self.write("TRIG:A:LEV:%s %g" % (source, level))
        else:
            self.write("TRIG:A SETLEVEL")
            self.query("*OPC?")
        self.write("TRIG:A:TIMEO:TIM %g" % time)
        return self.query_float("TRIG:A:TIMEO:TIM?")

    def set_runt(
        self,
        source: str,
        upper: float,
        lower: float,
        polarity: RuntPolarity | str = "POSITIVE",
        when: RuntWhen | str = "OCCURS",
        width: float | None = None,
    ) -> float:
        """Enable and configure trigger type as: Runt.

        Enum arg: polarity = <scope>.RuntPolarity.<TAB>
                  when     = <scope>.RuntWhen.<TAB>

        A runt crosses one threshold but not the other. ``when`` LESSTHAN /
        MORETHAN / EQUAL / UNEQUAL additionally qualify on the runt ``width``.

        ### Parameters
        1. source : str
            - ['CHx']
        2. upper : float
            - upper threshold voltage
        3. lower : float
            - lower threshold voltage
        4. *polarity : str (default 'POSITIVE')
            - ['POSITIVE', 'NEGATIVE', 'EITHER']
        5. *when : str (default 'OCCURS')
            - ['OCCURS', 'LESSTHAN', 'MORETHAN', 'EQUAL', 'UNEQUAL']
        6. *width : float (default None)
            - runt width in seconds (required unless ``when`` is 'OCCURS')
        ### Returns
        1. (upper, lower) : tuple[float, float]
            - actual upper and lower thresholds in volts
        """
        if when != "OCCURS" and width is None:
            raise ValueError("width required unless 'when' is 'OCCURS'")
        self.write("TRIG:A:TYP RUNT")
        self.write(f"TRIG:A:RUNT:SOU {source}")
        self.write(f"TRIG:A:RUNT:POL {token(polarity, RuntPolarity, 'polarity')}")
        self.write(f"TRIG:A:RUNT:WHE {token(when, RuntWhen, 'when')}")
        self.write("TRIG:A:UPP:%s %g" % (source, upper))
        self.write("TRIG:A:LOW:%s %g" % (source, lower))
        if width is not None:
            self.write("TRIG:A:RUNT:WID %g" % width)
        return (
            self.query_float("TRIG:A:UPP:%s?" % source),
            self.query_float("TRIG:A:LOW:%s?" % source),
        )

    def set_window(
        self,
        source: str,
        upper: float,
        lower: float,
        when: WindowWhen | str = "ENTERS",
        crossing: WindowCrossing | str = "UPPER",
        width: float | None = None,
    ) -> float:
        """Enable and configure trigger type as: Window.

        Enum arg: when     = <scope>.WindowWhen.<TAB>
                  crossing = <scope>.WindowCrossing.<TAB>

        Triggers on the signal entering/exiting a voltage window, or staying
        inside/outside it longer than ``width``.

        ### Parameters
        1. source : str
            - ['CHx']
        2. upper : float
            - upper window threshold voltage
        3. lower : float
            - lower window threshold voltage
        4. *when : str (default 'ENTERS')
            - ['ENTERS', 'EXITS', 'INSIDE', 'OUTSIDE']
        5. *crossing : str (default 'UPPER')
            - ['UPPER', 'LOWER', 'EITHER', 'NONE'] (event level for ENTERS/EXITS)
        6. *width : float (default None)
            - time in seconds (required for INSIDE/OUTSIDE)
        ### Returns
        1. (upper, lower) : tuple[float, float]
            - actual upper and lower thresholds in volts
        """
        if when in ("INSIDE", "OUTSIDE") and width is None:
            raise ValueError("width required when 'when' is 'INSIDE' or 'OUTSIDE'")
        self.write("TRIG:A:TYP WINDOW")
        self.write(f"TRIG:A:WIN:SOU {source}")
        self.write(f"TRIG:A:WIN:WHE {token(when, WindowWhen, 'when')}")
        self.write(f"TRIG:A:WIN:CROSSI {token(crossing, WindowCrossing, 'crossing')}")
        self.write("TRIG:A:UPP:%s %g" % (source, upper))
        self.write("TRIG:A:LOW:%s %g" % (source, lower))
        if width is not None:
            self.write("TRIG:A:WIN:WID %g" % width)
        return (
            self.query_float("TRIG:A:UPP:%s?" % source),
            self.query_float("TRIG:A:LOW:%s?" % source),
        )

    def set_transition(
        self,
        source: str,
        delta_time: float,
        upper: float,
        lower: float,
        polarity: RuntPolarity | str = "POSITIVE",
        when: TransitionWhen | str = "FASTER",
    ) -> float:
        """Enable and configure trigger type as: Transition (rise/fall time).

        Enum arg: polarity = <scope>.RuntPolarity.<TAB>
                  when     = <scope>.TransitionWhen.<TAB>

        Triggers when an edge transitions between the two thresholds faster or
        slower than ``delta_time``.

        ### Parameters
        1. source : str
            - ['CHx']
        2. delta_time : float
            - transition time in seconds to compare against
        3. upper : float
            - upper threshold voltage
        4. lower : float
            - lower threshold voltage
        5. *polarity : str (default 'POSITIVE')
            - ['POSITIVE', 'NEGATIVE', 'EITHER']
        6. *when : str (default 'FASTER')
            - ['FASTER', 'SLOWER', 'EQUAL', 'UNEQUAL']
        ### Returns
        1. delta_time : float
            - actual transition time comparison value in seconds
        """
        self.write("TRIG:A:TYP TRANSITION")
        self.write(f"TRIG:A:TRAN:SOU {source}")
        self.write(f"TRIG:A:TRAN:POL {token(polarity, RuntPolarity, 'polarity')}")
        self.write(f"TRIG:A:TRAN:WHE {token(when, TransitionWhen, 'when')}")
        self.write("TRIG:A:UPP:%s %g" % (source, upper))
        self.write("TRIG:A:LOW:%s %g" % (source, lower))
        self.write("TRIG:A:TRAN:DELT %g" % delta_time)
        return self.query_float("TRIG:A:TRAN:DELT?")

    def set_holdoff_by(self, mode: HoldoffBy | str) -> str:
        """Set the A-trigger holdoff type (user TIME or internal RANDOM).

        Enum arg: <scope>.HoldoffBy.<TAB>
        ### Parameters
        1. mode : str
            - ['TIME', 'RANDOM']
        ### Returns
        1. mode : str
            - actual holdoff type read back from the scope
        """
        self.write(f"TRIG:A:HOLD:BY {token(mode, HoldoffBy, 'mode')}")
        return self.query("TRIG:A:HOLD:BY?").strip()

    def get_holdoff_by(self) -> HoldoffBy:
        """Query the A-trigger holdoff type (TIME/RANDOM)."""
        return from_response(self.query("TRIG:A:HOLD:BY?").strip(), HoldoffBy, "holdoff_by")

    def set_user_hysteresis(self, state: bool, value: float | None = None) -> float | None:
        """Enable user-defined trigger hysteresis and optionally set its value.
        ### Parameters
        1. state : bool
            - True to enable user hysteresis, False to use the automatic value
        2. value : float (default None)
            - hysteresis value (only written when given)
        ### Returns
        1. value : float | None
            - actual user hysteresis value, or None if only the state was changed
        """
        self.write(f"TRIGger:HYSTeresis:USER:STATe {'ON' if state else 'OFF'}")
        if value is not None:
            self.write(f"TRIGger:HYSTeresis:USER:VALue {value:.12g}")
            return self.query_float("TRIGger:HYSTeresis:USER:VALue?")
        return None

    def get_user_hysteresis(self) -> tuple[bool, float]:
        """Query the user-hysteresis (enabled, value) pair."""
        return (
            self.query_bool("TRIGger:HYSTeresis:USER:STATe?"),
            self.query_float("TRIGger:HYSTeresis:USER:VALue?"),
        )

    def get_state(self) -> str:
        """Query the trigger-system state.

        Returns tokens such as ARMED / AUTO / READY / SAVE / TRIGGER. Prefer
        this over :meth:`get_status` - the ``TRIGger:STATUs?`` query is not
        implemented on this firmware and hangs (ADR-015).
        """
        return self.query("TRIGger:STATE?").strip()

    def set_b_state(self, state: bool) -> bool:
        """Enable or disable the B (delayed) trigger in the A->B sequence.
        ### Parameters
        1. state : bool
            - True to include the B trigger in the sequence, False for A only
        ### Returns
        1. state : bool
            - actual B-trigger state read back from the scope
        """
        self.write(f"TRIGger:B:STATE {'ON' if state else 'OFF'}")
        return self.query_bool("TRIGger:B:STATE?")

    def get_b_state(self) -> bool:
        """Query whether the B trigger is active."""
        return self.query_bool("TRIGger:B:STATE?")

    def set_b_by(self, mode: TriggerBBy | str) -> str:
        """Set whether the B trigger fires after N events or a set time.

        Enum arg: <scope>.TriggerBBy.<TAB>
        ### Parameters
        1. mode : str
            - ['EVENTS', 'TIME']
        ### Returns
        1. mode : str
            - actual B-trigger delay type read back from the scope
        """
        self.write(f"TRIGger:B:BY {token(mode, TriggerBBy, 'mode')}")
        return self.query("TRIGger:B:BY?").strip()

    def set_b_events_count(self, count: int) -> int:
        """Set the number of events before the B trigger; returns readback."""
        self.write(f"TRIGger:B:EVENTS:COUNt {int(count)}")
        return self.query_int("TRIGger:B:EVENTS:COUNt?")

    def set_b_time(self, seconds: float) -> float:
        """Set the B-trigger delay time in seconds; returns readback."""
        self.write(f"TRIGger:B:TIMe {seconds:.12g}")
        return self.query_float("TRIGger:B:TIMe?")

    def set_b_edge(
        self,
        source: str,
        level: float | None = None,
        slope: Slope | str = "RISING",
        coupling: TriggerCoupling | str = "DC",
    ) -> float:
        """Configure the B-trigger edge event (source/slope/level/coupling).

        Enum arg: slope    = <scope>.Slope.<TAB>
                  coupling = <scope>.TriggerCoupling.<TAB>

        Wraps the ``TRIGger:B:EDGE:*`` branch plus ``TRIGger:B:LEVel:<source>``.
        That branch is not in the published command list for this family but is
        real on the bench firmware (FV:2.6.38.1348) and answers cleanly - see
        ADR-016. Set the B state and delay type separately via
        :meth:`set_b_state` / :meth:`set_b_by`.

        ### Parameters
        1. source : str
            - ['CHx', 'CHx_Dy']
        2. *level : float (default None)
            - B-trigger threshold voltage; None leaves the current level
        3. *slope : str (default 'RISING')
            - ['RISING', 'FALLING', 'BOTH']
        4. *coupling : str (default 'DC')
            - ['DC', 'HF REJECT', 'LF REJECT', 'NOISE REJECT']
        ### Returns
        1. level : float
            - actual B-trigger level read back from the scope
        """
        self.write("TRIGger:B:TYPe EDGE")
        self.write(f"TRIGger:B:EDGE:SOUrce {source}")
        self.write(f"TRIGger:B:EDGE:SLOpe {token(slope, Slope, 'slope')}")
        self.write(f"TRIGger:B:EDGE:COUPling {token(coupling, TriggerCoupling, 'coupling')}")
        if level is not None:
            self.write(f"TRIGger:B:LEVel:{source} {level:.12g}")
        return self.query_float(f"TRIGger:B:LEVel:{source}?")


class _Channel(Node):
    def __init__(self, link: InstrumentLink, channel: int) -> None:
        super().__init__(link, name=f"CH{channel}")
        self._ch = channel
        self.sv = _Spectrum(link, name=f"CH{channel}:SV", channel=channel)
        self.probe = _Probe(link, channel)

    def set_bandwidth(self, bw: float | Literal["FULL"]) -> float | str:
        """Set the channel bandwidth.

        The instrument rounds a numeric request to the nearest available
        bandwidth (a :class:`ValueSnappedWarning` is emitted if it snaps). The
        available bandwidths depend on the model, termination, and attached
        probe. Pass the token ``"FULL"`` to disable bandwidth limiting and run
        at the channel's maximum bandwidth.

        ### Parameters
        1. bw : float | "FULL"
            - bandwidth in Hz, or the token "FULL"
        ### Returns
        1. bw : float | str
            - realized bandwidth in Hz (float), or the token echoed back
        """
        if isinstance(bw, str):
            token = bw.strip().upper()
            if token != "FULL":
                raise ValueError(f"Invalid bandwidth token {bw!r}; only 'FULL' is documented")
            return self.set_and_verify(f"CH{self._ch}:BAN", "FUL")
        return self.set_and_verify(f"CH{self._ch}:BAN", bw)

    def set_coupling(self, coupling: Coupling | str) -> None:
        """Set the channel input coupling

        Enum arg: <scope>.Coupling.<TAB>
        ### Options
        ['AC', 'DC', 'DC REJECT']
        * AC only available with 1MΩ termination
        ### Parameters
        1. coupling : str
            - input coupling mode
        """
        self.write(f"CH{self._ch}:COUP {token(coupling, Coupling, 'coupling')}")

    def get_clipping(self) -> bool:
        """Query if a channel is clipping (out of ADC range)
        ### Returns
        1. cliping : bool
            - True if clipping, false if OK
        """
        return self.query_bool(f"CH{self._ch}:CLIP?")

    def set_scale(self, scale: float) -> float:
        """Set the vertical scale in volts per division
        ### Parameters
        1. scale : float
            - volts per division
        ## Returns
        1. scale : float
            - actual volts per division
        """
        self.write(f"CH{self._ch}:SCA {scale:.12g}")
        return self.query_float(f"CH{self._ch}:SCA?")

    def get_scale(self) -> float:
        """Get the vertical scale in volts per division
        ## Returns
        1. scale : float
            - actual volts per division
        """
        return self.query_float(f"CH{self._ch}:SCA?")

    def set_position(self, position: float) -> None:
        """Set the vertical channel position in divisions (-5 to +5)
        ### Parameters
        1. position : float
            - vertical position in divisions
        """
        self.write(f"CH{self._ch}:POS {position:.12g}")

    def set_offset(self, offset: float) -> float:
        """Set the channel offset in volts
        ### Parameters
        1. offset : float
            - offset voltage
        ### Returns
        1. offset : float
            - actual offset voltage
        """
        self.write(f"CH{self._ch}:OFFS {offset:.12g}")
        return self.query_float(f"CH{self._ch}:OFFS?")

    def set_termination(self, termination: float) -> float:
        """Set the channel termination impedance.
        *[50, 1e6]
        ### Parameters
        1. termination : float
            - termination impedance in ohms]
        ### Returns
        1. termination : float
            - actual termination impedance in ohms
        """
        self.write(f"CH{self._ch}:TER {termination:.12g}")
        return self.query_float(f"CH{self._ch}:TER?")

    def set_units(self, units: str | None) -> str:
        """Set the external unit string.

        The units string is sent as a quoted SCPI argument. An embedded double
        quote would let the argument break out of its quoting, so a units value
        containing ``"`` is rejected with :class:`ValueError` rather than being
        silently escaped or truncated.

        ### Parameters
        1. units : str | None
            - units display string, or None to disable external units
        ### Returns
        1. units : str
            - actual units display string (only when a string is set)
        """
        if units is not None:
            if '"' in units:
                raise ValueError(
                    f"units string must not contain a double-quote character; got {units!r}"
                )
            self.write(f"CH{self._ch}:PROBEF:EXTU:STATE ON")
            self.write(f'CH{self._ch}:PROBEF:EXTU "{units}"')
            return self.query(f"CH{self._ch}:PROBEF:EXTU?")
        else:
            self.write(f"CH{self._ch}:PROBEF:EXTU:STATE 0")

    def set_attenuation(self, atten: float) -> float:
        """Set the channel external attenuation (i.e. 200 = 200x)
        ### Parameters
        1. atten : float
            - external attenuation factor
        ### Returns
        1. atten : float
            - actual external attenuation factor
        """
        self.write(f"CH{self._ch}:PROBEF:EXTA {atten:.12g}")
        return self.query_float(f"CH{self._ch}:PROBEF:EXTA?")

    def set_invert(self, invert: bool) -> None:
        """Set the channel invert state
        ### Parameters
        1. invert : bool
            - True -> inverted
            - False -> non-inverted
        """
        self.write(f"CH{self._ch}:INV {int(bool(invert))}")

    def set_display_state(self, state: bool) -> bool:
        """Show or hide this channel's waveform. Returns the read-back state.

        This controls visibility only. It does not stop the channel from being
        acquired or measured.
        """
        self.write(f"DIS:WAVEV:CH{self._ch}:STATE {'ON' if state else 'OFF'}")
        return self.get_display_state()

    def get_display_state(self) -> bool:
        """Read whether this channel's waveform is shown."""
        return self.query_bool(f"DIS:WAVEV:CH{self._ch}:STATE?")

    def get_bandwidth(self) -> float:
        """Get the realized channel bandwidth in Hz."""
        return self.query_float(f"CH{self._ch}:BAN?")

    def set_bandwidth_optimization(self, mode: BandwidthOptimization | str) -> None:
        """Set the bandwidth-limit filter optimization.

        Enum arg: <scope>.BandwidthOptimization.<TAB>
        ### Parameters
        1. mode : str
            - ['STEP RESPONSE', 'FLATNESS']
        """
        self.write(
            f"CH{self._ch}:BAN:FILT:OPTIMIZATION {token(mode, BandwidthOptimization, 'mode')}"
        )

    def set_label(self, label: str) -> str:
        """Set the channel label (waveform name badge).

        The label is sent as a quoted SCPI argument; an embedded double quote is
        rejected with :class:`ValueError` so it cannot break out of its quoting.

        ### Parameters
        1. label : str
            - label display string
        ### Returns
        1. label : str
            - actual label string read back from the scope
        """
        if '"' in label:
            raise ValueError(f"label must not contain a double-quote character; got {label!r}")
        self.write(f'CH{self._ch}:LAB:NAM "{label}"')
        return self.query(f"CH{self._ch}:LAB:NAM?").strip().strip('"')

    def get_label(self) -> str:
        """Get the channel label string."""
        return self.query(f"CH{self._ch}:LAB:NAM?").strip().strip('"')

    def set_deskew(self, deskew: float) -> float:
        """Set the channel deskew time in seconds.
        ### Parameters
        1. deskew : float
            - deskew time in seconds
        ### Returns
        1. deskew : float
            - actual deskew time in seconds
        """
        self.write(f"CH{self._ch}:DESK {deskew:.12g}")
        return self.query_float(f"CH{self._ch}:DESK?")

    def get_deskew(self) -> float:
        """Get the channel deskew time in seconds."""
        return self.query_float(f"CH{self._ch}:DESK?")

    def get_offset(self) -> float:
        """Get the channel offset in volts."""
        return self.query_float(f"CH{self._ch}:OFFS?")

    def get_termination(self) -> float:
        """Get the channel termination impedance in ohms."""
        return self.query_float(f"CH{self._ch}:TER?")

    def set_probe_control(self, mode: ProbeControl | str) -> None:
        """Set whether the probe attenuation is chosen automatically or manually.

        Enum arg: <scope>.ProbeControl.<TAB>
        ### Parameters
        1. mode : str
            - ['AUTO', 'MANUAL']
        """
        self.write(f"CH{self._ch}:PROBECO {token(mode, ProbeControl, 'mode')}")

    def get_probe_type(self) -> str:
        """Query the attached probe type (e.g. ANALOG, DIGITAL)."""
        return self.query(f"CH{self._ch}:PROBETYPE?").strip()

    def get_config(self) -> list[str]:
        """Query the full channel configuration string, parsed.

        The scope answers with a long semicolon-delimited compound response
        whose field order is undocumented for this family (ADR-018), so the
        result is returned as an ordered list of parsed fields (quote-aware,
        no naive quote strip) rather than a named dict.
        ### Returns
        1. fields : list[str]
            - the ordered configuration fields
        """
        return parse_compound(self.query(f"CH{self._ch}?"))

    def get_probe_config(self) -> list[str]:
        """Query the full probe configuration string, parsed.

        Returned as an ordered list of parsed fields (see :meth:`get_config`).
        """
        return parse_compound(self.query(f"CH{self._ch}:PRObe?"))

    def get_probe_cal_status(self) -> str:
        """Query the probe-calibration status.

        One of DEFAULT / FAILED / PASSED / RUNNING.
        """
        return self.query(f"CH{self._ch}:PROBECal?").strip()

    def set_selected(self, state: bool) -> bool:
        """Turn this channel on/off and select it.

        This is the vertical-domain channel-enable/select. Contrast with
        :meth:`set_display_state` (``DISplay:WAVEView`` waveform visibility) - this
        one both enables the channel and makes it the selected source.
        ### Parameters
        1. state : bool
            - True to enable/select the channel, False to turn it off
        ### Returns
        1. state : bool
            - actual selected state read back from the scope
        """
        self.write(f"SELect:CH{self._ch} {'ON' if state else 'OFF'}")
        return self.query_bool(f"SELect:CH{self._ch}?")

    def get_selected(self) -> bool:
        """Query whether this channel is on/selected."""
        return self.query_bool(f"SELect:CH{self._ch}?")

    def set_ext_db_attenuation(self, db: float) -> float:
        """Set the external attenuation expressed in dB.
        ### Parameters
        1. db : float
            - external attenuation in dB
        ### Returns
        1. db : float
            - actual external dB attenuation read back from the scope
        """
        self.write(f"CH{self._ch}:PROBEFunc:EXTDBatten {db:.12g}")
        return self.query_float(f"CH{self._ch}:PROBEFunc:EXTDBatten?")

    def get_ext_db_attenuation(self) -> float:
        """Query the external attenuation in dB."""
        return self.query_float(f"CH{self._ch}:PROBEFunc:EXTDBatten?")


class _Spectrum(Node):
    def __init__(self, link: InstrumentLink, name: str = "SV", channel: int | None = None) -> None:
        super().__init__(link, name=name)
        self._ch = channel

    def _require_channel(self) -> int:
        if self._ch is None:
            raise ValueError("this command is only valid on a per-channel Spectrum View node")
        return self._ch

    def set_center(self, frequency: float) -> None:
        """Set the spectrum view global center frequency (all channels locked).

        Sets ``SV:LOCKC ON`` so every channel's center frequency tracks the
        one written to CH1 - use this for a common center across channels. For
        an independent per-channel center, use ``set_channel_center`` (on a
        per-channel node), which first sets ``SV:LOCKC OFF``.

        ### Parameters:
        1. frequency : float
            - center frequency in Hz
        """
        self.write("SV:LOCKC ON")
        self.write(f"CH1:SV:CENTERF {frequency:.12g}")

    def set_channel_center(self, frequency: float) -> float:
        """Set THIS channel's spectrum center frequency independently (per-channel).

        Unlocks the global center (``SV:LOCKC OFF``) so channels can hold
        different centers, then writes ``CH<x>:SV:CENTERFrequency``. Contrast
        with the global ``set_center`` (all channels locked together).

        ### Parameters
        1. frequency : float
            - center frequency in Hz
        ### Returns
        1. frequency : float
            - actual center frequency read back from the scope
        """
        ch = self._require_channel()
        self.write("SV:LOCKC OFF")
        self.write(f"CH{ch}:SV:CENTERF {frequency:.12g}")
        return self.query_float(f"CH{ch}:SV:CENTERF?")

    def get_channel_center(self) -> float:
        """Query this channel's spectrum center frequency in Hz (per-channel)."""
        ch = self._require_channel()
        return self.query_float(f"CH{ch}:SV:CENTERF?")

    def set_span(self, frequency: float) -> None:
        """Set the spectrum view global frequency span
        ### Parameters:
        1. frequency : float
            - frequency span in Hz
        """
        self.write(f"SV:SPAN {frequency:.12g}")

    def set_rbw(self, frequency: float) -> None:
        """Set the spectrum view manual resolution bandwidth
        ### Parameters:
        1. frequency : float
            - resolution bandwidth span in Hz
        """
        self.write("SV:RBWM MAN")
        self.write(f"SV:RBW {frequency:.12g}")

    def set_span_rbw_ratio(self, ratio: float) -> float:
        """Set the spectrum view automatic span to resolution bandwidth ratio
        ### Parameters
        1. ratio : float
            - ratio (20, 50, 100, 200, ... , 1M)
        ### Returns
        1. ratio : float
            - actual ratio
        """
        self.write("SV:RBWM AUTO")
        self.write(f"SV:SPANRBWR {ratio:.12g}")
        return self.query_float("SV:SPANRBWR?")

    def set_window(self, window: SpectrumWindow | str) -> None:
        """Set the spectrum view window function

        Enum arg: <scope>.SpectrumWindow.<TAB>
        ### Parameters
        1. window : str
            - ['KAISER', 'RECT', 'HAMMING', 'HANNING', 'BLACKMAN', 'FLATTOP']
        """
        self.write(f"SV:WINDOW {token(window, SpectrumWindow, 'window')}")

    def set_peak_markers(self, state: bool) -> None:
        """Turn the automatic peak markers on or off (global Spectrum View)."""
        self.write(f"SV:MARKER:PEAK:STATE {'ON' if state else 'OFF'}")

    def get_peak_markers(self) -> bool:
        """Query whether the automatic peak markers are on."""
        return self.query_bool("SV:MARKER:PEAK:STATE?")

    def set_peak_count(self, count: int) -> int:
        """Set the maximum number of peak markers (1 to 11).
        ### Parameters
        1. count : int
            - number of peak markers
        ### Returns
        1. count : int
            - actual maximum number of peak markers
        """
        self.write(f"SV:MARKER:PEAK:MAX {int(count)}")
        return self.query_int("SV:MARKER:PEAK:MAX?")

    def set_peak_threshold(self, threshold: float) -> float:
        """Set the peak-marker amplitude threshold (only peaks above it count).
        ### Parameters
        1. threshold : float
            - amplitude threshold in the trace's vertical units
        ### Returns
        1. threshold : float
            - actual threshold read back from the scope
        """
        self.write(f"SV:MARKER:PEAK:THR {threshold:.12g}")
        return self.query_float("SV:MARKER:PEAK:THR?")

    def set_peak_excursion(self, excursion: float) -> float:
        """Set the peak-marker excursion (min rise+fall around a peak to count).
        ### Parameters
        1. excursion : float
            - excursion in dB
        ### Returns
        1. excursion : float
            - actual excursion read back from the scope
        """
        self.write(f"SV:MARKER:PEAK:EXCUR {excursion:.12g}")
        return self.query_float("SV:MARKER:PEAK:EXCUR?")

    def set_marker_type(self, marker_type: SpectrumMarkerType | str) -> None:
        """Set the peak marker readout type.

        Enum arg: <scope>.SpectrumMarkerType.<TAB>
        ### Parameters
        1. marker_type : str
            - ['ABSOLUTE', 'DELTA']
        """
        self.write(f"SV:MARKER:TYP {token(marker_type, SpectrumMarkerType, 'marker_type')}")

    def get_peak_frequencies(self) -> list[float]:
        """Return the frequencies (Hz) of the active peak markers."""
        return self._parse_float_list(self.query("SV:MARKER:PEAKS:FREQ?"))

    def get_peak_amplitudes(self) -> list[float]:
        """Return the amplitudes of the active peak markers (trace units)."""
        return self._parse_float_list(self.query("SV:MARKER:PEAKS:AMPLITUDE?"))

    def get_peaks(self) -> list[tuple[float, float]]:
        """Return the active peaks as ``(frequency_hz, amplitude)`` tuples."""
        freqs = self.get_peak_frequencies()
        amps = self.get_peak_amplitudes()
        return list(zip(freqs, amps))

    def get_reference_frequency(self) -> float:
        """Query the reference-marker frequency in Hz."""
        return self.query_float("SV:MARKER:REFER:FREQ?")

    def get_reference_amplitude(self) -> float:
        """Query the reference-marker amplitude in the trace's vertical units."""
        return self.query_float("SV:MARKER:REFER:AMPLITUDE?")

    def cmd_marker_to_center(self) -> None:
        """Move the reference marker to screen center (set center to its freq)."""
        self.write("SV:MARKER:REFER")

    @staticmethod
    def _parse_float_list(response: str) -> list[float]:
        out = []
        for piece in response.strip().split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                out.append(float(piece))
            except ValueError:
                continue
        return out

    def set_state(self, state: bool) -> None:
        """Enable or disable this channel's Spectrum View trace (per-channel)."""
        ch = self._require_channel()
        self.write(f"CH{ch}:SV:STATE {'ON' if state else 'OFF'}")

    def set_units(self, units: SpectrumUnits | str) -> None:
        """Set this channel's Spectrum View vertical units (per-channel).

        Enum arg: <scope>.SpectrumUnits.<TAB>
        ### Parameters
        1. units : str
            - ['DBM', 'DBUW', 'DBMV', 'DBUV', 'DBMA', 'DBUA']
        """
        ch = self._require_channel()
        self.write(f"SV:CH{ch}:UNI {token(units, SpectrumUnits, 'units')}")

    def set_select_trace(self, trace: SpectrumTraceSelect | str) -> None:
        """Select this channel's active Spectrum View trace (per-channel).

        Enum arg: <scope>.SpectrumTraceSelect.<TAB>
        ### Parameters
        1. trace : str
            - ['NORMAL', 'MAX HOLD', 'MIN HOLD', 'AVERAGE']
        """
        ch = self._require_channel()
        self.write(f"SV:CH{ch}:SELT {token(trace, SpectrumTraceSelect, 'trace')}")

    def set_scale(self, db_per_div: float, view: int = 1) -> float:
        """Set this channel's Spectrum View vertical scale in dB/div (per-channel).

        Continuous value - the scope does not snap it (bench-verified,
        e.g. 7 dB/div is accepted verbatim). Independent of units/position.

        ### Parameters
        1. db_per_div : float
            - vertical scale in dB per division
        2. *view : int (default 1)
            - Spectrum View index
        ### Returns
        1. db_per_div : float
            - actual vertical scale read back from the scope
        """
        ch = self._require_channel()
        self.write(f"DISPLAY:SPECVIEW{view}:CH{ch}:VERTICAL:SCALE {db_per_div:.12g}")
        return self.query_float(f"DISPLAY:SPECVIEW{view}:CH{ch}:VERTICAL:SCALE?")

    def get_scale(self, view: int = 1) -> float:
        """Query this channel's Spectrum View vertical scale in dB/div (per-channel)."""
        ch = self._require_channel()
        return self.query_float(f"DISPLAY:SPECVIEW{view}:CH{ch}:VERTICAL:SCALE?")

    def set_position(self, divisions: float, view: int = 1) -> float:
        """Set this channel's Spectrum View vertical DISPLAY position in divisions.

         This is the on-screen vertical placement of the trace
        . It is NOT the
         Spectrum Time control - for that see ``set_spectrum_time`` (which wraps
         the misleadingly named ``CH<x>:SV:POSition``). Fractional and negative
         values accepted (bench-verified).

         ### Parameters
         1. divisions : float
             - vertical position in divisions
         2. *view : int (default 1)
             - Spectrum View index
         ### Returns
         1. divisions : float
             - actual vertical position read back from the scope
        """
        ch = self._require_channel()
        self.write(f"DISPLAY:SPECVIEW{view}:CH{ch}:VERTICAL:POSITION {divisions:.12g}")
        return self.query_float(f"DISPLAY:SPECVIEW{view}:CH{ch}:VERTICAL:POSITION?")

    def get_position(self, view: int = 1) -> float:
        """Query this channel's Spectrum View vertical DISPLAY position in divisions."""
        ch = self._require_channel()
        return self.query_float(f"DISPLAY:SPECVIEW{view}:CH{ch}:VERTICAL:POSITION?")

    def set_spectrum_time(self, percent: float) -> float:
        """Set this channel's Spectrum Time as a percent of the record (per-channel).

        Wraps ``CH<x>:SV:POSition`` - a FALSE FRIEND whose SCPI name says
        "POSition" but which actually sets the Spectrum Time (the point in the
        acquisition record the spectrum is computed at), 0.0% to 100%. The
        on-screen vertical position is the separate ``set_position`` above;
        the two must not be confused.

        ### Parameters
        1. percent : float
            - spectrum time as a percentage of the record length (0.0 to 100.0)
        ### Returns
        1. percent : float
            - actual spectrum time read back from the scope
        """
        ch = self._require_channel()
        self.write(f"CH{ch}:SV:POSition {percent:.12g}")
        return self.query_float(f"CH{ch}:SV:POSition?")

    def get_spectrum_time(self) -> float:
        """Query this channel's Spectrum Time as a percent of the record (per-channel).

        Reads ``CH<x>:SV:POSition`` (Spectrum Time, not vertical position).
        """
        ch = self._require_channel()
        return self.query_float(f"CH{ch}:SV:POSition?")

    def set_squelch_state(self, state: bool) -> bool:
        """Enable/disable Squelch for this channel's RF-vs-Time traces (per-channel).

        ### Parameters
        1. state : bool
            - True enables squelch, False disables
        ### Returns
        1. state : bool
            - actual squelch state read back from the scope
        """
        ch = self._require_channel()
        self.write(f"SV:CH{ch}:SQUELCH:STATE {'ON' if state else 'OFF'}")
        return self.query_bool(f"SV:CH{ch}:SQUELCH:STATE?")

    def get_squelch_state(self) -> bool:
        """Query this channel's Squelch state (per-channel)."""
        ch = self._require_channel()
        return self.query_bool(f"SV:CH{ch}:SQUELCH:STATE?")

    def set_squelch_threshold(self, threshold: float) -> float:
        """Set this channel's Squelch threshold in volts (per-channel).

        ### Parameters
        1. threshold : float
            - squelch threshold in volts
        ### Returns
        1. threshold : float
            - actual squelch threshold read back from the scope
        """
        ch = self._require_channel()
        self.write(f"SV:CH{ch}:SQUELCH:THReshold {threshold:.12g}")
        return self.query_float(f"SV:CH{ch}:SQUELCH:THReshold?")

    def get_squelch_threshold(self) -> float:
        """Query this channel's Squelch threshold in volts (per-channel)."""
        ch = self._require_channel()
        return self.query_float(f"SV:CH{ch}:SQUELCH:THReshold?")

    def get_start_frequency(self) -> float:
        """Query this channel's Spectrum View start frequency in Hz (per-channel)."""
        ch = self._require_channel()
        return self.query_float(f"CH{ch}:SV:STARTF?")

    def get_stop_frequency(self) -> float:
        """Query this channel's Spectrum View stop frequency in Hz (per-channel)."""
        ch = self._require_channel()
        return self.query_float(f"CH{ch}:SV:STOPF?")

    def get_span_above_bw(self) -> float:
        """How far above the channel bandwidth this channel's spectrum extends, in Hz."""
        ch = self._require_channel()
        return self.query_float(f"CH{ch}:SV:SPANABovebw?")

    def get_span_below_dc(self) -> float:
        """How far below DC this channel's spectrum extends, in Hz."""
        ch = self._require_channel()
        return self.query_float(f"CH{ch}:SV:SPANBELowdc?")

    def set_rf_normal(self, state: bool) -> bool:
        """Enable/disable this channel's RF-vs-Time normal trace (per-channel).

        Wraps ``SV:CH<x>:SELect:RF_NORMal``; returns the read-back state.
        """
        ch = self._require_channel()
        self.write(f"SV:CH{ch}:SELect:RF_NORMal {'ON' if state else 'OFF'}")
        return self.query_bool(f"SV:CH{ch}:SELect:RF_NORMal?")

    def set_rf_phase(self, state: bool) -> bool:
        """Enable/disable this channel's RF phase-vs-Time trace (per-channel).

        Wraps ``SV:CH<x>:SELect:RF_PHASe``; returns the read-back state.
        """
        ch = self._require_channel()
        self.write(f"SV:CH{ch}:SELect:RF_PHASe {'ON' if state else 'OFF'}")
        return self.query_bool(f"SV:CH{ch}:SELect:RF_PHASe?")

    def set_spectrogram(self, state: bool) -> bool:
        """Enable/disable this channel's spectrogram (per-channel).

        Wraps ``SV:CH<x>:SELect:SPECtrogram``; returns the read-back state.
        """
        ch = self._require_channel()
        self.write(f"SV:CH{ch}:SELect:SPECtrogram {'ON' if state else 'OFF'}")
        return self.query_bool(f"SV:CH{ch}:SELect:SPECtrogram?")

    def set_lock_spectrum(self, state: bool) -> bool:
        """Lock all spectrum traces together; returns readback."""
        self.write(f"SV:LOCKSpectrum {'ON' if state else 'OFF'}")
        return self.query_bool("SV:LOCKSpectrum?")

    def get_lock_spectrum(self) -> bool:
        """Query whether spectrum traces are locked together."""
        return self.query_bool("SV:LOCKSpectrum?")

    def set_spectrogram_cscale(self, minimum: float, maximum: float) -> tuple[float, float]:
        """Set the spectrogram color-scale min and max in dB (global SV).
        ### Parameters
        1. minimum : float
            - color-scale minimum in dBm
        2. maximum : float
            - color-scale maximum in dBm
        ### Returns
        1. (minimum, maximum) : tuple[float, float]
            - actual color-scale min/max read back from the scope
        """
        self.write(f"SV:SPECtrogram:CSCale:MIN {minimum:.12g}")
        self.write(f"SV:SPECtrogram:CSCale:MAX {maximum:.12g}")
        return (
            self.query_float("SV:SPECtrogram:CSCale:MIN?"),
            self.query_float("SV:SPECtrogram:CSCale:MAX?"),
        )

    def set_spectrogram_cursor(self, cursor: str, state: bool) -> bool:
        """Enable/disable a spectrogram slice cursor (global SV).
        ### Parameters
        1. cursor : str
            - ['A', 'B']
        2. state : bool
            - True to enable, False to disable
        ### Returns
        1. state : bool
            - actual cursor state read back from the scope
        """
        key = cursor.strip().upper()
        if key not in ("A", "B"):
            raise ValueError(f"cursor must be 'A' or 'B'; got {cursor!r}")
        self.write(f"SV:SPECtrogram:CURSor:{key} {'ON' if state else 'OFF'}")
        return self.query_bool(f"SV:SPECtrogram:CURSor:{key}?")

    def set_rf_phase_reference_master(self, source: str) -> str:
        """Set the RF-phase reference master source (global SV).
        ### Parameters
        1. source : str
            - a channel token (e.g. ``CH1``) or ``NONE``
        ### Returns
        1. source : str
            - actual reference master read back from the scope
        """
        self.write(f"SV:RF_PHASe:REFerence:MASTer {source}")
        return self.query("SV:RF_PHASe:REFerence:MASTer?").strip()

    @staticmethod
    def _require_table_name(name: str) -> str:
        """Validate the ``TABLE<n>`` name PEAKSTABle requires (bench-verified).

        A free-form name (e.g. ``PK1``) is rejected by the scope with
        "Invalid string data", so the format is enforced client-side with a
        clear error instead.
        """
        if not re.fullmatch(r"TABLE\d+", name.strip().upper()):
            raise ValueError(
                f"peaks-table name must be of the form 'TABLE<n>' (n>=1); got {name!r}"
            )
        return name

    def cmd_add_peaks_table(self, name: str) -> list[str]:
        """Add a Spectrum-View peaks results table.

        ``name`` must be of the form ``TABLE<n>`` (the scope rejects other
        names with "Invalid string data").
        """
        self.write(f"PEAKSTABle:ADDNew {quote(self._require_table_name(name))}")
        return self.get_peaks_tables()

    def cmd_delete_peaks_table(self, name: str) -> list[str]:
        """Delete a Spectrum-View peaks results table.

        ``name`` must be of the form ``TABLE<n>``.
        """
        self.write(f"PEAKSTABle:DELete {quote(self._require_table_name(name))}")
        return self.get_peaks_tables()

    def get_peaks_tables(self) -> list[str]:
        """List the defined Spectrum-View peaks results tables."""
        listing = self.query("PEAKSTABle:LIST?").strip()
        if not listing or listing.upper() in ("NONE", '""'):
            return []
        return [t.strip().strip('"') for t in listing.split(",") if t.strip()]


class _RefLevels(Node):
    """Reference-level configuration for a measurement scope (Measurement.md).

    One instance wraps a single reference-level command scope, selected by the
    SCPI ``prefix`` it is built with:

    * global:        ``MEASUrement:REFLevels``
    * per-channel:   ``MEASUrement:CH<x>:REFLevels``
    * per-measurement: ``MEASUrement:MEAS<x>:REFLevels<y>``

    Rise/fall HIGH/MID/LOW levels exist in both PERCent and ABSolute flavors;
    ``METHod`` selects which one is used for the measurement.
    """

    _LEVEL_KEYS = ("rise_high", "rise_mid", "rise_low", "fall_high", "fall_mid", "fall_low")
    _LEVEL_SCPI = {
        "rise_high": "RISEHigh",
        "rise_mid": "RISEMid",
        "rise_low": "RISELow",
        "fall_high": "FALLHigh",
        "fall_mid": "FALLMid",
        "fall_low": "FALLLow",
    }

    def __init__(
        self, link: InstrumentLink, prefix: str, name: str, has_source_type: bool = False
    ) -> None:
        super().__init__(link, name=name)
        self._prefix = prefix
        self._has_source_type = has_source_type

    def set_method(self, method: RefLevelMethod | str) -> str:
        """Select PERCent or ABSolute reference levels.

        Enum arg: <scope>.RefLevelMethod.<TAB>
        ### Parameters
        1. method : str
            - ['PERCENT', 'ABSOLUTE']
        ### Returns
        1. method : str
            - actual method read back from the scope
        """
        self.write(f"{self._prefix}:METHod {token(method, RefLevelMethod, 'method')}")
        return self.query(f"{self._prefix}:METHod?").strip()

    def get_method(self) -> RefLevelMethod:
        """Query the active reference-level method (PERCENT/ABSOLUTE)."""
        return from_response(
            self.query(f"{self._prefix}:METHod?").strip(), RefLevelMethod, "method"
        )

    def set_percent_type(self, ref_type: RefLevelPercentType | str) -> str:
        """Set the percent reference-level preset.

        Enum arg: <scope>.RefLevelPercentType.<TAB>
        ### Parameters
        1. ref_type : str
            - ['10-90', '20-80', 'CUSTOM']
        ### Returns
        1. ref_type : str
            - actual preset read back from the scope
        """
        self.write(
            f"{self._prefix}:PERCent:TYPE {token(ref_type, RefLevelPercentType, 'ref_type')}"
        )
        return self.query(f"{self._prefix}:PERCent:TYPE?").strip()

    def set_absolute_type(self, ref_type: RefLevelAbsoluteType | str) -> str:
        """Set whether absolute rise and fall levels are SAME or UNIQUE.

        Enum arg: <scope>.RefLevelAbsoluteType.<TAB>
        ### Parameters
        1. ref_type : str
            - ['SAME', 'UNIQUE']
        ### Returns
        1. ref_type : str
            - actual type read back from the scope
        """
        self.write(
            f"{self._prefix}:ABSolute:TYPE {token(ref_type, RefLevelAbsoluteType, 'ref_type')}"
        )
        return self.query(f"{self._prefix}:ABSolute:TYPE?").strip()

    def set_basetop(self, method: RefLevelBaseTop | str) -> str:
        """Set the base/top detection method.

        Enum arg: <scope>.RefLevelBaseTop.<TAB>
        ### Parameters
        1. method : str
            - ['AUTO', 'MINMAX', 'MEAN HISTOGRAM', 'MODE HISTOGRAM', 'EYE HISTOGRAM']
        ### Returns
        1. method : str
            - actual base/top method read back from the scope
        """
        self.write(f"{self._prefix}:BASETop {token(method, RefLevelBaseTop, 'method')}")
        return self.query(f"{self._prefix}:BASETop?").strip()

    def get_basetop(self) -> RefLevelBaseTop:
        """Query the base/top detection method."""
        return from_response(
            self.query(f"{self._prefix}:BASETop?").strip(), RefLevelBaseTop, "basetop"
        )

    def _set_levels(self, kind: str, values: dict[str, float | None]) -> dict[str, float]:
        for key, value in values.items():
            if value is None:
                continue
            if key not in self._LEVEL_SCPI:
                raise ValueError(f"unknown reference level {key!r}; valid: {self._LEVEL_KEYS}")
            self.write(f"{self._prefix}:{kind}:{self._LEVEL_SCPI[key]} {value:.12g}")
        return self._get_levels(kind)

    def _get_levels(self, kind: str) -> dict[str, float]:
        return {
            key: self.query_float(f"{self._prefix}:{kind}:{scpi}?")
            for key, scpi in self._LEVEL_SCPI.items()
        }

    def set_percent_levels(
        self,
        rise_high: float | None = None,
        rise_mid: float | None = None,
        rise_low: float | None = None,
        fall_high: float | None = None,
        fall_mid: float | None = None,
        fall_low: float | None = None,
    ) -> dict[str, float]:
        """Set any of the PERCent rise/fall high/mid/low levels (0-100).

        Only the provided levels are written; the rest are left unchanged.
        ### Returns
        1. levels : dict[str, float]
            - all six percent levels read back after the write
        """
        return self._set_levels(
            "PERCent",
            {
                "rise_high": rise_high,
                "rise_mid": rise_mid,
                "rise_low": rise_low,
                "fall_high": fall_high,
                "fall_mid": fall_mid,
                "fall_low": fall_low,
            },
        )

    def get_percent_levels(self) -> dict[str, float]:
        """Read all six PERCent rise/fall high/mid/low levels."""
        return self._get_levels("PERCent")

    def set_absolute_levels(
        self,
        rise_high: float | None = None,
        rise_mid: float | None = None,
        rise_low: float | None = None,
        fall_high: float | None = None,
        fall_mid: float | None = None,
        fall_low: float | None = None,
    ) -> dict[str, float]:
        """Set any of the ABSolute rise/fall high/mid/low levels (volts).

        Only the provided levels are written; the rest are left unchanged.
        ### Returns
        1. levels : dict[str, float]
            - all six absolute levels (volts) read back after the write
        """
        return self._set_levels(
            "ABSolute",
            {
                "rise_high": rise_high,
                "rise_mid": rise_mid,
                "rise_low": rise_low,
                "fall_high": fall_high,
                "fall_mid": fall_mid,
                "fall_low": fall_low,
            },
        )

    def get_absolute_levels(self) -> dict[str, float]:
        """Read all six ABSolute rise/fall high/mid/low levels (volts)."""
        return self._get_levels("ABSolute")

    def set_hysteresis(self, value: float, method: str = "PERCENT") -> float:
        """Set the reference-level hysteresis for PERCent or ABSolute levels.
        ### Parameters
        1. value : float
            - hysteresis (percent or volts, matching ``method``)
        2. method : str (default 'PERCENT')
            - ['PERCENT', 'ABSOLUTE']
        ### Returns
        1. value : float
            - actual hysteresis read back from the scope
        """
        kind = _REFLEVEL_METHOD[method.strip().upper()]
        self.write(f"{self._prefix}:{kind}:HYSTeresis {value:.12g}")
        return self.query_float(f"{self._prefix}:{kind}:HYSTeresis?")

    def get_hysteresis(self, method: str = "PERCENT") -> float:
        """Query the reference-level hysteresis for PERCent or ABSolute levels."""
        kind = _REFLEVEL_METHOD[method.strip().upper()]
        return self.query_float(f"{self._prefix}:{kind}:HYSTeresis?")

    def set_source_type(self, source_type: RefLevelType | str) -> str:
        """Set whether reference levels are GLOBAL or PER SOURCE (global node only).

        Enum arg: <scope>.RefLevelType.<TAB>
        ### Parameters
        1. source_type : str
            - ['GLOBAL', 'PER SOURCE']
        ### Returns
        1. source_type : str
            - actual type read back from the scope
        """
        if not self._has_source_type:
            raise ValueError("source type is only settable on the global reference-level node")
        self.write(f"{self._prefix}:TYPE {token(source_type, RefLevelType, 'source_type')}")
        return self.query(f"{self._prefix}:TYPE?").strip()

    def get_source_type(self) -> RefLevelType:
        """Query whether reference levels are GLOBAL or PER SOURCE (global node only)."""
        if not self._has_source_type:
            raise ValueError("source type is only queryable on the global reference-level node")
        return from_response(
            self.query(f"{self._prefix}:TYPE?").strip(), RefLevelType, "source_type"
        )


class _Gating(Node):
    """Measurement gating configuration (Measurement.md).

    Wraps one gating command scope selected by ``prefix``:

    * global:          ``MEASUrement:GATing``
    * per-measurement: ``MEASUrement:MEAS<x>:GATing``

    Gating restricts a measurement to a region of the record (a time window,
    the screen, cursors, a logic/search qualifier).
    """

    def __init__(
        self, link: InstrumentLink, prefix: str, name: str, per_meas: bool = False
    ) -> None:
        super().__init__(link, name=name)
        self._prefix = prefix
        self._per_meas = per_meas

    def set_mode(self, mode: GatingMode | str) -> str:
        """Set the gating mode.

        Enum arg: <scope>.GatingMode.<TAB>
        ### Parameters
        1. mode : str
            - ['NONE', 'SCREEN', 'CURSOR', 'LOGIC', 'SEARCH', 'TIME']
        ### Returns
        1. mode : str
            - actual gating mode read back from the scope
        """
        self.write(f"{self._prefix} {token(mode, GatingMode, 'mode')}")
        return self.query(f"{self._prefix}?").strip()

    def get_mode(self) -> GatingMode:
        """Query the gating mode."""
        return from_response(self.query(f"{self._prefix}?").strip(), GatingMode, "mode")

    def set_active(self, active: GatingActive | str) -> str:
        """Set the active gating polarity (which logic level gates measurements).

        Enum arg: <scope>.GatingActive.<TAB>
        ### Parameters
        1. active : str
            - ['HIGH', 'LOW']
        ### Returns
        1. active : str
            - actual active polarity read back from the scope
        """
        self.write(f"{self._prefix}:ACTive {token(active, GatingActive, 'active')}")
        return self.query(f"{self._prefix}:ACTive?").strip()

    def set_time_window(self, start: float, end: float) -> tuple[float, float]:
        """Set the TIME-gating window start and end times (seconds).
        ### Parameters
        1. start : float
            - window start time in seconds
        2. end : float
            - window end time in seconds
        ### Returns
        1. (start, end) : tuple[float, float]
            - actual window start/end read back from the scope
        """
        self.write(f"{self._prefix}:STARTtime {start:.12g}")
        self.write(f"{self._prefix}:ENDtime {end:.12g}")
        return (
            self.query_float(f"{self._prefix}:STARTtime?"),
            self.query_float(f"{self._prefix}:ENDtime?"),
        )

    def get_time_window(self) -> tuple[float, float]:
        """Query the TIME-gating window (start, end) times in seconds."""
        return (
            self.query_float(f"{self._prefix}:STARTtime?"),
            self.query_float(f"{self._prefix}:ENDtime?"),
        )

    def set_hysteresis(self, value: float) -> float:
        """Set the gating hysteresis; returns the read-back value."""
        self.write(f"{self._prefix}:HYSTeresis {value:.12g}")
        return self.query_float(f"{self._prefix}:HYSTeresis?")

    def set_midref(self, value: float) -> float:
        """Set the gating mid-reference level; returns the read-back value."""
        self.write(f"{self._prefix}:MIDRef {value:.12g}")
        return self.query_float(f"{self._prefix}:MIDRef?")

    def set_global(self, state: bool) -> bool:
        """Tie this measurement's gating to the global gating (per-measurement only).
        ### Parameters
        1. state : bool
            - True to follow global gating, False for independent gating
        ### Returns
        1. state : bool
            - actual global-follow state read back from the scope
        """
        if not self._per_meas:
            raise ValueError("GLOBal is only valid on a per-measurement gating node")
        self.write(f"{self._prefix}:GLOBal {'ON' if state else 'OFF'}")
        return self.query_bool(f"{self._prefix}:GLOBal?")

    def get_global(self) -> bool:
        """Query whether this measurement follows the global gating (per-meas only)."""
        if not self._per_meas:
            raise ValueError("GLOBal is only valid on a per-measurement gating node")
        return self.query_bool(f"{self._prefix}:GLOBal?")


class _Measure(Node):
    """Measurement system (manual command group: Measurement).

    A measurement occupies a numbered slot. :meth:`custom` is the
    low-level primitive: it ensures the slot exists and is the right type,
    points it at a source, waits for the current-acquisition population to
    become non-zero, and reads the badge value. Named wrappers are organized
    into browsable sub-nodes for autocomplete:

    * :attr:`vert` - amplitude-domain measurements (single source)
    * :attr:`time` - timing measurements (single source)
    * :attr:`dual` - two-source measurements (delay, phase, skew, ...)

    The most common wrappers are also kept directly on this node for continuity
    with earlier releases. Option-gated types (jitter, eye, DDR, DPM, IMDA, WBG,
    power) are unwrapped and reached through :meth:`custom`.
    """

    def __init__(self, link: InstrumentLink, name: str = "MEASUREMENT") -> None:
        super().__init__(link, name=name)
        self.vert = _MeasVert(self)
        self.time = _MeasTime(self)
        self.dual = _MeasDual(self)
        self.ref_levels = _RefLevels(
            link, "MEASUrement:REFLevels", name="MEAS:REFLEVELS", has_source_type=True
        )
        self.gating = _Gating(link, "MEASUrement:GATing", name="MEAS:GATING")

    def ref_levels_ch(self, channel: int) -> "_RefLevels":
        """Per-channel reference levels."""
        return _RefLevels(
            self.link, f"MEASUrement:CH{int(channel)}:REFLevels", name=f"MEAS:CH{channel}:REFLEVELS"
        )

    def ref_levels_meas(self, num: int, index: int = 1) -> "_RefLevels":
        """Per-measurement reference levels."""
        return _RefLevels(
            self.link,
            f"MEASUrement:MEAS{int(num)}:REFLevels{int(index)}",
            name=f"MEAS:MEAS{num}:REFLEVELS{index}",
        )

    def gating_meas(self, num: int) -> "_Gating":
        """Per-measurement gating."""
        return _Gating(
            self.link,
            f"MEASUrement:MEAS{int(num)}:GATing",
            name=f"MEAS:MEAS{num}:GATING",
            per_meas=True,
        )

    def cmd_delete_all(self) -> None:
        """Delete every measurement slot."""
        self.write("MEASUrement:DELETEALL")

    def set_label(self, num: int, label: str) -> str:
        """Set measurement slot ``num``'s label; returns the read-back label.

        The label is sent as a quoted SCPI argument; an embedded double quote is
        rejected with :class:`ValueError` so it cannot break out of its quoting.
        """
        if '"' in label:
            raise ValueError(f"label must not contain a double-quote character; got {label!r}")
        self.write(f'MEASUrement:MEAS{int(num)}:LABel "{label}"')
        return self.query(f"MEASUrement:MEAS{int(num)}:LABel?").strip().strip('"')

    def get_label(self, num: int) -> str:
        """Query measurement slot ``num``'s label string."""
        return self.query(f"MEASUrement:MEAS{int(num)}:LABel?").strip().strip('"')

    def get_status(self, num: int) -> str:
        """Query measurement slot ``num``'s status."""
        return self.query(f"MEASUrement:MEAS{int(num)}:STATUS?").strip().strip('"')

    def get_xunit(self, num: int) -> str:
        """Query measurement slot ``num``'s horizontal unit string."""
        return self.query(f"MEASUrement:MEAS{int(num)}:XUNIT?").strip().strip('"')

    def get_yunit(self, num: int) -> str:
        """Query measurement slot ``num``'s vertical unit string."""
        return self.query(f"MEASUrement:MEAS{int(num)}:YUNIT?").strip().strip('"')

    def get_curr_max(self, num: int) -> float:
        """Current-acquisition maximum result for slot ``num``."""
        return self.query_float(f"MEASUrement:MEAS{int(num)}:RESUlts:CURRentacq:MAXimum?")

    def get_curr_min(self, num: int) -> float:
        """Current-acquisition minimum result for slot ``num``."""
        return self.query_float(f"MEASUrement:MEAS{int(num)}:RESUlts:CURRentacq:MINimum?")

    def get_curr_pk2pk(self, num: int) -> float:
        """Current-acquisition peak-to-peak result for slot ``num``."""
        return self.query_float(f"MEASUrement:MEAS{int(num)}:RESUlts:CURRentacq:PK2PK?")

    def get_curr_std(self, num: int) -> float:
        """Current-acquisition standard deviation for slot ``num``."""
        return self.query_float(f"MEASUrement:MEAS{int(num)}:RESUlts:CURRentacq:STDDev?")

    def get_allacqs_pk2pk(self, num: int) -> float:
        """All-acquisitions peak-to-peak result for slot ``num``."""
        return self.query_float(f"MEASUrement:MEAS{int(num)}:RESUlts:ALLAcqs:PK2PK?")

    def get_meas_active(self, num: int) -> bool:
        """Return True if measurement slot ``num`` currently exists."""
        listing = self.query("MEASUrement:LIST?").strip()
        if not listing or listing.upper() in ("NONE", '""'):
            return False
        for piece in listing.replace('"', "").split(","):
            piece = piece.strip()
            if not piece:
                continue
            digits = piece.upper().replace("MEAS", "")
            if digits.isdigit() and int(digits) == num:
                return True
        return False

    def cmd_delete(self, num: int | str = "ALL") -> None:
        """Delete measurement ``num`` (int) or every measurement (``'ALL'``)."""
        if num == "ALL":
            listing = self.query("MEASUrement:LIST?").strip()
            for token in listing.replace('"', "").split(","):
                token = token.strip()
                digits = token.upper().replace("MEAS", "")
                if digits.isdigit():
                    self.write(f'MEASUrement:DELete "MEAS{int(digits)}"')
        else:
            self.write(f'MEASUrement:DELete "MEAS{int(num)}"')

    def set_statistics(self, num: int, state: bool = True) -> None:
        """Show or hide the statistics readout for measurement ``num``."""
        self.write(f"MEASUrement:MEAS{num}:DISPlay:ENABle {'ON' if state else 'OFF'}")

    def wait_for_population(self, num: int = 1, timeout: float = 3.0) -> int:
        """Block until slot ``num`` reports a non-zero current-acq population."""
        deadline = time.monotonic() + timeout
        while True:
            popu = self.query_int(f"MEASUrement:MEAS{num}:RESUlts:CURRentacq:POPUlation?")
            if popu:
                return popu
            if time.monotonic() > deadline:
                raise InstrumentTimeoutError(f"MEAS{num} population stayed 0 after {timeout:g}s")
            time.sleep(0.05)

    def _wait_fresh_population(self, num: int, timeout: float) -> int:
        """Wait for a populated result computed AFTER the slot was configured.

        A freshly added slot defaults to PERIOD and immediately has a non-zero
        population, so a bare population poll can return a value from the old
        type. Requiring the acquisition counter to advance ensures the badge
        reflects the just-set type before it is read.
        """
        deadline = time.monotonic() + timeout
        try:
            baseline = self.link.query_int("ACQ:NUMAC?")
        except InstrumentError:
            baseline = None
        while True:
            popu = self.query_int(f"MEASUrement:MEAS{num}:RESUlts:CURRentacq:POPUlation?")
            fresh = baseline is None or self.link.query_int("ACQ:NUMAC?") > baseline
            if popu and fresh:
                return popu
            if time.monotonic() > deadline:
                raise InstrumentTimeoutError(
                    f"MEAS{num} produced no fresh populated result after {timeout:g}s"
                )
            time.sleep(0.05)

    def get_value(self, num: int) -> float:
        """Current-acquisition badge value (this acquisition only)."""
        return self.query_float(f"MEASUrement:MEAS{num}:RESUlts:CURRentacq:MEAN?")

    def get_mean(self, num: int) -> float:
        """Mean accumulated over all acquisitions since the last stats reset."""
        return self.query_float(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:MEAN?")

    def get_min(self, num: int) -> float:
        """Minimum over all acquisitions since the last stats reset."""
        return self.query_float(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:MINimum?")

    def get_max(self, num: int) -> float:
        """Maximum over all acquisitions since the last stats reset."""
        return self.query_float(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:MAXimum?")

    def get_std(self, num: int) -> float:
        """Standard deviation over all acquisitions since the last stats reset."""
        return self.query_float(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:STDDev?")

    def get_statistics(self, num: int = 1) -> MeasStats:
        """Read current value plus accumulated statistics for slot ``num``.

        ### Parameters
        1. num : int (default 1)
            - measurement slot number
        ### Returns
        1. stats : MeasStats
            - current-acq value plus all-acqs mean/min/max/stddev/population
        """
        return MeasStats(
            value=self.query_float(f"MEASUrement:MEAS{num}:RESUlts:CURRentacq:MEAN?"),
            mean=self.query_float(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:MEAN?"),
            minimum=self.query_float(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:MINimum?"),
            maximum=self.query_float(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:MAXimum?"),
            stddev=self.query_float(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:STDDev?"),
            population=self.query_int(f"MEASUrement:MEAS{num}:RESUlts:ALLAcqs:POPUlation?"),
        )

    def _configure(self, num: int, token: str, sources: list[str]) -> None:
        """Ensure slot ``num`` is the requested type and pointed at ``sources``."""
        token = token.strip().upper()
        if self.get_meas_active(num):
            current = self.query(f"MEASUrement:MEAS{num}:TYPe?").strip().strip('"').upper()
            if current != token:
                self.write(f'MEASUrement:DELete "MEAS{num}"')
                self.write(f'MEASUrement:ADDNew "MEAS{num}"')
        else:
            self.write(f'MEASUrement:ADDNew "MEAS{num}"')

        self.write(f"MEASUrement:MEAS{num}:TYPe {token}")
        if len(sources) == 1:
            self.write(f"MEASUrement:MEAS{num}:SOUrce {sources[0]}")
        else:
            for index, src in enumerate(sources, start=1):
                self.write(f"MEASUrement:MEAS{num}:SOURCE{index} {src}")
        self.query("*OPC?")

    def custom(
        self,
        type_token: str,
        source: str,
        num: int = 1,
        persist: bool = True,
        timeout: float = 3.0,
    ) -> float:
        """Configure, populate, and read one measurement slot (escape hatch).

        This is the general primitive behind every named wrapper and the way to
        reach any measurement type token the driver does not wrap explicitly
        (e.g. option-gated jitter/eye/power types).

        ### Parameters
        1. type_token : str
            - measurement type token (e.g. 'FREQUENCY'); see Measurement.md
        2. source : str
            - CH<x>, MATH<x>, REF<x>
        3. num : int (default 1)
            - measurement slot number
        4. persist : bool (default True)
            - keep the slot after reading; delete it when False
        5. timeout : float (default 3.0)
            - max seconds to wait for the population to become non-zero
        ### Returns
        1. value : float
            - the current-acquisition badge value
        """
        self._configure(num, type_token, [source])
        self._wait_fresh_population(num=num, timeout=timeout)
        value = self.query_float(f"MEASUrement:MEAS{num}:RESUlts:CURRentacq:MEAN?")
        if not persist:
            self.write(f'MEASUrement:DELete "MEAS{num}"')
        return value

    def custom_dual(
        self,
        type_token: str,
        source1: str,
        source2: str,
        num: int = 1,
        persist: bool = True,
        timeout: float = 3.0,
    ) -> float:
        """Configure and read a two-source measurement slot (escape hatch).

        ### Parameters
        1. type_token : str
            - two-source type token (e.g. 'DELAY'); see Measurement.md
        2. source1 : str
            - first (reference) source: CH<x>, MATH<x>, REF<x>
        3. source2 : str
            - second source
        4. num : int (default 1)
        5. persist : bool (default True)
        6. timeout : float (default 3.0)
        ### Returns
        1. value : float
            - the current-acquisition badge value
        """
        self._configure(num, type_token, [source1, source2])
        self._wait_fresh_population(num=num, timeout=timeout)
        value = self.query_float(f"MEASUrement:MEAS{num}:RESUlts:CURRentacq:MEAN?")
        if not persist:
            self.write(f'MEASUrement:DELete "MEAS{num}"')
        return value

    def measure_stats(
        self,
        type_token: str,
        source: str,
        source2: str | None = None,
        num: int = 1,
        persist: bool = True,
        timeout: float = 3.0,
    ) -> MeasStats:
        """Configure a measurement and return its full statistics.

        Accepts one or two sources (pass ``source2`` for a two-source type).

        ### Returns
        1. stats : MeasStats
            - current value plus all-acqs mean/min/max/stddev/population
        """
        sources = [source] if source2 is None else [source, source2]
        self._configure(num, type_token, sources)
        self._wait_fresh_population(num=num, timeout=timeout)
        result = self.get_statistics(num)
        if not persist:
            self.write(f'MEASUrement:DELete "MEAS{num}"')
        return result

    def generic_scpi(
        self,
        scpi: str,
        source: str,
        num: int = 1,
        persist: bool = True,
        timeout: float = 3.0,
    ) -> float:
        """Deprecated alias for :meth:`custom` (kept for continuity)."""
        return self.custom(scpi, source, num=num, persist=persist, timeout=timeout)

    def frequency(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Frequency (reciprocal of period) of ``source``."""
        return self.custom("FREQUENCY", source, num=num, persist=persist, timeout=timeout)

    def amplitude(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Amplitude (High - Low) of ``source``."""
        return self.custom("AMPLITUDE", source, num=num, persist=persist, timeout=timeout)

    def maximum(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Maximum data point of ``source``."""
        return self.custom("MAXIMUM", source, num=num, persist=persist, timeout=timeout)

    def minimum(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Minimum data point of ``source``."""
        return self.custom("MINIMUM", source, num=num, persist=persist, timeout=timeout)

    def peak_to_peak(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Peak-to-peak (Maximum - Minimum) of ``source``."""
        return self.custom("PK2PK", source, num=num, persist=persist, timeout=timeout)

    def period(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Period of ``source``."""
        return self.custom("PERIOD", source, num=num, persist=persist, timeout=timeout)

    def pulse_width_pos(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Positive pulse width of ``source``."""
        return self.custom("PWIDTH", source, num=num, persist=persist, timeout=timeout)

    def pulse_width_neg(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Negative pulse width of ``source``."""
        return self.custom("NWIDTH", source, num=num, persist=persist, timeout=timeout)

    def mean(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """Arithmetic mean of ``source``."""
        return self.custom("MEAN", source, num=num, persist=persist, timeout=timeout)

    def rms(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """True RMS of ``source``."""
        return self.custom("RMS", source, num=num, persist=persist, timeout=timeout)

    def ac_rms(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """AC RMS (RMS about the mean) of ``source``."""
        return self.custom("ACRMS", source, num=num, persist=persist, timeout=timeout)


class _MeasSubNode(Node):
    """Base for measurement sub-nodes; delegates to the owning :class:`_Measure`."""

    def __init__(self, owner: "_Measure", name: str) -> None:
        super().__init__(owner.link, name=name)
        self._m = owner


class _MeasVert(_MeasSubNode):
    """Amplitude-domain measurements (single source)."""

    def __init__(self, owner: "_Measure") -> None:
        super().__init__(owner, name="MEAS:VERT")

    def amplitude(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Amplitude (High - Low) of ``source``."""
        return self._m.custom("AMPLITUDE", source, num=num, persist=persist, timeout=timeout)

    def maximum(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Maximum data point of ``source``."""
        return self._m.custom("MAXIMUM", source, num=num, persist=persist, timeout=timeout)

    def minimum(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Minimum data point of ``source``."""
        return self._m.custom("MINIMUM", source, num=num, persist=persist, timeout=timeout)

    def peak_to_peak(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Peak-to-peak (Maximum - Minimum) of ``source``."""
        return self._m.custom("PK2PK", source, num=num, persist=persist, timeout=timeout)

    def top(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """Top value (most common level above the midpoint) of ``source``."""
        return self._m.custom("TOP", source, num=num, persist=persist, timeout=timeout)

    def base(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """Base value (most common level below the midpoint) of ``source``."""
        return self._m.custom("BASE", source, num=num, persist=persist, timeout=timeout)

    def high(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """High (logic 1) reference level of ``source``."""
        return self._m.custom("HIGH", source, num=num, persist=persist, timeout=timeout)

    def low(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """Low (logic 0) reference level of ``source``."""
        return self._m.custom("LOW", source, num=num, persist=persist, timeout=timeout)

    def mean(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """Arithmetic mean of ``source``."""
        return self._m.custom("MEAN", source, num=num, persist=persist, timeout=timeout)

    def rms(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """True RMS of ``source``."""
        return self._m.custom("RMS", source, num=num, persist=persist, timeout=timeout)

    def ac_rms(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """AC RMS (RMS about the mean) of ``source``."""
        return self._m.custom("ACRMS", source, num=num, persist=persist, timeout=timeout)

    def area(self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0) -> float:
        """Area under the curve (integral of the data points) of ``source``."""
        return self._m.custom("AREA", source, num=num, persist=persist, timeout=timeout)

    def overshoot_pos(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Positive overshoot ((Maximum - Top) / Amplitude) of ``source``."""
        return self._m.custom("POVERSHOOT", source, num=num, persist=persist, timeout=timeout)

    def overshoot_neg(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Negative overshoot ((Base - Minimum) / Amplitude) of ``source``."""
        return self._m.custom("NOVERSHOOT", source, num=num, persist=persist, timeout=timeout)


class _MeasTime(_MeasSubNode):
    """Timing-domain measurements (single source)."""

    def __init__(self, owner: "_Measure") -> None:
        super().__init__(owner, name="MEAS:TIME")

    def frequency(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Frequency (reciprocal of period) of ``source``."""
        return self._m.custom("FREQUENCY", source, num=num, persist=persist, timeout=timeout)

    def period(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Period of ``source``."""
        return self._m.custom("PERIOD", source, num=num, persist=persist, timeout=timeout)

    def rise_time(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Rise time (Base ref to Top ref on a rising edge) of ``source``."""
        return self._m.custom("RISETIME", source, num=num, persist=persist, timeout=timeout)

    def fall_time(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Fall time (Top ref to Base ref on a falling edge) of ``source``."""
        return self._m.custom("FALLTIME", source, num=num, persist=persist, timeout=timeout)

    def rise_slew_rate(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Rising slew rate (dV/dt on a rising edge) of ``source``."""
        return self._m.custom("RISESLEWRATE", source, num=num, persist=persist, timeout=timeout)

    def fall_slew_rate(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Falling slew rate (dV/dt on a falling edge) of ``source``."""
        return self._m.custom("FALLSLEWRATE", source, num=num, persist=persist, timeout=timeout)

    def width_pos(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Positive pulse width of ``source``."""
        return self._m.custom("PWIDTH", source, num=num, persist=persist, timeout=timeout)

    def width_neg(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Negative pulse width of ``source``."""
        return self._m.custom("NWIDTH", source, num=num, persist=persist, timeout=timeout)

    def duty_pos(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Positive duty cycle (Positive Width / Period) of ``source``."""
        return self._m.custom("PDUTY", source, num=num, persist=persist, timeout=timeout)

    def duty_neg(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Negative duty cycle (Negative Width / Period) of ``source``."""
        return self._m.custom("NDUTY", source, num=num, persist=persist, timeout=timeout)

    def burst_width(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Burst width (duration of adjacent mid-level crossings) of ``source``.

        Bench-verified: does NOT populate on a continuous waveform (raises
        InstrumentTimeoutError) - the signal must contain distinct bursts
        separated by idle time.
        """
        return self._m.custom("BURSTWIDTH", source, num=num, persist=persist, timeout=timeout)

    def high_time(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Time the signal remains above the Top reference level."""
        return self._m.custom("HIGHTIME", source, num=num, persist=persist, timeout=timeout)

    def low_time(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Time the signal remains below the Base reference level."""
        return self._m.custom("LOWTIME", source, num=num, persist=persist, timeout=timeout)

    def n_period(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Duration of N periods (time to complete N cycles) of ``source``."""
        return self._m.custom("NPERIOD", source, num=num, persist=persist, timeout=timeout)

    def data_rate(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Data rate (reciprocal of unit interval) of ``source``."""
        return self._m.custom("DATARATE", source, num=num, persist=persist, timeout=timeout)

    def time_outside_level(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Time the signal spends outside the Top/Base reference levels."""
        return self._m.custom("TIMEOUTSIDELEVEL", source, num=num, persist=persist, timeout=timeout)

    def unit_interval(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Unit interval (time difference between two successive bits) of ``source``."""
        return self._m.custom("UNITINTERVAL", source, num=num, persist=persist, timeout=timeout)

    def time_to_max(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Time from the trigger point to the maximum data point of ``source``."""
        return self._m.custom("TIMETOMAX", source, num=num, persist=persist, timeout=timeout)

    def time_to_min(
        self, source: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Time from the trigger point to the minimum data point of ``source``."""
        return self._m.custom("TIMETOMIN", source, num=num, persist=persist, timeout=timeout)


class _MeasDual(_MeasSubNode):
    """Two-source measurements (SOURCE1 = reference, SOURCE2 = second source)."""

    def __init__(self, owner: "_Measure") -> None:
        super().__init__(owner, name="MEAS:DUAL")

    def delay(
        self, source1: str, source2: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Delay from a mid-level crossing on ``source1`` to one on ``source2``."""
        return self._m.custom_dual(
            "DELAY", source1, source2, num=num, persist=persist, timeout=timeout
        )

    def phase(
        self, source1: str, source2: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Phase (skew relative to the period of ``source1``) between the sources."""
        return self._m.custom_dual(
            "PHASE", source1, source2, num=num, persist=persist, timeout=timeout
        )

    def skew(
        self, source1: str, source2: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Skew (time between mid-level crossings on the two sources)."""
        return self._m.custom_dual(
            "SKEW", source1, source2, num=num, persist=persist, timeout=timeout
        )

    def setup(
        self, source1: str, source2: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Setup time (data ``source1`` to clock ``source2`` crossing)."""
        return self._m.custom_dual(
            "SETUP", source1, source2, num=num, persist=persist, timeout=timeout
        )

    def hold(
        self, source1: str, source2: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """Hold time (clock ``source1`` to data ``source2`` crossing)."""
        return self._m.custom_dual(
            "HOLD", source1, source2, num=num, persist=persist, timeout=timeout
        )

    def common_mode(
        self, source1: str, source2: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """DC common mode (mean common-mode voltage of two sources).

        Requires the DJA option and is not available on 4 Series MSO.
        """
        return self._m.custom_dual(
            "COMMONMODE", source1, source2, num=num, persist=persist, timeout=timeout
        )

    def ac_common_mode(
        self, source1: str, source2: str, num: int = 1, persist: bool = True, timeout: float = 3.0
    ) -> float:
        """AC common mode (peak-to-peak common-mode voltage of two sources).

        Requires the DJA option and is not available on 4 Series MSO.
        """
        return self._m.custom_dual(
            "ACCOMMONMODE", source1, source2, num=num, persist=persist, timeout=timeout
        )


class _File(Node):
    """File system, screenshots, and waveform/spectrum file transfer.

    (Manual command groups: File System, Save and Recall.) Transfers round-trip
    through the scope disk: SAVe to a temp file, read it back with FILESystem,
    then delete it. Distinct from the ``data`` subsystem, which transfers curve
    data straight from memory. ``model`` is populated from the driver's parsed
    IDN on connect (this class does not re-query ``*IDN?``).
    """

    _MAX_TRANSFER_TIMEOUT_MS = 45000
    _MAX_UPLOAD_BYTES = 2_000_000_000
    # Dedicated scratch dir on the scope: keeps the LDIR size-lookup listing
    # small/deterministic instead of scanning a busy C:/ root (see ADR-005).
    _SCRATCH_DIR = "C:/pyvisa_tmp"

    def __init__(self, link: InstrumentLink, name: str = "FILE") -> None:
        super().__init__(link, name=name)
        self.model: str | None = None

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _ensure_scratch_dir(self) -> None:
        """Create the scope scratch dir on demand (see ADR-005)."""
        self.write('FILESystem:CWD "C:/"')
        _, _, leaf = self._SCRATCH_DIR.rpartition("/")
        if leaf not in self._list_dir_names():
            self.write(f"FILESystem:MKDir {quote(self._SCRATCH_DIR)}")

    def _scratch_path(self, filename: str) -> str:
        """Return a full path inside the scope scratch dir (dir created if absent)."""
        self._ensure_scratch_dir()
        return f"{self._SCRATCH_DIR}/{filename}"

    def _temp_remote(self, ext: str) -> str:
        """Ensure the scope scratch dir exists and return a temp file path in it."""
        return self._scratch_path(f"Temp.{ext}")

    def cmd_save_setup(self, scope_path: str) -> str:
        """Save the current instrument setup to a scope-side ``.set`` file.

        ### Parameters
        1. scope_path : str
            - full scope path for the setup file (e.g. ``C:/pyvisa_tmp/x.set``)
        ### Returns
        1. str
            - the scope path written to
        """
        self.write(f"SAVe:SETUp {quote(scope_path)}")
        self.query("*OPC?")
        return scope_path

    def cmd_recall_setup(self, scope_path: str) -> None:
        """Recall an instrument setup from a scope-side ``.set`` file.

        ### Parameters
        1. scope_path : str
            - full scope path of a setup file previously saved with cmd_save_setup
        """
        with self.link.temporary_timeout(self._MAX_TRANSFER_TIMEOUT_MS):
            self.write(f"RECAll:SETUp {quote(scope_path)}")
            self.query("*OPC?")

    def cmd_recall_setup_factory(self) -> None:
        """Reset the scope to its factory default setup."""
        with self.link.temporary_timeout(self._MAX_TRANSFER_TIMEOUT_MS):
            self.write("RECAll:SETUp FACtory")
            self.query("*OPC?")

    def cmd_save_session(self, scope_path: str) -> str:
        """Save a full session (setup + reference waveforms) to a ``.tss`` file.

        ### Parameters
        1. scope_path : str
            - full scope path for the session file
        ### Returns
        1. str
            - the scope path written to
        """
        self.write(f"SAVe:SESsion {quote(scope_path)}")
        self.query("*OPC?")
        return scope_path

    def cmd_recall_session(self, scope_path: str) -> None:
        """Recall a full session (setup + reference waveforms) from a ``.tss`` file.

        ### Parameters
        1. scope_path : str
            - full scope path of a session file previously saved
        """
        with self.link.temporary_timeout(self._MAX_TRANSFER_TIMEOUT_MS):
            self.write(f"RECAll:SESsion {quote(scope_path)}")
            self.query("*OPC?")

    def cmd_delete(self, scope_path: str) -> None:
        """Delete one file, or one empty directory, on the instrument."""
        self.write(f"FILESystem:DELEte {quote(scope_path)}")

    def cmd_copy(self, source_path: str, dest_path: str) -> None:
        """Copy one instrument file to a new path. Wild cards are not allowed."""
        self.write(f"FILESystem:COPy {quote(source_path)},{quote(dest_path)}")

    def cmd_rename(self, old_path: str, new_path: str) -> None:
        """Rename or move one instrument file or folder."""
        self.write(f"FILESystem:REName {quote(old_path)},{quote(new_path)}")

    def cmd_rmdir(self, dir_path: str) -> None:
        """Delete an empty instrument directory."""
        self.write(f"FILESystem:RMDir {quote(dir_path)}")

    @staticmethod
    def _split_remote(remote_path: str) -> tuple[str, str]:
        directory, _, filename = remote_path.rpartition("/")
        return (directory + "/" if directory else ""), filename

    def _local_path(self, filename: str, folder_path: str | None) -> str:
        """Normalize and validate a local destination path before opening it."""
        filename = os.path.basename(filename)
        if folder_path is not None:
            folder = os.path.abspath(os.path.normpath(folder_path))
            if not os.path.isdir(folder):
                raise ValueError(f"destination folder does not exist: {folder_path!r}")
            return os.path.join(folder, filename)
        return os.path.abspath(filename)

    def _list_dir_names(self) -> list[str]:
        listing = self.query("FILESystem:DIR?").strip()
        return [s.strip().strip('"') for s in listing.split(",") if s.strip()]

    def _get_file_size(self, directory: str, filename: str) -> int | None:
        if directory:
            self.write(f'FILESystem:CWD "{directory}"')
        pattern = re.compile(rf'"?{re.escape(filename)};FILE;(\d+);')
        # A file can lag its *OPC? in the directory listing, so retry briefly.
        for attempt in range(10):
            match = pattern.search(self.query("FILESystem:LDIR?"))
            if match:
                return int(match.group(1))
            if attempt < 9:
                time.sleep(0.2)
        return None

    def _download(self, remote_path: str, timeout_ms: int | None = None) -> bytes:
        """Read a scope file back over the interface, then delete it."""
        timeout_ms = timeout_ms or self._MAX_TRANSFER_TIMEOUT_MS
        directory, filename = self._split_remote(remote_path)
        with self.link.temporary_timeout(timeout_ms):
            size = self._get_file_size(directory, filename)
            if size is None:
                raise InstrumentQueryError(
                    f"could not determine size of scope file {remote_path!r} from LDIR"
                )
            self.write(f'FILESystem:READFile "{remote_path}"')
            data = self.read_bytes(size)
            self._drain_terminator()
        self.write(f'FILESystem:DELEte "{remote_path}"')
        return data

    def _drain_terminator(self) -> None:
        """Consume the trailing message terminator left after a binary read.

        FILESystem:READFile emits the file bytes followed by the interface
        message terminator. Reading exactly ``size`` bytes leaves that
        terminator in the input buffer, which would corrupt the next query, so
        drain it (a short timeout guards the case where none is pending).
        """
        with self.link.temporary_timeout(500):
            try:
                self.link.read_bytes(1)
            except Exception:  # noqa: BLE001 - nothing pending is a valid outcome
                pass

    def get_screenshot(self, title: str | None = None, folder_path: str | None = None) -> str:
        """Save a screenshot PNG locally; returns the local file path."""
        title = f"_{title}" if title else ""
        remote = self._temp_remote("png")
        self.write(f'SAVe:IMAGe "{remote}"')
        self.query("*OPC?")
        data = self._download(remote)
        local = self._local_path(f"{self.model}_{self._timestamp()}{title}.png", folder_path)
        with open(local, "wb") as fh:
            fh.write(data)
        return local

    def get_waveform(
        self,
        channel: str,
        title: str,
        folder_path: str | None = None,
        format: str = "csv",
    ) -> str:
        """Save a channel waveform to a local .csv/.wfm/.mat file; returns path."""
        fmt = format.lower()
        remote = self._temp_remote(fmt)
        with self.link.temporary_timeout(self._MAX_TRANSFER_TIMEOUT_MS):
            self.write(f'SAVe:WAVEform {channel}, "{remote}"')
            self.query("*OPC?")
            data = self._download(remote)
        local = self._local_path(
            f"{title}_{channel}_{self.model}_{self._timestamp()}.{fmt}", folder_path
        )
        with open(local, "wb") as fh:
            fh.write(data)
        return local

    def get_waveform_data(self, channel: str) -> bytes:
        """Return a channel waveform as raw ``.wfm`` bytes (no local file)."""
        remote = self._temp_remote("wfm")
        with self.link.temporary_timeout(self._MAX_TRANSFER_TIMEOUT_MS):
            self.write(f'SAVe:WAVEform {channel}, "{remote}"')
            self.query("*OPC?")
            return self._download(remote)

    def get_spectrum(
        self,
        channel: int,
        title: str,
        trace: str = "NORMAL",
        folder_path: str | None = None,
        format: str = "csv",
    ) -> str:
        """Save a channel Spectrum View trace to a local file; returns path."""
        fmt = format.lower()
        source = f"CH{channel}_{token(trace, SpectrumTrace, 'trace')}"
        remote = self._temp_remote(fmt)
        with self.link.temporary_timeout(self._MAX_TRANSFER_TIMEOUT_MS):
            self.write(f'SAVe:WAVEform {source}, "{remote}"')
            self.query("*OPC?")
            data = self._download(remote)
        trace_tag = trace.replace(" ", "")
        local = self._local_path(
            f"{title}_CH{channel}_SV_{trace_tag}_{self.model}_{self._timestamp()}.{fmt}",
            folder_path,
        )
        with open(local, "wb") as fh:
            fh.write(data)
        return local

    def cmd_write_file(self, filepath: str, scope_path: str | None = None) -> str:
        """Upload a local file to the scope, creating scope dirs as needed.

        ### Parameters
        1. filepath : str
            - local source file path
        2. scope_path : str (default None)
            - destination directory on the scope, relative to the home dir
        ### Returns
        1. str
            - full destination path on the scope
        """
        filepath = os.path.abspath(os.path.normpath(filepath))
        if not os.path.isfile(filepath):
            raise ValueError(f"source file not found: {filepath!r}")

        homedir = self.query("FILESystem:HOMEDir?").strip().strip('"').rstrip("/") + "/"
        self.write(f"FILESystem:CWD {quote(homedir)}")
        filedir = homedir

        if scope_path is not None:
            for part in (p for p in scope_path.split("/") if p):
                if part not in self._list_dir_names():
                    self.write(f"FILESystem:MKDir {quote(part)}")
                self.write(f"FILESystem:CWD {quote(part)}")
                filedir = self.query("FILESystem:CWD?").strip().strip('"').rstrip("/") + "/"

        filename = os.path.basename(filepath)
        with open(filepath, "rb") as fh:
            file_bytes = fh.read()
        if len(file_bytes) > self._MAX_UPLOAD_BYTES:
            raise ValueError(f"file too large ({len(file_bytes)} bytes); must be < 2 GB")

        self.write_binary_block(f"FILESystem:WRITEFile {quote(filename)},", file_bytes)
        self.write(f"FILESystem:CWD {quote(homedir)}")
        return filedir + filename

    def set_image_composition(self, mode: ImageComposition | str) -> str:
        """Set whether saved screen images use NORMAL or INVERTED colors.

        Enum arg: <scope>.ImageComposition.<TAB>
        ### Parameters
        1. mode : str
            - ['NORMAL', 'INVERTED']
        ### Returns
        1. mode : str
            - actual composition read back from the scope
        """
        self.write(f"SAVe:IMAGe:COMPosition {token(mode, ImageComposition, 'mode')}")
        return self.query("SAVe:IMAGe:COMPosition?").strip()

    def get_image_composition(self) -> ImageComposition:
        """Query the saved-image color composition (NORMAL/INVERTED)."""
        return from_response(
            self.query("SAVe:IMAGe:COMPosition?").strip(), ImageComposition, "image_composition"
        )

    def _save_and_download(self, save_cmd: str, ext: str) -> bytes:
        """Run a ``SAVe:*`` export to scratch and download the resulting bytes."""
        remote = self._temp_remote(ext)
        with self.link.temporary_timeout(self._MAX_TRANSFER_TIMEOUT_MS):
            self.write(f"{save_cmd} {quote(remote)}")
            self.query("*OPC?")
            return self._download(remote)

    def get_measurement_table(self) -> bytes:
        """Export the measurement results table to CSV and return the bytes.

        Wraps ``SAVe:EVENTtable:MEASUrement`` to a scope scratch file, downloads
        it via the file-transfer path, and deletes the scratch file.
        ### Returns
        1. data : bytes
            - the CSV measurement-results table
        """
        return self._save_and_download("SAVe:EVENTtable:MEASUrement", "csv")

    def get_peaks_table(self) -> bytes:
        """Export the Spectrum-View peaks results table to CSV and return the bytes.

        Wraps ``SAVe:EVENTtable:PEAKS`` to a scratch file and downloads it.
        """
        return self._save_and_download("SAVe:EVENTtable:PEAKS", "csv")

    def get_plot_data(self) -> bytes:
        """Export the active plot's underlying data to CSV and return the bytes.

        Wraps ``SAVe:PLOTData`` to a scratch file and downloads it (the modern
        replacement for the legacy plot-data export).
        """
        return self._save_and_download("SAVe:PLOTData", "csv")

    def parse_wfm_to_numpy(self, wfm_path: str) -> tuple[np.ndarray, list[np.ndarray]]:
        """Parse a local ``.wfm`` file into ``(time_samples, waveforms)``."""
        return wfm_reader.read_wfm_file(wfm_path)

    def parse_wfm_data_to_numpy(self, wfm_data: bytes) -> tuple[np.ndarray, list[np.ndarray]]:
        """Parse ``.wfm`` bytes into ``(time_samples, waveforms)``."""
        return wfm_reader.read_wfm_data(wfm_data)


class _Probe(Node):
    """Per-channel probe info and control (manual command group: Vertical)."""

    def __init__(self, link: InstrumentLink, channel: int) -> None:
        super().__init__(link, name=f"CH{channel}:PROBE")
        self._ch = channel

    def get_id(self) -> tuple[str, str]:
        """Query the probe identity as ``(serial, type)``.

        The scope answers with a compound pair like '"C164798";"TPP1000"'
        ('"N/A";"1X"' on a bare BNC input) - bench-verified.
        """
        resp = self.query(f"CH{self._ch}:PRO:ID?").strip()
        parts = [p.strip().strip('"') for p in resp.split(";")]
        return (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], "")

    def get_type(self) -> str:
        """Query the probe model / type string."""
        return self.query(f"CH{self._ch}:PRO:ID:TYP?").strip().strip('"')

    def get_serial(self) -> str:
        """Query the probe serial number string."""
        return self.query(f"CH{self._ch}:PRO:ID:SER?").strip().strip('"')

    def get_gain(self) -> float:
        """Query the probe gain (reciprocal of attenuation)."""
        return self.query_float(f"CH{self._ch}:PRO:GAIN?")

    def get_units(self) -> str:
        """Query the probe measurement units (e.g. V, A)."""
        return self.query(f"CH{self._ch}:PRO:UNI?").strip().strip('"')

    def get_resistance(self) -> float:
        """Query the probe input resistance in ohms."""
        return self.query_float(f"CH{self._ch}:PRO:RES?")

    def get_status(self) -> str:
        """Query the probe status string."""
        return self.query(f"CH{self._ch}:PRO:STAT?").strip().strip('"')

    def set_input_mode(self, mode: ProbeInputMode | str) -> str:
        """Set the differential-probe input mode.

        Enum arg: <scope>.ProbeInputMode.<TAB>
        ### Parameters
        1. mode : str
            - ['A', 'B', 'COMMON MODE', 'DIFFERENTIAL']
        ### Returns
        1. mode : str
            - actual input mode token read back from the scope
        """
        self.write(f"CH{self._ch}:PRO:INPUTM {token(mode, ProbeInputMode, 'mode')}")
        return self.query(f"CH{self._ch}:PRO:INPUTM?").strip()

    def set_forced_range(self, prange: float) -> float:
        """Set the forced probe range.
        ### Parameters
        1. prange : float
            - forced range in the probe's units
        ### Returns
        1. prange : float
            - actual forced range read back from the scope
        """
        self.write(f"CH{self._ch}:PRO:FORCEDR {prange:.12g}")
        return self.query_float(f"CH{self._ch}:PRO:FORCEDR?")

    def cmd_autozero(self) -> None:
        """Execute the probe auto-zero routine."""
        self.write(f"CH{self._ch}:PRO:AUTOZ EXECute")

    def cmd_degauss(self) -> None:
        """Run the current-probe degauss routine."""
        self.write(f"CH{self._ch}:PRO:DEGAUSS EXECute")

    def get_degauss_state(self) -> str:
        """Read whether the probe needs degauss, is running it, or has passed."""
        return self.query(f"CH{self._ch}:PRO:DEGAUSS:STATE?").strip()

    def cmd_run_selfcal(self) -> None:
        """Start the probe self-calibration routine."""
        self.write(f"CH{self._ch}:PRObe:SELFCal EXECute")

    def get_selfcal_state(self) -> str:
        """Query the probe self-cal state.

        One of DEFAULT / RUNNING / PASSED / FAIL.
        """
        return self.query(f"CH{self._ch}:PRObe:SELFCal:State?").strip()


class _Data(Node):
    """Direct-to-memory waveform transfer (manual command group: Waveform Transfer).

    Reads curve data straight out of instrument memory with ``CURVe?`` and
    scales it via the ``WFMOutpre?`` preamble - the fast path that avoids the
    scope-disk round trip used by the ``file`` subsystem.
    """

    _MAX_TRANSFER_TIMEOUT_MS = 45000

    # WFMOutpre? on this family answers as a bare, unlabeled positional list
    # when HEADer is OFF (bench-verified), so we read the labeled per-field
    # subqueries instead - header-agnostic and free of global state mutation.
    # Each entry: preamble key -> (python caster, WFMOutpre subquery).
    _PREAMBLE_QUERIES = {
        "byt_nr": (int, "WFMOutpre:BYT_Nr?"),
        "bit_nr": (int, "WFMOutpre:BIT_Nr?"),
        "bn_fmt": (str, "WFMOutpre:BN_Fmt?"),
        "byt_or": (str, "WFMOutpre:BYT_Or?"),
        "encdg": (str, "WFMOutpre:ENCdg?"),
        "nr_pt": (int, "WFMOutpre:NR_Pt?"),
        "pt_off": (int, "WFMOutpre:PT_Off?"),
        "xincr": (float, "WFMOutpre:XINcr?"),
        "xzero": (float, "WFMOutpre:XZEro?"),
        "ymult": (float, "WFMOutpre:YMUlt?"),
        "yoff": (float, "WFMOutpre:YOFf?"),
        "yzero": (float, "WFMOutpre:YZEro?"),
    }

    _BN_FMT_TO_DTYPE = {"RI": "i", "RP": "u", "FP": "f"}
    _BYT_OR_TO_ENDIAN = {"MSB": ">", "LSB": "<"}

    def set_source(self, source: str) -> str:
        """Set the CURVe? data source (e.g. ``CH5``, ``MATH1``, ``REF2``).

        ### Parameters
        1. source : str
            - waveform source token
        ### Returns
        1. source : str
            - actual source read back from the scope
        """
        self.write(f"DATa:SOUrce {source}")
        return self.query("DATa:SOUrce?").strip()

    def get_source(self) -> str:
        """Query the current CURVe? data source token."""
        return self.query("DATa:SOUrce?").strip()

    def set_start(self, point: int) -> int:
        """Set the first data point transferred by CURVe? (1-based).

        ### Parameters
        1. point : int
            - first data-point index
        ### Returns
        1. point : int
            - actual start point read back from the scope
        """
        self.write(f"DATa:STARt {int(point)}")
        return self.query_int("DATa:STARt?")

    def get_start(self) -> int:
        """Query the first data point transferred by CURVe?."""
        return self.query_int("DATa:STARt?")

    def set_stop(self, point: int) -> int:
        """Set the last data point transferred by CURVe?.

        ### Parameters
        1. point : int
            - last data-point index
        ### Returns
        1. point : int
            - actual stop point read back from the scope
        """
        self.write(f"DATa:STOP {int(point)}")
        return self.query_int("DATa:STOP?")

    def get_stop(self) -> int:
        """Query the last data point transferred by CURVe?."""
        return self.query_int("DATa:STOP?")

    def set_frame_start(self, frame: int) -> int:
        """Set the first FastFrame frame transferred by CURVe? (segmented readout).

        ### Parameters
        1. frame : int
            - first frame index
        ### Returns
        1. frame : int
            - actual start frame read back from the scope
        """
        self.write(f"DATa:FRAMESTARt {int(frame)}")
        return self.query_int("DATa:FRAMESTARt?")

    def set_frame_stop(self, frame: int | str) -> int | str:
        """Set the last FastFrame frame transferred by CURVe? (or ``'MAX'``).

        When set to ``'MAX'`` the scope echoes the literal token ``MAX`` back
        (bench-verified), so the read-back is returned as the string ``'MAX'``
        rather than coerced to an int; a numeric read-back is returned as int.

        ### Parameters
        1. frame : int | 'MAX'
            - last frame index, or the token ``'MAX'`` for the final frame
        ### Returns
        1. frame : int | str
            - actual stop frame read back (int, or ``'MAX'``)
        """
        if isinstance(frame, str):
            if frame.strip().upper() != "MAX":
                raise ValueError(f"invalid frame stop token {frame!r}; only 'MAX' is allowed")
            self.write("DATa:FRAMESTOP MAX")
        else:
            self.write(f"DATa:FRAMESTOP {int(frame)}")
        return self.get_frame_stop()

    def get_frame_start(self) -> int:
        """Query the first FastFrame frame transferred by CURVe?."""
        return self.query_int("DATa:FRAMESTARt?")

    def get_frame_stop(self) -> int | str:
        """Query the last FastFrame frame transferred by CURVe? (int, or ``'MAX'``)."""
        resp = self.query("DATa:FRAMESTOP?").strip()
        try:
            return int(resp)
        except ValueError:
            return resp.upper()

    def set_encoding(self, encoding: DataEncoding | str) -> str:
        """Set the CURVe? data encoding.

        Enum arg: <scope>.DataEncoding.<TAB>

        ### Parameters
        1. encoding : str
            - ['ASCII', 'RIBINARY', 'RPBINARY', 'FPBINARY', 'SRIBINARY',
              'SRPBINARY', 'SFPBINARY']
        ### Returns
        1. encoding : str
            - actual encoding read back from the scope
        """
        self.write(f"DATa:ENCdg {token(encoding, DataEncoding, 'encoding')}")
        return self.query("DATa:ENCdg?").strip()

    def get_encoding(self) -> DataEncoding:
        """Query the current CURVe? data encoding token."""
        return from_response(self.query("DATa:ENCdg?").strip(), DataEncoding, "encoding")

    def set_width(self, width: int) -> int:
        """Set the byte width per data point (1 or 2 for analog channels).

        ### Parameters
        1. width : int
            - bytes per data point
        ### Returns
        1. width : int
            - actual width read back from the scope
        """
        self.write(f"DATa:WIDth {int(width)}")
        return self.query_int("DATa:WIDth?")

    def get_width(self) -> int:
        """Query the byte width per data point."""
        return self.query_int("DATa:WIDth?")

    def get_preamble(self) -> dict:
        """Read the ``WFMOutpre`` formatting/scaling fields into a dict.

        Reads the labeled per-field subqueries (header-agnostic) rather than
        the aggregate ``WFMOutpre?`` list, which this family returns unlabeled
        when HEADer is OFF.

        ### Returns
        1. preamble : dict
            - keys: byt_nr, bit_nr, bn_fmt, byt_or, encdg, nr_pt, pt_off,
              xincr, xzero, ymult, yoff, yzero
        """
        preamble: dict = {}
        for key, (caster, query) in self._PREAMBLE_QUERIES.items():
            if caster is str:
                preamble[key] = self.query(query).strip().strip('"').upper()
            elif caster is int:
                preamble[key] = self.query_int(query)
            else:
                preamble[key] = self.query_float(query)
        return preamble

    def _dtype(self, preamble: dict):
        """Build a numpy dtype from the preamble's format/order/width fields."""
        code = self._BN_FMT_TO_DTYPE.get(preamble.get("bn_fmt", "RI"), "i")
        endian = self._BYT_OR_TO_ENDIAN.get(preamble.get("byt_or", "MSB"), ">")
        width = preamble.get("byt_nr", 1)
        return np.dtype(f"{endian}{code}{width}")

    def get_waveform(self, source: str | None = None, width: int = 1) -> np.ndarray:
        """Transfer a full waveform record straight from memory into a numpy array.

        Sets the source (if given), forces signed binary encoding, transfers
        the whole record with ``CURVe?`` and scales the raw codes to physical
        units via the ``WFMOutpre?`` preamble:
        ``y = YZEro + YMUlt * (raw - YOFf)``.

        ### Parameters
        1. source : str (default None)
            - waveform source token (e.g. ``CH5``); None keeps the current source
        2. width : int (default 1)
            - bytes per data point (1 or 2)
        ### Returns
        1. samples : numpy.ndarray
            - the scaled waveform in the preamble's vertical units
        """
        if source is not None:
            self.set_source(source)
        self.set_encoding("RIBINARY")
        self.set_width(width)
        self.write("DATa:STARt 1")
        record = self.query_int("HOR:RECO?")
        self.write(f"DATa:STOP {record}")
        preamble = self.get_preamble()
        with self.link.temporary_timeout(self._MAX_TRANSFER_TIMEOUT_MS):
            self.write("CURVe?")
            block = self.read_binary_block()
        raw = np.frombuffer(block, dtype=self._dtype(preamble)).astype(np.float64)
        ymult = preamble.get("ymult", 1.0)
        yoff = preamble.get("yoff", 0.0)
        yzero = preamble.get("yzero", 0.0)
        return yzero + ymult * (raw - yoff)

    def get_time_axis(self, preamble: dict | None = None) -> np.ndarray:
        """Return the time axis (seconds) for the most recent transfer.

        ### Parameters
        1. preamble : dict (default None)
            - a preamble from get_preamble(); queried fresh if None
        ### Returns
        1. time : numpy.ndarray
            - ``XZEro + XINcr * (n - PT_Off)`` for each transferred point
        """
        if preamble is None:
            preamble = self.get_preamble()
        n = preamble.get("nr_pt", 0)
        xincr = preamble.get("xincr", 1.0)
        xzero = preamble.get("xzero", 0.0)
        pt_off = preamble.get("pt_off", 0)
        return xzero + xincr * (np.arange(n, dtype=np.float64) - pt_off)


class _Reference(Node):
    """Reference waveform (manual command group: Vertical / REF<x>).

    Reference slots are created on demand: :meth:`add` allocates the slot,
    :meth:`cmd_save_from` captures a live source into it, and per-slot
    source/label/deskew controls live on the same node. :meth:`get_all` is a
    collection-level query but is exposed here for convenience.
    """

    def __init__(self, link: InstrumentLink, ref: int) -> None:
        super().__init__(link, name=f"REF{ref}")
        self._ref = ref

    def cmd_add(self) -> list[str]:
        """Allocate this reference slot; returns the new list."""
        self.write(f'REF:ADDNew "REF{self._ref}"')
        return self.get_all()

    def cmd_delete(self) -> list[str]:
        """Delete this reference slot; returns the remaining list."""
        self.write(f'REF:DELete "REF{self._ref}"')
        return self.get_all()

    def get_all(self) -> list[str]:
        """List all defined reference waveforms."""
        listing = self.query("REF:LIST?").strip()
        if not listing or listing.upper() in ("NONE", '""'):
            return []
        return [t.strip().strip('"') for t in listing.split(",") if t.strip()]

    def cmd_save_from(self, source: str) -> list[str]:
        """Create this reference slot from a live source (e.g. ``CH5``).

        ``SAVe:WAVEform <source>, REF<x>`` is NOT implemented on this family
        (bench-verified: "Save waveform to REF or EMEM not implemented",
        ADR-020). Instead this selects ``source`` (``SELect:CH<x>`` for a
        channel) and adds the reference with ``REF:ADDNew`` - which snapshots
        the selected source into the new reference slot - then returns the
        reference list so the caller can confirm the slot exists.
        ### Parameters
        1. source : str
            - source token (e.g. ``CH5``); channels are selected first
        ### Returns
        1. references : list[str]
            - the reference list after the capture
        """
        if source.upper().startswith("CH"):
            self.write(f"SELect:{source} ON")
        self.write(f"DISplay:SELect:SOUrce {source}")
        self.write(f'REF:ADDNew "REF{self._ref}"')
        self.query("*OPC?")
        return self.get_all()

    def set_source_file(self, filename: str) -> str:
        """Set this reference's source file; returns readback."""
        self.write(f"REF:REF{self._ref}:SOUrce {quote(filename)}")
        return self.query(f"REF:REF{self._ref}:SOUrce?").strip().strip('"')

    def set_label(self, label: str) -> str:
        """Set this reference's label name; returns readback.

        An embedded double quote is rejected so it cannot break out of quoting.
        """
        if '"' in label:
            raise ValueError(f"label must not contain a double-quote character; got {label!r}")
        self.write(f'REF:REF{self._ref}:LABel:NAMe "{label}"')
        return self.query(f"REF:REF{self._ref}:LABel:NAMe?").strip().strip('"')

    def get_label(self) -> str:
        """Query this reference's label name."""
        return self.query(f"REF:REF{self._ref}:LABel:NAMe?").strip().strip('"')

    def set_deskew(self, deskew: float) -> float:
        """Set this reference's deskew time in seconds."""
        self.write(f"REF:REF{self._ref}:DESKew {deskew:.12g}")
        return self.query_float(f"REF:REF{self._ref}:DESKew?")

    def get_deskew(self) -> float:
        """Query this reference's deskew time in seconds."""
        return self.query_float(f"REF:REF{self._ref}:DESKew?")


class _Bus(Node):
    """Bus decode (manual command group: Bus)."""

    def __init__(self, link: InstrumentLink, bus: int) -> None:
        super().__init__(link, name=f"BUS{bus}")
        self._bus = bus


class _Math(Node):
    """Math waveform (manual command group: Math)."""

    def __init__(self, link: InstrumentLink, math: int) -> None:
        super().__init__(link, name=f"MATH{math}")
        self._math = math


class _Zoom(Node):
    """Zoom control (manual command group: Zoom).

    Wraps the ``DISplay:WAVEView<v>:ZOOM:ZOOM<z>:...`` group. Each method takes
    optional ``zoom`` and ``view`` indices, defaulting to zoom 1 in waveform
    view 1.
    """

    def _p(self, zoom: int, view: int) -> str:
        return f"DISplay:WAVEView{int(view)}:ZOOM:ZOOM{int(zoom)}"

    def set_state(self, state: bool, zoom: int = 1, view: int = 1) -> bool:
        """Turn a zoom on or off; returns the read-back state."""
        p = self._p(zoom, view)
        self.write(f"{p}:STATe {'ON' if state else 'OFF'}")
        return self.query_bool(f"{p}:STATe?")

    def get_state(self, zoom: int = 1, view: int = 1) -> bool:
        """Query whether a zoom is on."""
        return self.query_bool(f"{self._p(zoom, view)}:STATe?")

    def set_horizontal_scale(self, scale: float, zoom: int = 1, view: int = 1) -> float:
        """Set the zoom horizontal scale (seconds/div); returns readback."""
        p = self._p(zoom, view)
        self.write(f"{p}:HORizontal:SCALe {scale:.12g}")
        return self.query_float(f"{p}:HORizontal:SCALe?")

    def set_horizontal_position(self, position: float, zoom: int = 1, view: int = 1) -> float:
        """Set the zoom horizontal position (percent of record); returns readback."""
        p = self._p(zoom, view)
        self.write(f"{p}:HORizontal:POSition {position:.12g}")
        return self.query_float(f"{p}:HORizontal:POSition?")

    def set_horizontal_winscale(self, winscale: float, zoom: int = 1, view: int = 1) -> float:
        """Set the zoom window horizontal scale (seconds/div); returns readback."""
        p = self._p(zoom, view)
        self.write(f"{p}:HORizontal:WINSCALe {winscale:.12g}")
        return self.query_float(f"{p}:HORizontal:WINSCALe?")

    def set_vertical_scale(self, scale: float, zoom: int = 1, view: int = 1) -> float:
        """Set the zoom vertical scale; returns readback."""
        p = self._p(zoom, view)
        self.write(f"{p}:VERTical:SCALe {scale:.12g}")
        return self.query_float(f"{p}:VERTical:SCALe?")

    def set_vertical_position(self, position: float, zoom: int = 1, view: int = 1) -> float:
        """Set the zoom vertical position; returns readback."""
        p = self._p(zoom, view)
        self.write(f"{p}:VERTical:POSition {position:.12g}")
        return self.query_float(f"{p}:VERTical:POSition?")

    def get_horizontal_scale(self, zoom: int = 1, view: int = 1) -> float:
        """Query the zoom horizontal scale (seconds/div)."""
        return self.query_float(f"{self._p(zoom, view)}:HORizontal:SCALe?")

    def get_horizontal_position(self, zoom: int = 1, view: int = 1) -> float:
        """Query the zoom horizontal position (percent of record)."""
        return self.query_float(f"{self._p(zoom, view)}:HORizontal:POSition?")

    def get_vertical_scale(self, zoom: int = 1, view: int = 1) -> float:
        """Query the zoom vertical scale."""
        return self.query_float(f"{self._p(zoom, view)}:VERTical:SCALe?")

    def get_vertical_position(self, zoom: int = 1, view: int = 1) -> float:
        """Query the zoom vertical position."""
        return self.query_float(f"{self._p(zoom, view)}:VERTical:POSition?")


class _Cursors(Node):
    """Cursor control (manual command group: Cursor).

    The 4/5/6 Series scopes scope cursors to a waveform view, so the real
    commands are ``DISplay:WAVEView<v>:CURSor:CURSOR<c>:...`` (bench-verified -
    the bare ``CURSor:...`` form of older scopes is absent here, ADR-019). Each
    method takes optional ``cursor`` (1 or 2) and ``view`` indices, defaulting
    to cursor 1 in waveform view 1.
    """

    def _p(self, cursor: int, view: int) -> str:
        return f"DISplay:WAVEView{int(view)}:CURSor:CURSOR{int(cursor)}"

    def set_state(self, state: bool, cursor: int = 1, view: int = 1) -> bool:
        """Turn the cursor pair on or off; returns the read-back state."""
        p = self._p(cursor, view)
        self.write(f"{p}:STATE {'ON' if state else 'OFF'}")
        return self.query_bool(f"{p}:STATE?")

    def get_state(self, cursor: int = 1, view: int = 1) -> bool:
        """Query whether the cursor pair is on."""
        return self.query_bool(f"{self._p(cursor, view)}:STATE?")

    def set_function(self, function: CursorFunction | str, cursor: int = 1, view: int = 1) -> str:
        """Set the cursor type.

        Enum arg: <scope>.CursorFunction.<TAB>
        ### Parameters
        1. function : str
            - ['HBARS', 'VBARS', 'SCREEN', 'WAVEFORM']
        2. *cursor : int (default 1)
        3. *view : int (default 1)
        ### Returns
        1. function : str
            - actual cursor function read back from the scope
        """
        p = self._p(cursor, view)
        self.write(f"{p}:FUNCtion {token(function, CursorFunction, 'function')}")
        return self.query(f"{p}:FUNCtion?").strip()

    def set_mode(self, mode: CursorMode | str, cursor: int = 1, view: int = 1) -> str:
        """Set cursor navigation mode.

        Enum arg: <scope>.CursorMode.<TAB>
        ### Parameters
        1. mode : str
            - ['TRACK', 'INDEPENDENT']
        ### Returns
        1. mode : str
            - actual cursor mode read back from the scope
        """
        p = self._p(cursor, view)
        self.write(f"{p}:MODe {token(mode, CursorMode, 'mode')}")
        return self.query(f"{p}:MODe?").strip()

    def set_source_a(self, source: str, cursor: int = 1, view: int = 1) -> str:
        """Set cursor A's source (e.g. ``CH5``, ``AUTO``); returns readback."""
        p = self._p(cursor, view)
        self.write(f"{p}:ASOUrce {source}")
        return self.query(f"{p}:ASOUrce?").strip()

    def set_source_b(self, source: str, cursor: int = 1, view: int = 1) -> str:
        """Set cursor B's source (e.g. ``CH5``, ``AUTO``); returns readback."""
        p = self._p(cursor, view)
        self.write(f"{p}:BSOUrce {source}")
        return self.query(f"{p}:BSOUrce?").strip()

    def set_vbar_positions(
        self, a: float, b: float, cursor: int = 1, view: int = 1
    ) -> tuple[float, float]:
        """Set the vertical-bar (time) cursor A/B positions in seconds.
        ### Returns
        1. (a, b) : tuple[float, float]
            - actual A/B positions read back from the scope
        """
        p = self._p(cursor, view)
        self.write(f"{p}:VBArs:APOSition {a:.12g}")
        self.write(f"{p}:VBArs:BPOSition {b:.12g}")
        return (
            self.query_float(f"{p}:VBArs:APOSition?"),
            self.query_float(f"{p}:VBArs:BPOSition?"),
        )

    def get_vbar_delta(self, cursor: int = 1, view: int = 1) -> float:
        """Query the vertical-bar cursor delta (B - A) in horizontal units."""
        return self.query_float(f"{self._p(cursor, view)}:VBArs:DELTa?")

    def get_vbar_units(self, cursor: int = 1, view: int = 1) -> str:
        """Query the vertical-bar cursor units."""
        return self.query(f"{self._p(cursor, view)}:VBArs:UNIts?").strip().strip('"')

    def set_hbar_positions(
        self, a: float, b: float, cursor: int = 1, view: int = 1
    ) -> tuple[float, float]:
        """Set the horizontal-bar (amplitude) cursor A/B positions in vertical units.
        ### Returns
        1. (a, b) : tuple[float, float]
            - actual A/B positions read back from the scope
        """
        p = self._p(cursor, view)
        self.write(f"{p}:HBArs:APOSition {a:.12g}")
        self.write(f"{p}:HBArs:BPOSition {b:.12g}")
        return (
            self.query_float(f"{p}:HBArs:APOSition?"),
            self.query_float(f"{p}:HBArs:BPOSition?"),
        )

    def get_hbar_delta(self, cursor: int = 1, view: int = 1) -> float:
        """Query the horizontal-bar cursor delta (B - A) in vertical units."""
        return self.query_float(f"{self._p(cursor, view)}:HBArs:DELTa?")


class _Display(Node):
    """Display control (manual command group: Display / Display Control)."""

    def set_select_source(self, source: str) -> str:
        """Set the globally selected (active) waveform source.
        ### Parameters
        1. source : str
            - ['NONE', 'CH<x>', 'BUS<x>', 'MATH<x>', 'PLOT<x>', 'REF<x>']
        ### Returns
        1. source : str
            - actual selected source read back from the scope
        """
        self.write(f"DISplay:SELect:SOUrce {source}")
        return self.query("DISplay:SELect:SOUrce?").strip()

    def get_select_source(self) -> str:
        """Query the globally selected (active) waveform source."""
        return self.query("DISplay:SELect:SOUrce?").strip()

    def set_select_view(self, view: str) -> str:
        """Set the selected view (e.g. WAVEVIEW1, MATHFFT<x>, PLOTVIEW<x>, REFFFT<x>)."""
        self.write(f"DISplay:SELect:VIEW {view}")
        return self.query("DISplay:SELect:VIEW?").strip()

    def get_select_view(self) -> str:
        """Query the selected view."""
        return self.query("DISplay:SELect:VIEW?").strip()

    def set_select_math(self, math: str) -> str:
        """Set the selected math waveform (e.g. ``MATH1``); returns readback."""
        self.write(f"DISplay:SELect:MATH {math}")
        return self.query("DISplay:SELect:MATH?").strip()

    def set_select_bus(self, bus: str) -> str:
        """Set the selected bus (e.g. ``BUS1``); returns readback."""
        self.write(f"DISplay:SELect:BUS {bus}")
        return self.query("DISplay:SELect:BUS?").strip()

    def set_select_reference(self, reference: str) -> str:
        """Set the selected reference waveform (``NONE`` or ``REF<x>``); returns readback."""
        self.write(f"DISplay:SELect:REFerence {reference}")
        return self.query("DISplay:SELect:REFerence?").strip()

    def set_waveview_source(self, source: str, view: int = 1) -> str:
        """Set the active source for a waveform view.
        ### Parameters
        1. source : str
            - ['CH<x>', 'MATH<x>', 'BUS<x>', 'REF<x>', 'PLOT<x>']
        2. *view : int (default 1)
            - waveform view index
        ### Returns
        1. source : str
            - actual active source read back from the scope
        """
        self.write(f"DISplay:SELect:WAVEView{int(view)}:SOUrce {source}")
        return self.query(f"DISplay:SELect:WAVEView{int(view)}:SOUrce?").strip()

    def get_waveview_source(self, view: int = 1) -> str:
        """Query the active source for a waveform view."""
        return self.query(f"DISplay:SELect:WAVEView{int(view)}:SOUrce?").strip()


class _Search(Node):
    """Search and mark (manual command group: Search and Mark)."""


class _Dvm(Node):
    """Digital voltmeter (manual command group: DVM).

    The DVM is a free, registration-enabled feature; its ``installed`` flag is
    set from the R2 runtime probe on connect. On the bench MSO58 it is NOT
    enabled (``dvm.installed`` is False), so every method here raises
    :class:`HardwareNotPresentError` via the Node gate rather than issuing SCPI
    that would hang. Implemented from DVM.md; command-acceptance is not
    bench-testable on this unit (see the captures README).
    """

    def reset(self) -> None:
        """Reset the DVM measurement history (``DVM RESET``)."""
        self.write("DVM RESET")

    def set_mode(self, mode: DvmMode | str) -> str:
        """Set the DVM measurement mode.

        Enum arg: <scope>.DvmMode.<TAB>
        ### Parameters
        1. mode : str
            - ['ACRMS', 'ACDCRMS', 'DC', 'OFF']
        ### Returns
        1. mode : str
            - actual DVM mode read back from the scope
        """
        self.write(f"DVM:MODe {token(mode, DvmMode, 'mode')}")
        return self.query("DVM:MODe?").strip()

    def get_mode(self) -> DvmMode:
        """Query the DVM measurement mode."""
        return from_response(self.query("DVM:MODe?").strip(), DvmMode, "mode")

    def set_source(self, source: str) -> str:
        """Set the DVM source channel; returns readback."""
        self.write(f"DVM:SOUrce {source}")
        return self.query("DVM:SOUrce?").strip()

    def get_source(self) -> str:
        """Query the DVM source channel."""
        return self.query("DVM:SOUrce?").strip()

    def set_autorange(self, state: bool) -> bool:
        """Enable/disable DVM autorange; returns readback."""
        self.write(f"DVM:AUTORange {'ON' if state else 'OFF'}")
        return self.query_bool("DVM:AUTORange?")

    def get_autorange(self) -> bool:
        """Query whether DVM autorange is enabled."""
        return self.query_bool("DVM:AUTORange?")

    def set_frequency_counter(self, state: bool) -> bool:
        """Enable/disable the DVM trigger frequency counter; returns readback."""
        self.write(f"DVM:TRIGger:FREQuency:COUNTer {'ON' if state else 'OFF'}")
        return self.query_bool("DVM:TRIGger:FREQuency:COUNTer?")

    def get_value(self) -> float:
        """Query the current DVM measurement value."""
        return self.query_float("DVM:MEASUrement:VALue?")

    def get_frequency(self) -> float:
        """Query the DVM frequency-counter reading in Hz."""
        return self.query_float("DVM:MEASUrement:FREQuency?")

    def get_history_average(self) -> float:
        """Query the DVM measurement history average."""
        return self.query_float("DVM:MEASUrement:HIStory:AVErage?")

    def get_history_maximum(self) -> float:
        """Query the DVM measurement history maximum."""
        return self.query_float("DVM:MEASUrement:HIStory:MAXimum?")

    def get_history_minimum(self) -> float:
        """Query the DVM measurement history minimum."""
        return self.query_float("DVM:MEASUrement:HIStory:MINImum?")

    def get_inf_maximum(self) -> float:
        """Query the DVM infinite-persistence maximum."""
        return self.query_float("DVM:MEASUrement:INFMAXimum?")

    def get_inf_minimum(self) -> float:
        """Query the DVM infinite-persistence minimum."""
        return self.query_float("DVM:MEASUrement:INFMINimum?")


class _AuxOut(Node):
    """Rear-panel Auxiliary OUTPUT BNC (manual command group: AUXout).

    Named ``_AuxOut`` (not ``_Aux``) because some models also have a separate
    auxiliary trigger INPUT connector; this node is only the aux output.

    The single Aux Out BNC emits one of: the A (main) trigger (``'ATRIGGER'``),
    the 10 MHz reference clock (``'REFOUT'``), or the built-in AFG output
    (``'AFG'``). The edge polarity applies when the source is the trigger.
    Setters return the scope's read-back token.
    """

    def set_source(self, source: str) -> str:
        """Set the Aux Out signal source.

        ### Parameters
        1. source : str
            - ['ATRIGGER' (A/main trigger), 'REFOUT' (10 MHz reference),
              'AFG' (function generator)]
        ### Returns
        1. source : str
            - actual source token read back (ATRIGGER / REFOUT / AFG)
        """
        self.write(f"AUXout:SOUrce {token(source, AuxOutSource, 'source')}")
        return self.get_source()

    def get_source(self) -> AuxOutSource:
        """What is currently driving the Aux Out connector."""
        return from_response(self.query("AUXout:SOUrce?"), AuxOutSource, "source")

    def set_edge(self, edge: str) -> str:
        """Set the Aux Out trigger transition edge: RISING or FALLING.

        Applies when the source is the A trigger ('ATRIGGER'). Returns read-back.
        """
        self.write(f"AUXout:EDGE {token(edge, AuxOutEdge, 'edge')}")
        return self.get_edge()

    def get_edge(self) -> AuxOutEdge:
        """Which edge the Aux Out pulse presents."""
        return from_response(self.query("AUXout:EDGE?"), AuxOutEdge, "edge")


class _Afg(Node):
    """Built-in arbitrary function generator. Requires option AFG.

    One `set_waveform_*` method per function. The method you call selects the
    function, so no string names a waveform shape. On the bench MSO58 the
    output is looped to CH5.

    Additive noise applies to every function except DC and NOISE. Those two
    methods therefore have no `noise_add` argument. The instrument declines the
    write silently on those functions and reports no error, so the signature is
    the only warning a caller gets (bench-verified 2026-08-06).

    Every helper writes only the values that differ from the present setting. A
    query costs 0.74 ms and a synchronized write costs about 135 ms on this LAN
    link, so a redundant write is roughly 180 times more expensive than the
    check that avoids it (bench-verified 2026-08-06).
    """

    # ---------- internal apply ----------

    def _apply(
        self,
        function: str,
        *,
        frequency: float | None = None,
        amplitude: float | None = None,
        offset: float | None = None,
        duty: float | None = None,
        symmetry: float | None = None,
        width: float | None = None,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Write the function then only the values that differ. Return the state.

        Order follows the design standard: mode first, then levels. Frequency,
        amplitude and offset all survive a function change on this instrument,
        so no value needs rewriting afterwards (bench-verified 2026-08-06).
        """
        wanted = token(function, Function, "function")
        if self.query("AFG:FUNCtion?").strip().upper() != wanted.upper():
            self.write(f"AFG:FUNCtion {wanted}")

        for cmd, value in (
            ("AFG:FREQuency", frequency),
            ("AFG:AMPLitude", amplitude),
            ("AFG:OFFSet", offset),
            ("AFG:SQUare:DUty", duty),
            ("AFG:RAMP:SYMmetry", symmetry),
            ("AFG:PULse:WIDth", width),
        ):
            if value is None:
                continue
            if self.query_float(f"{cmd}?") != float(value):
                self.write(f"{cmd} {float(value):.12g}")

        if noise_add is not None:
            self._apply_noise(noise_add)
        return self.get_waveform()

    def _apply_noise(self, noise_add: float | bool) -> None:
        """Set additive noise. False disables it. A number sets and enables it."""
        if noise_add is False:
            if self.query_bool("AFG:NOISEAdd:STATE?"):
                self.write("AFG:NOISEAdd:STATE OFF")
            return
        percent = float(noise_add)
        if self.query_float("AFG:NOISEAdd:PERCent?") != percent:
            self.write(f"AFG:NOISEAdd:PERCent {percent:.12g}")
        if not self.query_bool("AFG:NOISEAdd:STATE?"):
            self.write("AFG:NOISEAdd:STATE ON")

    # ---------- full waveform specifications ----------

    def set_waveform_sine(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a sine wave at freq in Hz and ampl in volts peak to peak."""
        return self._apply(
            "SINE", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_square(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        duty: float | None = None,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a square wave at freq in Hz, ampl in Vpp, duty in percent."""
        return self._apply(
            "SQUARE",
            frequency=freq,
            amplitude=ampl,
            offset=offset,
            duty=duty,
            noise_add=noise_add,
        )

    def set_waveform_pulse(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        width: float | None = None,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a pulse train at freq in Hz, ampl in Vpp, width in seconds."""
        return self._apply(
            "PULSE",
            frequency=freq,
            amplitude=ampl,
            offset=offset,
            width=width,
            noise_add=noise_add,
        )

    def set_waveform_ramp(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        symmetry: float | None = None,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a ramp at freq in Hz, ampl in Vpp, symmetry in percent."""
        return self._apply(
            "RAMP",
            frequency=freq,
            amplitude=ampl,
            offset=offset,
            symmetry=symmetry,
            noise_add=noise_add,
        )

    def set_waveform_sinc(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a sinc pulse at freq in Hz and ampl in volts peak to peak."""
        return self._apply(
            "SINC", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_gaussian(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a gaussian pulse at freq in Hz and ampl in Vpp."""
        return self._apply(
            "GAUSSIAN", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_lorentz(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a lorentz pulse at freq in Hz and ampl in Vpp."""
        return self._apply(
            "LORENTZ", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_exp_rise(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output an exponential rise at freq in Hz and ampl in Vpp."""
        return self._apply(
            "ERISE", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_exp_decay(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output an exponential decay at freq in Hz and ampl in Vpp."""
        return self._apply(
            "EDECAY", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_haversine(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a haversine at freq in Hz and ampl in volts peak to peak."""
        return self._apply(
            "HAVERSINE", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_cardiac(
        self,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output a cardiac waveform at freq in Hz and ampl in Vpp."""
        return self._apply(
            "CARDIAC", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_arb_file(
        self,
        path: str,
        freq: float,
        ampl: float,
        *,
        offset: float = 0.0,
        noise_add: float | bool | None = None,
    ) -> WaveformState:
        """Output the arbitrary waveform in the named scope-side file.

        The MSO loads an arb from a file on the instrument. The bench signal
        generators upload sample arrays instead, so their method takes samples
        and this one takes a path.
        """
        if self.query("AFG:ARBitrary:SOUrce?").strip().strip('"') != path:
            self.write(f"AFG:ARBitrary:SOUrce {quote(path)}")
        return self._apply(
            "ARBITRARY", frequency=freq, amplitude=ampl, offset=offset, noise_add=noise_add
        )

    def set_waveform_noise(self, ampl: float, *, offset: float = 0.0) -> WaveformState:
        """Output band-limited noise at ampl in volts peak to peak.

        This function takes no `noise_add` argument. The output is already
        noise, and the instrument declines additive noise here without an error.
        """
        return self._apply("NOISE", amplitude=ampl, offset=offset)

    def set_waveform_dc(self, offset: float) -> WaveformState:
        """Output a DC level at offset in volts.

        This function takes no amplitude and no `noise_add` argument. The
        instrument declines additive noise here without an error.
        """
        return self._apply("DC", offset=offset)

    def get_waveform(self) -> WaveformState:
        """Read the complete output setting."""
        function = self.query("AFG:FUNCtion?").strip()
        upper = function.upper()
        return WaveformState(
            function=function,
            frequency=self.query_float("AFG:FREQuency?"),
            amplitude=self.query_float("AFG:AMPLitude?"),
            offset=self.query_float("AFG:OFFSet?"),
            output_state=self.query_bool("AFG:OUTPut:STATE?"),
            duty=self.query_float("AFG:SQUare:DUty?") if upper.startswith("SQU") else None,
            symmetry=self.query_float("AFG:RAMP:SYMmetry?") if upper.startswith("RAMP") else None,
            width=self.query_float("AFG:PULse:WIDth?") if upper.startswith("PULS") else None,
            noise_add=(
                self.query_float("AFG:NOISEAdd:PERCent?")
                if self.query_bool("AFG:NOISEAdd:STATE?")
                else None
            ),
        )

    # ---------- atomic setters (justified exceptions) ----------

    def set_frequency(self, hz: float) -> float:
        """Set the output frequency in Hz without changing the function.

        Kept public so a loop that steps frequency does not rewrite amplitude
        on every pass. See the design standard, section 5.3.
        """
        self.write(f"AFG:FREQuency {hz:.12g}")
        return self.query_float("AFG:FREQuency?")

    def get_frequency(self) -> float:
        """Read the output frequency in Hz."""
        return self.query_float("AFG:FREQuency?")

    def set_amplitude(self, volts: float) -> float:
        """Set the output amplitude in volts peak to peak."""
        self.write(f"AFG:AMPLitude {volts:.12g}")
        return self.query_float("AFG:AMPLitude?")

    def get_amplitude(self) -> float:
        """Read the output amplitude in volts peak to peak."""
        return self.query_float("AFG:AMPLitude?")

    def set_offset(self, volts: float) -> float:
        """Set the output DC offset in volts."""
        self.write(f"AFG:OFFSet {volts:.12g}")
        return self.query_float("AFG:OFFSet?")

    def get_offset(self) -> float:
        """Read the output DC offset in volts."""
        return self.query_float("AFG:OFFSet?")

    def set_levels(self, high: float, low: float) -> tuple[float, float]:
        """Set the waveform high and low levels in volts.

        This is the alternative to amplitude and offset. Returns the read-back
        pair.
        """
        self.write(f"AFG:HIGHLevel {high:.12g}")
        self.write(f"AFG:LOWLevel {low:.12g}")
        return self.get_levels()

    def get_levels(self) -> tuple[float, float]:
        """Read the waveform high and low levels in volts."""
        return (
            self.query_float("AFG:HIGHLevel?"),
            self.query_float("AFG:LOWLevel?"),
        )

    # ---------- private one-to-one setters ----------

    def _set_period(self, seconds: float) -> float:
        """Write the output period (AFG:PERIod). Reciprocal of frequency."""
        self.write(f"AFG:PERIod {seconds:.12g}")
        return self.query_float("AFG:PERIod?")

    def _set_high_level(self, level: float) -> float:
        """Write the waveform high level (AFG:HIGHLevel)."""
        self.write(f"AFG:HIGHLevel {level:.12g}")
        return self.query_float("AFG:HIGHLevel?")

    def set_low_level(self, level: float) -> float:
        """Set the waveform low level in volts.

        ### Parameters
        1. level : float
            - low level in volts
        ### Returns
        1. level : float
            - actual low level read back from the scope
        """
        self.write(f"AFG:LOWLevel {level:.12g}")
        return self.query_float("AFG:LOWLevel?")

    def get_low_level(self) -> float:
        """Query the waveform low level in volts."""
        return self.query_float("AFG:LOWLevel?")

    def _set_square_duty(self, percent: float) -> float:
        """Write the square-wave duty cycle (AFG:SQUare:DUty), 10.0 to 90.0 percent."""
        self.write(f"AFG:SQUare:DUty {percent:.12g}")
        return self.query_float("AFG:SQUare:DUty?")

    def _set_ramp_symmetry(self, percent: float) -> float:
        """Write the ramp symmetry (AFG:RAMP:SYMmetry), 0.0 to 100.0 percent."""
        self.write(f"AFG:RAMP:SYMmetry {percent:.12g}")
        return self.query_float("AFG:RAMP:SYMmetry?")

    def _set_pulse_width(self, width: float) -> float:
        """Write the pulse width in seconds (AFG:PULse:WIDth)."""
        self.write(f"AFG:PULse:WIDth {width:.12g}")
        return self.query_float("AFG:PULse:WIDth?")

    def _set_noise_add_state(self, state: bool) -> bool:
        """Write the additive-noise enable (AFG:NOISEAdd:STATE).

        The instrument declines this on the DC and NOISE functions and reports
        no error. The read-back is the only way to detect the refusal
        (bench-verified 2026-08-06).
        """
        self.write(f"AFG:NOISEAdd:STATE {'ON' if state else 'OFF'}")
        return self.query_bool("AFG:NOISEAdd:STATE?")

    def _set_noise_add_percent(self, percent: float) -> float:
        """Write the additive-noise level (AFG:NOISEAdd:PERCent), 0.0 to 100.0."""
        self.write(f"AFG:NOISEAdd:PERCent {percent:.12g}")
        return self.query_float("AFG:NOISEAdd:PERCent?")

    def _set_run_mode(self, mode: AfgOutputMode | str) -> str:
        """Write the output run mode (AFG:OUTPut:MODe): OFF, CONTinuous or BURSt."""
        self.write(f"AFG:OUTPut:MODe {token(mode, AfgOutputMode, 'mode')}")
        return self.query("AFG:OUTPut:MODe?").strip()

    # ---------- output ----------

    def set_output_state(self, state: bool) -> bool:
        """Drive or stop driving the output connector."""
        self.write(f"AFG:OUTPut:STATE {'ON' if state else 'OFF'}")
        return self.query_bool("AFG:OUTPut:STATE?")

    def get_output_state(self) -> bool:
        """Read whether the output connector is driven."""
        return self.query_bool("AFG:OUTPut:STATE?")

    def set_load(self, load: float | str) -> float:
        """Set the expected far-end load in ohms. Accepts 50 or Load.HIGHZ.

        This tells the generator what the cable terminates into, so the
        amplitude setting matches what the load actually sees. It does not
        change the source impedance. Returns ohms, or inf for high impedance.
        """
        if isinstance(load, (int, float)) and not isinstance(load, bool):
            if float(load) == 50.0:
                wanted = Load.FIFTY.value
            elif float(load) == float("inf"):
                wanted = Load.HIGHZ.value
            else:
                raise ValueError(f"this AFG accepts only 50 ohm or high impedance, got {load!r}")
        else:
            wanted = token(load, Load, "load")
        self.write(f"AFG:OUTPut:LOAd:IMPEDance {wanted}")
        return self.get_load()

    def get_load(self) -> float:
        """Read the expected far-end load in ohms. Returns inf for high impedance."""
        token = self.query("AFG:OUTPut:LOAd:IMPEDance?").strip().upper()
        return 50.0 if token.startswith("FIF") else float("inf")

    # ---------- burst ----------

    def set_burst(self, count: int) -> int:
        """Switch to burst mode and set the cycles per burst. Returns the count.

        Call :meth:`cmd_burst_trigger` to fire one burst.
        """
        self._set_run_mode("BURST")
        self.write(f"AFG:BURSt:CCOUnt {int(count)}")
        return self.query_int("AFG:BURSt:CCOUnt?")

    def get_burst(self) -> int:
        """Read the cycles per burst."""
        return self.query_int("AFG:BURSt:CCOUnt?")

    def set_continuous(self) -> str:
        """Switch back to continuous output. Returns the run-mode token."""
        return self._set_run_mode("CONTINUOUS")

    def cmd_burst_trigger(self) -> None:
        """Fire one burst. Requires burst mode."""
        self.write("AFG:BURSt:TRIGger")

    # ---------- arbitrary source ----------

    def get_arb_file(self) -> str:
        """Read the scope-side path of the loaded arbitrary waveform file."""
        return self.query("AFG:ARBitrary:SOUrce?").strip().strip('"')

    def _set_arb_file(self, source_path: str) -> str:
        """Write the arbitrary-waveform source path (AFG:ARBitrary:SOUrce)."""
        self.write(f"AFG:ARBitrary:SOUrce {quote(source_path)}")
        return self.get_arb_file()


class _Digital(Node):
    """Digital channel configuration (manual command group: Digital)."""
