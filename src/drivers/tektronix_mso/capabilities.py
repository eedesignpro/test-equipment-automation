"""Option-string parsing and runtime capability probes for the MSO 4/5/6.

``*OPT?`` on the MSO 4/5/6 returns verbose per-license bundle text with no
discrete AFG/DVM/DIG code, so it is NOT a usable presence gate (ADR-004).
``parse_opt`` here is a best-effort reader of that bundle text (unit-testable),
while the runtime probes (``probe_afg``/``probe_dvm``/``probe_channel_digital``)
are the AUTHORITATIVE presence check wired into connect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.link import InstrumentLink

# Split the *OPT? text at each ", <NAME>:" boundary without splitting the
# description prose that follows each license entry.
_OPT_SPLIT_RE = re.compile(r",(?=[A-Z0-9][A-Z0-9-]*:)")

# License-name affixes to strip so bundle names normalize to a bare feature.
_PREFIXES = ("SUP5-", "LIC5-", "5-")
_SUFFIXES = ("-PER", "-1Y", "-FL")

# Best-effort bundle expansion. STARTER includes AFG/SREMBD/SRCOMP/PWR;
# ULTIMATE is a superset of STARTER (plus e.g. SV-RFVT). Reality (runtime
# probes) always wins over this table.
_STARTER_CONTENTS = frozenset({"AFG", "SREMBD", "SRCOMP", "PWR"})
BUNDLE_CONTENTS: dict[str, frozenset[str]] = {
    "STARTER": _STARTER_CONTENTS,
    "ULTIMATE": _STARTER_CONTENTS | frozenset({"SVRFVT"}),
}

# ESR bits: command error (CME, bit 5) and execution error (EXE, bit 4). Either
# being set after a subsystem query means that subsystem is not supported.
_ESR_CME = 0b00100000
_ESR_EXE = 0b00010000


@dataclass
class Capabilities:
    """Parsed view of an ``*OPT?`` response (advisory, not authoritative)."""

    raw: str
    options: list[str] = field(default_factory=list)
    bandwidth_mhz: int | None = None
    features: frozenset[str] = frozenset()

    def has(self, feature: str) -> bool:
        """Whether ``feature`` appears in the expanded feature set."""
        return feature.upper() in self.features


def _normalize(name: str) -> str:
    """Strip license prefixes/suffixes from a bundle name."""
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _expand_bundles(options: list[str]) -> frozenset[str]:
    """Expand any known bundle names into their member features."""
    features = set(options)
    for option in options:
        contents = BUNDLE_CONTENTS.get(option)
        if contents:
            features |= set(contents)
    return frozenset(features)


def parse_opt(opt_string: str) -> Capabilities:
    """Parse an ``*OPT?`` response into a :class:`Capabilities`.

    ### Parameters
    1. opt_string : str
        - raw ``*OPT?`` response text
    ### Returns
    1. Capabilities
        - normalized option names, bandwidth (from ``BW5-<MHz>``) and the
          bundle-expanded feature set
    """
    options: list[str] = []
    bandwidth_mhz: int | None = None
    for token in _OPT_SPLIT_RE.split(opt_string or ""):
        name = token.split(":", 1)[0].strip()
        if not name:
            continue
        bw_match = re.search(r"BW5-(\d+)", name)
        if bw_match:
            bandwidth_mhz = int(bw_match.group(1))
        options.append(_normalize(name))
    return Capabilities(
        raw=opt_string or "",
        options=options,
        bandwidth_mhz=bandwidth_mhz,
        features=_expand_bundles(options),
    )


def _probe_query(link: InstrumentLink, query: str) -> bool:
    """Return True if ``query`` succeeds with no CME/EXE event (subsystem present).

    Follows the connect-safe pattern: ``*CLS`` -> query -> ``*ESR?`` CME/EXE
    check -> ``ALLEV?`` drain. Never raises: a failure means "not present".
    """
    try:
        link.write("*CLS")
        link.query(query)
        esr = link.query_int("*ESR?")
        present = not (esr & (_ESR_CME | _ESR_EXE))
        link.query("ALLEV?")
        return present
    except Exception:  # noqa: BLE001 - probe failure means "not present", never raise
        return False


def probe_afg(link: InstrumentLink) -> bool:
    """Probe whether the AFG option is present (``AFG:FUNCTION?``)."""
    return _probe_query(link, "AFG:FUNCTION?")


def probe_dvm(link: InstrumentLink) -> bool:
    """Probe whether the DVM option is present (``DVM:MODE?``)."""
    return _probe_query(link, "DVM:MODE?")


def probe_channel_digital(link: InstrumentLink, channel: int) -> bool:
    """Probe whether a channel currently has a digital probe attached.

    Uses ``CH<x>:PROBETYPE?`` and is safe to call repeatedly - digital probes
    are hot-pluggable, so this is a refreshable check rather than a
    connect-time-frozen flag.

    ### Parameters
    1. link : InstrumentLink
        - the connected link
    2. channel : int
        - analog channel number
    ### Returns
    1. bool
        - True if the channel reports a digital probe
    """
    try:
        link.write("*CLS")
        response = link.query(f"CH{channel}:PROBETYPE?").strip().upper()
        link.query("ALLEV?")
        return "DIG" in response
    except Exception:  # noqa: BLE001 - probe failure means "not digital", never raise
        return False
