"""Build mock-VISA responses from committed live-bench capture traces.

Every capture in ``tests/captures/<instrument>/*.json`` is a recording of real
request/response traffic off a bench instrument (see each subfolder's README).
These helpers turn a trace into an exact-match ``{command: response}`` map so
unit tests replay genuine instrument answers - no synthetic/hand-written
responses. Trace names are resolved relative to ``tests/captures`` and may
include the instrument subfolder (e.g. ``"tek_mso/afg_subsystem_trace.json"``).
"""

from __future__ import annotations

import json
import os

from drivers.tektronix_mso import Tektronix_MSO

from tests.mocks.mock_visa import MockVisaResource

_CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "captures")


def load_trace(name: str) -> list[dict]:
    """Load a capture trace (list of op records) by file name."""
    with open(os.path.join(_CAPTURE_DIR, name)) as fh:
        return json.load(fh)


def response_map(name: str) -> dict[str, str]:
    """Return an exact-match ``{query: response}`` map from a capture trace.

    Later occurrences win, matching the instrument's most-recent answer for a
    repeated query within the session.
    """
    responses: dict[str, str] = {}
    for entry in load_trace(name):
        if entry.get("op") == "query":
            responses[entry["cmd"]] = entry["resp"]
    return responses


def connect_replay(name: str | None = None, extra: dict[str, str] | None = None):
    """Connect a driver to a mock backed by a capture trace's responses.

    ### Parameters
    1. name : str | None
        - capture file to source query responses from (None for none)
    2. extra : dict | None
        - additional/override query responses
    ### Returns
    1. (scope, mock) : tuple[Tektronix_MSO, MockVisaResource]
        - the connected driver and its mock resource (writes cleared)
    """
    responses = response_map(name) if name else {}
    if extra:
        responses.update(extra)
    responses.setdefault("*IDN?", "TEKTRONIX,MSO58,C000000,CF:91.1CT FV:2.6.38.1348")
    mock = MockVisaResource(responses=responses, default_response="")
    scope = Tektronix_MSO("MOCK::INSTR")
    scope.link._open_resource = lambda rm, name: mock
    scope.connect()
    mock.writes.clear()
    return scope, mock
