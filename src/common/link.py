"""InstrumentLink: owns the single pyvisa connection for one instrument."""

from __future__ import annotations

import contextlib
import math
import re
import warnings

import pyvisa as visa
from pyvisa.resources import MessageBasedResource

from common.errors import (
    InstrumentQueryError,
    NotConnectedError,
    ValueSnappedWarning,
)

# Default snap tolerance for set_and_verify: 0.1% relative, tiny absolute floor
# for values near zero. Loose enough to ignore float/format rounding, tight
# enough to surface a real instrument snap (e.g. a scale coerced to a step).
_SNAP_REL_TOL = 1e-3
_SNAP_ABS_TOL = 1e-9

_FLOAT_RE = re.compile(
    r"""
    [+-]?
    (?:
        (?:\d+\.\d*) |
        (?:\.\d+)   |
        (?:\d+)
    )
    (?:[eE][+-]?\d+)?
    """,
    re.VERBOSE,
)

_INT_RE = re.compile(
    r"""
    [+-]?
    \d+
    (?!\.)
    """,
    re.VERBOSE,
)


class InstrumentLink:
    """Owns the one pyvisa MessageBasedResource for an instrument.

    The constructor does not open the connection; call ``connect()`` (or use
    the object as a context manager) to open it. Native pyvisa access is via
    the ``visa`` property; there is no ``__getattr__`` delegation.
    """

    def __init__(self, resource_name: str, timeout_ms: int = 3000):
        """
        Parameters
        ----------
        resource_name : str
            full pyvisa instrument connection string
        timeout_ms : int
            default I/O timeout applied on connect, in milliseconds

        This link does not pace I/O. Inter-command timing is a property of a
        specific instrument and its transport, so a driver that needs it owns
        it in its own link subclass (see ``_PsuSerialLink`` in
        ``drivers/keysight_e364xa.py`` and ``_SigGenSerialLink`` in
        ``drivers/keysight_33120a.py``). See ADR-033.
        """
        self.resource_name = resource_name
        self.default_timeout_ms = timeout_ms
        self._instr: MessageBasedResource | None = None

    # ---------- connection lifecycle ----------

    def connect(self, timeout_ms: int | None = None) -> None:
        """Open the resource via a ResourceManager, clear it, set the timeout."""
        rm = visa.ResourceManager()
        self._instr = self._open_resource(rm, self.resource_name)
        self._instr.clear()
        self._instr.timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms

    def _open_resource(self, rm, resource_name: str) -> MessageBasedResource:
        """Open the underlying resource. Seam for tests to inject a mock."""
        return rm.open_resource(resource_name)

    def disconnect(self) -> None:
        """Close the resource if open."""
        if self._instr is not None:
            self._instr.close()
            self._instr = None

    @property
    def is_connected(self) -> bool:
        return self._instr is not None

    @property
    def visa(self) -> MessageBasedResource:
        """The underlying pyvisa resource. Raises when not connected."""
        if self._instr is None:
            raise NotConnectedError(f"Instrument {self.resource_name!r} is not connected")
        return self._instr

    def __enter__(self) -> InstrumentLink:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ---------- basic I/O ----------

    def write(self, message: str, **kwargs) -> int:
        return self.visa.write(message, **kwargs)

    def query(self, message: str, **kwargs) -> str:
        return self.visa.query(message, **kwargs)

    def query_float(self, cmd: str, **kwargs) -> float:
        """Query a float, tolerating quotes/junk; raise on parse failure."""
        resp = self.query(cmd, **kwargs)
        try:
            return float(resp)
        except (TypeError, ValueError):
            pass

        cleaned = resp.strip().strip('"').strip("'")
        match = _FLOAT_RE.search(cleaned)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                pass

        raise InstrumentQueryError(f"query_float failed\n  Command : {cmd}\n  Response: {resp!r}")

    def query_int(self, cmd: str, **kwargs) -> int:
        """Query an int, tolerating quotes/junk; raise on parse failure."""
        resp = self.query(cmd, **kwargs)
        try:
            return int(resp)
        except (TypeError, ValueError):
            pass

        cleaned = resp.strip().strip('"').strip("'")
        match = _INT_RE.search(cleaned)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                pass

        raise InstrumentQueryError(f"query_int failed\n  Command : {cmd}\n  Response: {resp!r}")

    def query_bool(self, cmd: str, **kwargs) -> bool:
        """Query a boolean. Accepts 1/0, ON/OFF, TRUE/FALSE (any case)."""
        resp = self.query(cmd, **kwargs)
        token = resp.strip().strip('"').strip("'").upper()
        if token in ("1", "ON", "TRUE"):
            return True
        if token in ("0", "OFF", "FALSE"):
            return False
        match = _INT_RE.search(token)
        if match:
            return bool(int(match.group(0)))
        raise InstrumentQueryError(f"query_bool failed\n  Command : {cmd}\n  Response: {resp!r}")

    def write_choice(self, cmd: str, value: str, mapping: dict[str, str]) -> None:
        """Write ``cmd <scpi>`` where ``scpi`` = mapping[value].

        Raises ValueError listing the valid choices when ``value`` is not a key.
        """
        if value not in mapping:
            valid = ", ".join(mapping)
            raise ValueError(f"Invalid value {value!r} for {cmd!r}. Valid choices: {valid}")
        self.write(f"{cmd} {mapping[value]}")

    def set_and_verify(
        self,
        cmd: str,
        value: float | str,
        *,
        rel_tol: float = _SNAP_REL_TOL,
        abs_tol: float = _SNAP_ABS_TOL,
        strict: bool = False,
    ) -> float | str:
        """Write ``cmd value``, query ``cmd?`` and return the readback.

        Numeric ``value``: the float readback is compared against the request.
        When the instrument snaps it to a different value (outside ``rel_tol`` /
        ``abs_tol``) a :class:`ValueSnappedWarning` is emitted and the applied
        value returned. This NEVER raises by default; pass ``strict=True`` to
        raise :class:`ValueError` on a snap instead.

        String ``value`` is treated as an instrument token: it is written
        verbatim, no numeric snap-check is applied, and the raw (stripped)
        query response is returned.
        """
        if isinstance(value, str):
            self.write(f"{cmd} {value}")
            return self.query(f"{cmd}?").strip()

        self.write(f"{cmd} {value:g}")
        applied = self.query_float(f"{cmd}?")
        if not math.isclose(applied, value, rel_tol=rel_tol, abs_tol=abs_tol):
            message = (
                f"{cmd}: requested {value:g} but instrument applied {applied:g} "
                f"(snapped beyond tolerance rel={rel_tol:g}, abs={abs_tol:g})"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(ValueSnappedWarning(message), stacklevel=2)
        return applied

    # ---------- binary blocks (IEEE-488.2 definite length) ----------

    def write_binary_block(self, cmd: str, data: bytes) -> int:
        """Send ``cmd`` followed by an IEEE-488.2 definite-length binary block."""
        header = self._format_binary_header(len(data))
        payload = cmd.encode("ascii") + b" " + header + bytes(data)
        return self.visa.write_raw(payload)

    def read_binary_block(self) -> bytes:
        """Read and unwrap an IEEE-488.2 definite-length binary block."""
        raw = self.visa.read_raw()
        return self._parse_binary_block(raw)

    def read_bytes(self, count: int, **kwargs) -> bytes:
        """Read exactly ``count`` raw bytes from the resource."""
        return self.visa.read_bytes(count, **kwargs)

    def read_raw(self, size: int | None = None) -> bytes:
        """Read raw bytes from the resource (no block-header parsing)."""
        return self.visa.read_raw(size)

    @staticmethod
    def _format_binary_header(n_bytes: int) -> bytes:
        digits = str(n_bytes)
        return b"#" + str(len(digits)).encode("ascii") + digits.encode("ascii")

    @staticmethod
    def _parse_binary_block(raw: bytes) -> bytes:
        start = raw.find(b"#")
        if start < 0:
            raise InstrumentQueryError("No binary block header ('#') found in response")
        try:
            ndigits = int(chr(raw[start + 1]))
        except (IndexError, ValueError) as exc:
            raise InstrumentQueryError("Malformed binary block length header") from exc
        len_start = start + 2
        len_end = len_start + ndigits
        try:
            nbytes = int(raw[len_start:len_end])
        except ValueError as exc:
            raise InstrumentQueryError("Malformed binary block byte count") from exc
        data_start = len_end
        payload = raw[data_start : data_start + nbytes]
        if len(payload) < nbytes:
            raise InstrumentQueryError(
                f"Truncated binary block: header declared {nbytes} bytes "
                f"but only {len(payload)} were present"
            )
        return payload

    # ---------- timeout + errors ----------

    @contextlib.contextmanager
    def temporary_timeout(self, timeout_ms: int):
        """Temporarily set the pyvisa timeout (ms), restoring it on exit."""
        instr = self.visa
        old = instr.timeout
        instr.timeout = timeout_ms
        try:
            yield
        finally:
            instr.timeout = old

    def clear_status(self) -> None:
        """Clear the status byte, event queue and event registers (``*CLS``)."""
        self.write("*CLS")

    def wait_for_complete(self, timeout_ms: int | None = None) -> None:
        """Block until all pending operations complete (``*WAI`` + ``*OPC?``).

        Sends ``*WAI`` (which defers later commands until the current
        operations finish) then queries ``*OPC?`` (which returns ``1`` only
        once the operation-complete flag is set).

        ### Parameters
        1. timeout_ms : int (default None)
            - temporary I/O timeout for the wait; the previous timeout is
              restored afterward. ``None`` uses the current timeout.
        """
        if timeout_ms is None:
            self.write("*WAI")
            self.query("*OPC?")
            return
        with self.temporary_timeout(timeout_ms):
            self.write("*WAI")
            self.query("*OPC?")

    def check_errors(self) -> None:
        """Drain SYST:ERR? until a '0,...' code; raise collecting any errors."""
        errors = []
        for _ in range(100):
            resp = self.query("SYST:ERR?").strip()
            code = resp.split(",", 1)[0].strip()
            if code.lstrip("+") == "0" or code == "0":
                break
            errors.append(resp)
        if errors:
            raise InstrumentQueryError("Instrument reported errors:\n  " + "\n  ".join(errors))
