"""Node base: a subsystem/front-panel entity that shares an InstrumentLink."""

from __future__ import annotations

from common.errors import HardwareNotPresentError
from common.link import InstrumentLink


class Node:
    """A subsystem of an instrument (e.g. a channel or the trigger block).

    Holds a reference to the shared :class:`InstrumentLink` and a front-panel
    ``name`` (e.g. ``"CH3"``). When ``installed`` is False every I/O helper
    raises :class:`HardwareNotPresentError`, so unpopulated subsystems fail
    loudly rather than issuing SCPI silently.
    """

    def __init__(self, link: InstrumentLink, name: str = ""):
        self.link = link
        self.name = name
        self.installed = True

    def _require_installed(self) -> None:
        if not self.installed:
            label = self.name or type(self).__name__
            raise HardwareNotPresentError(f"{label} is not installed on this instrument")

    def write(self, message: str, **kwargs) -> int:
        self._require_installed()
        return self.link.write(message, **kwargs)

    def query(self, message: str, **kwargs) -> str:
        self._require_installed()
        return self.link.query(message, **kwargs)

    def query_float(self, cmd: str, **kwargs) -> float:
        self._require_installed()
        return self.link.query_float(cmd, **kwargs)

    def query_int(self, cmd: str, **kwargs) -> int:
        self._require_installed()
        return self.link.query_int(cmd, **kwargs)

    def query_bool(self, cmd: str, **kwargs) -> bool:
        self._require_installed()
        return self.link.query_bool(cmd, **kwargs)

    def write_choice(self, cmd: str, value: str, mapping: dict[str, str]) -> None:
        self._require_installed()
        self.link.write_choice(cmd, value, mapping)

    def set_and_verify(
        self,
        cmd: str,
        value: float | str,
        *,
        rel_tol: float | None = None,
        abs_tol: float | None = None,
        strict: bool = False,
    ) -> float | str:
        self._require_installed()
        kwargs: dict[str, float] = {}
        if rel_tol is not None:
            kwargs["rel_tol"] = rel_tol
        if abs_tol is not None:
            kwargs["abs_tol"] = abs_tol
        return self.link.set_and_verify(cmd, value, strict=strict, **kwargs)

    def read_bytes(self, count: int, **kwargs) -> bytes:
        self._require_installed()
        return self.link.read_bytes(count, **kwargs)

    def read_raw(self, size: int | None = None) -> bytes:
        self._require_installed()
        return self.link.read_raw(size)

    def read_binary_block(self) -> bytes:
        self._require_installed()
        return self.link.read_binary_block()

    def write_binary_block(self, cmd: str, data: bytes) -> int:
        self._require_installed()
        return self.link.write_binary_block(cmd, data)
