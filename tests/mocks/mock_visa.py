"""A drop-in mock of a pyvisa MessageBasedResource for hardware-free tests."""

from __future__ import annotations

from collections.abc import Callable


class MockVisaResource:
    """Records every write and plays back scripted query responses.

    Query responses come from ``responses``: either an exact-string value or a
    callable taking the query string and returning the response string. Missing
    keys fall back to ``default_response``. All writes are appended (verbatim)
    to ``writes`` in order.
    """

    def __init__(
        self,
        responses: dict[str, str | Callable[[str], str]] | None = None,
        default_response: str = "",
    ):
        self.responses = dict(responses or {})
        self.default_response = default_response
        self.writes: list[str] = []
        self.raw_writes: list[bytes] = []
        self.read_buffer: list[bytes] = []
        self.timeout = 2000
        self.closed = False
        self.cleared = False

    def set_response(self, query: str, response: str | Callable[[str], str]) -> None:
        self.responses[query] = response

    def queue_read(self, data: bytes) -> None:
        """Queue bytes to be returned by the next ``read_raw`` call."""
        self.read_buffer.append(data)

    def write(self, message: str, **kwargs) -> int:
        self.writes.append(message)
        return len(message)

    def write_raw(self, data: bytes) -> int:
        self.raw_writes.append(bytes(data))
        return len(data)

    def read(self, **kwargs) -> str:
        return self._resolve(self.writes[-1] if self.writes else "")

    def read_raw(self, size: int | None = None) -> bytes:
        if self.read_buffer:
            return self.read_buffer.pop(0)
        return b""

    def read_bytes(self, count: int, **kwargs) -> bytes:
        if self.read_buffer:
            return self.read_buffer.pop(0)
        return b""

    def query(self, message: str, **kwargs) -> str:
        self.writes.append(message)
        return self._resolve(message)

    def _resolve(self, message: str) -> str:
        resp = self.responses.get(message, self.default_response)
        if callable(resp):
            return resp(message)
        return resp

    def clear(self) -> None:
        self.cleared = True

    def close(self) -> None:
        self.closed = True
