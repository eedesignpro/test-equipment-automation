"""Instrument base: a Node that owns and manages its InstrumentLink."""

from __future__ import annotations

from common.link import InstrumentLink
from common.node import Node


class Instrument(Node):
    """Base for a full instrument.

    Subclasses build their entire subsystem tree in ``__init__`` (so that
    attribute autocompletion works before connecting). ``connect()`` opens the
    link, reads ``*IDN?`` into ``self.idn`` and calls the ``_on_connect`` hook;
    ``disconnect()`` calls the ``safe_state`` hook then closes the link.
    """

    def __init__(self, resource_name: str, name: str = ""):
        self.link = InstrumentLink(resource_name)
        super().__init__(self.link, name=name)
        self.resource_name = resource_name
        self.idn: str | None = None

    def connect(self, timeout_ms: int | None = None) -> str:
        """Open the link, query ``*IDN?``, run the ``_on_connect`` hook."""
        self.link.connect(timeout_ms=timeout_ms)
        self.idn = self.link.query("*IDN?").strip()
        self._on_connect(self.idn)
        return self.idn

    def disconnect(self) -> None:
        """Run the ``safe_state`` hook (if connected) then close the link."""
        if self.link.is_connected:
            self.safe_state()
        self.link.disconnect()

    def _on_connect(self, idn: str) -> None:
        """Hook run after *IDN? is read. Default no-op."""

    def safe_state(self) -> None:
        """Hook run before disconnect to put the instrument in a safe state.

        Default no-op.
        """

    def reset(self) -> None:
        """Reset the instrument to its factory default state (``*RST``).

        WARNING: this wipes the current configuration to a known default -
        every channel, horizontal, trigger and subsystem setting is lost.
        Wrap work in ``preserved_setup()`` (driver-level) if the prior state
        must survive.
        """
        self.link.write("*RST")
        self.link.query("*OPC?")

    def check_errors(self) -> None:
        """Raise if the instrument reports queued errors.

        Default drains SYST:ERR? via the link; override for instruments
        with a different error-reporting dialect.
        """
        self.link.check_errors()

    def __enter__(self) -> Instrument:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()
