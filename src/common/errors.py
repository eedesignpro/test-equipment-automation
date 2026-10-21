"""Core exception hierarchy for the instrument framework."""


class InstrumentError(Exception):
    """Base class for all instrument framework errors."""


class NotConnectedError(InstrumentError):
    """Raised when instrument I/O is attempted on a closed/unopened link."""


class HardwareNotPresentError(InstrumentError):
    """Raised when a subsystem/node that is not installed is accessed."""


class InstrumentQueryError(InstrumentError):
    """Raised when a SCPI query response cannot be parsed as expected."""


class InstrumentTimeoutError(InstrumentError):
    """Raised when an instrument operation does not complete within a timeout."""


class InstrumentUnresponsiveError(InstrumentTimeoutError):
    """Raised when an instrument stops answering and likely needs a power cycle.

    Distinct from a routine timeout: it signals that a query got no response at
    all, so the caller (or operator) should stop issuing commands and power
    cycle the instrument rather than retrying. Drivers raise this instead of
    auto-retrying a hung link.
    """


class InstrumentWarning(UserWarning):
    """Warned for non-fatal instrument event-queue entries.

    Emitted (rather than raised) for events the instrument reports that are
    informational rather than errors - e.g. measurement warnings such as
    signal clipping - so a script continues instead of aborting.
    """


class ValueSnappedWarning(UserWarning):
    """Warned when an instrument snaps a requested value to a different one.

    Emitted by ``set_and_verify`` when the numeric readback differs from the
    requested value beyond tolerance. This is informational: the instrument
    applied a nearby legal value rather than the exact request. It never
    interrupts a script unless the caller opts into ``strict`` mode.
    """
