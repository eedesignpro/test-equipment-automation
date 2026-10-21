"""Tektronix MSO 4/5/6-series oscilloscope driver package.

Public import path::

    from drivers.tektronix_mso import Tektronix_MSO

That is the same shape a single-file driver gives at script level (compare
``from drivers.keysight_e364xa import Keysight_E364xA``), which is the point:
the package split is invisible to a caller. The implementation lives in
:mod:`drivers.tektronix_mso.driver`, and private node classes stay there rather
than being re-exported here (ADR-021).
"""

from drivers.tektronix_mso.driver import Tektronix_MSO

__all__ = ["Tektronix_MSO"]
