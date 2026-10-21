"""Tektronix vendor-common layer.

Code shared across Tektronix instrument families (not instrument-agnostic, so
it lives under ``drivers/`` rather than ``common/``). Currently the ``.wfm``
internal-waveform parser, which is a Tektronix file format used by the MSO
4/5/6 today and the AWG5200 later.
"""

from drivers.tektronix_common import wfm_reader

__all__ = ["wfm_reader"]
