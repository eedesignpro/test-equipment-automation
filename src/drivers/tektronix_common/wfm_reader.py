"""Parser for Tektronix MSO 4/5/6 internal ``.wfm`` files.

The ``.wfm`` binary layout parsed here is documented in the Tektronix
Reference Waveform File Format manual, 077-0220-11. The format is shared
across instrument families (MSO 4/5/6 today, AWG5200 later), which is why this
parser lives in the ``drivers/tektronix_common`` vendor-common layer rather
than in a single driver package.

Standalone: no dependency on the driver or pyvisa. Ported from the live
``_MSO456_WfmRead`` implementation used on the bench (the earlier, dead first
definition in the legacy driver was dropped).

Public entry points:

- :func:`read_wfm_file` - parse a ``.wfm`` file on disk
- :func:`read_wfm_data` - parse ``.wfm`` bytes already in memory

Both return ``(time_samples, waveforms)`` where ``time_samples`` is a 1-D
``numpy`` array of sample times and ``waveforms`` is a list of 1-D arrays of
scaled vertical values (one per stored trace).
"""

from __future__ import annotations

import io
import struct as st

import numpy as np

# format_enum (ExpDim1.Format) -> (numpy dtype code, human name)
_FORMAT_TO_DTYPE = {
    0: ("i2", "INT16"),
    1: ("i4", "INT32"),
    2: ("u4", "UINT32"),
    3: ("u8", "UINT64"),
    4: ("f4", "FP32"),
    5: ("f8", "FP64"),
    6: ("u1", "UINT8"),
    7: ("i1", "INT8"),
}


class WfmReader:
    """Reads a Tektronix MSO 4/5/6 ``.wfm`` binary into numpy arrays."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.wflist: list[np.ndarray] | None = None
        self.time_samples: np.ndarray | None = None

    def _dbg(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _read_at(self, file_like, offset: int, fmt: str, label: str):
        file_like.seek(offset)
        size = st.calcsize(fmt)
        raw = file_like.read(size)
        if len(raw) != size:
            raise EOFError(
                f"EOF reading {label} at 0x{offset:08X}: expected {size}, got {len(raw)}"
            )
        val = st.unpack(fmt, raw)
        sval = val[0] if len(val) == 1 else val
        self._dbg(f"{label:28s} @0x{offset:08X}  fmt={fmt:>4s}  raw={raw.hex(' ')}  -> {sval}")
        return sval

    def _parse_file_like(self, file_like) -> None:
        # ---- Endianness ----
        bow_le = self._read_at(file_like, 0x0000, "<H", "byte_order_word (<H)")
        bow_be = self._read_at(file_like, 0x0000, ">H", "byte_order_word (>H)")

        if bow_le == 0x0F0F:
            endian = "<"
        elif bow_le == 0xF0F0:
            endian = ">"
        elif bow_be == 0x0F0F:
            endian = ">"
        elif bow_be == 0xF0F0:
            endian = "<"
        else:
            raise ValueError(f"Unrecognized byte order word: le=0x{bow_le:04X} be=0x{bow_be:04X}")

        self._dbg(f"SELECTED endian = {endian}")

        # ---- Header basics ----
        self.ending = self._read_at(file_like, 0x0000, "2s", "ending[0:2]")
        version = self._read_at(file_like, 0x0002, "8s", "version_string")
        self._dbg(f"version_string decoded: {version!r}")

        self.bytes_to_eof = self._read_at(file_like, 0x000B, f"{endian}i", "bytes_to_eof")
        self.point_size = self._read_at(file_like, 0x000F, f"{endian}b", "header_bytes_per_point")
        self.startpoint = self._read_at(file_like, 0x0010, f"{endian}i", "curve_buffer_start")
        self.num_wfms = self._read_at(file_like, 0x0052, f"{endian}I", "num_wfms")
        self.num_fframes = self._read_at(
            file_like, 0x0096, f"{endian}I", "num_acquired_fast_frames"
        )

        # Offsets below are byte offsets from the manual (168/176/240/488 decimal).
        self.dimscale = self._read_at(file_like, 0x00A8, f"{endian}d", "dimscale (ExpDim1.Scale)")
        self.dimoffset = self._read_at(
            file_like, 0x00B0, f"{endian}d", "dimoffset (ExpDim1.Offset)"
        )
        self.time_scale = self._read_at(
            file_like, 0x01E8, f"{endian}d", "time_scale (ImpDim1.Scale)"
        )

        format_enum = self._read_at(file_like, 0x00F0, f"{endian}i", "format_enum (ExpDim1.Format)")

        if int(format_enum) in _FORMAT_TO_DTYPE:
            dtype_code, dtype_name = _FORMAT_TO_DTYPE[int(format_enum)]
            dtype = np.dtype(endian + dtype_code)
            self._dbg(
                f"dtype from format_enum: {dtype_name} -> {dtype.str} (itemsize={dtype.itemsize})"
            )
        else:
            # Fallback: use header bytes/point if the enum is unknown
            bpp = int(self.point_size)
            fallback = {1: "i1", 2: "i2", 4: "f4", 8: "f8"}.get(bpp)
            if fallback is None:
                raise ValueError(
                    f"Unknown format_enum={format_enum} and unsupported bytes/point={bpp}"
                )
            dtype = np.dtype(endian + fallback)
            self._dbg(f"WARNING: unknown format_enum={format_enum}; fallback dtype={dtype.str}")

        bytes_per_point = dtype.itemsize

        # Curve offsets are byte offsets inside the curve buffer:
        # preChargeStart @0x332, dataStart @0x336, postChargeStart @0x33A, postChargeStop @0x33E
        pre_charge_start_b = self._read_at(
            file_like, 0x0332, f"{endian}I", "preChargeStart (bytes)"
        )
        _data_start_b = self._read_at(file_like, 0x0336, f"{endian}I", "dataStart (bytes)")
        _post_charge_start_b = self._read_at(
            file_like, 0x033A, f"{endian}I", "postChargeStart (bytes)"
        )
        post_charge_stop_b = self._read_at(
            file_like, 0x033E, f"{endian}I", "postChargeStop (bytes)"
        )

        # Total bytes of stored curve (pre + data + post)
        curve_bytes = int(post_charge_stop_b - pre_charge_start_b)
        if curve_bytes % bytes_per_point != 0:
            raise ValueError(
                f"curve_bytes={curve_bytes} not aligned to bytes_per_point={bytes_per_point}"
            )

        npts = curve_bytes // bytes_per_point
        self._dbg(f"curve_bytes={curve_bytes}, bytes_per_point={bytes_per_point}, npts={npts}")

        curve_abs = int(self.startpoint) + int(pre_charge_start_b)
        self._dbg(
            f"curve_abs_offset = curve_buffer_start({self.startpoint}) + "
            f"preChargeStart({pre_charge_start_b}) = {curve_abs}"
        )

        file_like.seek(curve_abs)
        raw = file_like.read(curve_bytes)
        if len(raw) != curve_bytes:
            raise EOFError(f"Unexpected EOF reading curve: expected {curve_bytes}, got {len(raw)}")

        y = np.frombuffer(raw, dtype=dtype).astype(np.float64, copy=False)
        y = y * float(self.dimscale) + float(self.dimoffset)
        self.wflist = [y]

        # X axis: implicit dimension scale is per-sample spacing.
        self.time_samples = np.arange(npts, dtype=np.float64) * float(self.time_scale)

    def read_wfm_file(self, wfm_path: str) -> None:
        with open(wfm_path, "rb") as f:
            self._parse_file_like(f)

    def read_wfm_data(self, wfm_data: bytes) -> None:
        self._parse_file_like(io.BytesIO(wfm_data))


def read_wfm_file(wfm_path: str, verbose: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
    """Parse a ``.wfm`` file and return ``(time_samples, waveforms)``."""
    reader = WfmReader(verbose=verbose)
    reader.read_wfm_file(wfm_path)
    return reader.time_samples, reader.wflist


def read_wfm_data(wfm_data: bytes, verbose: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
    """Parse ``.wfm`` bytes in memory and return ``(time_samples, waveforms)``."""
    reader = WfmReader(verbose=verbose)
    reader.read_wfm_data(wfm_data)
    return reader.time_samples, reader.wflist
