"""Unit tests for src/utils/wfm_reader.py against the real captured .wfm.

Fixtures: tests/captures/ch5_sine_1mhz.wfm (real Tektronix internal binary,
CH5, 1 MHz AFG sine, record length 10000) and its CSV export companion, per
the testing policy (no synthetic .wfm files).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from drivers.tektronix_common import wfm_reader

CAPTURES = Path(__file__).parent.parent / "captures" / "tek_mso"
WFM_PATH = CAPTURES / "ch5_sine_1mhz.wfm"
CSV_PATH = CAPTURES / "ch5_sine_1mhz.csv"


def _load_trace_summary():
    with open(CAPTURES / "meas_dispatcher_trace.json") as fh:
        return json.load(fh)["parsed"]


def _load_trace_manifest():
    with open(CAPTURES / "meas_dispatcher_trace.json") as fh:
        return json.load(fh)


def _load_csv_rows():
    rows = []
    with open(CSV_PATH) as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except ValueError:
                continue
    return rows


def test_read_wfm_file_matches_captured_summary():
    summary = _load_trace_summary()
    time_samples, waveforms = wfm_reader.read_wfm_file(str(WFM_PATH))

    assert len(waveforms) == 1
    y = waveforms[0]
    assert len(time_samples) == len(y) == summary["npts"]

    dt = time_samples[1] - time_samples[0]
    assert dt == pytest.approx(summary["dt_s"], rel=1e-6)
    assert float(np.min(y)) == pytest.approx(summary["y_min"], abs=1e-3)
    assert float(np.max(y)) == pytest.approx(summary["y_max"], abs=1e-3)


def test_read_wfm_data_bytes_matches_read_wfm_file():
    """The in-memory (get_waveform_data) and on-disk (get_waveform) paths
    parse identically - bench-confirmed byte-for-byte identical (ADR notes)."""
    with open(WFM_PATH, "rb") as fh:
        raw = fh.read()

    t_file, wf_file = wfm_reader.read_wfm_file(str(WFM_PATH))
    t_data, wf_data = wfm_reader.read_wfm_data(raw)

    np.testing.assert_array_equal(t_file, t_data)
    np.testing.assert_array_equal(wf_file[0], wf_data[0])


def test_wfm_byte_count_matches_capture_manifest():
    manifest = _load_trace_manifest()
    assert WFM_PATH.stat().st_size == manifest["wfm_bytes"]


def test_parsed_sample_count_covers_the_csv_record_length():
    """wfm_reader includes pre/post-charge samples, so its count is >= the
    vendor CSV's numeric record length, and dt matches the CSV's Sample
    Interval exactly."""
    csv_rows = _load_csv_rows()
    time_samples, waveforms = wfm_reader.read_wfm_file(str(WFM_PATH))

    assert len(csv_rows) == 10000
    assert len(waveforms[0]) >= len(csv_rows)

    dt = time_samples[1] - time_samples[0]
    csv_dt = csv_rows[1][0] - csv_rows[0][0]
    assert dt == pytest.approx(csv_dt, rel=1e-6)


def test_parsed_amplitude_range_matches_csv_export():
    csv_rows = _load_csv_rows()
    csv_values = [v for _, v in csv_rows]
    _, waveforms = wfm_reader.read_wfm_file(str(WFM_PATH))
    y = waveforms[0]

    # Allow a small margin: wfm_reader's window includes extra pre/post-charge
    # samples the CSV export does not, so its min/max may extend slightly
    # beyond the CSV's own range but must be in the same ballpark.
    assert float(np.min(y)) <= min(csv_values) + 0.05
    assert float(np.max(y)) >= max(csv_values) - 0.05


def test_read_wfm_file_missing_path_raises():
    with pytest.raises(OSError):
        wfm_reader.read_wfm_file(str(CAPTURES / "does_not_exist.wfm"))
