# Bench captures

Real request/response traffic and artifacts captured from live bench
instruments. These are the fixtures for the unit suite (mock-VISA playback via
`tests/mocks/capture_replay.py`) and parser validation. Per the testing policy
nothing here is synthetic - every byte came off an instrument.

## Layout

Captures are organized by instrument in per-instrument subfolders:

```
tests/captures/
  tek_mso/            Tektronix MSO 4/5/6 (MSO58) - see tek_mso/README.md
  keysight_e364xa/    Keysight/Agilent E364xA PSU family (added iteration 4)
  rigol_dl3000/       Rigol DL3000 e-load family (added iteration 4)
```

`capture_replay.py` resolves trace names relative to this directory, so a test
references a trace by its subfolder-qualified name, e.g.
`connect_replay("tek_mso/afg_subsystem_trace.json")`.

## Format standard for NEW captures (mandatory)

NEW capture subfolders (`keysight_e364xa/`, `rigol_dl3000/`, and any future
instrument) MUST record traces in the raw, replayable per-operation record
format consumable by `tests/mocks/capture_replay.py`:

```json
[
  {"op": "write", "cmd": "VOLT 5.0", "resp": null},
  {"op": "query", "cmd": "VOLT?", "resp": "5.000000E+00"}
]
```

Each record is one instrument operation:

- `op`: `"write"` or `"query"`.
- `cmd`: the exact SCPI string sent.
- `resp`: the raw response string for a `query` (`null`/omitted for a write).

`response_map()` builds an exact-match `{query: response}` map from the `query`
records (later occurrences win). This is the format QA replays into unit tests.

Do NOT commit summary/derived traces (labels + parsed results only) as the sole
capture for a new instrument: those are not replayable and were flagged in the
iteration-3 QA review. The existing `tek_mso/` folder contains a mix of raw and
older summary traces for historical reasons; new work does not add to that mix.
