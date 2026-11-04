"""Math across runs: one new run folder from several saved ones.

Old scripts did this by hand, for example coupling ``k12 = (c2_coupled -
c2_alone) / c1_alone`` from three runs. :func:`combine_runs` interpolates the
named quantities of each run onto one frequency grid, evaluates an expression
on them, and writes the result as an ordinary run folder, so the standard plot,
condensed table and review come out for free.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from sweep.interp import common_grid, interpolate
from sweep.store import Run, RunWriter


class _Track(dict):
    def __init__(self, d):
        super().__init__(d)
        self.used = set()

    def __getitem__(self, k):
        self.used.add(k)
        return super().__getitem__(k)


def combine_runs(
    runs: dict,
    expressions: dict[str, Callable | tuple[Callable, str]],
    *,
    folder,
    title: str = "combined",
    points: int | None = None,
    plot: bool = True,
) -> Run:
    """Evaluate ``expressions`` on quantities from several runs.

    ``runs`` maps a short label to a run folder or :class:`Run`. Each
    expression receives a dict keyed ``"label.name"`` (for example
    ``v["coupled.c2"]``) with complex arrays on the common grid, which covers
    the frequencies all runs share. Phase stays valid only when every input
    it reads had a valid phase.
    """
    loaded = {k: (v if isinstance(v, Run) else Run(v)) for k, v in runs.items()}
    series = {}
    for label, run in loaded.items():
        for name in run.names():
            series[f"{label}.{name}"] = run.series(name)
    grid = common_grid([s.freqs for s in series.values()], mode="overlap", points=points)
    data, valid = {}, {}
    for key, s in series.items():
        v, _ = interpolate(s.freqs, s.values, grid)
        data[key] = np.asarray(v, dtype=complex)
        valid[key] = s.phase_valid
    meta = {
        "combined_from": {k: str(r.path) for k, r in loaded.items()},
        "expressions": {},
        "frequencies": {"count": len(grid)},
    }
    w = RunWriter(folder, title, meta)
    rows = []
    for name, spec in expressions.items():
        fn, unit = spec if isinstance(spec, tuple) else (spec, "")
        tv = _Track(data)
        vals = np.asarray(fn(tv), dtype=complex) * np.ones(len(grid))
        ok = all(valid.get(k, False) for k in tv.used)
        w.meta["expressions"][name] = {"unit": unit, "inputs": sorted(tv.used)}
        for i, (f, v) in enumerate(zip(grid, vals)):
            rows.append(
                {
                    "point": i,
                    "freq_cmd": f,
                    "freq_set": f,
                    "freq_used": f,
                    "measurement": "combine",
                    "group": "combined" if ok else "",
                    "signal": name,
                    "kind": "expr",
                    "order": 1,
                    "tone_freq": f,
                    "re": v.real,
                    "im": v.imag,
                    "mag": abs(v),
                    "phase_deg": math.degrees(math.atan2(v.imag, v.real)),
                    "unit": unit,
                    "phase_valid": int(ok),
                    "flags": [],
                }
            )
    w.add_rows(rows)
    w.finish("complete")
    if plot:
        from sweep.plot import plot_run
        from sweep.views import write_views

        plot_run(w.path)
        write_views(w.path)
    return Run(w.path)
