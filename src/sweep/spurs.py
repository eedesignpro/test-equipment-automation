"""Recurring spurs across a run, and where known tones alias.

The old waveform analysis found spurs that come and go as the adaptive
timebase changes the sample rate. :func:`spur_report` reads ``spurs.csv``,
groups spurs by frequency, and checks where each ``known`` tone (for example a
switching frequency) lands at each point's sample rate, so a spur that moves
with the sample rate reads as an alias, not as the device.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

from sweep.analysis import alias_frequency
from sweep.store import Run


def _bucket(f: float) -> float:
    """Round to three significant figures, so one spur groups across points."""
    if f <= 0:
        return 0.0
    digits = 2 - int(math.floor(math.log10(f)))
    return round(f, digits)


def spur_report(
    run, *, known: dict[str, float] | None = None, tolerance: float = 0.01, top: int = 15
) -> str:
    """Markdown table of recurring spurs, with the known tone each one matches."""
    run = run if isinstance(run, Run) else Run(run)
    rates = {r["point"]: r["sample_rate"] for r in run.rows if r["kind"] == "signal"}
    groups = defaultdict(list)
    p = run.path / "spurs.csv"
    if p.exists():
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                f = float(r["spur_freq"])
                groups[(r["signal"], _bucket(f))].append((int(r["point"]), float(r["dbc"]), f))
    out = [
        f"## Recurring spurs in {run.title}",
        "",
        "| Signal | Spur near (Hz) | Points | Worst (dBc) | Matches |",
        "| --- | --- | --- | --- | --- |",
    ]
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top]
    for (sig, key), hits in ranked:
        match = []
        for name, ft in (known or {}).items():
            n = sum(
                1
                for pt, _, f in hits
                if rates.get(pt) and abs(alias_frequency(ft, rates[pt]) - f) <= tolerance * f
            )
            if n:
                match.append(f"{name} alias at {n} points")
        worst = max(h[1] for h in hits)
        out.append(f"| {sig} | {key:.6g} | {len(hits)} | {worst:.1f} | {', '.join(match)} |")
    if not ranked:
        out.append("| none | | | | |")
    return "\n".join(out) + "\n"
