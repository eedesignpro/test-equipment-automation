"""Command line for running and saved runs. No instrument is touched.

python -m sweep review  <run_folder>             print review.md (writes it if missing)
python -m sweep plot    <run_folder>             remake plot.png and plot.svg
python -m sweep condensed <run_folder> [-n 100]  print the condensed table as CSV
python -m sweep compare <name> <run> <run> ...   overlay plot and markdown table
python -m sweep fit     <run_folder> <name> [--max-order 4] [--delay]
python -m sweep spurs   <run_folder> [--known switcher=2e6]   recurring spurs and aliases
python -m sweep status  <run_or_parent> [--plot] [--review]   progress of a running sweep
python -m sweep stop    <run_or_parent>          end after the current point, cleanly
python -m sweep estop   <run_or_parent>          emergency stop: output off at the next step
python -m sweep impedance <run_folder> [name]    |Z|, phase, R, series L and C plot
python -m sweep list    <folder>                 one line per run: status, points, title
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

from sweep.store import Run, safe_name


def _cmd_review(a) -> None:
    from sweep.views import write_views

    p = Path(a.run) / "review.md"
    if a.rewrite or not p.exists():
        write_views(a.run)
    print(p.read_text(encoding="utf-8"))


def _cmd_plot(a) -> None:
    from sweep.plot import plot_harmonics, plot_run

    plot_run(a.run)
    if a.harmonics:
        plot_harmonics(a.run, a.harmonics)
    print(Path(a.run) / "plot.png")


def _cmd_condensed(a) -> None:
    from sweep.views import condensed

    header, rows = condensed(a.run, a.n)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    print(buf.getvalue(), end="")


def _cmd_compare(a) -> None:
    from sweep.plot import overlay_runs
    from sweep.views import compare_review

    out = Path(a.out) if a.out else Path(a.runs[-1]) / f"compare_{safe_name(a.name)}"
    overlay_runs(a.runs, a.name, save_to=out)
    md = compare_review(a.runs, a.name, points=a.points)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"PLOT {out.with_suffix('.png')}")


def _cmd_fit(a) -> None:
    from sweep.fit import auto_fit

    s = Run(a.run).series(a.name)
    fit = auto_fit(s.freqs, s.values, max_order=a.max_order, delay=a.delay)
    print(fit.describe())


def _cmd_spurs(a) -> None:
    from sweep.spurs import spur_report

    known = {}
    for item in a.known or []:
        name, _, hz = item.partition("=")
        known[name] = float(hz)
    print(spur_report(a.run, known=known))


def _latest(folder) -> Path:
    runs = [d for d in Path(folder).iterdir() if (d / "meta.json").exists()]
    if not runs:
        raise SystemExit(f"no run folders in {folder}")
    # Run folder names start with their start time, so the name sorts by age;
    # a file written later into an old run does not make it the newest.
    return max(runs, key=lambda d: d.name)


def _resolve(run) -> Path:
    p = Path(run)
    if (p / "meta.json").exists():
        return p
    try:
        return _running(p)
    except SystemExit:
        return _latest(p)


def _running(folder) -> Path:
    """The run a stop is meant for: the given run, or the one running in a parent folder."""
    import json

    p = Path(folder)
    if (p / "meta.json").exists():
        return p
    live = []
    for d in p.iterdir():
        try:
            if (
                json.loads((d / "progress.json").read_text(encoding="utf-8")).get("status")
                == "running"
            ):
                live.append(d)
        except (OSError, ValueError):
            continue
    if len(live) == 1:
        return live[0]
    if not live:
        raise SystemExit(f"no running sweep in {p}; give the run folder itself")
    names = ", ".join(sorted(d.name for d in live))
    raise SystemExit(f"{len(live)} sweeps are running in {p} ({names}); give the run folder itself")


def _cmd_status(a) -> None:
    import json

    from sweep.views import review

    run = _resolve(a.run)
    prog = run / "progress.json"
    d = json.loads(prog.read_text(encoding="utf-8")) if prog.exists() else {}
    if not d:
        r = Run(run)
        d = {
            "status": r.meta.get("status"),
            "reason": r.meta.get("reason"),
            "points_done": len({x["point"] for x in r.rows}),
        }
    print(f"RUN_FOLDER {run}")
    keys = (
        "status",
        "reason",
        "points_done",
        "points_planned",
        "freq_hz",
        "elapsed_s",
        "remaining_s",
        "updated",
    )
    print("  ".join(f"{k}={d.get(k)}" for k in keys if d.get(k) is not None))
    for name, v in (d.get("latest") or {}).items():
        ph = v.get("phase_deg")
        print(
            f"  latest {name}: {v.get('mag'):.6g} {v.get('unit') or ''}"
            + (f", {ph:.2f} deg" if ph is not None else "")
        )
    if d.get("flags_so_far"):
        print("  flags so far:", ", ".join(f"{k} {n}" for k, n in d["flags_so_far"].items()))
    if a.plot:
        from sweep.plot import plot_run

        plot_run(run)
        print(f"PLOT {run / 'plot.png'}")
    if a.review:
        print(review(Run(run), 12))


def _cmd_stop(a) -> None:
    run = _running(a.run)
    name = "ESTOP" if a.estop else "STOP"
    (run / name).write_text("requested from the command line", encoding="utf-8")
    print(
        f"{name} written to {run}. "
        + (
            "The sweep turns the output off at its next step."
            if a.estop
            else "The sweep ends after the current point."
        )
    )


def _cmd_impedance(a) -> None:
    from sweep.plot import plot_impedance

    plot_impedance(a.run, a.name)
    print(f"PLOT {Path(a.run) / f'impedance_{safe_name(a.name)}.png'}")


def _cmd_list(a) -> None:
    for d in sorted(Path(a.folder).iterdir()):
        if (d / "meta.json").exists():
            r = Run(d)
            n = len({x["point"] for x in r.rows})
            print(f"{d.name}  {r.meta.get('status', '')}  {n} points  {r.title}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sweep", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("review")
    s.add_argument("run")
    s.add_argument("--rewrite", action="store_true")
    s.set_defaults(fn=_cmd_review)
    s = sub.add_parser("plot")
    s.add_argument("run")
    s.add_argument("--harmonics", metavar="SIGNAL")
    s.set_defaults(fn=_cmd_plot)
    s = sub.add_parser("condensed")
    s.add_argument("run")
    s.add_argument("-n", type=int, default=100)
    s.set_defaults(fn=_cmd_condensed)
    s = sub.add_parser("compare")
    s.add_argument("name")
    s.add_argument("runs", nargs="+")
    s.add_argument("--points", type=int, default=25)
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_compare)
    s = sub.add_parser("fit")
    s.add_argument("run")
    s.add_argument("name")
    s.add_argument("--max-order", type=int, default=4)
    s.add_argument("--delay", action="store_true")
    s.set_defaults(fn=_cmd_fit)
    s = sub.add_parser("spurs")
    s.add_argument("run")
    s.add_argument("--known", nargs="*", metavar="NAME=HZ", help="tones to check for aliases")
    s.set_defaults(fn=_cmd_spurs)
    s = sub.add_parser("status")
    s.add_argument("run", help="a run folder, or a parent folder for its newest run")
    s.add_argument("--plot", action="store_true", help="redraw plot.png from the points so far")
    s.add_argument("--review", action="store_true", help="print the review of the points so far")
    s.set_defaults(fn=_cmd_status)
    s = sub.add_parser("stop")
    s.add_argument("run")
    s.set_defaults(fn=_cmd_stop, estop=False)
    s = sub.add_parser("estop")
    s.add_argument("run")
    s.set_defaults(fn=_cmd_stop, estop=True)
    s = sub.add_parser("impedance")
    s.add_argument("run")
    s.add_argument("name", nargs="?", default="Z")
    s.set_defaults(fn=_cmd_impedance)
    s = sub.add_parser("list")
    s.add_argument("folder")
    s.set_defaults(fn=_cmd_list)
    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
