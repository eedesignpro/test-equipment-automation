# SPDX-License-Identifier: Apache-2.0
# Copyright: see the NOTICE file.
"""Run a driver's whole read and write-back surface against a real instrument.

    python tools/api_sweep.py drivers.siglent_sdg3000x:Siglent_SDG3000X "TCPIP::192.0.2.1::INSTR" out.jsonl

Every ``get_*`` with no required argument runs. Every ``set_x`` with one required argument and a
matching ``get_x`` writes the value it read and compares the return. Calls that the present mode
refuses appear in the file with the reply. Nothing turns an output on by intent, but a write-back
restores what it read, so run this with the outputs off and nothing attached that a level can harm.
Do not run it on an instrument in use: it starts from no reset and leaves written settings as read.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import time
import warnings

from common.node import Node

SKIP_SET = {"set_front_panel_keys_state", "set_output_state_all"}


def nodes(obj, path, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield path, obj
    for name, value in vars(obj).items():
        if not name.startswith("_") and isinstance(value, Node):
            yield from nodes(value, f"{path}.{name}", seen)


def required(fn):
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return [p for p in inspect.signature(fn).parameters.values() if p.default is p.empty and p.kind in kinds]


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))
    return a == b


def call(inst, out, counts, method, kind, fn, *args):
    t0 = time.time()
    rec = {"method": method, "kind": kind, "args": repr(args)[:200]}
    value, ok = None, False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = fn(*args)
        ok = True
        rec.update(result="ok", value=repr(value)[:300], warnings=[str(w.message)[:200] for w in caught])
    except Exception as exc:  # noqa: BLE001
        rec.update(result=type(exc).__name__, error=str(exc)[:400])
    rec["ms"] = round((time.time() - t0) * 1000)
    if hasattr(inst, "get_errors"):
        try:
            rec["queue"] = inst.get_errors()
        except Exception as exc:  # noqa: BLE001
            rec["queue"] = [f"read failed: {exc!r}"[:200]]
    counts[rec["result"]] = counts.get(rec["result"], 0) + 1
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return value, ok


def sweep(inst, root, out):
    counts, mismatches = {}, []
    for path, node in list(nodes(inst, root, set())):
        names = [n for n in dir(type(node)) if not n.startswith("_")]
        for name in names:
            fn = getattr(node, name)
            if name.startswith("get_") and callable(fn) and not required(fn):
                call(inst, out, counts, f"{path}.{name}", "get", fn)
        for name in names:
            if not name.startswith("set_") or name in SKIP_SET or f"get_{name[4:]}" not in names:
                continue
            setter, getter = getattr(node, name), getattr(node, f"get_{name[4:]}")
            if len(required(setter)) != 1 or required(getter):
                continue
            value, ok = call(inst, out, counts, f"{path}.get_{name[4:]}", "get_for_set", getter)
            if ok:
                back, ok = call(inst, out, counts, f"{path}.{name}", "set", setter, value)
                if ok and not same(back, value):
                    mismatches.append(f"{path}.{name}: wrote {value!r}, returned {back!r}")
    return counts, mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("driver", help="module:Class, for example drivers.keysight_33120a:Keysight_33120A")
    ap.add_argument("resource", help="the VISA address, read off the instrument")
    ap.add_argument("out", help="result file, one JSON record per call")
    a = ap.parse_args()
    module, cls = a.driver.split(":")
    inst = getattr(importlib.import_module(module), cls)(a.resource)
    inst.connect()
    try:
        with open(a.out, "w", encoding="utf-8") as out:
            counts, mismatches = sweep(inst, cls.lower(), out)
    finally:
        inst.disconnect()
    print(counts)
    print(f"write-back mismatches: {len(mismatches)}")
    print("\n".join(mismatches))


if __name__ == "__main__":
    main()
