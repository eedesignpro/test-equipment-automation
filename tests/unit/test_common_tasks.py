"""Every driver's "Common tasks" recipes must still match the driver.

Design standard section 10 criterion 1 puts the five most common bench tasks in
each module docstring, and section 9.3 says an agent uses them as starting
code. A recipe that has quietly stopped matching the API is worse than none: it
is wrong at the bench, where finding out is expensive.

These tests resolve each call path in each recipe against the live classes and
check the signature accepts the arguments written. They need no hardware,
because the failure they catch is a rename, not a behaviour.

They deliberately do NOT execute anything. Whether the instrument likes the
values is a bench question; whether the call still exists is not.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re

import pytest

DRIVERS = pathlib.Path(__file__).resolve().parents[2] / "src" / "drivers"

# module path -> the bench handle its recipes are written against
MODULES = {
    "drivers.keysight_e364xa": "psu1",
    "drivers.owon_odp": "psu2",
    "drivers.rigol_dl3000": "eload",
    "drivers.keysight_b2900": "smu",
    "drivers.siglent_sdm": "dmm",
    "drivers.keysight_33120a": "siggen",
    "drivers.keysight_33522a": "awg",
    "drivers.tektronix_mso.driver": "scope",
}

CALL_RE = re.compile(r"^\s*(?:\w+(?:,\s*\w+)*\s*=\s*)?([a-z_][\w.]*)\(")


def _recipes(module_name: str, handle: str) -> list[tuple[str, str]]:
    """Return (call_path, source_line) for every call in the Common tasks block."""
    module = importlib.import_module(module_name)
    doc = module.__doc__ or ""
    if "Common tasks:" not in doc:
        return []
    block = doc.split("Common tasks:", 1)[1]
    found = []
    for line in block.splitlines():
        m = CALL_RE.match(line)
        if not m:
            continue
        path = m.group(1)
        if path.split(".")[0] != handle:
            continue  # a loop variable or a local, not a driver call
        found.append((path, line.strip()))
    return found


def _instrument(module):
    """Build the driver without connecting.

    Section 5.2 requires __init__ to do no I/O, so this is safe offline and
    gives the real object graph, including the nodes __init__ attaches. A
    driver that reached for the bench here would fail this test, which is the
    right place to find out.
    """
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    cls = next(
        getattr(module, n.name)
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and any(getattr(b, "id", "") == "Instrument" for b in n.bases)
    )
    # The MSO requires its resource string; everything else has a pinned default.
    try:
        return cls()
    except TypeError:
        return cls("TCPIP::0.0.0.0::INSTR")


def _resolve(module, handle: str, path: str):
    """Resolve ``handle.a.b.method`` on a built-but-unconnected instrument."""
    obj = _instrument(module)
    parts = path.split(".")[1:]
    for part in parts[:-1]:
        nxt = getattr(obj, part, None)
        if nxt is None:
            # a node attached by connect(); its class is declared as an annotation
            ann = getattr(type(obj), "__annotations__", {}).get(part)
            if ann is None:
                return None
            name = ann.strip("\"'") if isinstance(ann, str) else ann.__name__
            nxt = getattr(module, name, None)
            if nxt is None:
                return None
        obj = nxt
    return getattr(obj, parts[-1], None)


@pytest.mark.parametrize("module_name,handle", sorted(MODULES.items()))
def test_every_driver_has_a_common_tasks_block(module_name: str, handle: str) -> None:
    """Criterion 1: the module docstring lists the common bench tasks."""
    module = importlib.import_module(module_name)
    assert "Common tasks:" in (module.__doc__ or ""), (
        f"{module_name} has no 'Common tasks:' block in its module docstring"
    )


@pytest.mark.parametrize("module_name,handle", sorted(MODULES.items()))
def test_recipe_calls_still_exist(module_name: str, handle: str) -> None:
    """Every method a recipe names is still on the class it is called on."""
    module = importlib.import_module(module_name)
    missing = []
    for path, line in _recipes(module_name, handle):
        if _resolve(module, handle, path) is None:
            missing.append(f"{path}  in: {line}")
    assert not missing, (
        f"{module_name} recipes naming calls that no longer resolve:\n  " + "\n  ".join(missing)
    )


@pytest.mark.parametrize("module_name,handle", sorted(MODULES.items()))
def test_recipe_arguments_match_the_signature(module_name: str, handle: str) -> None:
    """The arguments a recipe passes are ones the method actually accepts.

    Catches the rename that matters most: a keyword argument that quietly
    changed name, which resolves fine and then fails at the bench.
    """
    module = importlib.import_module(module_name)
    wrong = []
    for path, line in _recipes(module_name, handle):
        func = _resolve(module, handle, path)
        if func is None or not callable(func):
            continue
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            continue
        args = line[line.index("(", line.index(path)) :]
        for kw in re.findall(r"[(,]\s*(\w+)\s*=", args):
            if kw in sig.parameters:
                continue
            if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
                continue
            wrong.append(f"{path} has no parameter {kw!r}  in: {line}")
    assert not wrong, (
        f"{module_name} recipes passing arguments the method does not take:\n  "
        + "\n  ".join(wrong)
    )
