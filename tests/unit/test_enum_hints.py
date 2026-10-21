"""Every public method taking an enum must say what to type to find its members.

A signature reading ``function: Function | str`` does not tell an operator what
to type, and an enum nobody can reach by autocomplete may as well be a string.
The docstring carries an ``Enum arg:`` line naming the path from the instrument
handle; see tools/add_enum_hints.py and docs/hardware-not-software.md.

These tests keep the hint honest. They check that the hint exists, that it names
the enum the signature actually takes, and that the path it tells you to type
really resolves on the class. Without them the hint drifts the first time an
argument changes type, and a wrong hint is worse than none.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

DRIVERS = pathlib.Path(__file__).resolve().parents[2] / "src" / "drivers"

# Drivers whose enums have been converted to the instrument-handle convention.
# Add a driver here as it is converted.
CONVERTED = {
    "keysight_33120a.py": "siggen",
    "keysight_33522a.py": "awg",
    "keysight_b2900.py": "smu",
    "tektronix_mso/driver.py": "scope",
}

HINT_RE = re.compile(r"Enum arg:\s*(?:(\w+)\s*=\s*)?<(\w+)>\.(\w+)\.<TAB>")
CONT_RE = re.compile(r"^\s+(\w+)\s*=\s*<(\w+)>\.(\w+)\.<TAB>", re.MULTILINE)


def _module_enums(tree: ast.Module, path: pathlib.Path | None = None) -> set[str]:
    found = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and any(getattr(b, "id", "") == "ScpiEnum" for b in n.bases)
    }
    # A driver may keep its enums in a sibling module; the MSO does.
    if path is not None and (path.parent / "enums.py").exists():
        found |= _module_enums(ast.parse((path.parent / "enums.py").read_text(encoding="utf-8")))
    return found


def _enum_args(method: ast.FunctionDef, enums: set[str]) -> list[tuple[str, str]]:
    found = []
    for a in method.args.args + method.args.kwonlyargs:
        ann = ast.unparse(a.annotation) if a.annotation else ""
        for e in sorted(enums, key=len, reverse=True):
            if re.search(rf"\b{e}\b", ann):
                found.append((a.arg, e))
                break
    return found


def _public_methods_taking_enums(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    enums = _module_enums(tree, path)
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for m in cls.body:
            if not isinstance(m, ast.FunctionDef) or m.name.startswith("_"):
                continue
            pairs = _enum_args(m, enums)
            if pairs:
                yield cls.name, m, pairs


@pytest.mark.parametrize("filename,handle", sorted(CONVERTED.items()))
def test_every_enum_argument_has_a_hint(filename: str, handle: str) -> None:
    """A method taking an enum names the path to its members."""
    path = DRIVERS / filename
    missing = []
    for cls_name, method, pairs in _public_methods_taking_enums(path):
        doc = ast.get_docstring(method, clean=False) or ""
        if "Enum arg:" not in doc:
            missing.append(f"{cls_name}.{method.name} takes {[e for _, e in pairs]}")
    assert not missing, "methods taking an enum with no 'Enum arg:' hint:\n  " + "\n  ".join(
        missing
    )


@pytest.mark.parametrize("filename,handle", sorted(CONVERTED.items()))
def test_hints_name_the_right_enum_and_handle(filename: str, handle: str) -> None:
    """The hint names the enum the signature takes, and the bench handle."""
    path = DRIVERS / filename
    wrong = []
    for cls_name, method, pairs in _public_methods_taking_enums(path):
        doc = ast.get_docstring(method, clean=False) or ""
        hinted = {(m.group(3)) for m in HINT_RE.finditer(doc)}
        hinted |= {m.group(3) for m in CONT_RE.finditer(doc)}
        handles = {m.group(2) for m in HINT_RE.finditer(doc)}
        expected = {e for _, e in pairs}
        if hinted != expected:
            wrong.append(
                f"{cls_name}.{method.name}: hint says {hinted}, signature takes {expected}"
            )
        if handles and handles != {handle}:
            wrong.append(f"{cls_name}.{method.name}: handle {handles}, expected {{'{handle}'}}")
    assert not wrong, "hints disagreeing with their signature:\n  " + "\n  ".join(wrong)


@pytest.mark.parametrize("filename,handle", sorted(CONVERTED.items()))
def test_the_hinted_path_actually_resolves(filename: str, handle: str) -> None:
    """``<handle>.EnumName`` must exist on the instrument class, uninstantiated.

    This is the check that would have caught the B2902B, whose enums are nested
    inside node classes so the instrument-handle path does not exist at all.
    """
    import importlib

    module = importlib.import_module("drivers." + filename[:-3].replace("/", "."))
    tree = ast.parse((DRIVERS / filename).read_text(encoding="utf-8"))
    enums = _module_enums(tree, DRIVERS / filename)
    # The instrument class is the one subclassing Instrument. Selecting the
    # first public class instead picks up the returned dataclasses.
    instrument = next(
        getattr(module, n.name)
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and any(getattr(b, "id", "") == "Instrument" for b in n.bases)
    )
    unreachable = [e for e in sorted(enums) if not hasattr(instrument, e)]
    assert not unreachable, (
        f"{instrument.__name__} enums a hint could point at but that do not resolve "
        f"from the instrument handle: {unreachable}"
    )
