"""Add an ``Enum arg:`` hint to every public method that takes an enum.

The operator reads a docstring at bench scale, not driver scale. A signature
saying ``function: Function | str`` does not say what to TYPE to see the
options, and an enum you cannot find by autocomplete may as well be a string.
The hint gives the shortest path from the instrument handle to the members.

The placeholder is always the instrument handle, whatever node the method sits
on, because an operator working with an instrument has the whole instrument in
scope; see docs/hardware-not-software.md section 2.

    Enum arg: <awg>.Function.<TAB>
    Enum arg: shape  = <awg>.ModulationShape.<TAB>
              source = <awg>.ModulationSource.<TAB>

The hint goes after the summary line, so the api-map generator (which takes the
first line only) never sees it. It costs an agent nothing and an operator one
hover.

Dry run by default; pass --apply to write.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

# Bench handle per driver, matching the handle names in a user bench file.
HANDLE = {
    "keysight_e364xa.py": "psu1",
    "owon_odp.py": "psu2",
    "rigol_dl3000.py": "eload",
    "keysight_b2900.py": "smu",
    "siglent_sdm.py": "dmm",
    "keysight_33120a.py": "siggen",
    "keysight_33522a.py": "awg",
    "driver.py": "scope",  # tektronix_mso/driver.py
}


def enum_args(method: ast.FunctionDef, enums: set[str]) -> list[tuple[str, str]]:
    """Return (argument_name, enum_class) for each enum-annotated argument."""
    found = []
    for a in method.args.args + method.args.kwonlyargs:
        ann = ast.unparse(a.annotation) if a.annotation else ""
        for e in sorted(enums, key=len, reverse=True):
            if re.search(rf"\b{e}\b", ann):
                found.append((a.arg, e))
                break
    return found


def build_hint(pairs: list[tuple[str, str]], handle: str, indent: str) -> list[str]:
    """Render the hint block for one method."""
    if len(pairs) == 1:
        return [f"{indent}Enum arg: <{handle}>.{pairs[0][1]}.<TAB>\n"]
    width = max(len(a) for a, _ in pairs)
    lines = []
    for i, (arg, enum) in enumerate(pairs):
        label = "Enum arg: " if i == 0 else "          "
        lines.append(f"{indent}{label}{arg:<{width}} = <{handle}>.{enum}.<TAB>\n")
    return lines


def process(path: pathlib.Path, apply: bool) -> int:
    handle = HANDLE.get(path.name)
    if handle is None:
        print(f"  no bench handle known for {path.name}; skipped")
        return 0
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    enums = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and any(getattr(b, "id", "") == "ScpiEnum" for b in n.bases)
    }
    # A driver may keep its enums in a sibling module (the MSO does, because its
    # driver file is already the largest in the repo).
    sibling = path.parent / "enums.py"
    if sibling.exists():
        enums |= {
            n.name
            for n in ast.walk(ast.parse(sibling.read_text(encoding="utf-8")))
            if isinstance(n, ast.ClassDef)
            and any(getattr(b, "id", "") == "ScpiEnum" for b in n.bases)
        }
    if not enums:
        return 0

    edits = []  # (insert_at_line_index, hint_lines, description)
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for m in cls.body:
            if not isinstance(m, ast.FunctionDef) or m.name.startswith("_"):
                continue
            pairs = enum_args(m, enums)
            if not pairs:
                continue
            doc = ast.get_docstring(m, clean=False)
            if doc is None:
                print(f"  {cls.name}.{m.name}: NO DOCSTRING - skipped, fix by hand")
                continue
            node = m.body[0]  # the docstring expression
            indent = " " * (m.col_offset + 4)
            if "Enum arg:" in doc:
                continue  # already hinted
            hint = build_hint(pairs, handle, indent)
            start, end = node.lineno - 1, node.end_lineno - 1
            if start == end:
                # one-line docstring: expand it so the hint has somewhere to go
                summary = lines[start].strip()
                assert summary.endswith('"""'), summary
                body = summary[3:-3]
                new = [f'{indent}"""{body}\n', "\n"] + hint + [f'{indent}"""\n']
                edits.append((start, end + 1, new, f"{cls.name}.{m.name}"))
            else:
                # multi-line: insert a blank line + hint right after the summary
                edits.append((start + 1, start + 1, ["\n"] + hint, f"{cls.name}.{m.name}"))

    for start, end, new, desc in sorted(edits, key=lambda e: -e[0]):
        print(f"  {desc}: {''.join(new).strip()}")
        if apply:
            lines[start:end] = new
    if apply and edits:
        path.write_text("".join(lines), encoding="utf-8")
    return len(edits)


def main() -> int:
    apply = "--apply" in sys.argv
    targets = [pathlib.Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    total = 0
    for path in targets:
        print(f"--- {path} ---")
        total += process(path, apply)
    print(f"\n{total} hints {'APPLIED' if apply else '(dry run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
