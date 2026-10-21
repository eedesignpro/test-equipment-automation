"""Strip SCPI command names from the first docstring line of public methods.

Design standard 9.1 rule 1: the first docstring line is the whole map entry, and
naming a command there pulls the reader back to the command tree. Private
methods keep their command names, where the mapping is one to one.

Dry run by default; pass --apply to write.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DEF_RE = re.compile(r"^(\s*)def ([A-Za-z_][A-Za-z0-9_]*)\(")

# A parenthetical that is nothing but a SCPI command, optionally with its
# argument list or a short token list: (FUNC:SQU:DCYC), (*RST), (OUTP<n>:LOAD),
# (AM:DEPT), (SWE:SPAC): LINEAR or LOGARITHMIC -> only the parenthetical goes.
SCPI_PAREN = re.compile(
    r"\s*\((?:see\s+)?"
    r"(?P<body>[*A-Z][A-Za-z0-9:*<>?\[\]]*"
    r"(?:\s+[A-Za-z0-9:,<>/ ]+)?)"
    r"\)"
)


# Parentheticals that are units or ordinary prose, never SCPI. Without this the
# stripper eats "(Hz)" out of "the frequencies (Hz) of the active peak markers".
NOT_SCPI = {
    "HZ",
    "KHZ",
    "MHZ",
    "GHZ",
    "V",
    "MV",
    "UV",
    "A",
    "MA",
    "UA",
    "W",
    "MW",
    "S",
    "MS",
    "US",
    "NS",
    "PS",
    "OHM",
    "OHMS",
    "DB",
    "DBM",
    "AH",
    "WH",
    "VPP",
    "VRMS",
    "SA/S",
    "PERCENT",
    "DEG",
}


def looks_like_scpi(body: str) -> bool:
    """True when the parenthetical is a command name rather than real prose."""
    head = body.split()[0] if body.split() else ""
    if not head:
        return False
    if head.upper() in NOT_SCPI:
        return False
    if head.startswith("*"):  # *RST, *CLS, *TRG
        return True
    if ":" not in head:
        # A bare all-caps mnemonic like (DISP) or (TRACk). Require it to be a
        # plausible SCPI keyword: mostly caps, no lowercase words after it.
        if not re.fullmatch(r"[A-Z][A-Za-z0-9<>?\[\]]{1,15}", head):
            return False
        if len(body.split()) > 1:
            return False
        if head.upper() == head:
            return True
        # Mixed case is a SCPI short-form marker only with 2+ leading capitals:
        # TRACk and NTSc are commands, Kelvin and True are prose.
        return re.fullmatch(r"[A-Z]{2,}[a-z]+", head) is not None
    # Has a colon: SOUR:FREQ, FUNC:ARB:SRAT. Require the segments to be caps-led.
    return all(re.fullmatch(r"[A-Za-z0-9*<>?\[\]]+", seg) for seg in head.split(":"))


def main() -> int:
    apply = "--apply" in sys.argv
    paths = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    total = 0
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        out = list(lines)
        i = 0
        while i < len(lines):
            m = DEF_RE.match(lines[i])
            if not m or m.group(2).startswith("_"):
                i += 1
                continue
            # find the first line of the docstring after the signature
            j = i
            while j < len(lines) and '"""' not in lines[j]:
                j += 1
                if j - i > 12:
                    break
            if j >= len(lines) or '"""' not in lines[j]:
                i += 1
                continue
            line = out[j]
            new = line
            for match in list(SCPI_PAREN.finditer(line)):
                if looks_like_scpi(match.group("body")):
                    new = new.replace(match.group(0), "", 1)
            if new != line:
                total += 1
                print(f"{path.name}:{j + 1}  {m.group(2)}")
                print(f"  -  {line.strip()}")
                print(f"  +  {new.strip()}")
                out[j] = new
            i = j + 1
        if apply:
            path.write_text("".join(out), encoding="utf-8")
    print(f"\n{total} first-line rewrites{' APPLIED' if apply else ' (dry run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
