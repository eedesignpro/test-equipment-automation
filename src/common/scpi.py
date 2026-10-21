"""Shared SCPI string helpers."""

from __future__ import annotations

from enum import Enum


class ScpiEnum(Enum):
    """Base for a driver's multi-option argument enums.

    Deliberately NOT ``StrEnum``. A ``StrEnum`` member inherits every string
    method, so ``psu2.Coupling.<TAB>`` offered 51 completions of which 4 were
    the enum and 47 were ``capitalize``, ``casefold``, ``encode`` and friends.
    Autocomplete is the discovery tool for these drivers, so the namespace is
    part of the interface (design standard section 2), and 4-in-51 defeats it.

    ``__str__`` returns the SCPI token, so ``f"{Mode.CURRENT}"`` and
    ``str(mode)`` still produce ``CURR`` and every existing call site keeps
    working. The one thing given up is direct equality against a bare string:
    ``Mode.CURRENT == "CURR"`` is False. Compare ``.value``, or pass the member
    to the driver, whose token helper accepts a member, a member name, or a
    token string, case-insensitively.
    """

    def __str__(self) -> str:
        return str(self.value)


def member_from_response(response: str, enum_cls: type[ScpiEnum], label: str) -> ScpiEnum:
    """Resolve an instrument's answer to a member of ``enum_cls``.

    Instruments differ in what they answer with, and the difference is not
    predictable from the command. The B2902B and the 33522A answer with the
    same short token they accept; the MSO58 accepts an abbreviation and answers
    with the full keyword, so it is set with ``SAM`` and answers ``SAMPLE``.

    This matches the uppercased write token first, then the member name with
    underscores removed, which covers both conventions. An answer it cannot
    place raises rather than defaulting, because an unplaceable response means
    the enum has drifted from the instrument and that is worth knowing.
    """
    text = response.strip().upper().strip('"')
    for member in enum_cls:
        if text == str(member.value).upper():
            return member
    for member in enum_cls:
        if text == member.name.replace("_", ""):
            return member
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(
        f"the instrument answered {response!r} for {label}, which is not a known "
        f"{enum_cls.__name__}; expected one of: {valid}"
    )


def quote(value: str) -> str:
    """Wrap a string as a SCPI quoted string (``<QString>``).

    Instruments that take double-quoted string arguments (file paths, source
    names) expect this ``<QString>`` form. An embedded double quote is rejected
    rather than silently mangling the argument.

    ### Parameters
    1. value : str
        - the raw string (e.g. an instrument file path)
    ### Returns
    1. str
        - the double-quoted SCPI string literal
    """
    if '"' in value:
        raise ValueError(f"cannot SCPI-quote a string containing a double quote: {value!r}")
    return f'"{value}"'


def parse_compound(response: str, sep: str = ";") -> list[str]:
    """Split a compound SCPI response into fields, honoring quoted segments.

    Compound query responses (a single query returning several fields) are
    separator-delimited but may contain quoted substrings that themselves
    contain the separator. This walks
    the string so a separator inside a double-quoted run is not a split point,
    then strips surrounding quotes from each field (a naive ``strip('"')`` would
    corrupt fields whose content legitimately starts or ends with a quote).

    ### Parameters
    1. response : str
        - the raw compound response
    2. sep : str (default ';')
        - field separator
    ### Returns
    1. fields : list[str]
        - the ordered, unquoted fields
    """
    fields: list[str] = []
    current: list[str] = []
    in_quotes = False
    for char in response.strip():
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
        elif char == sep and not in_quotes:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    fields.append("".join(current).strip())

    out: list[str] = []
    for field in fields:
        if len(field) >= 2 and field[0] == '"' and field[-1] == '"':
            out.append(field[1:-1])
        else:
            out.append(field)
    return out
