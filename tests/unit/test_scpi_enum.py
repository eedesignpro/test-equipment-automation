"""Every driver enum is a ScpiEnum, and its string behavior is pinned.

These enums deliberately do NOT inherit from ``str``. A ``StrEnum`` member
carries all 47 string methods, which turned ``psu2.Coupling.<TAB>`` into 51
completions of which 4 were the enum. Autocomplete is the discovery tool for
these drivers, so that mattered (design standard 2 and 4.6).

Dropping the ``str`` base has one sharp edge: code that branched on
``isinstance(value, str)`` used to catch enum members and no longer does. That
edge already bit the scope's ``set_load``, so it is pinned here.

No hardware: enums and token helpers are pure.
"""

from __future__ import annotations

import enum
import importlib

import pytest

from common.scpi import ScpiEnum
from drivers.owon_odp import Owon_ODP
from drivers.tektronix_mso import Tektronix_MSO

# Module path per driver. Enums are attached to whichever class takes them,
# which is often a node class such as _SmuChannel or _Trace rather than the
# instrument, so the whole module is walked.
DRIVER_MODULES = [
    "drivers.keysight_b2900",
    "drivers.keysight_e364xa",
    "drivers.owon_odp",
    "drivers.rigol_dl3000",
    "drivers.tektronix_mso.driver",
]


def _driver_enums(module_path: str) -> list[type]:
    module = importlib.import_module(module_path)
    found: dict[str, type] = {}
    for owner in vars(module).values():
        if not isinstance(owner, type):
            continue
        if issubclass(owner, enum.Enum):
            found[owner.__qualname__] = owner
        for attr in vars(owner).values():
            if isinstance(attr, type) and issubclass(attr, enum.Enum):
                found[attr.__qualname__] = attr
    return list(found.values())


ALL_DRIVERS = DRIVER_MODULES


@pytest.mark.parametrize("driver", ALL_DRIVERS)
def test_driver_enums_are_scpi_enums(driver):
    """Every enum attached to a driver uses the shared base, not StrEnum."""
    found = _driver_enums(driver)
    assert found, f"{driver.__name__} attaches no enums"
    for member_type in found:
        assert issubclass(member_type, ScpiEnum), f"{driver.__name__}.{member_type.__name__}"


@pytest.mark.parametrize("driver", ALL_DRIVERS)
def test_enum_members_are_not_strings(driver):
    """The point of the exercise: members must not inherit the str namespace."""
    for member_type in _driver_enums(driver):
        for member in member_type:
            assert not isinstance(member, str), f"{member_type.__name__}.{member.name}"


@pytest.mark.parametrize("driver", ALL_DRIVERS)
def test_str_yields_the_scpi_token(driver):
    """``str(member)`` and f-strings still produce the token every call site uses."""
    for member_type in _driver_enums(driver):
        for member in member_type:
            assert str(member) == member.value
            assert f"{member}" == member.value


@pytest.mark.parametrize("driver", ALL_DRIVERS)
def test_completion_is_only_the_members(driver):
    """Nothing but the members shows up in the completion list."""
    for member_type in _driver_enums(driver):
        public = [n for n in dir(member_type) if not n.startswith("_")]
        assert sorted(public) == sorted(member_type.__members__)


def test_token_helper_accepts_member_name_or_token():
    """The driver token helpers still take a member, a name, or a raw token."""
    from drivers.owon_odp import _token

    channel_enum = Owon_ODP.Coupling
    assert _token(channel_enum.SERIES, channel_enum, "coupling") == "SERIES"
    assert _token("series", channel_enum, "coupling") == "SERIES"
    assert _token("SERIES", channel_enum, "coupling") == "SERIES"
    with pytest.raises(ValueError, match="invalid coupling"):
        _token("sideways", channel_enum, "coupling")


def test_scope_set_load_still_accepts_its_enum():
    """Regression: set_load branched on isinstance(str), which enums no longer are."""
    scope = Tektronix_MSO("TCPIP::0.0.0.0::INSTR")
    sent: list[str] = []
    scope.afg.write = lambda message, **kw: sent.append(message)  # type: ignore[method-assign]
    scope.afg.query = lambda message, **kw: "HIGHZ"  # type: ignore[method-assign]

    assert scope.afg.set_load(scope.Load.HIGHZ) == float("inf")
    assert any("HIGHZ" in message for message in sent)

    sent.clear()
    scope.afg.query = lambda message, **kw: "FIFty"  # type: ignore[method-assign]
    assert scope.afg.set_load(scope.Load.FIFTY) == 50.0
    assert any("FIFty" in message for message in sent)
