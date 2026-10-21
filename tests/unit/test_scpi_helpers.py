"""Direct unit tests for src/common/scpi.py (quote/parse_compound).

Pure-Python string logic with no instrument I/O - tested directly per the QA
policy allowing pure parsing logic to be exercised without a capture trace.
These functions are exercised indirectly elsewhere (e.g. test_file.py,
test_display_vertical_ref.py) but had no dedicated unit-level coverage of
their own edge cases (embedded quotes, ADR-018's quote-aware field splitting).
"""

import pytest

from common.scpi import parse_compound, quote


def test_quote_wraps_value_in_double_quotes():
    assert quote("C:/pyvisa_tmp/x.set") == '"C:/pyvisa_tmp/x.set"'


def test_quote_rejects_embedded_double_quote():
    with pytest.raises(ValueError):
        quote('bad"path')


def test_quote_allows_empty_string():
    assert quote("") == '""'


def test_parse_compound_bench_ch5_style_response():
    """CH5? bench-style compound response (ADR-018): 59 fields, field 0 self-cal."""
    response = 'PASSED;1;0.3;0;50.0OHM;1.0E-9;DCR;OFF;"extra;field"'
    fields = parse_compound(response)
    assert fields[0] == "PASSED"
    assert fields[1] == "1"
    assert fields[-1] == "extra;field"


def test_parse_compound_does_not_split_separator_inside_quotes():
    response = 'FIELD1;"contains;a;separator";FIELD3'
    fields = parse_compound(response)
    assert fields == ["FIELD1", "contains;a;separator", "FIELD3"]


def test_parse_compound_field_starting_and_ending_with_quote_is_unwrapped():
    response = '"quoted field";plain'
    assert parse_compound(response) == ["quoted field", "plain"]


def test_parse_compound_unmatched_quote_swallows_rest_as_one_field():
    """An unterminated quote opens a quoted span that swallows any separators
    for the rest of the string (single unmatched-quote edge case)."""
    response = 'A"B;plain'
    fields = parse_compound(response)
    assert fields == ['A"B;plain']


def test_parse_compound_custom_separator():
    assert parse_compound("A,B,C", sep=",") == ["A", "B", "C"]


def test_parse_compound_single_field_no_separator():
    assert parse_compound("SOLO") == ["SOLO"]


def test_parse_compound_empty_string_returns_single_empty_field():
    assert parse_compound("") == [""]
