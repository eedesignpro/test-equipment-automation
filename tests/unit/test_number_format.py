"""Numbers go to the instrument with 12 significant digits, not 6.

With 6 digits, 1 234 567.89 Hz went out as 1.23457e+06, a 2.11 Hz error that
the read-back check did not catch.
"""

import pytest

from drivers import (
    keysight_33120a,
    keysight_33522a,
    keysight_b2900,
    keysight_e364xa,
    owon_odp,
    siglent_sdm,
)

VALUE = 1234567.89


def test_set_and_verify_sends_twelve_digits(make_link):
    link, mock = make_link({"FREQ?": "1234567.89"})
    link.set_and_verify("FREQ", VALUE)
    assert mock.writes[0] == "FREQ 1234567.89"


@pytest.mark.parametrize(
    "module",
    [keysight_33120a, keysight_33522a, keysight_b2900, keysight_e364xa, owon_odp, siglent_sdm],
)
def test_driver_number_helper_keeps_twelve_digits(module):
    assert module._num(VALUE) == "1234567.89"
    assert float(module._num(1 / 3)) == pytest.approx(1 / 3, rel=1e-12)
