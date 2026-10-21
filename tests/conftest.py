"""Shared pytest fixtures: mock-VISA backed InstrumentLink and MSO driver."""

import pytest

from common.link import InstrumentLink

from tests.mocks.mock_visa import MockVisaResource


@pytest.fixture
def mock_visa():
    """A bare MockVisaResource with no scripted responses."""
    return MockVisaResource()


@pytest.fixture
def make_link(monkeypatch):
    """Factory returning a connected InstrumentLink wired to a mock resource.

    Usage::

        link, mock = make_link({"*IDN?": "TEKTRONIX,MSO58,..."})
    """

    def _factory(responses=None, default_response="", resource_name="MOCK::INSTR"):
        mock = MockVisaResource(responses=responses, default_response=default_response)
        link = InstrumentLink(resource_name)
        monkeypatch.setattr(link, "_open_resource", lambda rm, name: mock)
        link.connect()
        return link, mock

    return _factory
