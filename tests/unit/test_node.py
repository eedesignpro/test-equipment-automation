"""Unit tests for Node install gating."""

import pytest

from common.errors import HardwareNotPresentError
from common.node import Node


def test_installed_node_forwards_write(make_link):
    link, mock = make_link()
    node = Node(link, name="CH1")
    node.write("FOO 1")
    assert mock.writes[-1] == "FOO 1"


def test_uninstalled_node_raises_on_write(make_link):
    link, _ = make_link()
    node = Node(link, name="CH8")
    node.installed = False
    with pytest.raises(HardwareNotPresentError) as exc:
        node.write("FOO 1")
    assert "CH8" in str(exc.value)


def test_uninstalled_node_raises_on_query(make_link):
    link, _ = make_link({"FOO?": "1"})
    node = Node(link, name="CH8")
    node.installed = False
    with pytest.raises(HardwareNotPresentError):
        node.query("FOO?")


def test_uninstalled_node_raises_on_query_float(make_link):
    link, _ = make_link({"FOO?": "1.0"})
    node = Node(link, name="CH8")
    node.installed = False
    with pytest.raises(HardwareNotPresentError):
        node.query_float("FOO?")


def test_uninstalled_node_raises_on_query_int(make_link):
    link, _ = make_link({"FOO?": "1"})
    node = Node(link, name="CH8")
    node.installed = False
    with pytest.raises(HardwareNotPresentError):
        node.query_int("FOO?")


def test_uninstalled_node_raises_on_query_bool(make_link):
    link, _ = make_link({"FOO?": "1"})
    node = Node(link, name="CH8")
    node.installed = False
    with pytest.raises(HardwareNotPresentError):
        node.query_bool("FOO?")


def test_uninstalled_node_raises_on_write_choice(make_link):
    link, _ = make_link()
    node = Node(link, name="CH8")
    node.installed = False
    with pytest.raises(HardwareNotPresentError):
        node.write_choice("FOO", "A", {"A": "A"})


def test_uninstalled_node_raises_on_set_and_verify(make_link):
    link, _ = make_link({"FOO?": "1"})
    node = Node(link, name="CH8")
    node.installed = False
    with pytest.raises(HardwareNotPresentError):
        node.set_and_verify("FOO", 1)


def test_node_installed_defaults_true_and_no_name_uses_class_name(make_link):
    link, _ = make_link()
    node = Node(link)
    assert node.installed is True
    node.installed = False
    with pytest.raises(HardwareNotPresentError) as exc:
        node.write("X")
    assert "Node" in str(exc.value)
