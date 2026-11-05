import builtins
import sys

import pytest

from spatialmath.base import environment, isnotebook

# These tests are mocked: a fake get_ipython() and sys.modules / sys.platform
# stand in for the real front ends, which cannot be run under CI.


def fake_shell(name):
    shell = type(name, (), {})()
    return lambda: shell


@pytest.fixture
def plain(monkeypatch):
    """Standard interpreter: no IPython, not Colab, not Pyodide"""
    monkeypatch.delattr(builtins, "get_ipython", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    return monkeypatch


def test_terminal(plain):
    assert environment() == "terminal"
    assert not isnotebook()


def test_ipython_terminal(plain):
    plain.setattr(
        builtins, "get_ipython", fake_shell("TerminalInteractiveShell"), raising=False
    )
    assert environment() == "ipython"
    assert not isnotebook()


def test_jupyter(plain):
    plain.setattr(
        builtins, "get_ipython", fake_shell("ZMQInteractiveShell"), raising=False
    )
    assert environment() == "jupyter"
    assert isnotebook()


def test_unknown_shell(plain):
    plain.setattr(builtins, "get_ipython", fake_shell("SomethingElse"), raising=False)
    assert environment() == "terminal"
    assert not isnotebook()


def test_colab(plain):
    # Colab's shell is not a ZMQInteractiveShell, so detect it by the module
    plain.setattr(builtins, "get_ipython", fake_shell("Shell"), raising=False)
    plain.setitem(sys.modules, "google.colab", object())
    assert environment() == "colab"
    assert isnotebook()


def test_jupyterlite(plain):
    plain.setattr(sys, "platform", "emscripten")
    assert environment() == "jupyterlite"
    assert isnotebook()
