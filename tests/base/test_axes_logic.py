"""Tests of ``axes_logic``, which decides which axes a plot_xxx() function draws into."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from spatialmath.base.graphics import axes_logic


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def nfigs():
    return len(plt.get_fignums())


# --- behaviour that must not change ---------------------------------------------


def test_no_figure_creates_one():
    ax = axes_logic(None, 2)
    assert nfigs() == 1
    assert ax is plt.gca()


def test_current_axes_are_reused():
    fig, ax0 = plt.subplots()
    ax = axes_logic(None, 2)
    assert ax is ax0
    assert nfigs() == 1


def test_new_forces_a_new_figure():
    fig, ax0 = plt.subplots()
    ax = axes_logic(None, 2, new=True)
    assert ax is not ax0
    assert nfigs() == 2
    assert ax is plt.gca()


def test_dimension_mismatch_makes_a_new_figure():
    fig = plt.figure()
    ax3 = fig.add_subplot(projection="3d")
    ax = axes_logic(None, 2)
    assert ax is not ax3
    assert nfigs() == 2


def test_explicit_axes_of_the_right_dimension_are_used():
    fig, ax0 = plt.subplots()
    fig2, ax1 = plt.subplots()
    assert axes_logic(ax0, 2) is ax0
    assert nfigs() == 2


def test_explicit_axes_of_the_wrong_dimension_make_a_new_figure():
    fig, ax0 = plt.subplots()
    ax = axes_logic(ax0, 3)
    assert ax is not ax0
    assert nfigs() == 2


# --- an empty current figure, eg. from a bare plt.figure() -----------------------


def test_empty_current_figure_is_used_2d():
    fig = plt.figure()
    ax = axes_logic(None, 2)
    assert nfigs() == 1
    assert ax.figure is fig
    assert ax is plt.gca()


def test_empty_current_figure_is_used_3d():
    fig = plt.figure()
    ax = axes_logic(None, 3)
    assert nfigs() == 1
    assert ax.figure is fig
    assert ax.name == "3d"


def test_only_the_current_figure_matters():
    # an earlier figure has axes, but the current one is empty: use the current one
    plt.subplots()
    fig = plt.figure()
    ax = axes_logic(None, 2)
    assert nfigs() == 2
    assert ax.figure is fig


def test_new_with_an_empty_current_figure_still_makes_another():
    # new=True is an explicit request for a new figure
    fig = plt.figure()
    ax = axes_logic(None, 2, new=True)
    assert nfigs() == 2
    assert ax.figure is not fig
