"""Tests of parse_fmt, which is checked against what Matplotlib's plot() makes of
the same string."""

import itertools
import re

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import _base

from spatialmath.base import parse_fmt

COLORS = ["b", "g", "r", "c", "m", "y", "k", "w", "C0", "C3", "C12"]
MARKERS = list(".,ov^<>1234sp*hH+xDd|_PX8")
LINESTYLES = ["-", "--", "-.", ":"]


def matplotlib_says(fmt):
    """What plot() makes of fmt: (linestyle, marker, color or None)."""
    linestyle, marker, color = _base._process_plot_format(fmt)
    # a string that is only a color comes back with no marker or linestyle at all
    if marker is None:
        marker = "None"
    if linestyle is None:
        linestyle = mpl.rcParams["lines.linestyle"]
    return linestyle, marker, color


def n_colors(fmt):
    """Number of color symbols in fmt, a letter or a cycle color like C2."""
    return len(re.findall(r"C\d+|[bgrcmykw]", fmt))


def same_as_matplotlib(fmt):
    """Check parse_fmt against Matplotlib, filling in what plot() fills in."""
    ours = parse_fmt(fmt)
    linestyle, marker, color = matplotlib_says(fmt)

    # plot() takes a missing marker as none, and a missing linestyle as the rc
    # default, unless there is a marker, when it is none
    assert ours.get("marker", "None") == marker, fmt
    assert ours.get("linestyle", mpl.rcParams["lines.linestyle"]) == linestyle, fmt
    if color is None:
        assert "color" not in ours, fmt
    else:
        assert mcolors.to_rgba(ours["color"]) == mcolors.to_rgba(color), fmt
    assert set(ours) <= {"color", "marker", "linestyle"}


def test_every_single_symbol():
    for sym in COLORS + MARKERS + LINESTYLES:
        same_as_matplotlib(sym)


def test_every_pair_in_either_order():
    symbols = COLORS + MARKERS + LINESTYLES
    n = 0
    for a, b in itertools.product(symbols, repeat=2):
        fmt = a + b
        if n_colors(fmt) > 1:
            continue  # see test_two_colors_are_refused
        try:
            matplotlib_says(fmt)
        except ValueError:
            with pytest.raises(ValueError):
                parse_fmt(fmt)
        else:
            same_as_matplotlib(fmt)
            n += 1
    assert n > 800


def test_every_color_marker_linestyle_in_every_order():
    n = 0
    for color, marker, ls in itertools.product(COLORS, MARKERS, LINESTYLES):
        for order in itertools.permutations([color, marker, ls]):
            fmt = "".join(order)
            if n_colors(fmt) > 1:
                continue
            try:
                matplotlib_says(fmt)
            except ValueError:
                # such as a marker "1" after "C", which is the cycle color "C1"
                with pytest.raises(ValueError):
                    parse_fmt(fmt)
            else:
                same_as_matplotlib(fmt)
                n += 1
    assert n > 5000


@pytest.mark.parametrize(
    "fmt",
    ["red", "0.5", "0.0", "#ff8000", "#f80", "tab:blue", "C4", "xkcd:sky blue", "gray"],
)
def test_whole_string_colors(fmt):
    same_as_matplotlib(fmt)
    assert parse_fmt(fmt) == {"color": fmt}


def test_one_is_a_marker_not_a_gray_level():
    # "1" is a tri_down marker, not the gray level 1
    same_as_matplotlib("1")
    assert parse_fmt("1") == {"marker": "1", "linestyle": "None"}


def test_zero_is_refused():
    # "0" could be black, but Matplotlib refuses it for consistency with "1"
    with pytest.raises(ValueError):
        matplotlib_says("0")
    with pytest.raises(ValueError):
        parse_fmt("0")


def test_dashed_line_with_dot_marker():
    same_as_matplotlib("--.")
    assert parse_fmt("--.") == {"linestyle": "--", "marker": "."}


@pytest.mark.parametrize("fmt", ["rg", "rr", "bC0", "C0C1", "C1r", "kC12--"])
def test_two_colors_are_refused(fmt):
    # Matplotlib refuses two letter colors, but takes the last of a cycle color
    # and a letter; we refuse them all
    with pytest.raises(ValueError, match="two color symbols"):
        parse_fmt(fmt)


@pytest.mark.parametrize(
    "fmt",
    ["roo", "r?", "ro--?", "C", "Cx", "xyz", "r--:", "-:", "--:", "ab"],
)
def test_invalid_strings_raise_as_matplotlib_does(fmt):
    with pytest.raises(ValueError):
        matplotlib_says(fmt)
    with pytest.raises(ValueError, match="not a valid format string"):
        parse_fmt(fmt)


def test_empty_and_none():
    assert parse_fmt("") == {}
    assert parse_fmt(None) == {}
    assert parse_fmt(()) == {}
    assert parse_fmt([]) == {}


def test_tuple_or_list_takes_the_first():
    # as the *fmt argument of the plot functions arrives
    assert parse_fmt(("r--",)) == {"color": "r", "linestyle": "--"}
    assert parse_fmt(["r--", "ignored"]) == {"color": "r", "linestyle": "--"}


def test_not_a_string():
    with pytest.raises(ValueError, match="must be a string"):
        parse_fmt(3)


def test_only_what_the_string_sets():
    assert parse_fmt("--") == {"linestyle": "--"}
    assert parse_fmt("r") == {"color": "r"}
    assert parse_fmt("r--") == {"color": "r", "linestyle": "--"}
    assert parse_fmt("-.b") == {"color": "b", "linestyle": "-."}
    assert parse_fmt("C2:") == {"color": "C2", "linestyle": ":"}


def test_marker_alone_means_no_line():
    assert parse_fmt("o") == {"marker": "o", "linestyle": "None"}
    assert parse_fmt("rx") == {"color": "r", "marker": "x", "linestyle": "None"}
    assert parse_fmt("ro--") == {"color": "r", "marker": "o", "linestyle": "--"}


def test_the_dot_is_a_marker_not_part_of_dash_dot():
    assert parse_fmt("-.") == {"linestyle": "-."}
    assert parse_fmt(".-") == {"marker": ".", "linestyle": "-"}


def test_merges_over_defaults_without_wiping_them():
    defaults = {"color": "red", "linewidth": 2}
    assert defaults | parse_fmt("--") == {
        "color": "red",
        "linewidth": 2,
        "linestyle": "--",
    }
    assert defaults | parse_fmt("b--") == {
        "color": "b",
        "linewidth": 2,
        "linestyle": "--",
    }


def test_result_is_accepted_by_plot():
    fig, ax = plt.subplots()
    try:
        for fmt in ["r--", "go", "C2:", "--", "red", "k.-", "+"]:
            (line,) = ax.plot([0, 1], [0, 1], **parse_fmt(fmt))
            (ref,) = ax.plot([0, 1], [0, 1], fmt)
            assert line.get_linestyle() == ref.get_linestyle(), fmt
            assert line.get_marker() == ref.get_marker(), fmt
            if fmt not in ("--", "+"):  # no color: the color cycle differs
                assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                    ref.get_color()
                ), fmt
    finally:
        plt.close(fig)
