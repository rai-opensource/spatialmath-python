"""Animation style changes must reach the replacement patch."""

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy.testing as nt
import pytest
from spatialmath import Polygon2, SE2


@pytest.mark.parametrize("pose", [SE2(), SE2(2, -1), SE2(1, 2, 0.4)])
def test_animate_replaces_style_and_keeps_it(pose):
    fig, ax = plt.subplots()
    try:
        polygon = Polygon2([(0, 0), (2, 0), (0, 1)])
        polygon.plot(ax=ax, facecolor="red", edgecolor="black", linewidth=1)
        old_patch = polygon.patch
        polygon.animate(pose, facecolor="blue", linewidth=4)
        assert old_patch not in ax.patches
        assert len(ax.patches) == 1
        nt.assert_allclose(polygon.patch.get_facecolor(), to_rgba("blue"))
        assert polygon.patch.get_linewidth() == 4
        # New arguments replace the construction style, as documented.
        assert polygon.kwargs == {"facecolor": "blue", "linewidth": 4}
        expected = pose.R @ polygon.path0.vertices.T + pose.t[:, None]
        nt.assert_allclose(polygon.vertices(unique=False), expected, atol=1e-14)
        polygon.animate(SE2(-1, 3))
        nt.assert_allclose(polygon.patch.get_facecolor(), to_rgba("blue"))
        assert polygon.patch.get_linewidth() == 4
        assert len(ax.patches) == 1
    finally:
        plt.close(fig)


def test_animate_without_style_keeps_plot_options():
    fig, ax = plt.subplots()
    try:
        polygon = Polygon2([(0, 0), (2, 0), (0, 1)])
        polygon.plot(ax=ax, facecolor="green", edgecolor="black", linewidth=2)
        polygon.animate(SE2(1, 2))
        nt.assert_allclose(polygon.patch.get_facecolor(), to_rgba("green"))
        assert polygon.patch.get_linewidth() == 2
        assert polygon.kwargs == dict(facecolor="green", edgecolor="black", linewidth=2)
    finally:
        plt.close(fig)
