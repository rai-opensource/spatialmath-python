"""A transformed polygon has its own animation reference geometry."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import Polygon2, SE2


@pytest.mark.parametrize("first", [SE2(), SE2(2, -1), SE2(1, 2, 0.4)])
@pytest.mark.parametrize("second", [SE2(), SE2(-1, 3, -0.2)])
def test_transformed_copy_can_animate(first, second):
    original = Polygon2([(0, 0), (2, 0), (0, 1)])
    original_vertices = original.vertices(unique=False).copy()
    transformed = original.transformed(first)
    initial = first.R @ original_vertices + first.t[:, None]
    nt.assert_allclose(transformed.vertices(unique=False), initial, atol=1e-14)
    fig, ax = plt.subplots()
    try:
        transformed.plot(ax=ax, facecolor="blue")
        old_patch = transformed.patch
        transformed.animate(second)
        assert old_patch not in ax.patches
        assert len(ax.patches) == 1
        expected = second.R @ initial + second.t[:, None]
        nt.assert_allclose(transformed.vertices(unique=False), expected, atol=1e-14)
        # Frames are absolute relative to the transformed copy, not cumulative.
        transformed.animate(SE2())
        nt.assert_allclose(transformed.vertices(unique=False), initial, atol=1e-14)
        nt.assert_allclose(original.vertices(unique=False), original_vertices)
        assert len(ax.patches) == 1
    finally:
        plt.close(fig)


def test_chained_transformed_copy_has_latest_reference():
    polygon = Polygon2([(0, 0), (2, 0), (0, 1)])
    polygon = polygon.transformed(SE2(1, 2)).transformed(SE2.Rot(0.3))
    initial = polygon.vertices(unique=False).copy()
    fig, ax = plt.subplots()
    try:
        polygon.plot(ax=ax)
        polygon.animate(SE2(1, 0))
        nt.assert_allclose(
            polygon.vertices(unique=False), initial + np.array([[1], [0]])
        )
    finally:
        plt.close(fig)
