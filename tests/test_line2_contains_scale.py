import numpy as np
import pytest

from spatialmath.geom2d import Line2, Polygon2


@pytest.mark.parametrize("scale", [1, -1, 1e-20, -1e-20, 1e-8, -1e8, 1e20])
@pytest.mark.parametrize(
    "coefficients,on,off",
    [([1, 0, -1], [1, 2], [2, 2]), ([3, 4, -10], [2, 1], [2, 2])],
)
def test_contains_coefficient_scale(scale, coefficients, on, off):
    line = Line2(scale * np.array(coefficients))
    assert line.contains(on)
    assert not line.contains(off)
    assert line.contains(np.r_[on, 1])
    assert not line.contains(np.r_[off, 1])


@pytest.mark.parametrize("scale", [1, -1, 1e-8, -1e8])
def test_contains_geometric_tolerance(scale):
    eps = np.finfo(float).eps
    line = Line2(scale * np.array([0, 1, 0]))
    assert line.contains([0, 10 * eps])
    assert not line.contains([0, 30 * eps])
    assert line.contains([0, 30 * eps], tol=40)


@pytest.mark.parametrize("scale", [1, 1e-20, -1e-20, 1e8])
def test_polygon_line_nonintersection_is_scale_invariant(scale):
    polygon = Polygon2([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    outside = Line2(scale * np.array([0, 1, -2]))
    crossing = Line2(scale * np.array([0, 1, 0]))
    assert not polygon.intersects(outside)
    assert polygon.intersects(crossing)
