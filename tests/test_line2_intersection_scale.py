import numpy as np
import pytest
from spatialmath import Line2


@pytest.mark.parametrize(
    "scales",
    [
        (1, 1),
        (1e-12, 1),
        (1, 1e-12),
        (1e-12, 1e-12),
        (-1e-12, 1e12),
        (1e12, -1e-12),
        (1e200, 1e200),
        (1e-200, -1e-200),
    ],
)
@pytest.mark.parametrize(
    "coefficients, expected",
    [
        ([0, 1, -3], True),
        ([1, 1, -3], True),
        ([1, 0, -3], False),
        ([1, 0, -2], False),
    ],
)
def test_intersection_is_invariant_to_homogeneous_scale(scales, coefficients, expected):
    a = Line2(scales[0] * np.array([1.0, 0, -2]))
    b = Line2(scales[1] * np.array(coefficients, dtype=float))
    with np.errstate(over="raise", invalid="raise"):
        assert bool(a.intersect(b)) is expected
        assert bool(b.intersect(a)) is expected


@pytest.mark.parametrize("scale", [1e-12, 1, -1e12])
@pytest.mark.parametrize("tol, expected", [(1, True), (20, False), (100, False)])
def test_normalized_angle_tolerance(scale, tol, expected):
    a = Line2([1, 0, -2])
    b = Line2(scale * np.array([1.0, 10 * np.finfo(float).eps, -3]))
    assert bool(a.intersect(b, tol=tol)) is expected


@pytest.mark.parametrize("coefficients", [[0, 0, 0], [0, 0, 1]])
def test_degenerate_line_keeps_false_result(coefficients):
    a = Line2(coefficients)
    b = Line2([1, 0, -2])
    assert not a.intersect(b)
    assert not b.intersect(a)
