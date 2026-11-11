"""Point containment must not depend on homogeneous Plucker scale."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import Line3


@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 1, 0], [1, 2, -1]])
@pytest.mark.parametrize("scale", [1.0, -1.0, 1e-8, -1e-8, 1e8])
def test_contains_scalar_and_batch_are_scale_invariant(direction, scale):
    direction = np.asarray(direction, dtype=float)
    normal = np.cross(direction, [0, 0, 1])
    normal /= np.linalg.norm(normal)
    origin = np.array([1.0, 2.0, 3.0])
    line = Line3.PointDir(origin, scale * direction)
    direction /= np.linalg.norm(direction)
    # Far outside tolerance even for the translated/oblique lines.
    on_line = origin + 2 * direction
    off_line = on_line + 1e-8 * normal
    assert line.contains(on_line, tol=100)
    assert not line.contains(off_line, tol=100)
    nt.assert_array_equal(
        line.contains(np.column_stack([on_line, off_line]), tol=100), [True, False]
    )


@pytest.mark.parametrize("scale", [1.0, -1.0, 1e-8, 1e8])
def test_contains_tolerance_is_a_geometric_distance(scale):
    line = Line3.PointDir([0, 0, 0], [scale, 0, 0])
    eps = np.finfo(float).eps
    assert line.contains([1, 10 * eps, 0], tol=20)
    assert not line.contains([1, 30 * eps, 0], tol=20)
    assert line.contains([1, 30 * eps, 0], tol=40)
    nt.assert_array_equal(
        line.contains(np.array([[1, 1], [10 * eps, 30 * eps], [0, 0]]), tol=20),
        [True, False],
    )
