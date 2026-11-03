import numpy as np
import pytest

from spatialmath import Plane3


@pytest.mark.parametrize("scale", [1, -1, 2.5])
@pytest.mark.parametrize("factory", ["coefficients", "point-normal", "three-points"])
def test_translated_plane_contains_points(factory, scale):
    point = np.array([2.0, 3.0, 4.0])
    tangent1 = np.array([1.0, -1.0, 0.0])
    tangent2 = np.array([0.0, 1.0, -1.0])
    normal = np.cross(tangent1, tangent2)
    if factory == "coefficients":
        plane = Plane3(scale * np.r_[normal, -normal @ point])
    elif factory == "point-normal":
        plane = Plane3.PointNormal(point, scale * normal)
    else:
        plane = Plane3.ThreePoints(
            np.column_stack((point, point + scale * tangent1, point + tangent2))
        )

    for candidate in (point, point + tangent1, point + tangent2):
        assert plane.contains(candidate)
    assert not plane.contains(point + normal)
    # The old sign incorrectly accepted the reflected plane.
    assert not plane.contains(-point)


def test_plane_contains_tolerance():
    plane = Plane3([1, 0, 0, -2])
    assert plane.contains([2 + 1e-10, 0, 0], tol=1e6)
    assert not plane.contains([2 + 1e-10, 0, 0])


def test_plane_through_origin():
    plane = Plane3([0, 0, 1, 0])
    assert plane.contains([2, 3, 0])
    assert not plane.contains([2, 3, 1])
