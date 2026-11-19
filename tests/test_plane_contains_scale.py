import numpy as np
import pytest
from spatialmath import Plane3


@pytest.mark.parametrize("scale", [1, -1, 1e-12, -1e-12, 1e12, -1e12])
@pytest.mark.parametrize("offset, expected", [(0, True), (1e-6, False), (-1e-6, False)])
def test_homogeneous_scaling_keeps_containment(scale, offset, expected):
    plane = Plane3(scale * np.array([1.0, 0, 0, -2]))
    assert bool(plane.contains([2 + offset, 3, -4])) is expected


@pytest.mark.parametrize("scale", [1, -1, 1e-12, -1e-12, 1e12, -1e12])
@pytest.mark.parametrize("tol, expected", [(1, False), (20, True), (100, True)])
def test_distance_tolerance_is_scale_independent(scale, tol, expected):
    plane = Plane3(scale * np.array([0.0, 0, 1, 0]))
    assert bool(plane.contains([3, -4, 10 * np.finfo(float).eps], tol=tol)) is expected


@pytest.mark.parametrize("scale", [1, -1, 1e-12, 1e12])
def test_oblique_normal_uses_euclidean_distance(scale):
    normal = np.array([3.0, 4, 0])
    plane = Plane3(scale * np.r_[normal, 0])
    eps = np.finfo(float).eps
    assert plane.contains(normal / 5 * 10 * eps, tol=20)
    assert not plane.contains(normal / 5 * 40 * eps, tol=20)


@pytest.mark.parametrize("d", [0, 1])
def test_degenerate_coefficients_keep_existing_behavior(d):
    plane = Plane3([0, 0, 0, d])
    assert bool(plane.contains([1, 2, 3])) is (d == 0)
