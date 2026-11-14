"""Quaternion sign cannot change the shortest rotation distance."""

import numpy as np
import pytest
from spatialmath.base import qangle


@pytest.mark.parametrize("theta", [0, 0.3, 2.0, np.pi])
@pytest.mark.parametrize("signs", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_shortest_angle_is_sign_invariant(theta, signs):
    # Unit quaternions about the same oblique axis, with known separation.
    axis = np.array([1, -2, 3], dtype=float) / np.sqrt(14)
    start = 0.7
    first = np.r_[np.cos(start / 2), np.sin(start / 2) * axis]
    second = np.r_[np.cos((start + theta) / 2), np.sin((start + theta) / 2) * axis]
    first *= signs[0]
    second *= signs[1]
    assert qangle(first, second) == pytest.approx(theta, abs=1e-14)
    assert qangle(second, first) == pytest.approx(theta, abs=1e-14)


@pytest.mark.parametrize("theta", [1e-10, 2 * np.pi - 0.2, 2 * np.pi])
def test_angle_uses_shortest_equivalent_rotation(theta):
    first = [1, 0, 0, 0]
    second = [np.cos(theta / 2), np.sin(theta / 2), 0, 0]
    expected = min(theta, 2 * np.pi - theta)
    assert qangle(first, second) == pytest.approx(expected, abs=1e-14)
