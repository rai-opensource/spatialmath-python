"""Closest-point queries must return finite results for parallel lines."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import Line3


@pytest.mark.parametrize("direction", [[1, 0, 0], [1, 2, -1]])
@pytest.mark.parametrize("scales", [(1, 1), (2, -3), (1e-8, -1e8)])
@pytest.mark.parametrize("coincident", [False, True])
def test_parallel_closest_point(direction, scales, coincident):
    direction = np.array(direction, dtype=float)
    first_point = np.array([1.0, 2.0, 3.0])
    offset = np.zeros(3) if coincident else np.array([2.0, -1.0, 4.0])
    first = Line3.PointDir(first_point, scales[0] * direction)
    second = Line3.PointDir(first_point + 3 * direction + offset, scales[1] * direction)
    for a, b in [(first, second), (second, first)]:
        point, distance = a.closest_to_line(b)
        assert point.shape == (3,)
        assert np.all(np.isfinite(point))
        nt.assert_allclose(
            np.cross(point - first_point, direction)
            if a is first
            else np.cross(point - first_point - offset, direction),
            0,
            atol=1e-12,
        )
        # Independent Euclidean projection onto the second line.
        delta = (
            first_point + 3 * direction + offset - point
            if a is first
            else first_point - point
        )
        perpendicular = delta - direction * (delta @ direction) / (
            direction @ direction
        )
        assert distance == pytest.approx(np.linalg.norm(perpendicular), abs=1e-12)
        assert distance == pytest.approx(a.distance(b), abs=1e-12)


def test_parallel_closest_sequence_broadcast():
    a = Line3.PointDir([1, 2, 3], [1, 0, 0])
    b = Line3.PointDir([0, 5, 7], [-2, 0, 0])
    sequence = Line3([a, b])
    for left, right in [(a, sequence), (sequence, a), (sequence, sequence)]:
        points, distances = left.closest_to_line(right)
        assert points.shape == (3, 2)
        assert np.all(np.isfinite(points))
        nt.assert_allclose(distances, [0, 0] if left is right else [0, 5], atol=1e-12)


@pytest.mark.parametrize("second_point", [[1, 2, 3], [1, 2, 7]])
@pytest.mark.parametrize("scale", [1, -3, 1e-8])
def test_nonparallel_closest_unchanged(second_point, scale):
    first = Line3.PointDir([1, 2, 3], [scale, 0, 0])
    second = Line3.PointDir(second_point, [scale, scale, 0])
    point, distance = first.closest_to_line(second)
    nt.assert_allclose(point, [1, 2, 3], atol=1e-12)
    assert distance == pytest.approx(abs(second_point[2] - 3), abs=1e-12)
