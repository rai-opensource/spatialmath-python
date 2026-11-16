import numpy as np
import pytest
from spatialmath import Line3


@pytest.mark.parametrize("scale", [1, 3, -1, -3, 1e-10, -1e-10, 1e10, -1e10])
@pytest.mark.parametrize(
    "point,direction",
    [([0, 0, 0], [1, 0, 0]), ([1, 2, 3], [2, -1, 1]), ([-2, 4, 1], [0, 1, 2])],
)
def test_line_equality_homogeneous_sign(point, direction, scale):
    line = Line3.PointDir(point, direction)
    equivalent = Line3(scale * line.vec)
    for first, second in [(line, equivalent), (equivalent, line)]:
        assert first.isequal(second)
        assert first == second
        assert not first != second


def test_line_equality_reversed_endpoints():
    first, second = [1, 2, 3], [4, -1, 5]
    assert Line3.Join(first, second) == Line3.Join(second, first)


@pytest.mark.parametrize("scale", [1, -1, 3, -3])
@pytest.mark.parametrize("kind", ["parallel", "intersecting", "skew"])
def test_line_equality_distinct_lines(kind, scale):
    line = Line3.PointDir([0, 0, 0], [1, 0, 0])
    point, direction = {
        "parallel": ([0, 1, 0], [1, 0, 0]),
        "intersecting": ([0, 0, 0], [0, 1, 0]),
        "skew": ([0, 0, 1], [0, 1, 0]),
    }[kind]
    other = Line3.PointDir(point, np.array(direction) * scale)
    assert not line.isequal(other)
    assert not line == other
    assert line != other
