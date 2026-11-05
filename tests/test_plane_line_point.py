import numpy as np
import pytest

from spatialmath import Line3, Plane3


@pytest.mark.parametrize(
    "anchor,direction,point",
    [
        ([1, 2, 3], [1, 0, 0], [1, 4, 3]),
        ([2, -1, 4], [1, 2, -1], [3, 2, 5]),
        ([0, 0, 0], [0, 0, 1], [2, 3, 4]),
    ],
)
@pytest.mark.parametrize("scale", [1, -2, 0.25])
def test_line_point_plane_contains_translated_line(anchor, direction, point, scale):
    anchor = np.asarray(anchor, dtype=float)
    direction = np.asarray(direction, dtype=float) * scale
    line = Line3.PointDir(anchor, direction)
    plane = Plane3.LinePoint(line, np.asarray(point, dtype=float))

    # Use the plane equation directly, independent of Plane3.contains.
    for p in [point, anchor, anchor + direction, anchor - 3 * direction]:
        np.testing.assert_allclose(plane.n @ p + plane.d, 0, atol=1e-12)
    expected_normal = np.cross(direction, np.asarray(point) - anchor)
    np.testing.assert_allclose(plane.n, expected_normal, atol=1e-12)
