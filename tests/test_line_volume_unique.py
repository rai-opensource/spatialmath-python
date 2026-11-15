import numpy as np
import numpy.testing as nt
import pytest

from spatialmath import Line3


@pytest.mark.parametrize("scale", [1, -1, 1e-8, 1e8])
@pytest.mark.parametrize("direction", [[1, 1, 1], [1, 1, 0], [1, 0, 0]])
def test_volume_unique_chord_endpoints(scale, direction):
    direction = np.array(direction, dtype=float)
    line = Line3.PointDir([0, 0, 0], scale * direction)
    result = line.intersect_volume(np.array([-1, 1, -1, 1, -1, 1]))
    assert result.p.shape == (3, 2)
    assert np.asarray(result.lam).shape == (2,)
    expected = np.column_stack((-direction, direction))
    if scale < 0:
        expected = expected[:, ::-1]
    nt.assert_allclose(result.p, expected, atol=1e-14)
    nt.assert_allclose(line.point(result.lam), result.p)
    assert result.lam[0] < result.lam[1]


def test_volume_single_vertex_tangency():
    line = Line3.PointDir([1, 1, 0], [1, -1, 0])
    result = line.intersect_volume(np.array([-1, 1, -1, 1, -1, 1]))
    nt.assert_allclose(result.p, [[1], [1], [0]])
    assert np.asarray(result.lam).shape == (1,)


def test_volume_outside():
    line = Line3.PointDir([0, 2, 0], [1, 0, 0])
    result = line.intersect_volume(np.array([-1, 1, -1, 1, -1, 1]))
    assert result.p.shape == (3, 0)
    assert len(result.lam) == 0


def test_volume_thin_chord_remains_two_points():
    line = Line3.PointDir([0, 0, 0], [1, 0, 0])
    result = line.intersect_volume(np.array([-1e-12, 1e-12, -1, 1, -1, 1]))
    assert result.p.shape == (3, 2)
    nt.assert_allclose(result.p[0], [-1e-12, 1e-12], atol=0)
    assert result.lam[0] < result.lam[1]
