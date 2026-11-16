import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import Ellipse


@pytest.mark.parametrize("resolution", [3, 4, 5, 10, 20, 51])
@pytest.mark.parametrize(
    "radii,theta,centre",
    [((1, 1), 0, (0, 0)), ((1, 2), 0.6, (3, -4)), ((3, 0.5), -0.4, (-2, 1))],
)
def test_ellipse_polygon_vertex_count(resolution, radii, theta, centre):
    ellipse = Ellipse(radii=radii, theta=theta, centre=centre)
    polygon = ellipse.polygon(resolution=resolution)
    vertices = polygon.vertices()
    assert len(polygon) == resolution
    assert vertices.shape == (2, resolution)
    closed = polygon.vertices(unique=False)
    assert closed.shape == (2, resolution + 1)
    nt.assert_array_equal(closed[:, 0], closed[:, -1])
    nt.assert_allclose(vertices, ellipse.points(resolution=resolution))
    offsets = vertices - np.array(centre)[:, None]
    nt.assert_allclose(np.einsum("in,ij,jn->n", offsets, ellipse.E, offsets), 1)
    expected_area = resolution * np.prod(radii) * np.sin(2 * np.pi / resolution) / 2
    assert polygon.area() == pytest.approx(expected_area)


def test_ellipse_polygon_default_resolution():
    assert len(Ellipse(E=np.eye(2)).polygon()) == 10
