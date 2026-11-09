import numpy as np
import numpy.testing as nt
import pytest

from spatialmath import Ellipse


@pytest.mark.parametrize("radii", [(1, 2), (2.0, 0.5), np.array([3.0, 4.0])])
@pytest.mark.parametrize("centre", [(0, 0), (3, -2)])
@pytest.mark.parametrize("orientation", ["omitted", None, 0.0, 0.4])
def test_radii_constructor_orientation(radii, centre, orientation):
    kwargs = {} if orientation == "omitted" else {"theta": orientation}
    ellipse = Ellipse(radii=radii, centre=centre, **kwargs)
    theta = 0.0 if orientation in ("omitted", None) else orientation
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    # Constructor convention: theta rotates the principal ellipse axes.
    expected = rotation @ np.diag(np.asarray(radii, dtype=float) ** -2) @ rotation.T
    nt.assert_allclose(ellipse.E, expected, atol=1e-14)
    nt.assert_allclose(ellipse.centre, centre)
    assert ellipse.contains(centre)
    nt.assert_allclose(ellipse.area, np.pi * np.prod(radii))
