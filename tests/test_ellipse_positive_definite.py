"""An ellipse's quadratic form must be positive definite, not just det > 0."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import Ellipse


@pytest.mark.parametrize("eigenvalues", [(-1, -1), (-1, -4), (-0.25, -9)])
@pytest.mark.parametrize("theta", [0, 0.4, -0.7])
def test_reject_negative_definite_matrix(eigenvalues, theta):
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    matrix = rotation @ np.diag(eigenvalues) @ rotation.T
    assert np.linalg.det(matrix) > 0
    with pytest.raises(ValueError, match="positive definite"):
        Ellipse(E=matrix)


@pytest.mark.parametrize("theta", [0, 0.4, -0.7])
def test_positive_definite_matrix_retains_geometry(theta):
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    matrix = rotation @ np.diag([0.25, 1 / 9]) @ rotation.T
    ellipse = Ellipse(E=matrix, centre=(3, -4))
    nt.assert_array_equal(ellipse.E, matrix)
    nt.assert_allclose(np.sort(ellipse.radii), [2, 3])
    nt.assert_allclose(ellipse.area, 6 * np.pi)


@pytest.mark.parametrize(
    "matrix", [np.zeros((2, 2)), np.diag([1, 0]), np.diag([1, -1])]
)
def test_reject_degenerate_and_indefinite_matrix(matrix):
    with pytest.raises(ValueError):
        Ellipse(E=matrix)
