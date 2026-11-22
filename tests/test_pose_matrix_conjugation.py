import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from spatialmath import SO2, SE2, SO3, SE3
from spatialmath.base import skewa, skew


POSES = [
    SO2(0.4),
    SO3.RPY([0.2, -0.3, 0.4]),
    SE2(1, -2, 0),
    SE2(1, -2, 0.4),
    SE3(1, -2, 3),
    SE3(1, -2, 3) * SE3.RPY([0.2, -0.3, 0.4]),
]


@pytest.mark.parametrize("pose", POSES)
@pytest.mark.parametrize("kind", ["identity", "dense", "diagonal"])
def test_matrix_conjugation_matches_similarity_transform(pose, kind):
    n = pose.shape[0]
    if kind == "identity":
        matrix = np.eye(n)
    elif kind == "diagonal":
        matrix = np.diag(np.arange(1.0, n + 1))
    else:
        matrix = np.arange(n * n, dtype=float).reshape(n, n) / 3
    before, pose_before = matrix.copy(), pose.A.copy()
    expected = pose.A @ matrix @ np.linalg.inv(pose.A)
    actual = pose.conjugation(matrix)
    assert_allclose(actual, expected, atol=2e-14)
    assert_allclose(pose.inv().conjugation(actual), matrix, atol=5e-14)
    assert_array_equal(matrix, before)
    assert_array_equal(pose.A, pose_before)


@pytest.mark.parametrize("pose", POSES)
def test_conjugation_preserves_lie_algebra_and_matches_adjoint(pose):
    if pose.isSE:
        v = (
            np.array([0.3, -0.2, 0.5])
            if pose.N == 2
            else np.array([0.3, -0.2, 0.5, -0.4, 0.6, 0.2])
        )
        matrix = skewa(v)
        expected = skewa(pose.Ad() @ v)
    else:
        matrix = skew(0.3 if pose.N == 2 else [0.3, -0.2, 0.5])
        expected = pose.A @ matrix @ pose.A.T
    actual = pose.conjugation(matrix)
    assert_allclose(actual, expected, atol=2e-14)
    if pose.isSE:
        assert_allclose(actual[-1], 0, atol=1e-15)
    else:
        assert_allclose(actual + actual.T, 0, atol=1e-15)
