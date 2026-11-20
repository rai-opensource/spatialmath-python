import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from spatialmath import SO3, SE3
from spatialmath.base import trnorm, unitvec, rotx, roty, rotz


def previous_normalization(matrix):
    o, a = matrix[:3, 1], matrix[:3, 2]
    n = np.cross(o, a)
    o = np.cross(a, n)
    return np.stack((unitvec(n), unitvec(o), unitvec(a)), axis=1)


@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize(
    "rotation",
    [
        np.zeros((3, 3)),
        np.array([[1.0, 1, 1], [0, 0, 0], [0, 0, 0]]),
        np.diag([1.0, 0, 1]),
        np.diag([1.0, 1, 0]),
        np.diag([1.0, 1e-16, 1]),
        np.diag([1.0, 1, 1e-8]),
    ],
)
def test_degenerate_axes_raise_without_runtime_warning(rotation, homogeneous):
    matrix = np.eye(4) if homogeneous else rotation.copy()
    if homogeneous:
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [1, -2, 3]
    original = matrix.copy()
    with pytest.raises(ValueError):
        previous_normalization(matrix)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="zero norm vector"):
            trnorm(matrix)
    assert_array_equal(matrix, original)


@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.2, -1.3, np.pi])
def test_valid_drift_matches_previous_algorithm(angle, homogeneous):
    rotation = rotx(angle) @ roty(0.4) @ rotz(-0.7)
    rotation[:, 1] *= 1.0001
    rotation[:, 2] += 1e-5 * rotation[:, 1]
    matrix = np.eye(4) if homogeneous else rotation.copy()
    if homogeneous:
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [1, -2, 3]
    original = matrix.copy()
    actual = trnorm(matrix)
    assert_allclose(actual[:3, :3], previous_normalization(matrix), atol=1e-15)
    assert_allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-15)
    assert_allclose(np.linalg.det(actual[:3, :3]), 1, atol=1e-15)
    if homogeneous:
        assert_array_equal(actual[:3, 3], matrix[:3, 3])
        assert_array_equal(actual[3], [0, 0, 0, 1])
    assert_array_equal(matrix, original)


@pytest.mark.parametrize("cls, size", [(SO3, 3), (SE3, 4)])
def test_pose_norm_propagates_failure(cls, size):
    matrix = np.eye(size)
    matrix[:3, :3] = 0
    pose = cls(matrix, check=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError):
            pose.norm()
