import contextlib
import io
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from spatialmath.base import ICP2d, homtrans, xyt2tr
from scipy.spatial import KDTree


SOURCE = np.array([[0.0, 10, 0, 10, 20, -10], [0.0, 0, 10, 10, 5, 15]])


@pytest.mark.parametrize("pose", [[1.0, 2, 0.2], [-3.0, 1, -0.4], [2.0, -3, 0.8]])
@pytest.mark.parametrize(
    "delta",
    [[0.05, -0.03, 0.02], [-0.04, 0.02, -0.03], [0.0, 0.0, 0.025], [0.1, -0.1, 0.0]],
)
def test_one_step_applies_reference_frame_correction(pose, delta):
    initial = xyt2tr(pose)
    correction = xyt2tr(delta)
    expected = correction @ initial
    reference = homtrans(expected, SOURCE)
    # Widely spaced, asymmetric points guarantee exact nearest-neighbor matches.
    distances, indices = KDTree(reference.T).query(homtrans(initial, SOURCE).T)
    assert_array_equal(indices, np.arange(SOURCE.shape[1]))
    source_before, reference_before, initial_before = (
        SOURCE.copy(),
        reference.copy(),
        initial.copy(),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        actual = ICP2d(reference, SOURCE, T=initial, max_iter=1)
    assert_allclose(actual, expected, atol=2e-14)
    assert_allclose(homtrans(actual, SOURCE), reference, atol=5e-13)
    assert_array_equal(SOURCE, source_before)
    assert_array_equal(reference, reference_before)
    assert_array_equal(initial, initial_before)


@pytest.mark.parametrize("initial_pose", [[0.0, 0, 0], [1.0, 2, 0.2], [-3.0, 1, -0.4]])
def test_repeated_steps_recover_known_transform(initial_pose):
    initial = xyt2tr(initial_pose)
    expected = xyt2tr([0.05, -0.03, 0.02]) @ initial
    reference = homtrans(expected, SOURCE)
    with contextlib.redirect_stdout(io.StringIO()):
        actual = ICP2d(reference, SOURCE, T=initial, max_iter=20)
    assert_allclose(actual, expected, atol=3e-14)


@pytest.mark.parametrize("pose", [[0.0, 0, 0], [1.0, 2, 0.2]])
def test_already_aligned_cloud_is_unchanged(pose):
    initial = xyt2tr(pose)
    reference = homtrans(initial, SOURCE)
    with contextlib.redirect_stdout(io.StringIO()):
        actual = ICP2d(reference, SOURCE, T=initial, max_iter=1)
    assert_allclose(actual, initial, atol=2e-14)


def test_default_identity_initial_pose():
    expected = xyt2tr([0.05, -0.03, 0.02])
    with contextlib.redirect_stdout(io.StringIO()):
        actual = ICP2d(homtrans(expected, SOURCE), SOURCE, max_iter=1)
    assert_allclose(actual, expected, atol=2e-14)
