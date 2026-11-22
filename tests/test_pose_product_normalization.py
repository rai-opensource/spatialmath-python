import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from spatialmath import SO2, SE2, SO3, SE3
from spatialmath.base import trnorm, trnorm2


def poses(cls):
    if cls is SO2:
        return [SO2(0.2), SO2(-0.5), SO2(0.8)]
    if cls is SE2:
        return [SE2(1, 2, 0.2), SE2(-3, 1, -0.5), SE2(2, -4, 0.8)]
    if cls is SO3:
        return [SO3.Rx(0.2), SO3.Ry(-0.5), SO3.Rz(0.8)]
    return [
        SE3(1, 2, 3) * SE3.Rx(0.2),
        SE3(-3, 1, -2) * SE3.Ry(-0.5),
        SE3(2, -4, 1) * SE3.Rz(0.8),
    ]


@pytest.mark.parametrize("cls", [SO2, SE2, SO3, SE3])
@pytest.mark.parametrize("length", [0, 1, 3, 90])
@pytest.mark.parametrize("check", [False, True])
def test_normalized_product_dispatches_by_dimension(cls, length, check):
    entries = poses(cls)
    seq = cls([entries[i % 3] for i in range(length)]) if length else cls.Empty()
    original = [a.copy() for a in seq.data]
    raw = np.eye(cls().shape[0])
    for a in seq.data:
        raw = raw @ a
    normalizer = trnorm2 if seq.N == 2 else trnorm
    expected = normalizer(raw)
    actual = seq.prod(norm=True, check=check)
    assert type(actual) is cls
    assert len(actual) == 1
    assert cls.isvalid(actual.A)
    assert_allclose(actual.A, expected, atol=1e-14)
    if seq.isSE:
        assert_array_equal(actual.A[:-1, -1], raw[:-1, -1])
        assert_array_equal(actual.A[-1], np.r_[np.zeros(seq.N), 1])
    for before, after in zip(original, seq.data):
        assert_array_equal(before, after)


@pytest.mark.parametrize("cls", [SO2, SE2, SO3, SE3])
def test_unnormalized_path_is_unchanged(cls):
    seq = cls(poses(cls))
    expected = np.linalg.multi_dot(seq.data)
    assert_allclose(seq.prod(norm=False, check=False).A, expected, atol=1e-14)
