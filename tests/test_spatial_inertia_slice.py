import numpy as np
import pytest
from numpy.testing import assert_allclose
from spatialmath import SpatialInertia, SpatialVelocity, SpatialAcceleration


@pytest.mark.parametrize("count", [0, 1, 3, 6])
@pytest.mark.parametrize(
    "selection",
    [
        slice(None),
        slice(0, 1),
        slice(0, 0),
        slice(None, None, -1),
        slice(1, None, 2),
        slice(-2, None),
        slice(99, 100),
    ],
)
def test_slice_matches_python_list(count, selection):
    inertia = SpatialInertia.Empty()
    for k in range(count):
        inertia.append(SpatialInertia(m=k + 1, r=[0.1 * k, 0.2, -0.3], I=np.eye(3)))
    original = [a.copy() for a in inertia.data]
    sliced = inertia[selection]
    assert type(sliced) is SpatialInertia
    assert sliced.data is not inertia.data
    expected = inertia.data[selection]
    assert len(sliced) == len(expected)
    for actual, reference in zip(sliced.data, expected):
        assert_allclose(actual, reference)
    sliced.append(SpatialInertia())
    assert len(inertia) == count
    for actual, reference in zip(inertia.data, original):
        assert_allclose(actual, reference)


@pytest.mark.parametrize("index", [0, 1, -1])
def test_integer_index_and_dynamics(index):
    inertia = SpatialInertia.Empty()
    for m in [1, 2, 3]:
        inertia.append(SpatialInertia(m=m, r=[1, 2, 3], I=np.eye(3)))
    assert_allclose(inertia[index].A, inertia.data[index])
    one = inertia[index : index + 1] if index >= 0 else inertia[-1:]
    for vector_type in [SpatialVelocity, SpatialAcceleration]:
        vector = vector_type([1, 2, 3, 4, 5, 6])
        assert_allclose((one * vector).A, inertia.data[index] @ vector.A)


def test_invalid_index_still_raises():
    inertia = SpatialInertia()
    with pytest.raises(IndexError):
        inertia[1]
    with pytest.raises(TypeError):
        inertia["bad"]
    with pytest.raises(ValueError):
        inertia[::0]
