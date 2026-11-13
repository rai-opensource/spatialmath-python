"""Row and column vectors must use the same flat internal representation."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import (
    SpatialVelocity,
    SpatialAcceleration,
    SpatialForce,
    SpatialMomentum,
)


@pytest.mark.parametrize(
    "cls", [SpatialVelocity, SpatialAcceleration, SpatialForce, SpatialMomentum]
)
@pytest.mark.parametrize("size", [3, 6])
@pytest.mark.parametrize("shape", ["flat", "row", "column"])
def test_shaped_vector_constructor(cls, size, shape):
    values = np.arange(1, size + 1, dtype=float)
    supplied = (
        values
        if shape == "flat"
        else values.reshape((1, size) if shape == "row" else (size, 1))
    )
    original = supplied.copy()
    vector = cls(supplied)
    expected = values if size == 6 else np.r_[values, 0, 0, 0]
    assert len(vector) == 1
    assert vector.A.shape == (6,)
    assert vector.isvalid(vector.A, check=False)
    nt.assert_array_equal(vector.A, expected)
    nt.assert_array_equal(vector[0].A, expected)
    assert str(vector).startswith(cls.__name__ + "[")
    nt.assert_array_equal((-vector).A, -expected)
    nt.assert_array_equal(supplied, original)
