"""Preserve unit status only when both product operands are unit."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import DualQuaternion, Quaternion, SE3, UnitDualQuaternion


@pytest.mark.parametrize("pose", [SE3(), SE3(1, 2, 3), SE3.Rx(0.7) * SE3(2, -1, 3)])
@pytest.mark.parametrize("unit_on_left", [True, False])
def test_mixed_product_is_general(pose, unit_on_left):
    unit = UnitDualQuaternion(pose)
    general = DualQuaternion(Quaternion([2, 1, -3, 4]), Quaternion([-1, 2, 0, 3]))
    left, right = (unit, general) if unit_on_left else (general, unit)
    product = left * right
    assert type(product) is DualQuaternion
    nt.assert_allclose(product.vec, left.matrix() @ right.vec, atol=1e-14)
    # A nonunit product must not acquire the unit-only point transform path.
    assert product.__mul__(np.array([1.0, 2.0, 3.0])) is None


@pytest.mark.parametrize("pose", [SE3(), SE3(1, 2, 3), SE3.Ry(-0.4)])
def test_unit_products_keep_unit_type_and_transform(pose):
    left = UnitDualQuaternion(pose)
    right = UnitDualQuaternion(SE3.Rz(0.3) * SE3(-2, 1, 0))
    product = left * right
    assert type(product) is UnitDualQuaternion
    nt.assert_allclose(product.SE3().A, (pose * right.SE3()).A, atol=1e-14)


def test_general_product_stays_general():
    left = DualQuaternion(Quaternion([2, 1, -3, 4]), Quaternion([-1, 2, 0, 3]))
    right = DualQuaternion(Quaternion([3, -2, 1, 0]), Quaternion([1, 0, 2, -1]))
    product = left * right
    assert type(product) is DualQuaternion
    nt.assert_allclose(product.vec, left.matrix() @ right.vec, atol=1e-14)
