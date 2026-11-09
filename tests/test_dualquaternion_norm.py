import numpy as np
import numpy.testing as nt
import pytest

from spatialmath import DualQuaternion, Quaternion, SE3, UnitDualQuaternion


@pytest.mark.parametrize("real", [2.0, -2.0, 0.25])
@pytest.mark.parametrize("dual", [3.0, -3.0, 0.0])
def test_scalar_dual_quaternion_norm(real, dual):
    dq = DualQuaternion(Quaternion([real, 0, 0, 0]), Quaternion([dual, 0, 0, 0]))
    nt.assert_allclose(dq.norm(), (abs(real), np.sign(real) * dual))


@pytest.mark.parametrize("seed", range(10))
def test_norm_squares_to_product_with_conjugate(seed):
    rng = np.random.default_rng(seed)
    dq = DualQuaternion(Quaternion(rng.normal(size=4)), Quaternion(rng.normal(size=4)))
    real, dual = dq.norm()
    squared = dq * dq.conj()
    nt.assert_allclose(
        [real**2, 2 * real * dual], [squared.real.s, squared.dual.s], atol=1e-14
    )


@pytest.mark.parametrize("angle", [0.0, 0.3, -2.0])
def test_unit_norm_with_translation(angle):
    dq = UnitDualQuaternion(SE3(1, -2, 3) * SE3.Rx(angle))
    nt.assert_allclose(dq.norm(), (1, 0), atol=1e-14)


def test_zero_real_part_keeps_existing_zero_norm():
    dq = DualQuaternion(Quaternion([0, 0, 0, 0]), Quaternion([1, 2, 3, 4]))
    nt.assert_allclose(dq.norm(), (0, 0))
