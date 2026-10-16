import numpy as np
import pytest
from scipy.linalg import expm

from spatialmath import Twist2, Twist3


@pytest.mark.parametrize("speed", [-2.0, -0.3, 0.3, 1.0, 2.0])
def test_twist2_pole_is_rotation_center(speed):
    """A change in angular speed must not move the center of rotation."""
    center = np.array([3.0, -2.0])
    twist = Twist2(speed * np.r_[center[1], -center[0], 1])
    np.testing.assert_allclose(twist.pole, center)
    transform = expm(twist.skewa() * 0.7)
    np.testing.assert_allclose(
        transform @ np.r_[twist.pole, 1], np.r_[center, 1], atol=1e-14
    )


@pytest.mark.parametrize("speed", [-2.0, -0.3, 0.3, 1.0, 2.0])
@pytest.mark.parametrize("pitch", [0, 0.4])
def test_twist3_pole_is_on_screw_axis(speed, pitch):
    """The closest axis point is fixed except for translation along the axis."""
    axis = np.array([1.0, 2.0, -2.0]) / 3
    center = np.array([3.0, -2.0, 1.0])
    center -= (center @ axis) * axis
    angular = speed * axis
    linear = -np.cross(angular, center) + pitch * angular
    twist = Twist3(linear, angular)
    np.testing.assert_allclose(twist.pole, center)
    transform = expm(twist.skewa() * 0.7)
    expected = center + pitch * angular * 0.7
    np.testing.assert_allclose(
        (transform @ np.r_[twist.pole, 1])[:3], expected, atol=1e-14
    )
