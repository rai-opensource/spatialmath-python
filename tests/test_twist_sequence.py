import numpy as np
import pytest
from scipy.linalg import expm

from spatialmath import SE2, SE3, Twist2, Twist3


@pytest.mark.parametrize("twist,pose", [(Twist2, SE2), (Twist3, SE3)])
@pytest.mark.parametrize("method", ["exp", "pose"])
@pytest.mark.parametrize("theta", [None, 0, -0.4, [0.2, -0.7]])
def test_twist_sequence_exponential(twist, pose, method, theta):
    """A shared step and individual steps both integrate every twist."""
    vectors = (
        [[1, -2, 0.3], [2, 1, -0.4]]
        if twist is Twist2
        else [[1, -2, 3, 0.2, 0.3, -0.4], [2, 1, -1, -0.1, 0.5, 0.2]]
    )
    sequence = twist([np.array(v) for v in vectors])
    convert = getattr(sequence, "exp" if method == "exp" else pose.__name__)
    result = convert() if theta is None else convert(theta)
    steps = np.broadcast_to(1 if theta is None else theta, (2,))
    assert isinstance(result, pose)
    assert len(result) == 2
    for value, source, step in zip(result, sequence, steps):
        np.testing.assert_allclose(value.A, expm(source.skewa() * step), atol=1e-14)


@pytest.mark.parametrize("twist,pose", [(Twist2, SE2), (Twist3, SE3)])
@pytest.mark.parametrize("method", ["exp", "pose"])
def test_twist_sequence_degrees_and_singleton(twist, pose, method):
    """Degrees, singleton expansion and mismatched lengths retain their rules."""
    source = twist([1, 2, 0.3] if twist is Twist2 else [1, 2, 3, 0.2, 0.3, 0.4])
    name = "exp" if method == "exp" else pose.__name__
    angles = [15, -30]
    expanded = getattr(source, name)(angles, unit="deg")
    assert len(expanded) == 2
    for value, angle in zip(expanded, angles):
        np.testing.assert_allclose(
            value.A, expm(source.skewa() * np.deg2rad(angle)), atol=1e-14
        )
    sequence = twist([source, source])
    shared = getattr(sequence, name)(30, unit="deg")
    assert len(shared) == 2
    for value in shared:
        np.testing.assert_allclose(
            value.A, expm(source.skewa() * np.deg2rad(30)), atol=1e-14
        )
    with pytest.raises(ValueError):
        getattr(sequence, name)([0.1, 0.2, 0.3])
