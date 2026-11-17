import numpy as np
import numpy.testing as nt
import pytest

from spatialmath import (
    SE3,
    Twist3,
    SpatialVelocity,
    SpatialAcceleration,
    SpatialForce,
    SpatialMomentum,
)


@pytest.mark.parametrize(
    "cls", [SpatialVelocity, SpatialAcceleration, SpatialForce, SpatialMomentum]
)
@pytest.mark.parametrize("pose_type", [SE3, Twist3])
@pytest.mark.parametrize(
    "lengths",
    [(1, 1), (1, 2), (2, 1), (2, 2), (1, 6), (6, 1), (6, 6), (1, 0), (0, 1), (0, 0)],
)
def test_spatial_transform_broadcast(cls, pose_type, lengths):
    npose, nvector = lengths
    poses = [SE3(0.2 * i, 1, -2) * SE3.RPY(0.1 * i, -0.3, 0.2) for i in range(npose)]
    pose = SE3.Empty()
    for item in poses:
        pose.append(item)
    if pose_type is Twist3:
        twist = Twist3.Empty()
        for item in pose:
            twist.append(Twist3(item))
        pose = twist
    vectors = [np.arange(1, 7) + i for i in range(nvector)]
    vector = cls.Empty()
    for v in vectors:
        vector.append(cls(v))
    result = pose * vector
    assert isinstance(result, cls)
    count = 0 if npose == 0 or nvector == 0 else max(npose, nvector)
    assert len(result) == count
    for i in range(count):
        t = poses[0 if npose == 1 else i]
        v = vectors[0 if nvector == 1 else i]
        adjoint = t.Ad()
        expected = (
            adjoint @ v
            if cls in (SpatialVelocity, SpatialAcceleration)
            else np.linalg.solve(adjoint.T, v)
        )
        nt.assert_allclose(result[i].A, expected, atol=1e-12)
        assert result[i].A.shape == (6,)


@pytest.mark.parametrize(
    "cls", [SpatialVelocity, SpatialAcceleration, SpatialForce, SpatialMomentum]
)
@pytest.mark.parametrize("pose_type", [SE3, Twist3])
def test_spatial_transform_incompatible_lengths(cls, pose_type):
    pose = SE3([SE3(), SE3.Tx(1)])
    if pose_type is Twist3:
        twist = Twist3.Empty()
        for item in pose:
            twist.append(Twist3(item))
        pose = twist
    with pytest.raises(ValueError):
        pose * cls(np.arange(18).reshape(6, 3))


def test_spatial_transform_sequence_power_invariant():
    transforms = SE3([SE3(1, 2, 3), SE3.RPY(0.3, -0.2, 0.4) * SE3.Tx(-2)])
    values = np.arange(12).reshape(6, 2)
    velocity, force = SpatialVelocity(values), SpatialForce(values + 3)
    transformed_velocity = transforms * velocity
    transformed_force = transforms * force
    for i in range(2):
        nt.assert_allclose(
            transformed_force[i].A @ transformed_velocity[i].A,
            force[i].A @ velocity[i].A,
        )
