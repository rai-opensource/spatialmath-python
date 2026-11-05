"""Spline fitting must compare poses at corresponding trajectory times."""

import numpy as np
import numpy.testing as nt
import pytest

from spatialmath import SE3, SplineFit


@pytest.fixture(autouse=True)
def deterministic_removal_order(monkeypatch):
    # Keep the candidate order identical when comparing parameterizations.
    monkeypatch.setattr(np.random, "shuffle", lambda values: values.reverse())


@pytest.mark.parametrize("check_type", ["local", "global"])
@pytest.mark.parametrize(
    "start, duration", [(0, 1), (0, 5), (10, 10), (1e6, 10), (0, 0.01), (-3, 7)]
)
def test_normalized_linear_trajectory(check_type, start, duration):
    phase = np.linspace(0, 1, 5)
    times = start + duration * phase
    poses = [SE3.Trans([2 * u, -u, 0.5 * u]) * SE3.Rz(0.3 * u) for u in phase]
    fit = SplineFit(times.tolist(), poses)
    spline, kept = fit.stochastic_downsample_interpolation(
        check_type=check_type, epsilon_xyz=1e-7, epsilon_angle=1e-7
    )

    assert kept == [0, len(poses) - 1]
    nt.assert_array_equal(spline.timepoints, [0, 1])
    nt.assert_array_equal(fit.time_data, times)
    for u, pose in zip(phase, poses):
        nt.assert_allclose(spline(u).A, pose.A, atol=1e-12)
    assert fit.max_euclidean_error() < 1e-7
    assert fit.max_angular_error() < 1e-7
    assert len(fit.euclidean_errors()) == len(poses)
    assert len(fit.angular_errors()) == len(poses)


@pytest.mark.parametrize("check_type", ["local", "global"])
@pytest.mark.parametrize(
    "start, duration", [(0, 1), (0, 5), (10, 10), (1e6, 10), (0, 0.01), (-3, 7)]
)
def test_normalization_preserves_downsampling_decisions(check_type, start, duration):
    phase = np.linspace(0, 1, 9)
    times = start + duration * phase
    poses = [
        SE3.Trans([u, np.sin(2 * np.pi * u), u**2])
        * SE3.Rz(0.6 * np.sin(2 * np.pi * u))
        for u in phase
    ]
    fit = SplineFit(times.tolist(), poses)
    reference = SplineFit(times.tolist(), poses)
    raw_spline, raw_kept = reference.stochastic_downsample_interpolation(
        normalize_time=False,
        check_type=check_type,
        epsilon_xyz=0.15,
        epsilon_angle=0.1,
    )
    # Reusing the fitter must update its diagnostic time domain in both directions.
    for normalize_time in [True, False, True]:
        spline, kept = fit.stochastic_downsample_interpolation(
            normalize_time=normalize_time,
            check_type=check_type,
            epsilon_xyz=0.15,
            epsilon_angle=0.1,
        )
        sample_times = phase if normalize_time else times
        assert kept == raw_kept
        assert 2 < len(kept) < len(poses)
        nt.assert_allclose(
            fit.euclidean_errors(), reference.euclidean_errors(), atol=1e-9
        )
        nt.assert_allclose(fit.angular_errors(), reference.angular_errors(), atol=1e-7)
        manual_xyz = []
        manual_angle = []
        for t, raw_t, pose in zip(sample_times, times, poses):
            nt.assert_allclose(spline(t).A, raw_spline(raw_t).A, atol=1e-9)
            manual_xyz.append(np.linalg.norm(pose.t - spline(t).t))
            manual_angle.append(pose.angdist(spline(t)))
        nt.assert_allclose(fit.euclidean_errors(), manual_xyz, atol=1e-9)
        nt.assert_allclose(fit.angular_errors(), manual_angle, atol=1e-7)
        for index, control_pose in zip(kept, spline.control_poses):
            nt.assert_array_equal(control_pose.A, poses[index].A)
        if check_type == "global":
            assert fit.max_euclidean_error() <= 0.15
            assert fit.max_angular_error() <= 0.1
    nt.assert_array_equal(fit.time_data, times)


@pytest.mark.parametrize("check_type", ["local", "global"])
def test_normalized_two_pose_diagnostics(check_type):
    poses = [SE3.Tx(1), SE3.Tx(3) * SE3.Ry(0.5)]
    fit = SplineFit([10, 20], poses)
    spline, kept = fit.stochastic_downsample_interpolation(check_type=check_type)
    assert kept == [0, 1]
    nt.assert_allclose(fit.euclidean_errors(), [0, 0], atol=1e-12)
    nt.assert_allclose(fit.angular_errors(), [0, 0], atol=1e-7)
    nt.assert_array_equal(spline.timepoints, [0, 1])


@pytest.mark.parametrize("check_type", ["local", "global"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_irregular_array_timestamps(check_type, dtype):
    times = np.array([10, 10.125, 10.5, 12, 15], dtype=dtype)
    original_times = times.copy()
    phase = (times - times[0]) / (times[-1] - times[0])
    poses = [SE3.Tx(float(u)) * SE3.Rx(float(u) * 0.25) for u in phase]
    fit = SplineFit(times, poses)
    spline, kept = fit.stochastic_downsample_interpolation(
        check_type=check_type, epsilon_xyz=1e-7, epsilon_angle=1e-7
    )
    assert kept == [0, len(poses) - 1]
    nt.assert_array_equal(times, original_times)
    assert times.dtype == dtype
    for t, pose in zip(phase, poses):
        assert spline(t).A.shape == (4, 4)
        nt.assert_allclose(spline(t).A, pose.A, atol=1e-12)
    nt.assert_allclose(fit.euclidean_errors(), np.zeros(len(poses)), atol=1e-12)
    nt.assert_allclose(fit.angular_errors(), np.zeros(len(poses)), atol=1e-7)
