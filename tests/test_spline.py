import os
import pytest

import numpy.testing as nt
import numpy as np
import matplotlib.pyplot as plt
import unittest

from spatialmath import BSplineSE3, SE3, InterpSplineSE3, SplineFit, SO3


class TestBSplineSE3(unittest.TestCase):
    control_poses = [
        SE3.Trans([e, 2 * np.cos(e / 2 * np.pi), 2 * np.sin(e / 2 * np.pi)])
        * SE3.Ry(e / 8 * np.pi)
        for e in range(0, 8)
    ]

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        BSplineSE3(self.control_poses)

    def test_evaluation(self):
        spline = BSplineSE3(self.control_poses)
        nt.assert_almost_equal(spline(0).A, self.control_poses[0].A)
        nt.assert_almost_equal(spline(1).A, self.control_poses[-1].A)

    @pytest.mark.skipif(
        plt.get_backend().lower() == "agg" or os.environ.get("CI") == "true",
        reason="animate=True busy-waits on a timer callback that never "
        "fires under the non-interactive Agg backend",
    )
    def test_visualize(self):
        spline = BSplineSE3(self.control_poses)
        spline.visualize(
            sample_times=np.linspace(0, 1.0, 100), animate=True, repeat=False
        )


class TestInterpSplineSE3:
    waypoints = [
        SE3.Trans([e, 2 * np.cos(e / 2 * np.pi), 2 * np.sin(e / 2 * np.pi)])
        * SE3.Ry(e / 8 * np.pi)
        for e in range(0, 8)
    ]
    time_horizon = 10
    times = np.linspace(0, time_horizon, len(waypoints))

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        InterpSplineSE3(self.times, self.waypoints)

    def test_evaluation(self):
        spline = InterpSplineSE3(self.times, self.waypoints)
        for time, pose in zip(self.times, self.waypoints):
            nt.assert_almost_equal(spline(time).angdist(pose), 0.0)
            nt.assert_almost_equal(np.linalg.norm(spline(time).t - pose.t), 0.0)

        spline = InterpSplineSE3(self.times, self.waypoints, normalize_time=True)
        norm_time = spline.timepoints
        for time, pose in zip(norm_time, self.waypoints):
            nt.assert_almost_equal(spline(time).angdist(pose), 0.0)
            nt.assert_almost_equal(np.linalg.norm(spline(time).t - pose.t), 0.0)

    def test_small_delta_t(self):
        InterpSplineSE3(
            np.linspace(0, InterpSplineSE3._e, len(self.waypoints)), self.waypoints
        )

    @pytest.mark.skipif(
        plt.get_backend().lower() == "agg" or os.environ.get("CI") == "true",
        reason="animate=True busy-waits on a timer callback that never "
        "fires under the non-interactive Agg backend",
    )
    def test_visualize(self):
        spline = InterpSplineSE3(self.times, self.waypoints)
        spline.visualize(
            sample_times=np.linspace(0, self.time_horizon, 100),
            animate=True,
            repeat=False,
        )


class TestSplineFit:
    num_data_points = 300
    time_horizon = 5
    num_viz_points = 100

    # make a helix
    timestamps = np.linspace(0, 1, num_data_points)
    trajectory = [
        SE3.Rt(
            t=[
                t * 0.4,
                0.4 * np.sin(t * 2 * np.pi * 0.5),
                0.4 * np.cos(t * 2 * np.pi * 0.5),
            ],
            R=SO3.Rx(t * 2 * np.pi * 0.5),
        )
        for t in timestamps * time_horizon
    ]

    def test_spline_fit(self):
        fit = SplineFit(self.timestamps, self.trajectory)
        spline, kept_indices = fit.stochastic_downsample_interpolation()

        fraction_points_removed = 1.0 - len(kept_indices) / self.num_data_points

        assert fraction_points_removed > 0.2
        assert len(spline.control_poses) == len(kept_indices)
        assert len(spline.timepoints) == len(kept_indices)

        assert fit.max_angular_error() < np.deg2rad(5.0)
        assert fit.max_angular_error() < 0.1
        # animate=True busy-waits on a timer callback that never fires
        # under the non-interactive Agg backend
        if plt.get_backend().lower() != "agg" and os.environ.get("CI") != "true":
            spline.visualize(
                sample_times=np.linspace(0, self.time_horizon, 100),
                animate=True,
                repeat=False,
            )
