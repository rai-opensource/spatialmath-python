# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute.
# MIT Licence, see details in top-level file: LICENCE

"""
Classes for parameterizing a trajectory in SE3 with splines.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from spatialmath import SE3, SO3, Twist3
from spatialmath.base.transforms3d import tranimate


class SplineSE3(ABC):
    def __init__(self) -> None:
        self.control_poses: SE3

    @abstractmethod
    def __call__(self, t: float) -> SE3:
        pass

    def visualize(
        self,
        sample_times: List[float],
        input_trajectory: Optional[List[SE3]] = None,
        pose_marker_length: float = 0.2,
        animate: bool = False,
        repeat: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        """Displays an animation of the trajectory with the control poses against an optional input trajectory.

        Args:
            sample_times: which times to sample the spline at and plot
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection="3d")

        samples = [self(t) for t in sample_times]
        if not animate:
            pos = np.array([pose.t for pose in samples])
            ax.plot(
                pos[:, 0], pos[:, 1], pos[:, 2], "c", linewidth=1.0
            )  # plot spline fit

        pos = np.array([pose.t for pose in self.control_poses])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "r*")  # plot control_poses

        if input_trajectory is not None:
            pos = np.array([pose.t for pose in input_trajectory])
            ax.plot(
                pos[:, 0], pos[:, 1], pos[:, 2], "go", fillstyle="none"
            )  # plot compare to input poses

        if animate:
            tranimate(
                samples, length=pose_marker_length, wait=True, repeat=repeat
            )  # animate pose along trajectory
        else:
            plt.show()


class InterpSplineSE3(SplineSE3):
    """Class for an interpolated trajectory in SE3, as a function of time, through control_poses with a cubic spline.

    A combination of scipy.interpolate.CubicSpline and scipy.spatial.transform.RotationSpline (itself also cubic)
    under the hood.
    """

    _e = 1e-12

    def __init__(
        self,
        timepoints: List[float],
        control_poses: List[SE3],
        *,
        normalize_time: bool = False,
        bc_type: str = "not-a-knot",  # not-a-knot is scipy default; None is invalid
    ) -> None:
        """Construct a InterpSplineSE3 object

        Extends the scipy CubicSpline object
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#cubicspline

        Args :
            timepoints : list of times corresponding to provided poses
            control_poses : list of SE3 objects that govern the shape of the spline.
            normalize_time : flag to map times into the range [0, 1]
            bc_type : boundary condition provided to scipy CubicSpline backend.
                      string options: ["not-a-knot" (default), "clamped", "natural", "periodic"].
                      For tuple options and details see the scipy docs link above.
        """
        super().__init__()
        self.control_poses = control_poses
        self.timepoints = np.array(timepoints)

        if self.timepoints[-1] < self._e:
            raise ValueError(
                "Difference between start and end timepoints is less than {self._e}"
            )

        if len(self.control_poses) != len(self.timepoints):
            raise ValueError("Length of control_poses and timepoints must be equal.")

        if len(self.timepoints) < 2:
            raise ValueError("Need at least 2 data points to make a trajectory.")

        if normalize_time:
            self.timepoints = self.timepoints - self.timepoints[0]
            self.timepoints = self.timepoints / self.timepoints[-1]

        self.spline_xyz = CubicSpline(
            self.timepoints,
            np.array([pose.t for pose in self.control_poses]),
            bc_type=bc_type,
        )
        self.spline_so3 = RotationSpline(
            self.timepoints,
            Rotation.from_matrix(np.array([(pose.R) for pose in self.control_poses])),
        )

    def __call__(self, t: float) -> SE3:
        """Compute function value at t.
        Return:
            pose: SE3
        """
        return SE3.Rt(t=self.spline_xyz(t), R=self.spline_so3(t).as_matrix())

    def derivative(self, t: float, order: int = 1) -> Twist3:
        linear = self.spline_xyz.derivative(order)(t)
        angular = self.spline_so3(
            t, order
        )  # 1 is angular rate, 2 is angular acceleration
        return Twist3(linear, angular)


class SplineFit:
    """A general class to fit various SE3 splines to data."""

    def __init__(
        self,
        time_data: List[float],
        pose_data: List[SE3],
    ) -> None:
        self.time_data = time_data
        self.pose_data = pose_data
        self.spline: Optional[SplineSE3] = None

    def stochastic_downsample_interpolation(
        self,
        epsilon_xyz: float = 1e-3,
        epsilon_angle: float = 1e-1,
        normalize_time: bool = True,
        bc_type: str = "not-a-knot",
        check_type: str = "local",
    ) -> Tuple[InterpSplineSE3, List[int]]:
        """
        Uses a random dropout to downsample a trajectory with an interpolated spline. Keeps the start and
        end points of the trajectory. Takes a random order of the remaining indices, and then checks the error bound
        of just that point if check_type=="local", checks the error of the whole trajectory is check_type=="global".
        Local is **much** faster.

            Return:
                downsampled interpolating spline,
                list of removed indices from input data
        """

        interpolation_indices = list(range(len(self.pose_data)))

        # randomly attempt to remove poses from the trajectory
        # always keep the start and end
        removal_choices = interpolation_indices.copy()
        removal_choices.remove(0)
        removal_choices.remove(len(self.pose_data) - 1)
        np.random.shuffle(removal_choices)
        for candidate_removal_index in removal_choices:
            interpolation_indices.remove(candidate_removal_index)

            self.spline = InterpSplineSE3(
                [self.time_data[i] for i in interpolation_indices],
                [self.pose_data[i] for i in interpolation_indices],
                normalize_time=normalize_time,
                bc_type=bc_type,
            )

            sample_time = self.time_data[candidate_removal_index]
            if check_type == "local":
                angular_error = SO3(self.pose_data[candidate_removal_index]).angdist(
                    SO3(self.spline.spline_so3(sample_time).as_matrix())
                )
                euclidean_error = np.linalg.norm(
                    self.pose_data[candidate_removal_index].t
                    - self.spline.spline_xyz(sample_time)
                )
            elif check_type == "global":
                angular_error = self.max_angular_error()
                euclidean_error = self.max_euclidean_error()
            else:
                raise ValueError(
                    f"check_type must be 'local' of 'global', is {check_type}."
                )

            if (angular_error > epsilon_angle) or (euclidean_error > epsilon_xyz):
                interpolation_indices.append(candidate_removal_index)
                interpolation_indices.sort()

        self.spline = InterpSplineSE3(
            [self.time_data[i] for i in interpolation_indices],
            [self.pose_data[i] for i in interpolation_indices],
            normalize_time=normalize_time,
            bc_type=bc_type,
        )

        return self.spline, interpolation_indices

    def max_angular_error(self) -> float:
        return np.max(self.angular_errors())

    def angular_errors(self) -> List[float]:
        return [
            pose.angdist(self.spline(t))
            for pose, t in zip(self.pose_data, self.time_data)
        ]

    def max_euclidean_error(self) -> float:
        return np.max(self.euclidean_errors())

    def euclidean_errors(self) -> List[float]:
        return [
            np.linalg.norm(pose.t - self.spline(t).t)
            for pose, t in zip(self.pose_data, self.time_data)
        ]


class BSplineSE3(SplineSE3):
    """A class to parameterize a trajectory in SE3 with a 6-dimensional B-spline.

    The SE3 control poses are converted to se3 twists (the lie algebra) and a B-spline
    is created for each dimension of the twist, using the corresponding element of the twists
    as the control point for the spline.

    For detailed information about B-splines, please see this wikipedia article.
    https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
    """

    def __init__(
        self,
        control_poses: List[SE3],
        degree: int = 3,
        knots: Optional[List[float]] = None,
    ) -> None:
        """Construct BSplineSE3 object. The default arguments generate a cubic B-spline
        with uniformly spaced knots.

        - control_poses: list of SE3 objects that govern the shape of the spline.
        - degree: int that controls degree of the polynomial that governs any given point on the spline.
        - knots: list of floats that govern which control points are active during evaluating the spline
        at a given t input. If none, they are automatically, uniformly generated based on number of control poses and
        degree of spline on the range [0,1].
        """
        super().__init__()
        self.control_poses = control_poses

        # a matrix where each row is a control pose as a twist
        # (so each column is a vector of control points for that dim of the twist)
        self.control_pose_matrix = np.vstack(
            [np.array(element.twist()) for element in control_poses]
        )

        self.degree = degree

        if knots is None:
            knots = np.linspace(0, 1, len(control_poses) - degree + 1, endpoint=True)
            knots = np.append(
                [0.0] * degree, knots
            )  # ensures the curve starts on the first control pose
            knots = np.append(
                knots, [1] * degree
            )  # ensures the curve ends on the last control pose
        self.knots = knots

        self.splines = [
            BSpline(knots, self.control_pose_matrix[:, i], degree)
            for i in range(0, 6)  # twists are length 6
        ]

    def __call__(self, t: float) -> SE3:
        """Returns pose of spline at t.

        t: Normalized time value [0,1] to evaluate the spline at.
        """
        twist = np.hstack([spline(t) for spline in self.splines])
        return SE3.Exp(twist)
