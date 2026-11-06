"""Angle-axis construction preserves unit-quaternion and zero-angle contracts."""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from spatialmath import UnitQuaternion
from spatialmath.base import angvec2r


class TestAngVecNonunitAxis(unittest.TestCase):
    def test_unit_norm(self):
        assert_allclose(np.linalg.norm(UnitQuaternion.AngVec(0.5, [1, 2, 3]).vec), 1)

    def test_angle_round_trip(self):
        theta, axis = UnitQuaternion.AngVec(0.5, [1, 2, 3]).angvec()
        assert_allclose(theta, 0.5)
        assert_allclose(axis, np.array([1, 2, 3]) / np.sqrt(14))

    def test_rotation_matrix(self):
        assert_allclose(
            UnitQuaternion.AngVec(0.5, [1, 2, 3]).R,
            angvec2r(0.5, [1, 2, 3]),
            atol=1e-12,
        )

    def test_degrees(self):
        assert_allclose(
            UnitQuaternion.AngVec(90, [0, 0, 4], unit="deg").R,
            angvec2r(math.pi / 2, [0, 0, 1]),
            atol=1e-12,
        )

    def test_unit_axis_control(self):
        assert_allclose(
            UnitQuaternion.AngVec(0.5, [1, 0, 0]).vec,
            [math.cos(0.25), math.sin(0.25), 0, 0],
        )

    def test_zero_angle_zero_axis(self):
        assert_allclose(UnitQuaternion.AngVec(0, [0, 0, 0]).vec, [1, 0, 0, 0])

    def test_zero_degrees_zero_axis(self):
        assert_allclose(
            UnitQuaternion.AngVec(0, [0, 0, 0], unit="deg").vec, [1, 0, 0, 0]
        )

    def test_nonzero_angle_requires_nonzero_axis(self):
        with self.assertRaises(ValueError):
            UnitQuaternion.AngVec(0.5, [0, 0, 0])

    def test_near_zero_angle_zero_axis(self):
        for theta in [1e-20, -1e-20]:
            with self.subTest(theta=theta):
                assert_allclose(
                    UnitQuaternion.AngVec(theta, [0, 0, 0]).vec, [1, 0, 0, 0]
                )

    def test_near_zero_degrees_zero_axis(self):
        for theta in [1e-20, -1e-20]:
            with self.subTest(theta=theta):
                assert_allclose(
                    UnitQuaternion.AngVec(theta, [0, 0, 0], unit="deg").vec,
                    [1, 0, 0, 0],
                )

    def test_nonzero_small_angle_requires_nonzero_axis(self):
        for theta in [1e-10, -1e-10]:
            with self.subTest(theta=theta):
                with self.assertRaises(ValueError):
                    UnitQuaternion.AngVec(theta, [0, 0, 0])

    def test_nonzero_small_angle_preserves_rotation(self):
        for theta in [1e-10, -1e-10]:
            with self.subTest(theta=theta):
                assert_allclose(
                    UnitQuaternion.AngVec(theta, [2, 0, 0]).vec,
                    [math.cos(theta / 2), math.sin(theta / 2), 0, 0],
                    atol=0,
                )
