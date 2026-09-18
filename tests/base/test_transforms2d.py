#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep

"""

import numpy as np
import numpy.testing as nt
import unittest
from math import pi
import math
from scipy.linalg import logm
import pytest
import sys

from spatialmath.base.transforms2d import *
from spatialmath.base.transformsNd import (
    isR,
    t2r,
    r2t,
    rt2tr,
    skew,
    vexa,
    skewa,
    homtrans,
)

import matplotlib.pyplot as plt


class Test2D(unittest.TestCase):
    def test_rot2(self):
        R = np.array([[1, 0], [0, 1]])
        nt.assert_array_almost_equal(rot2(0), R)
        nt.assert_array_almost_equal(rot2(0, unit="rad"), R)
        nt.assert_array_almost_equal(rot2(0, unit="deg"), R)
        nt.assert_array_almost_equal(rot2(0, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rot2(0)), 1)

        R = np.array([[0, -1], [1, 0]])
        nt.assert_array_almost_equal(rot2(pi / 2), R)
        nt.assert_array_almost_equal(rot2(pi / 2, unit="rad"), R)
        nt.assert_array_almost_equal(rot2(90, unit="deg"), R)
        nt.assert_array_almost_equal(rot2(90, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rot2(pi / 2)), 1)

        R = np.array([[-1, 0], [0, -1]])
        nt.assert_array_almost_equal(rot2(pi), R)
        nt.assert_array_almost_equal(rot2(pi, unit="rad"), R)
        nt.assert_array_almost_equal(rot2(180, unit="deg"), R)
        nt.assert_array_almost_equal(rot2(180, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rot2(pi)), 1)

    def test_trot2(self):
        nt.assert_array_almost_equal(
            trot2(pi / 2, t=[3, 4]), np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]])
        )
        nt.assert_array_almost_equal(
            trot2(pi / 2, t=(3, 4)), np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]])
        )
        nt.assert_array_almost_equal(
            trot2(pi / 2, t=np.array([3, 4])),
            np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]),
        )

    def test_Rt(self):
        nt.assert_array_almost_equal(rot2(0.3), t2r(trot2(0.3)))
        nt.assert_array_almost_equal(trot2(0.3), r2t(rot2(0.3)))

        R = rot2(0.2)
        t = [1, 2]
        T = rt2tr(R, t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl2(T), np.array(t))
        # TODO

    def test_trlog2(self):
        R = rot2(0.5)
        nt.assert_array_almost_equal(trlog2(R), skew(0.5))

        nt.assert_array_almost_equal(trlog2(R, twist=True), 0.5)

        T = transl2(1, 2) @ trot2(0.5)
        nt.assert_array_almost_equal(trlog2(T), logm(T))

        nt.assert_array_almost_equal(trlog2(T, twist=True), vexa(logm(T)))

    def test_trexp2(self):
        R = trexp2(skew(0.5))
        nt.assert_array_almost_equal(R, rot2(0.5))

        T = transl2(1, 2) @ trot2(0.5)
        nt.assert_array_almost_equal(trexp2(logm(T)), T)

    def test_trnorm2(self):
        R = rot2(0.4)
        R = np.round(R, 3)  # approx SO(2)
        R = trnorm2(R)
        self.assertTrue(isrot2(R, check=True))

        R = rot2(0.4)
        R = np.round(R, 3)  # approx SO(2)
        T = rt2tr(R, [3, 4])

        T = trnorm2(T)
        self.assertTrue(ishom2(T, check=True))
        nt.assert_almost_equal(T[:2, 2], [3, 4])

    def test_transl2(self):
        nt.assert_array_almost_equal(
            transl2(1, 2), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])
        )
        nt.assert_array_almost_equal(
            transl2([1, 2]), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])
        )

    def test_pos2tr2(self):
        nt.assert_array_almost_equal(
            pos2tr2(1, 2), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])
        )
        nt.assert_array_almost_equal(
            transl2([1, 2]), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])
        )
        nt.assert_array_almost_equal(tr2pos2(pos2tr2(1, 2)), np.array([1, 2]))

    def test_tr2jac2(self):
        T = trot2(0.3, t=[4, 5])
        jac2 = tr2jac2(T)
        nt.assert_array_almost_equal(jac2[:2, :2], smb.t2r(T))
        nt.assert_array_almost_equal(jac2[:3, 2], np.array([0, 0, 1]))
        nt.assert_array_almost_equal(jac2[2, :3], np.array([0, 0, 1]))

    def test_xyt2tr(self):
        T = xyt2tr([1, 2, 0])
        nt.assert_array_almost_equal(T, transl2(1, 2))

        T = xyt2tr([1, 2, 0.2])
        nt.assert_array_almost_equal(T, rt2tr(rot2(0.2), [1, 2]))

    def test_trinv2(self):
        T = rt2tr(rot2(0.2), [1, 2])
        nt.assert_array_almost_equal(trinv2(T) @ T, np.eye(3))

    def test_tr2adjoint2(self):
        T = xyt2tr([1, 2, 0.2])
        X = [1, 2, 3]
        nt.assert_almost_equal(tr2adjoint2(T) @ X, vexa(T @ skewa(X) @ trinv2(T)))

    def test_tradjoint2_deprecated(self):
        T = xyt2tr([1, 2, 0.2])
        X = [1, 2, 3]
        with self.assertWarns(DeprecationWarning):
            result = tradjoint2(T)
        nt.assert_almost_equal(result @ X, vexa(T @ skewa(X) @ trinv2(T)))

    def test_points2tr2(self):
        p1 = np.random.uniform(size=(2, 5))
        T = xyt2tr([1, 2, 0.2])
        p2 = homtrans(T, p1)
        T2 = points2tr2(p1, p2)
        nt.assert_almost_equal(T, T2)

    def test_icp2d(self):
        p1 = np.random.uniform(size=(2, 30))
        T = xyt2tr([1, 2, 0.2])

        p2 = homtrans(T, p1)
        k = np.random.permutation(p2.shape[1])
        p2 = p2[:, k]

        T2 = ICP2d(p2, p1, T=xyt2tr([1, 2, 0.2]))
        nt.assert_almost_equal(T, T2)

    def test_icp2d_returns_valid_se2(self):
        # noisy correspondences plus a forced iteration count accumulate
        # roundoff error in the T = T @ new_T composition -- the result
        # must still pass ishom2(check=True), ie. be exactly orthonormal.
        # seed fixed: this configuration reliably triggers the drift.
        np.random.seed(0)
        p1 = np.random.uniform(size=(2, 30))
        T = xyt2tr([1, 2, 0.2])

        p2 = homtrans(T, p1) + np.random.normal(scale=0.01, size=(2, 30))
        k = np.random.permutation(p2.shape[1])
        p2 = p2[:, k]

        T2 = ICP2d(p2, p1, max_iter=100, min_delta_err=0)
        self.assertTrue(ishom2(T2, check=True))

    def test_print2(self):
        T = transl2(1, 2) @ trot2(0.3)

        s = trprint2(T, file=None)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 15)

    def test_checks(self):
        # 2D case, with rotation matrix
        R = np.eye(2)
        nt.assert_equal(isR(R), True)
        nt.assert_equal(isrot2(R), True)

        nt.assert_equal(ishom2(R), False)
        nt.assert_equal(isrot2(R, True), True)

        nt.assert_equal(ishom2(R, True), False)

        # 2D case, invalid rotation matrix
        R = np.array([[1, 1], [0, 1]])
        nt.assert_equal(isR(R), False)
        nt.assert_equal(isrot2(R), True)
        nt.assert_equal(ishom2(R), False)
        nt.assert_equal(isrot2(R, True), False)
        nt.assert_equal(ishom2(R, True), False)

        # 2D case, with homogeneous transformation matrix
        T = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot2(T), False)

        nt.assert_equal(ishom2(T), True)
        nt.assert_equal(isrot2(T, True), False)

        nt.assert_equal(ishom2(T, True), True)

        # 2D case, invalid rotation matrix
        T = np.array([[1, 1, 3], [0, 1, 4], [0, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot2(T), False)

        nt.assert_equal(ishom2(T), True)
        nt.assert_equal(isrot2(T, True), False)

        nt.assert_equal(ishom2(T, True), False)

        # 2D case, invalid bottom row
        T = np.array([[1, 1, 3], [0, 1, 4], [9, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot2(T), False)

        nt.assert_equal(ishom2(T), True)
        nt.assert_equal(isrot2(T, True), False)

        nt.assert_equal(ishom2(T, True), False)

    def test_trinterp2(self):
        R0 = rot2(-0.3)
        R1 = rot2(0.3)

        nt.assert_array_almost_equal(trinterp2(start=None, end=R1, s=0), np.eye(2))
        nt.assert_array_almost_equal(trinterp2(start=None, end=R1, s=1), R1)
        nt.assert_array_almost_equal(
            trinterp2(start=None, end=R1, s=0.5), rot2(0.3 / 2)
        )

        nt.assert_array_almost_equal(trinterp2(start=None, end=R1, s=0), np.eye(2))
        nt.assert_array_almost_equal(trinterp2(start=None, end=R1, s=1), R1)
        nt.assert_array_almost_equal(
            trinterp2(start=None, end=R1, s=0.5), rot2(0.3 / 2)
        )

        T0 = trot2(-0.3)
        T1 = trot2(0.3)

        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=0.5), np.eye(3))

        nt.assert_array_almost_equal(trinterp2(start=None, end=T1, s=0), np.eye(3))
        nt.assert_array_almost_equal(trinterp2(start=None, end=T1, s=1), T1)
        nt.assert_array_almost_equal(
            trinterp2(start=None, end=T1, s=0.5), trot2(0.3 / 2)
        )

        T0 = transl2(-1, -2)
        T1 = transl2(1, 2)

        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=0.5), np.eye(3))

        T0 = transl2(-1, -2) @ trot2(-0.3)
        T1 = transl2(1, 2) @ trot2(0.3)

        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=0.5), np.eye(3))

        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal(trinterp2(start=T0, end=T1, s=0.5), np.eye(3))

        nt.assert_array_almost_equal(trinterp2(start=None, end=T1, s=0), np.eye(3))
        nt.assert_array_almost_equal(trinterp2(start=None, end=T1, s=1), T1)
        nt.assert_array_almost_equal(
            trinterp2(start=None, end=T1, s=0.5), xyt2tr([0.5, 1, 0.15])
        )

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot(self):
        plt.figure()
        trplot2(transl2(1, 2), block=False, frame="A", rviz=True, width=1)
        trplot2(transl2(3, 1), block=False, color="red", arrow=True, width=3, frame="B")
        trplot2(
            transl2(4, 3) @ trot2(math.pi / 3), block=False, color="green", frame="c"
        )
        plt.close("all")


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
