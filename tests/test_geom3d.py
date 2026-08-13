#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:37:24 2020

@author: corkep
"""

from spatialmath.geom3d import *
from spatialmath.pose3d import SE3

import unittest
import numpy.testing as nt
import spatialmath.base as base
import pytest
import sys
import matplotlib.pyplot as plt


class Line3Test(unittest.TestCase):
    # Primitives
    def test_constructor1(self):
        # construct from 6-vector

        with self.assertRaises(ValueError):
            L = Line3([1, 2, 3, 4, 5, 6], check=True)

        L = Line3([1, 2, 3, 4, 5, 6], check=False)
        self.assertIsInstance(L, Line3)
        nt.assert_array_almost_equal(L.v, np.r_[1, 2, 3])
        nt.assert_array_almost_equal(L.w, np.r_[4, 5, 6])

        # construct from object
        L2 = Line3(L, check=False)
        self.assertIsInstance(L, Line3)
        nt.assert_array_almost_equal(L2.v, np.r_[1, 2, 3])
        nt.assert_array_almost_equal(L2.w, np.r_[4, 5, 6])

        # construct from point and direction
        L = Line3.PointDir([1, 2, 3], [4, 5, 6])
        self.assertTrue(L.contains([1, 2, 3]))
        nt.assert_array_almost_equal(L.uw, base.unitvec([4, 5, 6]))

    def test_vec(self):
        # verify double
        L = Line3([1, 2, 3, 4, 5, 6], check=False)
        nt.assert_array_almost_equal(L.vec, np.r_[1, 2, 3, 4, 5, 6])

    def test_constructor2(self):
        # 2, point constructor
        P = np.r_[2, 3, 7]
        Q = np.r_[2, 1, 0]
        L = Line3.Join(P, Q)
        nt.assert_array_almost_equal(L.w, P - Q)
        nt.assert_array_almost_equal(L.v, np.cross(P - Q, Q))

        # TODO, all combos of list and ndarray
        # test all possible input shapes
        # L2, = Line3(P, Q)
        # self.assertEqual(double(L2), double(L))
        # L2, = Line3(P, Q')
        # self.assertEqual(double(L2), double(L))
        # L2, = Line3(P', Q')
        # self.assertEqual(double(L2), double(L))
        # L2, = Line3(P, Q)
        # self.assertEqual(double(L2), double(L))

        # # planes constructor
        # P = [10, 11, 12]'; w = [1, 2, 3]
        # L = Line3.PointDir(P, w)
        # self.assertEqual(double(L), [cross(w,P) w]'); %FAIL
        # L2, = Line3.PointDir(P', w)
        # self.assertEqual(double(L2), double(L))
        # L2, = Line3.PointDir(P, w')
        # self.assertEqual(double(L2), double(L))
        # L2, = Line3.PointDir(P', w')
        # self.assertEqual(double(L2), double(L))

    def test_pp(self):
        # validate pp and ppd
        L = Line3.Join([-1, 1, 2], [1, 1, 2])
        nt.assert_array_almost_equal(L.pp, np.r_[0, 1, 2])
        self.assertEqual(L.ppd, math.sqrt(5))

        # validate pp
        self.assertTrue(L.contains(L.pp))

    def test_contains(self):
        P = [2, 3, 7]
        Q = [2, 1, 0]
        L = Line3.Join(P, Q)

        # validate contains
        self.assertTrue(L.contains([2, 3, 7]))
        self.assertTrue(L.contains([2, 1, 0]))
        self.assertFalse(L.contains([2, 1, 4]))

    def test_closest(self):
        P = [2, 3, 7]
        Q = [2, 1, 0]
        L = Line3.Join(P, Q)

        p, d = L.closest_to_point(P)
        nt.assert_array_almost_equal(p, P)
        self.assertAlmostEqual(d, 0)

        # validate closest with given points and origin
        p, d = L.closest_to_point(Q)
        nt.assert_array_almost_equal(p, Q)
        self.assertAlmostEqual(d, 0)

        L = Line3.Join([-1, 1, 2], [1, 1, 2])
        p, d = L.closest_to_point([0, 1, 2])
        nt.assert_array_almost_equal(p, np.r_[0, 1, 2])
        self.assertAlmostEqual(d, 0)

        p, d = L.closest_to_point([5, 1, 2])
        nt.assert_array_almost_equal(p, np.r_[5, 1, 2])
        self.assertAlmostEqual(d, 0)

        p, d = L.closest_to_point([0, 0, 0])
        nt.assert_array_almost_equal(p, L.pp)
        self.assertEqual(d, L.ppd)

        p, d = L.closest_to_point([5, 1, 0])
        nt.assert_array_almost_equal(p, [5, 1, 2])
        self.assertAlmostEqual(d, 2)

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot(self):
        P = [2, 3, 7]
        Q = [2, 1, 0]
        L = Line3.Join(P, Q)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10, 10)
        ax.set_zlim3d(-10, 10)

        L.plot(color="red", linewidth=2)

    def test_eq(self):
        w = np.r_[1, 2, 3]
        P = np.r_[-2, 4, 3]

        L1 = Line3.Join(P, P + w)
        L2 = Line3.Join(P + 2 * w, P + 5 * w)
        L3 = Line3.Join(P + np.r_[1, 0, 0], P + w)

        self.assertTrue(L1 == L2)
        self.assertFalse(L1 == L3)

        self.assertFalse(L1 != L2)
        self.assertTrue(L1 != L3)

    def test_skew(self):
        P = [2, 3, 7]
        Q = [2, 1, 0]
        L = Line3.Join(P, Q)

        m = L.skew()

        self.assertEqual(m.shape, (4, 4))
        nt.assert_array_almost_equal(m + m.T, np.zeros((4, 4)))

    def test_rmul(self):
        P = [1, 2, 0]
        Q = [1, 2, 10]  # vertical line through (1,2)
        L = Line3.Join(P, Q)

        # check transformation by SE3

        L2 = SE3() * L
        p = L2.intersect_plane([0, 0, 1, 0])[0]  # intersects z=0
        nt.assert_array_almost_equal(p, [1, 2, 0])

        L2 = SE3(2, 0, 0) * L  # shift line in the x direction
        p = L2.intersect_plane([0, 0, 1, 0])[0]  # intersects z=0
        nt.assert_array_almost_equal(p, [3, 2, 0])

        L2 = SE3(0, 2, 0) * L  # shift line in the y direction
        p = L2.intersect_plane([0, 0, 1, 0])[0]  # intersects z=0
        nt.assert_array_almost_equal(p, [1, 4, 0])

        L2 = SE3.Rx(np.pi / 2) * L  # rotate line about x-axis, now horizontal
        nt.assert_array_almost_equal(L2.uw, [0, -1, 0])

    def test_parallel(self):
        L1 = Line3.PointDir([4, 5, 6], [1, 2, 3])
        L2 = Line3.PointDir([5, 5, 6], [1, 2, 3])
        L3 = Line3.PointDir([4, 5, 6], [3, 2, 1])

        # L1, || L2, but doesnt intersect
        # L1, intersects L3

        self.assertTrue(L1.isparallel(L1))
        self.assertTrue(L1 | L1)

        self.assertTrue(L1.isparallel(L2))
        self.assertTrue(L1 | L2)
        self.assertTrue(L2.isparallel(L1))
        self.assertTrue(L2 | L1)
        self.assertFalse(L1.isparallel(L3))
        self.assertFalse(L1 | L3)

    def test_intersect(self):
        L1 = Line3.PointDir([4, 5, 6], [1, 2, 3])
        L2 = Line3.PointDir([5, 5, 6], [1, 2, 3])
        L3 = Line3.PointDir([4, 5, 6], [0, 0, 1])
        L4 = Line3.PointDir([5, 5, 6], [1, 0, 0])

        # L1, || L2, but doesnt intersect
        # L3, intersects L4
        self.assertFalse(
            L1 ^ L2,
        )

        self.assertTrue(
            L3 ^ L4,
        )

    def test_commonperp(self):
        L1 = Line3.PointDir([4, 5, 6], [0, 0, 1])
        L2 = Line3.PointDir([6, 5, 6], [0, 1, 0])

        self.assertFalse(L1 | L2)
        self.assertFalse(L1 ^ L2)

        self.assertEqual(L1.distance(L2), 2)

        L = L1.commonperp(L2)  # common perp intersects both lines

        self.assertTrue(L ^ L1)
        self.assertTrue(L ^ L2)

    def test_line(self):
        # mindist
        # intersect
        # char
        # intersect_volume
        # mindist
        # mtimes
        # or
        # side
        pass

    def test_contains(self):
        P = [2, 3, 7]
        Q = [2, 1, 0]
        L = Line3.Join(P, Q)

        self.assertTrue(L.contains(L.point(0)))
        self.assertTrue(L.contains(L.point(1)))
        self.assertTrue(L.contains(L.point(-1)))

    def test_point(self):
        P = [2, 3, 7]
        Q = [2, 1, 0]
        L = Line3.Join(P, Q)

        nt.assert_array_almost_equal(L.point(0).flatten(), L.pp)

        for x in (-2, 0, 3):
            nt.assert_array_almost_equal(L.lam(L.point(x)), x)

    def test_char(self):
        P = [2, 3, 7]
        Q = [2, 1, 0]
        L = Line3.Join(P, Q)

        s = str(L)
        self.assertIsInstance(s, str)

    def test_plane(self):
        xyplane = [0, 0, 1, 0]
        xzplane = [0, 1, 0, 0]
        L = Line3.TwoPlanes(xyplane, xzplane)  # x axis
        nt.assert_array_almost_equal(L.vec, np.r_[0, 0, 0, -1, 0, 0])

        L = Line3.Join([-1, 2, 3], [1, 2, 3])
        # line at y=2,z=3
        x6 = [1, 0, 0, -6]  # x = 6

        # plane_intersect
        p, lam = L.intersect_plane(x6)
        nt.assert_array_almost_equal(p, np.r_[6, 2, 3])
        nt.assert_array_almost_equal(L.point(lam).flatten(), np.r_[6, 2, 3])

        x6s = Plane3.PointNormal(n=[1, 0, 0], p=[6, 0, 0])
        p, lam = L.intersect_plane(x6s)
        nt.assert_array_almost_equal(p, np.r_[6, 2, 3])

        nt.assert_array_almost_equal(L.point(lam).flatten(), np.r_[6, 2, 3])

    def test_methods(self):
        # intersection
        px = Line3.Join([0, 0, 0], [1, 0, 0])
        # x-axis
        py = Line3.Join([0, 0, 0], [0, 1, 0])
        # y-axis
        px1 = Line3.Join([0, 1, 0], [1, 1, 0])
        # offset x-axis

        self.assertEqual(px.ppd, 0)
        self.assertEqual(px1.ppd, 1)
        nt.assert_array_almost_equal(px1.pp, [0, 1, 0])

        px.intersects(px)
        px.intersects(py)
        px.intersects(px1)

    # def test_intersect(self):
    #     px = Line3([0, 0, 0], [1, 0, 0]);  # x-axis
    #     py = Line3([0, 0, 0], [0, 1, 0]);  # y-axis
    #
    #     plane.d = [1, 0, 0]; plane.p = 2; # plane x=2
    #
    #     px.intersect_plane(plane)
    #     py.intersect_plane(plane)


if __name__ == "__main__":
    unittest.main()
