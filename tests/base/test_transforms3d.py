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
from scipy.linalg import logm

from spatialmath.base.transforms3d import *
from spatialmath.base.transformsNd import isR, t2r, r2t, rt2tr, skew


class Test3D(unittest.TestCase):
    def test_checks(self):
        # 2D case, with rotation matrix
        R = np.eye(2)
        nt.assert_equal(isR(R), True)
        nt.assert_equal(isrot(R), False)
        nt.assert_equal(ishom(R), False)
        nt.assert_equal(isrot(R, True), False)
        nt.assert_equal(ishom(R, True), False)

        # 2D case, invalid rotation matrix
        R = np.array([[1, 1], [0, 1]])
        nt.assert_equal(isR(R), False)
        nt.assert_equal(isrot(R), False)
        nt.assert_equal(ishom(R), False)
        nt.assert_equal(isrot(R, True), False)
        nt.assert_equal(ishom(R, True), False)

        # 2D case, with homogeneous transformation matrix
        T = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot(T), True)
        nt.assert_equal(ishom(T), False)
        nt.assert_equal(isrot(T, True), False)
        nt.assert_equal(ishom(T, True), False)

        # 2D case, invalid rotation matrix
        T = np.array([[1, 1, 3], [0, 1, 4], [0, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot(T), True)
        nt.assert_equal(ishom(T), False)
        nt.assert_equal(isrot(T, True), False)
        nt.assert_equal(ishom(T, True), False)

        # 2D case, invalid bottom row
        T = np.array([[1, 1, 3], [0, 1, 4], [9, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot(T), True)
        nt.assert_equal(ishom(T), False)
        nt.assert_equal(isrot(T, True), False)
        nt.assert_equal(ishom(T, True), False)

        # reflection case
        T = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot(T), True)
        nt.assert_equal(ishom(T), False)
        nt.assert_equal(isrot(T, True), False)
        nt.assert_equal(ishom(T, True), False)

    def test_trinv(self):
        T = np.eye(4)
        nt.assert_array_almost_equal(trinv(T), T)

        T = trotx(0.3)
        nt.assert_array_almost_equal(trinv(T) @ T, np.eye(4))

        T = transl(1, 2, 3)
        nt.assert_array_almost_equal(trinv(T) @ T, np.eye(4))

    def test_rotx(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotx(0), R)
        nt.assert_array_almost_equal(rotx(0, unit="rad"), R)
        nt.assert_array_almost_equal(rotx(0, unit="deg"), R)
        nt.assert_array_almost_equal(rotx(0, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rotx(0)), 1)

        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        nt.assert_array_almost_equal(rotx(pi / 2), R)
        nt.assert_array_almost_equal(rotx(pi / 2, unit="rad"), R)
        nt.assert_array_almost_equal(rotx(90, unit="deg"), R)
        nt.assert_array_almost_equal(rotx(90, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rotx(pi / 2)), 1)

        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(rotx(pi), R)
        nt.assert_array_almost_equal(rotx(pi, unit="rad"), R)
        nt.assert_array_almost_equal(rotx(180, unit="deg"), R)
        nt.assert_array_almost_equal(rotx(180, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rotx(pi)), 1)

    def test_roty(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(roty(0), R)
        nt.assert_array_almost_equal(roty(0, unit="rad"), R)
        nt.assert_array_almost_equal(roty(0, unit="deg"), R)
        nt.assert_array_almost_equal(roty(0, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(roty(0)), 1)

        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        nt.assert_array_almost_equal(roty(pi / 2), R)
        nt.assert_array_almost_equal(roty(pi / 2, unit="rad"), R)
        nt.assert_array_almost_equal(roty(90, unit="deg"), R)
        nt.assert_array_almost_equal(roty(90, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(roty(pi / 2)), 1)

        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(roty(pi), R)
        nt.assert_array_almost_equal(roty(pi, unit="rad"), R)
        nt.assert_array_almost_equal(roty(180, unit="deg"), R)
        nt.assert_array_almost_equal(roty(180, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(roty(pi)), 1)

    def test_rotz(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(0), R)
        nt.assert_array_almost_equal(rotz(0, unit="rad"), R)
        nt.assert_array_almost_equal(rotz(0, unit="deg"), R)
        nt.assert_array_almost_equal(rotz(0, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rotz(0)), 1)

        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(pi / 2), R)
        nt.assert_array_almost_equal(rotz(pi / 2, unit="rad"), R)
        nt.assert_array_almost_equal(rotz(90, unit="deg"), R)
        nt.assert_array_almost_equal(rotz(90, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rotz(pi / 2)), 1)

        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(pi), R)
        nt.assert_array_almost_equal(rotz(pi, unit="rad"), R)
        nt.assert_array_almost_equal(rotz(180, unit="deg"), R)
        nt.assert_array_almost_equal(rotz(180, "deg"), R)
        nt.assert_almost_equal(np.linalg.det(rotz(pi)), 1)

    def test_trotX(self):
        T = np.array([[1, 0, 0, 3], [0, 0, -1, 4], [0, 1, 0, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(trotx(pi / 2, t=[3, 4, 5]), T)
        nt.assert_array_almost_equal(trotx(pi / 2, t=(3, 4, 5)), T)
        nt.assert_array_almost_equal(trotx(pi / 2, t=np.array([3, 4, 5])), T)

        T = np.array([[0, 0, 1, 3], [0, 1, 0, 4], [-1, 0, 0, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(troty(pi / 2, t=[3, 4, 5]), T)
        nt.assert_array_almost_equal(troty(pi / 2, t=(3, 4, 5)), T)
        nt.assert_array_almost_equal(troty(pi / 2, t=np.array([3, 4, 5])), T)

        T = np.array([[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(trotz(pi / 2, t=[3, 4, 5]), T)
        nt.assert_array_almost_equal(trotz(pi / 2, t=(3, 4, 5)), T)
        nt.assert_array_almost_equal(trotz(pi / 2, t=np.array([3, 4, 5])), T)

    def test_rpy2r(self):
        r2d = 180 / pi

        # default zyx order
        R = rotz(0.3) @ roty(0.2) @ rotx(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3]), R)
        nt.assert_array_almost_equal(
            rpy2r(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit="deg"), R
        )
        nt.assert_array_almost_equal(
            rpy2r([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit="deg"), R
        )

        # xyz order
        R = rotx(0.3) @ roty(0.2) @ rotz(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3, order="xyz"), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3], order="xyz"), R)
        nt.assert_array_almost_equal(
            rpy2r(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit="deg", order="xyz"), R
        )
        nt.assert_array_almost_equal(
            rpy2r([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit="deg", order="xyz"), R
        )

        # yxz order
        R = roty(0.3) @ rotx(0.2) @ rotz(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3, order="yxz"), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3], order="yxz"), R)
        nt.assert_array_almost_equal(
            rpy2r(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit="deg", order="yxz"), R
        )
        nt.assert_array_almost_equal(
            rpy2r([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit="deg", order="yxz"), R
        )

    def test_rpy2tr(self):
        r2d = 180 / pi

        # default zyx order
        T = trotz(0.3) @ troty(0.2) @ trotx(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3]), T)
        nt.assert_array_almost_equal(
            rpy2tr(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit="deg"), T
        )
        nt.assert_array_almost_equal(
            rpy2tr([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit="deg"), T
        )

        # xyz order
        T = trotx(0.3) @ troty(0.2) @ trotz(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3, order="xyz"), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3], order="xyz"), T)
        nt.assert_array_almost_equal(
            rpy2tr(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit="deg", order="xyz"), T
        )
        nt.assert_array_almost_equal(
            rpy2tr([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit="deg", order="xyz"), T
        )

        # yxz order
        T = troty(0.3) @ trotx(0.2) @ trotz(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3, order="yxz"), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3], order="yxz"), T)
        nt.assert_array_almost_equal(
            rpy2tr(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit="deg", order="yxz"), T
        )
        nt.assert_array_almost_equal(
            rpy2tr([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit="deg", order="yxz"), T
        )

    def test_eul2r(self):
        r2d = 180 / pi

        # default zyx order
        R = rotz(0.1) @ roty(0.2) @ rotz(0.3)
        nt.assert_array_almost_equal(eul2r(0.1, 0.2, 0.3), R)
        nt.assert_array_almost_equal(eul2r([0.1, 0.2, 0.3]), R)
        nt.assert_array_almost_equal(
            eul2r(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit="deg"), R
        )
        nt.assert_array_almost_equal(
            eul2r([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit="deg"), R
        )

    def test_eul2tr(self):
        r2d = 180 / pi

        # default zyx order
        T = trotz(0.1) @ troty(0.2) @ trotz(0.3)
        nt.assert_array_almost_equal(eul2tr(0.1, 0.2, 0.3), T)
        nt.assert_array_almost_equal(eul2tr([0.1, 0.2, 0.3]), T)
        nt.assert_array_almost_equal(
            eul2tr(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit="deg"), T
        )
        nt.assert_array_almost_equal(
            eul2tr([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit="deg"), T
        )

    def test_angvec2r(self):
        r2d = 180 / pi

        nt.assert_array_almost_equal(angvec2r(0, [1, 0, 0]), rotx(0))
        nt.assert_array_almost_equal(angvec2r(pi / 4, [1, 0, 0]), rotx(pi / 4))
        nt.assert_array_almost_equal(angvec2r(-pi / 4, [1, 0, 0]), rotx(-pi / 4))

        nt.assert_array_almost_equal(angvec2r(0, [0, 1, 0]), roty(0))
        nt.assert_array_almost_equal(angvec2r(pi / 4, [0, 1, 0]), roty(pi / 4))
        nt.assert_array_almost_equal(angvec2r(-pi / 4, [0, 1, 0]), roty(-pi / 4))

        nt.assert_array_almost_equal(angvec2r(0, [0, 0, 1]), rotz(0))
        nt.assert_array_almost_equal(angvec2r(pi / 4, [0, 0, 1]), rotz(pi / 4))
        nt.assert_array_almost_equal(angvec2r(-pi / 4, [0, 0, 1]), rotz(-pi / 4))

    def test_angvec2tr(self):
        r2d = 180 / pi

        nt.assert_array_almost_equal(angvec2tr(0, [1, 0, 0]), trotx(0))
        nt.assert_array_almost_equal(angvec2tr(pi / 4, [1, 0, 0]), trotx(pi / 4))
        nt.assert_array_almost_equal(angvec2tr(-pi / 4, [1, 0, 0]), trotx(-pi / 4))

        nt.assert_array_almost_equal(angvec2tr(0, [0, 1, 0]), troty(0))
        nt.assert_array_almost_equal(angvec2tr(pi / 4, [0, 1, 0]), troty(pi / 4))
        nt.assert_array_almost_equal(angvec2tr(-pi / 4, [0, 1, 0]), troty(-pi / 4))

        nt.assert_array_almost_equal(angvec2tr(0, [0, 0, 1]), trotz(0))
        nt.assert_array_almost_equal(angvec2tr(pi / 4, [0, 0, 1]), trotz(pi / 4))
        nt.assert_array_almost_equal(angvec2tr(-pi / 4, [0, 0, 1]), trotz(-pi / 4))

        r2d = 180 / pi

        nt.assert_array_almost_equal(angvec2r(0, [1, 0, 0]), rotx(0))
        nt.assert_array_almost_equal(angvec2r(pi / 4, [1, 0, 0]), rotx(pi / 4))
        nt.assert_array_almost_equal(angvec2r(-pi / 4, [1, 0, 0]), rotx(-pi / 4))

    def test_trlog(self):
        R = np.eye(3)
        nt.assert_array_almost_equal(trlog(R), skew([0, 0, 0]))
        nt.assert_array_almost_equal(trlog(R, twist=True), [0, 0, 0])

        R = rotx(0.5)
        nt.assert_array_almost_equal(trlog(R), skew([0.5, 0, 0]))
        nt.assert_array_almost_equal(trlog(R, twist=True), [0.5, 0, 0])

        R = roty(0.5)
        nt.assert_array_almost_equal(trlog(R), skew([0, 0.5, 0]))
        nt.assert_array_almost_equal(trlog(R, twist=True), [0, 0.5, 0])

        R = rotz(0.5)
        nt.assert_array_almost_equal(trlog(R), skew([0, 0, 0.5]))
        nt.assert_array_almost_equal(trlog(R, twist=True), [0, 0, 0.5])

        R = rpy2r(0.1, 0.2, 0.3)
        nt.assert_array_almost_equal(logm(R), trlog(R))

        T = transl(1, 2, 3) @ rpy2tr(0.1, 0.2, 0.3)
        nt.assert_array_almost_equal(logm(T), trlog(T))

    def test_trlog_near_identity(self):
        # Regression test for #63: trlog's general-case branch divides by
        # sin(theta), computed from trace(R) via acos. A near-identity R
        # that isn't *exactly* np.eye(3) (as produced by real computation,
        # not hand-constructed) must not raise or blow up.

        # theta small enough that cos(theta) underflows to exactly 1.0 in
        # float64, so trace(R) rounds to exactly 3 and acos(1.0) == 0.0
        # exactly: this is the case that used to divide by sin(theta) == 0
        # with no guard. R itself is not bitwise np.eye(3) (it still has
        # sin(theta) noise off the diagonal), so this also exercises the
        # "is this actually the identity" branch on a matrix that isn't one.
        R = rotx(1e-9)
        assert not np.array_equal(R, np.eye(3))
        nt.assert_array_almost_equal(trlog(R, twist=True), [0, 0, 0])
        nt.assert_array_almost_equal(trlog(R), skew([0, 0, 0]))

        # theta small enough to be well within the near-identity regime, but
        # not small enough to clamp: the general-case formula must stay
        # accurate here rather than falling back to a coarse zero. This is
        # the case a fuzzy identity-tolerance (checking R against I before
        # computing theta, as originally proposed in #63) risks getting
        # wrong in the other direction, by discarding a real small rotation.
        theta = 1e-7
        for R in (rotx(theta), roty(theta), rotz(theta)):
            v = trlog(R, twist=True)
            nt.assert_almost_equal(np.linalg.norm(v), theta, decimal=12)

    def test_trexp(self):
        R = trexp(skew([0.5, 0, 0]))
        nt.assert_array_almost_equal(R, rotx(0.5))
        R = trexp(skew([0, 0.5, 0]))
        nt.assert_array_almost_equal(R, roty(0.5))
        R = trexp(skew([0, 0, 0.5]))
        nt.assert_array_almost_equal(R, rotz(0.5))

        R = rpy2r(0.1, 0.2, 0.3)
        nt.assert_array_almost_equal(trexp(logm(R)), R)

        T = transl(1, 2, 3) @ rpy2tr(0.1, 0.2, 0.3)
        nt.assert_array_almost_equal(trexp(logm(T)), T)

    def test_exp2r(self):
        r2d = 180 / pi

        nt.assert_array_almost_equal(exp2r([0, 0, 0]), rotx(0))
        nt.assert_array_almost_equal(exp2r([pi / 4, 0, 0]), rotx(pi / 4))
        nt.assert_array_almost_equal(exp2r([-pi / 4, 0, 0]), rotx(-pi / 4))

        nt.assert_array_almost_equal(exp2r([0, 0, 0]), roty(0))
        nt.assert_array_almost_equal(exp2r([0, pi / 4, 0]), roty(pi / 4))
        nt.assert_array_almost_equal(exp2r([0, -pi / 4, 0]), roty(-pi / 4))

        nt.assert_array_almost_equal(exp2r([0, 0, 0]), rotz(0))
        nt.assert_array_almost_equal(exp2r([0, 0, pi / 4]), rotz(pi / 4))
        nt.assert_array_almost_equal(exp2r([0, 0, -pi / 4]), rotz(-pi / 4))

    def test_exp2tr(self):
        r2d = 180 / pi

        nt.assert_array_almost_equal(exp2tr([0, 0, 0]), trotx(0))
        nt.assert_array_almost_equal(exp2tr([pi / 4, 0, 0]), trotx(pi / 4))
        nt.assert_array_almost_equal(exp2tr([-pi / 4, 0, 0]), trotx(-pi / 4))

        nt.assert_array_almost_equal(exp2tr([0, 0, 0]), troty(0))
        nt.assert_array_almost_equal(exp2tr([0, pi / 4, 0]), troty(pi / 4))
        nt.assert_array_almost_equal(exp2tr([0, -pi / 4, 0]), troty(-pi / 4))

        nt.assert_array_almost_equal(exp2tr([0, 0, 0]), trotz(0))
        nt.assert_array_almost_equal(exp2tr([0, 0, pi / 4]), trotz(pi / 4))
        nt.assert_array_almost_equal(exp2tr([0, 0, -pi / 4]), trotz(-pi / 4))

    def test_tr2rpy(self):
        rpy = np.r_[0.1, 0.2, 0.3]
        R = rpy2r(rpy)
        nt.assert_array_almost_equal(tr2rpy(R), rpy)
        nt.assert_array_almost_equal(tr2rpy(R, unit="deg"), rpy * 180 / pi)

        T = rpy2tr(rpy)
        nt.assert_array_almost_equal(
            tr2rpy(T),
            rpy,
        )
        nt.assert_array_almost_equal(tr2rpy(T, unit="deg"), rpy * 180 / pi)

        # xyz order
        R = rpy2r(rpy, order="xyz")
        nt.assert_array_almost_equal(tr2rpy(R, order="xyz"), rpy)
        nt.assert_array_almost_equal(tr2rpy(R, unit="deg", order="xyz"), rpy * 180 / pi)

        T = rpy2tr(rpy, order="xyz")
        nt.assert_array_almost_equal(tr2rpy(T, order="xyz"), rpy)
        nt.assert_array_almost_equal(tr2rpy(T, unit="deg", order="xyz"), rpy * 180 / pi)

        # corner cases
        seq = "zyx"
        ang = [pi, 0, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi / 2, 0]  # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi / 2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)

        seq = "xyz"
        ang = [pi, 0, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi / 2, 0]  # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi / 2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)

        seq = "yxz"
        ang = [pi, 0, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi / 2, 0]  # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi / 2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)

    def test_trnorm(self):
        R = rpy2r(0.2, 0.3, 0.4)
        R = np.round(R, 3)  # approx SO(3)
        R = trnorm(R)
        self.assertTrue(isrot(R, check=True))

        R = rpy2r(0.2, 0.3, 0.4)
        R = np.round(R, 3)  # approx SO(3)
        T = rt2tr(R, [3, 4, 5])

        T = trnorm(T)
        self.assertTrue(ishom(T, check=True))
        nt.assert_almost_equal(T[:3, 3], [3, 4, 5])

    def test_tr2eul(self):
        eul = np.r_[0.1, 0.2, 0.3]
        R = eul2r(eul)
        nt.assert_array_almost_equal(tr2eul(R), eul)
        nt.assert_array_almost_equal(tr2eul(R, unit="deg"), eul * 180 / pi)

        T = eul2tr(eul)
        nt.assert_array_almost_equal(tr2eul(T), eul)
        nt.assert_array_almost_equal(tr2eul(T, unit="deg"), eul * 180 / pi)

        # test singularity case
        eul = [0.1, 0, 0.3]
        R = eul2r(eul)
        nt.assert_array_almost_equal(eul2r(tr2eul(R)), R)
        nt.assert_array_almost_equal(eul2r(tr2eul(R, unit="deg"), unit="deg"), R)

        # test flip
        eul = [-0.1, 0.2, 0.3]
        R = eul2r(eul)
        eul2 = tr2eul(R, flip=True)
        nt.assert_equal(eul2[0] > 0, True)
        nt.assert_array_almost_equal(eul2r(eul2), R)

    def test_tr2angvec(self):
        # null rotation
        # - vector isn't defined here, but RTB sets it (0 0 0)
        [theta, v] = tr2angvec(np.eye(3, 3))
        nt.assert_array_almost_equal(theta, 0.0)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 0])

        # canonic rotations
        [theta, v] = tr2angvec(rotx(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[1, 0, 0])

        [theta, v] = tr2angvec(roty(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])

        [theta, v] = tr2angvec(rotz(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 1])

        # null rotation
        [theta, v] = tr2angvec(np.eye(4))
        nt.assert_array_almost_equal(theta, 0.0)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 0])

        # canonic rotations
        [theta, v] = tr2angvec(trotx(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[1, 0, 0])

        [theta, v] = tr2angvec(troty(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])

        [theta, v] = tr2angvec(trotz(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 1])

        [theta, v] = tr2angvec(roty(pi / 2), unit="deg")
        nt.assert_array_almost_equal(theta, 90)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])

        true_ang = 1.51
        true_vec = np.array([0.0, 1.0, 0.0])
        eps = 1e-08

        # show that tr2angvec works on true rotation matrix
        ang, vec = tr2angvec(roty(true_ang), check=True)
        nt.assert_almost_equal(ang, true_ang)
        nt.assert_almost_equal(vec, true_vec)

        # check a rotation matrix that should fail
        badR = roty(true_ang) + eps
        with self.assertRaises(ValueError):
            tr2angvec(badR, check=True)

        # run without check
        ang, vec = tr2angvec(badR, check=False)
        nt.assert_almost_equal(ang, true_ang)
        nt.assert_almost_equal(vec, true_vec)

    def test_print(self):
        R = rotx(0.3) @ roty(0.4)
        s = trprint(R, file=None)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 30)

        T = transl(1, 2, 3) @ trotx(0.3) @ troty(0.4)
        s = trprint(T, file=None)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 42)
        self.assertTrue("rpy" in s)
        self.assertTrue("zyx" in s)

        s = trprint(T, file=None, orient="rpy/xyz")
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 39)
        self.assertTrue("rpy" in s)
        self.assertTrue("xyz" in s)

        s = trprint(T, file=None, orient="eul")
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 37)
        self.assertTrue("eul" in s)
        self.assertFalse("zyx" in s)

    def test_trinterp(self):
        R0 = rotx(-0.3)
        R1 = rotx(0.3)

        nt.assert_array_almost_equal(trinterp(start=R0, end=R1, s=0), R0)
        nt.assert_array_almost_equal(trinterp(start=R0, end=R1, s=1), R1)
        nt.assert_array_almost_equal(trinterp(start=R0, end=R1, s=0.5), np.eye(3))

        nt.assert_array_almost_equal(trinterp(start=None, end=R1, s=0), np.eye(3))
        nt.assert_array_almost_equal(trinterp(start=None, end=R1, s=1), R1)
        nt.assert_array_almost_equal(trinterp(start=None, end=R1, s=0.5), rotx(0.3 / 2))

        T0 = trotx(-0.3)
        T1 = trotx(0.3)

        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=0.5), np.eye(4))

        nt.assert_array_almost_equal(trinterp(start=None, end=T1, s=0), np.eye(4))
        nt.assert_array_almost_equal(trinterp(start=None, end=T1, s=1), T1)
        nt.assert_array_almost_equal(
            trinterp(start=None, end=T1, s=0.5), trotx(0.3 / 2)
        )

        T0 = transl(-1, -2, -3)
        T1 = transl(1, 2, 3)

        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=0.5), np.eye(4))

        T0 = transl(-1, -2, -3) @ trotx(-0.3)
        T1 = transl(1, 2, 3) @ trotx(0.3)

        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=0.5), np.eye(4))

        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal(trinterp(start=T0, end=T1, s=0.5), np.eye(4))

    def test_tr2delta(self):
        # unit testing tr2delta with a tr matrix
        nt.assert_array_almost_equal(
            tr2delta(transl(0.1, 0.2, 0.3)), np.r_[0.1, 0.2, 0.3, 0, 0, 0]
        )
        nt.assert_array_almost_equal(
            tr2delta(transl(0.1, 0.2, 0.3), transl(0.2, 0.4, 0.6)),
            np.r_[0.1, 0.2, 0.3, 0, 0, 0],
        )
        nt.assert_array_almost_equal(
            tr2delta(trotx(0.001)), np.r_[0, 0, 0, 0.001, 0, 0]
        )
        nt.assert_array_almost_equal(
            tr2delta(troty(0.001)), np.r_[0, 0, 0, 0, 0.001, 0]
        )
        nt.assert_array_almost_equal(
            tr2delta(trotz(0.001)), np.r_[0, 0, 0, 0, 0, 0.001]
        )
        nt.assert_array_almost_equal(
            tr2delta(trotx(0.001), trotx(0.002)), np.r_[0, 0, 0, 0.001, 0, 0]
        )

        # %Testing with a scalar number input
        # verifyError(tc, @()tr2delta(1),'SMTB:tr2delta:badarg');
        # verifyError(tc, @()tr2delta( ones(3,3) ),'SMTB:tr2delta:badarg');

    def test_delta2tr(self):
        # test with standard numbers
        nt.assert_array_almost_equal(
            delta2tr([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            np.array(
                [
                    [1.0, -0.6, 0.5, 0.1],
                    [0.6, 1.0, -0.4, 0.2],
                    [-0.5, 0.4, 1.0, 0.3],
                    [0, 0, 0, 1.0],
                ]
            ),
        )

        # test, with, zeros
        nt.assert_array_almost_equal(delta2tr([0, 0, 0, 0, 0, 0]), np.eye(4))

        # test with scalar input
        # verifyError(testCase, @()delta2tr(1),'MATLAB:badsubscript');

    def test_tr2jac(self):
        # NOTE, create these matrices using pyprint() in MATLAB
        # TODO change to forming it from block R matrices directly
        nt.assert_array_almost_equal(
            tr2jac(trotx(pi / 2)).T,
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, -1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, -1, 0],
                ]
            ),
        )

        nt.assert_array_almost_equal(
            tr2jac(transl(1, 2, 3)).T,
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
        )

        # test with scalar value
        # verifyError(tc, @()tr2jac(1),'SMTB:t2r:badarg');

    def test_r2x(self):
        R = rpy2r(0.2, 0.3, 0.4)

        nt.assert_array_almost_equal(r2x(R, representation="eul"), tr2eul(R))
        nt.assert_array_almost_equal(
            r2x(R, representation="rpy/xyz"), tr2rpy(R, order="xyz")
        )
        nt.assert_array_almost_equal(
            r2x(R, representation="rpy/zyx"), tr2rpy(R, order="zyx")
        )
        nt.assert_array_almost_equal(
            r2x(R, representation="rpy/yxz"), tr2rpy(R, order="yxz")
        )

        nt.assert_array_almost_equal(
            r2x(R, representation="arm"), tr2rpy(R, order="xyz")
        )
        nt.assert_array_almost_equal(
            r2x(R, representation="vehicle"), tr2rpy(R, order="zyx")
        )
        nt.assert_array_almost_equal(
            r2x(R, representation="camera"), tr2rpy(R, order="yxz")
        )

        nt.assert_array_almost_equal(r2x(R, representation="exp"), trlog(R, twist=True))

    def test_x2r(self):
        x = [0.2, 0.3, 0.4]

        nt.assert_array_almost_equal(x2r(x, representation="eul"), eul2r(x))
        nt.assert_array_almost_equal(
            x2r(x, representation="rpy/xyz"), rpy2r(x, order="xyz")
        )
        nt.assert_array_almost_equal(
            x2r(x, representation="rpy/zyx"), rpy2r(x, order="zyx")
        )
        nt.assert_array_almost_equal(
            x2r(x, representation="rpy/yxz"), rpy2r(x, order="yxz")
        )

        nt.assert_array_almost_equal(
            x2r(x, representation="arm"), rpy2r(x, order="xyz")
        )
        nt.assert_array_almost_equal(
            x2r(x, representation="vehicle"), rpy2r(x, order="zyx")
        )
        nt.assert_array_almost_equal(
            x2r(x, representation="camera"), rpy2r(x, order="yxz")
        )

        nt.assert_array_almost_equal(x2r(x, representation="exp"), trexp(x))

    def test_tr2x(self):
        t = [1, 2, 3]
        R = rpy2tr(0.2, 0.3, 0.4)
        T = transl(t) @ R

        x = tr2x(T, representation="eul")
        nt.assert_array_almost_equal(x[:3], t)
        nt.assert_array_almost_equal(x[3:], tr2eul(R))

        x = tr2x(T, representation="rpy/xyz")
        nt.assert_array_almost_equal(x[:3], t)
        nt.assert_array_almost_equal(x[3:], tr2rpy(R, order="xyz"))

        x = tr2x(T, representation="rpy/zyx")
        nt.assert_array_almost_equal(x[:3], t)
        nt.assert_array_almost_equal(x[3:], tr2rpy(R, order="zyx"))

        x = tr2x(T, representation="rpy/yxz")
        nt.assert_array_almost_equal(x[:3], t)
        nt.assert_array_almost_equal(x[3:], tr2rpy(R, order="yxz"))

        x = tr2x(T, representation="arm")
        nt.assert_array_almost_equal(x[:3], t)
        nt.assert_array_almost_equal(x[3:], tr2rpy(R, order="xyz"))

        x = tr2x(T, representation="vehicle")
        nt.assert_array_almost_equal(x[:3], t)
        nt.assert_array_almost_equal(x[3:], tr2rpy(R, order="zyx"))

        x = tr2x(T, representation="camera")
        nt.assert_array_almost_equal(x[:3], t)
        nt.assert_array_almost_equal(x[3:], tr2rpy(R, order="yxz"))

        x = tr2x(T, representation="exp")
        nt.assert_array_almost_equal(x[:3], t)
        nt.assert_array_almost_equal(x[3:], trlog(t2r(R), twist=True))

    def test_x2tr(self):
        t = [1, 2, 3]
        gamma = [0.3, 0.2, 0.1]
        x = np.r_[t, gamma]

        nt.assert_array_almost_equal(
            x2tr(x, representation="eul"), transl(t) @ eul2tr(gamma)
        )

        nt.assert_array_almost_equal(
            x2tr(x, representation="rpy/xyz"), transl(t) @ rpy2tr(gamma, order="xyz")
        )
        nt.assert_array_almost_equal(
            x2tr(x, representation="rpy/zyx"), transl(t) @ rpy2tr(gamma, order="zyx")
        )
        nt.assert_array_almost_equal(
            x2tr(x, representation="rpy/yxz"), transl(t) @ rpy2tr(gamma, order="yxz")
        )

        nt.assert_array_almost_equal(
            x2tr(x, representation="arm"), transl(t) @ rpy2tr(gamma, order="xyz")
        )
        nt.assert_array_almost_equal(
            x2tr(x, representation="vehicle"), transl(t) @ rpy2tr(gamma, order="zyx")
        )
        nt.assert_array_almost_equal(
            x2tr(x, representation="camera"), transl(t) @ rpy2tr(gamma, order="yxz")
        )

        nt.assert_array_almost_equal(
            x2tr(x, representation="exp"), transl(t) @ r2t(trexp(gamma))
        )


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
