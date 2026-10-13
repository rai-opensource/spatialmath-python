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

from spatialmath.base.transformsNd import *
from spatialmath.base.transforms3d import trotx, transl, rotx, isrot, ishom
from spatialmath.base.transforms2d import trot2, transl2, rot2, isrot2, ishom2

try:
    import sympy as sp

    _symbolics = True
    from spatialmath.base.symbolic import symbol
except ImportError:
    _symbolics = False


class TestND(unittest.TestCase):
    def test_iseye(self):
        self.assertTrue(iseye(np.eye(1)))
        self.assertTrue(iseye(np.eye(2)))
        self.assertTrue(iseye(np.eye(3)))
        self.assertTrue(iseye(np.eye(5)))

        self.assertFalse(iseye(2 * np.eye(3)))
        self.assertFalse(iseye(-np.eye(3)))
        self.assertFalse(iseye(np.array([[1, 0, 0], [0, 1, 0]])))
        self.assertFalse(iseye(np.array([1, 0, 0])))

    def test_r2t(self):
        # 3D
        R = rotx(0.3)
        T = r2t(R)
        nt.assert_array_almost_equal(T[0:3, 3], np.r_[0, 0, 0])
        nt.assert_array_almost_equal(T[:3, :3], R)

        # 2D
        R = rot2(0.3)
        T = r2t(R)
        nt.assert_array_almost_equal(T[0:2, 2], np.r_[0, 0])
        nt.assert_array_almost_equal(T[:2, :2], R)

        with self.assertRaises(ValueError):
            r2t(3)

        with self.assertRaises(ValueError):
            r2t(np.eye(3, 4))

        _ = r2t(np.ones((3, 3)), check=False)
        with self.assertRaises(ValueError):
            r2t(np.ones((3, 3)), check=True)

    @unittest.skipUnless(_symbolics, "sympy required")
    def test_r2t_sym(self):
        theta = symbol("theta")
        R = rot2(theta)
        T = r2t(R)
        self.assertEqual(r2t(R).dtype, "O")
        nt.assert_array_almost_equal(T[0:2, 2], np.r_[0, 0])
        nt.assert_array_almost_equal(T[:2, :2], R)

        theta = symbol("theta")
        R = rotx(theta)
        T = r2t(R)
        self.assertEqual(r2t(R).dtype, "O")
        nt.assert_array_almost_equal(T[0:3, 3], np.r_[0, 0, 0])
        # nt.assert_array_almost_equal(T[:3,:3], R)
        self.assertTrue((T[:3, :3] == R).all())

    def test_t2r(self):
        # 3D
        t = [1, 2, 3]
        T = trotx(0.3, t=t)
        R = t2r(T)
        nt.assert_array_almost_equal(T[:3, :3], R)
        nt.assert_array_almost_equal(transl(T), np.array(t))

        # 2D
        t = [1, 2]
        T = trot2(0.3, t=t)
        R = t2r(T)
        nt.assert_array_almost_equal(T[:2, :2], R)
        nt.assert_array_almost_equal(transl2(T), np.array(t))

        with self.assertRaises(ValueError):
            t2r(3)

        with self.assertRaises(ValueError):
            r2t(np.eye(3, 4))

    def test_rt2tr(self):
        # 3D
        R = rotx(0.2)
        t = [3, 4, 5]
        T = rt2tr(R, t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl(T), np.array(t))

        # 2D
        R = rot2(0.2)
        t = [3, 4]
        T = rt2tr(R, t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl2(T), np.array(t))

        with self.assertRaises(ValueError):
            rt2tr(3, 4)

        with self.assertRaises(ValueError):
            rt2tr(np.eye(3, 4), [1, 2, 3, 4])

        with self.assertRaises(ValueError):
            rt2tr(np.eye(4, 4), [1, 2, 3, 4])

        _ = rt2tr(np.ones((3, 3)), [1, 2, 3], check=False)
        with self.assertRaises(ValueError):
            rt2tr(np.ones((3, 3)), [1, 2, 3], check=True)

    @unittest.skipUnless(_symbolics, "sympy required")
    def test_rt2tr_sym(self):
        theta = symbol("theta")
        R = rotx(theta)
        self.assertEqual(r2t(R).dtype, "O")

        theta = symbol("theta")
        R = rot2(theta)
        self.assertEqual(r2t(R).dtype, "O")

    def test_tr2rt(self):
        # 3D
        T = trotx(0.3, t=[1, 2, 3])
        R, t = tr2rt(T)
        nt.assert_array_almost_equal(T[:3, :3], R)
        nt.assert_array_almost_equal(T[:3, 3], t)

        # 2D
        T = trot2(0.3, t=[1, 2])
        R, t = tr2rt(T)
        nt.assert_array_almost_equal(T[:2, :2], R)
        nt.assert_array_almost_equal(T[:2, 2], t)

        with self.assertRaises(ValueError):
            R, t = tr2rt(3)

        with self.assertRaises(ValueError):
            R, t = tr2rt(np.eye(3, 4))

    def test_Ab2M(self):
        # 3D
        R = np.ones((3, 3))
        t = [3, 4, 5]
        T = Ab2M(R, t)
        nt.assert_array_almost_equal(T[:3, :3], R)
        nt.assert_array_almost_equal(T[:3, 3], np.array(t))
        nt.assert_array_almost_equal(T[3, :], np.array([0, 0, 0, 0]))

        # 2D
        R = np.ones((2, 2))
        t = [3, 4]
        T = Ab2M(R, t)
        nt.assert_array_almost_equal(T[:2, :2], R)
        nt.assert_array_almost_equal(T[:2, 2], np.array(t))
        nt.assert_array_almost_equal(T[2, :], np.array([0, 0, 0]))

        with self.assertRaises(ValueError):
            Ab2M(3, 4)

        with self.assertRaises(ValueError):
            Ab2M(np.eye(3, 4), [1, 2, 3, 4])

        with self.assertRaises(ValueError):
            Ab2M(np.eye(4, 4), [1, 2, 3, 4])

    def test_checks(self):
        # 3D case, with rotation matrix
        R = np.eye(3)
        self.assertTrue(isR(R))
        self.assertFalse(isrot2(R))
        self.assertTrue(isrot(R))
        self.assertFalse(ishom(R))
        self.assertTrue(ishom2(R))
        self.assertFalse(isrot2(R, True))
        self.assertTrue(isrot(R, True))
        self.assertFalse(ishom(R, True))
        self.assertTrue(ishom2(R, True))

        # 3D case, invalid rotation matrix
        R = np.eye(3)
        R[0, 1] = 2
        self.assertFalse(isR(R))
        self.assertFalse(isrot2(R))
        self.assertTrue(isrot(R))
        self.assertFalse(ishom(R))
        self.assertTrue(ishom2(R))
        self.assertFalse(isrot2(R, True))
        self.assertFalse(isrot(R, True))
        self.assertFalse(ishom(R, True))
        self.assertFalse(ishom2(R, True))

        # 3D case, with rotation matrix
        T = np.array([[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        self.assertFalse(isR(T))
        self.assertFalse(isrot2(T))
        self.assertFalse(isrot(T))
        self.assertTrue(ishom(T))
        self.assertFalse(ishom2(T))
        self.assertFalse(isrot2(T, True))
        self.assertFalse(isrot(T, True))
        self.assertTrue(ishom(T, True))
        self.assertFalse(ishom2(T, True))

        # 3D case, invalid rotation matrix
        T = np.array([[1, 0, 0, 3], [0, 1, 1, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        self.assertFalse(isR(T))
        self.assertFalse(isrot2(T))
        self.assertFalse(isrot(T))
        self.assertTrue(
            ishom(T),
        )
        self.assertFalse(ishom2(T))
        self.assertFalse(isrot2(T, True))
        self.assertFalse(isrot(T, True))
        self.assertFalse(ishom(T, True))
        self.assertFalse(ishom2(T, True))

        # 3D case, invalid bottom row
        T = np.array([[1, 0, 0, 3], [0, 1, 1, 4], [0, 0, 1, 5], [9, 0, 0, 1]])
        self.assertFalse(isR(T))
        self.assertFalse(isrot2(T))
        self.assertFalse(isrot(T))
        self.assertTrue(ishom(T))
        self.assertFalse(ishom2(T))
        self.assertFalse(isrot2(T, True))
        self.assertFalse(isrot(T, True))
        self.assertFalse(ishom(T, True))
        self.assertFalse(ishom2(T, True))

        # skew matrices
        S = np.array([[0, 2], [-2, 0]])
        nt.assert_equal(isskew(S), True)
        S[0, 0] = 1
        nt.assert_equal(isskew(S), False)

        S = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        nt.assert_equal(isskew(S), True)
        S[0, 0] = 1
        nt.assert_equal(isskew(S), False)

    def test_homog(self):
        nt.assert_almost_equal(e2h([1, 2, 3]), np.c_[1, 2, 3, 1].T)

        nt.assert_almost_equal(h2e([2, 4, 6, 2]), np.c_[1, 2, 3].T)

    def test_e2h_vector(self):
        # list, tuple, 1D array, column vector all give an (N+1,1) column
        expected = np.c_[1, 2, 3, 1].T
        for v in ([1, 2, 3], (1, 2, 3), np.r_[1, 2, 3], np.c_[1, 2, 3].T):
            with self.subTest(type=type(v).__name__):
                out = e2h(v)
                self.assertEqual(out.shape, (4, 1))
                nt.assert_almost_equal(out, expected)

    def test_e2h_matrix(self):
        # each column gains a trailing 1
        e = np.c_[[1, 2], [3, 4], [5, 6]]
        nt.assert_almost_equal(e2h(e), np.array([[1, 3, 5], [2, 4, 6], [1, 1, 1]]))

    def test_h2e_vector(self):
        # the result is scaled by the last element
        expected = np.c_[1, 2, 3].T
        for v in ([2, 4, 6, 2], (2, 4, 6, 2), np.r_[2, 4, 6, 2], np.c_[2, 4, 6, 2].T):
            with self.subTest(type=type(v).__name__):
                out = h2e(v)
                self.assertEqual(out.shape, (3, 1))
                nt.assert_almost_equal(out, expected)

    def test_h2e_matrix(self):
        # each column is scaled by its own last element
        h = np.c_[[1, 2, 1], [3, 4, 2], [5, 6, 1]]
        nt.assert_almost_equal(h2e(h), np.array([[1, 1.5, 5], [2, 2, 6]]))

    def test_homog_roundtrip(self):
        v = np.c_[[1, 2, 3], [-4, 5, 0.5]]
        nt.assert_almost_equal(h2e(e2h(v)), v)

        h = np.c_[[2, 4, 6, 2], [3, 6, 9, 3]]
        nt.assert_almost_equal(e2h(h2e(h)), np.c_[[1, 2, 3, 1], [1, 2, 3, 1]])

    def test_homog_row_vector_is_matrix(self):
        # Documents (rather than endorses) current behavior: a 2D array is
        # always a matrix of column vectors, so a (1,N) row is N 1-vectors.
        # Changing this would silently break callers passing 1xN matrices.
        self.assertEqual(e2h(np.ones((1, 3))).shape, (2, 3))
        nt.assert_almost_equal(e2h(np.c_[[1, 2, 3]].T), np.array([[1, 2, 3], [1, 1, 1]]))
        nt.assert_almost_equal(h2e(np.array([[1.0, 2, 4], [1, 2, 2]])), np.array([[1, 1, 2]]))

    def test_h2e_too_few_rows(self):
        # a homogeneous point needs at least 2 elements, h2e of 1 row would be
        # an empty (0,N) array
        for bad in (np.ones((1, 3)), np.ones((1, 1)), [5], 5, np.ones((1,))):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(ValueError):
                    h2e(bad)

    def test_homog_bad_type(self):
        for convert in (e2h, h2e):
            for bad in ("hello", None, np.empty((0,))):
                with self.subTest(convert=convert.__name__, value=repr(bad)):
                    with self.assertRaises(ValueError):
                        convert(bad)

    def test_homtrans(self):
        # 3D
        T = trotx(pi / 2, t=[1, 2, 3])
        v = [10, 12, 14]
        v2 = homtrans(T, v)
        nt.assert_almost_equal(v2, np.c_[11, -12, 15].T)
        v = np.c_[[10, 12, 14], [-3, -4, -5]]
        v2 = homtrans(T, v)
        nt.assert_almost_equal(v2, np.c_[[11, -12, 15], [-2, 7, -1]])

        # 2D
        T = trot2(pi / 2, t=[1, 2])
        v = [10, 12]
        v2 = homtrans(T, v)
        nt.assert_almost_equal(v2, np.c_[-11, 12].T)
        v = np.c_[[10, 12], [-3, -4]]
        v2 = homtrans(T, v)
        nt.assert_almost_equal(v2, np.c_[[-11, 12], [5, -1]])

        with self.assertRaises(ValueError):
            T = trotx(pi / 2, t=[1, 2, 3])
            v = [10, 12]
            v2 = homtrans(T, v)

    def test_skew(self):
        # 3D
        sk = skew([1, 2, 3])
        self.assertEqual(sk.shape, (3, 3))
        nt.assert_almost_equal(sk + sk.T, np.zeros((3, 3)))
        self.assertEqual(sk[2, 1], 1)
        self.assertEqual(sk[0, 2], 2)
        self.assertEqual(sk[1, 0], 3)
        nt.assert_almost_equal(sk.diagonal(), np.r_[0, 0, 0])

        # 2D
        sk = skew([1])
        self.assertEqual(sk.shape, (2, 2))
        nt.assert_almost_equal(sk + sk.T, np.zeros((2, 2)))
        self.assertEqual(sk[1, 0], 1)
        nt.assert_almost_equal(sk.diagonal(), np.r_[0, 0])

        with self.assertRaises(ValueError):
            sk = skew([1, 2])

    def test_vex(self):
        # 3D
        t = [3, 4, 5]
        sk = skew(t)
        nt.assert_almost_equal(vex(sk), t)

        # 2D
        t = [3]
        sk = skew(t)
        nt.assert_almost_equal(vex(sk), t)

        _ = vex(np.ones((3, 3)), check=False)
        with self.assertRaises(ValueError):
            _ = vex(np.ones((3, 3)), check=True)

        with self.assertRaises(ValueError):
            _ = vex(np.eye(4, 4))

    def test_isskew(self):
        t = [3, 4, 5]
        sk = skew(t)
        self.assertTrue(isskew(sk))
        sk[0, 0] = 3
        self.assertFalse(isskew(sk))

        # 2D
        t = [3]
        sk = skew(t)
        self.assertTrue(isskew(sk))
        sk[0, 0] = 3
        self.assertFalse(isskew(sk))

    def test_isskewa(self):
        # 3D
        t = [3, 4, 5, 6, 7, 8]
        sk = skewa(t)
        self.assertTrue(isskewa(sk))
        sk[0, 0] = 3
        self.assertFalse(isskew(sk))
        sk = skewa(t)
        sk[3, 3] = 3
        self.assertFalse(isskew(sk))

        # 2D
        t = [3, 4, 5]
        sk = skew(t)
        self.assertTrue(isskew(sk))
        sk[0, 0] = 3
        self.assertFalse(isskew(sk))
        sk = skewa(t)
        sk[2, 2] = 3
        self.assertFalse(isskew(sk))

    def test_skewa(self):
        # 3D
        sk = skewa([1, 2, 3, 4, 5, 6])
        self.assertEqual(sk.shape, (4, 4))
        nt.assert_almost_equal(sk.diagonal(), np.r_[0, 0, 0, 0])
        nt.assert_almost_equal(sk[-1, :], np.r_[0, 0, 0, 0])
        nt.assert_almost_equal(sk[:3, 3], [1, 2, 3])
        nt.assert_almost_equal(vex(sk[:3, :3]), [4, 5, 6])

        # 2D
        sk = skewa([1, 2, 3])
        self.assertEqual(sk.shape, (3, 3))
        nt.assert_almost_equal(sk.diagonal(), np.r_[0, 0, 0])
        nt.assert_almost_equal(sk[-1, :], np.r_[0, 0, 0])
        nt.assert_almost_equal(sk[:2, 2], [1, 2])
        nt.assert_almost_equal(vex(sk[:2, :2]), [3])

        with self.assertRaises(ValueError):
            sk = skew([1, 2])

    def test_vexa(self):
        # 3D
        t = [1, 2, 3, 4, 5, 6]
        sk = skewa(t)
        nt.assert_almost_equal(vexa(sk), t)

        # 2D
        t = [1, 2, 3]
        sk = skewa(t)
        nt.assert_almost_equal(vexa(sk), t)

    def test_det(self):
        a = np.array([[1, 2], [3, 4]])
        self.assertAlmostEqual(np.linalg.det(a), det(a))

    @unittest.skipUnless(_symbolics, "sympy required")
    def test_det_sym(self):
        x, y = symbol("x y")
        a = np.array([[x, y], [y, x]])
        self.assertEqual(det(a), x**2 - y**2)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
