#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:49:29 2020

@author: corkep
"""

import unittest
import numpy as np
import numpy.testing as nt

from spatialmath.base.argcheck import *
from spatialmath.base.transformsNd import e2h, h2e


class Test_check(unittest.TestCase):
    def test_ismatrix(self):
        a = np.eye(3, 3)
        self.assertTrue(ismatrix(a, (3, 3)))
        self.assertFalse(ismatrix(a, (4, 3)))
        self.assertFalse(ismatrix(a, (3, 4)))
        self.assertFalse(ismatrix(a, (4, 4)))

        self.assertTrue(ismatrix(a, (-1, 3)))
        self.assertTrue(ismatrix(a, (3, -1)))
        self.assertTrue(ismatrix(a, (-1, -1)))

        self.assertFalse(ismatrix(1, (-1, -1)))

    def test_assertmatrix(self):
        with self.assertRaises(TypeError):
            assertmatrix(3)
        with self.assertRaises(TypeError):
            assertmatrix("not a matrix")

        with self.assertRaises(TypeError):
            a = np.eye(3, 3, dtype=complex)
            assertmatrix(a)

        a = np.eye(3, 3)

        assertmatrix(a)
        assertmatrix(a, (3, 3))
        assertmatrix(a, (None, 3))
        assertmatrix(a, (3, None))

        with self.assertRaises(ValueError):
            assertmatrix(a, (4, 3))
        with self.assertRaises(ValueError):
            assertmatrix(a, (4, None))
        with self.assertRaises(ValueError):
            assertmatrix(a, (None, 4))

    def test_getmatrix(self):
        a = np.random.rand(4, 3)
        self.assertEqual(getmatrix(a, (4, 3)).shape, (4, 3))
        self.assertEqual(getmatrix(a, (None, 3)).shape, (4, 3))
        self.assertEqual(getmatrix(a, (4, None)).shape, (4, 3))
        self.assertEqual(getmatrix(a, (None, None)).shape, (4, 3))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (5, 3))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (5, None))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (None, 4))

        with self.assertRaises(TypeError):
            m = getmatrix({}, (4, 3))

        a = np.r_[1, 2, 3, 4]
        self.assertEqual(getmatrix(a, (1, 4)).shape, (1, 4))
        self.assertEqual(getmatrix(a, (4, 1)).shape, (4, 1))
        self.assertEqual(getmatrix(a, (2, 2)).shape, (2, 2))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (5, None))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (None, 5))

        a = [1, 2, 3, 4]
        self.assertEqual(getmatrix(a, (1, 4)).shape, (1, 4))
        self.assertEqual(getmatrix(a, (4, 1)).shape, (4, 1))
        self.assertEqual(getmatrix(a, (2, 2)).shape, (2, 2))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (5, None))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (None, 5))

        a = 7
        self.assertEqual(getmatrix(a, (1, 1)).shape, (1, 1))
        self.assertEqual(getmatrix(a, (None, None)).shape, (1, 1))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (2, 1))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (1, 2))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (None, 2))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (2, None))

        a = 7.0
        self.assertEqual(getmatrix(a, (1, 1)).shape, (1, 1))
        self.assertEqual(getmatrix(a, (None, None)).shape, (1, 1))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (2, 1))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (1, 2))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (None, 2))
        with self.assertRaises(ValueError):
            m = getmatrix(a, (2, None))

    def test_verifymatrix(self):
        with self.assertRaises(TypeError):
            assertmatrix(3)
        with self.assertRaises(TypeError):
            verifymatrix([3, 4])

        a = np.eye(3, 3)

        verifymatrix(a, (3, 3))
        with self.assertRaises(ValueError):
            verifymatrix(a, (3, 4))

    def test_unit(self):
        # scalar -> vector
        self.assertEqual(getunit(1), np.array([1]))
        self.assertEqual(getunit(1, dim=0), np.array([1]))
        with self.assertRaises(ValueError):
            self.assertEqual(getunit(1, dim=1), np.array([1]))

        self.assertEqual(getunit(1, unit="deg"), np.array([1 * math.pi / 180.0]))
        self.assertEqual(getunit(1, dim=0, unit="deg"), np.array([1 * math.pi / 180.0]))
        with self.assertRaises(ValueError):
            self.assertEqual(
                getunit(1, dim=1, unit="deg"), np.array([1 * math.pi / 180.0])
            )

        # scalar -> scalar
        self.assertEqual(getunit(1, vector=False), 1)
        self.assertEqual(getunit(1, dim=0, vector=False), 1)
        with self.assertRaises(ValueError):
            self.assertEqual(getunit(1, dim=1, vector=False), 1)

        self.assertIsInstance(getunit(1.0, vector=False), float)
        self.assertIsInstance(getunit(1, vector=False), int)

        self.assertEqual(getunit(1, vector=False, unit="deg"), 1 * math.pi / 180.0)
        self.assertEqual(
            getunit(1, dim=0, vector=False, unit="deg"), 1 * math.pi / 180.0
        )
        with self.assertRaises(ValueError):
            self.assertEqual(
                getunit(1, dim=1, vector=False, unit="deg"), 1 * math.pi / 180.0
            )

        self.assertIsInstance(getunit(1.0, vector=False, unit="deg"), float)
        self.assertIsInstance(getunit(1, vector=False, unit="deg"), float)

        # vector -> vector
        self.assertEqual(getunit([1]), np.array([1]))
        self.assertEqual(getunit([1], dim=1), np.array([1]))
        with self.assertRaises(ValueError):
            getunit([1], dim=0)

        self.assertIsInstance(getunit([1, 2]), np.ndarray)
        self.assertIsInstance(getunit((1, 2)), np.ndarray)
        self.assertIsInstance(getunit(np.r_[1, 2]), np.ndarray)

        nt.assert_equal(getunit(5, "rad"), 5)
        nt.assert_equal(getunit(5, "deg"), 5 * math.pi / 180.0)
        nt.assert_equal(getunit([3, 4, 5], "rad"), [3, 4, 5])
        nt.assert_almost_equal(
            getunit([3, 4, 5], "deg"), [x * math.pi / 180.0 for x in [3, 4, 5]]
        )
        nt.assert_equal(getunit((3, 4, 5), "rad"), [3, 4, 5])
        nt.assert_almost_equal(
            getunit((3, 4, 5), "deg"),
            np.array([x * math.pi / 180.0 for x in [3, 4, 5]]),
        )

        nt.assert_equal(getunit(np.array([3, 4, 5]), "rad"), [3, 4, 5])
        nt.assert_almost_equal(
            getunit(np.array([3, 4, 5]), "deg"),
            [x * math.pi / 180.0 for x in [3, 4, 5]],
        )

    def test_isvector(self):
        # no length specified
        self.assertTrue(isvector(2))
        self.assertTrue(isvector(2.0))
        self.assertTrue(isvector([1, 2, 3]))
        self.assertTrue(isvector((1, 2, 3)))
        self.assertTrue(isvector(np.array([1, 2, 3])))
        self.assertTrue(isvector(np.array([[1, 2, 3]])))
        self.assertTrue(isvector(np.array([[1], [2], [3]])))

        # length specified
        self.assertTrue(isvector(2, 1))
        self.assertTrue(isvector(2.0, 1))
        self.assertTrue(isvector([1, 2, 3], 3))
        self.assertTrue(isvector((1, 2, 3), 3))
        self.assertTrue(isvector(np.array([1, 2, 3]), 3))
        self.assertTrue(isvector(np.array([[1, 2, 3]]), 3))
        self.assertTrue(isvector(np.array([[1], [2], [3]]), 3))

        # wrong length specified
        self.assertFalse(isvector(2, 4))
        self.assertFalse(isvector(2.0, 4))
        self.assertFalse(isvector([1, 2, 3], 4))
        self.assertFalse(isvector((1, 2, 3), 4))
        self.assertFalse(isvector(np.array([1, 2, 3]), 4))
        self.assertFalse(isvector(np.array([[1, 2, 3]]), 4))
        self.assertFalse(isvector(np.array([[1], [2], [3]]), 4))

    def test_assertvector(self):
        l = [1, 2, 3]
        nt.assert_raises(ValueError, assertvector, l, 4)

    def test_isvector_unsupported_array_shapes(self):
        for shape in [(), (2, 3), (1, 3, 2), (3, 1, 2), (1, 1, 1), (1, 3, 0)]:
            with self.subTest(shape=shape):
                value = np.ones(shape)
                self.assertFalse(isvector(value))
                with self.assertRaises(ValueError):
                    assertvector(value)

    def test_isvector_empty_arrays(self):
        for shape in [(0,), (1, 0), (0, 1), (0, 0)]:
            with self.subTest(shape=shape):
                self.assertFalse(isvector(np.empty(shape)))

    def test_homogeneous_conversion_unsupported_array_shapes(self):
        for convert in [e2h, h2e]:
            for shape in [(), (1, 3, 2), (3, 1, 2), (1, 1, 1)]:
                with self.subTest(convert=convert.__name__, shape=shape):
                    with self.assertRaises(ValueError):
                        convert(np.ones(shape))

    def test_getvector(self):
        l = [1, 2, 3]
        t = (1, 2, 3)
        a = np.array(l)
        r = np.array([[1, 2, 3]])
        c = np.array([[1], [2], [3]])

        # input is list
        v = getvector(l)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(l, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(l, out="sequence")
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(l, 3, out="sequence")
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(l, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(l, 3, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(l, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(l, 3, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(l, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(l, 3, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, l, 4)
        nt.assert_raises(ValueError, getvector, l, 4, "sequence")
        nt.assert_raises(ValueError, getvector, l, 4, "array")
        nt.assert_raises(ValueError, getvector, l, 4, "row")
        nt.assert_raises(ValueError, getvector, l, 4, "col")

        # input is tuple

        v = getvector(t)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(t, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(t, out="sequence")
        self.assertIsInstance(v, tuple)
        nt.assert_equal(len(v), 3)

        v = getvector(t, 3, out="sequence")
        self.assertIsInstance(v, tuple)
        nt.assert_equal(len(v), 3)

        v = getvector(t, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(t, 3, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(t, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(t, 3, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(t, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(t, 3, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, t, 4)
        nt.assert_raises(ValueError, getvector, t, 4, "sequence")
        nt.assert_raises(ValueError, getvector, t, 4, "array")
        nt.assert_raises(ValueError, getvector, t, 4, "row")
        nt.assert_raises(ValueError, getvector, t, 4, "col")

        # input is array

        v = getvector(a)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(a, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(a, out="sequence")
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(a, 3, out="sequence")
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(a, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(a, 3, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(a, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(a, 3, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(a, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(a, 3, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, a, 4)
        nt.assert_raises(ValueError, getvector, a, 4, "sequence")
        nt.assert_raises(ValueError, getvector, a, 4, "array")
        nt.assert_raises(ValueError, getvector, a, 4, "row")
        nt.assert_raises(ValueError, getvector, a, 4, "col")

        # input is row

        v = getvector(r)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(r, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(r, out="sequence")
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(r, 3, out="sequence")
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(r, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(r, 3, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(r, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(r, 3, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(r, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(r, 3, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, r, 4)
        nt.assert_raises(ValueError, getvector, r, 4, "sequence")
        nt.assert_raises(ValueError, getvector, r, 4, "array")
        nt.assert_raises(ValueError, getvector, r, 4, "row")
        nt.assert_raises(ValueError, getvector, r, 4, "col")

        # input is col

        v = getvector(c)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(c, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(c, out="sequence")
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(c, 3, out="sequence")
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(c, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(c, 3, out="array")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(c, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(c, 3, out="row")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(c, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(c, 3, out="col")
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, c, 4)
        nt.assert_raises(ValueError, getvector, c, 4, "sequence")
        nt.assert_raises(ValueError, getvector, c, 4, "array")
        nt.assert_raises(ValueError, getvector, c, 4, "row")
        nt.assert_raises(ValueError, getvector, c, 4, "col")

    def test_isnumberlist(self):
        nt.assert_equal(isnumberlist([1]), True)
        nt.assert_equal(isnumberlist([1, 2]), True)
        nt.assert_equal(isnumberlist((1,)), True)
        nt.assert_equal(isnumberlist((1, 2)), True)
        nt.assert_equal(isnumberlist(1), False)
        nt.assert_equal(isnumberlist([]), False)
        nt.assert_equal(isnumberlist(np.array([1, 2, 3])), False)

    def test_isvectorlist(self):
        a = [np.r_[1, 2], np.r_[3, 4], np.r_[5, 6]]
        self.assertTrue(isvectorlist(a, 2))

        a = [(1, 2), (3, 4), (5, 6)]
        self.assertFalse(isvectorlist(a, 2))

        a = [np.r_[1, 2], np.r_[3, 4], np.r_[5, 6, 7]]
        self.assertFalse(isvectorlist(a, 2))

    def test_islistof(self):
        a = [3, 4, 5]
        self.assertTrue(islistof(a, int))
        self.assertFalse(islistof(a, float))
        self.assertTrue(islistof(a, lambda x: isinstance(x, int)))

        self.assertTrue(islistof(a, int, 3))
        self.assertFalse(islistof(a, int, 2))

        a = [3, 4.5, 5.6]
        self.assertFalse(islistof(a, int))
        self.assertTrue(islistof(a, (int, float)))
        a = [[1, 2], [3, 4], [5, 6]]
        self.assertTrue(islistof(a, lambda x: islistof(x, int, 2)))


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover
    unittest.main()
